# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The prefix store: the state a request's prefill left at a multiple of GRID tokens (the chunked prefill's grid: one
sequence-parallel wave pair), kept in DRAM apart from the seats, and written back into a seat so that a later request
with the same first tokens is prefilled from there (`Glm53Generator.prefill_many`'s `begin`) instead of from position
0.  GLM53_PREFIX_ENTRIES = E entries or "auto" (the generator's default 0: off,
nothing allocated; the vLLM serving's default "auto", tools/glm53_vllm.py PREFIX).

A prefill call that ends at position A (a single request over the mesh rows, the sequence-parallel prefill) leaves what
the next call from A reads (as the chunked prefill resumes):
  - each mesh row's KDA wave state (fp32 [16, 128, 128] a layer) and conv history ([1, 3, 6144] bf16): each row's own
    (the request's last segment's scan on its own row, the chain of the segments' maps on the others);
  - the DSA caches' latents and pooled indexer keys of positions 0 .. A - 1: on the request's own row in its user, on
    the other rows in the sequence-parallel extra user (user U), the same bytes on every row;
  - the drafter's K / V of the request's user (its ring of the last 4,096 positions), on its own row.
A snapshot copies these into entry e of the store (each chip its own: no data crosses chips); a restore copies them
back, the DSA rows into the target seat's user on its row and into the extra user on the others, the drafter's into the
target user, the wave states into every row.  The target seat must be on the snapshot's mesh row (the rows' roles in
the wave states, own row against the others, stay where they were): then the call from A reads the bits a whole
prefill would have left (the design; tests/test_prefix_resume.py checks it).  Not stored: the recent pool ring (the
call from A writes it whole at its end), the seats' own KDA states (the call's commit writes them), positions past A.

The copies are one op a kind of buffer (`copy_pages`: segments of pages between pairs of interleaved DRAM buffers of one
page size, the same segments for each pair, per chip from a plan): five a snapshot or restore (the 11 DSA layers'
latents, their pooled keys, the 34 KDA layers' wave states, their histories, the drafter's K and V), their plans written
in one host write.  Three programs (by the pairs an op: 11, 34, 2; the hash has no tensor in it), made in the
generator's warm-up (`warm`) before any capture.  (One op a buffer pair, the earlier form, was ~92 ops a snapshot:
25-35 ms of host time a continuation call.)

The store's size (`auto_entries`, GLM53_PREFIX_ENTRIES=auto: the vLLM serving's default): an entry holds whole seats'
caches (the DSA rows of a user for the cache's length, max_model_len), so its bytes grow with the cache (a chip ~78 MiB
at 9K, ~120 MiB at 36K, ~265 MiB at 128K, ~1.67 GiB at 1M).  The entries are what fits in the chip's free DRAM when the
generator makes its states, less a reserve for what the warm-up and the serving allocate after them (decode and prefill
inputs, the constants, the prefill's working set: GLM53_PREFIX_RESERVE_GIB, default 4), at most a fraction of it
(GLM53_PREFIX_DRAM_FRACTION, default 0.5) and of its largest free block, and at most two a seat
(GLM53_PREFIX_MAX_ENTRIES); none (the store off, the reason in the log) when not one entry fits.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.glm53_flash.config import DSA, KDA

from .common import COLS, ROWS, core_rects, put, zeros
from .prefix_keys import AUTO, GRID, entries_setting, prefix_keys  # noqa: F401 (prefix_keys: the callers')

ENTRIES = entries_setting("0")  # (read by Glm53Generator, whose default is no store; "auto" or a number)
RESERVE_GIB = float(os.environ.get("GLM53_PREFIX_RESERVE_GIB", "4"))
DRAM_FRACTION = float(os.environ.get("GLM53_PREFIX_DRAM_FRACTION", "0.5"))
MAX_ENTRIES = os.environ.get("GLM53_PREFIX_MAX_ENTRIES")  # (unset: two a seat)
KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/prefix/"
PLAN_WIDTH = 128  # uint32 a chip's plan: S, then (source page, destination page, pages) a segment
MAX_SEGMENTS = (PLAN_WIDTH - 1) // 3
COPY_CORES = 64
SLOTS = 16
SLOT_BYTES = 4096  # the largest page (an fp32 tile)
MAX_PAIRS = 64  # buffer pairs an op (its runtime args: 4 + 2 a pair)


def _run(mesh, tensors: list, program) -> None:
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    ttnn.generic_op(tensors, mesh_program)


def page_bytes(t) -> int:
    return int(t.buffer_page_size())


def make_plans(mesh, kinds: list) -> ttnn.Tensor:
    """kinds[k][r]: plan k's segments on mesh row r [(source page, destination page, pages)] -> the plans, uint32
    row-major [len(kinds), PLAN_WIDTH] a chip (row k plan k, the same on a row's chips), in one host write.  The caller
    deallocates it."""
    host = torch.zeros(ROWS, len(kinds), PLAN_WIDTH, dtype=torch.int64)
    for k, rows in enumerate(kinds):
        assert len(rows) == ROWS, rows
        for r, segs in enumerate(rows):
            segs = [s for s in segs if s[2] > 0]
            if len(segs) > MAX_SEGMENTS:
                raise ValueError(f"row {r}: {len(segs)} segments, at most {MAX_SEGMENTS}")
            host[r, k, 0] = len(segs)
            for i, (a, b, n) in enumerate(segs):
                if min(a, b, n) < 0:
                    raise ValueError(f"row {r}: a negative page or count {segs[i]}")
                host[r, k, 1 + 3 * i : 4 + 3 * i] = torch.tensor([a, b, n])
    host = host.reshape(ROWS * len(kinds), PLAN_WIDTH).to(torch.int32)
    return put(mesh, host, dims=(0, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def make_plan(mesh, rows: list) -> ttnn.Tensor:
    """rows[r]: mesh row r's segments -> one plan, uint32 row-major [1, PLAN_WIDTH] a chip (`make_plans`)."""
    return make_plans(mesh, [rows])


def copy_pages(mesh, src, dst, plan, plan_row: int = 0) -> None:
    """Pages of src into dst (interleaved DRAM, the same page size: bytes only) by row `plan_row` of `plan`
    (`make_plans`), each chip its own segments, in one op (`copy_pages_many`)."""
    copy_pages_many(mesh, [(src, dst)], plan, plan_row)


def copy_pages_many(mesh, pairs: list, plan, plan_row: int = 0) -> None:
    """For each (src, dst) of `pairs` (interleaved DRAM buffers, all of one page size: bytes only), the pages of src
    into dst by row `plan_row` of `plan` (`make_plans`; the same segments for every pair), each chip its own segments,
    in one op (kernels/prefix/copy_pages.cpp; COPY_CORES cores, the pages dealt round-robin over the pairs' segments in
    order).  The program is one a number of pairs (a compile-time argument)."""
    if not 0 < len(pairs) <= MAX_PAIRS:
        raise ValueError(f"copy_pages: {len(pairs)} buffer pairs, 1 to {MAX_PAIRS}")
    src0, dst0 = pairs[0]
    page = page_bytes(src0)
    if page > SLOT_BYTES or page % 64:
        raise ValueError(f"copy_pages: pages of {page} bytes")
    src_ct = ttnn.TensorAccessorArgs(src0).get_compile_time_args()
    dst_ct = ttnn.TensorAccessorArgs(dst0).get_compile_time_args()
    addrs = []
    for src, dst in pairs:
        if page_bytes(src) != page or page_bytes(dst) != page:
            raise ValueError(f"copy_pages: pages of {page_bytes(src)} and {page_bytes(dst)} bytes, not all {page}")
        if (
            ttnn.TensorAccessorArgs(src).get_compile_time_args() != src_ct
            or ttnn.TensorAccessorArgs(dst).get_compile_time_args() != dst_ct
        ):
            raise ValueError("copy_pages: the pairs' buffers are not all of one kind (interleaved DRAM)")
        addrs += [src.buffer_address(), dst.buffer_address()]
    grid = mesh.compute_with_storage_grid_size()
    n = min(COPY_CORES, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
    cores = core_rects(coords)
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(coords):
        rt[c.x][c.y] = [plan.buffer_address(), page, i, int(plan_row)] + addrs
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "copy_pages.cpp",
        core_ranges=cores,
        compile_time_args=[n, 4 * PLAN_WIDTH, SLOTS, SLOT_BYTES, 0, len(pairs)]
        + src_ct
        + ttnn.TensorAccessorArgs(plan).get_compile_time_args()
        + dst_ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    size = 64 + 4 * PLAN_WIDTH + SLOTS * SLOT_BYTES
    cb = ttnn.CBDescriptor(
        total_size=size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)],
    )
    _run(mesh, [src0, plan, dst0], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))


def _tiles(t) -> int:
    return math.prod(int(d) for d in t.padded_shape) // 1024


def dram_free(mesh) -> tuple[int, int]:
    """A chip's free DRAM (bytes, every chip the same) and the largest free block over its banks."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    return int(view.total_bytes_free_per_bank) * banks, int(view.largest_contiguous_bytes_free_per_bank) * banks


def _layout(model, states: list, drafter, users: int) -> dict:
    """What an entry holds, from the seats' tensors (no allocation): per DSA layer (state, latent rows a user a chip,
    the latent split, pool pages a user, tokens a pool page, tiles a pool page), per KDA layer (state, tiles of its wave
    state, of its history), the drafter's (tiles a page, pages a user a layer) and an entry's bytes a chip."""
    out = {"dsa": [], "kda": [], "drafter": None, "bytes": 0}
    for layer, st in zip(model.layers, states):
        if layer.kind == DSA:
            att = layer.attention
            if not att.long or st.rm is None:
                raise ValueError("the prefix store needs the long DSA cache (a cache past 2,051 positions)")
            total = st.rm.shape[-2]
            if total % (users + 1):
                raise ValueError(f"the latent cache's {total} rows are not {users + 1} users' (the SP extra user)")
            per = total // (users + 1)
            split = COLS if att.shard_kv() else 1
            pool_tokens = att.kpool * 32  # (a pool page: 32 pools, 4 tiles of [32, 32] across the 128 key values)
            tiles = st.pools.shape[-1] // 32  # tiles a pool page (the copy's pages are tiles)
            out["dsa"].append((st, per, split, att.pool_pages, pool_tokens, tiles))
            out["bytes"] += per * page_bytes(st.rm) + att.pool_pages * tiles * page_bytes(st.pools)
        elif layer.kind == KDA:
            s_pages, h_pages = _tiles(st.wave_state), _tiles(st.wave_history)
            out["kda"].append((st, s_pages, h_pages))
            out["bytes"] += s_pages * page_bytes(st.wave_state) + h_pages * page_bytes(st.wave_history)
    if drafter is not None:  # a user's 2 virtual users (its 2 KV heads a chip) a layer, `pages` each
        d_tiles, d_pages = drafter.cfg.head_dim // 32, 2 * drafter.pages
        out["drafter"] = (d_tiles, d_pages)
        out["bytes"] += 2 * drafter.n_layers * d_pages * d_tiles * page_bytes(drafter.k_cache)
    return out


def entry_bytes(model, states: list, drafter, users: int) -> int:
    """One entry's DRAM a chip (its pages; the store's tensors round each up to whole pages a bank, a few KiB)."""
    return _layout(model, states, drafter, users)["bytes"]


def auto_entries(model, states: list, drafter, users: int, free: tuple | None = None) -> tuple[int, str]:
    """GLM53_PREFIX_ENTRIES=auto: the entries the free DRAM holds now (the generator's states made, before its warm-up)
    -> (entries, the reason in a line).  The budget: the free DRAM less RESERVE_GIB, at most DRAM_FRACTION of it and the
    largest free block; at most MAX_ENTRIES (default two a seat: each seat's request extends one entry while the last
    turn of its conversation stays in another).  `free`: (free, largest block) bytes for the device's (tests)."""
    seats = ROWS * users
    try:
        one = entry_bytes(model, states, drafter, users)
    except ValueError as err:  # (no long DSA cache: nothing the store could hold)
        return 0, f"prefix store off: {err}"
    total, block = free if free is not None else dram_free(model.mesh)
    return plan_entries(one, total, block, seats)


def plan_entries(one: int, total: int, block: int, seats: int) -> tuple[int, str]:
    """`auto_entries`' arithmetic: an entry of `one` bytes, `total` free bytes a chip and its largest block `block`."""
    gib = 2**30
    cap = int(MAX_ENTRIES) if MAX_ENTRIES else 2 * seats
    budget = min(total - RESERVE_GIB * gib, DRAM_FRACTION * total, block)
    n = max(0, min(cap, int(budget // one))) if one > 0 else 0
    why = (
        f"an entry {one / gib:.3f} GiB a chip; DRAM free {total / gib:.2f} GiB (largest block {block / gib:.2f}), less "
        f"the reserve {RESERVE_GIB:g} GiB, at most {DRAM_FRACTION:g} of it: {max(budget, 0) / gib:.2f} GiB; at most "
        f"{cap} entries"
    )
    if n == 0:
        return 0, f"prefix store off: not one entry fits ({why})"
    return n, f"prefix store: {n} entries, {n * one / gib:.2f} GiB a chip ({why})"


@dataclass
class Entry:
    length: int = 0  # the prefix's tokens (0: empty)
    row: int = -1  # the mesh row of the request it was taken from
    key: str | None = None  # prefix_keys(..)[length // GRID - 1]
    tick: int = 0  # last use (LRU)
    pins: int = 0  # held for a restore to come (not evicted)


class PrefixStore:
    """E entries of a generator's prefill state (the module's docstring)."""

    def __init__(self, model, states: list, drafter, users: int, entries: int) -> None:
        if entries <= 0:
            raise ValueError("a prefix store of no entries")
        from .dsa import _zeros

        self.mesh = model.mesh
        self.users = users
        self.entries = [Entry() for _ in range(entries)]
        self.tick = 0
        layout = _layout(model, states, drafter, users)
        self.entry_bytes = layout["bytes"]
        # (state, latent rows a user a chip, tokens a latent row's chip split, pool pages a user, tokens a pool page,
        # tiles a pool page, the store's latents, its pools)
        self.dsa = []
        self.kda = []  # (state, the store's wave states, its wave histories, pages of each)
        for st, per, split, pool_pages, pool_tokens, tiles in layout["dsa"]:
            rm = _zeros(self.mesh, [1, 1, entries * per, st.rm.shape[-1]], st.rm.dtype, ttnn.ROW_MAJOR_LAYOUT)
            pools = _zeros(self.mesh, [entries * pool_pages, 1, 32, st.pools.shape[-1]], st.pools.dtype)
            self.dsa.append((st, per, split, pool_pages, pool_tokens, tiles, rm, pools))
        for st, s_pages, h_pages in layout["kda"]:
            ws, wh = st.wave_state, st.wave_history
            store_s = zeros(self.mesh, [entries * ws.shape[0], ws.shape[1], ws.shape[2]], ttnn.float32)
            store_h = zeros(self.mesh, [entries, wh.shape[1], wh.shape[2]], wh.dtype)
            assert _tiles(store_s) == entries * s_pages and _tiles(store_h) == entries * h_pages
            self.kda.append((st, store_s, store_h, s_pages, h_pages))
        self.drafter = drafter
        if drafter is not None:  # a user's 2 virtual users (its 2 KV heads a chip) a layer, `pages` each
            d = drafter
            self.d_tiles, self.d_pages = layout["drafter"]  # (tiles a page [32, head_dim]; a user's pages a layer)
            shape = [entries * d.n_layers * self.d_pages, 1, 32, d.cfg.head_dim]
            if d.kv_dtype == ttnn.bfloat16:
                self.d_k, self.d_v = zeros(self.mesh, shape), zeros(self.mesh, shape)
            else:
                from .dsa_indexer import zeros_bytes

                self.d_k = zeros_bytes(self.mesh, shape, d.kv_dtype, ttnn.TILE_LAYOUT)
                self.d_v = zeros_bytes(self.mesh, shape, d.kv_dtype, ttnn.TILE_LAYOUT)
        gib, one = self.bytes_a_chip() / 2**30, self.entry_bytes / 2**30
        print(f"prefix store: {entries} entries, {gib:.2f} GiB a chip ({one:.3f} GiB an entry)", flush=True)

    def bytes_a_chip(self) -> int:
        """The store's DRAM a chip (its tensors' pages)."""
        out = 0
        for *_, rm, pools in self.dsa:
            out += rm.shape[-2] * page_bytes(rm) + _tiles(pools) * page_bytes(pools)
        for _, s, h, _, _ in self.kda:
            out += _tiles(s) * page_bytes(s) + _tiles(h) * page_bytes(h)
        if self.drafter is not None:
            out += 2 * _tiles(self.d_k) * page_bytes(self.d_k)
        return out

    # -- the copies ---------------------------------------------------------------------------------------------------
    def _copy(self, entry: int, row: int, user: int, length: int, have: int, to_store: bool) -> None:
        """Seat (row, user) <-> entry: the DSA rows [have, length) (the seat's user on `row`, the extra user on the
        other rows), every row's KDA wave states and the drafter's K / V of `user` on `row`."""
        mesh, extra = self.mesh, self.users

        def rows_of(seg):  # seg(r) -> (seat page, entry page, pages) or None -> each row's segments, in direction
            rows = []
            for r in range(ROWS):
                s = seg(r)
                rows.append([] if s is None else [s if to_store else (s[1], s[0], s[2])])
            return rows

        kinds, groups = [], []  # (each plan's rows; each op's (seat tensor, store tensor) pairs, by plan)
        if self.dsa:
            st0, per, split, pp, pool_tokens, pt, _, _ = self.dsa[0]
            r0, r1 = have // split, length // split
            p0, p1 = have // pool_tokens, length // pool_tokens

            def lat(r):
                u = user if r == row else extra
                return (u * per + r0, entry * per + r0, r1 - r0)

            def pool(r):  # (in tiles)
                u = user if r == row else extra
                return ((u * pp + p0) * pt, (entry * pp + p0) * pt, (p1 - p0) * pt)

            kinds += [rows_of(lat), rows_of(pool)]
            groups.append([(st.rm, rm) for st, *_, rm, _ in self.dsa])
            groups.append([(st.pools, pools) for st, *_, pools in self.dsa])
        if self.kda:
            _, _, _, sp, hp = self.kda[0]
            kinds += [rows_of(lambda r: (0, entry * sp, sp)), rows_of(lambda r: (0, entry * hp, hp))]
            groups.append([(st.wave_state, s) for st, s, _, _, _ in self.kda])
            groups.append([(st.wave_history, h) for st, _, h, _, _ in self.kda])
        if self.drafter is not None:
            d, dp, dt = self.drafter, self.d_pages, self.d_tiles
            rows = []
            for r in range(ROWS):
                segs = []
                if r == row:  # (in tiles)
                    for layer in range(d.n_layers):
                        seat = (layer * d.layer_pages + 2 * user * d.pages) * dt
                        store = (entry * d.n_layers + layer) * dp * dt
                        segs.append((seat, store, dp * dt) if to_store else (store, seat, dp * dt))
                rows.append(segs)
            kinds.append(rows)
            groups.append([(d.k_cache, self.d_k), (d.v_cache, self.d_v)])
        plans = make_plans(mesh, kinds)  # (one host write for all the plans)
        for k, pairs in enumerate(groups):
            copy_pages_many(mesh, [(a, b) if to_store else (b, a) for a, b in pairs], plans, k)
        ttnn.deallocate(plans)

    def snapshot(self, entry: int, row: int, user: int, length: int, key: str | None, have: int = 0) -> None:
        """The seat (row, user)'s prefill state at `length` (a multiple of GRID; its last prefill call ended there, no
        other prefill since) into `entry`; `have`: the DSA rows [0, have) are already the entry's (an earlier snapshot of
        the same request: only [have, length) are copied)."""
        if length <= 0 or length % GRID or have % GRID or have > length:
            raise ValueError(f"a snapshot at {length} (from {have}): multiples of {GRID}")
        e = self.entries[entry]
        if have and (e.row != row or e.length != have):
            raise ValueError(f"entry {entry} holds {e.length} tokens of row {e.row}, not {have} of row {row}")
        self._copy(entry, row, user, length, have, to_store=True)
        self.tick += 1
        e.length, e.row, e.key, e.tick = length, row, key, self.tick

    def restore(self, entry: int, row: int, user: int) -> int:
        """Entry -> the seat (row, user), which must be on the entry's row -> the prefix's length."""
        e = self.entries[entry]
        if e.length <= 0:
            raise ValueError(f"entry {entry} is empty")
        if row != e.row:
            raise ValueError(f"entry {entry} is of mesh row {e.row}: a seat of that row only, not {row}")
        self._copy(entry, row, user, e.length, 0, to_store=False)
        self.tick += 1
        e.tick = self.tick
        return e.length

    # -- the entries ---------------------------------------------------------------------------------------------------
    def find(self, keys: list[str], limit: int) -> tuple[int, int] | None:
        """The longest stored prefix among `keys` (prefix_keys of a prompt) of at most `limit` tokens -> (entry,
        length), or None."""
        by_key = {e.key: i for i, e in enumerate(self.entries) if e.length and e.key is not None}
        for j in range(min(len(keys), limit // GRID) - 1, -1, -1):
            i = by_key.get(keys[j])
            if i is not None and self.entries[i].length == (j + 1) * GRID:
                return i, (j + 1) * GRID
        return None

    def take(self, key: str | None = None) -> int:
        """An entry to write: the one holding `key` (the same prefix), else an empty one, else the least recently used
        unpinned one (its prefix forgotten)."""
        free = [i for i, e in enumerate(self.entries) if e.pins == 0]
        if not free:
            raise RuntimeError("every prefix entry is pinned")
        for i in free:
            if key is not None and self.entries[i].key == key:
                return i
        empty = [i for i in free if self.entries[i].length == 0]
        i = empty[0] if empty else min(free, key=lambda i: self.entries[i].tick)
        self.entries[i] = Entry()
        return i

    def warm(self) -> None:
        """The copy op's program (and each plan's) made once, before any capture (the warm-up's states are meaningless:
        entry 0 from and back into seat (0, 0), GRID tokens)."""
        if self.dsa:
            st, per, split, *_ = self.dsa[0]
            assert per * split >= GRID, (per, split)
        self._copy(0, 0, 0, GRID, 0, to_store=True)
        self._copy(0, 0, 0, GRID, 0, to_store=False)
        self.entries[0] = Entry()
