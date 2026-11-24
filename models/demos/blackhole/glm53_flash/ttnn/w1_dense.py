# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The dense L1 landing (GLM53_W1D): part of the next KDA layer's x_proj weights (the decode_matmul with its fg made
inside) brought into L1 during the decode step's DRAM-light programs, by "riders" on the cores those programs leave
idle, so that the x_proj's decode_matmul reads those chunks from L1 in place of DRAM.  The values are the same: the
riders copy the DRAM's bytes into each core's L1 in the decode_matmul's own chunk order and slot layout, and the
decode_matmul copies a chunk from there into the same ring slot it would have read it into (decode_matmul_weights.cpp
DM_LANDING, kernels/w1_dense/dm_rider.cpp).

The hosts (the programs the riders ride in): the attention's row all-reduce and the MoE's (row_ar2; col_scatter_ar
for the 8-token way back) and the router and routing (route_only).  On each core of the host program without a kernel
of it (an "idle" core) that holds a block of the next x_proj, a rider reads that core's next chunks (up to the host
kind's budget and the region's room) from DRAM on NoC 0 (RISCV_1) into the core's region, then writes the region's
seal (tag << 16) | its chunks.  The decode_matmul takes (a, b) chunks a core (`consume`), each region only when its
seal says exactly that, the rest from DRAM; the seals cleared after.

The regions: L1 tensors on the 12 x 10 grid (the same address on every core), held from the first host to the
decode_matmul, so that tt-metal keeps the circular buffers of every program of that window below them (its check of
each program's CB region against the lowest L1 buffer, at every enqueue and capture: a program whose CBs would reach
into them fails, it never runs over them).  Region A (chunks from 0) from the layer's attention all-reduce over the
router and routing, the experts op and the way back, the next layer's mHC; region B (the chunks after A's) from the
MoE's way back (after the experts op's large CBs are gone).  Each region: 64 B of seal, then chunks of the core's
weight stream (decode_matmul_streams: cr x sb tiles a chunk).

Switches (read at each step):
  GLM53_W1D=1             on (off by default)
  GLM53_W1D_A_KB / _B_KB  the regions' bytes a core (KB; 0: none; B off: its host fills A); by default just what the
                          budgets can fill in one layer (A: ar + router, B: moe_ar; 154 / 60 KB at the default budgets),
                          at most ROOM_KB's for the step's rows a mesh row (the forms whose L1 room was measured:
                          32, i.e. 32 users K = 7); none for the other forms (the riders off)
  GLM53_W1D_BUDGET        "kind=KB,..": the bytes a rider core takes in one program of that kind (ar: the attention's
                          all-reduce, router: the router and routing, moe_ar: the MoE's way back); 0: none there
"""

from __future__ import annotations

import contextlib
import os
from typing import Optional

import ttnn

RIDER = "models/demos/blackhole/glm53_flash/ttnn/kernels/w1_dense/dm_rider.cpp"
GRID = (12, 10)
SEAL = 64
HOSTS = {"ar": "a", "router": "a", "moe_ar": "b"}  # each host kind's region
# rows a mesh row -> (A, B) KB a core: 32 (32 users K = 7, the default and preset C): A under the experts op's down
# CBs (172 KB of room), B under the mHC write-read's (385 KB); 8 rows have 196 KB for A and B together (the 8-token
# way back's own CBs), 64 rows (128 users K = 3) 37 KB for A: not measured for speed, off unless set
ROOM_KB = {32: (164, 112)}
# the KB a rider core takes in one host program (32 users K = 7): 64 / 96 / 64 the step's best in three processes
# (-0.44 / -0.29 / -0.38 ms; 32 / 48 / 32: -0.32 / -0.13 ms; the hosts' own kernels slow with the riders' DRAM reads by
# about what the x_proj saves, so more does not pay)
DEFAULT_BUDGET = {"ar": 64, "router": 96, "moe_ar": 64}

CURRENT: Optional["DenseLanding"] = None  # the step's (`step`), else None
_PRINTED: set = set()


def enabled() -> bool:
    return os.environ.get("GLM53_W1D", "0") == "1"


def _env_kb(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


def kbs(m_rows: int) -> dict:
    """The regions' KB a core at most for a step of `m_rows` rows a mesh row (GLM53_W1D_A_KB / _B_KB, else
    ROOM_KB's)."""
    a, b = ROOM_KB.get(m_rows, (0, 0))
    return {"a": _env_kb("GLM53_W1D_A_KB", a), "b": _env_kb("GLM53_W1D_B_KB", b)}


def _set(r: str) -> bool:
    """The region's size set by its switch (else sized to the budgets, `DenseLanding._kb`)."""
    return os.environ.get(f"GLM53_W1D_{r.upper()}_KB", "").strip() != ""


def _budgets() -> dict:
    out = dict(DEFAULT_BUDGET)
    for item in os.environ.get("GLM53_W1D_BUDGET", "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = int(v)
    return out


def _cores(crs) -> set:
    """A CoreRangeSet's cores as (x, y)."""
    out = set()
    for r in crs.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                out.add((x, y))
    return out


def _grid_mem(words: int):
    """An L1 tensor of `words` 32-bit words a core on the grid (height-sharded: the same address on every core)."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID[0] - 1, GRID[1] - 1))]),
            [1, words],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


class Consumer:
    """A decode_matmul to come (its weight `w`, `m_rows` rows of x, fg's K tiles) and what the riders brought it."""

    def __init__(self, w, m_rows: int, fg_k: int, layer: int):
        self.w, self.layer = w, layer
        self.streams = ttnn.experimental.deepseek_prefill.decode_matmul_streams(w, m_rows, 0, fg_k)
        self.n = len(self.streams)
        self.slot = int(self.streams[0][7])  # a chunk's bytes
        self.tile = self.slot // (int(self.streams[0][8]) * int(self.streams[0][9]))
        self.fill_a = [0] * self.n  # chunks in region A (from chunk 0)
        self.base_b = [None] * self.n  # region B's first chunk (A's count when B's first rider came)
        self.fill_b = [0] * self.n  # chunks in region B
        self.region = {"a": None, "b": None}
        self.cap = {"a": 0, "b": 0}

    def allocate(self, mesh, r: str, kb: int) -> None:
        """Region r ("a" / "b") of kb KB a core (an L1 tensor on the grid, the same address on every core)."""
        words = (kb * 1024) // 4
        self.region[r] = ttnn.allocate_tensor_on_device(
            ttnn.Shape([GRID[0] * GRID[1], words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, _grid_mem(words)
        )
        self.cap[r] = (words * 4 - SEAL) // self.slot

    def ct(self) -> list:
        """The rider kernel's compile-time arguments for this weight (the tile's bytes and its accessor's)."""
        return [self.tile] + list(ttnn.TensorAccessorArgs(self.w).get_compile_time_args())

    def _range(self, i: int, r: str, left: int):
        """Core i's next chunks in region r, at most `left`: (start, end, base) or None."""
        chunks = int(self.streams[i][6])
        if chunks == 0 or left <= 0:
            return None
        if r == "a":
            if self.base_b[i] is not None:  # (B began: A stays as it was)
                return None
            base, start = 0, self.fill_a[i]
            end = min(start + left, self.cap["a"], chunks)
        else:
            base = self.fill_a[i] if self.base_b[i] is None else self.base_b[i]
            start = base + self.fill_b[i]
            end = min(start + left, base + self.cap["b"], chunks)
        return (start, end, base) if end > start else None

    def jobs(self, r: str, busy: set, per: int, tag: int) -> dict:
        """Region r's rider runtime words {(x, y): words}: on each core not in `busy` that holds a block, its next
        chunks, at most `per` of them and as many as the region holds (the counts advanced)."""
        out = {}
        if self.region[r] is None:
            return out
        for i, s in enumerate(self.streams):
            core = (int(s[0]), int(s[1]))
            if core in busy:
                continue
            got = self._range(i, r, per)
            if got is None:
                continue
            start, end, base = got
            x, y, k0, kc, n0, ncb, chunks, _, cr, sb, stride, nt = (int(v) for v in s)
            out[core] = [self.w.buffer_address(), self.region[r].buffer_address(), (tag << 16) | end, k0, n0, ncb,
                         nt, kc, cr, sb, stride, start, end, base]  # fmt: skip
            if r == "a":
                self.fill_a[i] = end
            else:
                self.base_b[i], self.fill_b[i] = base, end - base
        return out

    def riders(self, r: str, busy: set, per: int, tag: int) -> list:
        """Region r's rider kernels alone (tests: every core `per` chunks)."""
        return _kernels(self, self.jobs(r, busy, per, tag))

    def landing(self, tag: int) -> Optional[list]:
        """decode_matmul's `landing`: [A, B, tag, (a, b) a core], or None when nothing landed."""
        a, b = self.region["a"], self.region["b"]
        if not any(self.fill_a) and not any(self.fill_b):
            return None
        out = [a.buffer_address() if a is not None else 0, b.buffer_address() if b is not None else 0, tag]
        for i in range(self.n):
            out += [self.fill_a[i], self.fill_b[i]]
        return out

    def release(self) -> None:
        for k in ("a", "b"):
            if self.region[k] is not None:
                ttnn.deallocate(self.region[k])
                self.region[k] = None


def _kernels(c: Consumer, rts: dict) -> list:
    """The rider kernel descriptor for {(x, y): runtime words} (dm_rider.cpp), [] when none."""
    if not rts:
        return []
    from .common import core_rects

    rt = ttnn.RuntimeArgs()
    for (x, y), words in rts.items():
        rt[x][y] = words
    cfg = ttnn.DataMovementConfigDescriptor()
    cfg.processor, cfg.noc = ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0
    return [
        ttnn.KernelDescriptor(
            kernel_source=RIDER,
            core_ranges=core_rects([ttnn.CoreCoord(x, y) for x, y in rts]),
            compile_time_args=c.ct(),
            runtime_args=rt,
            config=cfg,
        )
    ]


class DenseLanding:
    """A decode step's riders and regions (`step`)."""

    def __init__(self, model, m_rows: int):
        self.model, self.mesh, self.m_rows = model, model.mesh, m_rows
        self.kb = kbs(m_rows)
        self.budget = _budgets()
        self.tag = getattr(model, "_w1d_tag", None)
        if self.tag is None:  # (one a process: a seal of another process's run never matches)
            self.tag = model._w1d_tag = 1 + int.from_bytes(os.urandom(2), "little") % 0x7FFF
        self.ready: Optional[Consumer] = None  # the x_proj this layer takes
        self.next: Optional[Consumer] = None  # the x_proj this layer's riders fill (the next layer's)
        self.kind: Optional[str] = None
        self.riders = 0
        self.sizes = {}  # the regions' KB as allocated
        self.got = []  # (layer, chunks from A, from B, of all) per x_proj taken

    def _region(self, kind: str) -> str:
        """The kind's region: B's host fills A where B is off and A is on."""
        r = HOSTS[kind]
        return "a" if r == "b" and self.kb["b"] <= 0 and self.kb["a"] > 0 else r

    # -- the model's loop
    def at_layer(self, i: int) -> None:
        """Before layer i's step: its x_proj takes what layer i - 1's riders brought; layer i's riders fill layer i +
        1's x_proj (a KDA layer with its fg made inside x_proj's decode_matmul)."""
        from .kda import TtKda, fg_fused_on

        if self.ready is not None:  # (not taken: e.g. by another path)
            self.ready.release()
        self.ready, self.next = self.next, None
        layers = self.model.layers
        if i + 1 < len(layers):
            nxt = layers[i + 1].attention
            if isinstance(nxt, TtKda) and fg_fused_on(nxt.x_proj):
                self.next = Consumer(nxt.x_proj, self.m_rows, int(nxt.fg_proj.shape[-2]) // 32, i + 1)

    def finish(self) -> None:
        for c in (self.ready, self.next):
            if c is not None:
                c.release()
        self.ready = self.next = None
        key = (self.m_rows, tuple(sorted(self.kb.items())), tuple(sorted(self.budget.items())))
        if key not in _PRINTED and self.got:  # (once a form: what one step brought into L1)
            _PRINTED.add(key)
            a, b, n = (sum(v[k] for v in self.got) for k in (1, 2, 3))
            print(f"GLM53_W1D {self.m_rows} rows a mesh row, regions {self.sizes} KB, budgets {self.budget}: "
                  f"{len(self.got)} x_proj, chunks from L1 {a} + {b} of {n} ({100 * (a + b) / max(1, n):.1f} %), "
                  f"{self.riders} programs with riders", flush=True)  # fmt: skip

    # -- the hosts
    @contextlib.contextmanager
    def host(self, kind: str):
        """A program of this kind runs inside (its builder calls `take`); its region is allocated first (so that the
        program's own L1 outputs go below it)."""
        prev, self.kind = self.kind, kind
        c, r = self.next, self._region(kind)
        if c is not None and self.kb[r] > 0 and c.region[r] is None and self.budget.get(kind, 0) > 0:
            self.sizes[r] = self._kb(c, r)
            c.allocate(self.mesh, r, self.sizes[r])
        try:
            yield
        finally:
            self.kind = prev

    def _per(self, kind: str, c: Consumer) -> int:
        """A rider core's chunks in one program of this kind."""
        return max(1, (self.budget[kind] * 1024) // c.slot) if self.budget.get(kind, 0) > 0 else 0

    def _kb(self, c: Consumer, r: str) -> int:
        """Region r's KB: its switch's, else what one layer's hosts of it can fill (at most its room)."""
        if _set(r):
            return self.kb[r]
        need = sum(self._per(k, c) for k in HOSTS if self._region(k) == r)
        return min(self.kb[r], -(-(SEAL + need * c.slot) // 1024))

    def take(self, used) -> list:
        """The rider kernels for the program being built (its kernels' cores `used`, a CoreRangeSet or a set of (x,
        y)), once a call (the same kernels go into every device's program of the call); [] when none."""
        kind, c = self.kind, self.next
        if kind is None or c is None or self.budget.get(kind, 0) <= 0:
            return []
        per = self._per(kind, c)
        rts = c.jobs(self._region(kind), used if isinstance(used, set) else _cores(used), per, self.tag)
        if rts:
            self.riders += 1
        return _kernels(c, rts)

    # -- the consumer
    def consume(self, w) -> Optional[list]:
        """The landing argument for the decode_matmul of weight `w` (the ready x_proj's), or None."""
        c = self.ready
        return c.landing(self.tag) if c is not None and c.w is w else None

    def consumed(self, w) -> None:
        c = self.ready
        if c is None or c.w is not w:
            return
        self.got.append((c.layer, sum(c.fill_a), sum(c.fill_b), sum(int(v[6]) for v in c.streams)))
        c.release()
        self.ready = None


@contextlib.contextmanager
def step(model, m_rows: int):
    """A decode step's riders (GLM53_W1D=1), else nothing: `CURRENT` for the hosts and the consumer inside."""
    global CURRENT
    if not enabled() or CURRENT is not None or not any(v > 0 for v in kbs(m_rows).values()):
        yield None
        return
    # The L1 state the decode step's ops make once and keep, made first (before any region of ours is held, so that
    # it never lands below one and stays low for the process: the row all-reduce's async state's 16 KB landing made
    # under a region clashed with a later program's CBs; tt-metal's guard stopped it).  The model's load makes the
    # fabric ops' state (prealloc_fabric_ops); this one is made at its first use
    key = ("_w1d_warm", m_rows)
    if key not in getattr(model, "_w1d_warmed", set()):
        model._w1d_warmed = getattr(model, "_w1d_warmed", set()) | {key}
        from .common import ar_input_config

        ar_input_config(model.mesh, m_rows, model.config.hidden_size)
    CURRENT = DenseLanding(model, m_rows)
    try:
        yield CURRENT
    finally:
        CURRENT.finish()
        CURRENT = None


def host(kind: str):
    """A host program's context (a no-op outside a step)."""
    return CURRENT.host(kind) if CURRENT is not None else contextlib.nullcontext()


def take(used) -> list:
    return CURRENT.take(used) if CURRENT is not None else []


def consume(w) -> Optional[list]:
    return CURRENT.consume(w) if CURRENT is not None else None


def consumed(w) -> None:
    if CURRENT is not None:
        CURRENT.consumed(w)


__all__ = ["step", "host", "take", "consume", "consumed", "enabled"]
