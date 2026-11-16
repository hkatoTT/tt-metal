# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The decode MoE's routing by generic_op (`route_only`): in place of moe_decode_route
for a mesh column's experts (`local` 72, the column layout), the same rows bit for bit, faster at every decode shape.

The token rows are dealt out over up to ROUTE_WORKERS cores (`plan`: 2 a core where the cores suffice, one for each
data-movement RISC; 16 a core at most; a core's rows inside one 16-row group of the logits' tiles when that costs no
row more on the busiest core).  A core's reader takes its rows of the logits in the block layout (kernels/route/
rt_common.hpp: one read a block of 16 experts a 16-row group, not one a row), its writer the bias's row once (copied in
L1 to the other rows), the compute makes the scores sigmoid(logits) and the choice keys scores + bias (the same SFPU
steps), and the writer writes each row as moe_decode_route does: the top 8 by the keys as ordered integers (ties to
the lower expert), the chosen scores summed in increasing expert order, bf16((score * (1 / (sum + 1e-20))) * 2.5) for
the column's experts, then the shared expert; with ROUTE_SHARE the reader routes the second half of a core's rows (its
writer the first).  Against moe_decode_route (trace): 1 / 8 / 32 / 128 / 512 / 1,024 tokens 21.7 / 20.5 / 21.1 / 31.7 /
62.8 / 105.5 -> 15.2 / 13.6 / 15.9 / 23.1 / 59.2 / 88.0 us (with one read a row and the bias read for every row, as
in an earlier form, 21.5 of the op's 35.5 us of kernel time at 256 tokens went in those small DRAM reads).  Only inside
a decode step (`decoding`: `forward`, `forward_spec`); the prefill keeps moe_decode_route.

A decode step's tokens of seats without a request (position -1: `LiveTokens`) choose no expert: their rows are written
zero, so the experts op reads no weights for them (the requests' rows the same).
"""

from __future__ import annotations

import struct
from contextlib import contextmanager
from dataclasses import dataclass

import torch

import ttnn

from .common import COLS, ROWS, put, write_rows

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/route/"
# the decode's routing by `route_only` (the same rows, decode 8 / 32 users -0.2 ms a step, the rest within the
# noise; 2,000-step stresses clean).  Read at each call (tests switch it).  False: moe_decode_route (the prefill's)
ROUTE_FAST = True
ROUTE_WORKERS = 120  # cores at most (the rows dealt out evenly)
# a core's rows over both its data-movement RISCs (the writer alone was slower than moe_decode_route at 256 / 512
# tokens: 40.4 / 62.8 -> 43.6 / 69.3 us).  False: the writer alone (tests switch these at each call)
ROUTE_SHARE = True
MAX_ROWS = 16  # a core's rows at most (its scratch's staging)
# rows a core while the cores suffice (2: one for each RISC that routes); 0: dealt out evenly over ROUTE_WORKERS
ROUTE_PAIR = 2
# a core's rows inside one 16-row group of the logits when that keeps the busiest core's rows (one read a block then)
ROUTE_GROUPS = True
# cores that read the bias from DRAM, each writing it on to the cores after it (every core reading the same words was
# slow); the other cores wait for theirs
ROUTE_BIAS_SEEDS = 8
SEM_BIAS = 0
SEM_LOGITS = 1  # (router_mm.router_route: the router's cores tell the readers that the logits are written)
# up to this many tokens the reader reads the bias for each of its rows (few reads; the compute then waits for no
# seed), above it the seeds (16 tokens 1.0 us slower with the seeds than with one read a row)
ROUTE_BIAS_HERE_MAX = 32
PER = 72  # a mesh column's experts
SCR_BYTES = 192 + 16 * 2 * 64 + 64 + 256 + 64 + 64  # kernels/route/rt_common.hpp
# a core of 3 rows or more gives a third of them to its compute's unpack thread (kernels/route/rt_compute.cpp; the
# writer writes them): three routers a core where the 256 tokens' 16 cores of 3 rows had the writer route 2 in a row.
# Read at each call (tests switch it).  False: the writer and the reader only
ROUTE_TRISC = True
CB_LOG, CB_BIAS, CB_SC, CB_CH, CB_SCR, CB_TMP = 0, 1, 16, 17, 24, 25
# the shared expert spread over the mesh columns: in a decode step of SHARED_SPLIT_MIN tokens of the mesh or more (past
# the MoE's SHARED_IN_OP_MAX, where the experts op leaves it to the TP4 shared MLP), column c's chips add the shared
# expert (slot 72, weight 1) to the tokens c n / 4 .. (c + 1) n / 4 - 1 only (`TtMoe.local_base_split`; rt_common.hpp
# `shared_span`), so each column's experts op computes it for a quarter of the tokens and the reduce-scatter and
# all-reduce sum it, in place of the TP4 MLP (2 ops of matmul, 2 slices, the multiply and the add: 1 layer ~39 us at 128
# users' verify step).  Synthetic weights, traced, without / with: the verify step at 128 / 256 users 52.95 / 68.86 ->
# 51.87 / 67.98 ms, the plain step at 128 / 256 / 512 users 47.27 / 51.42 / 68.61 -> 46.18 / 50.50 / 67.69 ms (-0.9 ..
# -1.1 ms; an earlier form before route_only: 128 / 512 users -0.3 / +0.8 ms).  The real weights' teacher forcing at 128
# users the same (KLD 0.00256 / 0.00249 -> 0.00237 / 0.00240); 2,000-step traced stress at the verify step's 128 users
# and the plain step's 128 / 256 users and 4 x 480 at 512 without a hang.  Read at each call (tests switch it).  False:
# the TP4 MLP, as before (tests)
SHARED_SPLIT = True
SHARED_SPLIT_MIN = 128

LIVE_MAX = 1024  # tokens of the mesh the mask covers (kernels/route/rt_common.hpp: 64 words at LIVE)

_DECODE: list = [False, None]  # inside a decode step (`decoding`), its LiveTokens


@dataclass
class LiveTokens:
    """A decode step's live tokens of the mesh (t a row: row r's at r t .., as the MoE gathers them) as a bit mask at a
    fixed address, the same on every chip: bit i % 16 of word i / 16 set for a token with a request (`route_only`
    writes the others' routing rows zero).  Made all live; written with the step's inputs (`write`)."""

    tensor: object  # [1, words] uint32 row-major per chip
    tokens: int  # of the mesh: ROWS t
    last: object = None  # the host words last written

    @staticmethod
    def host(live: torch.Tensor) -> torch.Tensor:
        """live [ROWS, t] bool (the rows' tokens in order) -> [ROWS, words] int32: the mask, the same in every row."""
        flat = live.reshape(-1).to(torch.int32)
        words = -(-flat.numel() // 16)
        bits = torch.zeros(words * 16, dtype=torch.int32)
        bits[: flat.numel()] = flat
        w = (bits.view(words, 16) << torch.arange(16, dtype=torch.int32)).sum(1, dtype=torch.int32)
        return w.expand(ROWS, words).contiguous()

    @classmethod
    def alloc(cls, mesh, t: int) -> "LiveTokens":
        live = cls.host(torch.ones(ROWS, t, dtype=torch.bool))
        tensor = put(mesh, live, dims=(0, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return cls(tensor=tensor, tokens=ROWS * t, last=live)

    def write(self, mesh, live: torch.Tensor, pack=None) -> None:
        """live [ROWS, t] bool -> the device mask: into the step's input pack (its one write), else written apart
        when it changed."""
        words = self.host(live)
        if pack is not None and pack.has(self.tensor):
            pack.set(self.tensor, words)
        elif not torch.equal(words, self.last):
            write_rows(mesh, words, self.tensor, ttnn.uint32)
        self.last = words


@contextmanager
def decoding(live: LiveTokens | None = None):
    """A decode step's layers (`forward`, `forward_spec`): where ROUTE_FAST applies; `live`: its tokens' mask."""
    old = list(_DECODE)
    _DECODE[0], _DECODE[1] = True, live
    try:
        yield
    finally:
        _DECODE[:] = old


def in_decode() -> bool:
    """Whether a decode step's layers run now (`decoding`)."""
    return bool(_DECODE[0])


def fits(moe, tokens: int) -> bool:
    """Whether `route_only` takes this call: ROUTE_FAST, a decode step, a column's experts, MAX_ROWS rows a core at
    most."""
    return ROUTE_FAST and _DECODE[0] and moe.op_experts == PER and -(-tokens // ROUTE_WORKERS) <= MAX_ROWS


def fits_prefill(moe, tokens: int) -> bool:
    """Whether `route_only` takes a prefill chunk's routing (moe.PREFILL_ROUTE_ONLY): a column's experts, MAX_ROWS
    rows a core at most, outside a decode step (so `_shared_mode` gives 0: moe_decode_route's shared=False)."""
    return not _DECODE[0] and moe.op_experts == PER and -(-tokens // ROUTE_WORKERS) <= MAX_ROWS


def split(tokens: int, workers: int) -> list:
    """Token rows 0 .. tokens - 1 dealt out evenly over at most `workers` -> [(start, count)], the first ones a row
    more."""
    k = max(1, min(tokens, workers))
    base, extra = divmod(tokens, k)
    out, start = [], 0
    for i in range(k):
        count = base + (1 if i < extra else 0)
        out.append((start, count))
        start += count
    return out


def ranges(cores: list) -> ttnn.CoreRangeSet:
    """Cores -> a CoreRangeSet of their runs along each grid row (few ranges: the dispatch sends each range's once)."""
    rows: dict = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    out = []
    for y, xs in sorted(rows.items()):
        xs = sorted(xs)
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            out.append(ttnn.CoreRange(ttnn.CoreCoord(start, y), ttnn.CoreCoord(prev, y)))
            if x is not None:
                start = prev = x
    return ttnn.CoreRangeSet(out)


def _cb(index: int, fmt, page: int, pages: int, cores) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def plan(tokens: int) -> list:
    """Token rows 0 .. tokens - 1 -> [(start, count)] a core: ROUTE_PAIR rows a core while ROUTE_WORKERS cores suffice
    (one for each RISC that routes), else dealt out evenly; with ROUTE_GROUPS each core's rows inside one 16-row group
    of the logits' tiles (one read a block, `rt_reader.cpp`) if the busiest core gets no row more that way."""
    workers = min(ROUTE_WORKERS, tokens)
    if ROUTE_PAIR:
        workers = min(workers, -(-tokens // ROUTE_PAIR))
    even = split(tokens, workers)
    if not ROUTE_GROUPS:
        return even
    sizes = [min(16, tokens - g) for g in range(0, tokens, 16)]
    if len(sizes) > workers:
        return even
    # cores a group in proportion to its rows (at least 1), the rest to the groups with the most rows a core
    cores = [max(1, workers * size // tokens) for size in sizes]
    while sum(cores) < workers:
        i = max(range(len(sizes)), key=lambda g: (-(-sizes[g] // cores[g]), sizes[g]))
        if cores[i] >= sizes[i]:
            break
        cores[i] += 1
    grouped, base = [], 0
    for size, c in zip(sizes, cores):
        grouped += [(base + start, count) for start, count in split(size, min(c, size))]
        base += size
    busiest = lambda parts: max(count for _, count in parts)  # noqa: E731
    return grouped if busiest(grouped) <= busiest(even) and len(grouped) <= ROUTE_WORKERS else even


def route_only(mesh, moe, logits, tokens: int, h1=None, router=None):
    """moe_decode_route(logits, moe.router_bias, moe.local_base, tokens, moe.k, 72, moe.scale, shared=
    moe._shared_mode(n, tokens) == 1) for the column layout: logits [1, 1, n, 288] fp32 tiled -> w [1, 1, n, 32] fp32
    tiled, the same rows bit for bit (rows from `tokens` on zero); with the shared expert spread over the columns
    (`_shared_mode` 2, SHARED_SPLIT) each column adds it to its quarter of the tokens.

    `h1` (w_all, the moe's experts op H1, moe.py `_routed_h1`): the logits are a mesh row's own `tokens` tokens, and
    mesh row r writes their rows into the column's w_all [8 tokens, 32] at rows r tokens .. (the rows the other mesh
    rows write stay; the shared expert's span counted over the column's 8 tokens, as the column's routing does); returns
    w_all.

    `router` ((x, router_wt), router_mm.router_route): the router's matmul in the same program, writing `logits` (here
    allocated, unwritten) on the grid's last rows; the readers wait for its cores before reading them."""
    n = logits.shape[-2]
    col = ROWS * tokens if h1 is not None else tokens  # the column's tokens (the shared expert's span)
    mode = moe._shared_mode(ROWS * tokens if h1 is not None else n, col)
    local_base = moe.local_base_split if mode == 2 else moe.local_base
    live = _DECODE[1]
    if live is not None and (live.tokens != col or col > LIVE_MAX):  # (H1: the column's tokens, row r's at r tokens)
        live = None  # (another step's tokens: the MTP layer's first drafts)
    live_t = live.tensor if live is not None else local_base  # (a stand-in for the unused accessor)
    live_args = [live_t.buffer_address(), live_t.buffer_page_size()] if live is not None else [0, 0]
    grid = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(grid.x * grid.y)]
    parts = plan(tokens)
    per = max(count for _, count in parts)
    if per > MAX_ROWS or len(parts) > len(cores):
        raise ValueError(f"{tokens} tokens: {per} rows a core ({MAX_ROWS} at most) on {len(parts)} cores")
    workers = cores[: len(parts)]
    everywhere = ranges(workers)
    et = 9  # tiles of 288
    np_tiles = -(-per * et * 32 // 1024)  # a core's rows in the block layout (kernels/route/rt_common.hpp)
    if h1 is not None:
        w = h1
        assert tuple(w.shape) == (1, 1, ROWS * tokens, 32) and ROWS * tokens % 32 == 0, (w.shape, tokens)
    else:
        spec = ttnn.TensorSpec(
            ttnn.Shape([1, 1, n, 32]), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        )
        w = ttnn.allocate_tensor_on_device(spec, mesh)
    # the bias's seeds: the first core of each of up to ROUTE_BIAS_SEEDS runs of cores, writing it on to the run's
    # others
    # (none with the bias read by each reader)
    bias_here = int(tokens <= ROUTE_BIAS_HERE_MAX)
    seeds = len(workers) if bias_here else max(1, min(ROUTE_BIAS_SEEDS, len(workers)))
    runs = split(len(workers), seeds)
    span = max(count for _, count in runs)  # (the runtime args padded to the longest run: their count fixed)
    seed_of = {}
    for first, size in runs:
        for i in range(first, first + size):
            seed_of[i] = first

    rparts = None
    if router is not None:
        from . import router_mm

        notify = [(xy.x, xy.y) for xy in (mesh.worker_core_from_logical_core(c) for c in workers)]
        rparts = router_mm.router_parts(mesh, router[0], router[1], logits, notify, SEM_LOGITS)
        taken = {(c.x, c.y) for c in rparts[3]}
        assert not any((c.x, c.y) in taken for c in workers), "the router's cores overlap the routing's"

    def program(out_base: int):
        p = _program_core(out_base)
        if rparts is None:
            return p
        kernels, cbs, sems, rcores, count = rparts
        union = ranges(workers + rcores)
        sems = sems + [ttnn.SemaphoreDescriptor(id=SEM_LOGITS, core_ranges=union, initial_value=0)]
        return ttnn.ProgramDescriptor(
            kernels=list(p.kernels) + kernels, semaphores=list(p.semaphores) + sems, cbs=list(p.cbs) + cbs
        )

    def _program_core(out_base: int):
        return _program(
            mesh,
            moe,
            logits,
            local_base,
            w,
            tokens,
            n,
            mode,
            workers,
            parts,
            everywhere,
            np_tiles,
            bias_here,
            seed_of,
            span,
            out_base,
            col,
            live,
            live_t,
            live_args,
            fused=rparts[4] if rparts is not None else 0,
        )

    mesh_program = ttnn.MeshProgramDescriptor()
    if h1 is None:
        whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
        mesh_program[whole] = program(0)
    else:
        for r in range(ROWS):  # mesh row r's rows at r tokens ..
            row = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, 0), ttnn.MeshCoordinate(r, COLS - 1))
            mesh_program[row] = program(r * tokens)
    inputs = [logits, moe.router_bias, local_base] + ([live.tensor] if live is not None else [])
    if router is not None:
        inputs = [router[0], router[1]] + inputs
    ttnn.generic_op(inputs + [w], mesh_program)
    return w


def _program(
    mesh,
    moe,
    logits,
    local_base,
    w,
    tokens,
    n,
    mode,
    workers,
    parts,
    everywhere,
    np_tiles,
    bias_here,
    seed_of,
    span,
    out_base,
    col,
    live,
    live_t,
    live_args,
    fused: int = 0,
):
    """route_only's program (its rows written from row out_base of w on; the shared expert's span over `col` tokens:
    no zero rows past `tokens` when out_base is set).  `fused` (router_mm.router_route): the readers wait for that many
    adds to SEM_LOGITS (the router's cores, its logits written) before reading the logits."""
    et = 9  # tiles of 288
    acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    # the reader RISCV_1 and the writer RISCV_0 on NoC 0 / NoC 1 (Blackhole's Reader / WriterConfigDescriptor); with
    # the router in the same program and its x multicast on NoC 1 (router_mm) the other way round: the writer's DRAM
    # reads at its start (the bias, the base) then on NoC 0 with the router's weight stream, none on the multicast's
    # NoC while it runs (the reader reads only once the router has written the logits)
    swap = bool(fused)

    def dm(processor, noc):
        cfg = ttnn.DataMovementConfigDescriptor()
        cfg.processor, cfg.noc = processor, noc
        return cfg

    reader_cfg = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1 if swap else ttnn.NOC.NOC_0)
    writer_cfg = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0 if swap else ttnn.NOC.NOC_1)
    n_rows = tokens if out_base or col != tokens else n
    for i, (c, (start, count)) in enumerate(zip(workers, parts)):
        rd[c.x][c.y] = (
            [
                logits.buffer_address(),
                local_base.buffer_address(),
                w.buffer_address(),
                start,
                count,
                moe.router_bias.buffer_address(),
                tokens,
            ]
            + live_args
            + [out_base, col]
        )  # (reader runtime args 7 / 8 the live mask, 9 / 10 H1's)
        cp[c.x][c.y] = [count]
        receivers = [j for j in range(len(workers)) if seed_of[j] == i and j != i]
        coords = []
        for j in receivers:
            xy = mesh.worker_core_from_logical_core(workers[j])
            coords += [xy.x, xy.y]
        coords += [0, 0] * (span - 1 - len(receivers))
        wr[c.x][c.y] = [
            moe.router_bias.buffer_address(),
            local_base.buffer_address(),
            w.buffer_address(),
            start,
            count,
            i,
            len(parts),
            tokens,
            n_rows,
            int(seed_of[i] != i),
            len(receivers),
            out_base,
            col,
        ] + coords
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[CB_LOG] = modes[CB_BIAS] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    cfg.unpack_to_dest_mode = modes
    shared = int(mode != 0)
    scale = struct.unpack("<I", struct.pack("<f", float(moe.scale)))[0]
    base_page = 32  # the local base page: [1, 8] uint32
    k_arg = moe.k
    if getattr(moe, "prog", False):  # (the progressive form: every pick's global id too, rt_common.hpp stage_row)
        k_arg |= 1 << 16
    common = [k_arg, moe.op_experts, scale, shared, base_page]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rt_reader.cpp",
            core_ranges=everywhere,
            compile_time_args=[CB_LOG, CB_SC, CB_CH, CB_SCR, et, logits.shape[-1] // 32, int(ROUTE_SHARE)]
            + common
            + [bias_here, CB_BIAS, int(ROUTE_TRISC), int(live is not None), fused, SEM_LOGITS]
            + acc(logits)
            + acc(local_base)
            + acc(w)
            + acc(moe.router_bias)
            + acc(live_t),
            runtime_args=rd,
            config=reader_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rt_compute.cpp",
            core_ranges=everywhere,
            compile_time_args=[CB_LOG, CB_BIAS, CB_SC, CB_CH, et * 32, CB_SCR, k_arg, moe.op_experts, scale]
            + [int(ROUTE_TRISC), int(ROUTE_SHARE)],
            runtime_args=cp,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rt_writer.cpp",
            core_ranges=everywhere,
            compile_time_args=[CB_BIAS, CB_SC, CB_CH, CB_SCR, et]
            + common
            + [int(ROUTE_SHARE), CB_TMP, SEM_BIAS, bias_here, int(ROUTE_TRISC), int(live is not None)]
            + acc(moe.router_bias)
            + acc(local_base)
            + acc(w),
            runtime_args=wr,
            config=writer_cfg,
        ),
    ]
    f32 = ttnn.float32
    cbs = [
        _cb(CB_SCR, ttnn.uint32, SCR_BYTES, 1, everywhere),
        _cb(CB_LOG, f32, 4096, np_tiles, everywhere),
        _cb(CB_BIAS, f32, 4096, np_tiles, everywhere),
        _cb(CB_SC, f32, 4096, np_tiles, everywhere),
        _cb(CB_CH, f32, 4096, np_tiles, everywhere),
        _cb(CB_TMP, ttnn.uint32, 2 * et * 64, 1, everywhere),  # the bias's row 0 (rt_writer.cpp)
    ]
    sems = [ttnn.SemaphoreDescriptor(id=SEM_BIAS, core_ranges=everywhere, initial_value=0)]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
