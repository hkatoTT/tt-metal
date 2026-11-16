# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The prefill MoE that reads each expert's weights once a chunk, through ttnn.generic_op, the
same program on every chip (each chip its own 256 intermediate columns of its mesh column's 72 experts, col8):

  plan(route): the route op's rows [1, 1, N, 32] fp32 (columns 0-7 the chosen local experts, 16-23 their weights) ->
      the pair table: per expert its pairs at tile-row-aligned offsets (seg), per pair row its token and weight (ptok,
      pw), per token its pair rows (tpair), on 32 + 72 cores in two programs;
  gate_up(x_rm, e_w1): per pair row h = silu(min(g, limit)) x clamp(u, -limit, limit) x weight, bf16 -> H [M, 256];
  down(H, e_w2): y = h W2 per pair row, fp32 -> Y [M, 4096] row-major;
  combine(Y, tpair): per token the sum of its pairs' rows (fp32), bf16 -> [N, 4096] tiled.

`experts` runs the four on the column's tokens [1, 1, N, 4096] (tiled) -> this chip's partial sums [1, 1, N, 4096]
(tiled), as `moe_decode_experts` gives them.  M (the table's rows) is fixed by N for a trace: 8 N + 72 x 31, rounded
to a tile row; only the rows in use are computed.
"""

from __future__ import annotations

import os
import struct

import ttnn

from .common import COLS, ROWS, core_rects, fidelity

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/moe_prefill/"
LOCAL = 72  # a mesh column's experts
SEG_WORDS = 256
K = 8
TILE_BF16, TILE_FP32, TILE_BFP8, TILE_BFP4 = 2048, 4096, 1088, 576
# gate | up's and down's matmuls (on the weights) at GLM53_MOE_PREFILL_FIDELITY (HiFi4 by default); the reductions and
# SwiGLU are SFPU's, not affected
# (auto: by the weights' format, LoFi on BFP4 and HiFi4 else: layers of mixed formats)
FIDELITY_NAME = os.environ.get("GLM53_MOE_PREFILL_FIDELITY", "hifi4")
DOWN_FIDELITY_NAME = os.environ.get("GLM53_MOE_PREFILL_DOWN_FIDELITY", FIDELITY_NAME)
FIDELITY = None if FIDELITY_NAME == "auto" else fidelity("GLM53_MOE_PREFILL_FIDELITY")
DOWN_FIDELITY = None if DOWN_FIDELITY_NAME == "auto" else fidelity("GLM53_MOE_PREFILL_DOWN_FIDELITY", FIDELITY_NAME)


def _fidelity(fixed, w):
    """`fixed`, or by the weights' format (auto)."""
    if fixed is not None:
        return fixed
    return ttnn.MathFidelity.LoFi if w.dtype == ttnn.bfloat4_b else ttnn.MathFidelity.HiFi4


def _wfmt(w) -> tuple:
    """A weight tensor's CB format and tile bytes: BFP8, or BFP4 (the low-precision patterns)."""
    return (ttnn.bfloat4_b, TILE_BFP4) if w.dtype == ttnn.bfloat4_b else (ttnn.bfloat8_b, TILE_BFP8)


def table_rows(n: int) -> int:
    return -(-(K * n + LOCAL * 31) // 32) * 32


def _run(mesh, tensors: list, program) -> None:
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    ttnn.generic_op(tensors, mesh_program)


def _cb(index: int, fmt, page: int, pages: int, cores) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def _alloc(mesh, shape, dtype, layout):
    return ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(shape), dtype, layout, buffer_type=ttnn.BufferType.DRAM), mesh
    )


def _acc(*tensors) -> list:
    out = []
    for t in tensors:
        out += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    return out


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    coords = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
    return coords


def _spread(mesh, n_items: int):
    """Items 0 .. n - 1 round robin over the cores -> [(core, [items])] for the cores with any."""
    coords = _grid(mesh)
    per = [[] for _ in coords]
    for i in range(n_items):
        per[i % len(coords)].append(i)
    work = [(c, items) for c, items in zip(coords, per) if items]
    cores = core_rects([c for c, _ in work])
    return work, cores


# (mp_plan_push sums the counts in one pass of plain loads, a token core's prefix on the way: the same table, the op
# -0.015 ms at 1,024 tokens a column.  tpair, each token's pair rows of 16 words, a page a token tile row (2 KB) in
# place of a page a token (64 bytes): combine reads an item's 32 tokens' rows in one request, not 32 (its 32 small
# reads took ~8-10 us an item): the experts op 0.963 -> 0.952 ms at 1,024 tokens a column, 1.863 -> 1.833 at 4,096
# (uniform; real text's 1.070 -> 1.061, 2.108 -> 2.077), the same words in the same order.  The forms before,
# GLM53_MOE_PLAN_ONE_PASS=0 and GLM53_MOE_PREFILL_TPAIR_ROWS=0, were removed)


class PlanPush:
    """`plan(.., defer_push=True)`'s second program, not run: its kernel, semaphore and CB on its cores (none of the
    `avoid` ones) and its tensors, for a program on other cores to carry (`x2d_gather(.., extra=)`: the push beside
    the x all-gather); `done()` after that program (the counts freed)."""

    def __init__(self, kernels, sems, cbs, tensors, cnt):
        self.kernels, self.semaphores, self.cbs, self.tensors, self.cnt = kernels, sems, cbs, tensors, cnt

    def done(self) -> None:
        ttnn.deallocate(self.cnt)


def plan(mesh, route, mask=None, avoid=(), defer_push=False):
    """route [1, 1, N, 32] fp32 tiled -> (seg [1, 1, 1, 256], ptok [1, 1, M / 16, 16], pw (the same, fp32), tpair [1, 1,
    N / 32, 512]) uint32 / fp32 row-major; `mask` [1, 1, N, 1] bf16 tiled (or None): its 0 tokens have no pairs.  Two
    programs: P token cores count the pairs per expert over their tile rows (mp_plan_count.cpp), then the token cores
    push their pairs into the 72 expert cores' L1 and write their tokens' pair rows, and the expert cores their experts'
    segments of the table (mp_plan_push.cpp: 0.49 -> 0.13 ms at 4,096 tokens against each expert core scanning every
    route row), each core only its own pages.  `avoid`: (x, y) cores the two programs leave out (the cores in
    order otherwise, the same tables); `defer_push`: both as one program, not run, returned (`PlanPush`) after the
    tables (the count on the token cores' BRISC; each count core then increments S_COUNTED on one gathering core, the
    first push core, which once it has the P of them sets S_GO and writes it to the other push cores: P atomics, no
    more; the push reads the counts after S_GO).  (The push alone deferred, the count run before, was removed.)"""
    n = route.shape[-2]
    m, tiles = table_rows(n), n // 32
    p = min(32, tiles)
    rm = ttnn.ROW_MAJOR_LAYOUT
    cnt = _alloc(mesh, [1, 1, p, 128], ttnn.uint32, rm)
    seg = _alloc(mesh, [1, 1, 1, SEG_WORDS], ttnn.uint32, rm)
    ptok = _alloc(mesh, [1, 1, m // 16, 16], ttnn.uint32, rm)
    pw = _alloc(mesh, [1, 1, m // 16, 16], ttnn.float32, rm)
    tpair = _alloc(mesh, [1, 1, n // 32, 512], ttnn.uint32, rm)
    skip = set(avoid)
    coords = [c for c in _grid(mesh) if (c.x, c.y) not in skip]
    assert len(coords) >= p + LOCAL, (len(coords), p + LOCAL)
    ranges = [(c * tiles // p, (c + 1) * tiles // p) for c in range(p)]
    fused = defer_push
    s_counted, s_go = 1, 2  # (the push's own semaphore is 0)
    noc = mesh.worker_core_from_logical_core
    table_cores = core_rects(coords[: p + LOCAL])
    all_xys = [noc(c).x | noc(c).y << 16 for c in coords[: p + LOCAL]]  # (the gathering core: the first)

    count_cores = core_rects(coords[:p])
    masked = [int(mask is not None)] + _acc(mask if mask is not None else route)
    mask_addr = mask.buffer_address() if mask is not None else 0
    rt = ttnn.RuntimeArgs()
    for c, (tr0, tr1) in enumerate(ranges):
        rt[coords[c].x][coords[c].y] = [route.buffer_address(), cnt.buffer_address(), c, tr0, tr1, mask_addr] + (
            [all_xys[0]] if fused else []
        )
    count_cb = 1 if fused else 0  # (beside the push's CB 0, made after it: the push's at one address on its cores)
    count_kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "mp_plan_count.cpp",
        core_ranges=count_cores,
        compile_time_args=[LOCAL, count_cb] + _acc(route, cnt) + masked + [int(fused), s_counted],
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor() if fused else ttnn.ReaderConfigDescriptor(),
    )
    count_cbs = [_cb(count_cb, ttnn.uint32, 2 * 4096 + 512 + 2048, 1, count_cores)]
    if not fused:
        program = ttnn.ProgramDescriptor(kernels=[count_kernel], semaphores=[], cbs=count_cbs)
        _run(mesh, [route, cnt] + ([mask] if mask is not None else []), program)

    rt = ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in (route, cnt, seg, ptok, pw, tpair)]
    xys = all_xys[p:]
    stage = -(-(-(-tiles // p) * 32 * K + LOCAL * 8) // 4) * 4  # a token core's pairs at most, + alignment
    for c in range(p + LOCAL):
        role, who = (0, c) if c < p else (1, c - p)
        tr0, tr1 = ranges[c] if c < p else (0, 0)
        tail = ([1, len(all_xys) - 1] + all_xys[1:] if c == 0 else [0]) if fused else []
        rt[coords[c].x][coords[c].y] = addrs + [role, who, tr0, tr1, mask_addr] + xys + tail
    l1 = p * 512 + SEG_WORDS * 4 + 2 * 4096 + 64 + max(2 * (n + 32) * 4, 3 * 512 + 64 * 64 + 2048 + 2 * stage * 4)
    sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=table_cores, initial_value=0)]
    if fused:
        sems += [ttnn.SemaphoreDescriptor(id=i, core_ranges=table_cores, initial_value=0) for i in (s_counted, s_go)]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "mp_plan_push.cpp",
        core_ranges=table_cores,
        compile_time_args=[tiles, p, LOCAL, 0, stage]
        + _acc(route, cnt, seg, ptok, pw, tpair)
        + masked
        + [int(fused), s_counted, s_go],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cbs = [_cb(0, ttnn.uint32, l1, 1, table_cores)]
    tensors = [route, cnt, seg, ptok, pw, tpair] + ([mask] if mask is not None else [])
    if defer_push:
        return seg, ptok, pw, tpair, PlanPush([kernel, count_kernel], sems, cbs + count_cbs, tensors, cnt)
    _run(mesh, tensors, ttnn.ProgramDescriptor(kernels=[kernel], semaphores=sems, cbs=cbs))
    ttnn.deallocate(cnt)
    return seg, ptok, pw, tpair


def _groups(mesh) -> list[list]:
    """The cores in groups of 8 whose rectangles a multicast can cover: 8 x 1 along each row of the grid, then the
    columns left over in 4 x 2 blocks (a 12 x 10 grid: 10 + 5 groups) -> [[member 0 .. 7 CoreCoord]]."""
    g = mesh.compute_with_storage_grid_size()
    out = [[ttnn.CoreCoord(x0 + i, y) for i in range(8)] for y in range(g.y) for x0 in range(0, g.x - 7, 8)]
    left = g.x - g.x // 8 * 8
    if left == 4:
        for y in range(0, g.y - 1, 2):
            out.append([ttnn.CoreCoord(g.x - 4 + i % 4, y + i // 4) for i in range(8)])
    return out


# gate | up (mp_gu5_*.cpp) by groups of 8 cores (`_groups`), split along K: each member its x block of every row
# against W1's K block of all 8 column tiles; the partial sums, 8 KB fp32 a column tile, to the tile's owner, which
# sums the 8 in fp32 (gate_up 1.26 -> 0.93 ms against the column tiles' split, mp_gu4; 3 x 2,000 runs without a
# hang).  The reader reads each row's x with the next row's token pages and an even share of the next expert's W1, one
# barrier a row.
# mp_gu5's reader on both NoCs (2): GU5_NOC1_QUARTERS of every 4 of its DRAM reads (W1's tiles of a K row, x rows) on
# NoC 1, handed over by the writer (which only writes after its seg page; mp_noc1.hpp) as decode_matmul's weight
# stream.  On NoC 0 alone the cores of the grid's low rows read DRAM about half as fast as the high rows' (the same
# work ended at ~890 against ~430 us; their W1 waits 400 against 20-40 us); on both, every row ends at 600-680 us.
# gate_up 0.94 -> 0.79 ms at 4,096 tokens, 0.59 -> 0.50 at 512; 2 quarters is the best (1: 0.84, 3: 1.05); the prefill
# at chunks of 512 (synthetic, traced) 1 x 8,192 / 32,768 14,767 / 15,038 -> 15,098 / 15,410 tok/s, 8 x 2,048 / 8,192
# 16,637 / 16,374 -> 17,123 / 16,834.  The same values.  The default (5 x 2,000 runs of the experts at 4,096 tokens and
# 2,000 at 512, 2 x 600 rounds of the prefill stress without a hang).  1: NoC 0 only
GU5_NOCS = 2
GU5_NOC1_QUARTERS = 2
# ... each core's share by its place on the die: rules "x>=11:1,y<=4:3" (physical NoC 0 x, y: the harvested columns
# counted, as the reader takes them from its NIU; later rules win) over GU5_NOC1_QUARTERS, from GU5_NOC1_RULES_MIN
# tokens a column up.  Under all cores reading DRAM alone, NoC 1's reads end by physical column (x = 1 first, the
# right edge ~4 times later) and NoC 0's by row (the top rows last); in mp_gu5 more NoC 1 anywhere
# was slower.  The grid's top two rows (physical y 2, 3) at 1 of 4: at 4,096 tokens a column 0.885 -> 0.828 ms with
# real text's routing (layers 0 / 10 / 20 / 30 / 40: -10 / -5 / -7 / +2 / -10 %), uniform 0.790 -> 0.782; at 2,048
# real +6 %, at 1,024 +4 % (so not below 4,096).  The same values on every chip
GU5_NOC1_RULES = "y<=3:1"
GU5_NOC1_RULES_MIN = 4096
# mp_gu5's members' roles (a member's K block, its slot in the others' G5_RED, its output column tile: role c reads
# W1's shards of K block c, all on DRAM bank c) rotated by GU5_ROLE_ROT x the group's index: in place of role c on
# the group's c-th core.  At 1,024 tokens a column the readers of physical column 5 (role 4 in each row group: bank
# 4) ended ~450 us, the rest of the row 370-400, and the whole group waits for them each row (profiler zones); W1's
# reads alone 413 -> 387 us (a read bench).  The same values (each role's work and the sums'
# order unchanged).  0: role c on core c
GU5_ROLE_ROT = 3
# (mp_gu5's reader reads a K row's (gate, up) tiles of a column tile, adjacent in their bank, in one request of 2,176
# bytes in place of two, NoC 1 for GU5_NOC1_QUARTERS of every 4 pairs in place of tiles: W1's and x's reads alone at
# 1,024 tokens a column 433 -> 396 us, at 4,096 575 -> 538 us, with GU5_ROLE_ROT 3, a read bench; the
# experts op 1.022 -> 1.014 ms at 1,024 tokens, 1.895 -> 1.863 at 4,096, uniform routing; real text's 1.133 -> 1.122,
# 2.122 -> 2.111.  The same bytes into the same places; a tile a request, GLM53_MOE_GU5_W1_PAIRS=0, was removed)
Q1_X, Q1_Y = 17, 12  # mp_common.hpp


def _q1_table(base: int, rules: str) -> list[int]:
    """mp_gu5_reader's NoC 1 shares: 2 bits for each physical (x, y), y-major, 16 to a word."""
    q = [[base] * Q1_X for _ in range(Q1_Y)]
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b}
    for rule in filter(None, rules.split(",")):
        cond, v = rule.split(":")
        axis, op, bound = cond[0], cond[1:3], int(cond[3:])
        assert axis in "xy" and op in ops and 0 <= int(v) <= 3, rule
        for y in range(Q1_Y):
            for x in range(Q1_X):
                if ops[op](x if axis == "x" else y, bound):
                    q[y][x] = int(v)
    flat = [v for row in q for v in row]
    return [sum(v << (2 * i) for i, v in enumerate(flat[w : w + 16])) for w in range(0, len(flat), 16)]


# the column's tile rows split evenly over mp_gu5's 15 groups and down's 120 cores (BALANCE, mp_common.hpp) in place
# of whole experts round robin (group g: experts g, g + 15, ..; core k: items (e, qq) k, k + 120, ..): real text's
# routing gives an expert up to ~1,150 of a column's 4,096 tokens' pairs, and the busiest group 2.4 times the mean
# rows (58 against 26 with an expert's weights' load counted as 1 row; uniform routing 22 against 20).  Each unit takes
# the rows whose place on the line of all rows (an expert's start weighing BALANCE_A rows) falls in its share; an
# expert cut between two units has its weights read by both.  The same values (each row computed as before): the real
# weights' 1 x 8K prefill 709.5 -> 529.2 ms, 1 x 1K 151.9 -> 136.2 ms, the logits bit for bit; stress 24,000 ops and
# 600 balanced 8-row rounds clean.  0: round robin
BALANCE = True
BALANCE_A = 1
# ... decided on the device per stage: the even split when the round robin's busiest unit has more than BALANCE_PCT %
# of the even share (uniform routing ~110 %: round robin; real text's 200-400 %)
BALANCE_PCT = 130
# the progressive form (moe.py's GLM53_MOE_PROG_F): e_w1 / e_w2 hold every expert's BFP4 base in slots 0 .. slots - 1
# and its BFP4 correction `slots` later; the prefill reads both for every expert (K twice: x times the base, then times
# the correction, summed in the same fp32 Dest), whatever F
PROG = os.environ.get("GLM53_MOE_PROG_F") is not None
GU5_PROBE = 0  # mp_gu5's timing probes (tests only; wrong results): compute 1 no tilize, 2 no finish, 4 no matmul;
# reader 8 W1 for the first expert only, 16 no x reads, 64 the first expert's W1 not read; writer 32 no exchange


def gate_up(mesh, x_rm, e_w1, seg, ptok, pw, limit: float, corr: int = 0):
    """x_rm [1, 1, N, 4096] bf16 row-major, e_w1 ([slots 8 4096, 64] bfp8 or bfp4) -> H [1, 1, M, 256] bf16 tiled
    (mp_gu5, above); `corr` (PROG): the slots past each expert's base where its correction is (0: none)."""
    sets = 2 if corr else 1
    m = ptok.shape[-2] * 16
    wfmt, wtile = _wfmt(e_w1)
    h = _alloc(mesh, [1, 1, m, 256], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    groups = _groups(mesh)
    if GU5_ROLE_ROT:  # member list order = roles: core k of group g takes role (k - rot g) mod 8
        groups = [grp[(GU5_ROLE_ROT * gi) % 8 :] + grp[: (GU5_ROLE_ROT * gi) % 8] for gi, grp in enumerate(groups)]
    cores = core_rects([c for grp in groups for c in grp])
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    noc = mesh.worker_core_from_logical_core
    q1 = _q1_table(GU5_NOC1_QUARTERS, GU5_NOC1_RULES if x_rm.shape[-2] >= GU5_NOC1_RULES_MIN else "")
    for gi, grp in enumerate(groups):
        n_items = len(range(gi, LOCAL, len(groups)))
        xys = [noc(o).x | noc(o).y << 16 for o in grp]
        for c, core in enumerate(grp):
            rd[core.x][core.y] = [
                x_rm.buffer_address(),
                e_w1.buffer_address(),
                seg.buffer_address(),
                ptok.buffer_address(),
                pw.buffer_address(),
                gi,
                len(groups),
                c,
            ] + q1
            cp[core.x][core.y] = [n_items]
            wr[core.x][core.y] = [h.buffer_address(), seg.buffer_address(), gi, len(groups), c] + xys
    bits = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]  # noqa: E731
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[8] = ttnn.UnpackToDestMode.UnpackToDestFp32  # G5_RED: the partials summed in fp32 in Dest
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=_fidelity(FIDELITY, e_w1), fp32_dest_acc_en=True)
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "mp_gu5_reader.cpp",
            core_ranges=cores,
            compile_time_args=[x_rm.shape[-1] * 2, wtile, LOCAL]
            + _acc(x_rm, e_w1, seg, ptok, pw)
            + [GU5_PROBE, GU5_NOCS, int(BALANCE), BALANCE_A, BALANCE_PCT, corr],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "mp_gu5_compute.cpp",
            core_ranges=cores,
            compile_time_args=[bits(limit), bits(-limit), GU5_PROBE, int(BALANCE), sets],
            runtime_args=cp,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "mp_gu5_writer.cpp",
            core_ranges=cores,
            compile_time_args=[TILE_BF16, LOCAL, 8]
            + _acc(h, seg)
            + [GU5_PROBE, GU5_NOCS, int(BALANCE), BALANCE_A, BALANCE_PCT],
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cbs = [
        _cb(0, wfmt, wtile, 512 * sets, cores),  # W1's K block c of every column tile, two experts (PROG: base, corr)
        _cb(1, ttnn.bfloat16, TILE_BF16, 32, cores),  # this member's x block of a row, row-major, two rows
        _cb(2, ttnn.bfloat16, TILE_BF16, 32, cores),  # the same tilized, two rows
        _cb(3, ttnn.bfloat16, TILE_BF16, 4, cores),  # weights column
        _cb(4, ttnn.bfloat16, TILE_BF16, 2, cores),  # h
        _cb(5, ttnn.uint32, SEG_WORDS * 4 + 512, 1, cores),  # (the seg page, two rows' token and weight pages)
        _cb(6, ttnn.uint32, SEG_WORDS * 4 + 64, 1, cores),
        _cb(7, ttnn.float32, TILE_FP32, 32, cores),  # this member's partials of a row (16 tiles), two rows
        _cb(8, ttnn.float32, TILE_FP32, 32, cores),  # the 8 members' partials of its column tile, two rows
    ]
    sems = _SEMS3(cores)  # S_READY, S_XVALID, S_NOC1
    _run(mesh, [x_rm, e_w1, seg, ptok, pw, h], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
    return h


# down's and combine's readers on both NoCs (2): DNCB_NOC1_QUARTERS of every 4 W2 / h tiles and Y rows read on NoC 1,
# handed over by the writers (which then read nothing: combine's slots come from its reader) as mp_gu5's (GU5_NOCS).
# On NoC 0 alone their low grid rows' cores end 3 to 4 times later than the high rows', but their writers already
# load NoC 1 (Y, out): at 4,096 tokens down 0.70 -> 0.74-0.77 ms, combine 0.54 -> 0.51-0.55; with Y in bf16 (half the
# writes) down 0.58 -> 0.54 (1 quarter), combine 0.38 -> 0.35.  The same values.  The default: with Y in bf16 the op
# 1,024 tokens 1.12-1.25 -> 1.06-1.18 ms, 4,096 2.03 -> 1.96 ms, 1 x 8K 528.7 -> 507.9 ms (synthetic, the logits the
# same); stress 12,000 ops and 600 8-row rounds clean.  1: NoC 0 only
DNCB_NOCS = 2
DNCB_NOC1_QUARTERS = 1


def _SEMS3(cores):  # noqa: N802  (semaphores 0 .. 2: mp_gu5's S_READY, S_XVALID; S_NOC1 for all three)
    return [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in (0, 1, 2)]


# Y (down's sums of each pair, combine's input) in bf16 in place of fp32: each pair's sum rounded to bf16 once before
# combine adds a token's pairs (in fp32, as before); half the bytes of Y's round trip through DRAM (written by down,
# read by combine: 2 x 149 MB a chip a layer at 4,096 tokens): down 0.70 -> 0.58 ms, combine 0.54 -> 0.38.
# Values change (the experts' sums against fp32 Y: PCC 0.99994, mean |d| 0.13 % of mean |y|; against torch at 512
# tokens mean |d| 7.4e-4 against 5.4e-4); the real weights' teacher forcing (3 passkey prompts x 24 positions)
# within the one-row / eight-row forms' own spread (KLD to the CPU reference 0.0172 -> 0.0206 / 0.0212 -> 0.0180).
# Default on (the prefill +8-10 % with GU5_NOCS 2); 0: fp32
Y_BF16 = os.environ.get("GLM53_MOE_PREFILL_Y_BF16", "1") == "1"


# down's reader reads a K row's 4 tiles of a column block (adjacent in their bank) in one request of 4,352 bytes in
# place of four (NoC 1 for DNCB_NOC1_QUARTERS of every 4 rows in place of tiles), up to DN_W2_RUNS_MAX tokens a column:
# the experts op (with GU5_W1_PAIRS) at 1,024 tokens 1.024 -> 0.964 ms (uniform; real text's 1.135 -> 1.071), but at
# 4,096 slower (1.880 -> 1.897 alone, 2.111 -> 2.150 real with GU5_W1_PAIRS: down's DRAM traffic, its Y writes
# too, near the rate there).  The same bytes into the same places; False: a tile a request (tests)
DN_W2_RUNS = True
DN_W2_RUNS_MAX = 2048

# Y in bfp8 (a quality-changing trial; values change: each Y row's 16 consecutive values share an exponent, the
# CPU estimate's "bfp8 along a row", KLD +0.00037): down untilizes Dest to fp32 rows (as with Y_BF16 off), then each
# flat 1,024-value piece (two rows of the item's 512 columns) is packed as a bfp8_b tile (64 exponents, 1,024
# mantissas: value i's exponent the (i / 16)-th), and each row's half written into its Y8 record: a pair row's 1,024
# columns as one flat bfp8 piece of Y8_REC bytes (exponents 64, mantissas 1,024).  combine takes blocks of 1,024 columns
# (32 tiles an item), a record a read straight into a CB_S page (one token row a piece), and sums them as bf16 pieces.
# Y8 [1, 1, M, Y8_ROW / 4] uint32 row-major; 0: off
Y_BFP8 = os.environ.get("GLM53_MOE_PREFILL_Y_BFP8", "0") == "1"
Y8_REC = 1088
Y8_ROW = 4 * Y8_REC


def down(mesh, h, e_w2, seg, slots: int, corr: int = 0):
    """H [1, 1, M, 256] bf16 tiled, e_w2 (col8: [32 slots 256, 128] bfp8 or bfp4) -> Y [1, 1, M, 4096] fp32 (Y_BF16: bf16)
    row-major (Y_BFP8: its records [1, 1, M, Y8_ROW / 4] uint32); `corr` (PROG): the slots past each expert's base where
    its correction is (`slots` then counts both)."""
    sets = 2 if corr else 1
    m = h.shape[-2]
    runs = DN_W2_RUNS and m <= table_rows(DN_W2_RUNS_MAX)  # (H's rows: table_rows of the column's tokens)
    wfmt, wtile = _wfmt(e_w2)
    y8 = Y_BFP8
    ydt, ytile, ysize = (ttnn.bfloat16, TILE_BF16, 2) if Y_BF16 and not y8 else (ttnn.float32, TILE_FP32, 4)
    if y8:  # (Y8 records; the untilized rows fp32 in L1 only)
        y = _alloc(mesh, [1, 1, m, Y8_ROW // 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    else:
        y = _alloc(mesh, [1, 1, m, 4096], ydt, ttnn.ROW_MAJOR_LAYOUT)
    work, cores = _spread(mesh, LOCAL * 8)  # (e, qq): 4 column blocks an item
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    if BALANCE:  # every core its share of the items' rows: (the core count, this core's index)
        grid = _grid(mesh)
        work = [(c, [k]) for k, c in enumerate(grid)]
        cores = core_rects(grid)
        for c, items in work:
            rd[c.x][c.y] = [h.buffer_address(), e_w2.buffer_address(), seg.buffer_address(), len(grid)] + items
            cp[c.x][c.y] = [len(grid)]
            wr[c.x][c.y] = [y.buffer_address(), seg.buffer_address(), len(grid)] + items
    for c, items in work if not BALANCE else []:
        rd[c.x][c.y] = [h.buffer_address(), e_w2.buffer_address(), seg.buffer_address(), len(items)] + items
        cp[c.x][c.y] = [len(items)]
        wr[c.x][c.y] = [y.buffer_address(), seg.buffer_address(), len(items)] + items
    dn_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=_fidelity(DOWN_FIDELITY, e_w2), fp32_dest_acc_en=True)
    if y8:  # (the fp32 rows unpacked to Dest as they are, then packed in bfp8: one rounding)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
        dn_cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "mp_dn_reader.cpp",
            core_ranges=cores,
            compile_time_args=[TILE_BF16, wtile, slots]
            + _acc(h, e_w2, seg)
            + [DNCB_NOCS, DNCB_NOC1_QUARTERS, int(BALANCE), BALANCE_A, BALANCE_PCT, int(runs), corr],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "mp_dn_compute.cpp",
            core_ranges=cores,
            compile_time_args=[int(BALANCE), int(y8), sets],
            runtime_args=cp,
            config=dn_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "mp_dn_writer.cpp",
            core_ranges=cores,
            compile_time_args=[Y8_ROW if y8 else 4096 * ysize]
            + _acc(y, seg)
            + [DNCB_NOCS, int(BALANCE), BALANCE_A, BALANCE_PCT, int(y8)],
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cbs = [
        _cb(0, wfmt, wtile, 256 * sets, cores),  # W2(e, q) of the item's 4 column blocks, two items (PROG: 2 sets)
        _cb(1, ttnn.bfloat16, TILE_BF16, 16, cores),  # h tiles, two rows
        _cb(2, ydt, ytile, 32, cores),  # y untilized, two rows of the item's 4 blocks
        _cb(3, ttnn.uint32, SEG_WORDS * 4, 1, cores),
        _cb(4, ttnn.uint32, SEG_WORDS * 4, 1, cores),
    ]
    if y8:
        cbs += [
            _cb(5, ttnn.bfloat8_b, TILE_BFP8, 32, cores),  # a row's 16 flat pieces in bfp8, two rows
            _cb(6, ttnn.uint32, 32 * 64, 1, cores),  # the writer's exponents twice, 64 bytes a pair row
        ]
    _run(mesh, [h, e_w2, seg, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=_SEMS3(cores), cbs=cbs))
    return y


# combine with CB_S holding two items (Y in bf16) and the writer mailing the next item's slots before writing this
# one's tiles (mp_cb_writer.cpp): the reader reads item i + 1's Y rows while the compute sums item i (the same sums;
# the op -0.01 to -0.04 ms); 0: one item at a time.  With GU5_ROLE_ROT 3 and the plan's one-pass count: the op 1.067
# -> 1.024 ms at 1,024 tokens a column, 1.963 -> 1.900 at 4,096 (uniform routing; real text's 1.153 -> 1.134, 2.178 ->
# 2.126), the prefill 8 x 128 / 8 x 2,048 / 1 x 8K +1.5 / +0.9 / +0.9 %, the logits bit for bit, 600 rounds of the
# 8-row prefill's stress and 2,000 runs of the op each.  (Y in fp32, Y_BF16
# off, goes one item at a time: two items of fp32 Y are 1 MB of L1.)  False: one item at a time (tests)
CB_AHEAD = True


def combine(mesh, y, tpair, batches: int = 1):
    """Y [1, 1, M, 4096] fp32 (or bf16; or Y_BFP8's records [1, 1, M, Y8_ROW / 4] uint32) row-major, tpair [1, 1,
    N / 32, 512] -> [1, 1, N, 4096] bf16 tiled; `batches` > 1: its rows batch-major (the
    column's 8 mesh rows' N / 8 tokens each cut into `batches`: batch b's rows are each mesh row's tokens b N / 8 /
    batches .., in mesh-row order; `col_row_rs`)."""
    n = tpair.shape[-2] * 32  # (the tokens: tpair [N / 32, 512])
    bf16 = y.dtype == ttnn.bfloat16
    assert batches == 1 or (n % (32 * ROWS * batches) == 0), (n, batches)
    out = _alloc(mesh, [1, 1, n, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    y8 = y.dtype == ttnn.uint32  # (Y_BFP8's records)
    pieces = 32 if y8 else 16  # (an item's columns / 32: its tiles and CB_S's pieces a slot)
    work, cores = _spread(mesh, n // 32 * (128 // pieces))
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, items in work:
        rd[c.x][c.y] = [y.buffer_address(), tpair.buffer_address(), len(items)] + items
        cp[c.x][c.y] = [len(items)]
        wr[c.x][c.y] = [out.buffer_address(), tpair.buffer_address(), len(items)] + items
    # the items (their count per core) change with N, which the program cache's key does not see in the runtime args
    # (only their count of cores): N goes into every kernel's compile-time args (a 512-token program reused for 4,096
    # tokens hung, its writer's shorter item list)
    key = [n]
    ahead = CB_AHEAD and (bf16 or y8)  # (two items of fp32 Y: 1 MB of L1)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    if not bf16 and not y8:
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "mp_cb_reader.cpp",
            core_ranges=cores,
            compile_time_args=[Y8_ROW if y8 else 4096 * (2 if bf16 else 4)]
            + _acc(y, tpair)
            + key
            + [DNCB_NOCS, DNCB_NOC1_QUARTERS, int(y8)],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "mp_cb_compute.cpp",
            core_ranges=cores,
            compile_time_args=key + [pieces],
            runtime_args=cp,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "mp_cb_writer.cpp",
            core_ranges=cores,
            compile_time_args=[TILE_BF16]
            + _acc(out, tpair)
            + key
            + [DNCB_NOCS, int(ahead), pieces, ROWS]
            + ([n // 32 // ROWS, n // 32 // ROWS // batches] if batches > 1 else [1, 1]),
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    s_fmt = (ttnn.bfloat8_b, TILE_BFP8) if y8 else (ttnn.bfloat16, TILE_BF16) if bf16 else (ttnn.float32, TILE_FP32)
    cbs = [  # CB_S: every slot's block (32 rows x 512 values) of one item, or of two (CB_AHEAD)
        _cb(0, *s_fmt, pieces * K * (1 + ahead), cores),
        _cb(1, ttnn.bfloat16, TILE_BF16, 2 * pieces, cores),  # the sums row-major, two items
        _cb(2, ttnn.uint32, 32 * 64, 1, cores),
        _cb(3, ttnn.uint32, 32 * 64, 1, cores),
        _cb(4, ttnn.bfloat16, TILE_BF16, 2 * pieces, cores),  # the sums tilized, two items
        _cb(5, ttnn.uint32, 32, 4, cores),  # the items' slots, the reader to the writer
    ]
    _run(mesh, [y, tpair, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=_SEMS3(cores), cbs=cbs))
    return out


def experts(
    mesh,
    x_all,
    route,
    e_w1,
    e_w2,
    slots: int,
    limit: float,
    row_major: bool = False,
    mask=None,
    batches: int = 1,
    planned=None,
):
    """The column's tokens x_all [1, 1, N, 4096] bf16 tiled (row_major: row-major, kept), the route op's rows -> this
    chip's partial sums [1, 1, N, 4096] bf16 tiled (`batches` > 1: rows batch-major, `combine`); `mask` (`plan`'s): its
    0 tokens' sums are 0; `planned`: `plan`'s tables made already (seg, ptok, pw, tpair: taken over)."""
    seg, ptok, pw, tpair = planned if planned is not None else plan(mesh, route, mask)
    x_rm = x_all if row_major else ttnn.to_layout(x_all, ttnn.ROW_MAJOR_LAYOUT)
    corr = slots if PROG else 0
    h = gate_up(mesh, x_rm, e_w1, seg, ptok, pw, limit, corr)
    if not row_major:
        ttnn.deallocate(x_rm)
    ttnn.deallocate(ptok)
    ttnn.deallocate(pw)
    y = down(mesh, h, e_w2, seg, slots * (2 if PROG else 1), corr)
    ttnn.deallocate(h)
    ttnn.deallocate(seg)
    out = combine(mesh, y, tpair, batches)
    ttnn.deallocate(y)
    ttnn.deallocate(tpair)
    return out


__all__ = ["PlanPush", "combine", "down", "experts", "gate_up", "plan", "table_rows"]
