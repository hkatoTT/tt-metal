"""The KDA verify step's decode_prepare and chunked recurrence (recurrent_decode_step's chunked form) with the
recurrence's state read ahead, across the two programs (kda.d2_form, GLM53_KDA_D2=2; on by default at 128 users K = 3).

decode_prepare's reader, its own reads done, reads each recurrence core's units' state (bf16, all of them: 96 KB a core
at 16 users a mesh row and T = 4) into an L1 buffer of that core, under the prepare's compute and writes (DRAM nearly
idle there); the chunked recurrence's CB_S lies on that buffer and its reader does not read the state.  Two programs as
the C++ ops make them (their binaries within a few hundred bytes of the C++ ops': the kernel config ring as before), the
same values bit for bit (the same reads into the same tiles, the same compute).

The two programs are built here (Python generic_op) as the C++ factories build them for the model's verify call
(kda.TtKda.step_spec: users, keep, saved, tokens, pair_major, l1_outputs, split_channels; o into L1):
decode_prepare_program_factory.cpp and recurrent_decode_step_program_factory.cpp's create_chunk_descriptor, line for
line (the same CBs, args, core split and NoC rows); the read-ahead form's two reader kernels in kernels/kda_d2/.  Also
here, for comparison (GLM53_KDA_D2=1, the fused form): the two as one program (kernels/kda_d2/d2_*.cpp), every core its
prepare item then its recurrence units, the recurrence reader waiting for a semaphore count from the prepare items of
its heads; the same values, but 68 - 73 KB of kernels that do not fit the dispatcher's kernel config ring beside
x_proj's.

    tests/test_kda_d2.py (the device); run as a script (D2_CASES "users:T,..", D2_FORMS "ops,d2p", D2_STEPS,
    D2_CHECK=1 compares bit for bit)
"""

from __future__ import annotations

import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# the kernels: ttnn/kernels/kda_d2/ (or kernels/ beside this file)
KD = os.environ.get("D2_KERNEL_DIR") or (
    os.path.join(_HERE, "kernels", "kda_d2")
    if os.path.isdir(os.path.join(_HERE, "kernels", "kda_d2"))
    else os.path.join(_HERE, "kernels")
)
TREE_KDA = "ttnn/cpp/ttnn/operations/experimental/kda/"
H, DH, N = 16, 128, 4
LOWER = -5.0
CH = 3 * H * DH

# decode_prepare's CB indices (decode_prepare_program_factory.cpp)
P_CUR, P_H0, P_H1, P_H2, P_TAP, P_F, P_DT, P_RATE, P_BETA, P_ONES, P_STG = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
P_QKV, P_SQ, P_INV, P_OUT, P_BOUT, P_WSCR, P_CURH, P_KEEP, P_ROWS, P_SAVED, P_XT = (
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    12,
)
P_SEQ, P_HST, P_SQP, P_SQX = 28, 29, 30, 31
# recurrent_decode_step's chunked form's (create_chunk_descriptor), + C_BASE in the fused program
(
    C_KQ,
    C_AL,
    C_BT,
    C_MZ,
    C_V,
    C_S,
    C_CONST,
    C_CONSTM,
    C_G,
    C_KQP,
    C_KM,
    C_M,
    C_N,
    C_P,
    C_Y,
    C_YM,
    C_F,
    C_TB,
    C_E,
    C_C,
    C_KZ,
    C_GC,
    C_X1,
    C_O,
    C_SO,
    C_KEEP,
    C_SCR,
) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27)
CHUNK_ROW_PCT = [70, 60, 70, 70, 80, 90, 100, 100, 100, 100]
D2_SEM = 2


def seq_lines() -> int:
    d = os.environ.get("TT_METAL_KSIM_DUMP")
    p = os.path.join(d, "seq.jsonl") if d else ""
    if not d or not os.path.exists(p):
        return 0
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def capture(on: bool):
    os.environ["TT_METAL_KSIM_ON"] = "1" if on else "0"


def fbits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def col_major(n, gy):
    return [(i // gy, i % gy) for i in range(n)]


def rects(cores):
    """A set of (x, y) as few rectangles: per column its runs of y, then neighbouring columns with the same runs
    merged."""
    import ttnn

    cols = {}
    for x, y in sorted(cores):
        cols.setdefault(x, []).append(y)
    runs = {}
    for x, ys in cols.items():
        r, a = [], ys[0]
        for i in range(1, len(ys) + 1):
            if i == len(ys) or ys[i] != ys[i - 1] + 1:
                r.append((a, ys[i - 1]))
                if i < len(ys):
                    a = ys[i]
        runs[x] = r
    out, xs = [], sorted(runs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1 and runs[xs[j + 1]] == runs[xs[i]]:
            j += 1
        for a, b in runs[xs[i]]:
            out.append(ttnn.CoreRange(ttnn.CoreCoord(xs[i], a), ttnn.CoreCoord(xs[j], b)))
        i = j + 1
    return ttnn.CoreRangeSet(out)


def chunk_split(units, cores, pct):
    CP, CBC = 15, 4

    def ranges(budget):
        out, a = [], 0
        while a < units:
            if len(out) == cores:
                return []
            cap = budget * pct(len(out)) // 100
            b, cost = a, 0
            while b < units:
                add = CBC + (CP if (b == a or b % 4 == 0) else 0)
                if cost + add > cap:
                    break
                cost += add
                b += 1
            if b == a:
                return []
            out.append((a, b - a))
            a = b
        return out

    lo, hi = CP + CBC, 2 * (CP + CBC) * units
    while lo < hi:
        mid = (lo + hi) // 2
        if ranges(mid):
            hi = mid
        else:
            lo = mid + 1
    return ranges(lo)


def stack_tokens(users, tokens):
    if tokens < 2 or users > 4 or 16 % users != 0 or users * tokens > 32:
        return 0
    g = min(16 // users, tokens)
    return g if g >= 2 else 0


class Cb:
    """A CB as the factories add them: (index, n pages, page size, ttnn dtype), or shared indices."""

    def __init__(self, idx, n, page, dt, shared=None):
        self.idx, self.n, self.page, self.dt, self.shared = idx, n, page, dt, shared

    def desc(self, cores, base=0):
        import ttnn

        fds = [
            ttnn.CBFormatDescriptor(buffer_index=base + i, data_format=self.dt, page_size=self.page)
            for i in ([self.idx] + ([self.shared] if self.shared is not None else []))
        ]
        return ttnn.CBDescriptor(total_size=self.n * self.page, core_ranges=cores, format_descriptors=fds)


def acc(t):
    import ttnn

    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def addr(t):
    return t.buffer_address()


class Plan:
    """One of the two programs' parts: CBs (indices from 0), the reader / writer / compute CT args and defines, and per
    core (x, y): its RT args and whether its data movers take the swapped (upper rows') form."""

    def __init__(self):
        self.cbs, self.cores = [], []
        self.reader_ct, self.writer_ct, self.compute_ct = [], [], []
        self.reader_def, self.writer_def, self.compute_def = [], [], []
        self.rt_r, self.rt_w, self.rt_c, self.upper = {}, {}, {}, set()
        self.unpack_fp32 = []
        self.sems = []
        self.extra = {}


def prepare_plan(
    mesh, proj, fg, hist, taps, gc, keep, saved, outs, users, tokens, heads=H, lower=LOWER, q_scale=1.0 / DH**0.5
):
    import ttnn

    p = Plan()
    f32, bf16, u32 = ttnn.float32, ttnn.bfloat16, ttnn.uint32
    T32, T16 = 4096, 2048
    grid = mesh.compute_with_storage_grid_size()
    gy = grid.y
    blocks = (users + 15) // 16
    units = 4 * heads * blocks
    items = []
    for u in range(units):
        if (u // blocks) % 4 < 3:
            items += [(u, 0, 2), (u, 2, 2)]
        else:
            items.append((u, 0, N))
    assert len(items) <= grid.x * grid.y
    ncores = len(items)
    p.cores = col_major(ncores, gy)
    stack_g = stack_tokens(users, tokens)
    stack_p = (tokens + stack_g - 1) // stack_g if stack_g else 0
    depth = 2 if tokens <= 4 else 1
    cb = p.cbs
    if stack_g:
        cb += [Cb(P_SEQ, N, T16, bf16), Cb(P_HST, 2 * N, T32, f32)]
    cb += [
        Cb(P_CUR, depth * tokens * N, T16, bf16),
        Cb(P_H0, 2 * N, T32, f32),
        Cb(P_H1, 2 * N, T32, f32),
        Cb(P_H2, 2 * N, T32, f32),
        Cb(P_TAP, 2 * 4 * N, T32, f32),
        Cb(P_F, depth * tokens * N, T16, bf16),
        Cb(P_DT, 2 * N, T32, f32),
        Cb(P_RATE, 2 * N, T32, f32),
        Cb(P_BETA, depth * tokens, T32, f32),
        Cb(P_ONES, 1, T32, f32),
        Cb(P_STG, 1, T16, bf16),
        Cb(P_QKV, N, T32, f32),
        Cb(P_SQ, 1, T32, f32),
        Cb(P_INV, 1, T32, f32),
        Cb(P_OUT, 2 * N, T32, f32),
        Cb(P_BOUT, 2, T32, f32),
        Cb(P_WSCR, 1, T32, f32),
        Cb(P_CURH, 2 * N, T32, f32),
        Cb(P_SQP, 1, T32, f32),
        Cb(P_SQX, 2, T32, f32),
    ]
    p.sems.append(0)
    if tokens > 1 and users % 16 != 0:
        cb.append(Cb(P_ROWS, 1, T16, bf16))
    if tokens > 1:
        cb.append(Cb(P_XT, max(tokens - 1, 2) * N, T32, f32))
    keep_bytes = -(-users * 4 // 32) * 32
    cb.append(Cb(P_SAVED, (tokens - 1 if tokens > 1 else 1) * N, T32, f32))
    cb.append(Cb(P_KEEP, 1, keep_bytes, u32))
    proj_tiles = int(proj.padded_shape[-1]) // 32
    fg_tiles = int(fg.padded_shape[-1]) // 32
    hist_tiles = int(hist[0].padded_shape[-1]) // 32
    saved_pages = -(-int(hist[0].padded_shape[-2]) // 32) * hist_tiles
    ins = [proj, fg, hist[0], hist[1], hist[2], taps, gc]
    p.reader_ct = [
        P_CUR,
        P_H0,
        P_H1,
        P_H2,
        P_TAP,
        P_F,
        P_DT,
        P_RATE,
        P_BETA,
        P_ONES,
        P_STG,
        heads,
        proj_tiles,
        fg_tiles,
        hist_tiles,
        users,
    ]
    for t in ins:
        p.reader_ct += acc(t)
    p.reader_ct += [1, P_KEEP, keep_bytes, P_ROWS, P_SAVED, tokens, saved_pages]
    p.reader_ct += acc(keep) + acc(saved) + [stack_g, stack_p, P_SEQ, P_HST]
    p.writer_ct = [P_OUT, P_BOUT, P_WSCR, heads, users, P_CURH, hist_tiles]
    for t in outs:
        p.writer_ct += acc(t)
    p.writer_ct += acc(hist[2]) + [tokens, 1, saved_pages] + acc(saved) + [stack_g, stack_p, 1, P_SQP, P_SQX]
    p.compute_ct = [fbits(lower), fbits(q_scale), stack_g * users if stack_g else users]
    if stack_g:
        p.compute_def = [("KDA_PREP_STACK", str(stack_p))]
    elif tokens > 1:
        p.compute_def = [("KDA_PREP_TOKENS", str(tokens))]
    for lst in (p.compute_def, p.reader_def, p.writer_def):
        lst.append(("KDA_PREP_SPLIT", "1"))
    p.unpack_fp32 = [P_H0, P_H1, P_H2, P_TAP, P_DT, P_RATE, P_BETA, P_QKV, P_INV, P_SQX] + (
        [P_XT] if tokens > 1 else []
    )
    # split: the top 5 grid rows (virtual y 2 .. 6) swap their data movers' NoCs
    for i, (x, y) in enumerate(p.cores):
        v = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        if 2 <= v.y < 7:
            p.upper.add((x, y))
    for i, (x, y) in enumerate(p.cores):
        unit, c_lo, c_n = items[i]
        px = py = 0
        if c_n == 2:
            j = i + 1 if c_lo == 0 else i - 1
            pv = mesh.worker_core_from_logical_core(ttnn.CoreCoord(j // gy, j % gy))
            px, py = pv.x, pv.y
        p.rt_r[(x, y)] = [addr(t) for t in ins] + [
            unit,
            1,
            addr(keep),
            addr(saved),
            c_lo,
            c_n,
            addr(hist[0]),
            addr(hist[1]),
            addr(saved),
        ]
        p.rt_w[(x, y)] = [addr(t) for t in outs] + [
            addr(hist[2]),
            unit,
            1,
            addr(hist[1]),
            users * heads,
            addr(hist[0]),
            addr(saved),
            c_lo,
            c_n,
            px,
            py,
        ]
        p.rt_c[(x, y)] = [unit, 1, c_lo, c_n]
    p.extra = dict(items=items, users=users, heads=heads)
    return p


def chunk_plan(mesh, state, q, k, v, decay, beta, keep, saved, out, users, tokens, heads=H):
    import ttnn

    p = Plan()
    f32, bf16, u32 = ttnn.float32, ttnn.bfloat16, ttnn.uint32
    T32, T16 = 4096, 2048
    grid = mesh.compute_with_storage_grid_size()
    gy = grid.y
    pairs = users * heads
    by_rows = gy == len(CHUNK_ROW_PCT)
    if by_rows and pairs > grid.x * grid.y:
        split = chunk_split(pairs * N, grid.x * grid.y, lambda k_: CHUNK_ROW_PCT[k_ % gy])
    else:
        split = chunk_split(pairs * N, grid.x * grid.y, lambda k_: 100)
    p.cores = col_major(len(split), gy)
    # (the fused form's D2_PREFETCH_ALL: CB_S holds every unit of a core, its state all read at the start)
    s_tiles = max(2 * N * N, max(c for _, c in split) * N) if os.environ.get("D2_PREFETCH_ALL") == "1" else 2 * N * N
    slot_bytes = 1536 * tokens + 64
    keep_bytes = -(-users * 4 // 32) * 32
    p.cbs = [
        Cb(C_KQ, 2 * N, T32, f32),
        Cb(C_AL, 2 * N, T32, f32),
        Cb(C_BT, 2, T32, f32),
        Cb(C_MZ, 2, T32, f32),
        Cb(C_V, 2 * N, T32, f32),
        Cb(C_S, s_tiles, T16, bf16),
        Cb(C_CONST, 4, T32, f32, shared=C_CONSTM),
        Cb(C_G, N, T32, f32),
        Cb(C_KQP, N, T32, f32),
        Cb(C_KM, N, T32, f32),
        Cb(C_M, 1, T32, f32),
        Cb(C_N, 1, T32, f32),
        Cb(C_P, 1 if tokens <= 2 else 2 if tokens <= 4 else 3, T32, f32),
        Cb(C_Y, 2, T32, f32, shared=C_YM),
        Cb(C_F, 1, T32, f32),
        Cb(C_TB, 1, T32, f32),
        Cb(C_E, 1, T32, f32),
        Cb(C_C, N, T32, f32),
        Cb(C_KZ, N, T32, f32),
        Cb(C_GC, N, T32, f32),
        Cb(C_X1, 1, T32, f32),
        Cb(C_O, 2, T32, f32),
        Cb(C_SO, 4 * N, T16, bf16),
        Cb(C_SCR, 1, 2 * slot_bytes + 512 + 2048, f32),
        Cb(C_KEEP, 1, keep_bytes, u32),
    ]
    rows = [q, k, v, decay, beta]
    p.reader_ct = [tokens, pairs, heads, keep_bytes, 1]
    for t in rows:
        p.reader_ct += acc(t)
    for t in (state, saved, keep):
        p.reader_ct += acc(t)
    p.writer_ct = [tokens, pairs] + acc(state) + acc(out)
    p.compute_def = [("KDA_CHUNK_TOKENS", str(tokens))]
    p.unpack_fp32 = [C_KQ, C_AL, C_BT, C_MZ, C_CONST, C_G, C_M, C_Y, C_GC]
    up_rows = 3 if tokens >= 8 else 0
    for i, (x, y) in enumerate(p.cores):
        vv = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        if up_rows and 2 <= vv.y < 2 + up_rows:
            p.upper.add((x, y))
        start, count = split[i]
        p.rt_r[(x, y)] = [addr(t) for t in rows] + [
            addr(state),
            addr(saved),
            addr(keep),
            start,
            count,
            addr(state),
            addr(saved),
        ]
        p.rt_w[(x, y)] = [addr(state), addr(out), start, count]
        p.rt_c[(x, y)] = [start, count]
    p.extra = dict(split=split, pairs=pairs)
    return p


def dm(proc, noc):
    import ttnn

    c = ttnn.DataMovementConfigDescriptor()
    c.processor, c.noc = proc, noc
    return c


def ccfg(fp32_cbs):
    import ttnn

    modes = [ttnn.UnpackToDestMode.Default] * 64
    for c in fp32_cbs:
        modes[c] = ttnn.UnpackToDestMode.UnpackToDestFp32
    c = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    c.dst_full_sync_en = True
    c.unpack_to_dest_mode = modes
    return c


def lower_cfgs():
    import ttnn

    P = ttnn.DataMovementProcessor
    return (
        (dm(P.RISCV_1, ttnn.NOC.NOC_0), dm(P.RISCV_0, ttnn.NOC.NOC_1)),  # reader, writer (Reader/WriterConfig)
        (dm(P.RISCV_0, ttnn.NOC.NOC_1), dm(P.RISCV_1, ttnn.NOC.NOC_0)),
    )  # the swapped rows'


def kernels_for(
    plan_cores, upper, src_r, src_w, src_c, ct_r, ct_w, ct_c, def_r, def_w, def_c, rt_r, rt_w, rt_c, fp32_cbs
):
    """reader / writer kernels per (lower, upper) set of cores and one compute."""
    import ttnn

    low, up = lower_cfgs()
    ks = []
    for cores, (cr, cw) in (
        ([c for c in plan_cores if c not in upper], low),
        ([c for c in plan_cores if c in upper], up),
    ):
        if not cores:
            continue
        rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for x, y in cores:
            rr[x][y] = rt_r[(x, y)]
            rw[x][y] = rt_w[(x, y)]
        crs = rects(cores)
        ks.append(
            ttnn.KernelDescriptor(
                kernel_source=src_r, core_ranges=crs, compile_time_args=ct_r, runtime_args=rr, defines=def_r, config=cr
            )
        )
        ks.append(
            ttnn.KernelDescriptor(
                kernel_source=src_w, core_ranges=crs, compile_time_args=ct_w, runtime_args=rw, defines=def_w, config=cw
            )
        )
    rc = ttnn.RuntimeArgs()
    for x, y in plan_cores:
        rc[x][y] = rt_c[(x, y)]
    ks.append(
        ttnn.KernelDescriptor(
            kernel_source=src_c,
            core_ranges=rects(plan_cores),
            compile_time_args=ct_c,
            runtime_args=rc,
            defines=def_c,
            config=ccfg(fp32_cbs),
        )
    )
    return ks


def run_program(mesh, kernels, sems, cbs, io):
    import ttnn

    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    mp = ttnn.MeshProgramDescriptor()
    s = mesh.shape
    mp[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(s[0] - 1, s[1] - 1))] = program
    ttnn.generic_op(io, mp)


def run_sep(mesh, pp, cp, io):
    """The two programs as the C++ ops build them (a check of the replication)."""
    import ttnn

    pk = TREE_KDA + "decode_prepare/device/kernels/"
    ck = TREE_KDA + "recurrent_decode_step/device/kernels/"
    pc = rects(pp.cores)
    run_program(
        mesh,
        kernels_for(
            pp.cores,
            pp.upper,
            pk + "dataflow/reader_decode_prepare.cpp",
            pk + "dataflow/writer_decode_prepare.cpp",
            pk + "compute/decode_prepare.cpp",
            pp.reader_ct,
            pp.writer_ct,
            pp.compute_ct,
            pp.reader_def,
            pp.writer_def,
            pp.compute_def,
            pp.rt_r,
            pp.rt_w,
            pp.rt_c,
            pp.unpack_fp32,
        ),
        [ttnn.SemaphoreDescriptor(id=0, core_ranges=pc, initial_value=0)],
        [c.desc(pc) for c in pp.cbs],
        io,
    )
    cc = rects(cp.cores)
    run_program(
        mesh,
        kernels_for(
            cp.cores,
            cp.upper,
            ck + "dataflow/reader_recurrent_decode_chunk.cpp",
            ck + "dataflow/writer_recurrent_decode_chunk.cpp",
            ck + "compute/recurrent_decode_chunk.cpp",
            cp.reader_ct,
            cp.writer_ct,
            cp.compute_ct,
            cp.reader_def,
            cp.writer_def,
            cp.compute_def,
            cp.rt_r,
            cp.rt_w,
            cp.rt_c,
            cp.unpack_fp32,
        ),
        [],
        [c.desc(cc) for c in cp.cbs],
        io,
    )


def state_buffer(mesh, cp):
    """The read-ahead form: an L1 buffer on each chunk core holding all its units' state tiles (bf16), the chunk
    program's CB_S."""
    import ttnn

    tiles = max(c for _, c in cp.extra["split"]) * N
    cores = rects(cp.cores)
    spec = ttnn.ShardSpec(cores, [tiles * 32, 32], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    return (
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, len(cp.cores) * tiles * 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, mc
        ),
        tiles,
    )


def run_d2p(mesh, pp, cp, io, state, sbuf, tiles):
    """The read-ahead form (two programs, as the C++ ops', the state read across the boundary): decode_prepare's
    program with its reader also reading each chunk core's units' state into sbuf (d2p_prep_reader.cpp), then the
    chunked recurrence with CB_S on sbuf and its reader not reading the state (d2p_chunk_reader.cpp).  The same
    values."""
    import ttnn

    pk = TREE_KDA + "decode_prepare/device/kernels/"
    ck = TREE_KDA + "recurrent_decode_step/device/kernels/"
    split = dict(zip(cp.cores, cp.extra["split"]))
    # (every recurrence core a prepare core: its state read by that core's prepare reader; true at the verify shapes,
    # 112 prepare items over cores 0 .. 111 and the recurrence's 64 / 112 cores the first of them)
    missing = [c for c in cp.cores if c not in set(pp.cores)]
    if missing:
        raise RuntimeError(
            f"GLM53_KDA_D2=2: recurrence cores {missing[:4]}.. run no prepare item (GLM53_KDA_D2=0 at this shape)"
        )
    rt_r = {}
    for c in pp.cores:
        st, cn = split.get(c, (0, 0))
        rt_r[c] = pp.rt_r[c] + [addr(state), st, cn, sbuf.buffer_address()]
    def_r = pp.reader_def + [("D2P_CT", str(len(pp.reader_ct))), ("D2P_RT", "16")]
    pc = rects(pp.cores)
    run_program(
        mesh,
        kernels_for(
            pp.cores,
            pp.upper,
            os.path.join(KD, "d2p_prep_reader.cpp"),
            pk + "dataflow/writer_decode_prepare.cpp",
            pk + "compute/decode_prepare.cpp",
            pp.reader_ct + acc(state),
            pp.writer_ct,
            pp.compute_ct,
            def_r,
            pp.writer_def,
            pp.compute_def,
            rt_r,
            pp.rt_w,
            pp.rt_c,
            pp.unpack_fp32,
        ),
        [ttnn.SemaphoreDescriptor(id=0, core_ranges=pc, initial_value=0)],
        [c.desc(pc) for c in pp.cbs],
        io + [sbuf],
    )
    cc = rects(cp.cores)
    cbs = [c.desc(cc) for c in cp.cbs if c.idx != C_S]
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(C_S, sbuf, 0, tiles * 2048, cc))
    def_cr = cp.reader_def + [("D2_PREFETCH_ALL", "1"), ("D2_STATE_RESIDENT", "1")]
    run_program(
        mesh,
        kernels_for(
            cp.cores,
            cp.upper,
            os.path.join(KD, "d2p_chunk_reader.cpp"),
            ck + "dataflow/writer_recurrent_decode_chunk.cpp",
            ck + "compute/recurrent_decode_chunk.cpp",
            cp.reader_ct,
            cp.writer_ct,
            cp.compute_ct,
            def_cr,
            cp.writer_def,
            cp.compute_def,
            cp.rt_r,
            cp.rt_w,
            cp.rt_c,
            cp.unpack_fp32,
        ),
        [],
        cbs,
        io + [sbuf],
    )


C_BASE = 32


def fused_parts(mesh, pp, cp):
    """The fused program's kernels, semaphores and CBs over the union of the two core sets."""
    import ttnn

    union = sorted(set(pp.cores) | set(cp.cores), key=lambda c: (c[0], c[1]))
    heads = pp.extra["heads"]
    # consumers: per chunk core the heads its unit range touches (pair p = user H + head)
    need = {}
    for c, (start, count) in zip(cp.cores, cp.extra["split"]):
        hs = set()
        for u in range(start, start + count):
            hs.add((u // N) % heads)
        need[c] = hs
    prod_head = {c: it[0] // 4 for c, it in zip(pp.cores, pp.extra["items"])}  # unit u = 4 h + part (one block)
    per_head = {}
    for c, h in prod_head.items():
        per_head[h] = per_head.get(h, 0) + 1
    zero_r = len(next(iter(pp.rt_r.values())))
    zero_w = len(next(iter(pp.rt_w.values())))
    zero_cr = len(next(iter(cp.rt_r.values())))
    zero_cw = len(next(iter(cp.rt_w.values())))
    rt_r, rt_w, rt_c = {}, {}, {}
    first_c = cp.rt_r[cp.cores[0]]
    first_w = cp.rt_w[cp.cores[0]]
    for c in union:
        pr = pp.rt_r.get(c)
        pw = pp.rt_w.get(c)
        pc = pp.rt_c.get(c)
        if pr is None:  # (no prepare item: count 0)
            pr = list(next(iter(pp.rt_r.values())))
            pr[8] = 0
            pw = list(next(iter(pp.rt_w.values())))
            pw[7] = 0
            pc = [0, 0, 0, N]
        cr = cp.rt_r.get(c)
        cw = cp.rt_w.get(c)
        cc = cp.rt_c.get(c)
        if cr is None:
            cr = list(first_c)
            cr[9] = 0
            cw = list(first_w)
            cw[3] = 0
            cc = [0, 0]
        # (+ 1: this core's own prepare done, the writer's increment)
        expected = sum(per_head[h] for h in need.get(c, ())) + (1 if c in need else 0)
        rt_r[c] = pr + cr + [expected]
        sig = []
        if c in prod_head:
            h = prod_head[c]
            for d in cp.cores:
                if h in need[d]:
                    v = mesh.worker_core_from_logical_core(ttnn.CoreCoord(d[0], d[1]))
                    sig += [v.x, v.y]
        rt_w[c] = pw + cw + [len(sig) // 2] + sig
        rt_c[c] = pc + cc
    assert zero_r == 16 and zero_w == 16 and zero_cr == 12 and zero_cw == 4, (zero_r, zero_w, zero_cr, zero_cw)
    d_common = [("D2_CB", str(C_BASE)), ("D2_PREP", "1"), ("D2_SEM", str(D2_SEM))]
    dm_os = [("D2_DM_OS", "1")] if os.environ.get("D2_DM_OS") == "1" else []
    if os.environ.get("D2_PREFETCH_ALL") == "1":
        d_common = d_common + [("D2_PREFETCH_ALL", "1")]
    def_r = (
        pp.reader_def + d_common + dm_os + [("D2_CT", str(len(pp.reader_ct))), ("D2_RT", "16"), ("D2_READY_RT", "28")]
    )
    def_w = pp.writer_def + d_common + dm_os + [("D2_CT", str(len(pp.writer_ct))), ("D2_RT", "16"), ("D2_SIG_RT", "20")]
    def_c = (
        pp.compute_def
        + cp.compute_def
        + d_common
        + [("D2_RT", "4")]
        + ([("D2_UNPACK_OS", "1")] if os.environ.get("D2_UNPACK_OS") == "1" else [])
    )
    upper = set(c for c in union if (c in pp.upper) or (c not in pp.rt_r and c in cp.upper))
    fp32 = pp.unpack_fp32 + [C_BASE + c for c in cp.unpack_fp32]
    ks = kernels_for(
        union,
        upper,
        os.path.join(KD, "d2_reader.cpp"),
        os.path.join(KD, "d2_writer.cpp"),
        os.path.join(KD, "d2_compute.cpp"),
        pp.reader_ct + cp.reader_ct,
        pp.writer_ct + cp.writer_ct,
        pp.compute_ct + cp.compute_ct,
        def_r,
        def_w,
        def_c,
        rt_r,
        rt_w,
        rt_c,
        fp32,
    )
    uc = rects(union)
    sems = [
        ttnn.SemaphoreDescriptor(id=0, core_ranges=uc, initial_value=0),
        ttnn.SemaphoreDescriptor(id=D2_SEM, core_ranges=uc, initial_value=0),
    ]
    cbs, alias = fused_cbs(pp, cp, uc)
    info = dict(
        union=len(union),
        prep=len(pp.cores),
        chunk=len(cp.cores),
        expected=sorted(set(sum(per_head[h] for h in need.get(c, ())) for c in union)),
        signals=sorted(set(rt_w[c][20] for c in union)),
        cb_kb=sum(d.total_size for d in cbs) / 1024,
        alias=alias,
    )
    return ks, sems, cbs, info


# chunk CBs that may share a prepare CB's memory (D2_ALIAS): written only after this core's prepare is done (the
# reader's wait counts its own writer's increment; the compute's are behind the reader's pushes). (index: exact tiles
# needed (the reader's per-slot setup of KQ / AL / V assumes their 2 slots), or the granularity a size must be a
# multiple of). S, SCR, KEEP, CONST, BT, MZ (written by the reader before its wait), P (a pair's tiles never straddle
# its end) and Y / YM stay apart.
ALIAS_EXACT = {C_KQ, C_AL, C_V}
ALIAS_GRAN = {
    C_G: 4,
    C_KQP: 4,
    C_KM: 4,
    C_C: 4,
    C_KZ: 4,
    C_GC: 4,
    C_SO: 4,
    C_M: 1,
    C_N: 1,
    C_F: 1,
    C_TB: 1,
    C_E: 1,
    C_X1: 1,
    C_O: 1,
}


def fused_cbs(pp, cp, cores):
    """The fused program's CB descriptors: the prepare CBs (CB_XT dropped in the stacked form: its compute never
    touches it), the chunk CBs at C_BASE + index, with D2_ALIAS=1 most chunk CBs sharing a prepare CB's memory."""
    import ttnn

    stacked = any(c.idx == P_SEQ for c in pp.cbs)
    prep = [c for c in pp.cbs if not (stacked and c.idx == P_XT)]
    alias_on = os.environ.get("D2_ALIAS") == "1"
    hosts = {}  # prepare idx -> chunk Cb
    if alias_on:
        free = sorted(prep, key=lambda c: c.n * c.page)
        for ch in sorted(cp.cbs, key=lambda c: -c.n * c.page):
            if ch.idx not in ALIAS_EXACT and ch.idx not in ALIAS_GRAN:
                continue
            need = ch.n * ch.page
            for h in free:
                tot = h.n * h.page
                if h.idx in hosts or h.shared is not None or tot % ch.page:
                    continue
                pages = tot // ch.page
                ok = pages == ch.n if ch.idx in ALIAS_EXACT else (tot >= need and pages % ALIAS_GRAN[ch.idx] == 0)
                if ok:
                    hosts[h.idx] = ch
                    break
    out = []
    for h in prep:
        if h.idx in hosts:
            ch = hosts[h.idx]
            fds = [
                ttnn.CBFormatDescriptor(buffer_index=h.idx, data_format=h.dt, page_size=h.page),
                ttnn.CBFormatDescriptor(buffer_index=C_BASE + ch.idx, data_format=ch.dt, page_size=ch.page),
            ]
            out.append(ttnn.CBDescriptor(total_size=h.n * h.page, core_ranges=cores, format_descriptors=fds))
        else:
            out.append(h.desc(cores))
    aliased = {ch.idx for ch in hosts.values()}
    out += [c.desc(cores, C_BASE) for c in cp.cbs if c.idx not in aliased]
    return out, {h: ch.idx for h, ch in hosts.items()}


def run_fused(mesh, pp, cp, io):
    ks, sems, cbs, _ = fused_parts(mesh, pp, cp)
    run_program(mesh, ks, sems, cbs, io)


def verify_step(
    mesh,
    proj,
    fg,
    history,
    taps,
    gate_rows,
    keep,
    saved,
    state,
    rows,
    users,
    tokens,
    heads=H,
    lower=LOWER,
    q_scale=None,
    form="2",
):
    """The model's verify step's KDA prepare + recurrence (kda.TtKda.step_spec with HANDOFF_L1, PREP_SPLIT, no
    ping-pong) as one program: -> o [1, 1, T B H, 128] fp32 in L1 (the same values as the two ops)."""
    import ttnn

    R = tokens * users * heads
    l1 = ttnn.L1_MEMORY_CONFIG
    mk = lambda w: ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, w]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, l1
    )  # noqa
    q, k, v, e, beta = mk(DH), mk(DH), mk(DH), mk(DH), mk(1)
    out = mk(DH)
    qs = 1.0 / DH**0.5 if q_scale is None else q_scale
    pp = prepare_plan(
        mesh, proj, fg, history, taps, gate_rows, keep, saved, [q, k, v, e, beta], users, tokens, heads, lower, qs
    )
    cp = chunk_plan(mesh, state, q, k, v, e, beta, keep, rows, out, users, tokens, heads)
    io = [proj, fg, *history, taps, gate_rows, keep, saved, state, rows, q, k, v, e, beta, out]
    if form == "2":  # (the read-ahead form: two programs, the state read across the boundary)
        sbuf, tiles = state_buffer(mesh, cp)
        run_d2p(mesh, pp, cp, io, state, sbuf, tiles)
        ttnn.deallocate(sbuf)
    else:
        run_fused(mesh, pp, cp, io)
    for t in (q, k, v, e, beta):
        ttnn.deallocate(t)
    return out


# ---------------------------------------------------------------- the test
def make_inputs(mesh, users, tokens, seed):
    import torch

    from models.demos.blackhole.glm53_flash.ttnn.kda import chunk_saved_host

    g = torch.Generator().manual_seed(seed)
    b, n = users, tokens
    host = dict(
        p=torch.randn(n * b, 6432, generator=g).to(torch.bfloat16),
        f=torch.randn(n * b, 2 * H * DH, generator=g).to(torch.bfloat16),
        hist=[torch.randn(b, CH, generator=g) for _ in range(3)],
        sv=torch.randn(1, n - 1, b, CH, generator=g),
        tp=torch.randn(4, CH, generator=g) * 0.5,
        gc=torch.stack([torch.randn(H * DH, generator=g), torch.rand(H * DH, generator=g) + 0.5]),
        flags=torch.randint(0, n, (1, b), generator=g, dtype=torch.int32),
        st=(torch.randn(b, H, DH, DH, generator=g) * 0.05).to(torch.bfloat16),
        rows=chunk_saved_host(b, H, n),
    )
    return host


def upload(mesh, host):
    import ttnn

    up = lambda t, dtype: ttnn.from_torch(t[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa
    d = dict(
        p=up(host["p"], ttnn.bfloat16),
        f=up(host["f"], ttnn.bfloat16),
        hist=[up(h, ttnn.float32) for h in host["hist"]],
        sv=ttnn.from_torch(host["sv"], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh),
        tp=up(host["tp"], ttnn.float32),
        gc=up(host["gc"], ttnn.float32),
        flags=ttnn.from_torch(host["flags"], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh),
        st=ttnn.from_torch(host["st"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh),
        rows=ttnn.from_torch(host["rows"], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh),
    )
    return d


def step_ops(mesh, d, users, tokens):
    """Today's two ops (kda.TtKda.step_spec's verify call)."""
    import ttnn

    mid = ttnn.experimental.kda.decode_prepare(
        d["p"],
        d["f"],
        *d["hist"],
        d["tp"],
        d["gc"],
        H,
        LOWER,
        1.0 / DH**0.5,
        users=users,
        keep=d["flags"],
        saved=d["sv"],
        tokens=tokens,
        pair_major=True,
        l1_outputs=True,
        split_channels=True,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, tokens * users * H, DH]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.L1_MEMORY_CONFIG
    )
    o = ttnn.experimental.kda.recurrent_decode_step(
        d["st"], *mid, keep=d["flags"], heads=H, tokens=tokens, pair_major=True, saved=d["rows"], output=out
    )
    return o, list(mid)


def step_py(mesh, d, users, tokens, form):
    import ttnn

    R = tokens * users * H
    l1 = ttnn.L1_MEMORY_CONFIG
    mk = lambda w: ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, w]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, l1
    )  # noqa
    q, k, v, e, beta = mk(DH), mk(DH), mk(DH), mk(DH), mk(1)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, R, DH]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, l1)
    pp = prepare_plan(
        mesh, d["p"], d["f"], d["hist"], d["tp"], d["gc"], d["flags"], d["sv"], [q, k, v, e, beta], users, tokens
    )
    cp = chunk_plan(mesh, d["st"], q, k, v, e, beta, d["flags"], d["rows"], out, users, tokens)
    io = [d["p"], d["f"], *d["hist"], d["tp"], d["gc"], d["flags"], d["sv"], d["st"], d["rows"], q, k, v, e, beta, out]
    if form == "sep":
        run_sep(mesh, pp, cp, io)
    elif form == "d2p":
        sbuf, tiles = state_buffer(mesh, cp)
        run_d2p(mesh, pp, cp, io, d["st"], sbuf, tiles)
        ttnn.deallocate(sbuf)
    else:
        run_fused(mesh, pp, cp, io)
    return out, [q, k, v, e, beta], (pp, cp)


def main() -> int:
    if not os.environ.get("TT_METAL_SIMULATOR") and not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        print("D2 REFUSED: neither TT_METAL_SIMULATOR nor a mock cluster is set", flush=True)
        return 0
    capture(False)
    import torch

    import ttnn

    check = os.environ.get("D2_CHECK", "1") == "1" and os.environ.get("TT_METAL_KSIM_NOLAUNCH") != "1"
    steps = int(os.environ.get("D2_STEPS", "2"))
    forms = os.environ.get("D2_FORMS", "ops,fused").split(",")
    t0 = time.time()
    # the model's L1 layout (common.DEVICE_PARAMS: l1_small_size, worker_l1_size with the 24 KB wider kernel config
    # ring, 93 KB in all)
    extra = int(os.environ.get("GLM53_KERNEL_CONFIG_EXTRA_KB", "24"))
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=32768,
        worker_l1_size=1460864 - (extra << 10),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    )
    g = mesh.compute_with_storage_grid_size()
    print(f"D2 open in {time.time() - t0:.1f}s, grid {g.x} x {g.y}, steps {steps}, check {check}", flush=True)
    bad = 0
    try:
        for case in os.environ.get("D2_CASES", "4:8,16:4").split(","):
            users, tokens = (int(x) for x in case.split(":"))
            host = make_inputs(mesh, users, tokens, 2600 + 10 * users + tokens)
            res = {}
            for form in forms:
                d = upload(mesh, host)
                outs = []
                n0 = seq_lines()
                capture(True)
                tf = time.time()
                for s in range(steps):
                    if form == "ops":
                        o, mid = step_ops(mesh, d, users, tokens)
                    else:
                        o, mid, plans = step_py(mesh, d, users, tokens, form)
                        if s == 0 and form == "fused":
                            print(f"D2 PLAN {case}: {fused_parts(mesh, *plans)[3]}", flush=True)
                    if check:
                        outs.append([ttnn.to_torch(t).float() for t in [o] + mid])
                    for t in [o] + mid:
                        ttnn.deallocate(t)
                capture(False)
                print(f"CASE {case} {form} enqueues {n0}..{seq_lines() - 1} ({time.time() - tf:.1f}s)", flush=True)
                if check:
                    fin = [ttnn.to_torch(t).float() for t in [d["st"], d["rows"], *d["hist"], d["sv"]]]
                    res[form] = (outs, fin)
                for t in (d["p"], d["f"], *d["hist"], d["sv"], d["tp"], d["gc"], d["flags"], d["st"], d["rows"]):
                    ttnn.deallocate(t)
            if check and len(res) > 1:
                ref = res[forms[0]]
                names = ["o", "q", "k", "v", "decay", "beta"]
                fnames = ["state", "rows", "h0", "h1", "h2", "saved"]
                for form in forms[1:]:
                    got = res[form]
                    diffs = []
                    for s in range(steps):
                        for nm, a, b in zip(names, ref[0][s], got[0][s]):
                            if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                                diffs.append(f"step{s}.{nm} max|d| {(a - b).abs().max().item():.3e}")
                    for nm, a, b in zip(fnames, ref[1], got[1]):
                        if nm == "rows":  # (a slot's beta row holds T floats; its other 16 - T are never written:
                            # whatever the reader's staging held, never read. Compared without them.)
                            from models.demos.blackhole.glm53_flash.ttnn.kda import chunk_slot_tiles

                            nt = chunk_slot_tiles(tokens)
                            a = a.reshape(users * H * 4, nt * 1024).clone()
                            b = b.reshape(users * H * 4, nt * 1024).clone()
                            a[:, 256 * tokens + tokens : 256 * tokens + 16] = 0
                            b[:, 256 * tokens + tokens : 256 * tokens + 16] = 0
                        same = (
                            torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))
                            if a.dtype == torch.float32
                            else torch.equal(a, b)
                        )
                        if not same:
                            diffs.append(f"final.{nm} max|d| {(a - b).abs().max().item():.3e}")
                            if nm == "rows":  # which units' slots (unit u = user H 4 + h 4 + j, nt tiles each)
                                from models.demos.blackhole.glm53_flash.ttnn.kda import chunk_slot_tiles

                                nt = chunk_slot_tiles(tokens)
                                dd = (a != b).reshape(users * H * 4, nt * 32 * 32).any(dim=1).nonzero().flatten()
                                diffs.append(f"rows units differing {len(dd)}: {dd[:40].tolist()}")
                                ee = (a != b).reshape(users * H * 4, nt * 1024)
                                first = ee[dd[0]].nonzero().flatten()[:8].tolist() if len(dd) else []
                                diffs.append(f"first unit's differing floats {first}")
                    bad += len(diffs)
                    fin_o = ref[0][-1][0]
                    print(
                        f"CHECK {case} {form} vs {forms[0]}: {'bit for bit SAME' if not diffs else 'DIFF: ' + '; '.join(diffs[:12])}"
                        f" (o finite {torch.isfinite(fin_o).all().item()}, |o| max {fin_o.abs().max().item():.3e})",
                        flush=True,
                    )
    finally:
        ttnn.close_mesh_device(mesh)
    print(f"D2 DONE bad={bad}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
