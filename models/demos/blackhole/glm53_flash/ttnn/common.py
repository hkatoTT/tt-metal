"""Mesh layout, weight placement and compute settings shared by the device components."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import NamedTuple

import torch

import ttnn

MESH_SHAPE = (8, 4)
DEVICE_TILIZE_MIN = 1 << 24  # elements: tile on the device above this (16 M bf16 values)
ROWS, COLS = MESH_SHAPE
ROW_AXIS = 0  # mesh axis 0: the 8 rows; users are split over it
TP_AXIS = 1  # mesh axis 1: the 4 chips of a row; tensor parallel

# How the device tests and runs open the mesh: TorusXY fabric (a ring on both axes), and an L1_SMALL region so the
# collectives' semaphores do not sit in the middle of L1 (they clashed with sdpa decode's circular buffers).
DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "l1_small_size": 32768}
# The dispatcher's ring for the programs' kernel configs (binaries, runtime args, CB configs) on each worker core: on
# Blackhole the space between the firmware's map (41,344 B) and the allocator's L1, 69 KB by default (1,460,864 B of L1
# for the allocator).  A program whose config does not fit next to those of the programs still running waits for them
# to finish before its config is written: a gap before it, as long as the writing of its binaries.  The custom ops'
# configs are large (KDA's prepare 46 KB, its verify recurrence 41 KB, mHC's write-read ~35 KB, the experts ~25 KB): the
# decode's gaps were ~1 ms a step.  KERNEL_CONFIG_EXTRA_KB (24) gives the ring that much of the allocator's L1: the
# gaps 1.0 -> 0.4 ms (at 32 users), the verify steps -0.3 to -0.6 ms (24 as 32 within 0.1 ms but 128 users at T = 4,
# -0.21 ms), the prefill as fast as before.  The tightest program, 128 users' T = 4 verify on 96 cores, overflows at
# +48: 24 leaves it ~19 KB (32: ~11 KB); the chunk-512 prefill fits up to +56.  vLLM: the plugin's tt config
# "worker_l1_size" (1,460,864 - 24,576 = 1,436,288).  GLM53_KERNEL_CONFIG_EXTRA_KB in the environment: another size
# (past +40 the 128 users' T = 4 program's static CBs overflow).
KERNEL_CONFIG_EXTRA_KB = int(os.environ.get("GLM53_KERNEL_CONFIG_EXTRA_KB", "24"))
DEVICE_PARAMS["worker_l1_size"] = 1460864 - (KERNEL_CONFIG_EXTRA_KB << 10)
# the fabric routers' largest packet payload in bytes (the 2D default 4,352; Blackhole allows up to 15,232): ttnn's
# collectives move 8 KB row-major rows in two packets at the default (they warn of it).  8,192: the prefill's
# collectives -4 to -8 %, the decode the same; stress clean (decode 2,000 steps at 8 / 32 users, plain and verify;
# 600 8-row prefill rounds).  15,232: 1.6x slower.  vLLM opens the mesh
# with the same (`fabric_router_config`, tools/glm53_vllm.py)
FABRIC_PAYLOAD = 8192


# GLM53_TRACE_ALLOC_GUARD: once the generator's warm-up has armed it (`arm_trace_alloc_guard`), a device buffer
# allocated after the traces were captured and still alive when one of them is replayed (the replay may write over it:
# the trace's own intermediates sat there at its capture) is printed once ("log") or stops the run ("1").
# Needs TT_METAL_TRACE_ALLOC_TRACKING=1 (and for the allocation's Python stack TT_METAL_TRACE_ALLOC_TRACEBACKS=1) in
# the environment before ttnn is imported.  0 (default): off, ttnn.execute_trace untouched
TRACE_ALLOC_GUARD = os.environ.get("GLM53_TRACE_ALLOC_GUARD", "0")
_GUARD = {"armed": False, "flagged": 0}


def core_rects(coords) -> ttnn.CoreRangeSet:
    """Distinct cores (CoreCoord) as few rectangles as their layout allows: each grid row's runs of consecutive x, then
    the runs that repeat on consecutive rows merged.  A program's kernels, CBs and semaphores go out range by range: a
    range a core (the generic_op ops' CoreRange(c, c)) left their dispatch showing between the programs (the drafter's
    64-core ops ~8 us before each sublayer's first convolution at 8 users)."""
    rows: dict = {}
    for c in coords:
        rows.setdefault(c.y, []).append(c.x)
    runs: dict = {}  # (x0, x1) -> its rows
    for y, xs in rows.items():
        xs = sorted(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
                j += 1
            runs.setdefault((xs[i], xs[j]), []).append(y)
            i = j + 1
    ranges = []
    for (x0, x1), ys in sorted(runs.items(), key=lambda kv: (min(kv[1]), kv[0])):
        ys = sorted(ys)
        i = 0
        while i < len(ys):
            j = i
            while j + 1 < len(ys) and ys[j + 1] == ys[j] + 1:
                j += 1
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x0, ys[i]), ttnn.CoreCoord(x1, ys[j])))
            i = j + 1
    return ttnn.CoreRangeSet(ranges)


def _guard_report(live: dict) -> str:
    """The flagged buffers with their allocation context and the model's frames of their Python stack."""
    from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_DIAGNOSTICS, TraceAllocationTracker

    if TRACE_ALLOC_DIAGNOSTICS:
        TraceAllocationTracker.reconcile_tracebacks()
    parts = []
    for buf, ctx in sorted(live.items()):
        stack = TraceAllocationTracker._tracebacks.get(buf, "")
        frames = ['  File "' + f.rstrip() for f in stack.split('  File "')[1:] if "glm53_flash" in f.split("\n")[0]]
        parts.append(f"buffer {buf} [{ctx or '?'}]\n" + ("\n".join(frames[-6:]) if frames else "  (no model frame)"))
    return "\n".join(parts)


def _guarded_execute_trace(mesh_device, trace_id, *, cq_id=None, blocking=True):
    from ttnn._ttnn.operations import trace as tr

    if _GUARD["armed"]:
        live = tr.get_unsafe_tracked_ids(mesh_device, trace_id)
        if live:
            import gc

            gc.collect()
            live = tr.get_unsafe_tracked_ids(mesh_device, trace_id)
        if live:
            report = _guard_report(live)
            if TRACE_ALLOC_GUARD == "1":
                raise RuntimeError(
                    f"{len(live)} device buffer(s) allocated after the captures alive at a replay:\n{report}"
                )
            _GUARD["flagged"] += len(live)
            print(
                f"[trace alloc guard] {len(live)} buffer(s) allocated after the captures alive at a replay "
                f"(trace {trace_id}):\n{report}",
                flush=True,
            )
            for buf in live:
                tr.remove_unsafe_tracked_id(mesh_device, buf)
    return tr.execute_trace(mesh_device, trace_id, cq_id=cq_id, blocking=blocking)


def arm_trace_alloc_guard(mesh, armed: bool = True) -> None:
    """The warm-up's end (`armed`): what the warm-up made is accepted (its traces' fixed inputs and outputs, written
    before each use; the order of its captures is test_warmup_order's), and from here on GLM53_TRACE_ALLOC_GUARD
    watches the replays.  The warm-up's start: not armed."""
    if TRACE_ALLOC_GUARD == "0":
        return
    from ttnn._ttnn.operations import trace as tr
    from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

    if not TRACE_ALLOC_TRACKING:
        raise RuntimeError("GLM53_TRACE_ALLOC_GUARD needs TT_METAL_TRACE_ALLOC_TRACKING=1 before ttnn is imported")
    if armed:
        import gc

        gc.collect()
        for buf in tr.get_all_unsafe_tracked_ids():
            tr.remove_unsafe_tracked_id(mesh, buf)
    _GUARD["armed"] = armed


if TRACE_ALLOC_GUARD != "0":
    ttnn.execute_trace = _guarded_execute_trace  # (every caller reads it from the module: ttnn.execute_trace(..))


# The parity check: the one-shot fabric ops and all_reduce_async alternate two sets of landing areas and semaphores by a
# host turn (`OneshotArea.turn`, `_TpArAsync.turn`), fixed at a trace's capture: inside a trace the parity alternates,
# but at a trace's seam (the same trace replayed again, or another trace next) the same set comes twice in a row when
# that trace made an odd number of calls of the state (row_ar: the model breaks in 10 moves; row_ar2 holds by itself).
# At every capture's end: each state's turn at the start and its calls in the capture; a state that starts odd or takes
# an odd number of calls is printed ("[parity check] ..", GLM53_PARITY_CHECK=1, the default) or raised (=raise); 0: off.
# The values are not touched.  Each capture's line names its caller (the stack's glm53_flash frames).
# Known and accepted: head_pick's areas (head_row / head_col) take one call in a DFlash2 step's trace, odd, so a
# replayed step meets the same set at its seam; that relies on the step holding both the row and the column collectives
# between two head_picks (they keep any chip from writing a set its peer is still reading, checked over all 8 x 4
# chips).  A change that puts head_pick or head_relax in a trace without both (an acceptance- or drafts-only trace, a
# separate trace for the sampled requests' head, ..) needs the parity on the device instead (a flag each chip flips at
# the call's end, as x2d_gather and maps_x).  So the odd counts of head_pick v2 / v3 and small_gather (their areas
# head_row / head_col; v3, the own mesh row's, holds the same) are marked on the capture's line ("ODD, accepted"), not
# flagged: GLM53_PARITY_CHECK=raise does not stop on them.  head_relax (head_row3 / head_col3) has not been checked: its
# odd counts stay flagged.
PARITY_CHECK = os.environ.get("GLM53_PARITY_CHECK", "1")
_PARITY = {"open": None, "last_end": {}, "captures": 0, "flagged": 0}
# the states whose odd counts are known and accepted (above): marked, not flagged
_PARITY_ACCEPTED_ODD = ("oneshot head_row ", "oneshot head_col ")


def _parity_states() -> dict:
    """label -> the state (every OneshotArea and _TpArAsync made so far)."""
    from .row_all_reduce import _AREAS  # (it imports this module)

    out = {}
    for (_, d, axis), st in list(_TP_AR_ASYNC.items()):
        out[f"ar_async d{d} axis{axis}"] = st
    for (_, family, pages, page), area in list(_AREAS.items()):
        out[f"oneshot {family} {pages}x{page}"] = area
    return out


def _parity_snapshot() -> dict:
    return {k: (st.turn, dict(getattr(st, "paths", {}))) for k, st in _parity_states().items()}


def _parity_caller() -> str:
    import traceback

    frames = [
        f
        for f in traceback.extract_stack()
        if "glm53_flash" in f.filename and not f.name.startswith(("_checked_", "_parity_"))
    ]
    return " < ".join(f"{f.filename.rsplit('/', 1)[-1][:-3]}.{f.name}:{f.lineno}" for f in reversed(frames[-4:]))


_begin_trace_capture = ttnn.begin_trace_capture
_end_trace_capture = ttnn.end_trace_capture


def _checked_begin_trace_capture(mesh_device, *args, **kwargs):
    if PARITY_CHECK != "0":
        _PARITY["open"] = (_parity_snapshot(), _parity_caller())
    return _begin_trace_capture(mesh_device, *args, **kwargs)


def _checked_end_trace_capture(mesh_device, *args, **kwargs):
    out = _end_trace_capture(mesh_device, *args, **kwargs)
    opened, _PARITY["open"] = _PARITY["open"], None
    if PARITY_CHECK == "0" or opened is None:
        return out
    before, caller = opened
    after = _parity_snapshot()
    _PARITY["captures"] += 1
    n = _PARITY["captures"]
    parts, bad = [], []
    for k, (t1, p1) in sorted(after.items()):
        t0, p0 = before.get(k, (0, {}))
        eager = t0 - _PARITY["last_end"].get(k, 0)  # (calls outside the captures since the last one's end)
        d = t1 - t0
        if d == 0:  # (a state this capture did not use)
            continue
        paths = {q: c - p0.get(q, 0) for q, c in p1.items() if c - p0.get(q, 0)}
        tag = f"{k}: start {t0} +{d}" + (f" {paths}" if paths else "") + (f" (eager {eager} before)" if eager else "")
        parts.append(tag)
        if d and (t0 % 2 or d % 2):
            odd = (" START ODD" if t0 % 2 else "") + (" CALLS ODD" if d % 2 else "")
            if k.startswith(_PARITY_ACCEPTED_ODD):  # (head_pick v2 / v3, small_gather: accepted; not head_relax's)
                parts[-1] = tag + odd + ", accepted"
            else:
                bad.append(tag + odd)
    _PARITY["last_end"] = {k: v[0] for k, v in after.items()}
    print(f"[parity check] capture {n} ({caller}): {'; '.join(parts) if parts else 'no parity state used'}", flush=True)
    if bad:
        _PARITY["flagged"] += len(bad)
        msg = f"[parity check] PARITY FLAG capture {n} ({caller}): " + " | ".join(bad)
        if PARITY_CHECK == "raise":
            raise RuntimeError(msg)
        print(msg, flush=True)
    return out


def _parity_at_exit() -> None:
    """At the process's end: each state's calls outside the captures since the last capture's end (eager calls between
    replays would take a parity of their own, outside the captures' alternation), and the flags counted."""
    if not _PARITY["captures"]:
        return
    after = {k: st.turn for k, st in _parity_states().items()}
    late = {k: t - _PARITY["last_end"].get(k, 0) for k, t in sorted(after.items()) if t - _PARITY["last_end"].get(k, 0)}
    print(
        f"[parity check] at exit: {_PARITY['captures']} captures, {_PARITY['flagged']} flags; calls after the last "
        f"capture's end: {late if late else 'none'}",
        flush=True,
    )


if PARITY_CHECK != "0":
    import atexit

    ttnn.begin_trace_capture = _checked_begin_trace_capture  # (every caller reads it from the module, as above)
    ttnn.end_trace_capture = _checked_end_trace_capture
    atexit.register(_parity_at_exit)


def fabric_router_config():
    """The fabric routers' config for FABRIC_PAYLOAD."""
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = FABRIC_PAYLOAD
    return router


DEVICE_PARAMS["fabric_router_config"] = fabric_router_config()
# with a DRAM region for captured traces (the decode step is ~11,000 ops), per chip over its DRAM banks:
# GLM53_TRACE_REGION_MB (default 256; every prefill chunk traced, GLM53_PREFILL_TRACE_ALL, may need more)
TRACE_REGION_MB = int(os.environ.get("GLM53_TRACE_REGION_MB", "256"))
DEVICE_PARAMS_TRACE = {**DEVICE_PARAMS, "trace_region_size": TRACE_REGION_MB << 20}

# v0 runs at the reference precision: bf16 weights, HiFi4 multiplies, fp32 accumulation.
HIFI4_FP32 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)

WEIGHT_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}

FIDELITIES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi3": ttnn.MathFidelity.HiFi3,
    "hifi4": ttnn.MathFidelity.HiFi4,
}


def fidelity(var: str, default: str = "hifi4"):
    """A math fidelity from the environment variable `var` (lofi / hifi2 / hifi3 / hifi4): the low-precision patterns.
    Matmuls only ever take it on a weight (in1, SrcA); the 0/1 picks and copies, whose
    values must come through whole, keep HIFI4_FP32."""
    key = os.environ.get(var, default).lower()
    if key not in FIDELITIES:
        raise ValueError(f"{var}={key}: one of {sorted(FIDELITIES)}")
    return FIDELITIES[key]


# the ttnn matmuls on weights (the projections, the dense MLPs, the LM head; not the router): GLM53_MM_FIDELITY,
# HiFi4 by default.  decode_matmul's own fidelity is DECODE_MATMUL_FIDELITY (C++), the experts' MOE_EXPERTS_FIDELITY
# and GLM53_MOE_PREFILL_FIDELITY.
WEIGHT_MM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=fidelity("GLM53_MM_FIDELITY"), math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def weight_dtype(name: str | None = None):
    """The precision of the model's large matrices (experts, attention projections, MLPs, LM head): GLM53_WEIGHT_DTYPE
    or `name`, bfp8 (BFP8, packed on the host from the bf16 checkpoint) by default, or bf16.  The router,
    norms, mHC, the KDA gates' vectors and the embedding stay as they are.  The components default to bf16; the
    model passes this down."""
    key = name or os.environ.get("GLM53_WEIGHT_DTYPE", "bfp8")
    if key not in WEIGHT_DTYPES:
        raise ValueError(f"GLM53_WEIGHT_DTYPE={key}: one of {sorted(WEIGHT_DTYPES)}")
    return WEIGHT_DTYPES[key]


def batched_matmul_config(mesh, k_tiles: int, n_tiles: int):
    """For [batch.., 32, K] @ [batch.., K, N] with a batched second input (one tile row per batch item): one batch
    item (its whole 1 x N tiles; this config requires per_core_N == N) per core and step, over the whole grid.
    ttnn's default ran these on 4 cores (KDA's k S at 32 users per row: 0.97 ms, with this 0.16 ms)."""
    grid = mesh.compute_with_storage_grid_size()
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k_tiles,
        out_subblock_h=1,
        out_subblock_w=max(w for w in (4, 2, 1) if n_tiles % w == 0),
        per_core_M=1,
        per_core_N=n_tiles,
    )


# decode's weight matmuls on one tile row of tokens (a mesh row's users, up to 32) with a chosen 1D config
# (`decode_linear`): ttnn's default read the FP8 weights at 70-280 GB/s (the shared expert's gate | up 65 us, DSA's
# a_proj 79 us), these at 210-345 GB/s (21 and 32 us; trace-timed per shape)
# ... for up to this many tokens (whole tile rows of x on every core: 512 users' 64 a row had taken ttnn's default,
# the KDA layer's three matmuls 162 us against 103 at 32 tokens, DSA's five 484 against 131).  32: one tile row
DECODE_MM_ROWS = 64
# ... and past it up to this many (128 users' verify step of 6 or 8 tokens a user: 96 / 128 a mesh row), the 1D config
# with every row on each core and the output spread over the whole grid (DFlash2's drafter's form at 128 rows:
# ttnn's default config ran those 3-5x slower), in decode steps only (the prefill's chunks of 128 as before):
# 128 users' verify step of T = 6 / 8 108.0 / 125.3 -> 103.5 / 120.4 ms (synthetic weights).  0: ttnn's default past
# DECODE_MM_ROWS
DECODE_MM_WIDE = 128
_DECODE_MM: dict = {}
# ... and the large ones (at least DECODE_MATMUL_MIN_TILES weight tiles) by decode_matmul (deepseek_prefill: each
# core's weight columns read as one bank's run at a time, x multicast to them; the fp32 sums in another order).  Alone
# (device time): KDA x_proj 78.1 -> 70.5, DSA o_proj 50.0 -> 47.6, KDA o_proj 26.8 -> 25.2, DSA a_proj 27.5 ->
# 25.9, q_b 20.8 -> 19.6 us; the small ones' gains go to its heavier dispatch (trace).  The KLD 0.00228 /
# 0.00254 (0.00235 / 0.00238 without).  Its weight stream on both NOCs, NoC 1 only after x's multicast (a NoC 1
# multicast and other cores' NoC 1 DRAM reads in one op jam the NoC now and then): real weights decode 8 / 32 /
# 128 users -0.4 / -0.4 / -0.2 ms, 16,000 stress steps clean; the op's
# DECODE_MATMUL_NOCS=1 for NoC 0 only
DECODE_MATMUL = True
DECODE_MATMUL_MIN_TILES = 6000
# ... for up to this many tokens (64: two tile rows of x, K whole on each core, x's slice k-major; 512 users' 64 a row:
# alone (trace) KDA x_proj 94.7 -> 81.9, DSA o_proj 69.7 -> 66.8, MLP gate | up 84.9 -> 79.6 us against the 1D
# config at 64 rows (74.6 / 54.7 / 74.8 at 32).  32: one tile row
DECODE_MATMUL_ROWS = 64


# The prefill's waves of more than WAVE_ROWS tokens a row: ttnn's own config for a
# weight matmul picks its K blocks from the shape, and with packer_l1_acc the K blocks fix the bits.  It takes a narrow
# shape (N over 8 M) in 1D with in0_block_w 2 (K tiles even) and a wider one in 2D with Kt / 12 or 1, so the shapes
# narrow at WAVE_ROWS but not at the wave's rows (N over 4,096: KDA's x_proj, the dense MLP's gate | up) summed
# otherwise at 1,024 rows.  Those get `wave_config`: ttnn's WAVE_ROWS K blocks on a 2D grid (a config's K blocks
# alone fix its bits).  The other shapes keep ttnn's own (the same bits at 1,024)
WAVE_ROWS = 512
WAVE_GRID_Y = 8
_WAVE_MM: dict = {}


def wave_config(mesh, m: int, k: int, n: int):
    """A program config for x [.., m, k] @ w [k, n] with ttnn's own K blocks at WAVE_ROWS rows, or None (ttnn's own
    config keeps them: up to WAVE_ROWS rows, or a shape that is not narrow there)."""
    if m <= WAVE_ROWS or m % 32 or n <= 8 * WAVE_ROWS:
        return None
    key = (m, k, n)
    if key not in _WAVE_MM:
        grid = mesh.compute_with_storage_grid_size()
        mt, kt, nt = m // 32, k // 32, -(-n // 32)
        gy = max(y for y in range(1, min(WAVE_GRID_Y, grid.y) + 1) if mt % y == 0)
        pm, pn = mt // gy, -(-nt // grid.x)
        sub = max(
            ((h, w) for h in range(1, 5) for w in range(1, 5) if h * w <= 4 and pm % h == 0 and pn % w == 0),
            key=lambda hw: (hw[0] * hw[1], hw[1]),
        )
        _WAVE_MM[key] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid.x, gy),
            in0_block_w=2 if kt % 2 == 0 else 1,
            out_subblock_h=sub[0],
            out_subblock_w=sub[1],
            per_core_M=pm,
            per_core_N=pn,
            transpose_mcast=False,
            fused_activation=None,
        )
    return _WAVE_MM[key]


# Larger K blocks in prefill (values change: the fp32 sums' K blocks): the prefill's weight matmuls
# (`wave_linear`: KDA x_proj / fg / o_proj, DSA a_proj / q_b / o_proj, the shared expert and the dense MLP) with K in
# blocks of up to PREFILL_KBLOCK tiles (2D multicast on the grid, the blocks as large as fit L1), in place of ttnn's own
# (in0_block_w mostly 1: ~5-50 % slower ops).  Read at each call (a quality test switches it in its process);
# 16 (default: prefill -2.7 / -3.7 %, wikitext KLD 0.0194 -> 0.0131); 0: ttnn's own config (and
# `wave_config`'s)
PREFILL_KBLOCK = int(os.environ.get("GLM53_PREFILL_KBLOCK", "16"))
# ... the bytes its CBs may take a core (GLM53_PREFILL_KBLOCK_L1, KB; read at each call, a key of the configs).  1000
# (the default): K blocks of 16 at 1,024 rows as at 128 and 512, so the matmuls' values do not depend on their rows: a
# pair of sequence-parallel waves (`prefill_sp_pair`: the MLP and MoE at 1,024 rows) sums as its two waves apart.  At
# 900 (the earlier default) the dense MLP's gate | up (K 4,096, N 6,144 a chip) and KDA's x_proj (N 6,432) took blocks
# of 8 at 1,024 rows (16 needs ~950 / ~993 KB at per_core_M 4): a pair summed otherwise than its waves apart, and the
# traced prefill otherwise than the eager one where one paired and the other did not.  With the experts' L1 landing
# held (GLM53_W1, `W1Landing`) those CBs would reach into it (58 / 85 / 120 KB at C = 3 / 4 / 5): `kblock_for_w1` keeps
# 900 and the split below there.  GLM53_PREFILL_KBLOCK_L1=900 GLM53_PREFILL_KBLOCK_SPLIT=0: the earlier form
PREFILL_KBLOCK_L1 = int(os.environ.get("GLM53_PREFILL_KBLOCK_L1", "1000")) * 1024
# ... or the same values within the room (GLM53_PREFILL_KBLOCK_SPLIT=1; read at each call; on with the GLM53_W1
# landing): a matmul of more than WAVE_ROWS rows whose K blocks at its rows are smaller than at WAVE_ROWS runs as blocks
# of WAVE_ROWS rows with WAVE_ROWS' config (a matmul a block, then a concat; the weights read once a block: 8 x 2,048
# -1.3 %, 1 x 8K -0.2 %).  0: off
PREFILL_KBLOCK_SPLIT = int(os.environ.get("GLM53_PREFILL_KBLOCK_SPLIT", "0"))
_KBLOCK_MM: dict = {}
_KBLOCK_BW: dict = {}  # (kblock_config's key) -> its K block (0: no config)


def kblock_for_w1() -> None:
    """The GLM53_W1 landing held in L1 (`W1Landing`): the prefill's K-blocked matmuls within 900 KB of CBs, 1,024 rows
    as blocks of 512 (the values of 1000 KB's), unless GLM53_PREFILL_KBLOCK_L1 / GLM53_PREFILL_KBLOCK_SPLIT set them."""
    global PREFILL_KBLOCK_L1, PREFILL_KBLOCK_SPLIT
    if "GLM53_PREFILL_KBLOCK_L1" not in os.environ:
        PREFILL_KBLOCK_L1 = 900 * 1024
    if "GLM53_PREFILL_KBLOCK_SPLIT" not in os.environ:
        PREFILL_KBLOCK_SPLIT = 1


def kblock_config(mesh, m: int, k: int, n: int, w_tile: int, out_tile: int, most: int):
    """A 2D multicast config for x [.., m, k] @ w [k, n] with K in blocks of up to `most` tiles (a divisor of K's
    tiles, as large as the CBs fit PREFILL_KBLOCK_L1: in0 and in1 blocks double-buffered, the output block), or None."""
    key = (m, k, n, w_tile, out_tile, most, PREFILL_KBLOCK_L1)
    if key not in _KBLOCK_MM:
        grid = mesh.compute_with_storage_grid_size()
        mt, kt, nt = -(-m // 32), k // 32, -(-n // 32)
        gy = max(y for y in range(1, min(WAVE_GRID_Y, grid.y) + 1) if mt % y == 0)
        gx = min(grid.x, nt)
        pm, pn = mt // gy, -(-nt // gx)
        cfg, chosen = None, 0
        for bw in range(min(most, kt), 0, -1):
            if kt % bw:
                continue
            l1 = 2 * pm * bw * 2048 + 2 * bw * pn * w_tile + pm * pn * out_tile
            if l1 > PREFILL_KBLOCK_L1:
                continue
            sub = max(
                ((h, w_) for h in range(1, 5) for w_ in range(1, 5) if h * w_ <= 4 and pm % h == 0 and pn % w_ == 0),
                key=lambda hw: (hw[0] * hw[1], hw[1]),
            )
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=bw,
                out_subblock_h=sub[0],
                out_subblock_w=sub[1],
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
            )
            chosen = bw
            break
        _KBLOCK_MM[key] = cfg
        _KBLOCK_BW[key] = chosen
    return _KBLOCK_MM[key]


def _kblock_split(mesh, x, w, w_tile: int, out_tile: int, **kw):
    """PREFILL_KBLOCK_SPLIT: x [.., m, k] @ w as blocks of WAVE_ROWS rows with WAVE_ROWS' config (a matmul a block,
    then a concat), or None (m is not blocks of WAVE_ROWS, or its own config's K blocks are WAVE_ROWS' already).  (One
    matmul over the blocks as a batch is not a form ttnn's 2D multicast config takes: TT_FATAL.)"""
    shape = list(x.shape)
    m, k, n = shape[-2], w.shape[-2], w.shape[-1]
    if m <= WAVE_ROWS or m % WAVE_ROWS:
        return None
    kblock_config(mesh, m, k, n, w_tile, out_tile, PREFILL_KBLOCK)
    part = kblock_config(mesh, WAVE_ROWS, k, n, w_tile, out_tile, PREFILL_KBLOCK)
    own = _KBLOCK_BW[(m, k, n, w_tile, out_tile, PREFILL_KBLOCK, PREFILL_KBLOCK_L1)]
    if part is None or _KBLOCK_BW[(WAVE_ROWS, k, n, w_tile, out_tile, PREFILL_KBLOCK, PREFILL_KBLOCK_L1)] == own:
        return None
    kw["program_config"] = part
    ys = []
    for b in range(m // WAVE_ROWS):
        part_x = ttnn.slice(x, [0] * (len(shape) - 2) + [b * WAVE_ROWS, 0], shape[:-2] + [(b + 1) * WAVE_ROWS, k])
        ys.append(ttnn.linear(part_x, w, **kw))
        ttnn.deallocate(part_x)
    out = ttnn.concat(ys, dim=len(shape) - 2, memory_config=kw.get("memory_config"))
    for y in ys:
        ttnn.deallocate(y)
    return out


def apply_setting(spec: str):
    """A test's in-process switch: "module.NAME=value[+module.NAME=value..]" (modules of ttnn/, int or 0/1 values;
    "" none) set -> its restore (quality tests measure several settings on one load of the weights)."""
    import importlib

    undo = []
    for one in (x for x in spec.split("+") if x):
        target, value = one.split("=")
        mod_name, name = target.rsplit(".", 1)
        mod = importlib.import_module(f"models.demos.blackhole.glm53_flash.ttnn.{mod_name}")
        keep = getattr(mod, name)
        setattr(mod, name, (value == "1") if isinstance(keep, bool) else type(keep)(int(value)))
        undo.append((mod, name, keep))
    return lambda: [setattr(m, n, k) for m, n, k in reversed(undo)]


def wave_linear(mesh, x, w, **kw):
    """`ttnn.linear(x, w, **kw)`, a long prefill wave's rows with `wave_config` (the bits of waves of WAVE_ROWS); with
    PREFILL_KBLOCK, a prefill's rows (not a decode step's) with `kblock_config`."""
    if "program_config" not in kw:
        cfg = None
        if PREFILL_KBLOCK > 0 and x.shape[-2] >= 128 and not _in_decode():
            w_tile = 1088 if w.dtype == ttnn.bfloat8_b else (576 if w.dtype == ttnn.bfloat4_b else 2048)
            out_tile = 4096 if kw.get("dtype") == ttnn.float32 else 2048
            if PREFILL_KBLOCK_SPLIT:
                out = _kblock_split(mesh, x, w, w_tile, out_tile, **kw)
                if out is not None:
                    return out
            cfg = kblock_config(mesh, x.shape[-2], w.shape[-2], w.shape[-1], w_tile, out_tile, PREFILL_KBLOCK)
        if cfg is None:
            cfg = wave_config(mesh, x.shape[-2], w.shape[-2], w.shape[-1])
        if cfg is not None:
            kw["program_config"] = cfg
    return ttnn.linear(x, w, **kw)


def _in_decode() -> bool:
    """A decode step's matmuls (moe_route.decoding): DECODE_MM_WIDE leaves the prefill's chunks of 128 as they were."""
    from .moe_route import in_decode  # (it imports this module)

    return in_decode()


def _dram_interleaved(t) -> bool:
    return t.memory_config().buffer_type == ttnn.BufferType.DRAM and not t.is_sharded()


def decode_matmul_takes(x, w) -> bool:
    """decode_linear takes decode_matmul for x and w: at most DECODE_MATMUL_ROWS rows of x, a weight of
    DECODE_MATMUL_MIN_TILES tiles or more, both DRAM interleaved."""
    return (
        DECODE_MATMUL
        and x.shape[-2] <= max(32, DECODE_MATMUL_ROWS)
        and (w.shape[-1] // 32) * (w.shape[-2] // 32) >= DECODE_MATMUL_MIN_TILES
        and _dram_interleaved(x)
        and _dram_interleaved(w)
    )


def decode_linear(mesh, x, w, memory_config=None, **kw):
    """`ttnn.linear(x, w, **kw)`, for at most one tile row of x with a 1D config (in0 multicast over the grid): one
    output tile per core up to 64 of them, else 4 per core; K in blocks of 8 tiles (16 for 32 output tiles or fewer).
    The best or within 4 % of the best of the configs tried for each of the model's decode shapes.  A
    weight's matmul: HIFI4_FP32 comes as WEIGHT_MM (GLM53_MM_FIDELITY; decode_matmul has DECODE_MATMUL_FIDELITY).
    `memory_config` (`ar_input_config`): decode_matmul's output there; the other paths write DRAM interleaved."""
    if kw.get("compute_kernel_config") is HIFI4_FP32:
        kw["compute_kernel_config"] = WEIGHT_MM
    if memory_config is not None and "dtype" not in kw:  # (GLM53_AR_BFP8: the row all-reduce's share in BFP8_b)
        d8 = ar_input_dtype(x.shape[-2])
        if d8 is not None:
            kw["dtype"] = d8
    if decode_matmul_takes(x, w) and set(kw) <= {"compute_kernel_config", "dtype"}:
        return ttnn.experimental.deepseek_prefill.decode_matmul(
            x, w, output_dtype=kw.get("dtype"), memory_config=memory_config
        )
    if x.shape[-2] <= max(32, DECODE_MM_ROWS):
        n, k, m = w.shape[-1] // 32, w.shape[-2] // 32, -(-x.shape[-2] // 32)
        if (n, k, m) not in _DECODE_MM:
            grid = mesh.compute_with_storage_grid_size()
            per_core = max(1 if n <= 64 else 4, -(-n // (grid.x * grid.y)))  # at most one block per core
            assert per_core <= 4, (  # (out_subblock_w = per_core: a wider subblock passes fp32 Dest's 4 tiles)
                f"decode_linear: {n} output tiles give {per_core} a core on {grid.x} x {grid.y}, past the 4-tile "
                "subblock this 1D config takes"
            )
            block = 16 if n <= 32 else 8
            while k % block:
                block //= 2
            _DECODE_MM[(n, k, m)] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=per_core,
                out_block_h=m,
                out_block_w=per_core,
                per_core_M=m,
                per_core_N=per_core,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        kw["program_config"] = _DECODE_MM[(n, k, m)]
    elif x.shape[-2] <= DECODE_MM_WIDE and "program_config" not in kw and _in_decode():
        n, k, m = w.shape[-1] // 32, w.shape[-2] // 32, -(-x.shape[-2] // 32)
        if (n, k, m) not in _DECODE_MM:
            grid = mesh.compute_with_storage_grid_size()
            per_core = -(-n // (grid.x * grid.y))
            block = 8
            while k % block:
                block //= 2
            _DECODE_MM[(n, k, m)] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=max(v for v in (4, 2, 1) if per_core % v == 0),
                out_block_h=m,
                out_block_w=per_core,
                per_core_M=m,
                per_core_N=per_core,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        kw["program_config"] = _DECODE_MM[(n, k, m)]
    return wave_linear(mesh, x, w, **kw)


# the all-reduce over a row's 4 chips for one tile row of tokens as tt-metal's all_reduce_async (Llama Galaxy's decode
# all-reduce: the input width-sharded in L1 over TP_AR_ASYNC_CORES cores, gathered into a persistent L1 buffer and
# summed there), on both links of the row's ring, moved from and back to DRAM interleaved around it.  Alone (trace)
# 28.7 -> 23.3 us (19.6 us without the moves; on 1 link or a line 31-39 us); decode 8 / 32 / 128 users
# -0.7 / -0.8 / -0.6 ms, trace bit for bit as eager; its sums in fp32 (Dest).  Above 4 tokens a row
# (TP_AR_CUSTOM below); ttnn.all_reduce for what it does not take
TP_AR_ASYNC = True
TP_AR_ASYNC_CORES = 32
_TP_AR_ASYNC: dict = {}
# ... its multicast split by output core (tt-metal's TT_AR_ASYNC_MCAST_SPLIT, read at each call: on the row's ring of 4
# the device sends 2 hops one way and 1 the other, even and odd devices the other way round, so the busiest link
# carried twice a device's share; with the split the odd output cores of each link go the other way round, every link
# 1.5 shares).  The sums the same bit for bit (8 - 64 tokens a row); alone 17.2 / 28.6 -> 16.5 / 25.6 us at 32 / 64
# tokens a row.  TT_AR_ASYNC_MCAST_SPLIT=0 in the environment before this import: as before
os.environ.setdefault("TT_AR_ASYNC_MCAST_SPLIT", "1")
# ... and at 8 / 16 tokens a row only the token rows sent (tt-metal's TT_AR_ASYNC_COMPACT, read at each call: each
# output core's share packed as its tiles' first t rows, 2 KB in place of 8 at 8 tokens, summed packed and put back, the
# padding rows zero).  The token rows' sums the same bit for bit (the op and the model); alone 18.3 / 18.7 -> 14.1 /
# 15.0 us at 8 / 16 tokens a row; the DFlash2 step at 8 users -0.26 ms.  On by default; TT_AR_ASYNC_COMPACT=0 in the
# environment before this import: as before
os.environ.setdefault("TT_AR_ASYNC_COMPACT", "1")
# ... and on the 2D fabric the writer's packet headers' multicast routes made on the host and passed as runtime args
# (tt-metal's TT_AR_ASYNC_HOST_ROUTES, read at each call), in place of encoding them in the kernel before its first
# packet (4 encodes ~1.9 us a call).  The same header bytes, the same sends (the check form 2 found them equal on every
# header; the op and the model bit for bit).  Alone -1.6 / -1.5 us at 32 / 64 tokens a row; the DFlash2 step less the
# kernel config ring's share -0.14 / -0.21 ms at 32 / 128 users.  On by default; 2: the check (encoded as before, the
# host's bytes compared, DPRINT); TT_AR_ASYNC_HOST_ROUTES=0 in the environment before this import: as before
os.environ.setdefault("TT_AR_ASYNC_HOST_ROUTES", "1")
# The sender of the experts op's in-op column gather (ttnn/moe.py H1, 42 calls a step at the fixed shapes): on the 2D
# fabric its 4 packet headers' multicast routes (4 / 3 hops each way along the mesh column) made on the host and passed
# as runtime args (tt-metal's MOE_H1_HOST_ROUTES, read at each call), in place of encoding them in the kernel before
# its first packet (~2.45 us a call on the way to the column's routing and x).  The same header bytes, the same sends
# (the check form 2 equal on every header, the MoE layers and the model bit for bit); the DFlash2 step -0.08 / -0.05 /
# -0.065 ms at 8 / 32 / 128 users.  On by default; 2: the check; MOE_H1_HOST_ROUTES=0 in the environment before this
# import: as before
os.environ.setdefault("MOE_H1_HOST_ROUTES", "1")
# The traces' kernel config ring placed by a plan over the whole trace (tt-metal's TT_METAL_TRACE_ALLOC_BEAM, read at
# each trace's capture; SimpleTraceAllocator `plan_binaries`): the Tensix binaries' places chosen by a beam search of
# width 8 so that a program seldom waits for the one just before it to end before its binary is written (the decode
# step had 23 / 45 such stalls a verify trace at 32 / 128 users, ~5 - 11 us each).  What is written and the kernels
# are the same, so are the values.
# TT_METAL_TRACE_ALLOC_BEAM=0 in the environment before this import: tt-metal's placement as before
os.environ.setdefault("TT_METAL_TRACE_ALLOC_BEAM", "8")
# The decode mHC write + read's comb (Sinkhorn) and post on a finisher core a group, off the mix core's chain (the
# mhc_stream_read factory's GLM53_MHC_FINISH, read at the program's creation: 2 = the mixes written before P, signalled
# after P's send).  Only the decode forms that leave the most units a write core makes unchanged: 8 / 32 tokens a mesh
# row (8 / 32 users' verify steps at K = 7), not 64.  The same calls on the same values; the DFlash2 step at 32 users
# -0.186 ms on the real weights.  GLM53_MHC_FINISH=0 in the environment before this import: as before
os.environ.setdefault("GLM53_MHC_FINISH", "2")
# ... up to this many tokens a row: 64 also takes two tile rows (128 users' verify step at T = 4, 512 users' plain
# step; ttnn.all_reduce's reduce-scatter + all-gather there, ~53 us x 90 calls), their shards [64, D / 64] on
# 2 x TP_AR_ASYNC_CORES cores, and every call's gathered landing area those 64 cores' [64, 4 D / 64] (2 x 32 KB a core,
# as the one tile row's on 32 cores: two tile rows on 32 cores took 2 x 64 KB and met the static CBs); the one
# tile row's calls keep their 32 cores and land there too (the op checks only the landing shard's size and cores).
# Alone (trace) 40.7 -> 35.8 us at 64 tokens, 32 tokens bit for bit and as fast, the landing areas at the same L1
# addresses; every prefill chunk up to 512 after 512 users' decode in one process, the L1 probe as before; the
# verify step at 128 users x T = 4 62.29 / 62.22 -> 60.52 / 60.58 ms, 512 users' plain step 64.4
# -> 62.9 ms (synthetic weights); the sums closer to fp32's than ttnn.all_reduce's (bf16 adds), the KLD at 512
# users the same (0.000601 -> 0.000631 against the CPU decode path, the paired difference's 95 % interval -0.00007 ..
# +0.00013; 0.000508 -> 0.000472 against the prefill path; top-1 1.0); stresses clean.  32: as before
TP_AR_ASYNC_ROWS = 64
# ... its sums left sharded in L1 for the mHC write that takes them (`mhc_stream_write` reads block_out through a
# TensorAccessor, interleaved or sharded), in place of the move back to DRAM: only inside `ar_keep_l1`, which the
# decoder layer opens around its sub-layers in decode (their all-reduce's output goes to its mHC write alone).  The
# outputs bit for bit; synthetic weights, traced, off / on twice: the verify step at 128 users 52.4 / 52.5
# -> 52.2 / 52.2 ms, the plain step 47.2 / 47.3 -> 47.0 / 46.9, 32 users -0.1 ms, 8 users (the custom all-reduce) the
# same.  0: the move, as before
AR_L1_OUT = True
_AR_KEEP_L1 = [False]
# ... and its input written there by the matmul before it (decode_matmul's memory_config: the KDA and DSA o_proj, the
# dense MLP's down), in place of the move in (`ar_input_config`).  The outputs bit for bit; synthetic weights,
# traced, off / on twice: the verify step at 128 users 4,721 / 4,720 -> 4,734 / 4,738 tok/s, the plain step 45.8 /
# 45.8 -> 45.6 / 45.3 ms, 32 users within the noise.  0: the move, as before
AR_L1_IN = True


@contextmanager
def ar_keep_l1(on: bool = True):
    """all_reduce_async's output stays sharded in L1 inside this block (AR_L1_OUT)."""
    prev = _AR_KEEP_L1[0]
    _AR_KEEP_L1[0] = on and AR_L1_OUT
    try:
        yield
    finally:
        _AR_KEEP_L1[0] = prev


# up to 4 tokens a row: the row all-reduce as one-shot fabric writes of the tokens' rows only (`row_all_reduce.py`,
# generic_op; the sums bit for bit as all_reduce_async's), straight from and to DRAM: 20.5 -> 9.1 / 9.6 us at 1 / 4
# tokens a row (trace, 100 calls), decode 8 / 32 users -1.2 / -1.0 ms (synthetic weights), real weights 23.2 / 29.4
# -> 21.9 / 28.3 ms, the KLD the same; 4 x 2,000-step stress runs at 8 and 32 users clean.  No
# NoC multicast inside it.  Its landing area in DRAM (`OneshotArea`) and its few semaphores (1.5 KB of L1) taken when
# the model loads (`model.prealloc_fabric_ops`), before any program or trace: an L1 landing area clashed with a later
# chunk-512 prefill's static circular buffers (up to 1.39 MB); now 9.5 / 10.0 us, chunk 512's L1 headroom 43 KB,
# decode 8-512 users and every prefill chunk traced and eager in one process clean
TP_AR_CUSTOM = True
# ... up to this many tokens a row (above 4 with 32 workers, a slice of a whole face pair of 4 tiles each: 128 users'
# all_reduce_async was 26-35 us, 90 calls a step)
TP_AR_CUSTOM_MAX = 4


def _ar_path(t: int, d: int, dtype=None, layout=None) -> str:
    """The row all-reduce's path for t tokens of width d (`tp_all_reduce`): "custom", "async" or "ttnn"."""
    dtype = ttnn.bfloat16 if dtype is None else dtype
    layout = ttnn.TILE_LAYOUT if layout is None else layout
    if (
        TP_AR_CUSTOM
        and t <= TP_AR_CUSTOM_MAX
        and dtype == ttnn.bfloat16
        and layout == ttnn.TILE_LAYOUT
        and d % (32 * (8 if t <= 4 else 32)) == 0
    ):
        return "custom"
    if TP_AR_ASYNC and t <= 32 and d % (32 * TP_AR_ASYNC_CORES) == 0:
        return "async"
    if TP_AR_ASYNC and t <= TP_AR_ASYNC_ROWS <= 64 and d % (32 * 2 * TP_AR_ASYNC_CORES) == 0:
        if not ar_land_narrow():
            return "async"
        if not _NARROW_NOTED:  # (the narrow landing holds one tile row: two go to ttnn.all_reduce, slower)
            _NARROW_NOTED.append(t)
            print(
                f"NOTE row all-reduce: {t} tokens a row with GLM53_AR_LAND_NARROW=1 go to ttnn.all_reduce", flush=True
            )
    return "ttnn"


_NARROW_NOTED: list = []


def ar_input_config(mesh, t: int, d: int):
    """Where all_reduce_async takes [1, 1, t, d] (L1 width-sharded) when the row all-reduce goes that way (AR_L1_IN), for
    the op before it to write there; else None (DRAM interleaved)."""
    if not AR_L1_IN or _ar_path(t, d) != "async" or (t > 32 and not AR_L1_IN_WIDE):
        return None
    return _tp_ar_async(mesh, d, TP_AXIS).shard_for(t)


# ... the matmuls before it (and the MoE's reduce-scatter) writing the two tile rows' shards too (TP_AR_ASYNC_ROWS 64):
# the logits bit for bit as with the move in, the verify step at 128 users x T = 4 60.52 -> 60.25 ms, 512 users'
# plain step 62.9 -> 62.7 ms.  0: the move in there
AR_L1_IN_WIDE = True


# ... or in two stages (`row_ar2.py`, generic_op: a one-shot reduce-scatter, then a multicast all-gather of the sums,
# 0.4375 of a chip's share on each link in place of 0.75; the sums all_reduce_async's bit for bit: its compute kernel
# on the same 4 slots): the DFlash2 step at 128 / 32 users -0.89 / -0.22 ms, 2 x 2,000 steps bit for bit at both.
# GLM53_AR_TWO_STAGE, read at each call: "2" (the default) above 16 tokens a row (whole tiles: 32 and 64), "1" above 32
# only, "0" all_reduce_async as before
def _ar_two_stage(t: int) -> bool:
    mode = os.environ.get("GLM53_AR_TWO_STAGE", "2")
    return (mode == "1" and t > 32) or (mode == "2" and t > 16)


# ... its landing areas narrow (GLM53_AR_LAND_NARROW=1, read when the state is made; 0: as above).  The device's L1
# allocator is lock-step (one address range on every core), so the two landing areas of 64 cores x 32 KB take 64 KB of
# every core's L1 under the experts' L1 landing (GLM53_W1), and every program's circular buffers must end below them
# (the prefill's highest at 8 / 32 users' C 10: the MoE's gate | up, 9 KB to spare).  Narrow, for models whose decode never takes two tile rows a mesh row (8 / 32 users K = 7; two go to ttnn.all_reduce):
#   - one tile row of 17 - 32 tokens on the 64 cores, [32, D / 64] a core (2 tiles; the two-stage form, row_ar2,
#     takes any core count), its landing [32, 4 D / 64] (16 KB a core);
#   - one tile row of 8 / 16 tokens packed (TT_AR_ASYNC_COMPACT) keeps its 32 cores, landing in the same areas: a
#     chip's packed pages t x 64 bytes a tile, ring x 1 / 2 pages a core (all_reduce_async takes a landing that holds
#     those: its validate and reduction CB, tt-metal);
#   - the rest of one tile row (unpacked) on the 64 cores as 17 - 32.
# Each output element is the same four values added in the same order on whichever core holds it: the same sums.
# 2 x 16 KB of every core's L1 back: the op alone the same bits at 8 / 16 / 32 tokens (watcher clean), 15.20 -> 14.59
# us at 32; in the generator flow at 32 / 8 seats with the GLM53_W1 landing at C 10 and the three L1 switches the tokens
# of every request the same as the wide landing's and the prefill's highest circular buffer (mp_gu5) 41,920 B below
# the lowest L1 in place of 9,152; the DFlash2 step on the real weights at 32 users K = 7 -0.23 ms at C 10 (3,635 ->
# 3,658 tok/s), -0.15 at C 5; the verify step's ids and logits the same, 2,000-step stresses at 32 / 8 users.  On by
# default (`ar_land_narrow_default`) for a model that serves a DFlash2 verify step of at most 32 tokens a mesh row (8 /
# 32 users K = 7); GLM53_AR_LAND_NARROW=0 in the environment: as before
def ar_land_narrow() -> bool:
    return os.environ.get("GLM53_AR_LAND_NARROW", "0") == "1"


def ar_land_narrow_default(dflash_shape) -> None:
    """GLM53_AR_LAND_NARROW's default for the process, from the DFlash2 verify step its model serves (`dflash_shape`:
    users a mesh row, tokens a user; TtGlm53Model's): on at most 32 tokens a mesh row, where the decode never takes two
    tile rows (above, the narrow landing would send them to ttnn.all_reduce); unset otherwise (off).  Before the
    first all-reduce (the state reads it when made)."""
    if dflash_shape and dflash_shape[0] * dflash_shape[1] <= 32:
        os.environ.setdefault("GLM53_AR_LAND_NARROW", "1")


def _ar_compact(t: int) -> bool:
    """Whether all_reduce_async packs these t tokens (8 / 16, TT_AR_ASYNC_COMPACT, as its invoke decides) on this call."""
    return t in (8, 16) and os.environ.get("TT_AR_ASYNC_COMPACT") == "1" and not _ar_two_stage(t)


# The row all-reduce's input share in BFP8 (GLM53_AR_BFP8=1, read at each call; value-changing; default off): where the
# row all-reduce goes in two stages (`_ar_two_stage`), the op before it writes its share in BFP8_b (decode_matmul's
# output_dtype), so the reduce-scatter's shares cross the links and the senders' readers in 1,088-B tiles in place of
# 2,048 (row_ar2.py; the sums, the all-gather and the output stay bf16: each chip's share rounded once to BFP8, the 4
# shares added in fp32 as before).
def ar_input_dtype(t: int):
    """ttnn.bfloat8_b for the row all-reduce's input share at t tokens a row with GLM53_AR_BFP8 on, else None (the
    default)."""
    return ttnn.bfloat8_b if os.environ.get("GLM53_AR_BFP8") == "1" and _ar_two_stage(t) else None


class _TpArAsync:
    """all_reduce_async's persistent state for a mesh, axis and width: two L1 buffers and two global semaphores, taken
    in turn (as Llama cycles them: consecutive all-reduces never share one; a step makes an even number of calls on
    each axis, so a replayed trace keeps the alternation).  Made on the first (eager) call, before any trace capture."""

    def __init__(self, mesh, d: int, axis: int = TP_AXIS):
        grid = mesh.compute_with_storage_grid_size()
        full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        cores = ttnn.num_cores_to_corerangeset(TP_AR_ASYNC_CORES, grid, row_wise=True)
        width = d // TP_AR_ASYNC_CORES

        def sharded(w, rows=32, on=cores):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(on, [rows, w], ttnn.ShardOrientation.ROW_MAJOR),
            )

        n_dev = COLS if axis == TP_AXIS else ROWS
        self.shard = sharded(width)
        self.wide = None  # two tile rows (TP_AR_ASYNC_ROWS 64): their shards, on twice the cores
        self.narrow_wanted = ar_land_narrow()  # (GLM53_AR_LAND_NARROW when made: `_tp_ar_async` remakes on a change)
        self.narrow = None  # one tile row on twice the cores (GLM53_AR_LAND_NARROW): its shards
        land_rows, land_width, land_cores, n_cores = 32, width, cores, TP_AR_ASYNC_CORES
        if self.narrow_wanted and d % (32 * 2 * TP_AR_ASYNC_CORES) == 0:
            n_cores = 2 * TP_AR_ASYNC_CORES
            land_cores = ttnn.num_cores_to_corerangeset(n_cores, grid, row_wise=True)
            land_width = d // n_cores
            self.narrow = sharded(land_width, 32, land_cores)
        elif TP_AR_ASYNC_ROWS > 32:
            n_cores = 2 * TP_AR_ASYNC_CORES
            land_cores = ttnn.num_cores_to_corerangeset(n_cores, grid, row_wise=True)
            land_rows, land_width = 64, d // n_cores
            self.wide = sharded(land_width, 64, land_cores)
        self.bufs = []
        for _ in range(2):
            b = put(mesh, torch.zeros(ROWS, COLS, land_rows, land_width * n_dev * n_cores), dims=(0, 1))
            self.bufs.append(ttnn.to_memory_config(b, sharded(land_width * n_dev, land_rows, land_cores)))
            ttnn.deallocate(b)
        self.sems = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(2)]
        self.mesh = mesh
        self.axis = axis
        self.turn = 0
        self.paths: dict = {}  # calls by path ("row_ar2" / "async"), for the parity check

    def release(self) -> None:
        """Its landing areas freed (a state made again for another form: no trace may still use them)."""
        for b in self.bufs:
            ttnn.deallocate(b)
        self.bufs = []

    def shard_for(self, t: int):
        """The input and output shards for t tokens a row: one tile row's, or two's (TP_AR_ASYNC_ROWS 64); narrow
        (GLM53_AR_LAND_NARROW): 32 cores' for 8 / 16 packed tokens, else the 64 cores'."""
        if self.narrow is not None:
            assert t <= 32, (t, "the narrow landing holds one tile row")
            return self.shard if _ar_compact(t) else self.narrow
        if t <= 32:
            return self.shard
        assert self.wide is not None and t <= 64, (t, TP_AR_ASYNC_ROWS)
        return self.wide

    def __call__(self, x):
        i = self.turn % 2
        self.turn += 1
        two = _ar_two_stage(x.shape[-2])
        path = "row_ar2" if two else "async"
        self.paths[path] = self.paths.get(path, 0) + 1  # (the parity check's counts by path)
        shard = self.shard_for(x.shape[-2])
        own = not (x.is_sharded() and x.memory_config() == shard)  # (the matmul before wrote it there)
        a = ttnn.to_memory_config(x, shard) if own else x
        if two:
            from .row_ar2 import row_ar2  # (it imports this module)

            s = row_ar2(self.mesh, a, self.bufs[i], self.sems[i])
        else:
            s = ttnn.experimental.all_reduce_async(
                a,
                self.bufs[i],
                cluster_axis=self.axis,
                mesh_device=self.mesh,
                multi_device_global_semaphore=self.sems[i],
                num_links=2,
                memory_config=shard,
                topology=ttnn.Topology.Ring,
                fp32_dest_acc=True,
            )
        if own:
            ttnn.deallocate(a)
        if _AR_KEEP_L1[0]:
            return s
        out = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s)
        return out


def _tp_ar_async(mesh, d: int, axis: int) -> "_TpArAsync":
    key = (id(mesh), d, axis)
    st = _TP_AR_ASYNC.get(key)
    if st is not None and st.mesh is mesh and st.narrow_wanted != ar_land_narrow():
        st.release()  # (a test's arms switch GLM53_AR_LAND_NARROW: the landing areas made again)
        st = None
    if st is None or st.mesh is not mesh:
        _TP_AR_ASYNC[key] = _TpArAsync(mesh, d, axis)
    return _TP_AR_ASYNC[key]


def all_reduce_async(mesh, x, axis: int):
    """x [1, 1, T <= 32 (64: TP_AR_ASYNC_ROWS), D] bf16 -> its sum over the mesh axis, by all_reduce_async
    (`_TpArAsync`), DRAM interleaved."""
    return _tp_ar_async(mesh, x.shape[-1], axis)(x)


# the prefill's row all-reduce (past 64 tokens a row: ttnn.all_reduce, its reduce_scatter + all_gather on the last dim)
# as those two ops called here with ttnn's tuning knobs, "<from tokens a row>:w<workers a link>c<chunks a sync>" (the
# same ops on the same dim: the sums bit for bit as ttnn.all_reduce's).  Alone (traced, back to back) 512 tokens a row
# 120.4 -> 114.1 us; 128 tokens gained nothing.  Read at each call (tests switch it).  "": ttnn.all_reduce
TP_AR_SPLIT_KNOBS = "256:w8c8"


def _tp_ar_split(t: int) -> dict | None:
    import re

    spec = TP_AR_SPLIT_KNOBS
    if not spec or ":" not in spec or t < int(spec.split(":")[0]):
        return None
    keys = {"w": "num_workers_per_link", "c": "chunks_per_sync", "b": "num_buffers_per_channel"}
    return {keys[a]: int(b) for a, b in re.findall(r"([wcb])(\d+)", spec.split(":")[1])}


def tp_all_reduce(mesh, x):
    """x [1, 1, T, D] bf16, each chip's partial -> their sum over the row's chips, on every one of them."""
    path = _ar_path(x.shape[-2], x.shape[-1], x.dtype, x.layout)
    if path == "custom":
        from .row_all_reduce import row_all_reduce  # (it imports this module)

        return row_all_reduce(mesh, x)
    if path == "async":
        return all_reduce_async(mesh, x, TP_AXIS)
    kw = _tp_ar_split(x.shape[-2])
    if kw is not None and x.layout == ttnn.TILE_LAYOUT:
        part = ttnn.reduce_scatter(x, dim=3, cluster_axis=TP_AXIS, **kw)
        if ROW_SPLIT_A2A and quarter_of_rows(x.shape[-2]):
            # (`row_quarter`: chip c's tokens c T / 4 .. of the sums, [1, 1, T / 4, D], in place of the all-gather)
            out = ttnn.experimental.all_to_all_async_generic(part, in_dim=3, out_dim=2, cluster_axis=TP_AXIS)
        else:
            out = ttnn.all_gather(part, dim=3, cluster_axis=TP_AXIS, **kw)
        ttnn.deallocate(part)
        return out
    return ttnn.all_reduce(x, cluster_axis=TP_AXIS)


# mHC over the row's chips by tokens in a long prefill wave.  The streams of a row's tokens are the same on its 4 chips
# (mhc.py): each chip read and wrote all of the row's tokens, 4 times the work (1 / 10 of a wave's time at 1,024 tokens
# a row).  With ROW_SPLIT (tokens a row from which it runs, a multiple of 128), chip c
# of a row keeps the streams of the row's tokens c T / 4 .. only:
#   - a sub-layer's output comes to it as that quarter: the row all-reduce's reduce-scatter, then an all-to-all over the
#     row (ROW_SPLIT_A2A: [T, D / 4] -> [T / 4, D]) in place of its all-gather; the MoE's one-op way back without its
#     row all-gather, then the same all-to-all (moe.ROW_SPLIT_MOE_A2A); else the output with its quarter taken
#     (`row_quarter_of`);
#   - mHC writes and reads T / 4 tokens (its groups of 32 tokens are independent: `_read_pieces`, `write_read`);
#   - the block input it reads goes to the row's T tokens by an all-gather over the row on the tokens.
# Every value the same as without (the reduce-scatter is the all-reduce's own, the rest copies): test_model_bits at
# chunks 1,024 / 512 all bit for bit.  A sub-layer's end alone at 1,024 tokens 523 -> 353 us; the prefill 8 x 2,048
# 0.6086 -> 0.5768 s (-5.2 %), 1 x 8K 0.3717 -> 0.3545 s (-4.6 %), synthetic weights.
# 512 (default): waves of 512 and 1,024 tokens a row; 0: off (every chip all of the row's tokens, as before)
ROW_SPLIT = int(os.environ.get("GLM53_MHC_ROW_SPLIT", "512"))
ROW_SPLIT_A2A = os.environ.get("GLM53_MHC_ROW_SPLIT_A2A", "1") == "1"
_ROW_QUARTER: list = []  # (`row_quarter`'s tokens a row, innermost last)


def row_split_on(t: int) -> bool:
    """A prefill wave of t tokens a row runs mHC on its quarter a chip (ROW_SPLIT)."""
    return ROW_SPLIT > 0 and t >= ROW_SPLIT and t % (COLS * 32) == 0


def quarter_of_rows(t: int) -> bool:
    """Whether the row all-reduce of t rows here returns this chip's quarter of the tokens (inside `row_quarter(t)`)."""
    return bool(_ROW_QUARTER) and _ROW_QUARTER[-1] == t and t % (COLS * 32) == 0


@contextmanager
def row_quarter(t: int):
    """Within it, the row all-reduce of t rows (`tp_all_reduce`, the reduce-scatter and all-gather path) returns this
    chip's quarter of the row's tokens [1, 1, t / 4, D] (ROW_SPLIT_A2A); the other paths are unchanged (callers take the
    quarter of a full output with `to_row_quarter`)."""
    _ROW_QUARTER.append(t)
    try:
        yield
    finally:
        _ROW_QUARTER.pop()


def row_quarter_of(mesh, x):
    """x [1, 1, T, D] tiled, the same on a row's 4 chips -> chip c's tokens c T / 4 .. as [1, 1, T / 4, D] (a copy of
    those pages: `row_split.take`)."""
    from .row_split import take  # (it imports this module)

    t, d = x.shape[-2], x.shape[-1]
    pages = (t // COLS // 32) * (d // 32)
    return take(mesh, x, [1, 1, t // COLS, d], 1, pages, 0, [c * pages for c in range(COLS)])


def to_row_quarter(mesh, y, t: int):
    """A sub-layer's output under `row_quarter(t)` -> this chip's quarter [1, 1, t / 4, D]: as it is when the
    all-to-all made it, else (the row's t tokens) its quarter taken and the source deallocated."""
    if y.shape[-2] == t // COLS:
        return y
    q = row_quarter_of(mesh, y)
    ttnn.deallocate(y)
    return q


def row_tokens_gather(x):
    """Each chip's quarter [1, 1, T / 4, D] -> the row's T tokens [1, 1, T, D] (an all-gather over the row on the
    tokens, the row all-reduce's knobs)."""
    kw = _tp_ar_split(x.shape[-2] * COLS) or {}
    return ttnn.all_gather(x, dim=2, cluster_axis=TP_AXIS, **kw)


def swiglu_activations(limit: float) -> dict:
    """GLM's clamped SwiGLU, silu(min(gate, limit)) * clamp(up, -limit, limit), as the input activations of one
    multiply: `ttnn.multiply(gate, up, **swiglu_activations(limit))` (one op instead of clamp, clamp, silu, multiply).
    """
    return dict(
        input_tensor_a_activations=[
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MINIMUM, float(limit)),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        ],
        input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, -float(limit), float(limit))],
    )


HOST_THREADS = 16


def host_threads() -> None:
    """Weight building stacks and transposes gigabytes on the host; under the test runner torch starts with 1 thread,
    which made the whole-model load ~1 minute per MoE layer."""
    if torch.get_num_threads() < HOST_THREADS:
        torch.set_num_threads(HOST_THREADS)


def mapper(mesh, dims):
    """dims = (tensor dim split over the rows or None, tensor dim split over the row's chips or None)."""
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=dims)


# The small inputs written from the host every step (positions, ids, flags): one host tensor per device buffer, made
# once from a torch buffer whose storage it borrows (from_torch keeps a row-major tensor's storage when the dtypes
# match), refilled in place each step: a write costs the copy alone, ~0.12 ms against ~0.32 with from_torch.
# Checked once per buffer (the copy reads back what the torch buffer holds), else made each step
HOST_REUSE = True
_HOST_INPUTS: dict = {}  # (id(device tensor), shape, dtype) -> (torch buffer, host tensor), or None: not borrowed


def write_rows(mesh, values: torch.Tensor, device_tensor, dtype=ttnn.int32) -> None:
    """values (integers, dim 0 split over the mesh rows) -> device_tensor, row-major int32 / uint32 as `put` made it
    with dims (0, None), through a reused host tensor (HOST_REUSE).  The device must be idle the first time (a read)."""
    rows = mapper(mesh, (0, None))
    key = (id(device_tensor), tuple(values.shape), dtype)
    entry = _HOST_INPUTS.get(key, False) if HOST_REUSE else None
    if entry is False:  # the first write to this buffer: make the host tensor and check that it borrows
        entry = None
        buf = torch.zeros(values.shape, dtype=torch.uint32 if dtype == ttnn.uint32 else torch.int32)
        try:
            host = ttnn.from_torch(buf, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rows)
            pattern = torch.arange(buf.numel(), dtype=torch.int32).view(buf.shape) + 7
            buf.view(torch.int32).copy_(pattern)
            ttnn.copy_host_to_device_tensor(host, device_tensor)
            got = ttnn.to_torch(ttnn.get_device_tensors(device_tensor)[0]).flatten()
            got = got.view(torch.int32) if got.dtype == torch.uint32 else got.to(torch.int32)
            want = pattern.reshape(ROWS, -1)[0].flatten()
            if torch.equal(got[: want.numel()], want):
                entry = (buf, host)
        except (RuntimeError, TypeError):
            pass
        _HOST_INPUTS[key] = entry
    if entry is None:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(values.to(torch.int32), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rows),
            device_tensor,
        )
        return
    buf, host = entry
    buf.view(torch.int32).copy_(values)
    ttnn.copy_host_to_device_tensor(host, device_tensor)


SYNTHETIC_RANGE = 0.02  # random weights (from a ShapeOnlyCheckpoint) are uniform in [-this, this)


# synthetic weights: the same values on every chip (a seed per tensor).  ttnn.rand's seed 0 draws a seed per core and
# chip, so a replicated weight such as the router differed from chip to chip: each chip of a mesh column routed the
# tokens differently, and the experts' op took up to 2x as long on some chips of a column as on others, which the
# real weights never do (1.8 ms of the 8-user step)
_SYNTHETIC_TENSORS = [0]


def synthetic(mesh, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Random values of a per-chip shape, made on every chip of the mesh (nothing crosses PCIe)."""
    if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        layout = ttnn.TILE_LAYOUT
    _SYNTHETIC_TENSORS[0] += 1
    seed = (_SYNTHETIC_TENSORS[0] * 4096) % (1 << 31) + 1  # a core takes seed + its index: streams kept apart
    return ttnn.rand(
        [int(d) for d in shape],
        device=mesh,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        low=-SYNTHETIC_RANGE,
        high=SYNTHETIC_RANGE,
        seed=seed,
    )


def slice_copy(t, begins, ends):
    """`ttnn.slice` that always returns a new tensor: over the whole tensor ttnn.slice returns its input, and
    deallocating that result would free the input (a 1-user KDA state, a 1-user DSA cache)."""
    if all(int(b) == 0 for b in begins) and [int(e) for e in ends] == [int(d) for d in t.shape]:
        return ttnn.clone(t)
    return ttnn.slice(t, begins, ends)


def zeros(mesh, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Zeros of a per-chip shape, made on every chip (states: sending host zeros took ~40 s at 32 users per row)."""
    return ttnn.zeros(
        [int(d) for d in shape], dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


_CONST_ZEROS: dict = {}


def const_zeros(mesh, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Read-only zeros of a per-chip shape, made once per mesh, shape and type and shared (never deallocate them): a
    captured trace may read them, whereas making new zeros inside a capture writes from the host (not allowed)."""
    key = (tuple(int(d) for d in shape), dtype, layout)
    hit = _CONST_ZEROS.get(key)
    if hit is None or hit[0] is not mesh:  # a new mesh (tests open several) makes its own
        hit = _CONST_ZEROS[key] = (mesh, zeros(mesh, shape, dtype, layout))
    return hit[1]


_CONSTS: dict = {}


def const_tensor(mesh, name: str, make) -> object:
    """A read-only replicated constant made once per mesh from `make()` (a host tensor, bf16 tiled) and shared (never
    deallocate it; as `const_zeros`, a captured trace may read it)."""
    hit = _CONSTS.get(name)
    if hit is None or hit[0] is not mesh:
        hit = _CONSTS[name] = (mesh, put(mesh, make(), dims=(None, None)))
    return hit[1]


class SpGroup(NamedTuple):
    """One request's part of a sequence-parallel prefill wave: its segments of T tokens on the mesh rows `rows` (up the
    ring order from the wave's first row, consecutive or with other groups' rows between them,
    `generator.SP_INTERLEAVE`; the last is the request's own row),
    rows[i] holding positions start + i T .. with valid[i] real tokens (only the last may have fewer than T)."""

    rows: tuple
    valid: tuple
    user: int  # the request's user on its own row
    start: int
    last: bool = True  # the request's last wave: its KDA state goes into its user

    @property
    def owner(self) -> int:
        return self.rows[-1]


def sp_first(groups) -> int:
    """A wave's first row in the ring order its groups follow (its first group's first row)."""
    return groups[0].rows[0]


def per_chip_shape(shape, dims) -> list:
    out = list(shape)
    for d, parts in zip(dims, MESH_SHAPE):
        if d is not None:
            out[d] //= parts
    return out


def put(mesh, t: torch.Tensor, dims, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Upload with the given mapping.  A large bf16 TILE tensor goes up row-major and is tiled on the device: the host
    tilize is single-threaded and made the whole-model load take ~1 minute per MoE layer.  A meta tensor (shape only,
    `ShapeOnlyCheckpoint`) becomes random values made on the device."""
    if t.is_meta:
        return synthetic(mesh, per_chip_shape(t.shape, dims), dtype, layout)
    if layout == ttnn.TILE_LAYOUT and dtype == ttnn.bfloat16 and t.numel() >= DEVICE_TILIZE_MIN:
        rm = ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper(mesh, dims),
        )
        tiled = ttnn.to_layout(rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rm)
        return tiled
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper(mesh, dims),
    )


# FixedInputs.place skips a write whose values its buffer holds already (the host tensor it last wrote there, compared
# on the host: a chunk's inputs are only read on the device).  A traced prefill's inputs took ~0.75 ms a buffer on the
# host (from_torch and the copy, 45 of them for a pair of sequence-parallel waves: 45 ms with the device idle).
# Read at each call.  False (a benchmark's, tests/test_prefill_bench.py GLM53_FIXED_INPUTS_SKIP=0: a repeated prompt
# timed as a new one): every write
FIXED_SKIP_SAME = True


class FixedInputs:
    """Inputs a captured trace reads, at fixed device addresses: `place(name, t, dims, dtype, layout)` writes `t` (a
    host tensor mapped as `put` maps it) into the buffer `name`, made on the first call (which must come before any
    capture), and returns that buffer (FIXED_SKIP_SAME: not written again with the values it holds).  Between `begin`
    and `end` with a pack (`make_pack`) the places only stage their values, and `end` writes them all in one copy (->
    the layout, for `unpack`): a traced prefill's inputs in one host write (`prefill_pack`, the pack's unpack the
    captured prefill's first op).  `eager_place` is the counterpart that makes a new
    tensor each time."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.buffers: dict = {}
        self.written: dict = {}  # name -> (the host tensor last written, dims, dtype, layout)
        self.pack = None  # prefill_pack.Pack (`make_pack`)
        self.session = None  # the prefill being placed (`begin`)
        self.most = 0  # the largest packed prefill seen, bytes a chip

    def begin(self) -> None:
        """A prefill's places from here to `end`."""
        from .prefill_pack import PackSession  # (it imports this module)

        self.session = PackSession()

    def end(self):
        """-> the packed layout written (with a pack), else None (the places wrote their buffers)."""
        session, self.session = self.session, None
        if session is None:
            return None
        if self.pack is None:
            self.most = max(self.most, session.layout()[1])
            return None
        return self.pack.write(session)

    def make_pack(self) -> None:
        """The packed row for the largest prefill placed so far (before any capture)."""
        if self.pack is None and self.most:
            from .prefill_pack import Pack

            self.pack = Pack(self.mesh, self.most)

    def unpack(self, layout) -> None:
        """The packed fields to their buffers (a device op: inside the capture, its first op)."""
        if layout:
            self.pack.unpack(layout)

    def place(self, name: str, t: torch.Tensor, dims, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        buf = self.buffers.get(name)
        if self.session is not None and buf is not None:
            self.session.add(name, buf, t, dims, dtype, layout)
            if self.pack is not None:  # (staged: `end` writes it with the others)
                self.written.pop(name, None)
                return buf
        if buf is None:
            buf = self.buffers[name] = put(self.mesh, t, dims, dtype, layout)
            if self.session is not None:
                self.session.add(name, buf, t, dims, dtype, layout)
        else:
            last = self.written.get(name)
            if (
                FIXED_SKIP_SAME
                and last is not None
                and last[1:] == (dims, dtype, layout)
                and last[0].dtype == t.dtype
                and last[0].shape == t.shape
                and torch.equal(last[0], t)
            ):
                return buf
            host = ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mapper(self.mesh, dims))
            ttnn.copy_host_to_device_tensor(host, buf)
        if FIXED_SKIP_SAME:
            self.written[name] = (t.detach().clone(), dims, dtype, layout)
        return buf

    def deallocate(self) -> None:
        for buf in self.buffers.values():
            ttnn.deallocate(buf)
        self.buffers.clear()
        self.written.clear()
        if self.pack is not None:
            self.pack.deallocate()
            self.pack = None


def eager_place(mesh):
    """`FixedInputs.place`'s signature over `put`: a new device tensor per call (the caller deallocates it)."""
    return lambda name, t, dims, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT: put(mesh, t, dims, dtype, layout)


# The weight upload's host threads (GLM53_UPLOAD_THREADS): one chip's shard built and converted (BFP8 packed) by each;
# the whole-model load is bound by this host work, not by the disk.  The packing already runs on every core inside
# ttnn: two MoE layers' experts took 6.0 / 5.7 / 6.2 / 6.2 s a layer with 8 / 16 / 32 / 48 threads
UPLOAD_THREADS = int(os.environ.get("GLM53_UPLOAD_THREADS", "16"))
# Each distinct shard built and converted once (GLM53_UPLOAD_ONCE, default 1; 0: the form before, for an A/B): a caller
# whose shards repeat over the mesh rows or columns says so (`upload_shards`' `by`; KDA's projections and conv taps and
# DSA's absorbed heads depend on the column only: one host tensor a column, its bytes written to the column's 8 chips,
# where 32 were built and BFP8-packed), and a caller that knows its weights are real says so (`meta=False`: the first
# shard was built on the main thread only to see whether it was a meta tensor, the pool idle meanwhile, and built again
# in the pool, ~40 s of the load).  The callers' shard functions likewise read each tensor once (KDA's channel indices
# made once, a MoE layer's expert views once).  The same bytes on every chip
UPLOAD_ONCE = os.environ.get("GLM53_UPLOAD_ONCE", "1") == "1"
# The host cut of a big shard (the MoE layers' experts, 151 MB bf16 each, 96 a layer) into a reused host buffer
# (GLM53_UPLOAD_SCRATCH, default 1; 0: a fresh tensor each, as before): `upload_shards(.., out_shape=)` hands the shard
# function a free buffer of that many bf16 values (`out=`), back to the free list once the shard is converted (the
# conversion to a block-float dtype copies it: the converted tensor keeps nothing of it).  A fresh tensor each time
# meant fresh pages, faulted in and zeroed by the kernel: a layer's cut took 2.5 s with them, 1.0 s into reused buffers
# (16 workers, page-cached).  `drop_scratch` frees them (the end of the model's load).
# The same bytes (torch.cat into a given tensor)
UPLOAD_SCRATCH = os.environ.get("GLM53_UPLOAD_SCRATCH", "1") == "1"
_scratch: dict = {}  # numel -> free 1D bf16 tensors
_scratch_lock = threading.Lock()


def _scratch_get(shape: tuple) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    with _scratch_lock:
        free = _scratch.setdefault(n, [])
        buf = free.pop() if free else None
    return (buf if buf is not None else torch.empty(n, dtype=torch.bfloat16)).view(shape)


def _scratch_put(buf: torch.Tensor) -> None:
    flat = buf.view(-1)
    with _scratch_lock:
        _scratch.setdefault(flat.numel(), []).append(flat)


def drop_scratch() -> None:
    """The upload's reused host buffers freed (`UPLOAD_SCRATCH`)."""
    with _scratch_lock:
        _scratch.clear()


def upload_shards(
    mesh,
    shard_fn,
    dtype=ttnn.bfloat16,
    threads: int | None = None,
    *,
    by: str = "rc",
    meta: bool | None = None,
    out_shape: tuple | None = None,
):
    """A mesh tensor from one host tensor per chip, `shard_fn(row, col)` -> torch tensor, built on a thread pool.

    Much faster than one big tensor mapped by ttnn (4.8 GB: 1.6 s against 10 s): no host-wide stack, and the per-chip
    conversions run in parallel.  bf16 goes up row-major and is tiled on the device; other dtypes (BFP8) are packed on
    the host (the packer the CPU tools replay).

    by: the coordinates a shard depends on, "rc" (each chip its own), "c" (the mesh column only), "r" (the row only)
    or "" (one for every chip); each distinct shard is built and converted once (UPLOAD_ONCE) and its host tensor given
    to every chip it maps to.  meta: the weights are `ShapeOnlyCheckpoint`'s meta tensors (True: random values made on
    the device), real (False), or not known (None: the first shard built tells, and is kept for its chips).  out_shape:
    the shard's shape, the shard function taking `out=` (a bf16 buffer of that shape to cut into, reused: UPLOAD_SCRATCH;
    for block-float dtypes only, whose conversion copies it)
    """
    from concurrent.futures import ThreadPoolExecutor

    if not UPLOAD_ONCE:  # (the form before: every chip's shard built, the first twice)
        by, meta = "rc", None
    first = None
    if meta is not False:
        first = shard_fn(0, 0)
        if first.is_meta:
            return synthetic(mesh, first.shape, dtype)
        if not UPLOAD_ONCE:
            first = None
    host_threads()
    layout = ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.bfloat16 else ttnn.TILE_LAYOUT

    def key(r: int, c: int) -> tuple:
        return (r if "r" in by else 0, c if "c" in by else 0)

    scratch = UPLOAD_SCRATCH and out_shape is not None and dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)

    def convert(t):  # (a shard function may give the host tensor itself: moe.PACK_SLICES)
        return t if isinstance(t, ttnn.Tensor) else ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout)

    def one(k):
        if first is not None and k == (0, 0):
            return convert(first)
        if not scratch:
            return convert(shard_fn(*k))
        buf = _scratch_get(tuple(out_shape))
        try:
            return convert(shard_fn(*k, out=buf))
        finally:
            _scratch_put(buf)

    coords = [(r, c) for r in range(ROWS) for c in range(COLS)]
    keys = list(dict.fromkeys(key(*rc) for rc in coords))
    with ThreadPoolExecutor(min(threads or UPLOAD_THREADS, len(keys))) as pool:
        built = dict(zip(keys, pool.map(one, keys)))
    shards = [built[key(*rc)] for rc in coords]
    dev = ttnn.to_device(ttnn.from_host_shards(shards, mesh.shape), mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if layout == ttnn.TILE_LAYOUT:
        return dev
    tiled = ttnn.to_layout(dev, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(dev)
    return tiled


def tp_weight(mesh, w: torch.Tensor, split: str, dtype=ttnn.bfloat16):
    """A torch Linear weight [out, in] as a ttnn [in, out] weight split over the row's chips.

    split="out" shards the output features (column parallel), split="in" the input features (row parallel); the
    weight is the same on every row.
    """
    wt = w.T.contiguous()
    dim = {"out": 1, "in": 0}[split]
    return put(mesh, wt, dims=(None, dim), dtype=dtype)


def gather_rows(mesh, t, *, tp_dim: int | None = None) -> torch.Tensor:
    """Back to torch: concatenate the rows on dim 0 (users), and the row's chips on `tp_dim` (or take chip 0)."""
    if tp_dim is not None:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MESH_SHAPE, dims=(0, tp_dim)))
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MESH_SHAPE, dims=(0, -1)))
    width = full.shape[-1] // COLS
    return full[..., :width]
