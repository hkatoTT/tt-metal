"""The decode matmul op (`ttnn.experimental.deepseek_prefill.decode_matmul`) on the decode step's weight matmuls
([32, K] @ [K, N], one tile row of tokens), against the model's matmul today (`common.decode_linear`; the router's own
1D config for the router) and torch.

Per shape: the time and weight GB/s of today's matmul and of the op (its own split, and any others listed), the op's
PCC and largest difference against torch (x rounded to bf16 @ the dequantized weights, fp32) and against today's
matmul.  Every chip of the mesh runs the same op on the same random values; every chip's result is checked.

The time: with the device profiler on (TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
TTNN_OP_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_DIR=<dir>), the device kernel time of REPS eager
runs (median over the runs of the slowest chip, and the median over all chips and runs); else trace-timed
(`test_matmul_bench.timed`: the host's time per op of 10 in a captured trace, ~10 us of dispatch included).

One pytest node per shape and config (a hang or failure costs one of them; each node opens the mesh, ~10 s):
    GLM53_DM_SHAPES  = name:K:N[:bf16w][:fp32out],...  (default: the 9 decode shapes below)
    GLM53_DM_CONFIGS = base,k_split:n_per_core[:d1|:d2][:aA][:cC],...  (default base,0:0: today's matmul and the
                       op's own choice; n_per_core 0: the op's for that k_split; a split that does not fit is skipped;
                       d1 / d2: timing only, without weight reads / without matmuls: wrong values, not checked; aA / cC:
                       A weight chunks in flight / C tiles a chunk; n2: the weight stream on both NoCs, NoC 1 once x's
                       multicast is done (the default), n1: NoC 0 only)
    GLM53_DM_CASES   = key/config,...  (instead of every shape x config; keys: router gateup down fg kda_o x_proj
                       a_proj q_b dsa_o)
Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
"""

from __future__ import annotations

import csv
import os
import statistics
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

# the weights' precision (bfp8, or bfp4: the low-precision patterns)
WDT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("GLM53_TEST_WDTYPE", "bfp8")]

from models.demos.blackhole.glm53_flash.tests.test_matmul_bench import timed  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS,
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    decode_linear,
    mapper,
)

# (name, K, N, weights bf16 (else BFP8), output fp32 (else bf16)): one chip's share of each decode matmul
SHAPES = [
    ("MoE router", 4096, 288, True, True),
    ("MoE shared gate|up", 4096, 1024, False, False),
    ("MoE shared down", 512, 4096, False, False),
    ("KDA fg", 256, 4096, False, False),
    ("KDA o_proj", 2048, 4096, False, False),
    ("KDA x_proj", 4096, 6432, False, False),
    ("DSA a_proj", 4096, 2048, False, False),
    ("DSA q_b", 1536, 4096, False, False),
    ("DSA o_proj", 4096, 4096, False, False),
]
KEYS = dict(zip(["router", "gateup", "down", "fg", "kda_o", "x_proj", "a_proj", "q_b", "dsa_o"], SHAPES))
PCC_MIN = 0.9999
REPS = 5
PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"


def shapes():
    spec = os.environ.get("GLM53_DM_SHAPES")
    if not spec:
        return SHAPES
    out = []
    for item in spec.split(","):
        f = item.split(":")
        out.append((f[0], int(f[1]), int(f[2]), "bf16w" in f[3:], "fp32out" in f[3:]))
    return out


def cases():
    spec = os.environ.get("GLM53_DM_CASES")
    if spec:
        return [(KEYS[c.split("/")[0]], c.split("/")[1]) for c in spec.split(",")]
    return [(s, c) for s in shapes() for c in os.environ.get("GLM53_DM_CONFIGS", "base,0:0").split(",")]


def router_config(n_tiles: int):
    """The model's router matmul config (`TtMoe._router_config`, one tile row): each output tile on its own core,
    K in blocks of 16 tiles, in0 multicast."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(n_tiles, 1),
        in0_block_w=16,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def per_chip(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


def first(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


class DeviceTimer:
    """The device kernel time of an op from the profiler's report (one row per program and chip, appended at every
    ReadDeviceProfiler)."""

    KERNEL = "DEVICE KERNEL DURATION [ns]"

    def __init__(self, mesh):
        self.mesh = mesh
        self.report = Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs" / "cpp_device_perf_report.csv"

    def _rows(self) -> list[dict]:
        if not self.report.exists():
            return []
        with open(self.report) as f:
            return list(csv.DictReader(f))

    def __call__(self, fn) -> tuple[float, float]:
        """(median over the runs of the slowest chip's time, median over all chips and runs), seconds."""
        ttnn.deallocate(fn())
        ttnn.synchronize_device(self.mesh)
        ttnn.ReadDeviceProfiler(self.mesh)
        before = len(self._rows())
        outs = [fn() for _ in range(REPS)]
        ttnn.synchronize_device(self.mesh)
        ttnn.ReadDeviceProfiler(self.mesh)
        rows = self._rows()[before:]
        for o in outs:
            ttnn.deallocate(o)
        calls: dict[str, list[float]] = {}
        for r in rows:
            if r.get(self.KERNEL):
                calls.setdefault(r["GLOBAL CALL COUNT"], []).append(float(r[self.KERNEL]) * 1e-9)
        slowest = [max(v) for v in calls.values()]
        every = [t for v in calls.values() for t in v]
        return statistics.median(slowest), statistics.median(every)


ROWS_X = int(os.environ.get("GLM53_DM_ROWS", "32"))  # x's rows (64: two tile rows, 32 users' verify step)


def make_inputs(mesh, k, n, w_bf16):
    torch.manual_seed(0)
    rep = mapper(mesh, (None, None))
    x_host = torch.randn(1, 1, ROWS_X, k)
    w_host = torch.randn(1, 1, k, n) * (1.0 / k**0.5)
    x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    wdtype = ttnn.bfloat16 if w_bf16 else WDT
    w = ttnn.from_torch(w_host, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    return x, w, x_host.bfloat16().float() @ first(w)


def set_env(name, value):
    if value:
        os.environ[name] = str(value)
    else:
        os.environ.pop(name, None)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("shape, config", cases(), ids=[f"{s[0].replace(' ', '_')}-{c}" for s, c in cases()])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_decode_matmul(mesh_device, shape, config):
    name, k, n, w_bf16, out_fp32 = shape
    grid = mesh_device.compute_with_storage_grid_size()
    banks = mesh_device.dram_grid_size().x
    kt, nt = k // 32, n // 32
    odtype = ttnn.float32 if out_fp32 else ttnn.bfloat16
    wbytes = 2048 if w_bf16 else 1088
    mb = kt * nt * wbytes / 1e6
    device_timer = DeviceTimer(mesh_device) if PROFILING else None

    def time_of(fn) -> tuple[float, float]:
        if device_timer:
            return device_timer(fn)
        sec = timed(mesh_device, fn)
        return sec, sec

    head = (
        f"[{ROWS_X}, {k}] @ [{k}, {n}] {name} ({'bf16' if w_bf16 else 'bfp8'} weights, {mb:.2f} MB, "
        f"{'fp32' if out_fp32 else 'bf16'} out; grid {grid.x}x{grid.y}, {banks} banks, "
        f"{'device profiler' if PROFILING else 'trace'} timing)"
    )
    x, w, want = make_inputs(mesh_device, k, n, w_bf16)

    if config == "base":
        if name == "MoE router":
            label = "router config (1D, 9 cores)"

            def base():
                return ttnn.matmul(
                    x, w, dtype=odtype, compute_kernel_config=HIFI4_FP32, program_config=router_config(nt)
                )

        else:
            label = "decode_linear"

            def base():
                return decode_linear(mesh_device, x, w, dtype=odtype, compute_kernel_config=HIFI4_FP32)

        p_got = pcc(first(base()), want)
        sec, sec_med = time_of(base)
        print(
            f"\n{head}\n  RESULT {name} | {label:34s} {sec * 1e6:6.1f} us ({sec_med * 1e6:5.1f}) "
            f"{mb / 1e3 / sec:5.0f} GB/s  pcc {p_got:.6f}",
            flush=True,
        )
        return

    parts = config.split(":")
    ks, nc = int(parts[0]), int(parts[1])
    flags = {p[0]: int(p[1:]) for p in parts[2:]}  # d: debug mode, a: chunks in flight, c: tiles a chunk
    debug = flags.get("d", 0)
    set_env("DECODE_MATMUL_AHEAD", flags.get("a", 0))
    set_env("DECODE_MATMUL_CHUNK", flags.get("c", 0))
    set_env("DECODE_MATMUL_NOCS", flags.get("n", 0))
    plan = ttnn.experimental.deepseek_prefill.decode_matmul_plan
    op = ttnn.experimental.deepseek_prefill.decode_matmul
    set_env("DECODE_MATMUL_NC", nc)
    try:
        try:
            p = plan(kt, nt, grid.x, grid.y, wbytes, ks, banks)
        except Exception:  # noqa: BLE001  (the split does not fit the grid)
            pytest.skip(f"{config} does not fit the grid")

        def run():
            return op(x, w, output_dtype=odtype, k_split=ks)

        label = f"op {config} -> k {p[0]} x {p[2]} blk of {p[1]} ({p[3]}x{p[4]})"
        if debug:
            set_env("DECODE_MATMUL_DEBUG", debug)
            sec, sec_med = time_of(run)
            what = {1: "no weight reads", 2: "no matmuls"}.get(debug, str(debug))
            print(
                f"\n{head}\n  RESULT {name} | {label + ' (' + what + ')':34s} {sec * 1e6:6.1f} us "
                f"({sec_med * 1e6:5.1f})",
                flush=True,
            )
            return
        # the reference: decode_linear with the op's defaults (this config's switches off for it)
        saved = {v: os.environ.get(v) for v in os.environ if v.startswith("DECODE_MATMUL_")}
        for v in saved:
            os.environ.pop(v)
        ref = first(decode_linear(mesh_device, x, w, dtype=odtype, compute_kernel_config=HIFI4_FP32))
        os.environ.update(saved)
        got = per_chip(run())
        p_got = pcc(got[0], want)
        err = (got[0] - want).abs().max().item()
        vs_base = (got[0] - ref).abs()
        same = all(torch.equal(g, got[0]) for g in got[1:])
        sec, sec_med = time_of(run)
        print(
            f"\n{head}\n  RESULT {name} | {label:34s} {sec * 1e6:6.1f} us ({sec_med * 1e6:5.1f}) "
            f"{mb / 1e3 / sec:5.0f} GB/s  pcc {p_got:.6f}  max|d| torch {err:.3g}, decode_linear "
            f"{vs_base.max().item():.3g} ({(vs_base > 0).float().mean().item() * 100:.2f} % differ)  chips same {same}",
            flush=True,
        )
        assert p_got >= PCC_MIN and same, f"pcc {p_got:.6f}, chips same {same}"
    finally:
        for v in ("NC", "DEBUG", "AHEAD", "CHUNK", "NOCS"):
            set_env("DECODE_MATMUL_" + v, 0)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_decode_matmul_one_row(mesh_device):
    """One row of tokens (M = 1) through the op's padding."""
    rep = mapper(mesh_device, (None, None))
    x1 = torch.randn(1, 1, 1, 256)
    w1 = torch.randn(1, 1, 256, 4096) / 16
    xt = ttnn.from_torch(x1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    wt = ttnn.from_torch(w1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    got = per_chip(ttnn.experimental.deepseek_prefill.decode_matmul(xt, wt))[0]
    p1 = pcc(got, x1.bfloat16().float() @ first(wt))
    print(f"\nM = 1, [1, 256] @ [256, 4096]: shape {list(got.shape)}, pcc {p1:.6f}", flush=True)
    assert p1 >= PCC_MIN and list(got.shape) == [1, 1, 1, 4096]


# the model's shapes that go to the op (common.decode_linear with GLM53_DECODE_MATMUL=1: 6,000 weight tiles or more),
# in the order a KDA and a DSA layer call them
REPEAT_SHAPES = ["x_proj", "kda_o", "a_proj", "q_b", "dsa_o"]
REPEATS = int(os.environ.get("GLM53_DM_REPEATS", "200"))
REPEAT_ROUNDS = 8  # rounds a captured trace


@pytest.mark.timeout(300)
@pytest.mark.parametrize("rows", [32, 2])
@pytest.mark.parametrize("mode", ["eager", "trace"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_decode_matmul_repeat(mesh_device, mode, rows):
    """The op's determinism: the same x and w through the op GLM53_DM_REPEATS (200) times, the 5 model shapes
    alternating as the model calls them, eagerly or replayed from a captured trace (REPEAT_ROUNDS rounds a trace), every
    output compared bitwise on the device (ttnn.ne) to the first eager output of its shape on the same chip, counted
    per element; then the counts read from all 32 chips.  Every count must be 0."""
    op = ttnn.experimental.deepseek_prefill.decode_matmul
    rep = mapper(mesh_device, (None, None))
    torch.manual_seed(1)
    cases = []
    for key in REPEAT_SHAPES:
        name, k, n, w_bf16, out_fp32 = KEYS[key]
        x = ttnn.from_torch(
            torch.randn(1, 1, rows, k),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        w = ttnn.from_torch(
            torch.randn(1, 1, k, n) / k**0.5,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        ref = op(x, w)
        acc = ttnn.zeros_like(ref)
        cases.append((name, x, w, ref, acc))

    def round_():
        kept = []
        for name, x, w, ref, acc in cases:
            out = op(x, w)
            diff = ttnn.ne(out, ref)
            ttnn.add(acc, diff, output_tensor=acc)
            kept += [out, diff]
        return kept

    if mode == "eager":
        for _ in range(REPEATS):
            for t in round_():
                ttnn.deallocate(t)
    else:
        for t in round_():  # once eagerly before the capture (programs and constants made outside the trace)
            ttnn.deallocate(t)
        for _, _, _, _, acc in cases:  # (that round counted too: the accumulators start again from zero)
            ttnn.multiply(acc, 0.0, output_tensor=acc)
        ttnn.synchronize_device(mesh_device)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        kept = [t for _ in range(REPEAT_ROUNDS) for t in round_()]
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        for _ in range(REPEATS // REPEAT_ROUNDS):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ttnn.release_trace(mesh_device, trace)
        for t in kept:
            ttnn.deallocate(t)
    calls = REPEATS if mode == "eager" else (REPEATS // REPEAT_ROUNDS) * REPEAT_ROUNDS
    bad = []
    for name, x, w, ref, acc in cases:
        counts = per_chip(acc)
        per = [int((c > 0).sum().item()) for c in counts]
        worst = max(float(c.max().item()) for c in counts)
        print(
            f"  REPEAT {mode} M={rows} {name} (DECODE_MATMUL_NOCS={os.environ.get('DECODE_MATMUL_NOCS', '1')}): "
            f"{calls} calls, elements differing from the first: "
            f"{sum(per)} over 32 chips (chips with any: {sum(1 for p in per if p)}; most calls on one element "
            f"{worst:.0f})",
            flush=True,
        )
        if sum(per):
            bad.append(name)
    assert not bad, f"not deterministic: {bad}"


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_decode_matmul_model(mesh_device):
    """The decode step (traced) with decode_matmul's settings in turn, GLM53_DM_AB_REPEATS times each in rotation:
    GLM53_DM_AB_VARIANTS = name:VAR=value,VAR=value;name:...  (the op reads its DECODE_MATMUL_* environment at each
    call, so each capture takes the variant set before it); GLM53_DM_AB_USERS (1,4,16) tokens a mesh row,
    GLM53_DM_AB_FORMS (plain,verify), GLM53_DM_AB_STEPS (12), GLM53_SYNTHETIC_WEIGHTS=1 for the times;
    GLM53_DM_AB_WRAP: the positions modulo this (even; a stress of many steps in a short cache)."""
    import time

    from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint
    from models.demos.blackhole.glm53_flash.ttnn.common import ROWS
    from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model

    variants = []
    for item in os.environ.get("GLM53_DM_AB_VARIANTS", "n1:DECODE_MATMUL_NOCS=1;n2:").split(";"):
        name, _, envs = item.partition(":")
        variants.append((name, [e.split("=") for e in envs.split(",") if e]))
    names = {v for e in variants for v, _ in e[1]}
    forms = os.environ.get("GLM53_DM_AB_FORMS", "plain,verify").split(",")
    steps = int(os.environ.get("GLM53_DM_AB_STEPS", "12"))
    wrap = int(os.environ.get("GLM53_DM_AB_WRAP", "0")) or 1 << 30
    repeats = int(os.environ.get("GLM53_DM_AB_REPEATS", "3"))
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    model = TtGlm53Model(
        mesh_device,
        ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(),
        max_seq=int(os.environ.get("GLM53_TRACE_MAX_SEQ", "512")),
        mtp="verify" in forms,
    )
    vocab = model.config.vocab_size
    med = lambda v: sorted(v[2:])[len(v[2:]) // 2]  # noqa: E731
    for u in [int(x) for x in os.environ.get("GLM53_DM_AB_USERS", "1,4,16").split(",")]:
        for form in forms:
            verify = form == "verify"
            got = {name: [] for name, _ in variants}
            sig = {}
            for _ in range(repeats):
                for name, envs in variants:
                    for v in names:
                        os.environ.pop(v, None)
                    for v, value in envs:
                        os.environ[v] = value
                    g = torch.Generator().manual_seed(4000 + u)
                    if verify:
                        states = model.empty_state(u, spec=1)
                        io = model.alloc_spec_io(u, drafts=1)
                        keep = torch.full((ROWS, u), 1)

                        def write(t):
                            toks = torch.randint(0, vocab, (ROWS, 2 * u), generator=g)
                            p0 = (2 * t) % wrap
                            pos = torch.cat([torch.full((ROWS, u), p0), torch.full((ROWS, u), p0 + 1)], dim=1)
                            model.write_spec_inputs(io, toks, pos, keep)

                        run = lambda: model.forward_spec(io, states)  # noqa: E731
                    else:
                        states = model.empty_state(u)
                        io = model.alloc_io(u)

                        def write(t):
                            toks = torch.randint(0, vocab, (ROWS, u), generator=g)
                            model.write_inputs(io, toks, torch.full((ROWS, u), t % wrap))

                        run = lambda: model.forward(io, states, greedy=True)  # noqa: E731
                    write(0)
                    first_out = run()
                    for x in first_out if isinstance(first_out, tuple) else (first_out,):
                        ttnn.deallocate(x)
                    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                    outs = run()
                    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                    times = []
                    for t in range(1, steps + 1):
                        a = time.perf_counter()
                        write(t)
                        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh_device)
                        times.append(time.perf_counter() - a)
                    got[name].append(med(times))
                    # the last step's outputs (chip 0), the same for every variant of a round
                    ttnn.synchronize_device(mesh_device)
                    host = [
                        ttnn.to_torch(ttnn.get_device_tensors(x)[0])
                        for x in (outs if isinstance(outs, tuple) else (outs,))
                    ]
                    sig.setdefault(len(got[name]), (name, host))
                    first_name_r, first_host = sig[len(got[name])]
                    assert all(torch.equal(a, b) for a, b in zip(host, first_host)), f"{name} != {first_name_r}"
                    ttnn.release_trace(mesh_device, trace)
                    for x in outs if isinstance(outs, tuple) else (outs,):
                        ttnn.deallocate(x)
                    for st in states:
                        st.deallocate()
            for v in names:
                os.environ.pop(v, None)
            medians = {n: sorted(t)[(len(t) - 1) // 2] * 1e3 for n, t in got.items()}
            first_name = variants[0][0]
            line = ", ".join(
                f"{n} {m:.2f} ({m - medians[first_name]:+.2f}; {[round(x * 1e3, 2) for x in got[n]]})"
                for n, m in medians.items()
            )
            print(f"AB {form}, {ROWS * u} users: {line}", flush=True)
