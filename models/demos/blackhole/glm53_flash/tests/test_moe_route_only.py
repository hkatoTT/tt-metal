"""The decode MoE's routing by generic_op (ttnn/moe_route.py `route_only`).

test_route_only: against moe_decode_route at the decode's shapes (tokens of a mesh column: 1 .. 32 in one tile row,
    then whole tile rows to 1,024; the shared expert on its column up to 64), on the router's logits of random tokens
    and on logits with many ties: the rows bit for bit on all 32 chips, a core's rows over both RISCs and on its writer
    alone, the rows 2 a core in 16-row groups and dealt out evenly (moe_route.plan); both ops' times (trace).
test_route_only_split: the shared expert spread over the mesh columns (moe_route.SHARED_SPLIT, TtMoe._shared_mode 2):
    on each chip of column c the rows of tokens c tokens / 4 .. (c + 1) tokens / 4 - 1 as moe_decode_route writes them
    with the shared expert on every row, the others as without it, bit for bit (both RISCs, the writer alone, the
    compute's third).
test_route_only_live: a decode step's idle tokens (moe_route.LiveTokens, a random mask and all live) choose nothing:
    their rows zero, the live rows bit for bit as without the mask (both RISCs, the writer alone, the compute's third;
    the shared expert on its column and spread over the columns); the times with the mask all live and without it.
test_router_route_times: the router's matmul, route_only and the two in a row at 16 / 64 / 128 / 256 / 512 tokens of
    a mesh column, and a program that does nothing on 1 / 9 / 120 cores (trace).
test_route_only_zones: route_only alone at GLM53_ROUTE_ZONE_TOKENS tokens (eager, for the device profiler: its kernel
    zones and the device kernel time with TT_METAL_PROFILER_CPP_POST_PROCESS=1).
test_route_only_model: the decode step with the routing by route_only and by moe_decode_route (GLM53_ROUTE_AB_ATTR:
    the moe_route switch toggled, ROUTE_FAST), off / on in turn (GLM53_ROUTE_REPEATS times), traced:
    GLM53_ROUTE_USERS (1,4,16) tokens a mesh row, GLM53_ROUTE_FORMS (plain,verify), GLM53_ROUTE_STEPS (12),
    GLM53_ROUTE_LIVE (the seats with a request, the first ones of each row; 0: all),
    or an environment variable an op reads at each call in turn (GLM53_ROUTE_AB_ENV=NAME=off:on, e.g.
    MOE_ROUTE_PACK=0:1),
    GLM53_TRACE_MAX_SEQ (512); GLM53_SYNTHETIC_WEIGHTS=1 for the times.
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_moe_route_only.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn import moe_route  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.moe import TtMoe  # noqa: E402

E, K, SCALE, D = 288, 8, 2.5, 4096


def traced(mesh, fn, reps=20):
    """Seconds per call of fn() replayed from a trace (one eager call first)."""
    fn()
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    s = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    return s


class FakeMoe:
    """What route_only and the router need of a TtMoe (column layout): a random router, bias, local base."""

    def __init__(self, mesh, g):
        self.mesh = mesh
        router = (torch.randn(E, D, generator=g) * 0.02).to(torch.bfloat16)
        self.router_wt = put(mesh, router.T.contiguous()[None, None], dims=(None, None))
        bias = (torch.randn(E, generator=g) * 0.05).view(1, 1, 1, -1)
        self.router_bias = put(mesh, bias, dims=(None, None), dtype=ttnn.float32)
        base = torch.zeros(ROWS, COLS, 1, 8, dtype=torch.int32)
        base[:, :, 0, 0] = torch.arange(COLS, dtype=torch.int32) * 72
        everywhere = base.clone()
        everywhere[:, :, 0, 1] = 1  # (the reference of the split: every column adds the shared expert to every row)
        self.local_base_all = put(mesh, everywhere, dims=(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        split = base.clone()  # as TtMoe.local_base_split: column c adds it to part c of COLS of the tokens
        split[:, :, 0, 1] = 2
        split[:, :, 0, 2] = torch.arange(COLS, dtype=torch.int32)
        split[:, :, 0, 3] = COLS
        self.local_base_split = put(mesh, split, dims=(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        base[:, 3, 0, 1] = 1  # the shared expert on column 3
        self.local_base = put(mesh, base, dims=(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.k, self.scale, self.op_experts, self.column = K, SCALE, 72, True
        self._router_cfgs = {}
        self.split = False  # route_only's shared expert spread over the columns (TtMoe._shared_mode 2)

    def _shared_in_op(self, n):
        return n <= 64

    def _shared_mode(self, n, tokens):
        return 2 if self.split else (1 if self._shared_in_op(n) else 0)

    def _router_config(self, n, *args):
        return TtMoe._router_config(self, n, *args)

    def router_logits(self, x):
        return TtMoe.router_logits(self, x)

    def route(self, logits, tokens, n):
        return ttnn.experimental.deepseek_prefill.moe_decode_route(
            logits, self.router_bias, self.local_base, tokens, K, 72, SCALE, shared=self._shared_in_op(n)
        )


def chips(t):
    return ttnn.get_device_tensors(t)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_route_only(mesh_device):
    g = torch.Generator().manual_seed(17)
    moe = FakeMoe(mesh_device, g)
    cases = [(32, 1), (32, 2), (32, 4), (32, 8), (32, 16), (32, 24), (32, 32), (64, 64), (128, 128), (256, 256)]
    cases += [(512, 512), (1024, 1024)]
    rows = []
    for n, tokens in cases:
        for kind in ("router", "ties"):
            if kind == "router":
                x = torch.zeros(n, D)
                x[:tokens] = torch.randn(tokens, D, generator=g)
                dx = put(mesh_device, x[None, None], dims=(None, None))
                logits = moe.router_logits(dx)
                ttnn.deallocate(dx)
            else:  # (a few distinct values: equal choice keys, the lower expert first)
                lt = (torch.randn(n, E, generator=g) * 2).round() / 2
                logits = put(mesh_device, lt[None, None], dims=(None, None), dtype=ttnn.float32)
            w_ref = moe.route(logits, tokens, n)
            same = True
            # a core's rows over both its RISCs or on its writer alone; 2 rows a core in 16-row groups or dealt out
            # evenly over all the cores (the plans: moe_route.plan)
            # (and a third of a core's rows on its compute's unpack thread, from 3 rows a core: ROUTE_TRISC)
            plans = ((True, 2, True, True), (False, 2, True, True), (True, 0, False, True), (False, 0, False, True))
            for share, pair, groups, trisc in plans + ((True, 2, True, False),):
                moe_route.ROUTE_SHARE, moe_route.ROUTE_PAIR, moe_route.ROUTE_GROUPS = share, pair, groups
                moe_route.ROUTE_TRISC = trisc
                w = moe_route.route_only(mesh_device, moe, logits, tokens)
                same = same and all(
                    torch.equal(ttnn.to_torch(p).view(torch.int32), ttnn.to_torch(q).view(torch.int32))
                    for p, q in zip(chips(w), chips(w_ref))
                )
                ttnn.deallocate(w)
            moe_route.ROUTE_SHARE, moe_route.ROUTE_PAIR, moe_route.ROUTE_GROUPS = True, 2, True
            moe_route.ROUTE_TRISC = True
            ttnn.deallocate(w_ref)
            line = f"n {n} tokens {tokens} ({kind}): rows bit-identical on all chips {same}"
            if kind == "router":
                t_ref = traced(mesh_device, lambda: ttnn.deallocate(moe.route(logits, tokens, n))) * 1e6
                moe_route.ROUTE_SHARE = False
                fast = lambda: ttnn.deallocate(moe_route.route_only(mesh_device, moe, logits, tokens))  # noqa: E731
                t_one = traced(mesh_device, fast) * 1e6
                moe_route.ROUTE_SHARE = True
                t_new = traced(mesh_device, fast) * 1e6
                line += (
                    f"; moe_decode_route {t_ref:.1f} us, route_only {t_new:.1f} us ({t_new - t_ref:+.1f}; the writer "
                    f"alone {t_one:.1f})"
                )
                rows.append((tokens, t_ref, t_new))
            moe_route.ROUTE_SHARE = True
            ttnn.deallocate(logits)
            print(line, flush=True)
            assert same, line
    table = "; ".join(f"{t} {a:.1f} {b:.1f}" for t, a, b in rows)
    print(f"tokens, moe_decode_route us, route_only us: {table}", flush=True)
    assert all(b <= a + 1.0 for _, a, b in rows), "route_only slower than moe_decode_route somewhere"


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_route_only_split(mesh_device):
    g = torch.Generator().manual_seed(29)
    moe = FakeMoe(mesh_device, g)
    ok = True
    for n, tokens in [(128, 128), (256, 256), (256, 240), (512, 512), (1024, 1024)]:
        for kind in ("router", "ties"):
            if kind == "router":
                x = torch.zeros(n, D)
                x[:tokens] = torch.randn(tokens, D, generator=g)
                dx = put(mesh_device, x[None, None], dims=(None, None))
                logits = moe.router_logits(dx)
                ttnn.deallocate(dx)
            else:
                lt = (torch.randn(n, E, generator=g) * 2).round() / 2
                logits = put(mesh_device, lt[None, None], dims=(None, None), dtype=ttnn.float32)
            dr = ttnn.experimental.deepseek_prefill.moe_decode_route
            none = dr(logits, moe.router_bias, moe.local_base, tokens, K, 72, SCALE, shared=False)
            every = dr(logits, moe.router_bias, moe.local_base_all, tokens, K, 72, SCALE, shared=True)
            want = []
            for i, (a, b) in enumerate(zip(chips(none), chips(every))):
                c = i % COLS  # (the chips in row-major mesh order)
                lo, hi = c * tokens // COLS, (c + 1) * tokens // COLS
                ref = ttnn.to_torch(a).view(torch.int32).clone()
                ref[..., lo:hi, :] = ttnn.to_torch(b).view(torch.int32)[..., lo:hi, :]
                want.append(ref)
            moe.split = True
            same = True
            for share, trisc in ((True, True), (False, True), (True, False)):
                moe_route.ROUTE_SHARE, moe_route.ROUTE_TRISC = share, trisc
                w = moe_route.route_only(mesh_device, moe, logits, tokens)
                same = same and all(torch.equal(ttnn.to_torch(p).view(torch.int32), q) for p, q in zip(chips(w), want))
                ttnn.deallocate(w)
            moe_route.ROUTE_SHARE, moe_route.ROUTE_TRISC = True, True
            moe.split = False
            for t in (none, every, logits):
                ttnn.deallocate(t)
            print(f"split: n {n} tokens {tokens} ({kind}): rows bit-identical on all chips {same}", flush=True)
            ok = ok and same
    assert ok


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_route_only_live(mesh_device):
    g = torch.Generator().manual_seed(41)
    moe = FakeMoe(mesh_device, g)
    ok = True
    for n, tokens in [(32, 8), (32, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]:
        x = torch.zeros(n, D)
        x[:tokens] = torch.randn(tokens, D, generator=g)
        dx = put(mesh_device, x[None, None], dims=(None, None))
        logits = moe.router_logits(dx)
        ttnn.deallocate(dx)
        mask = moe_route.LiveTokens.alloc(mesh_device, tokens // ROWS)
        for split in (False, True) if n >= 128 else (False,):
            moe.split = split
            plain = moe_route.route_only(mesh_device, moe, logits, tokens)
            for kind, live in (("random", torch.rand(tokens, generator=g) < 0.6), ("all", torch.ones(tokens) > 0)):
                live[0] = True  # (some live row, whatever the draw)
                mask.write(mesh_device, live.view(ROWS, -1))
                want = []
                for p in chips(plain):
                    ref = ttnn.to_torch(p).view(torch.int32).clone()
                    ref[..., :tokens, :][..., ~live, :] = 0
                    want.append(ref)
                same = True
                for share, trisc in ((True, True), (False, True), (True, False)):
                    moe_route.ROUTE_SHARE, moe_route.ROUTE_TRISC = share, trisc
                    with moe_route.decoding(mask):
                        w = moe_route.route_only(mesh_device, moe, logits, tokens)
                    same = same and all(
                        torch.equal(ttnn.to_torch(p).view(torch.int32), q) for p, q in zip(chips(w), want)
                    )
                    ttnn.deallocate(w)
                moe_route.ROUTE_SHARE, moe_route.ROUTE_TRISC = True, True
                line = (
                    f"live: n {n} tokens {tokens} split {split} ({kind}, {int(live.sum())} live): rows as wanted {same}"
                )
                if kind == "all" and not split:

                    def masked():
                        with moe_route.decoding(mask):
                            ttnn.deallocate(moe_route.route_only(mesh_device, moe, logits, tokens))

                    t_off = traced(
                        mesh_device, lambda: ttnn.deallocate(moe_route.route_only(mesh_device, moe, logits, tokens))
                    )
                    t_on = traced(mesh_device, masked)
                    line += f"; without the mask {t_off * 1e6:.1f} us, with it (all live) {t_on * 1e6:.1f} us"
                print(line, flush=True)
                ok = ok and same
            ttnn.deallocate(plain)
        moe.split = False
        ttnn.deallocate(mask.tensor)
        ttnn.deallocate(logits)
    assert ok


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_route_only_model(mesh_device):
    from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint
    from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model

    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    forms = os.environ.get("GLM53_ROUTE_FORMS", "plain,verify").split(",")
    steps = int(os.environ.get("GLM53_ROUTE_STEPS", "12"))
    repeats = int(os.environ.get("GLM53_ROUTE_REPEATS", "3"))
    max_seq = int(os.environ.get("GLM53_TRACE_MAX_SEQ", "512"))
    attr = os.environ.get("GLM53_ROUTE_AB_ATTR", "ROUTE_FAST")  # the moe_route switch toggled
    target = moe_route
    if "." in attr:  # another module's switch of ttnn/: GLM53_ROUTE_AB_ATTR=mhc.WRITE_READ
        import importlib

        name, attr = attr.split(".", 1)
        target = importlib.import_module(f"models.demos.blackhole.glm53_flash.ttnn.{name}")
    original = getattr(target, attr, None)
    # or an environment variable read at each op call: GLM53_ROUTE_AB_ENV=NAME=off:on (e.g. MOE_ROUTE_PACK=0:1)
    env_ab = os.environ.get("GLM53_ROUTE_AB_ENV")
    if env_ab:
        env_name, values = env_ab.split("=", 1)
        env_values = dict(zip((False, True), values.split(":")))
        attr = env_name
    live_k = int(os.environ.get("GLM53_ROUTE_LIVE", "0"))  # the seats with a request a row (the first ones; 0: all)
    model = TtGlm53Model(
        mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=max_seq, mtp="verify" in forms
    )
    vocab = model.config.vocab_size
    med = lambda v: sorted(v[2:])[len(v[2:]) // 2]  # noqa: E731
    for u in [int(x) for x in os.environ.get("GLM53_ROUTE_USERS", "1,4,16").split(",")]:
        seat = torch.arange(u) < (live_k or u)
        for form in forms:
            verify = form == "verify"
            live = (seat.repeat(2) if verify else seat).expand(ROWS, -1) if live_k else None
            got = {False: [], True: []}
            for on in [False, True] * repeats:
                if env_ab:
                    os.environ[env_name] = env_values[on]
                else:
                    setattr(target, attr, on)
                g = torch.Generator().manual_seed(3000 + u)
                if verify:
                    states = model.empty_state(u, spec=1)
                    io = model.alloc_spec_io(u, drafts=1)
                    keep = torch.full((ROWS, u), 1)

                    def write(t):
                        toks = torch.randint(0, vocab, (ROWS, 2 * u), generator=g)
                        pos = torch.cat([torch.full((ROWS, u), 2 * t), torch.full((ROWS, u), 2 * t + 1)], dim=1)
                        model.write_spec_inputs(io, toks, pos, keep, live=live)

                    run = lambda: model.forward_spec(io, states)  # noqa: E731
                else:
                    states = model.empty_state(u)
                    io = model.alloc_io(u)

                    def write(t):
                        toks = torch.randint(0, vocab, (ROWS, u), generator=g)
                        model.write_inputs(io, toks, torch.full((ROWS, u), t), live=live)

                    run = lambda: model.forward(io, states, greedy=True)  # noqa: E731
                write(0)
                first = run()
                for x in first if isinstance(first, tuple) else (first,):
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
                got[on].append(med(times))
                ttnn.release_trace(mesh_device, trace)
                for x in outs if isinstance(outs, tuple) else (outs,):
                    ttnn.deallocate(x)
                for st in states:
                    st.deallocate()
            off, on_ = (sorted(got[k])[(len(got[k]) - 1) // 2] * 1e3 for k in (False, True))
            print(
                f"{form}, {ROWS * u} users ({ROWS * int(seat.sum())} live): step {off:.2f} -> {on_:.2f} ms "
                f"({on_ - off:+.2f} ms; {attr} off "
                f"{[round(v * 1e3, 2) for v in got[False]]}, on {[round(v * 1e3, 2) for v in got[True]]})",
                flush=True,
            )
    if not env_ab:
        setattr(target, attr, original)


def _empty_op(mesh, t, out, count):
    """A program that does nothing (a reader kernel on `count` cores); generic_op wants an input and an output."""
    grid = mesh.compute_with_storage_grid_size()
    cores = moe_route.ranges([ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)])
    kernel = ttnn.KernelDescriptor(
        kernel_source="models/demos/blackhole/glm53_flash/tests/kernels/empty.cpp",
        core_ranges=cores,
        compile_time_args=[],
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    return lambda: ttnn.generic_op([t, out], mesh_program)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_router_route_times(mesh_device):
    g = torch.Generator().manual_seed(23)
    moe = FakeMoe(mesh_device, g)
    reps = int(os.environ.get("GLM53_ROUTE_TIME_REPS", "5"))
    med = lambda fn: sorted(traced(mesh_device, fn) for _ in range(reps))[reps // 2] * 1e6  # noqa: E731
    small = put(mesh_device, torch.zeros(1, 1, 32, 32), dims=(None, None))
    out = put(mesh_device, torch.zeros(1, 1, 32, 32), dims=(None, None))
    for count in (1, 9, 120):
        print(f"empty program on {count} cores: {med(_empty_op(mesh_device, small, out, count)):.1f} us", flush=True)
    for n, tokens in [(32, 16), (64, 64), (128, 128), (256, 256), (512, 512)]:
        x = torch.zeros(n, D)
        x[:tokens] = torch.randn(tokens, D, generator=g)
        dx = put(mesh_device, x[None, None], dims=(None, None))
        logits = moe.router_logits(dx)

        def both():
            lg = moe.router_logits(dx)
            ttnn.deallocate(moe_route.route_only(mesh_device, moe, lg, tokens))
            ttnn.deallocate(lg)

        t_mm = med(lambda: ttnn.deallocate(moe.router_logits(dx)))
        moe_route.ROUTE_TRISC = False
        t_two = med(lambda: ttnn.deallocate(moe_route.route_only(mesh_device, moe, logits, tokens)))
        moe_route.ROUTE_TRISC = True
        t_rt = med(lambda: ttnn.deallocate(moe_route.route_only(mesh_device, moe, logits, tokens)))
        t_both = med(both)
        print(
            f"{tokens} tokens (x {n} rows): router matmul {t_mm:.1f} us, route_only {t_rt:.1f} us (the writer and "
            f"reader only {t_two:.1f}), the two in a row "
            f"{t_both:.1f} us (sum {t_mm + t_rt:.1f})",
            flush=True,
        )
        ttnn.deallocate(logits)
        ttnn.deallocate(dx)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_route_only_zones(mesh_device):
    from models.demos.blackhole.glm53_flash.tests.test_decode_matmul import DeviceTimer

    g = torch.Generator().manual_seed(29)
    moe = FakeMoe(mesh_device, g)
    tokens = int(os.environ.get("GLM53_ROUTE_ZONE_TOKENS", "256"))
    n = -(-tokens // 32) * 32
    x = torch.zeros(n, D)
    x[:tokens] = torch.randn(tokens, D, generator=g)
    dx = put(mesh_device, x[None, None], dims=(None, None))
    logits = moe.router_logits(dx)
    run = lambda: moe_route.route_only(mesh_device, moe, logits, tokens)  # noqa: E731
    if os.environ.get("TT_METAL_PROFILER_CPP_POST_PROCESS") == "1":
        slow, every = DeviceTimer(mesh_device)(run)
        print(f"route_only {tokens} tokens: device kernel {slow * 1e6:.1f} us (chips' mean {every * 1e6:.1f})")
    else:
        for _ in range(4):
            ttnn.deallocate(run())
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_router_weight_bound(mesh_device):
    """How much of the router matmul is its weight stream (2.36 MB bf16, 9 reading cores, one output column tile
    each): the same ttnn.matmul and program config with the weights in DRAM (today) and in L1 (interleaved over the
    grid: a far faster read, the same arithmetic), traced, and whether the logits are bit for bit the same.  Prints."""
    g = torch.Generator().manual_seed(23)
    moe = FakeMoe(mesh_device, g)
    reps = int(os.environ.get("GLM53_ROUTE_TIME_REPS", "5"))
    med = lambda fn: sorted(traced(mesh_device, fn) for _ in range(reps))[reps // 2] * 1e6  # noqa: E731
    w_dram = moe.router_wt
    w_l1 = ttnn.to_memory_config(w_dram, ttnn.L1_MEMORY_CONFIG)
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    for n, tokens in [(32, 16), (64, 64), (256, 256)]:
        x = torch.zeros(n, D)
        x[:tokens] = torch.randn(tokens, D, generator=g)
        dx = put(mesh_device, x[None, None], dims=(None, None))
        times, outs = {}, {}
        for name, w in (("DRAM", w_dram), ("L1", w_l1)):
            moe.router_wt = w
            outs[name] = first(moe.router_logits(dx))
            times[name] = med(lambda: ttnn.deallocate(moe.router_logits(dx)))
        moe.router_wt = w_dram
        print(
            f"{tokens} tokens (x {n} rows): router matmul, weights in DRAM {times['DRAM']:.1f} us, in L1 "
            f"{times['L1']:.1f} us; logits bit for bit {torch.equal(outs['DRAM'], outs['L1'])}",
            flush=True,
        )
        ttnn.deallocate(dx)
    ttnn.deallocate(w_l1)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_router_mm(mesh_device):
    """ttnn/router_mm.py's router matmul against ttnn.matmul with `_router_config` (moe.router_logits): bit for bit on
    random x and the FakeMoe's router (8 / 32 / 64 tokens a row, as the decode steps' H1 router), and both times from a
    trace.  Prints; fails on a difference."""
    from models.demos.blackhole.glm53_flash.ttnn.router_mm import router_matmul

    g = torch.Generator().manual_seed(29)
    moe = FakeMoe(mesh_device, g)
    reps = int(os.environ.get("GLM53_ROUTE_TIME_REPS", "5"))
    med = lambda fn: sorted(traced(mesh_device, fn) for _ in range(reps))[reps // 2] * 1e6  # noqa: E731
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    bad = []
    for n in [int(v) for v in os.environ.get("GLM53_ROUTER_MM_ROWS", "8,32,64").split(",")]:
        x = (torch.randn(n, D, generator=g)).to(torch.bfloat16)
        dx = put(mesh_device, x[None, None], dims=(None, None))
        want = moe.router_logits(dx)
        got = router_matmul(mesh_device, dx, moe.router_wt)
        a, b = first(want), first(got)
        same = torch.equal(a, b)
        if not same:
            bad.append(n)
        once = os.environ.get("GLM53_ROUTER_MM_ONCE") == "1"  # (one eager call a size: a new form's first run)
        t_ttnn = 0.0 if once else med(lambda: ttnn.deallocate(moe.router_logits(dx)))
        t_mine = 0.0 if once else med(lambda: ttnn.deallocate(router_matmul(mesh_device, dx, moe.router_wt)))
        print(
            f"router matmul {n} rows: bit for bit {same} (differing {int((a != b).sum())}, max |d| "
            f"{(a.float() - b.float()).abs().max().item():.3e}); ttnn {t_ttnn:.1f} us, router_mm {t_mine:.1f} us",
            flush=True,
        )
        for t in (want, got, dx):
            ttnn.deallocate(t)
    assert not bad, f"router_mm differs at {bad} rows"


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_router_route(mesh_device):
    """router_mm.router_route (the router matmul and route_only in one program) against moe.router_logits + route_only
    bit for bit (w), GLM53_ROUTER_ROUTE_TOKENS (default 8,32,64) tokens; then the two forms' times from a trace (unless
    GLM53_ROUTER_MM_ONCE=1: one eager call a size).  Prints; fails on a difference."""
    from models.demos.blackhole.glm53_flash.ttnn.router_mm import router_route

    g = torch.Generator().manual_seed(31)
    moe = FakeMoe(mesh_device, g)
    reps = int(os.environ.get("GLM53_ROUTE_TIME_REPS", "5"))
    med = lambda fn: sorted(traced(mesh_device, fn) for _ in range(reps))[reps // 2] * 1e6  # noqa: E731
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    once = os.environ.get("GLM53_ROUTER_MM_ONCE") == "1"
    bad = []
    for tokens in [int(v) for v in os.environ.get("GLM53_ROUTER_ROUTE_TOKENS", "8,32,64").split(",")]:
        x = (torch.randn(tokens, D, generator=g)).to(torch.bfloat16)
        dx = put(mesh_device, x[None, None], dims=(None, None))

        def today():
            lg = moe.router_logits(dx)
            w = moe_route.route_only(mesh_device, moe, lg, tokens)
            ttnn.deallocate(lg)
            return w

        a, b = today(), router_route(mesh_device, moe, dx, tokens)
        ta, tb = first(a), first(b)
        same = torch.equal(ta, tb)
        if not same:
            bad.append(tokens)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        t_a = 0.0 if once else med(lambda: ttnn.deallocate(today()))
        t_b = 0.0 if once else med(lambda: ttnn.deallocate(router_route(mesh_device, moe, dx, tokens)))
        print(
            f"router + routing {tokens} tokens: bit for bit {same} (differing {int((ta != tb).sum())}); two ops "
            f"{t_a:.1f} us, one program {t_b:.1f} us",
            flush=True,
        )
        ttnn.deallocate(dx)
    assert not bad, f"router_route differs at {bad} tokens"


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_router_route_stress(mesh_device):
    """A race stress of router_mm.router_route (the router matmul and route_only in one program, its cores told by
    writes: rm_w.cpp): per GLM53_ROUTER_ROUTE_TOKENS (default 8,32,64) tokens, GLM53_ROUTER_ROUTE_REPEAT (300) eager
    calls each against moe.router_logits + route_only bit for bit, then GLM53_ROUTER_ROUTE_REPLAYS (2,000) replays of a
    trace of 4 calls (each call's own x), the last replay's outputs against theirs.  Prints; fails on a difference."""
    from models.demos.blackhole.glm53_flash.ttnn.router_mm import router_route

    g = torch.Generator().manual_seed(37)
    moe = FakeMoe(mesh_device, g)
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    repeat = int(os.environ.get("GLM53_ROUTER_ROUTE_REPEAT", "300"))
    replays = int(os.environ.get("GLM53_ROUTER_ROUTE_REPLAYS", "2000"))
    bad = []
    for tokens in [int(v) for v in os.environ.get("GLM53_ROUTER_ROUTE_TOKENS", "8,32,64").split(",")]:
        xs = [
            put(mesh_device, torch.randn(tokens, D, generator=g).to(torch.bfloat16)[None, None], dims=(None, None))
            for _ in range(4)
        ]
        want = []
        for dx in xs:
            lg = moe.router_logits(dx)
            w = moe_route.route_only(mesh_device, moe, lg, tokens)
            want.append(first(w))
            ttnn.deallocate(lg)
            ttnn.deallocate(w)
        t0 = time.perf_counter()
        differ = 0
        for i in range(repeat):
            w = router_route(mesh_device, moe, xs[i % 4], tokens)
            differ += int(not torch.equal(first(w), want[i % 4]))
            ttnn.deallocate(w)
        dt = time.perf_counter() - t0
        outs = [router_route(mesh_device, moe, dx, tokens) for dx in xs]  # (eager once: the programs made)
        for w in outs:
            ttnn.deallocate(w)
        ttnn.synchronize_device(mesh_device)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [router_route(mesh_device, moe, dx, tokens) for dx in xs]
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        t1 = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        dt2 = time.perf_counter() - t1
        last = sum(int(not torch.equal(first(w), want[j])) for j, w in enumerate(outs))
        ttnn.release_trace(mesh_device, trace)
        for t in outs + xs:
            ttnn.deallocate(t)
        print(
            f"router + routing {tokens} tokens: {repeat} eager calls in {dt:.1f} s, {differ} differing; {replays} "
            f"replays of 4 calls in {dt2:.1f} s, the last replay's outputs differing {last} / 4",
            flush=True,
        )
        if differ or last:
            bad.append(tokens)
    assert not bad, f"router_route differs at {bad} tokens"
