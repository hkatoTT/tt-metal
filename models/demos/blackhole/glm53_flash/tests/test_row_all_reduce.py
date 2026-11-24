"""The row all-reduce for a few tokens (`ttnn/row_all_reduce.py`: one-shot over the fabric, generic_op) against
today's (`common.tp_all_reduce`: all_reduce_async with its moves from and to DRAM) and torch.

One pytest node per case (a hang costs one), GLM53_RAR_CASES = comma list of
kind:t[:workers:links:mcast:fpu:split][:today][:dry] (default values:1,values:4; the op's configuration, else 8 / 2 /
1 / 2 / half; fpu 2 or 0; dry: kernels that do nothing, time only):
    values  eagerly, four calls (both parities of the receive buffers twice): every chip's rows against torch (the
            row's four chips added in order in fp32, rounded to bf16) and against today's; the row's chips bit for
            bit the same; the rows below t zero
    time    trace-timed (REPS calls a trace), the new op; with `today` after it, today's too
    today   trace-timed, today's only (any t: all_reduce_async, or above 32 rows the composite)
Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
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

from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    put,
    tp_all_reduce,
)
from models.demos.blackhole.glm53_flash.ttnn.row_all_reduce import RowAllReduce  # noqa: E402

REPS = int(os.environ.get("GLM53_RAR_REPS", "20"))
HIDDEN = 4096
CASES = os.environ.get("GLM53_RAR_CASES", "values:1,values:4").split(",")


def chips(t) -> list:
    return [ttnn.to_torch(d).float().reshape(-1, HIDDEN) for d in ttnn.get_device_tensors(t)]


def padded(t) -> list:
    """Every chip's whole tile rows (the padding below t too), or None where the tensor cannot give them."""
    try:
        return [d.cpu().to_torch_with_padded_shape().float().reshape(-1, HIDDEN) for d in ttnn.get_device_tensors(t)]
    except Exception as e:  # noqa: BLE001 -- only a report
        print(f"  (padding not read: {str(e).splitlines()[0][:120]})", flush=True)
        return None


def timed(mesh, fn) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    with trace_capture(mesh) as trace:
        outs = [fn() for _ in range(REPS)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return seconds


def inputs(mesh, t: int):
    g = torch.Generator().manual_seed(t)
    host = (torch.randn(ROWS, COLS, t, HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
    x = put(mesh, host, dims=(0, 1))  # chip (r, c): host[r, c] as [1, 1, t, 4096]
    acc = host[:, 0].float()
    for c in range(1, COLS):
        acc = acc + host[:, c].float()  # in the chips' order, fp32
    h = host.float()
    pairs = (h[:, 0] + h[:, 1]) + (h[:, 2] + h[:, 3])  # fpu 2's order
    return x, acc.to(torch.bfloat16).float(), pairs.to(torch.bfloat16).float()  # [ROWS, t, D]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("case", CASES, ids=[c.replace(":", "-") for c in CASES])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_row_all_reduce(mesh_device, case):
    f = case.split(":")
    kind, t = f[0], int(f[1])
    if kind == "today":
        x, _, _ = inputs(mesh_device, t)
        sec_today = timed(mesh_device, lambda: tp_all_reduce(mesh_device, x))
        print(f"\nt {t}: today's tp_all_reduce {sec_today * 1e6:7.1f} us", flush=True)
        return
    cfg = dict(zip(("workers", "links", "mcast", "fpu", "split"), [int(v) for v in f[2:] if v.isdigit()]))
    if "mcast" in cfg:
        cfg["mcast"] = bool(cfg["mcast"])
    op = RowAllReduce(mesh_device, HIDDEN, dry="dry" in f, **cfg)
    name = f"workers {op.workers}, links {op.links}, mcast {int(op.mcast)}, fpu {op.fpu}, split {op.split}" + (
        ", dry" if op.dry else ""
    )
    print(f"\n{case}: {name}; fabric max payload {op.max_payload} B", flush=True)
    x, want, want_pairs = inputs(mesh_device, t)
    if kind == "values":
        today = chips(tp_all_reduce(mesh_device, x))
        ok = True
        for call in range(4):
            out = op(x)
            got = chips(out)
            pad = padded(out)
            worst, differ, same_row, pad_max, vs_today, n_today, n_pairs = 0.0, 0, True, 0.0, 0.0, 0, 0
            for dev, g in enumerate(got):
                r, c = dev // COLS, dev % COLS
                g = g[:t]
                ref = want[r]
                worst = max(worst, ((g - ref).abs().max() / ref.abs().max()).item())
                differ += int((g != ref).sum().item())
                n_pairs += int((g != want_pairs[r]).sum().item())
                vs_today = max(vs_today, (g - today[dev][:t]).abs().max().item())
                n_today += int((g != today[dev][:t]).sum().item())
                same_row &= bool(torch.equal(g, got[r * COLS][:t]))
                if pad is not None:
                    pad_max = max(pad_max, pad[dev][t:32].abs().max().item() if t < 32 else 0.0)
            n = ROWS * COLS * t * HIDDEN
            print(
                f"t {t} ({name}) call {call}: vs torch (fp32 sum in order, bf16): max |d| / max {worst:.2e}, "
                f"{differ} of {n} differ ({n_pairs} vs (s0 + s1) + (s2 + s3)); vs today's: max |d| {vs_today:.3g}, "
                f"{n_today} differ; the row's chips the same {same_row}; padding rows max |x| {pad_max:.3g}",
                flush=True,
            )
            ok &= worst < 1e-2 and same_row and pad_max == 0.0
            ttnn.deallocate(out)
        assert ok
    else:
        sec_new = timed(mesh_device, lambda: op(x))
        print(f"t {t}: row all-reduce ({name}) {sec_new * 1e6:7.1f} us", flush=True)
        if "today" in f:
            sec_today = timed(mesh_device, lambda: tp_all_reduce(mesh_device, x))
            print(f"t {t}: today's tp_all_reduce (all_reduce_async + moves) {sec_today * 1e6:7.1f} us", flush=True)


# the launch cost of a program shape in a trace: empty kernels on `cores` cores (the row all-reduce's: 8 workers and
# a sender), with or without its parts, and a small ttnn op beside it
FLOOR_CASES = os.environ.get("GLM53_RAR_FLOOR", "1dm1,9dm1,9dm2,9dm2c,9dm2c_rt,9dm2c_rt_fab,add").split(",")


@pytest.mark.timeout(300)
@pytest.mark.parametrize("case", FLOOR_CASES)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_launch_floor(mesh_device, case):
    x, _, _ = inputs(mesh_device, 1)
    if case == "add":
        sec = timed(mesh_device, lambda: ttnn.add(x, x))
        print(f"floor {case:16s} {sec * 1e6:7.1f} us", flush=True)
        return
    n = int(case.split("dm")[0])
    dm = int(case.split("dm")[1][0])
    compute = "c" in case.split("dm")[1]
    rt = "_rt" in case
    fab = "_fab" in case
    gx = mesh_device.compute_with_storage_grid_size().x
    cores = [ttnn.CoreCoord(i, 0) for i in range(min(n, 8))] + ([ttnn.CoreCoord(gx - 1, 0)] if n > 8 else [])
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    empty = "models/demos/blackhole/glm53_flash/ttnn/kernels/row_ar/row_ar_empty.cpp"
    nodes = {
        (r, c): mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(ROWS) for c in range(COLS)
    }

    def program(r, c):
        kernels = []
        for k in range(dm):
            args = ttnn.RuntimeArgs()
            for core in cores:
                args[core.x][core.y] = list(range(15)) if rt else []
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=empty,
                    core_ranges=crs,
                    runtime_args=args,
                    config=ttnn.WriterConfigDescriptor() if k == 0 else ttnn.ReaderConfigDescriptor(),
                )
            )
        if compute:
            kernels.append(
                ttnn.KernelDescriptor(kernel_source=empty, core_ranges=crs, config=ttnn.ComputeConfigDescriptor())
            )
        p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
        if fab:
            s = cores[-1]
            for k in range(dm):
                a = p.kernels[k].runtime_args[s.x][s.y]
                for peer in ((r, (c + 1) % COLS), (r, (c - 1) % COLS)):
                    a.extend(ttnn.setup_fabric_connection(nodes[(r, c)], nodes[peer], k, p, s))
        return p

    def run():
        out = ttnn.allocate_tensor_on_device(x.spec, mesh_device)
        mp = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mp[ttnn.MeshCoordinateRange(coord, coord)] = program(r, c)
        ttnn.generic_op([x, out], mp)
        return out

    sec = timed(mesh_device, run)
    print(f"floor {case:16s} {sec * 1e6:7.1f} us", flush=True)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_row_ar_compact(mesh_device):
    """all_reduce_async as the model calls it in decode (its input already in its L1 shard, its output left in L1) with
    TT_AR_ASYNC_COMPACT 0 / 1 (read at each call, in the program's cache key: the token rows only, packed, at 8 / 16
    tokens a row) at GLM53_RAR_COMPACT_T (default 8,16) tokens a row, width 4,096.
    GLM53_RAR_COMPACT_MODE (comma list, default values,time):
        values  4 calls each way (both parities of the landing buffers twice): every chip's token rows bit for bit,
                the whole padded tiles too (the input's padding rows zero, so the default's are zero), compact's padding
                rows zero; printed, then asserted
        time    the call's time traced (REPS calls a trace), GLM53_RAR_COMPACT_ROUNDS (default 5) rounds of 0 / 1 in
                turn, medians
        zones   GLM53_RAR_COMPACT_CALLS (default 5) eager calls with TT_AR_ASYNC_COMPACT as set outside (for the
                device profiler's kernel zones)
    GLM53_RAR_AB (default TT_AR_ASYNC_COMPACT): the switch the values / time modes turn, "NAME" or "NAME=off/on"
    (default 0/1), e.g. TT_AR_ASYNC_HOST_ROUTES=0/1 (the writer's headers' routes made on the host) or 0/2
    (its check: the kernel encodes them as before and DPRINTs whether the host's bytes are the same)."""
    from models.demos.blackhole.glm53_flash.ttnn import common

    ts = [int(v) for v in os.environ.get("GLM53_RAR_COMPACT_T", "8,16").split(",") if v]
    modes = os.environ.get("GLM53_RAR_COMPACT_MODE", "values,time").split(",")
    rounds = int(os.environ.get("GLM53_RAR_COMPACT_ROUNDS", "5"))
    ab = os.environ.get("GLM53_RAR_AB", "TT_AR_ASYNC_COMPACT")
    switch, ab_values = (ab.split("=") + ["0/1"])[:2]
    off_value, on_value = ab_values.split("/")
    saved = os.environ.get(switch)
    ok = True

    def call(x):
        with common.ar_keep_l1(True):
            return common.all_reduce_async(mesh_device, x, common.TP_AXIS)

    def compact(on: bool):  # (the switch under test: GLM53_RAR_AB)
        os.environ[switch] = on_value if on else off_value

    try:
        for t in ts:
            g = torch.Generator().manual_seed(100 + t)
            host = (torch.randn(ROWS, COLS, t, HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
            x_dram = put(mesh_device, host, dims=(0, 1))
            x = ttnn.to_memory_config(x_dram, common._tp_ar_async(mesh_device, HIDDEN, common.TP_AXIS).shard_for(t))
            ttnn.deallocate(x_dram)
            if "zones" in modes:
                for _ in range(int(os.environ.get("GLM53_RAR_COMPACT_CALLS", "5"))):
                    ttnn.deallocate(call(x))
                ttnn.synchronize_device(mesh_device)
                print(f"COMPACT t {t}: zones run, {switch}={saved}", flush=True)
            if "values" in modes:
                # GLM53_RAR_COMPACT_VARY=1: another input at each call (a buffer left over from an earlier call then
                # holds other values: an op that misses a write cannot pass on stale bytes that happen to be right)
                vary = os.environ.get("GLM53_RAR_COMPACT_VARY") == "1"
                for c in range(4):
                    xc = x
                    if vary and c > 0:
                        gc_ = torch.Generator().manual_seed(1000 * c + t)
                        hc = (torch.randn(ROWS, COLS, t, HIDDEN, generator=gc_) * 0.1).to(torch.bfloat16)
                        xd = put(mesh_device, hc, dims=(0, 1))
                        xc = ttnn.to_memory_config(xd, x.memory_config())
                        ttnn.deallocate(xd)
                    outs = {}
                    for on in (False, True) if c % 2 == 0 else (True, False):  # (each way on both landing buffers)
                        compact(on)
                        y = call(xc)
                        outs[on] = (chips(y), padded(y))
                        ttnn.deallocate(y)
                    if xc is not x:
                        ttnn.deallocate(xc)
                    (a, pa), (b, pb) = outs[False], outs[True]
                    rows_same = sum(int(torch.equal(u[:t], v[:t])) for u, v in zip(a, b))
                    whole = None if pa is None or pb is None else sum(int(torch.equal(u, v)) for u, v in zip(pa, pb))
                    pad0 = None if pb is None else all(bool((v[t:32] == 0).all()) for v in pb)
                    print(
                        f"COMPACT t {t} call {c} ({switch} {off_value} / {on_value}): token rows bit for bit on "
                        f"{rows_same} of {len(a)} chips; whole padded tiles {whole} of {len(a)}; "
                        f"the on side's padding rows zero {pad0}",
                        flush=True,
                    )
                    ok &= rows_same == len(a) and whole in (None, len(a)) and pad0 in (None, True)
            if "time" in modes:
                times = {False: [], True: []}
                for _ in range(rounds):
                    for on in (False, True):
                        compact(on)
                        times[on].append(timed(mesh_device, lambda: call(x)) * 1e6)
                med = {on: sorted(v)[len(v) // 2] for on, v in times.items()}
                print(
                    f"COMPACT t {t} time ({switch}): off {med[False]:.1f} us, on {med[True]:.1f} us "
                    f"({med[True] - med[False]:+.1f}; "
                    f"rounds {', '.join(f'{v:.1f}' for v in times[False])} / {', '.join(f'{v:.1f}' for v in times[True])})",
                    flush=True,
                )
            ttnn.deallocate(x)
    finally:
        if saved is None:
            os.environ.pop(switch, None)
        else:
            os.environ[switch] = saved
    if os.environ.get("GLM53_RAR_COMPACT_PRINT_ONLY") != "1":
        assert ok


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_row_ar_bfp8(mesh_device):
    """GLM53_AR_BFP8: the two-stage row all-reduce (row_ar2.py) of an input share in BFP8_b, as the model calls it (the
    input in its L1 shard, the output left in L1), at GLM53_RAR_COMPACT_T (default 32,64) tokens a row.  The 4 shares
    added in fp32 either way, so the sums of the BFP8 input are those of its exact bf16 copy (typecast back): 4 calls
    (both landing parities twice, another input at each), every chip's output bit for bit against the bf16 copy's; then
    the time traced (REPS calls a trace), bf16 / BFP8 in turn, GLM53_RAR_COMPACT_ROUNDS (default 5) rounds.  Prints AR8
    lines; asserts nothing (a probe: exit 0)."""
    from models.demos.blackhole.glm53_flash.ttnn import common

    ts = [int(v) for v in os.environ.get("GLM53_RAR_COMPACT_T", "32,64").split(",") if v]
    rounds = int(os.environ.get("GLM53_RAR_COMPACT_ROUNDS", "5"))
    modes = os.environ.get("GLM53_RAR_COMPACT_MODE", "values,time").split(",")

    def call(x):
        with common.ar_keep_l1(True):
            return common.all_reduce_async(mesh_device, x, common.TP_AXIS)

    def shares(seed: int, t: int, shard):
        g = torch.Generator().manual_seed(seed)
        host = (torch.randn(ROWS, COLS, t, HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
        xd = put(mesh_device, host, dims=(0, 1))
        x8d = ttnn.typecast(xd, ttnn.bfloat8_b)
        xrd = ttnn.typecast(x8d, ttnn.bfloat16)
        x8, xr = ttnn.to_memory_config(x8d, shard), ttnn.to_memory_config(xrd, shard)
        for v in (xd, x8d, xrd):
            ttnn.deallocate(v)
        return x8, xr

    for t in ts:
        assert common._ar_two_stage(t), (t, "the two-stage form only (GLM53_AR_TWO_STAGE)")
        shard = common._tp_ar_async(mesh_device, HIDDEN, common.TP_AXIS).shard_for(t)
        if "values" in modes:
            for c in range(4):
                x8, xr = shares(500 * c + t, t, shard)
                pair = [(x8, "bfp8"), (xr, "bf16")] if c % 2 == 0 else [(xr, "bf16"), (x8, "bfp8")]
                got = {}
                for x, name in pair:
                    y = call(x)
                    got[name] = chips(y)
                    if c == 0:
                        print(f"AR8 t {t} call {c}: {name} input -> output dtype {y.dtype}", flush=True)
                    ttnn.deallocate(y)
                same = sum(int(torch.equal(a[:t], b[:t])) for a, b in zip(got["bfp8"], got["bf16"]))
                ref = sum(
                    int(torch.equal(got["bf16"][0][:t], b[:t])) for b in got["bf16"][:COLS]
                )  # (a row's chips alike)
                print(f"AR8 t {t} call {c}: BFP8 input = its bf16 copy's sums bit for bit on {same} of "
                      f"{len(got['bf16'])} chips (row 0's chips alike {ref} of {COLS})", flush=True)  # fmt: skip
                ttnn.deallocate(x8)
                ttnn.deallocate(xr)
        if "time" in modes:
            x8, xr = shares(9000 + t, t, shard)
            times = {"bf16": [], "bfp8": []}
            for _ in range(rounds):
                for x, name in ((xr, "bf16"), (x8, "bfp8")):
                    times[name].append(timed(mesh_device, lambda: call(x)) * 1e6)
            med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
            print(f"AR8 t {t} time (traced, {REPS} calls a trace): bf16 {med['bf16']:.1f} us, BFP8 {med['bfp8']:.1f} us "
                  f"({med['bfp8'] - med['bf16']:+.1f}; rounds {', '.join(f'{v:.1f}' for v in times['bf16'])} / "
                  f"{', '.join(f'{v:.1f}' for v in times['bfp8'])})", flush=True)  # fmt: skip
            ttnn.deallocate(x8)
            ttnn.deallocate(xr)
