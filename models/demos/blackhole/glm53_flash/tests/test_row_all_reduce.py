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
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(REPS)]
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
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
