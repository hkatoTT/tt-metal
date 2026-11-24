"""`fc_ksplit.fc_ksplit` (GLM53_DFLASH2_FC_KSPLIT: the drafter's context fc with its K split over 3 cores a column)
against the drafter's own call (`common.decode_linear(.., compute_kernel_config=DRAFTER_MM, program_config=None)`:
ttnn's 1D matmul), bit for bit on every chip, and both timed in a trace: x [1, 1, R, 36,864] bf16 @ w [36,864, 1,024]
BFP8 at R = 32 / 64 (32 / 128 users K = 7 / 3) and 8.
GLM53_FC_KSPLIT_REPEAT=N: N more calls each against the reference (a stress); GLM53_FC_KSPLIT_TIME=0: no traces (e.g.
under the watcher's NoC checks); GLM53_FC_KSPLIT_ROWS: a comma list of R (default 32,64,8).  Prints "FC_KSPLIT" lines,
ends normally.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_fc_ksplit.py -s
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

from models.demos.blackhole.glm53_flash.tests._trace import study  # noqa: E402
from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    decode_linear,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.dflash2 import DRAFTER_MM  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.fc_ksplit import fc_ksplit  # noqa: E402


def chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_fc_ksplit(mesh_device):
    # (an exception printed, not raised: the run ends cleanly; a device timeout raised)
    study(lambda: _run(mesh_device), label="FC_KSPLIT STUDY FAILED (printed, not raised)")


def _run(mesh):
    repeat = int(os.environ.get("GLM53_FC_KSPLIT_REPEAT", "0"))
    rows_list = [int(v) for v in os.environ.get("GLM53_FC_KSPLIT_ROWS", "32,64,8").split(",") if v]
    g = torch.Generator().manual_seed(5)
    k, n = 36864, 1024
    w_host = torch.randn(k, n, generator=g) * 0.02
    w = put(mesh, w_host, dims=(None, None), dtype=ttnn.bfloat8_b)
    fid = DRAFTER_MM.math_fidelity
    ok = True
    for rows in rows_list:
        xh = torch.randn(1, 1, rows, k, generator=g)
        xh[0, 0, 0, :4] = torch.tensor([0.0, -0.0, 1e-30, -1e-30])
        x = put(mesh, xh, dims=(None, None), dtype=ttnn.bfloat16)
        ref = decode_linear(mesh, x, w, compute_kernel_config=DRAFTER_MM, program_config=None)
        got = fc_ksplit(mesh, x, w, fidelity=fid)
        a, b = chips(ref), chips(got)
        same = sum(1 for p, q in zip(a, b) if same_bits(p, q))
        diff = 0
        if same < len(a):
            p, q = a[0].float(), b[0].float()
            diff = int((p != q).sum())
        stress = 0
        for _ in range(repeat):
            t = fc_ksplit(mesh, x, w, fidelity=fid)
            stress += sum(0 if same_bits(p, q) else 1 for p, q in zip(a, chips(t)))
            ttnn.deallocate(t)
        times = {"ttnn": float("nan"), "op": float("nan")}
        if os.environ.get("GLM53_FC_KSPLIT_TIME", "1") != "0":
            for name, fn in (
                ("ttnn", lambda: decode_linear(mesh, x, w, compute_kernel_config=DRAFTER_MM, program_config=None)),
                ("op", lambda: fc_ksplit(mesh, x, w, fidelity=fid)),
            ):
                ttnn.deallocate(fn())
                ttnn.synchronize_device(mesh)
                with trace_capture(mesh) as trace:
                    kept = [fn() for _ in range(20)]
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                times[name] = (time.perf_counter() - t0) / 20 * 1e6
                ttnn.release_trace(mesh, trace)
                for t in kept:
                    ttnn.deallocate(t)
        print(
            f"FC_KSPLIT rows {rows}: {same} / {len(a)} chips the same as ttnn's 1D matmul"
            + (f" (chip 0: {diff} values differ)" if same < len(a) else "")
            + f"; ttnn {times['ttnn']:.1f} -> op {times['op']:.1f} us"
            + (f"; {repeat} more calls: {stress} chips differ" if repeat else ""),
            flush=True,
        )
        ok &= same == len(a) and stress == 0
        for t in (x, ref, got):
            ttnn.deallocate(t)
    print(f"FC_KSPLIT RESULT: {'all the same' if ok else 'NOT all the same'}", flush=True)
