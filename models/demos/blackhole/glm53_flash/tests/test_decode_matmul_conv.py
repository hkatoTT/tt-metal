"""decode_matmul's conv (`conv_side` / `conv_base` / `conv_e` / `conv_s` / `conv_out`: DFlash2's dynamic convolution of
a sublayer's normed input made inside its kernel projection's decode_matmul on the cores its blocks leave free,
GLM53_DFLASH2_CONV_FUSED) against the separate ops on the same random values: delta = decode_matmul(h, kp)
and the conv op (dflash2_ops.conv) of h with that delta, bit for bit on all 32 chips, at M = 8 / 32 / 64 rows (the
drafter's block rows at 8 / 32 / 128 users: blocks of 8, 8 and 4 positions), sides 0 and 1; then GLM53_DMCV_REPEAT
more fused calls each against the first, and both forms traced and timed.

    GLM53_DMCV_M       the row counts (default 8,32,64)
    GLM53_DMCV_SIDES   the conv's sides (default 0,1; the model fuses side 0)
    GLM53_DMCV_TIME=0  no timing (e.g. under the watcher)
    GLM53_DMCV_REPEAT  the fused calls compared with the first (default 20)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_decode_matmul_conv.py -s

Prints "DMCV M .. side ..: delta the same / conv the same .." lines and "DMCV ALL SAME" when every comparison held.
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
from models.demos.blackhole.glm53_flash.ttnn import dflash2_ops as ops  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, mapper  # noqa: E402

K, N = 4096, 1024  # the drafter's hidden and a sublayer's kernel projection (sides x taps x 256 groups)
BLOCK = {8: 8, 32: 8, 64: 4}  # the drafter's block at 8 / 32 / 128 users (K = 7, 7, 3)
REPS = 10


def same_bits(a, b) -> bool:
    """Every chip's tiles of the two tensors equal bit for bit."""
    xs, ys = ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)
    return len(xs) == len(ys) and all(
        torch.equal(ttnn.to_torch(p).view(torch.int16), ttnn.to_torch(q).view(torch.int16)) for p, q in zip(xs, ys)
    )


def first(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def timed(mesh, fn) -> float:
    """fn's time per call, REPS calls in a captured trace (the host's time, a few us of dispatch included)."""
    for o in fn():
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    with trace_capture(mesh) as trace:
        outs = [fn() for _ in range(REPS)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for group in outs:
        for o in group:
            ttnn.deallocate(o)
    return seconds


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_decode_matmul_conv(mesh_device):
    mesh = mesh_device
    op = ttnn.experimental.deepseek_prefill.decode_matmul
    rep = mapper(mesh, (None, None))
    timing = os.environ.get("GLM53_DMCV_TIME", "1") != "0"
    repeat = int(os.environ.get("GLM53_DMCV_REPEAT", "20"))
    sides = [int(v) for v in os.environ.get("GLM53_DMCV_SIDES", "0,1").split(",")]
    g = torch.Generator().manual_seed(1821)
    kp = ttnn.from_torch(
        torch.randn(1, 1, K, N, generator=g) * (1.0 / K**0.5),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=rep,
    )
    bases = [ops.conv_base(mesh, torch.randn(2, K, generator=g) * 0.5) for _ in range(2)]
    ok = True
    for m in [int(v) for v in os.environ.get("GLM53_DMCV_M", "8,32,64").split(",")]:
        e, s = ops.conv_constants(mesh, BLOCK[m])
        h = ttnn.from_torch(
            torch.randn(1, 1, m, K, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=rep,
        )  # fmt: skip
        spec = ttnn.TensorSpec(
            ttnn.Shape([1, 1, m, K]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        )
        for side in sides:

            def today():
                delta = op(h, kp)
                return delta, ops.conv(mesh, h, delta, bases[side], (e, s), side)

            def fused():
                hc = ttnn.allocate_tensor_on_device(spec, mesh)
                delta = op(h, kp, conv_side=side, conv_base=bases[side], conv_e=e, conv_s=s, conv_out=hc)
                return delta, hc

            d0, c0 = today()
            d1, c1 = fused()
            d_same, c_same = same_bits(d0, d1), same_bits(c0, c1)
            print(
                f"DMCV M {m} side {side}: delta the {'same' if d_same else 'DIFFER'}, conv the "
                f"{'same' if c_same else 'DIFFER'} on {len(ttnn.get_device_tensors(c1))} chips",
                flush=True,
            )
            if not c_same:
                diff = (first(c0) - first(c1)).abs()
                cols = sorted({int(c) // 32 for c in torch.nonzero(diff.reshape(m, K) > 0)[:, 1].tolist()})[:12]
                print(f"DMCV M {m} side {side}: {int((diff > 0).sum())} values differ (column tiles {cols} ..)")
            ok = ok and d_same and c_same
            again = 0
            for _ in range(repeat):  # (a race would show as another value now and then)
                d2, c2 = fused()
                again += int(same_bits(d2, d1) and same_bits(c2, c1))
                ttnn.deallocate(d2)
                ttnn.deallocate(c2)
            if repeat:
                print(f"DMCV M {m} side {side}: {again} of {repeat} more fused calls the same", flush=True)
            ok = ok and again == repeat
            for t in (d0, c0, d1, c1):
                ttnn.deallocate(t)
            if timing and side == sides[0]:
                t_today, t_fused = timed(mesh, today), timed(mesh, fused)
                print(
                    f"DMCV TIME M {m}: decode_matmul + conv {t_today * 1e6:.1f} -> fused {t_fused * 1e6:.1f} us "
                    f"({(t_fused - t_today) * 1e6:+.1f})",
                    flush=True,
                )
        ttnn.deallocate(h)
    print(f"DMCV {'ALL SAME' if ok else 'NOT ALL SAME'}", flush=True)
