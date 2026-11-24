"""The TP4 MLP's clamped SwiGLU in one op (swiglu_split.swiglu, mlp.SWIGLU_FUSED)
alone, against today's three ops (two slices of gate | up and `ttnn.multiply(gate, up, **swiglu_activations(limit))`):

  bits:   for each shape (S tokens, q the chip's quarter: 512 the shared expert's, 3,072 the dense layers'), each
          chip's h against the three ops' bit for bit (counts of differing words printed, SW lines), and the largest
          error of both against the formula in fp32 on the host (a sanity check that the reference is the clamped
          SwiGLU);
  time:   the three ops' and the op's time from a trace (the median of 3 replays of GLM53_SW_REPS calls; the op's
          cores as rectangles);
  stress: GLM53_SW_STRESS calls of the op (a trace of GLM53_SW_REPS calls alternating two inputs, replayed), the last
          output read back every GLM53_SW_CHECK_EVERY replays against the reference of its input.

Inputs N(0, 4^2) per chip (a few percent past the limit 10 on both sides) with some +-30 and +-0.001 entries.
Ends with an assert that no word differed (the run's exit code 1 if one did).

    GLM53_SW_SHAPES      comma list of SxQ (default 1024x512,512x512,128x512,1024x3072,512x3072)
    GLM53_SW_STRESS      calls (default 0: none)
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
from models.demos.blackhole.glm53_flash.ttnn import swiglu_split  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    put,
    swiglu_activations,
)

LIMIT = 10.0
SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get(
    "GLM53_SW_SHAPES", "1024x512,512x512,128x512,1024x3072,512x3072").split(",") if s]  # fmt: skip
REPS = int(os.environ.get("GLM53_SW_REPS", "20"))
STRESS = int(os.environ.get("GLM53_SW_STRESS", "0"))
CHECK_EVERY = int(os.environ.get("GLM53_SW_CHECK_EVERY", "50"))


def inputs(s: int, q: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, COLS, s, 2 * q, generator=g) * 4
    pick = torch.rand(x.shape, generator=g)
    x = torch.where(pick < 0.01, torch.sign(x) * 30, x)
    x = torch.where((pick > 0.5) & (pick < 0.51), x * 2.5e-4, x)
    return x.to(torch.bfloat16)


def reference(gu, s: int, q: int):
    gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, s, q])
    up = ttnn.slice(gu, [0, 0, 0, q], [1, 1, s, 2 * q])
    h = ttnn.multiply(gate, up, **swiglu_activations(LIMIT))
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    return h


def host(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def differing(a, b) -> int:
    return sum(int((x.view(torch.int16) != y.view(torch.int16)).sum()) for x, y in zip(host(a), host(b)))


def traced(mesh, fn) -> float:
    fn()  # (compiled before the capture)
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for _ in range(REPS):
            ttnn.deallocate(fn())
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) / REPS)
    ttnn.release_trace(mesh, trace)
    return sorted(ts)[1]


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_swiglu_split(mesh_device):
    mesh = mesh_device
    bad = 0
    for s, q in SHAPES:
        x = inputs(s, q, 900 + s + q)
        gu = put(mesh, x, dims=(0, 1))
        ref = reference(gu, s, q)
        x0 = x[0, 0].float()  # the formula in fp32 on the host (chip 0's input)
        want = torch.nn.functional.silu(x0[:, :q].clamp(max=LIMIT)) * x0[:, q:].clamp(-LIMIT, LIMIT)
        err_ref = float((host(ref)[0].reshape(s, q).float() - want).abs().max())
        out = swiglu_split.swiglu(mesh, gu, q, LIMIT)
        n = differing(out, ref)
        bad += n
        err = float((host(out)[0].reshape(s, q).float() - want).abs().max())
        print(f"SW bits S={s} q={q} differing_words={n} of {ROWS * COLS * s * q} "
              f"max_err_vs_fp32 fused={err:.4g} ref={err_ref:.4g}", flush=True)  # fmt: skip
        ttnn.deallocate(out)
        ttnn.deallocate(ref)
        t_ref = traced(mesh, lambda: reference(gu, s, q))
        t_new = traced(mesh, lambda: swiglu_split.swiglu(mesh, gu, q, LIMIT))
        print(f"SW time S={s} q={q} three_ops={t_ref * 1e6:.1f} us fused={t_new * 1e6:.1f} us", flush=True)
        ttnn.deallocate(gu)
    if STRESS:
        s, q = SHAPES[0]
        xs = [put(mesh, inputs(s, q, 950 + i), dims=(0, 1)) for i in range(2)]
        refs = [reference(t, s, q) for t in xs]
        last = (REPS - 1) % 2
        out = None
        swiglu_split.swiglu(mesh, xs[0], q, LIMIT)  # (compiled)
        ttnn.synchronize_device(mesh)
        with trace_capture(mesh) as trace:
            for i in range(REPS):
                if out is not None:
                    ttnn.deallocate(out)
                out = swiglu_split.swiglu(mesh, xs[i % 2], q, LIMIT)
        replays, wrong, t0 = -(-STRESS // REPS), 0, time.perf_counter()
        for k in range(replays):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            if (k + 1) % CHECK_EVERY == 0 or k == replays - 1:
                ttnn.synchronize_device(mesh)
                n = differing(out, refs[last])
                wrong += n
                if n:
                    print(f"SW stress replay {k + 1}: {n} words differ", flush=True)
        ttnn.synchronize_device(mesh)
        ttnn.release_trace(mesh, trace)
        bad += wrong
        print(f"SW stress S={s} q={q} calls={replays * REPS} differing_words={wrong} "
              f"({time.perf_counter() - t0:.1f} s)", flush=True)  # fmt: skip
        for t in (*xs, *refs, out):
            ttnn.deallocate(t)
    print(f"SW done differing_words={bad}", flush=True)
    assert bad == 0, f"{bad} words differ"
