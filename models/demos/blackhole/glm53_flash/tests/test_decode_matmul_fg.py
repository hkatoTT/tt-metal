"""decode_matmul's fg (`fg_weight` / `fg_out`: KDA's f | gate made inside x_proj's decode_matmul on the cores its blocks
leave free, GLM53_KDA_FG_FUSED) against the separate ops on the same random values: proj =
decode_matmul(x, x_proj) and fg = ttnn.linear(proj's last 256 columns, fg_proj, WEIGHT_MM), bit for bit on all 32
chips, at M = 8 / 32 / 64 rows (the verify step's at 8 / 32 / 128 users), with fg_proj block-diagonal as the model's
(f_b, g_b: zeros off the blocks) and dense; then GLM53_DMFG_REPEAT more fused calls each against the first, and both
forms traced and timed.

    GLM53_DMFG_M       the row counts (default 8,32,64)
    GLM53_DMFG_FORMS   fg_proj's forms (default block,dense)
    GLM53_DMFG_TIME=0  no timing (e.g. under the watcher)
    GLM53_DMFG_REPEAT  the fused calls compared with the first (default 20)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_decode_matmul_fg.py -s

Prints "DMFG M .. form ..: proj the same / fg the same .." lines and "DMFG ALL SAME" when every comparison held.
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
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    WEIGHT_MM,
    mapper,
)

K, N, K2, N2 = 4096, 6432, 256, 4096  # KDA's x_proj [4096, 6144 + 32 + 256] and fg_proj [256, 2 x 2048] a chip
REPS = 10


def same_bits(a, b) -> bool:
    """Every chip's tiles of the two tensors equal bit for bit."""
    xs, ys = ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)
    return len(xs) == len(ys) and all(
        torch.equal(ttnn.to_torch(p).view(torch.int16), ttnn.to_torch(q).view(torch.int16)) for p, q in zip(xs, ys)
    )


def first(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


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
def test_decode_matmul_fg(mesh_device):
    mesh = mesh_device
    op = ttnn.experimental.deepseek_prefill.decode_matmul
    rep = mapper(mesh, (None, None))
    timing = os.environ.get("GLM53_DMFG_TIME", "1") != "0"
    repeat = int(os.environ.get("GLM53_DMFG_REPEAT", "20"))
    g = torch.Generator().manual_seed(1817)
    w_host = torch.randn(1, 1, K, N, generator=g) * (1.0 / K**0.5)
    w = ttnn.from_torch(w_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    dense = torch.randn(1, 1, K2, N2, generator=g) * (1.0 / K2**0.5)
    block = torch.zeros_like(dense)  # the model's: f_b on f_a's rows, g_b on g_a's
    block[..., : K2 // 2, : N2 // 2] = dense[..., : K2 // 2, : N2 // 2]
    block[..., K2 // 2 :, N2 // 2 :] = dense[..., K2 // 2 :, N2 // 2 :]
    forms = os.environ.get("GLM53_DMFG_FORMS", "block,dense").split(",")
    fg_ws = {
        name: ttnn.from_torch(m, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        for name, m in (("block", block), ("dense", dense))
        if name in forms
    }
    ok = True
    for m in [int(v) for v in os.environ.get("GLM53_DMFG_M", "8,32,64").split(",")]:
        x_host = torch.randn(1, 1, m, K, generator=g)
        x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        spec = ttnn.TensorSpec(
            ttnn.Shape([1, 1, m, N2]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        )
        for name, fg_w in fg_ws.items():

            def today():
                proj = op(x, w)
                fg_in = ttnn.slice(proj, [0, 0, 0, N - K2], [1, 1, m, N])
                fg = ttnn.linear(fg_in, fg_w, compute_kernel_config=WEIGHT_MM)
                ttnn.deallocate(fg_in)
                return proj, fg

            def fused():
                fg = ttnn.allocate_tensor_on_device(spec, mesh)
                return op(x, w, fg_weight=fg_w, fg_out=fg), fg

            proj0, fg0 = today()
            proj1, fg1 = fused()
            p_same, f_same = same_bits(proj0, proj1), same_bits(fg0, fg1)
            want = first(proj0)[..., N - K2 :].bfloat16().float() @ first(fg_w)
            print(
                f"DMFG M {m} fg_proj {name}: proj the {'same' if p_same else 'DIFFER'}, fg the "
                f"{'same' if f_same else 'DIFFER'} on {len(ttnn.get_device_tensors(fg1))} chips; fg pcc against "
                f"torch {pcc(first(fg1), want):.6f}",
                flush=True,
            )
            if not f_same:
                d = (first(fg0) - first(fg1)).abs()
                print(f"DMFG M {m} {name}: {int((d > 0).sum())} values differ, the largest {float(d.max()):.3e}")
            ok = ok and p_same and f_same
            again = 0
            for _ in range(repeat):  # (a race would show as another value now and then)
                p2, f2 = fused()
                again += int(same_bits(p2, proj1) and same_bits(f2, fg1))
                ttnn.deallocate(p2)
                ttnn.deallocate(f2)
            if repeat:
                print(f"DMFG M {m} fg_proj {name}: {again} of {repeat} more fused calls the same", flush=True)
            ok = ok and again == repeat
            for t in (proj0, fg0, proj1, fg1):
                ttnn.deallocate(t)
            if timing and name == forms[0]:
                t_today, t_fused = timed(mesh, today), timed(mesh, fused)
                print(
                    f"DMFG TIME M {m}: decode_matmul + slice + linear {t_today * 1e6:.1f} -> fused "
                    f"{t_fused * 1e6:.1f} us ({(t_fused - t_today) * 1e6:+.1f})",
                    flush=True,
                )
        ttnn.deallocate(x)
    print(f"DMFG {'ALL SAME' if ok else 'NOT ALL SAME'}", flush=True)
