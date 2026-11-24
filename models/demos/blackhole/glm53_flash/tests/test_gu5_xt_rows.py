"""The prefill MoE's gate | up (mp_gu5, `ttnn/moe_prefill.py`) with its tilized x one row deep (GLM53_GU5_XT_ROWS=1,
the default) against two rows (=2, the earlier form): the whole prefill MoE (`moe_prefill.experts`) on the
column-sliced layout with random weights and routing, at GLM53_MOE_PREFILL_TOKENS tokens a column, the outputs of all 32
chips bit for bit, twice each way (2, 1, 2, 1: a stale page would show as a difference), and the times.
"VERDICT GU5 XT PASS" when every comparison holds (the test ends normally either way).

    GLM53_MOE_PREFILL_TOKENS (512,1024,4096)
Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_gu5_xt_rows.py -s
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tests.test_moe_experts import LIMIT, D, Q, row_sharded  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import moe_prefill  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, put  # noqa: E402

K = 8


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_gu5_xt_rows(mesh_device):
    import torch

    e_all, e_col, s_col, base = 288, 72, 256, 144
    t_col = s_col // 32
    rand = lambda shape, dt: ttnn.rand(  # noqa: E731
        shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, low=-0.05, high=0.05
    )
    w1 = rand([1, 1, t_col * e_col * D, 64], ttnn.bfloat8_b)
    w2 = rand([1, 1, Q * e_col * s_col, D // Q], ttnn.bfloat8_b)
    nd1 = ttnn.to_memory_config(w1, row_sharded(mesh_device, 2, 16 * 32))
    nd2 = ttnn.to_memory_config(w2, row_sharded(mesh_device, D // Q // 32, e_col * s_col))
    ttnn.deallocate(w1)
    ttnn.deallocate(w2)
    rep = dict(dims=(None, None), dtype=ttnn.float32)
    d_bias = put(mesh_device, torch.zeros(1, 1, 1, e_all), **rep)
    bases = torch.zeros(1, 1, 1, 8, dtype=torch.int32)
    bases[..., 0] = base
    d_base = put(mesh_device, bases, dims=(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    keep = os.environ.get("GLM53_GU5_XT_ROWS")
    ok = True
    try:
        for n in [int(v) for v in os.environ.get("GLM53_MOE_PREFILL_TOKENS", "512,1024,4096").split(",") if v]:
            g = torch.Generator().manual_seed(n)
            logits = torch.randn(n, e_all, generator=g) * 2.0
            x = torch.randn(n, D, generator=g).to(torch.bfloat16)
            d_logits = put(mesh_device, logits[None, None], **rep)
            d_x = put(mesh_device, x[None, None], dims=(None, None), dtype=ttnn.bfloat16)
            w = ttnn.experimental.deepseek_prefill.moe_decode_route(d_logits, d_bias, d_base, n, K, e_col, 2.5)
            outs, times = [], []
            for rows in ("2", "1", "2", "1"):
                os.environ["GLM53_GU5_XT_ROWS"] = rows
                out = moe_prefill.experts(mesh_device, d_x, w, nd1, nd2, e_col, LIMIT)
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                for _ in range(3):
                    ttnn.deallocate(out)
                    out = moe_prefill.experts(mesh_device, d_x, w, nd1, nd2, e_col, LIMIT)
                ttnn.synchronize_device(mesh_device)
                times.append((time.perf_counter() - t0) / 3 * 1e3)
                outs.append([ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)])
                ttnn.deallocate(out)
            same = [all(torch.equal(a, b) for a, b in zip(outs[0], o)) for o in outs[1:]]
            differ_inputs = not torch.equal(outs[0][0], torch.zeros_like(outs[0][0]))
            ok &= all(same) and differ_inputs
            print(
                f"GU5 XT {n} tokens: rows 2 / 1 / 2 / 1 against the first (2), 32 chips bit for bit {same} (outputs "
                f"non-zero {differ_inputs}); host time a call {' / '.join(f'{t:.2f}' for t in times)} ms",
                flush=True,
            )
            for t in (d_logits, d_x, w):
                ttnn.deallocate(t)
    except Exception as exc:  # (the test ends normally either way: its verdict is the printed line)
        ok = False
        print(f"GU5 XT error: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if keep is None:
            os.environ.pop("GLM53_GU5_XT_ROWS", None)
        else:
            os.environ["GLM53_GU5_XT_ROWS"] = keep
    print(f"VERDICT GU5 XT {'PASS' if ok else 'FAIL'}", flush=True)
