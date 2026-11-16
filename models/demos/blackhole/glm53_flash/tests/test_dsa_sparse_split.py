"""The long-cache prefill's sparse attention split by queries over a row's 4 chips (each chip all 64 heads of a quarter
of the queries) instead of by heads (each chip its 16 heads, padded to 32, of all the queries): sparse_sdpa reads every
query's selected latents (2,176 rows of 1 KB) once a chip, so the K reads drop 4 times and the padded heads go away.

Checked: the 64-head call's rows of the first 16 heads against the 16-padded-to-32 call's, bit for bit (each head row is
computed on its own).  Timed (trace, replayed): both calls, and the row collectives the split needs (the absorbed
queries gathered over the row, the outputs gathered back).

    GLM53_SPLIT_CACHE = cache rows (default 32768), GLM53_SPLIT_S = queries a row (default 512)
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
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    TP_AXIS,
    put,
)

REPS = 10
SENT = 0xFFFFFFFF


def timed(mesh, fn) -> float:
    out = fn()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
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


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_sparse_split(mesh_device):
    n = int(os.environ.get("GLM53_SPLIT_CACHE", "32768"))
    s = int(os.environ.get("GLM53_SPLIT_S", "512"))
    width, d, heads = 2176, 512, 16
    fmt = getattr(ttnn.transformer, "SparseKVFormat", None) or getattr(ttnn, "SparseKVFormat")
    g = torch.Generator().manual_seed(5)
    rm = ttnn.ROW_MAJOR_LAYOUT
    cache = put(mesh_device, torch.randn(1, 1, n, d, generator=g), dims=(None, None), layout=rm)
    rows = torch.full((1, 1, s, width), SENT, dtype=torch.int64)
    for i in range(s):
        valid = width if i % 4 else 1024 + i  # most full, some shorter
        rows[0, 0, i, :valid] = torch.randperm(n, generator=g)[:valid]
    tokens = put(mesh_device, rows.to(torch.int32), dims=(None, None), dtype=ttnn.uint32, layout=rm)
    q = torch.randn(1, 64, s, d, generator=g) * 0.5
    scale = 256**-0.5
    sdpa = dict(v_dim=d, kv_format=fmt.BF16, scale=scale, k_chunk_size=128, compute_kernel_config=HIFI4_FP32)

    # today: heads 0-15 of all the queries, padded with 16 zero heads
    q32_host = torch.cat([q[:, :heads], torch.zeros(1, heads, s, d)], dim=1)
    q32 = put(mesh_device, q32_host, dims=(None, None), layout=rm)
    out32 = ttnn.transformer.sparse_sdpa(q32, cache, tokens, **sdpa)
    ref = ttnn.to_torch(ttnn.get_device_tensors(out32)[0]).float()  # [1, 32, s, d]

    # split: all 64 heads of a quarter of the queries
    quarter = s // 4
    bad = []
    q64s, toks = [], []
    for c in range(4):
        q64 = put(mesh_device, q[:, :, c * quarter : (c + 1) * quarter].contiguous(), dims=(None, None), layout=rm)
        tk = put(
            mesh_device,
            rows[:, :, c * quarter : (c + 1) * quarter].to(torch.int32).contiguous(),
            dims=(None, None),
            dtype=ttnn.uint32,
            layout=rm,
        )
        out64 = ttnn.transformer.sparse_sdpa(q64, cache, tk, **sdpa)
        got = ttnn.to_torch(ttnn.get_device_tensors(out64)[0]).float()  # [1, 64, quarter, d]
        want = ref[:, :heads, c * quarter : (c + 1) * quarter]
        equal = torch.equal(got[:, :heads], want)
        print(
            f"quarter {c}: heads 0-15 of the 64-head call vs the padded call: "
            f"{'bit for bit' if equal else f'DIFFERENT (max |d| {(got[:, :heads] - want).abs().max():g})'}",
            flush=True,
        )
        if not equal:
            bad.append(c)
        q64s.append(q64)
        toks.append(tk)
        ttnn.deallocate(out64)

    res = {
        "padded 32 heads x all queries": timed(
            mesh_device, lambda: ttnn.transformer.sparse_sdpa(q32, cache, tokens, **sdpa)
        ),
        "64 heads x a quarter": timed(
            mesh_device, lambda: ttnn.transformer.sparse_sdpa(q64s[0], cache, toks[0], **sdpa)
        ),
    }
    q16 = put(mesh_device, q[:, :heads].contiguous(), dims=(None, None), layout=rm)  # [1, 16, s, d] RM
    res["all_gather q [1,16,s,512] RM over the row (dim 1)"] = timed(
        mesh_device, lambda: ttnn.all_gather(q16, dim=1, cluster_axis=TP_AXIS)
    )
    q16t = ttnn.to_layout(q16, ttnn.TILE_LAYOUT)
    res["all_gather q [1,16,s,512] TILE over the row (dim 1)"] = timed(
        mesh_device, lambda: ttnn.all_gather(q16t, dim=1, cluster_axis=TP_AXIS)
    )
    res["all_gather out [1,64,s/4,512] RM over the row (dim 2)"] = timed(
        mesh_device, lambda: ttnn.all_gather(q64s[0], dim=2, cluster_axis=TP_AXIS)
    )
    res["untilize q [1,16,s,512]"] = timed(mesh_device, lambda: ttnn.to_layout(q16t, rm))
    for name, sec in res.items():
        print(f"  {name:56s} {sec * 1e6:9.1f} us", flush=True)
    assert not bad, bad
