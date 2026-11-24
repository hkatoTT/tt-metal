# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DSA decode's small ops after a_proj without the moves (the model's GLM53_DSA_SMALL_OPS: `TtDsa._project_small`,
`_out_proj(heads_in)`): decode_matmul's page layouts (out_heads, x_heads, split_columns, x in L1) against the ops they
replace, bit for bit on every chip of the mesh, at B tokens a mesh row (GLM53_SMALLOPS_ROWS, 8,32,64: one / one / two
tile rows), random values and BFP8 weights of the model's shapes.

The chain without them (mode 0, GLM53_DSA_SMALL_OPS=0): a = a_proj(x); q_resid, latent = two slices of a; q = q_b(q
norm(q_resid)) (the norm width-sharded on 16 cores: move in, norm, move out); the kv norm of latent; heads =
q_heads(q); o_proj(concat(heads)).  Modes (bits as GLM53_DSA_SMALL_OPS): 1 q_b writes the heads, o_proj reads them;
2 a_proj writes q_resid into the q norm's L1 shards and latent into its own tensor, q_b reads the norm's L1 output;
3 both (the default).  Compared: the heads, the normed latent and o_proj's output (and for mode 2 the matmul's two
outputs against the slices).  Each chain's time traced (`test_matmul_bench.timed`, ~10 us of dispatch included)
unless GLM53_SMALLOPS_TIME=0 (e.g. under the watcher).

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_small_ops_fuse.py -s
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tests.test_matmul_bench import timed  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    decode_linear,
    mapper,
)
from models.demos.blackhole.glm53_flash.ttnn.dsa_latent import q_heads  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm, norm_weight  # noqa: E402

D, Q_RANK, KV_RANK, HEADS, DK = 4096, 1536, 512, 16, 256  # one chip's shares of a DSA layer
EPS = 1e-6
ROWS = [int(r) for r in os.environ.get("GLM53_SMALLOPS_ROWS", "8,32,64").split(",")]
TIME = os.environ.get("GLM53_SMALLOPS_TIME", "1") == "1"


def per_chip(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


def differ(a, b) -> int:
    """Elements differing between a and b over the chips (shapes must match)."""
    pa, pb = per_chip(a), per_chip(b)
    assert [list(x.shape) for x in pa] == [list(x.shape) for x in pb], (pa[0].shape, pb[0].shape)
    return sum(int((x != y).sum().item()) for x, y in zip(pa, pb))


@pytest.mark.timeout(600)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_small_ops_fuse(mesh_device, rows):
    mesh = mesh_device
    torch.manual_seed(rows)
    rep = mapper(mesh, (None, None))

    def put(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

    x = put(torch.randn(1, 1, rows, D), ttnn.bfloat16)
    w_a = put(torch.randn(1, 1, D, Q_RANK + KV_RANK) / D**0.5, ttnn.bfloat8_b)
    w_qb = put(torch.randn(1, 1, Q_RANK, HEADS * DK) / Q_RANK**0.5, ttnn.bfloat8_b)
    w_o = put(torch.randn(1, 1, HEADS * DK, D) / D**0.5, ttnn.bfloat8_b)
    q_norm = TtRmsNorm(mesh, torch.rand(Q_RANK) + 0.5, EPS)
    kv_w = norm_weight(mesh, torch.rand(KV_RANK) + 0.5)
    mm = dict(compute_kernel_config=HIFI4_FP32)
    matmul = ttnn.experimental.deepseek_prefill.decode_matmul
    shard = q_norm.sharded_configs(rows)
    assert shard is not None, "the q norm is width-sharded at these rows"
    memory, program = shard

    def chain(mode: int, keep: bool):
        """`TtDsa._project_small` + `_out_proj` (mode 0: `_project` + q_heads + the concat, GLM53_DSA_SMALL_OPS=0) ->
        (heads, normed latent, o_proj's output, extras) or o_proj's output alone (keep False: the others freed)."""
        extras = {}
        if mode & 2:
            spec = ttnn.TensorSpec(
                ttnn.Shape([1, 1, rows, KV_RANK]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
            )
            latent = ttnn.allocate_tensor_on_device(spec, mesh)
            q_resid = matmul(x, w_a, memory_config=memory, split_columns=Q_RANK, split_out=latent)
            if keep:
                extras["q_resid"] = ttnn.to_memory_config(q_resid, ttnn.DRAM_MEMORY_CONFIG)
                extras["latent"] = ttnn.clone(latent)
            normed = q_norm.on_shard(q_resid, memory, program)
            ttnn.deallocate(q_resid)
            q = matmul(normed, w_qb, out_heads=HEADS if mode & 1 else 0)
            ttnn.deallocate(normed)
        else:
            a = decode_linear(mesh, x, w_a, **mm)
            q_resid = ttnn.slice(a, [0, 0, 0, 0], [1, 1, rows, Q_RANK])
            latent = ttnn.slice(a, [0, 0, 0, Q_RANK], [1, 1, rows, Q_RANK + KV_RANK])
            ttnn.deallocate(a)
            if keep:
                extras["q_resid"] = ttnn.clone(q_resid)
                extras["latent"] = ttnn.clone(latent)
            normed = q_norm(q_resid)
            ttnn.deallocate(q_resid)
            q = matmul(normed, w_qb, out_heads=HEADS) if mode & 1 else decode_linear(mesh, normed, w_qb, **mm)
            ttnn.deallocate(normed)
        lat = ttnn.rms_norm(latent, weight=kv_w, epsilon=EPS, **mm)
        ttnn.deallocate(latent)
        if mode & 1:
            heads = q
        else:
            heads = q_heads(mesh, q, HEADS)
            ttnn.deallocate(q)
        if mode & 1:
            out = matmul(heads, w_o, x_heads=HEADS)
        else:
            flat = ttnn.experimental.nlp_concat_heads(heads)
            out = decode_linear(mesh, flat, w_o, compute_kernel_config=HIFI4_FP32)
            ttnn.deallocate(flat)
        if not keep:
            ttnn.deallocate(heads)
            ttnn.deallocate(lat)
            return out
        return heads, lat, out, extras

    print(f"\n== DSA small ops, {rows} tokens a mesh row", flush=True)
    ref = chain(0, True)
    bad = []
    for mode in (1, 2, 3):
        got = chain(mode, True)
        counts = {name: differ(g, r) for name, g, r in zip(("heads", "latent norm", "o_proj"), got[:3], ref[:3])}
        if mode & 2:
            counts.update({f"a_proj {k}": differ(got[3][k], ref[3][k]) for k in ("q_resid", "latent")})
        print(f"  SMALLOPS rows={rows} mode={mode}: elements differing over 32 chips {counts}", flush=True)
        if any(counts.values()):
            bad.append(mode)
        for t in got[:3] + tuple(got[3].values()):
            ttnn.deallocate(t)
    if TIME:
        for mode in (0, 1, 2, 3):
            sec = timed(mesh, lambda: chain(mode, False))  # noqa: B023
            print(f"  SMALLOPS-TIME rows={rows} mode={mode}: {sec * 1e6:.1f} us a chain (traced)", flush=True)
    print(f"SMALLOPS RESULT rows={rows}: {'all bit for bit' if not bad else f'DIFFERENT in modes {bad}'}", flush=True)
    assert not bad
