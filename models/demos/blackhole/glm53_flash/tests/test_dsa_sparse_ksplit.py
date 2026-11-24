"""Decode's sparse attention with each query's keys split over cores (`dsa_indexer.sparse_split` + `sparse_combine`)
against `ttnn.transformer.sparse_sdpa` (one query a core) and torch (fp32), on every chip of the mesh (replicated
inputs, chip 0 checked): q [1, 32, S, 512] (16 heads, zeros below, as `TtDsa._sparse_attend` pads them), every
user's latents in one row-major cache with global row ids, each query's selected rows (`select_tokens`'s form: the
valid ones first, 0xFFFFFFFF after, width 2,176), the valid counts varied (a full selection, the tail's 1-3 more,
fewer than one chunk, a few chunks, under the split count).  Then both timed in a trace (replayed), with
sparse_sdpa's output conversion (to tiles, the 16 heads) against the split's (combine's tiles, the transpose).

    GLM53_KSPLIT_S = queries (default 1,2,4,8,16,32), GLM53_KSPLIT_G = splits (default 2,4,8,16),
    GLM53_KSPLIT_CACHE = cache rows (default 9728), GLM53_KSPLIT_KV = bf16 (default), fp8 (the KV cache study's FP8
    e4m3 latent cache: both ops read it as fp8, torch the latents rounded as they hold them) or a comma list
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
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import sparse_combine, sparse_split  # noqa: E402

REP = dict(dims=(None, None))
WIDTH, LATENT, SENT = 2176, 512, 0xFFFFFFFF


def sizes(name: str, default: str) -> list[int]:
    return [int(v) for v in os.environ.get(name, default).split(",") if v]


def first(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def traced(mesh, fn, reps: int = 10) -> float:
    for o in fn():
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    with trace_capture(mesh) as trace:
        outs = [fn() for _ in range(reps)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    for group in outs:
        for o in group:
            ttnn.deallocate(o)
    return seconds


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_sparse_ksplit(mesh_device):
    g = torch.Generator().manual_seed(7)
    ctx = int(os.environ.get("GLM53_KSPLIT_CACHE", "9728"))
    fmt = getattr(ttnn.transformer, "SparseKVFormat", None) or getattr(ttnn, "SparseKVFormat")
    scale = (256 + 0) ** -0.5  # the DSA layer's softmax scale (qk_head_dim 256)
    grid = mesh_device.compute_with_storage_grid_size()
    worst = 0.0
    from models.demos.blackhole.glm53_flash.tools.kv_round import round_kv
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import FP8_K_IN

    for fp8, s in [
        (kv_name == "fp8", s_)
        for kv_name in os.environ.get("GLM53_KSPLIT_KV", "bf16").split(",")
        for s_ in sizes("GLM53_KSPLIT_S", "1,2,4,8,16,32")
    ]:
        q = torch.zeros(1, 32, s, LATENT, dtype=torch.bfloat16)
        q[:, :16] = (torch.randn(1, 16, s, LATENT, generator=g) * 0.3).to(torch.bfloat16)
        kv = (torch.randn(1, 1, ctx, LATENT, generator=g) * 0.5).to(torch.bfloat16)  # (one cache: rows by id)
        if fp8:  # (values fp8 holds exactly: the host's conversion leaves them as they are)
            kv = round_kv(kv, "fp8a")
        counts = [2048, 2051, 2049, 100, 128, 129, 300, 2050][:s] + [2048 + (u % 4) for u in range(max(0, s - 8))]
        idx = torch.full((1, 1, s, WIDTH), SENT, dtype=torch.int64)
        sel = []
        for u in range(s):
            n = counts[u]
            rows = torch.randperm(ctx, generator=g)[:n]  # (any order: no positions)
            idx[0, 0, u, :n] = rows
            sel.append(rows)
        want = torch.empty(1, 16, s, LATENT)
        kv_held = round_kv(kv, "bfp8a") if fp8 and FP8_K_IN == "bfp8" else kv  # (as the K tiles hold it)
        for u in range(s):
            keys = kv_held[0, 0, sel[u]].float()
            p = torch.softmax(q[0, :16, u].float() @ keys.T * scale, dim=-1)
            want[0, :, u] = p @ keys
        d_q = put(mesh_device, q, layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
        d_kv = put(
            mesh_device,
            kv.float() if fp8 else kv,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.fp8_e4m3 if fp8 else ttnn.bfloat16,
            **REP,
        )
        d_idx = put(mesh_device, idx.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
        # the queries as tiles, a token's heads its tile row ([1, S, 16, 512], as `_split_attend` makes them)
        d_qt = put(mesh_device, q[:, :16].permute(0, 2, 1, 3).contiguous(), layout=ttnn.TILE_LAYOUT, **REP)

        def base():
            out = ttnn.transformer.sparse_sdpa(
                d_q,
                d_kv,
                d_idx,
                v_dim=LATENT,
                kv_format=fmt.FP8_E4M3 if fp8 else fmt.BF16,
                scale=scale,
                k_chunk_size=128,
                compute_kernel_config=HIFI4_FP32,
            )
            tiled = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
            ttnn.deallocate(out)
            heads = ttnn.slice(tiled, [0, 0, 0, 0], [1, 16, s, LATENT])
            ttnn.deallocate(tiled)
            return [heads]

        ref = first(base()[0]).float()
        e_ref = (ref - want).abs().max().item()
        for splits, tiled in [(g_, t_) for g_ in sizes("GLM53_KSPLIT_G", "2,4,8,16") for t_ in (0, 1)]:
            if s * splits > grid.x * grid.y:
                continue

            def split():
                parts = sparse_split(mesh_device, d_qt if tiled else d_q, d_kv, d_idx, splits, scale)
                heads = sparse_combine(mesh_device, parts, d_idx, splits, scale)  # [1, S, 16, 512]
                ttnn.deallocate(parts)
                out = ttnn.transpose(heads, 1, 2)  # [1, 16, S, 512]
                ttnn.deallocate(heads)
                return [out]

            if os.environ.get("GLM53_KSPLIT_DEBUG") == "1":  # each unit's partial against torch's
                parts = first(sparse_split(mesh_device, d_q, d_kv, d_idx, splits, scale)).float().reshape(-1, 576)
                for u in range(s):
                    nv = counts[u]
                    n_act = -(-nv // 128)
                    for j in range(splits):
                        c0 = n_act - ((splits - j) * n_act) // splits
                        c1 = n_act - ((splits - j - 1) * n_act) // splits
                        if c1 == c0:
                            continue
                        keys = kv[0, 0, sel[u][c0 * 128 : min(nv, c1 * 128)]].float()
                        sc = q[0, :16, u].float() @ keys.T
                        m = sc.max(-1).values
                        pr = torch.exp((sc - m[:, None]) * scale)
                        l_ = pr.sum(-1)
                        o = (pr @ keys) / l_[:, None]
                        blk = parts[(u * splits + j) * 32 : (u * splits + j) * 32 + 16]
                        print(
                            f"  unit {u}/{j} chunks {c0}-{c1}: |m| {(blk[:, 544] - m).abs().max():.3e}, "
                            f"l {blk[0, 512]:.3f} / {l_[0]:.3f} (max rel {((blk[:, 512] - l_) / l_).abs().max():.3e}), "
                            f"|o/l| {(blk[:, :512] / blk[:, 512:513] - o).abs().max():.3e}",
                            flush=True,
                        )
            got = first(split()[0]).float()
            e_got = (got - want).abs().max().item()
            d = (got - ref).abs().max().item()
            pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
            t_base, t_split = traced(mesh_device, base), traced(mesh_device, split)
            worst = max(worst, e_got / max(e_ref, 1e-6))
            print(
                f"{'fp8 ' if fp8 else ''}S {s:2d} G {splits:2d}{' q tiles' if tiled else ''}: max |split - torch| {e_got:.3e}, |sparse_sdpa - torch| {e_ref:.3e}, "
                f"|split - sparse_sdpa| {d:.3e}, pcc {pcc:.6f}; sparse_sdpa + tiles {t_base * 1e6:.1f} us, "
                f"split + combine + transpose {t_split * 1e6:.1f} us",
                flush=True,
            )
            assert pcc > 0.9995, (s, splits, pcc)
            assert e_got < 4 * max(e_ref, 2e-3), (s, splits, e_got, e_ref)
        for t in (d_q, d_qt, d_kv, d_idx):
            ttnn.deallocate(t)
    print(f"worst error ratio (split / sparse_sdpa, each against torch): {worst:.2f}", flush=True)
