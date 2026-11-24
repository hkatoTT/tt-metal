"""Probes of the ttnn ops the DSA indexer's decode would build on, each against torch and timed
in a trace, on every chip of the mesh (replicated inputs): the pool scores of a decode step as a batched matmul
(GLM53_DSAI_SCORES), the top 512 pools by `ttnn.experimental.topk_large_indices` (GLM53_DSAI_TOPK), and the attention
over the selected latents by `ttnn.transformer.sparse_sdpa` with every user's cache in one row-major tensor and global
row ids, one query row a user (GLM53_DSAI_SPARSE).  No checkpoint.

Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_indexer_ops.py -s
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

HEADS, DIM, TOPK_POOLS, KPOOL, LATENT = 32, 128, 512, 4, 512
REP = dict(dims=(None, None))


def traced(mesh, fn, reps: int = 10) -> float:
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    with trace_capture(mesh) as trace:
        outs = [fn() for _ in range(reps)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return seconds


def first(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def sizes(name: str, default: str) -> list[int]:
    return [int(v) for v in os.environ.get(name, default).split(",") if v]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_indexer_ops(mesh_device):
    users = int(os.environ.get("GLM53_DSAI_USERS", "16"))
    g = torch.Generator().manual_seed(0)

    # pool scores: s[u, p] = sum_h w[u, h] relu(q[u, h] . k[u, p]) (the 1 / sqrt(128) and 1 / sqrt(32) folded into w)
    for pools in sizes("GLM53_DSAI_SCORES", "2048,8192"):
        q = torch.randn(users, 1, HEADS, DIM, generator=g).to(torch.bfloat16)
        k = torch.randn(users, 1, pools, DIM, generator=g).to(torch.bfloat16)
        w = (torch.randn(users, 1, HEADS, 1, generator=g) * (DIM * HEADS) ** -0.5).to(torch.bfloat16)
        want = (torch.relu(q.float() @ k.float().transpose(-1, -2)) * w.float()).sum(dim=-2)  # [U, 1, P]
        d_q, d_k, d_w = (put(mesh_device, t, **REP) for t in (q, k, w))

        def scores():
            s = ttnn.matmul(d_q, d_k, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=HIFI4_FP32)
            r = ttnn.relu(s)
            ttnn.deallocate(s)
            m = ttnn.multiply(r, d_w)
            ttnn.deallocate(r)
            out = ttnn.sum(m, dim=-2, keepdim=True)
            ttnn.deallocate(m)
            return out

        got = first(scores()).float().reshape(users, 1, pools)
        rel = ((got - want).abs().max() / want.abs().max()).item()
        sec = traced(mesh_device, scores)
        print(f"scores, {users} users x {pools} pools: max |d| / max {rel:.2e}; {sec * 1e6:.1f} us", flush=True)
        assert rel < 1e-2

    # top 512 pools of each user's row (bf16 row-major), the invisible ones -inf
    for pools in sizes("GLM53_DSAI_TOPK", "2048,8192,32768"):
        s = torch.randn(1, 1, users, pools, generator=g).to(torch.bfloat16)
        s[0, 0, :, pools - pools // 8 :] = float("-inf")  # the last eighth not yet visible
        d_s = put(mesh_device, s, layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
        k = min(TOPK_POOLS, pools - pools // 8)
        topk = lambda: ttnn.experimental.topk_large_indices(d_s, k=k)  # noqa: E731
        got = first(topk()).long().reshape(users, k)
        rows = s.float().reshape(users, pools)
        want = torch.topk(rows, k, dim=-1)
        same = sum(len(set(got[u].tolist()) & set(want.indices[u].tolist())) for u in range(users)) / (users * k)
        # bf16 scores tie often at the k-th value: every pick at least the k-th value, and all distinct
        kth = want.values[:, -1:]
        ok = bool((rows.gather(1, got) >= kth).all()) and all(len(set(got[u].tolist())) == k for u in range(users))
        sec = traced(mesh_device, topk)
        print(
            f"topk {k} of {pools}, {users} rows: {same * 100:.2f} % the same pools (ties at the k-th value), every "
            f"pick >= the k-th value {ok}; {sec * 1e6:.1f} us",
            flush=True,
        )
        assert ok

    # attention over the selected latents: one query row a user, every user's cache [1, 1, U T, 512] row-major, global
    # row ids (2,048 tokens + up to 3 of the tail, padded to the k chunk with 0xFFFFFFFF at the end)
    for ctx in sizes("GLM53_DSAI_SPARSE", "8192"):
        heads = 32  # sparse_sdpa wants a multiple of 32 (a chip's 16 would be padded)
        width = 2048 + 3
        chunk = 128
        topk_w = -(-width // chunk) * chunk
        q = (torch.randn(1, heads, users, LATENT, generator=g) * 0.1).to(torch.bfloat16)
        kv = (torch.randn(1, 1, users * ctx, LATENT, generator=g) * 0.5).to(torch.bfloat16)
        idx = torch.full((1, 1, users, topk_w), 0xFFFFFFFF, dtype=torch.int64)
        sel = []
        for u in range(users):
            n = width - (u % 4)  # a few rows with fewer valid tokens
            rows = torch.randperm(ctx, generator=g)[:n].sort().values + u * ctx
            idx[0, 0, u, :n] = rows
            sel.append(rows)
        scale = LATENT**-0.5
        want = torch.empty(1, heads, users, LATENT)
        for u in range(users):
            keys = kv[0, 0, sel[u]].float()  # [n, 512]
            p = torch.softmax(q[0, :, u].float() @ keys.T * scale, dim=-1)  # [H, n]
            want[0, :, u] = p @ keys
        d_q = put(mesh_device, q, layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
        d_kv = put(mesh_device, kv, layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
        d_idx = put(mesh_device, idx.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)

        fmt = getattr(ttnn.transformer, "SparseKVFormat", None) or getattr(ttnn, "SparseKVFormat")
        back = first(d_idx).reshape(users, topk_w)
        print(f"  indices back: first {back[0, 0].item()}, sentinel {back[users - 1, -1].item()}", flush=True)

        # GLM53_DSAI_FIDELITY=hifi2: the matmuls at HiFi2 (is the op bound by its compute?)
        cfg = HIFI4_FP32
        if os.environ.get("GLM53_DSAI_FIDELITY") == "hifi2":
            cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)

        def attend():
            return ttnn.transformer.sparse_sdpa(
                d_q,
                d_kv,
                d_idx,
                v_dim=LATENT,
                kv_format=fmt.BF16,
                scale=scale,
                k_chunk_size=chunk,
                compute_kernel_config=cfg,
            )

        got = first(attend()).float().reshape(1, heads, users, LATENT)
        pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
        sec = traced(mesh_device, attend)
        print(
            f"sparse_sdpa, {users} users x {heads} heads, context {ctx}, {width} tokens each: pcc {pcc:.6f}, max |d| "
            f"{(got - want).abs().max().item():.3e}; {sec * 1e6:.1f} us",
            flush=True,
        )
        assert pcc > 0.999


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_indexer_steps(mesh_device):
    """The two generic_op steps of `ttnn/dsa_indexer.py` against torch: rows_write (each user's latent row into its row
    of a row-major cache; a 0xFFFFFFFF row writes nothing) and select_tokens (pools -> token rows, the tail, the
    sentinels last), for users with few visible pools (sentinels among the pools) and many, and their times."""
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import PARAMS, SENT, rows_write, select_tokens

    users, ctx, k, width = 16, 8192, TOPK_POOLS, 2176
    g = torch.Generator().manual_seed(1)

    # rows_write
    cache = torch.zeros(1, 1, users * ctx, LATENT, dtype=torch.bfloat16)
    new = cache_new = torch.randn(1, 1, users, LATENT, generator=g).to(torch.bfloat16)
    pos = torch.randint(0, ctx, (users,), generator=g)
    rows = torch.full((1, 1, 1, 32), -1, dtype=torch.int64)
    rows[0, 0, 0, :users] = torch.arange(users) * ctx + pos
    rows[0, 0, 0, 3] = -1  # user 3: nothing
    d_cache = put(mesh_device, cache, layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
    d_new = put(mesh_device, new, layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
    d_rows = put(mesh_device, rows.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
    rows_write(mesh_device, d_new, d_cache, d_rows)
    got = first(d_cache).reshape(users * ctx, LATENT)
    want = cache.reshape(users * ctx, LATENT).clone()
    for u in range(users):
        if u != 3:
            want[u * ctx + pos[u]] = new[0, 0, u]
    print(f"rows_write: equal {torch.equal(got, want)}", flush=True)
    assert torch.equal(got, want)
    # the same rows from the indexer's params (position, base; `dsa.TRIM_WRITES`): user 3 position 0xFFFFFFFF
    params = torch.zeros(1, 1, users, PARAMS, dtype=torch.int64)
    params[0, 0, :, 0] = pos
    params[0, 0, :, 1] = torch.arange(users) * ctx
    params[0, 0, 3, 0] = SENT
    d_cache2 = put(mesh_device, cache, layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
    d_params = put(mesh_device, params.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
    rows_write(mesh_device, d_new, d_cache2, None, params=d_params)
    got2 = first(d_cache2).reshape(users * ctx, LATENT)
    print(f"rows_write from params: equal {torch.equal(got2, want)}", flush=True)
    assert torch.equal(got2, want)

    # select_tokens: users 0-3 near the start (fewer than k visible pools), the rest far in
    pools = torch.full((1, 1, users, k), SENT, dtype=torch.int64)
    params = torch.zeros(1, 1, users, PARAMS, dtype=torch.int64)
    want = torch.full((users, width), SENT, dtype=torch.int64)
    for u in range(users):
        p = int(torch.randint(0, 200, (1,), generator=g)) if u < 4 else int(torch.randint(4096, ctx, (1,), generator=g))
        visible = (p + 1) // 4
        n = min(k, visible)
        chosen = torch.randperm(visible, generator=g)[:n]
        pools[0, 0, u, :n] = chosen
        params[0, 0, u, 0], params[0, 0, u, 1] = p, u * ctx
        toks = (chosen[:, None] * 4 + torch.arange(4)).flatten() + u * ctx
        tail = (p + 1) % 4
        toks = torch.cat([toks, u * ctx + (p + 1 - tail) + torch.arange(tail)])
        want[u, : toks.numel()] = toks
    params[0, 0, 5, 0] = SENT  # user 5: no query
    want[5] = SENT
    as_u32 = lambda t: put(
        mesh_device, t.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP
    )  # noqa: E731
    d_pools, d_params = as_u32(pools), as_u32(params)
    run = lambda: select_tokens(mesh_device, d_pools, d_params, width)  # noqa: E731
    got = first(run()).long().reshape(users, width) & 0xFFFFFFFF
    same = torch.equal(got, want)
    print(
        f"select_tokens: equal {same}"
        + ("" if same else f" (rows differing {(got != want).any(1).nonzero().flatten().tolist()})"),
        flush=True,
    )
    sec = traced(mesh_device, run)
    print(f"select_tokens, {users} users x {width}: {sec * 1e6:.1f} us", flush=True)
    assert same
    from models.demos.blackhole.glm53_flash.ttnn import dsa_indexer

    for sub in (1, 2, 4, users):  # a user's row over cores (dsa_indexer.SELECT_PARTS): the same rows
        dsa_indexer.SELECT_PARTS, dsa_indexer.SELECT_PARTS_MAX = True, 1 << 20  # (the parts at any users)
        d_sub = (as_u32(pools[:, :, :sub]), as_u32(params[:, :, :sub]))
        run_p = lambda: select_tokens(mesh_device, d_sub[0], d_sub[1], width)  # noqa: E731
        got_p = first(run_p()).long().reshape(sub, width) & 0xFFFFFFFF
        same_p = torch.equal(got_p, want[:sub])
        sec_p = traced(mesh_device, run_p)
        dsa_indexer.SELECT_PARTS = False
        sec_1 = traced(mesh_device, lambda: select_tokens(mesh_device, d_sub[0], d_sub[1], width))
        print(
            f"select_tokens over cores, {sub} users: equal {same_p}, {sec_p * 1e6:.1f} us (one core a row "
            f"{sec_1 * 1e6:.1f} us)",
            flush=True,
        )
        assert same_p

    # rows_write of a prefill chunk (512 rows over several cores, rows 500.. none)
    n = 512
    new = torch.randn(1, 1, n, LATENT, generator=g).to(torch.bfloat16)
    rows = torch.full((1, 1, 1, n), -1, dtype=torch.int64)
    rows[0, 0, 0, :500] = 2 * ctx + 1024 + torch.arange(500)
    d_new = put(mesh_device, new, layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
    d_rows = put(mesh_device, rows.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
    rows_write(mesh_device, d_new, d_cache, d_rows)
    got = first(d_cache).reshape(users * ctx, LATENT)
    want_cache = cache.reshape(users * ctx, LATENT).clone()
    for u in range(users):
        if u != 3:
            want_cache[u * ctx + pos[u]] = cache_new[0, 0, u]
    want_cache[2 * ctx + 1024 : 2 * ctx + 1524] = new[0, 0, :500]
    same = torch.equal(got, want_cache)
    print(f"rows_write of {n} rows: equal {same}", flush=True)
    assert same

    # pages_gather: a user's pool pages in a paged [U PP + 1, 1, 32, 128] cache -> contiguous
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import pages_gather

    pages_n = users * 64 + 1
    pool_cache = torch.randn(pages_n, 1, 32, 128, generator=g).to(torch.bfloat16)
    d_pool = put(mesh_device, pool_cache, **REP)
    for n_pages in (16, 64):
        table = (5 * 64 + torch.randperm(64, generator=g)[:n_pages]).to(torch.int32).reshape(1, n_pages)
        d_table = put(mesh_device, table, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, **REP)
        run = lambda: pages_gather(mesh_device, d_pool, d_table)  # noqa: E731
        got = first(run()).reshape(n_pages * 32, 128)
        want_g = pool_cache[table[0].long()].reshape(n_pages * 32, 128)
        same = torch.equal(got, want_g)
        sec = traced(mesh_device, run)
        print(f"pages_gather of {n_pages} pages: equal {same}, {sec * 1e6:.1f} us", flush=True)
        assert same


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_pool_scores(mesh_device):
    """`dsa_indexer.pool_scores` (one op: the pools read from the paged cache, relu(q k) weighted by the head weights,
    the unseen pools -inf, bf16 row-major) against torch in fp32 from the same bf16 inputs: decode (each user its own
    pages, positions spread over the cache) at 1 / 16 / 64 users, and a prefill chunk (512 queries of one user sharing
    its pages, the pools computed up to the chunk's end); the relative error of the seen scores, the -inf pattern,
    the top-512 agreement, and the times."""
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import PARAMS, pool_scores

    g = torch.Generator().manual_seed(3)
    pp = 64  # pages a user: 2,048 pools (a cache of 8,192 tokens)
    users_max = 64
    cache = torch.randn(users_max * pp + 1, 1, 32, DIM, generator=g).to(torch.bfloat16)
    d_cache = put(mesh_device, cache, **REP)
    keys = cache[: users_max * pp].reshape(users_max, pp * 32, DIM).float()

    def check(label, n, owners, positions, n_pools, shared, computed=None):
        q = torch.randn(n, 1, HEADS, DIM, generator=g).to(torch.bfloat16)
        hw = (torch.randn(1, 1, n, HEADS, generator=g) / 64).to(torch.bfloat16)
        params = torch.zeros(1, 1, n, PARAMS, dtype=torch.int64)
        params[0, 0, :, 0] = positions
        params[0, 0, :, 2] = owners * pp
        d_q, d_hw = put(mesh_device, q, **REP), put(mesh_device, hw, **REP)
        d_p = put(mesh_device, params.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
        run = lambda: pool_scores(mesh_device, d_q, d_cache, d_hw, d_p, n_pools, shared, computed)  # noqa: E731
        got = first(run()).float().reshape(n, n_pools)
        k = keys[owners][:, :n_pools]  # [n, P, D]
        s = torch.relu(torch.einsum("nhd,npd->nhp", q[:, 0].float(), k))
        want = (s * hw[0, 0].float()[:, :, None]).sum(1)  # [n, P]
        seen = torch.arange(n_pools)[None] < ((positions + 1) // KPOOL)[:, None]
        same_mask = torch.equal(torch.isinf(got) & (got < 0), ~seen)
        err = ((got - want).abs()[seen].max() / want.abs()[seen].max()).item() if seen.any() else 0.0
        agree = []
        for i in range(n):
            v = int(seen[i].sum())
            if v > TOPK_POOLS:
                a = set(got[i].topk(TOPK_POOLS).indices.tolist())
                b = set(want[i].masked_fill(~seen[i], float("-inf")).topk(TOPK_POOLS).indices.tolist())
                agree.append(len(a & b) / TOPK_POOLS)
        sec = traced(mesh_device, run)
        print(
            f"{label}: -inf where unseen {same_mask}, max err / max {err:.2e}, top-512 agreement "
            f"{min(agree) if agree else 1.0:.4f} (min over {len(agree)} queries), {sec * 1e6:.1f} us",
            flush=True,
        )
        assert same_mask and err < 1e-2
        if not shared:  # the queries row-major ([1, 1, N, 32 x 128], as decode passes them): the same bits
            d_qrm = put(mesh_device, q.reshape(1, 1, n, HEADS * DIM), layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
            run_rm = lambda: pool_scores(
                mesh_device, d_qrm, d_cache, d_hw, d_p, n_pools, shared, computed
            )  # noqa: E731
            got_rm = first(run_rm()).float().reshape(n, n_pools)
            same = torch.equal(torch.nan_to_num(got_rm, neginf=-1e30), torch.nan_to_num(got, neginf=-1e30))
            print(
                f"{label}, row-major queries: bit for bit {same}, {traced(mesh_device, run_rm) * 1e6:.1f} us",
                flush=True,
            )
            assert same

    for users in (1, 16, 64):
        owners = torch.arange(users)
        positions = torch.randint(0, pp * 32 * KPOOL, (users,), generator=g)
        positions[0] = pp * 32 * KPOOL - 1  # every pool seen
        check(f"decode, {users} users, 2,048 pools", users, owners, positions, pp * 32, False)
    start, t = 4096, 512
    owners = torch.full((t,), 5)
    positions = start + torch.arange(t)
    check(
        "prefill, 512 queries at 4,096, 2,048 pools (1,152 computed)",
        t,
        owners,
        positions,
        pp * 32,
        True,
        (start + t) // KPOOL,
    )


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_heads_ops(mesh_device):
    """The DSA prefill's head transforms (`TtDsa._q_heads`, `_kv_heads`, `_join_heads`) by the head ops against
    reshape + permute (+ slice), the form they replaced (written out here), bit for bit, and their traced times, at the
    chunks of GLM53_DSAI_HEADS_T (128, 512, 2,048)."""
    from types import SimpleNamespace

    from models.demos.blackhole.glm53_flash.ttnn import dsa

    me = SimpleNamespace(heads=16, dk=256, dv=256)
    for t in sizes("GLM53_DSAI_HEADS_T", "128,512,2048"):
        g = torch.Generator().manual_seed(t)
        q = put(mesh_device, torch.randn(1, 1, t, 16 * 256, generator=g), **REP, dtype=ttnn.bfloat16)
        kv = put(mesh_device, torch.randn(1, 1, t, 16 * 512, generator=g), **REP, dtype=ttnn.bfloat16)
        attn = put(mesh_device, torch.randn(1, 16, t, 256, generator=g), **REP, dtype=ttnn.bfloat16)

        def kv_reshaped():
            kvp = ttnn.permute(ttnn.reshape(kv, [1, t, 16, 512]), (0, 2, 1, 3))
            key = ttnn.slice(kvp, [0, 0, 0, 0], [1, 16, t, 256])
            value = ttnn.slice(kvp, [0, 0, 0, 256], [1, 16, t, 512])
            ttnn.deallocate(kvp)
            out = ttnn.concat([key, value], dim=-1)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            return out

        cases = {  # name: (the reshape + permute form, the head ops)
            "q": (
                lambda: ttnn.permute(ttnn.reshape(q, [1, t, 16, 256]), (0, 2, 1, 3)),
                lambda: dsa.TtDsa._q_heads(me, q),
            ),
            "kv": (kv_reshaped, lambda: ttnn.concat(list(dsa.TtDsa._kv_heads(me, kv)), dim=-1)),
            "join": (
                lambda: ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), [1, 1, t, 16 * 256]),
                lambda: dsa.TtDsa._join_heads(me, attn),
            ),
        }
        line = []
        for name, fns in cases.items():
            outs, ms = [], []
            for fn in fns:
                outs.append(first(fn()))
                ms.append(traced(mesh_device, fn) * 1e3)
            same = torch.equal(outs[0], outs[1])
            line.append(f"{name} equal {same} ({ms[0]:.3f} -> {ms[1]:.3f} ms)")
            assert same, name
        print(f"{t} tokens: " + ", ".join(line), flush=True)
        for x in (q, kv, attn):
            ttnn.deallocate(x)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_index_update(mesh_device):
    """`dsa_indexer.index_update` (decode's indexer update in one op) against torch: users
    1 / 4 / 16 a row, one segment (the plain step), two (MTP's verify: the draft one position on, crossing into a
    new pool where the token's pos mod 4 is 3) and eight (a verify step of T = 8, over two or three pools), random
    caches and rows: the recent pages bit for bit (row `recent_row(pos)`), the pool rows against the reference's key
    (fp32 softmax over the pool's 4 recent rows with the slot bias, the weights and the products rounded to bf16 as
    `TtDsa._compress`), every other row untouched (the time: the model's profile)."""
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import PARAMS, RECENT_RING, index_update, recent_row

    g = torch.Generator().manual_seed(9)
    pp = 8  # pool pages a user
    for users in (1, 4, 16):
        for segs in (1, 2, 8):
            b = users * segs
            rkey = torch.randn(users + 1, 1, 32, DIM, generator=g).to(torch.bfloat16)
            rgate = torch.randn(users + 1, 1, 32, DIM, generator=g).to(torch.bfloat16)
            pools = torch.randn(users * pp + 1, 1, 32, DIM, generator=g).to(torch.bfloat16)
            ik = torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16)
            ig = torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16)
            bias = torch.full((1, 1, 32, DIM), float("-inf"))
            bias[0, 0, :KPOOL] = torch.randn(KPOOL, DIM, generator=g) * 0.5
            pos0 = torch.randint(0, pp * 32 * KPOOL - max(2, segs), (users,), generator=g)
            pos0[0] = (pos0[0] // KPOOL) * KPOOL + KPOOL - 1  # a token completing its pool (the draft starts the next)
            positions = torch.cat([pos0 + s for s in range(segs)])
            owner = torch.arange(b) % users
            params = torch.zeros(1, 1, b, PARAMS, dtype=torch.int64)
            params[0, 0, :, 0] = positions
            params[0, 0, :, 2] = owner * pp
            want_k, want_g, want_p = rkey.clone(), rgate.clone(), pools.clone()
            for v in range(b):
                u, pos = int(owner[v]), int(positions[v])
                want_k[u, 0, recent_row(pos)] = ik[0, 0, v]
                want_g[u, 0, recent_row(pos)] = ig[0, 0, v]
                pool = pos // KPOOL
                lo = (pool % (RECENT_RING // KPOOL)) * KPOOL  # the pool's rows of the recent pages
                t = want_g[u, 0, lo : lo + KPOOL].float() + bias[0, 0, :KPOOL]
                w = torch.softmax(t, dim=0).to(torch.bfloat16)
                key = (w * want_k[u, 0, lo : lo + KPOOL]).to(torch.bfloat16).float().sum(0).to(torch.bfloat16)
                want_p[u * pp + pool // 32, 0, pool % 32] = key
            d = [put(mesh_device, x, **REP) for x in (ik, ig)]
            d_p = put(mesh_device, params.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
            d_rk, d_rg, d_pools = (put(mesh_device, x, **REP) for x in (rkey, rgate, pools))
            d_bias = put(mesh_device, bias, dtype=ttnn.float32, **REP)
            index_update(mesh_device, d[0], d[1], d_p, d_rk, d_rg, d_pools, d_bias, users, segs)
            got_k, got_g, got_p = (first(x) for x in (d_rk, d_rg, d_pools))
            same_recent = torch.equal(got_k, want_k) and torch.equal(got_g, want_g)
            moved = (want_p != pools).any(-1)  # [P, 1, 32]: the rows the reference wrote
            rest_same = torch.equal(got_p[~moved], pools[~moved])
            err = (got_p.float() - want_p.float()).abs()[moved].max().item() if moved.any() else 0.0
            scale = want_p.float()[moved].abs().max().item() if moved.any() else 1.0
            print(
                f"index_update, {users} users, {segs} segments: recent pages bit for bit {same_recent}, the other "
                f"pool rows untouched {rest_same}, pool rows max |d| {err:.3e} (of {scale:.3f})",
                flush=True,
            )
            assert same_recent and rest_same and err <= 2 * 2**-8 * max(scale, 1.0)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_index_update_verify(mesh_device):
    """`dsa_indexer.index_update` in the verify step's pattern against the plain step's, step after step: T segments a
    user at p .. p + T - 1 (the token and T - 1 drafts, every row written), then n drafts accepted (random, 0 .. T - 1)
    and the next step from p + n + 1, so the rejected drafts' rows are left behind; the plain path one segment at a
    time over the committed positions p .. p + n only (the same rows of ik / ig).  After every step the pool keys of
    the pools complete in the committed positions must be the same bit for bit (layout free: the recent pages' rows
    are not compared).  T in GLM53_DSAI_VERIFY_T (default 2,3,4,8), users 1 / 4 a row, starts at each position mod 4
    and near a pool page's end.  Prints only (VIDX lines, VIDX RESULT; no assert)."""
    from models.demos.blackhole.glm53_flash.ttnn.dsa import PAGE
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import PARAMS, index_update

    g = torch.Generator().manual_seed(31)
    pp, max_seq = 8, 8 * 32 * KPOOL  # pool pages a user (1,024 positions)
    bias = torch.full((1, 1, PAGE, DIM), float("-inf"))
    bias[0, 0, :KPOOL] = torch.randn(KPOOL, DIM, generator=g) * 0.5
    d_bias = put(mesh_device, bias, dtype=ttnn.float32, **REP)
    bad = []
    for n in sizes("GLM53_DSAI_VERIFY_T", "2,3,4,8"):
        for users in (1, 4):
            starts = [1, 2, 3, 125][:users] if users > 1 else [int(torch.randint(0, 4, (1,), generator=g))]
            zero = lambda shape: put(mesh_device, torch.zeros(shape, dtype=torch.bfloat16), **REP)  # noqa: E731
            shapes = ([users + 1, 1, PAGE, DIM], [users + 1, 1, PAGE, DIM], [users * pp + 1, 1, PAGE, DIM])
            ver = [zero(s) for s in shapes]  # the verify pattern's state
            plain = [zero(s) for s in shapes]  # the plain step's
            pos = list(starts)
            steps = differing = compared = 0
            first_bad = None

            def params_of(positions, owner):
                p = torch.zeros(1, 1, len(positions), PARAMS, dtype=torch.int64)
                p[0, 0, :, 0] = torch.tensor(positions)
                p[0, 0, :, 1] = torch.tensor(owner) * max_seq
                p[0, 0, :, 2] = torch.tensor(owner) * pp
                return put(mesh_device, p.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)

            while steps < 20:
                b = n * users
                owner = [v % users for v in range(b)]
                positions = [pos[v % users] + v // users for v in range(b)]  # the tokens, then the drafts
                ik_t = torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16)
                ig_t = torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16)
                ik, ig, d_p = put(mesh_device, ik_t, **REP), put(mesh_device, ig_t, **REP), params_of(positions, owner)
                index_update(mesh_device, ik, ig, d_p, ver[0], ver[1], ver[2], d_bias, users, n)
                for t in (ik, ig, d_p):
                    ttnn.deallocate(t)
                acc = [int(x) for x in torch.randint(0, n, (users,), generator=g)]
                if steps % 3 == 2:
                    acc = [n - 1] * users  # (every draft kept now and then)
                # the plain step over the committed positions p .. p + n (a user with fewer again at its last: the
                # same row and key, written again)
                for j in range(max(acc) + 1):
                    sel = [min(j, acc[u]) for u in range(users)]
                    rows = [s * users + u for u, s in enumerate(sel)]
                    ik1 = put(mesh_device, ik_t[:, :, rows].contiguous(), **REP)
                    ig1 = put(mesh_device, ig_t[:, :, rows].contiguous(), **REP)
                    d_p1 = params_of([pos[u] + sel[u] for u in range(users)], list(range(users)))
                    index_update(mesh_device, ik1, ig1, d_p1, plain[0], plain[1], plain[2], d_bias, users, 1)
                    for t in (ik1, ig1, d_p1):
                        ttnn.deallocate(t)
                pos = [pos[u] + acc[u] + 1 for u in range(users)]
                steps += 1
                pv, pl = first(ver[2]), first(plain[2])
                for u in range(users):
                    for k in range(starts[u] // KPOOL, pos[u] // KPOOL):  # the pools complete in committed positions
                        a = pv[u * pp + k // 32, 0, k % 32]
                        c = pl[u * pp + k // 32, 0, k % 32]
                        compared += 1
                        if not torch.equal(a.view(torch.int16), c.view(torch.int16)):
                            differing += 1
                            if first_bad is None:
                                first_bad = (steps, u, k)
            ok = differing == 0
            if not ok:
                bad.append((n, users))
            print(
                f"VIDX T={n} users={users} starts {starts}: {steps} steps, complete pools differing from the plain "
                f"step {differing} / {compared} (first: step, user, pool {first_bad})",
                flush=True,
            )
            for t in ver + plain:
                ttnn.deallocate(t)
    print(f"VIDX RESULT {'all the same' if not bad else 'DIFFER: ' + str(bad)}", flush=True)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_index_update_time(mesh_device):
    """`dsa_indexer.index_update`'s time in a trace (10 calls a replay, µs a call): users 1 / 4 / 16 a row x segments
    1 / 2 / 4 / 8 (the plain step, MTP's K = 1, T = 4 / 8), positions near a pool page's end.  Prints only (VIDXT
    lines)."""
    from models.demos.blackhole.glm53_flash.ttnn.dsa import PAGE
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import PARAMS, index_update

    g = torch.Generator().manual_seed(41)
    pp, max_seq = 8, 8 * 32 * KPOOL
    bias = torch.full((1, 1, PAGE, DIM), float("-inf"))
    bias[0, 0, :KPOOL] = torch.randn(KPOOL, DIM, generator=g) * 0.5
    d_bias = put(mesh_device, bias, dtype=ttnn.float32, **REP)
    for users in (1, 4, 16):
        for segs in (1, 2, 4, 8):
            b = users * segs
            owner = [v % users for v in range(b)]
            positions = [125 + (v % users) % 3 + v // users for v in range(b)]
            params = torch.zeros(1, 1, b, PARAMS, dtype=torch.int64)
            params[0, 0, :, 0] = torch.tensor(positions)
            params[0, 0, :, 1] = torch.tensor(owner) * max_seq
            params[0, 0, :, 2] = torch.tensor(owner) * pp
            d_p = put(mesh_device, params.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
            ik = put(mesh_device, torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16), **REP)
            ig = put(mesh_device, torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16), **REP)
            caches = [
                put(mesh_device, torch.randn(shape, generator=g).to(torch.bfloat16), **REP)
                for shape in ([users + 1, 1, PAGE, DIM], [users + 1, 1, PAGE, DIM], [users * pp + 1, 1, PAGE, DIM])
            ]

            index_update(mesh_device, ik, ig, d_p, *caches, d_bias, users, segs)  # (compiled before the capture)
            ttnn.synchronize_device(mesh_device)
            reps = 10
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:  # (device ops only in the capture)
                for _ in range(reps):
                    index_update(mesh_device, ik, ig, d_p, *caches, d_bias, users, segs)
            finally:
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            us = (time.perf_counter() - t0) / reps * 1e6
            ttnn.release_trace(mesh_device, trace)
            print(f"VIDXT users={users} segments={segs}: {us:.1f} us a call", flush=True)
            for t in [ik, ig, d_p] + caches:
                ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_index_update_vs_ops(mesh_device):
    """`dsa_indexer.index_update` against the ttnn ops it replaced (the decode's segment loop: paged_update_cache of the
    recent rows, the pool keys by `TtDsa._compress`, paged_update_cache of the pool rows; written out here) on the same
    states, step after step: MTP's verify (the token at p,
    the draft at p + 1; the next token at p + 2 when the draft is kept, else at p + 1) and the plain step, from starts
    just before a pool's end and a pool page's end (positions mod 4 = 3 -> 0, mod 128 = 127 -> 0), with the slot bias of
    the real weights' form (random, rows 4-31 -inf): rkey / rgate / pools compared after every step (bit for bit, or
    the differing rows and their largest difference)."""
    import types

    from models.demos.blackhole.glm53_flash.ttnn.dsa import PAGE, TtDsa
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import PARAMS, RECENT_RING, index_update, recent_row

    g = torch.Generator().manual_seed(21)
    pp, max_seq = 8, 8 * 32 * KPOOL  # pool pages a user (1,024 positions)
    bias = torch.full((1, 1, PAGE, DIM), float("-inf"))
    bias[0, 0, :KPOOL] = torch.randn(KPOOL, DIM, generator=g) * 0.5
    fake = types.SimpleNamespace(
        mesh=mesh_device,
        grid=mesh_device.compute_with_storage_grid_size(),
        index_dim=DIM,
        pool_bias=put(mesh_device, bias, dtype=ttnn.float32, **REP),
    )
    fake._compress = types.MethodType(TtDsa._compress, fake)

    def write(cache, new, positions, page_table):  # each user's row at its position (one user per core)
        g_, d_ = new.shape[1], new.shape[-1]
        grid = ttnn.num_cores_to_corerangeset(g_, fake.grid, row_wise=True)
        spec = ttnn.ShardSpec(grid, [32, d_], ttnn.ShardOrientation.ROW_MAJOR)
        sharded = ttnn.to_memory_config(
            new, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
        )
        ttnn.experimental.paged_update_cache(cache, sharded, update_idxs_tensor=positions, page_table=page_table)
        ttnn.deallocate(sharded)

    def pool_keys(rkey, rgate, seg_pos, owner_of):  # each user's pool's key from its pool's 4 rows of its recent pages
        batch = len(seg_pos)

        def rows(cache):  # [U, 1, 32, 128]: user u's pool's 4 rows at rows 0-3, zeros below (as the kernel's tiles)
            parts = []
            for u, p in enumerate(seg_pos):  # (padded row-major, as `TtDsa._chunk_pool_keys`: a tile's pad is not 0)
                lo = ((p // KPOOL) % (RECENT_RING // KPOOL)) * KPOOL
                four = ttnn.to_layout(
                    ttnn.slice(cache, [owner_of[u], 0, lo, 0], [owner_of[u] + 1, 1, lo + KPOOL, DIM]),
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                padded = ttnn.pad(four, [(0, 0), (0, 0), (0, PAGE - KPOOL), (0, 0)], 0.0)
                parts.append(ttnn.to_layout(padded, ttnn.TILE_LAYOUT))
                ttnn.deallocate(four)
                ttnn.deallocate(padded)
            out = ttnn.concat(parts, dim=0) if batch > 1 else parts[0]
            if batch > 1:
                for t in parts:
                    ttnn.deallocate(t)
            return out

        rk, rg = rows(rkey), rows(rgate)
        key = fake._compress(rk, rg)  # [U, 1, 1, 128]
        ttnn.deallocate(rk)
        ttnn.deallocate(rg)
        return ttnn.reshape(key, [1, batch, 1, DIM])

    table = lambda rows: put(  # noqa: E731
        mesh_device, torch.tensor(rows, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **REP
    )
    ints = lambda v: put(  # noqa: E731
        mesh_device, torch.tensor(v, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **REP
    )
    worst = 0
    for users, segs in ((1, 2), (2, 2), (4, 2), (4, 1)):
        b = users * segs
        owner = [v % users for v in range(b)]
        starts = [125, 126, 127, 3, 254, 381][:users] + [5] * max(0, users - 6)
        keeps = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
        rk0 = torch.randn(users + 1, 1, PAGE, DIM, generator=g).to(torch.bfloat16)
        rg0 = torch.randn(users + 1, 1, PAGE, DIM, generator=g).to(torch.bfloat16)
        pl0 = torch.randn(users * pp + 1, 1, PAGE, DIM, generator=g).to(torch.bfloat16)
        ops = [put(mesh_device, x, **REP) for x in (rk0, rg0, pl0)]  # today's path's state
        new = [put(mesh_device, x, **REP) for x in (rk0, rg0, pl0)]  # index_update's
        pos = list(starts)
        for step, keep in enumerate(keeps):
            positions = [pos[v % users] + v // users for v in range(b)]  # the tokens, then the drafts
            ik_t = torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16)
            ig_t = torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16)
            ik, ig = put(mesh_device, ik_t, **REP), put(mesh_device, ig_t, **REP)
            # today's path: a segment at a time, its users' recent rows, then every user's pool key, its users' rows
            for seg in range(segs):
                lo, hi = seg * users, (seg + 1) * users
                seg_pos = positions[lo:hi]
                recent = (ints([recent_row(p) for p in seg_pos]), table([[owner[v]] for v in range(lo, hi)]))
                for cache, src in ((ops[0], ik), (ops[1], ig)):
                    part = ttnn.slice(src, [0, 0, lo, 0], [1, 1, hi, DIM])
                    write(cache, ttnn.reshape(part, [1, users, 1, DIM]), *recent)
                key = pool_keys(ops[0], ops[1], seg_pos, owner[lo:hi])
                pools_t = (
                    ints([p // KPOOL for p in seg_pos]),
                    table([[owner[v] * pp + j for j in range(pp)] for v in range(lo, hi)]),
                )
                write(ops[2], key, *pools_t)
                ttnn.deallocate(key)
            params = torch.zeros(1, 1, b, PARAMS, dtype=torch.int64)
            params[0, 0, :, 0] = torch.tensor(positions)
            params[0, 0, :, 1] = torch.tensor(owner) * max_seq
            params[0, 0, :, 2] = torch.tensor(owner) * pp
            d_p = put(mesh_device, params.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
            index_update(mesh_device, ik, ig, d_p, new[0], new[1], new[2], fake.pool_bias, users, segs)
            report = []
            for name, a_, b_ in zip(("rkey", "rgate", "pools"), ops, new):
                x, y = first(a_), first(b_)
                rows = (x != y).any(-1)  # [N, 1, 32]
                if rows.any():
                    where = [tuple(i) for i in rows.nonzero()[:4, [0, 2]].tolist()]
                    report.append(
                        f"{name}: {int(rows.sum())} rows differ (max |d| "
                        f"{(x.float() - y.float()).abs().max().item():.3e}, first {where})"
                    )
                    worst = max(worst, int(rows.sum()) if name != "pools" else 0)
            print(
                f"index_update vs today's, {users} users x {segs}, step {step} at {positions}: "
                + ("; ".join(report) if report else "bit for bit"),
                flush=True,
            )
            for t in (ik, ig, d_p):
                ttnn.deallocate(t)
            # the next step's token: after the kept draft (p + 2) or at the draft's position (p + 1)
            pos = [p + (1 + keep if segs == 2 else 1) for p in pos]
    assert worst == 0, "the recent rows differ from today's path"


# index_update's split form (the arguments; dsa_indexer.index_update): the default, and the steps it is made of
SPLIT_FORMS = {
    "split": dict(split=True),  # a core a segment, the kept keys only, 4 tiles an op
    "parts1": dict(split=True, parts=1),  # a core a user
    "parts2": dict(split=True, parts=2),
    "all-keys": dict(split=True, skip=False, batch=1),  # every key made and sent, a tile an op
    "one-core-all": dict(split=True, parts=1, skip=False, batch=1),  # the one-core form's work in the new kernels
    "one-core-batch": dict(split=True, parts=1, skip=False),  # + 4 tiles an op
    "one-core": dict(split=False),  # the reference form (kernels/dsa/index_update_*.cpp)
}


def _split_positions(g, mode: int, base: int, segs: int, limit: int) -> list[int]:
    """A user's segments' positions: 0 consecutive (the verify step), 1 all the same (a free seat's), 2 random in 16
    on (rows and pools written twice), 3 descending, 4 16 apart (one ring row, different pools)."""
    if mode == 0:
        out = [base + s for s in range(segs)]
    elif mode == 1:
        out = [base] * segs
    elif mode == 2:
        out = [base + int(v) for v in torch.randint(0, 16, (segs,), generator=g)]
    elif mode == 3:
        out = [base + segs - 1 - s for s in range(segs)]
    else:
        out = [base + 16 * s for s in range(segs)]
    return [min(p, limit - 1) for p in out]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_index_update_split(mesh_device):
    """`index_update`'s split form against the one-core form (split=False) bit for bit, step after step on the same
    random states: the recent pages and the pool pages whole (the rows of pools not complete too) after every step.
    T segments a user in GLM53_DSAI_SPLIT_T (default 1,2,3,4,8,14), users 1 / 4 / 16 a row, two patterns:
    verify (positions p .. p + T - 1, then n accepted at random and the next step from p + n + 1: the rejected drafts'
    rows left behind and written again; starts at each position mod 4 and before a pool page's end) and random (each
    user's positions in one of `_split_positions`' forms a step).  The forms in GLM53_DSAI_SPLIT_FORMS (default split;
    the others at 4 users: parts1, parts2, all-keys, one-core-all, one-core-batch).  Prints only (VSPLIT lines,
    VSPLIT RESULT)."""
    from models.demos.blackhole.glm53_flash.ttnn.dsa import PAGE
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import PARAMS, index_update

    g = torch.Generator().manual_seed(53)
    pp, max_seq = 8, 8 * 32 * KPOOL  # pool pages a user (1,024 positions)
    steps = int(os.environ.get("GLM53_DSAI_SPLIT_STEPS", "6"))
    names = os.environ.get("GLM53_DSAI_SPLIT_FORMS", "parts1,parts2,all-keys,one-core-all,one-core-batch")
    extra = [f for f in names.split(",") if f]
    bias = torch.full((1, 1, PAGE, DIM), float("-inf"))
    bias[0, 0, :KPOOL] = torch.randn(KPOOL, DIM, generator=g) * 0.5
    d_bias = put(mesh_device, bias, dtype=ttnn.float32, **REP)
    bad, runs = [], 0
    for segs in sizes("GLM53_DSAI_SPLIT_T", "1,2,3,4,8,14"):
        for users in (1, 4, 16):
            for pattern in ("verify", "random"):
                forms = ["split"] + (extra if users == 4 else [])
                shapes = ([users + 1, 1, PAGE, DIM], [users + 1, 1, PAGE, DIM], [users * pp + 1, 1, PAGE, DIM])
                init = [torch.randn(s, generator=g).to(torch.bfloat16) for s in shapes]
                ref = [put(mesh_device, x, **REP) for x in init]
                states = {f: [put(mesh_device, x, **REP) for x in init] for f in forms}
                starts = [(125, 1, 2, 3)[u % 4] + 8 * (u // 4) for u in range(users)]
                pos = list(starts)
                differ = {f: [] for f in forms}
                for step in range(steps):
                    b = users * segs
                    if pattern == "verify":
                        positions = [pos[v % users] + v // users for v in range(b)]
                    else:
                        per_user = []
                        for u in range(users):
                            mode = int(torch.randint(0, 5, (1,), generator=g))
                            base = int(torch.randint(0, max_seq - 16 * segs, (1,), generator=g))
                            per_user.append(_split_positions(g, mode, base, segs, max_seq))
                        positions = [per_user[v % users][v // users] for v in range(b)]
                    owner = [v % users for v in range(b)]
                    p = torch.zeros(1, 1, b, PARAMS, dtype=torch.int64)
                    p[0, 0, :, 0] = torch.tensor(positions)
                    p[0, 0, :, 1] = torch.tensor(owner) * max_seq
                    p[0, 0, :, 2] = torch.tensor(owner) * pp
                    d_p = put(mesh_device, p.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
                    ik = put(mesh_device, torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16), **REP)
                    ig = put(mesh_device, torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16), **REP)
                    index_update(mesh_device, ik, ig, d_p, *ref, d_bias, users, segs, split=False)
                    want = [first(x).view(torch.int16) for x in ref]
                    for f in forms:
                        index_update(mesh_device, ik, ig, d_p, *states[f], d_bias, users, segs, **SPLIT_FORMS[f])
                        got = [first(x).view(torch.int16) for x in states[f]]
                        rows = [int((w != h).any(-1).sum()) for w, h in zip(want, got)]
                        if any(rows):
                            differ[f].append((step, rows))
                    for t in (ik, ig, d_p):
                        ttnn.deallocate(t)
                    if pattern == "verify":
                        acc = [int(x) for x in torch.randint(0, segs, (users,), generator=g)]
                        if step % 3 == 2:
                            acc = [segs - 1] * users
                        pos = [pos[u] + acc[u] + 1 for u in range(users)]
                        if max(pos) + segs >= max_seq:
                            pos = list(starts)
                for f in forms:
                    runs += 1
                    if differ[f]:
                        bad.append((segs, users, pattern, f))
                    print(
                        f"VSPLIT T={segs} users={users} {pattern} {f}: {steps} steps, rows differing (rkey, rgate, "
                        f"pools) {differ[f][:3] if differ[f] else 'none'}",
                        flush=True,
                    )
                for t in ref + [x for f in forms for x in states[f]]:
                    ttnn.deallocate(t)
    print(f"VSPLIT RESULT {'all the same' if not bad else 'DIFFER: ' + str(bad)} ({runs} runs)", flush=True)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_index_update_split_time(mesh_device):
    """`index_update`'s forms timed in a trace (10 calls a replay, µs a call), in turn twice (the forms in
    GLM53_DSAI_SPLIT_TIME_FORMS, default one-core, one-core-all, one-core-batch, parts1, split): users 1 / 4 / 16 a
    row x segments 1 / 2 / 4 / 8 (GLM53_DSAI_SPLIT_T), positions near a pool page's end (as
    test_dsa_index_update_time).  Prints only (VIDXS lines)."""
    from models.demos.blackhole.glm53_flash.ttnn.dsa import PAGE
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import PARAMS, index_update

    g = torch.Generator().manual_seed(41)
    pp, max_seq = 8, 8 * 32 * KPOOL
    names = os.environ.get("GLM53_DSAI_SPLIT_TIME_FORMS", "one-core,one-core-all,one-core-batch,parts1,split")
    forms = [f for f in names.split(",") if f]
    bias = torch.full((1, 1, PAGE, DIM), float("-inf"))
    bias[0, 0, :KPOOL] = torch.randn(KPOOL, DIM, generator=g) * 0.5
    d_bias = put(mesh_device, bias, dtype=ttnn.float32, **REP)
    for users in (1, 4, 16):
        for segs in sizes("GLM53_DSAI_SPLIT_T", "1,2,4,8"):
            b = users * segs
            owner = [v % users for v in range(b)]
            positions = [125 + (v % users) % 3 + v // users for v in range(b)]
            params = torch.zeros(1, 1, b, PARAMS, dtype=torch.int64)
            params[0, 0, :, 0] = torch.tensor(positions)
            params[0, 0, :, 1] = torch.tensor(owner) * max_seq
            params[0, 0, :, 2] = torch.tensor(owner) * pp
            d_p = put(mesh_device, params.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
            ik = put(mesh_device, torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16), **REP)
            ig = put(mesh_device, torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16), **REP)
            caches = [
                put(mesh_device, torch.randn(shape, generator=g).to(torch.bfloat16), **REP)
                for shape in ([users + 1, 1, PAGE, DIM], [users + 1, 1, PAGE, DIM], [users * pp + 1, 1, PAGE, DIM])
            ]
            times = {f: [] for f in forms}
            for _ in range(2):
                for f in forms:
                    kw = SPLIT_FORMS[f]
                    index_update(mesh_device, ik, ig, d_p, *caches, d_bias, users, segs, **kw)  # (compiled first)
                    ttnn.synchronize_device(mesh_device)
                    reps = 10
                    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                    try:
                        for _ in range(reps):
                            index_update(mesh_device, ik, ig, d_p, *caches, d_bias, users, segs, **kw)
                    finally:
                        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    times[f].append((time.perf_counter() - t0) / reps * 1e6)
                    ttnn.release_trace(mesh_device, trace)
            line = ", ".join(f"{f} {' / '.join(f'{t:.1f}' for t in times[f])}" for f in forms)
            print(f"VIDXS users={users} segments={segs}: {line} us a call", flush=True)
            for t in [ik, ig, d_p] + caches:
                ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_pool_keys(mesh_device):
    """`dsa_indexer.pool_keys` (the prefill chunk's pools' keys in one op) against the ttnn ops it replaces
    (`TtDsa._chunk_pool_keys` with PREFILL_POOL_KEYS_OP off), bit for bit on every chip, for T = GLM53_DPK_T (128, 512,
    1,024, 2,048) and gates / keys / slot biases of several forms: N(0, 1); the gates x 40 (a pool's softmax down to
    exp underflow: weights 0 and subnormal); the synthetic weights' bias 0; ties (the gates rounded to integers) and
    zero keys (the bf16 multiply's 0 rule).  Then both timed in a trace (GLM53_DPK_TIME=1)."""
    import types

    from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_mod
    from models.demos.blackhole.glm53_flash.ttnn.dsa import PAGE, TtDsa
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import pool_keys_bias

    g = torch.Generator().manual_seed(53)

    def fake_for(bias_rows):  # bias_rows [4, 128] or None (the synthetic weights' 0)
        bias = torch.full((1, 1, PAGE, DIM), float("-inf"))
        bias[0, 0, :KPOOL] = 0.0 if bias_rows is None else bias_rows
        fake = types.SimpleNamespace(
            mesh=mesh_device,
            index_dim=DIM,
            kpool=KPOOL,
            pool_bias=put(mesh_device, bias, dtype=ttnn.float32, **REP),
            pool_rows_bias=pool_keys_bias(mesh_device, bias_rows),
        )
        fake._compress = types.MethodType(TtDsa._compress, fake)
        fake._chunk_pool_keys = types.MethodType(TtDsa._chunk_pool_keys, fake)
        return fake

    def keys(fake, ik, ig, fused: bool):
        saved = dsa_mod.PREFILL_POOL_KEYS_OP
        dsa_mod.PREFILL_POOL_KEYS_OP = fused
        try:
            return fake._chunk_pool_keys(ik, ig)
        finally:
            dsa_mod.PREFILL_POOL_KEYS_OP = saved

    def chips(t):
        return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]

    forms = {
        "normal": lambda t: (torch.randn(t, DIM, generator=g), torch.randn(t, DIM, generator=g), True),
        "wide gates": lambda t: (torch.randn(t, DIM, generator=g), torch.randn(t, DIM, generator=g) * 40, True),
        "bias 0": lambda t: (torch.randn(t, DIM, generator=g), torch.randn(t, DIM, generator=g), False),
        "ties, zero keys": lambda t: (
            torch.randn(t, DIM, generator=g) * (torch.rand(t, DIM, generator=g) > 0.3),
            torch.randn(t, DIM, generator=g).mul(1.5).round(),
            True,
        ),
    }
    bad = 0
    timing = os.environ.get("GLM53_DPK_TIME", "1") == "1"
    for t in sizes("GLM53_DPK_T", "128,512,1024,2048"):
        for name, make in forms.items():
            k, gt, with_bias = make(t)
            fake = fake_for(torch.randn(KPOOL, DIM, generator=g) * 0.5 if with_bias else None)
            ik = put(mesh_device, k.to(torch.bfloat16).reshape(1, 1, t, DIM), **REP)
            ig = put(mesh_device, gt.to(torch.bfloat16).reshape(1, 1, t, DIM), **REP)
            ref, new = keys(fake, ik, ig, False), keys(fake, ik, ig, True)
            assert tuple(new.shape) == tuple(ref.shape) == (1, 1, t // KPOOL, DIM), (new.shape, ref.shape)
            r_all, n_all = chips(ref), chips(new)
            diff = 0
            for r_, n_ in zip(r_all, n_all):
                diff += int((r_.view(torch.int16) != n_.view(torch.int16)).sum())
            same_chips = all(torch.equal(r_all[0].view(torch.int16), x.view(torch.int16)) for x in n_all)
            line = f"pool_keys T {t} {name}: {diff} values differ over the 32 chips"
            if diff:
                r0, n0 = r_all[0].float(), n_all[0].float()
                at = (r_all[0].view(torch.int16) != n_all[0].view(torch.int16)).nonzero()[:4].tolist()
                line += f" (max |d| {(r0 - n0).abs().max().item():.3e}, first {at})"
            if not same_chips:
                line += "; the chips differ"
            if timing and name == "normal":
                us_ref = traced(mesh_device, lambda: keys(fake, ik, ig, False)) * 1e6
                us_new = traced(mesh_device, lambda: keys(fake, ik, ig, True)) * 1e6
                line += f"; traced {us_ref:.1f} -> {us_new:.1f} us"
            print(line, flush=True)
            bad += diff + (0 if same_chips else 1)
            for x in (ik, ig, ref, new):
                ttnn.deallocate(x)
    assert bad == 0, "pool_keys differs from the ttnn ops"
