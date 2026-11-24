"""The indexer's scores sharded over the mesh row (`dsa.SCORES_SHARD`): the decode indexer's pool scores split over a
row's 4 chips (`dsa_indexer.pool_scores(shard=True)`: mesh column c the blocks 4 j + c), each chip's top 512, the
candidates gathered over the row (`shard_candidates`, ttnn.all_gather) and picked (`shard_pick`: topk_large_indices of
the 2,048, mapped to pools by the candidates' columns, as select_tokens maps them), against today's form (every chip
the whole row's scores and topk_large_indices), on the same pools cache on every chip (made on the host).

Checked, printed (no assert: a probe's result to read): each chip's scores against the whole row's columns of its blocks
(bit for bit, the -inf of the unseen pools too), and the merged top 512 against today's (the same set; the same order
but for ties).  Timed (GLM53_SDS_TIME=1, trace): today's scores + topk, and the split form's scores + topk + candidates +
all_gather + merge.  Shapes: GLM53_SDS_SHAPES (default 1x8:8192,4x8:8192,1x8:32768,4x8:32768,16x4:32768, users x tokens
: pools a row; the watcher's first run small), positions spread (one user with fewer than 512 visible pools).
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_scores_shard.py -s
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
    TP_AXIS,
    put,
)

REP = dict(dims=(None, None))
RM = ttnn.ROW_MAJOR_LAYOUT
HEADS, DIM, KPOOL, TOPK = 32, 128, 4, 512
SENT = 0xFFFFFFFF


def traced(mesh, fn, reps: int = 10) -> float:
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = []
    try:
        for _ in range(reps):
            outs.append(fn())
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        seconds = (time.perf_counter() - t0) / reps
    finally:
        ttnn.release_trace(mesh, trace)
        for o in outs:
            ttnn.deallocate(o)
    return seconds


def chip(t, c: int) -> torch.Tensor:
    """Mesh row 0, column c's tensor (devices in row-major mesh order)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[c])


def shapes() -> list[tuple[int, int, int]]:
    out = []
    v = os.environ.get("GLM53_SDS_SHAPES", "1x8:8192,4x8:8192,1x8:32768,4x8:32768,16x4:32768")
    for s in v.split(","):
        ut, p = s.split(":")
        u, t = ut.split("x")
        out.append((int(u), int(t), int(p)))
    return out


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_scores_shard(mesh_device):
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import (
        PARAMS,
        pool_scores,
        select_tokens,
        shard_candidates,
        shard_pick,
    )

    mesh = mesh_device
    timing = os.environ.get("GLM53_SDS_TIME", "1") == "1"
    g = torch.Generator().manual_seed(21)
    total_bad = 0
    for users, t, p in shapes():
        n, pp = users * t, p // 32
        # bf16 keys with ties made likely in places (a few repeated rows): the merge's tie rule shows
        cache = (torch.randn(users * pp + 1, 1, 32, DIM, generator=g) * 0.5).to(torch.bfloat16)
        d_cache = put(mesh, cache, **REP)
        qi = (torch.randn(1, 1, n, HEADS * DIM, generator=g) * 0.1).to(torch.bfloat16)
        hw = (torch.randn(1, 1, n, HEADS, generator=g) / 64).to(torch.bfloat16)
        params = torch.zeros(1, 1, n, PARAMS, dtype=torch.int64)
        owner = torch.arange(n) % users
        tok = torch.arange(n) // users
        end = torch.randint(KPOOL * 600, KPOOL * p, (users,), generator=g)  # a user's last token
        end[0] = KPOOL * p - 1  # every pool seen
        if users > 1:
            end[1] = KPOOL * 300  # fewer than 512 pools seen: the SENT tail
        params[0, 0, :, 0] = end[owner] - (t - 1) + tok
        params[0, 0, :, 1] = owner * KPOOL * p
        params[0, 0, :, 2] = owner * pp
        d_qi = put(mesh, qi, layout=RM, **REP)
        d_hw = put(mesh, hw, **REP)
        d_p = put(mesh, params.to(torch.int32), layout=RM, dtype=ttnn.uint32, **REP)

        def today():
            rows = pool_scores(mesh, d_qi, d_cache, d_hw, d_p, p, False, group=(users, t))
            top = ttnn.experimental.topk_large_indices(rows, k=TOPK)
            ttnn.deallocate(rows)
            return top

        def split(keep=None):
            rows = pool_scores(mesh, d_qi, d_cache, d_hw, d_p, p, False, group=(users, t), shard=True)
            top = ttnn.experimental.topk_large_indices(rows, k=TOPK)
            cand = shard_candidates(mesh, rows, top)
            ttnn.deallocate(top)
            every = ttnn.all_gather(cand, dim=3, cluster_axis=TP_AXIS)
            ttnn.deallocate(cand)
            pick = shard_pick(mesh, every, TOPK)
            if keep is not None:
                keep += [rows, every]
            else:
                ttnn.deallocate(rows)
                ttnn.deallocate(every)
            return pick

        def pools_of(pick, every, c) -> torch.Tensor:
            """The picks' pools on chip c (select_tokens' map): pick x -> 32 (4 (col >> 5) + x / 512) + col mod 32."""
            x = chip(pick, c).reshape(n, TOPK).to(torch.int64) & 0xFFFFFFFF
            cols = chip(every, c).reshape(2, n, COLS * TOPK)[1].view(torch.int16).to(torch.int64) & 0xFFFF
            ok = x != SENT
            xs = torch.where(ok, x, torch.zeros_like(x))
            col = torch.gather(cols, 1, xs)
            pool = 32 * (COLS * (col >> 5) + xs // TOPK) + (col & 31)
            return torch.where(ok, pool, torch.full_like(pool, SENT))

        label = f"{users}x{t} P={p}"
        # the scores: chip c's columns 32 j + r = the whole row's pool 32 (4 j + c) + r
        full = pool_scores(mesh, d_qi, d_cache, d_hw, d_p, p, False, group=(users, t))
        whole = chip(full, 0).reshape(n, p)
        ref_top = ttnn.experimental.topk_large_indices(full, k=TOPK)
        ref = chip(ref_top, 0).reshape(n, TOPK).to(torch.int64) & 0xFFFFFFFF
        kept = []
        pick = split(kept)
        got0 = pools_of(pick, kept[1], 0)
        same_order = sum(torch.equal(ref[i], got0[i]) for i in range(n))
        if same_order < n:  # a first difference, for the record
            i = next(i for i in range(n) if not torch.equal(ref[i], got0[i]))
            k0 = int(next(j for j in range(TOPK) if ref[i, j] != got0[i, j]))
            print(
                f"SDS   query {i} from {k0}: today {ref[i, k0:k0 + 6].tolist()} picked {got0[i, k0:k0 + 6].tolist()} "
                f"scores {[float(whole[i, x]) for x in ref[i, k0:k0 + 6].tolist() if x != SENT]}",
                flush=True,
            )
        cols = torch.arange(p).reshape(p // 128, COLS, 32)  # [j, c, r] -> pool
        scores_ok = []
        for c in range(COLS):
            mine = chip(kept[0], c).reshape(n, p // COLS)
            want = whole[:, cols[:, c].reshape(-1)]
            scores_ok.append(torch.equal(mine.view(torch.int16), want.view(torch.int16)))
        # the top 512: today's on chip 0 against the picked form's on every chip of row 0
        set_bad = order_bad = 0
        ties = 0
        for c in range(COLS):
            got = pools_of(pick, kept[1], c)
            for i in range(n):
                a, b = ref[i], got[i]
                if not torch.equal(a, b):
                    order_bad += c == 0
                    sa, sb = set(a.tolist()) - {SENT}, set(b.tolist()) - {SENT}
                    if sa != sb:
                        # a tie at the 512th score explains a different set
                        va = whole[i].float()
                        kth = va[a[a != SENT][-1]] if (a != SENT).any() else None
                        diff = sa ^ sb
                        tie = kth is not None and all(float(va[x]) == float(kth) for x in diff)
                        ties += tie
                        set_bad += not tie
        # select_tokens' map (cand=): the token rows from the picks against those from today's top 512 (as sets)
        tok_today = select_tokens(mesh, ref_top, d_p, 2176, KPOOL)
        tok_split = select_tokens(mesh, pick, d_p, 2176, KPOOL, cand=kept[1])
        ta = chip(tok_today, 0).reshape(n, 2176).to(torch.int64) & 0xFFFFFFFF
        tb = chip(tok_split, 0).reshape(n, 2176).to(torch.int64) & 0xFFFFFFFF
        tok_same = sum((set(ta[i].tolist()) - {SENT}) == (set(tb[i].tolist()) - {SENT}) for i in range(n))
        tok_mapped = all(torch.equal(tb[i, :4 * TOPK].reshape(TOPK, 4)[:, 0] // 4, got0_pools)
                         for i, got0_pools in [(0, got0[0][got0[0] != SENT][: TOPK])]
                         if (got0[0] != SENT).all())  # fmt: skip
        for x in (tok_today, tok_split):
            ttnn.deallocate(x)
        for x in kept:
            ttnn.deallocate(x)
        for x in (full, ref_top, pick):
            ttnn.deallocate(x)
        total_bad += set_bad + (not all(scores_ok))
        print(
            f"SDS {label}: scores bit for bit per chip {scores_ok}; top 512 sets differ (not a tie) in {set_bad} "
            f"query-chips, by a tie at the 512th in {ties}; order differs (chip 0) in {order_bad} of {n} queries; "
            f"select_tokens' map: the same token set in {tok_same} / {n} (query 0 pool order {tok_mapped})",
            flush=True,
        )
        if timing:
            print(
                f"SDS {label}: today {traced(mesh, today) * 1e6:.1f} us, split {traced(mesh, split) * 1e6:.1f} us",
                flush=True,
            )
            # the split form's parts, each on inputs made once
            rows = pool_scores(mesh, d_qi, d_cache, d_hw, d_p, p, False, group=(users, t), shard=True)
            top = ttnn.experimental.topk_large_indices(rows, k=TOPK)
            cand = shard_candidates(mesh, rows, top)
            every = ttnn.all_gather(cand, dim=3, cluster_axis=TP_AXIS)
            part = {
                "scores": lambda: pool_scores(mesh, d_qi, d_cache, d_hw, d_p, p, False, group=(users, t), shard=True),
                "topk": lambda: ttnn.experimental.topk_large_indices(rows, k=TOPK),
                "candidates": lambda: shard_candidates(mesh, rows, top),
                "all_gather": lambda: ttnn.all_gather(cand, dim=3, cluster_axis=TP_AXIS),
                "pick": lambda: shard_pick(mesh, every, TOPK),
            }
            print(
                f"SDS {label} parts: " + ", ".join(f"{k} {traced(mesh, f) * 1e6:.1f}" for k, f in part.items()) + " us",
                flush=True,
            )
            for x in (rows, top, cand, every):
                ttnn.deallocate(x)
        for x in (d_cache, d_qi, d_hw, d_p):
            ttnn.deallocate(x)
    print(f"SDS done: {'all the same' if total_bad == 0 else f'{total_bad} DIFFERENT'}", flush=True)
