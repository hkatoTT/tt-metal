"""`dsa_indexer.pool_scores` with SCORES_BATCH (GLM53_DSA_SCORES_BATCH: the shared queries' two stages a batch of
queries at a time, the default 4) against one query at a time (0: the earlier form), bit for bit, and their traced
times.

Shapes (GLM53_SB_DECODE: users a mesh row x tokens : pools a row : s(sharded over the mesh row, as dsa.SCORES_SHARD) /
u(unsharded) : dtype of the pools): a decode group's T queries a core at a 1M cache (8 / 32 / 64 users' K = 7 verify
step), 128K, T = 3 / 4 / 2 / 5 / 7 (the MTP K = 2, DFlash2 K = 3, MTP K = 1 steps and odd T: batches shorter than 4
that do not divide the CB rings, a case that once went wrong), bf16 pools; and prefill chunks (GLM53_SB_PREFILL:
queries @ start : pools a user, the shared form, 32 queries a core, the pools computed to the chunk's end; 37 queries:
a core of 5; 1 / 2 / 3 queries: one short core).  Batches GLM53_SB_BATCHES (default 4,3,2).  The pools cache is made
on the device (synthetic).  Every chip of mesh row 0 is compared (sharded: each its own blocks), the -inf of the
unseen pools too.
Prints "SB RESULT: all bit for bit" or the count of differing cases; no assert (a probe: its printed lines are the
result).
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_scores_batch.py -s
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
    put,
    synthetic,
)

REP = dict(dims=(None, None))
RM = ttnn.ROW_MAJOR_LAYOUT
HEADS, DIM, KPOOL = 32, 128, 4
DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


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


def row0(t) -> list[torch.Tensor]:
    """Mesh row 0's chips' tensors as int16 bits (bf16 scores, -inf included)."""
    return [ttnn.to_torch(x).contiguous().view(torch.int16) for x in ttnn.get_device_tensors(t)[:COLS]]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_scores_batch(mesh_device):
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import PARAMS, pool_scores

    mesh = mesh_device
    batches = [int(b) for b in os.environ.get("GLM53_SB_BATCHES", "4,3,2").split(",")]
    timing = os.environ.get("GLM53_SB_TIME", "1") == "1"
    g = torch.Generator().manual_seed(29)
    bad = cases = 0

    def compare(label, run):
        nonlocal bad, cases
        os.environ["GLM53_DSA_SCORES_BATCH"] = "0"
        try:
            ref = run()
        finally:
            os.environ.pop("GLM53_DSA_SCORES_BATCH", None)
        want = row0(ref)
        ttnn.deallocate(ref)
        line = []
        for b in batches:
            os.environ["GLM53_DSA_SCORES_BATCH"] = str(b)
            try:
                out = run()
                got = row0(out)
                ttnn.deallocate(out)
            finally:
                os.environ.pop("GLM53_DSA_SCORES_BATCH", None)
            same = all(torch.equal(a, w) for a, w in zip(got, want))
            cases += 1
            bad += not same
            line.append(f"batch {b} {'bit for bit' if same else 'DIFFERENT'}")
        if timing:
            os.environ["GLM53_DSA_SCORES_BATCH"] = "0"
            try:
                t_off = traced(mesh, run) * 1e6
            finally:
                os.environ.pop("GLM53_DSA_SCORES_BATCH", None)
            ts = []
            for b in batches:
                os.environ["GLM53_DSA_SCORES_BATCH"] = str(b)
                try:
                    ts.append(f"{b}: {traced(mesh, run) * 1e6:.1f}")
                finally:
                    os.environ.pop("GLM53_DSA_SCORES_BATCH", None)
            line.append(f"times us: off {t_off:.1f}, " + ", ".join(ts))
        print(f"SB {label}: " + "; ".join(line), flush=True)

    # decode: a user's T queries a core (group), positions at the cache's end (every pool seen) but for user 1's
    # (half the cache: the visible blocks end inside the range)
    decode = os.environ.get(
        "GLM53_SB_DECODE",
        "1x8:262144:s:bfp4,4x8:262144:s:bfp4,8x8:262144:s:bfp4,4x8:32768:s:bfp4,4x3:32768:s:bfp4,16x4:32768:s:bfp4,"
        "8x2:32768:u:bfp4,4x8:32768:s:bf16,4x5:32768:s:bfp4,4x7:32768:s:bfp4,1x7:262144:s:bfp4,2x3:32768:u:bfp4",
    )
    for spec in [s for s in decode.split(",") if s]:
        ut, p, form, dt = spec.split(":")
        users, t = (int(v) for v in ut.split("x"))
        p, shard = int(p), form == "s"
        n, pp = users * t, p // 32
        cache = synthetic(mesh, [users * pp + 1, 1, 32, DIM], DTYPES[dt])
        qi = (torch.randn(1, 1, n, HEADS * DIM, generator=g) * 0.1).to(torch.bfloat16)
        hw = (torch.randn(1, 1, n, HEADS, generator=g) / 64).to(torch.bfloat16)
        params = torch.zeros(1, 1, n, PARAMS, dtype=torch.int64)
        owner, tok = torch.arange(n) % users, torch.arange(n) // users
        end = torch.full((users,), KPOOL * p - 1)
        if users > 1:
            end[1] = KPOOL * p // 2 + 37
        params[0, 0, :, 0] = end[owner] - (t - 1) + tok
        params[0, 0, :, 1] = owner * KPOOL * p
        params[0, 0, :, 2] = owner * pp
        d_qi = put(mesh, qi, layout=RM, **REP)
        d_hw = put(mesh, hw, **REP)
        d_p = put(mesh, params.to(torch.int32), layout=RM, dtype=ttnn.uint32, **REP)
        run = lambda: pool_scores(mesh, d_qi, cache, d_hw, d_p, p, False, group=(users, t), shard=shard)  # noqa: E731
        compare(f"decode {users}x{t} pools {p} {'sharded' if shard else 'unsharded'} {dt}", run)
        for x in (cache, d_qi, d_hw, d_p):
            ttnn.deallocate(x)

    # prefill: one user's chunk, the queries sharing its pages (32 a core), pools computed to the chunk's end
    prefill = os.environ.get(
        "GLM53_SB_PREFILL",
        "512@4096:8192:bfp4,128@30000:32768:bfp4,37@1000:2048:bf16,37@5000:8192:bfp4,1@3000:8192:bfp4,"
        "2@3000:8192:bfp4,3@3000:8192:bfp4",
    )
    for spec in [s for s in prefill.split(",") if s]:
        qs, p, dt = spec.split(":")
        t, start = (int(v) for v in qs.split("@"))
        p = int(p)
        pp = p // 32
        cache = synthetic(mesh, [2 * pp + 1, 1, 32, DIM], DTYPES[dt])
        q = (torch.randn(t, 1, HEADS, DIM, generator=g) * 0.1).to(torch.bfloat16)
        hw = (torch.randn(1, 1, t, HEADS, generator=g) / 64).to(torch.bfloat16)
        params = torch.zeros(1, 1, t, PARAMS, dtype=torch.int64)
        params[0, 0, :, 0] = start + torch.arange(t)
        params[0, 0, :, 2] = pp  # user 1's pages
        d_q, d_hw = put(mesh, q, **REP), put(mesh, hw, **REP)
        d_p = put(mesh, params.to(torch.int32), layout=RM, dtype=ttnn.uint32, **REP)
        computed = (start + t) // KPOOL
        run = lambda: pool_scores(mesh, d_q, cache, d_hw, d_p, p, True, computed)  # noqa: E731
        compare(f"prefill {t} queries at {start}, pools {p} ({computed} computed) {dt}", run)
        for x in (cache, d_q, d_hw, d_p):
            ttnn.deallocate(x)

    print(f"SB RESULT: {'all bit for bit' if bad == 0 else f'{bad} of {cases} DIFFERENT'} ({cases} cases)", flush=True)
