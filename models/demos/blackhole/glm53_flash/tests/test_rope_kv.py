"""dflash2_ops.rope_kv (the drafter's RoPE with the block's K / V into the BFP8_b caches in one op,
GLM53_DFLASH2_ROPE_KV, the default) against the separate ops (=0, `today` below) on the same random values: rope
(halves), bfp_cast of k and of qkv, kv_write of K and V (mode 0, as dflash2._attention calls them); q and both
caches (from the same initial contents) bit for bit on all 32 chips.  The drafter's shapes at 8 / 32 / 128 users
(users a mesh row U = 1 / 4 / 16, blocks B = 8 / 8 / 4), the ring and the plain caches, starts that cross a page, run
past the pages (skipped without the ring) and one below zero (skipped); then GLM53_ROPE_KV_REPEAT more calls against
the first, and both forms traced and timed.

    GLM53_ROPE_KV_U        users a mesh row (default 1,4,16)
    GLM53_ROPE_KV_TIME=0   no timing (e.g. under the watcher)
    GLM53_ROPE_KV_REPEAT   more calls compared with the first (default 10)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_rope_kv.py -s

Prints "ROPE_KV U .. ring ..: q the same / K the same / V the same" lines and "ROPE_KV ALL SAME" when all held.
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
from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import bfp_cast  # noqa: E402

HEADS, HQ, HD = 2, 8, 128  # K / V heads, q heads a chip, head dim
V0 = (HQ + HEADS) * HD // 128  # v's first 128-column block in qkv
BLOCK = {1: 8, 4: 8, 16: 4}
PAGES = 4  # a (user, head)'s pages: 128 positions
REPS = 10


def bits(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float().contiguous().view(torch.int32) for d in ttnn.get_device_tensors(t)]


def same(a, b) -> bool:
    xs, ys = bits(a), bits(b)
    return len(xs) == len(ys) and all(torch.equal(x, y) for x, y in zip(xs, ys))


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_rope_kv(mesh_device):
    mesh = mesh_device
    rep = mapper(mesh, (None, None))
    timing = os.environ.get("GLM53_ROPE_KV_TIME", "1") != "0"
    repeat = int(os.environ.get("GLM53_ROPE_KV_REPEAT", "10"))
    g = torch.Generator().manual_seed(1822)

    def put(t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=rep)

    ok = True
    for u in [int(v) for v in os.environ.get("GLM53_ROPE_KV_U", "1,4,16").split(",")]:
        b = BLOCK[u]
        m, vu = u * b, u * HEADS
        layer_pages = u * HEADS * PAGES
        page_base = layer_pages  # (the second layer's pages)
        n = put(torch.randn(1, 2 * vu, 32, 128, generator=g), ttnn.bfloat16)
        w = put(torch.randn(1, 1, 64, 128, generator=g), ttnn.bfloat16)
        cos = put(torch.rand(1, vu, 32, 128, generator=g) * 2 - 1, ttnn.float32)
        sin = put(torch.rand(1, vu, 32, 128, generator=g) * 2 - 1, ttnn.float32)
        qkv = put(torch.randn(1, 1, m, (HQ + 2 * HEADS) * HD, generator=g), ttnn.bfloat16)
        # starts: one within a page, one crossing a page, one past the pages (skipped without the ring), one < 0
        choice = [5, 32 - b // 2, PAGES * 32 - b // 2, -1]
        starts_host = torch.tensor([[choice[i % len(choice)] for i in range(u)]], dtype=torch.int32)
        starts = put(starts_host, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cache0 = torch.randn(2 * layer_pages, 1, 32, 128, generator=g)
        for ring in (False, True):
            caches = {name: [put(cache0, ttnn.bfloat8_b) for _ in range(2)] for name in ("k", "v")}

            def today(kc, vc):
                q, k = ops.rope(mesh, n, w, cos, sin, halves=vu)
                k8, qkv8 = bfp_cast(mesh, k, ttnn.bfloat8_b), bfp_cast(mesh, qkv, ttnn.bfloat8_b)
                ttnn.deallocate(k)
                writes = [(k8, kc, (2 * 32, 1, 0, 32), (0, 0, 0)), (qkv8, vc, (b, 1, 0, 0), (V0, 0, 1))]
                ops.kv_write(mesh, writes, starts, None, ring=ring, tokens=m, block=b, layers=1, heads=HEADS,
                             n_users=u, pages=PAGES, layer_pages=layer_pages, page_base=page_base)  # fmt: skip
                ttnn.deallocate(k8)
                ttnn.deallocate(qkv8)
                return q

            def fused(kc, vc):
                return ops.rope_kv(mesh, n, w, cos, sin, vu, qkv, V0, kc, vc, starts, block=b, heads=HEADS,
                                   n_users=u, pages=PAGES, page_base=page_base, ring=ring)  # fmt: skip

            q0 = today(caches["k"][0], caches["v"][0])
            q1 = fused(caches["k"][1], caches["v"][1])
            res = [same(q0, q1), same(caches["k"][0], caches["k"][1]), same(caches["v"][0], caches["v"][1])]
            print(
                f"ROPE_KV U {u} B {b} ring {int(ring)}: q the {'same' if res[0] else 'DIFFER'}, K the "
                f"{'same' if res[1] else 'DIFFER'}, V the {'same' if res[2] else 'DIFFER'} on "
                f"{len(ttnn.get_device_tensors(q1))} chips",
                flush=True,
            )
            ok = ok and all(res)
            again = 0
            for _ in range(repeat):  # (the same writes again: the caches must stay as they are)
                q2 = fused(caches["k"][1], caches["v"][1])
                kv = same(caches["k"][0], caches["k"][1]) and same(caches["v"][0], caches["v"][1])
                again += int(same(q2, q1) and kv)
                ttnn.deallocate(q2)
            if repeat:
                print(f"ROPE_KV U {u} ring {int(ring)}: {again} of {repeat} more calls the same", flush=True)
            ok = ok and again == repeat
            ttnn.deallocate(q0)
            ttnn.deallocate(q1)
            if timing and not ring:
                times = []
                for fn, kc, vc in ((today, caches["k"][0], caches["v"][0]), (fused, caches["k"][1], caches["v"][1])):
                    ttnn.deallocate(fn(kc, vc))
                    ttnn.synchronize_device(mesh)
                    with trace_capture(mesh) as trace:
                        outs = [fn(kc, vc) for _ in range(REPS)]
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    times.append((time.perf_counter() - t0) / REPS)
                    ttnn.release_trace(mesh, trace)
                    for o in outs:
                        ttnn.deallocate(o)
                print(
                    f"ROPE_KV TIME U {u}: rope + 2 bfp_cast + kv_write {times[0] * 1e6:.1f} -> rope_kv "
                    f"{times[1] * 1e6:.1f} us ({(times[1] - times[0]) * 1e6:+.1f})",
                    flush=True,
                )
            for t in caches["k"] + caches["v"]:
                ttnn.deallocate(t)
    print(f"ROPE_KV {'ALL SAME' if ok else 'NOT ALL SAME'}", flush=True)
