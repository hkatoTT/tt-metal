"""dflash2_ops.kv_write's context rows (mode 1, as dflash2.write_context calls it: K rows [R 16, 128] (r, l, h) and V
from [R, 4096] columns 16 + 2 l + h, 8 layers, 2 heads) by kv_write_runs.cpp (GLM53_DFLASH2_KVW_RUNS=1) against
kv_write.cpp on the same random values: both caches (from the same initial contents) bit for bit on all 32 chips.
The drafter's shapes at 8 / 32 / 128 users (users a mesh row U = 1 / 4 / 16, T = 8 / 8 / 4 tokens a user, R = U T),
each user's tokens at consecutive positions from a start (one within a page, one crossing a face, one crossing a page,
one past the pages, the rest random), the first a(u) of them written (the rest -1, some users none), the ring and the
plain caches, the caches' formats GLM53_KVW_RUNS_DTYPES (default bfp8,bf16); then GLM53_KVW_RUNS_REPEAT more calls
against the first, and both forms traced and timed (bfp8, the plain caches).

    GLM53_KVW_RUNS_U        users a mesh row (default 1,4,16)
    GLM53_KVW_RUNS_TIME=0   no timing (e.g. under the watcher)
    GLM53_KVW_RUNS_REPEAT   more calls compared with the first (default 10)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kv_write_runs.py -s

Prints "KVW_RUNS U .. .. ring ..: K the same / V the same" lines and "KVW_RUNS ALL SAME" when all held.
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

HEADS, LAYERS, HD = 2, 8, 128
TOKENS = {1: 8, 4: 8, 16: 4}  # T a user
PAGES = 4  # a (user, head)'s pages: 128 positions
REPS = 10
DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def bits(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float().contiguous().view(torch.int32) for d in ttnn.get_device_tensors(t)]


def same(a, b) -> bool:
    xs, ys = bits(a), bits(b)
    return len(xs) == len(ys) and all(torch.equal(x, y) for x, y in zip(xs, ys))


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kv_write_runs(mesh_device):
    mesh = mesh_device
    rep = mapper(mesh, (None, None))
    timing = os.environ.get("GLM53_KVW_RUNS_TIME", "1") != "0"
    repeat = int(os.environ.get("GLM53_KVW_RUNS_REPEAT", "10"))
    g = torch.Generator().manual_seed(1122)

    def put(t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=rep)

    ok = True
    for dname in os.environ.get("GLM53_KVW_RUNS_DTYPES", "bfp8,bf16").split(","):
        dt = DTYPES[dname]
        for u in [int(v) for v in os.environ.get("GLM53_KVW_RUNS_U", "1,4,16").split(",")]:
            t_u = TOKENS[u]
            r = u * t_u
            heads = LAYERS * HEADS
            layer_pages = u * HEADS * PAGES
            kr = put(torch.randn(1, 1, r * heads, HD, generator=g), dt)
            kv = put(torch.randn(1, 1, r, 2 * heads * HD, generator=g), dt)
            for ring in (False, True):
                span = PAGES * 32 * (2 if ring else 1)
                fixed = [5, 16 - t_u // 2, 32 - t_u // 2, PAGES * 32 - t_u // 2]
                pos = torch.full((r,), -1, dtype=torch.int32)
                usr = torch.zeros(r, dtype=torch.int32)
                for v in range(u):
                    start = fixed[v] if v < len(fixed) else int(torch.randint(0, span - t_u, (1,), generator=g))
                    acc = t_u if v < len(fixed) else int(torch.randint(0, t_u + 1, (1,), generator=g))
                    for j in range(t_u):
                        usr[v * t_u + j] = v
                        if j < acc:
                            pos[v * t_u + j] = start + j
                idx = put(pos.view(1, r), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                users = put(usr.view(1, r), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                cache0 = torch.randn(LAYERS * layer_pages + 1, 1, 32, 128, generator=g)
                caches = {name: [put(cache0, dt) for _ in range(2)] for name in ("k", "v")}

                def write(runs: bool, kc, vc):
                    os.environ["GLM53_DFLASH2_KVW_RUNS"] = "1" if runs else "0"
                    writes = [(kr, kc, (heads, 0, HEADS, 1), (0, 0, 0)), (kv, vc, (1, 0, 0, 0), (heads, HEADS, 1))]
                    ops.kv_write(mesh, writes, idx, users, ring=ring, tokens=r, block=1, layers=LAYERS, heads=HEADS,
                                 n_users=u, pages=PAGES, layer_pages=layer_pages)  # fmt: skip

                write(False, caches["k"][0], caches["v"][0])
                write(True, caches["k"][1], caches["v"][1])
                res = [same(caches["k"][0], caches["k"][1]), same(caches["v"][0], caches["v"][1])]
                print(
                    f"KVW_RUNS U {u} T {t_u} {dname} ring {int(ring)}: K the {'same' if res[0] else 'DIFFER'}, V the "
                    f"{'same' if res[1] else 'DIFFER'} on {len(ttnn.get_device_tensors(kr))} chips",
                    flush=True,
                )
                ok = ok and all(res)
                again = 0
                for _ in range(repeat):  # (the same writes again: the caches must stay as they are)
                    write(True, caches["k"][1], caches["v"][1])
                    again += int(same(caches["k"][0], caches["k"][1]) and same(caches["v"][0], caches["v"][1]))
                if repeat:
                    print(
                        f"KVW_RUNS U {u} {dname} ring {int(ring)}: {again} of {repeat} more calls the same", flush=True
                    )
                ok = ok and again == repeat
                if timing and not ring and dname == "bfp8":
                    # GLM53_KVW_RUNS_FORMS: the runs forms timed (and compared), "LG:TPU" (K's layers a unit, the
                    # column tiles a unit); "" = the default only
                    forms = [f for f in os.environ.get("GLM53_KVW_RUNS_FORMS", "").split(",") if f] or [""]

                    def timed_write(runs, kc, vc):
                        write(runs, kc, vc)
                        ttnn.synchronize_device(mesh)
                        with trace_capture(mesh) as trace:
                            for _ in range(REPS):
                                write(runs, kc, vc)
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        t0 = time.perf_counter()
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        ttnn.release_trace(mesh, trace)
                        return (time.perf_counter() - t0) / REPS

                    t_old = timed_write(False, caches["k"][0], caches["v"][0])
                    for form in forms:
                        lg, tpu = form.split(":") if form else (None, None)
                        for key, val in (("GLM53_DFLASH2_KVW_LG", lg), ("GLM53_DFLASH2_KVW_TPU", tpu)):
                            if val is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = val
                        t_new = timed_write(True, caches["k"][1], caches["v"][1])
                        res = same(caches["k"][0], caches["k"][1]) and same(caches["v"][0], caches["v"][1])
                        ok = ok and res
                        print(
                            f"KVW_RUNS TIME U {u}{' form ' + form if form else ''}: kv_write.cpp {t_old * 1e6:.1f} -> "
                            f"kv_write_runs.cpp {t_new * 1e6:.1f} us ({(t_new - t_old) * 1e6:+.1f}); the caches "
                            f"{'the same' if res else 'DIFFER'}",
                            flush=True,
                        )
                    os.environ.pop("GLM53_DFLASH2_KVW_LG", None)
                    os.environ.pop("GLM53_DFLASH2_KVW_TPU", None)
                for t in caches["k"] + caches["v"] + [idx, users]:
                    ttnn.deallocate(t)
            ttnn.deallocate(kr)
            ttnn.deallocate(kv)
    os.environ.pop("GLM53_DFLASH2_KVW_RUNS", None)
    print(f"KVW_RUNS {'ALL SAME' if ok else 'NOT ALL SAME'}", flush=True)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_heads_split_ahead(mesh_device):
    """dflash2_ops.heads_split (q and k as one tensor, as dflash2._attention calls it) with GLM53_DFLASH2_HEADS_AHEAD=1
    (the default) against =0 (the earlier form) on the same random qkv: the tiles bit for bit on all 32 chips at the
    drafter's shapes (U = 1 / 4 / 16, B = 8 / 8 / 4), then both forms traced and timed (GLM53_KVW_RUNS_TIME=0: no
    timing).  Prints "HEADS_AHEAD U ..: the same" lines and "HEADS_AHEAD ALL SAME"."""
    mesh = mesh_device
    rep = mapper(mesh, (None, None))
    timing = os.environ.get("GLM53_KVW_RUNS_TIME", "1") != "0"
    g = torch.Generator().manual_seed(1123)
    group, kvh, k0 = 4, 2, 8
    ok = True
    for u in [int(v) for v in os.environ.get("GLM53_KVW_RUNS_U", "1,4,16").split(",")]:
        b = {1: 8, 4: 8, 16: 4}[u]
        qkv = ttnn.from_torch(torch.randn(1, 1, u * b, 1536, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=mesh, mesh_mapper=rep)  # fmt: skip

        def split(ahead: bool):
            os.environ["GLM53_DFLASH2_HEADS_AHEAD"] = "1" if ahead else "0"
            return ops.heads_split(mesh, qkv, u, b, group, kvh, k0, combined=True)

        a, c = split(False), split(True)
        res = same(a, c)
        print(f"HEADS_AHEAD U {u} B {b}: the {'same' if res else 'DIFFER'} on {len(ttnn.get_device_tensors(a))} chips",
              flush=True)  # fmt: skip
        ok = ok and res
        ttnn.deallocate(a)
        ttnn.deallocate(c)
        if timing:
            times = []
            for ahead in (False, True):
                ttnn.deallocate(split(ahead))
                ttnn.synchronize_device(mesh)
                with trace_capture(mesh) as trace:
                    outs = [split(ahead) for _ in range(REPS)]
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                times.append((time.perf_counter() - t0) / REPS)
                ttnn.release_trace(mesh, trace)
                for o in outs:
                    ttnn.deallocate(o)
            print(f"HEADS_AHEAD TIME U {u}: {times[0] * 1e6:.1f} -> {times[1] * 1e6:.1f} us ({(times[1] - times[0]) * 1e6:+.1f})",
                  flush=True)  # fmt: skip
        ttnn.deallocate(qkv)
    os.environ.pop("GLM53_DFLASH2_HEADS_AHEAD", None)
    print(f"HEADS_AHEAD {'ALL SAME' if ok else 'NOT ALL SAME'}", flush=True)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_drafter_writes_stress(mesh_device):
    """A stress of the drafter's two run-wise writes: kv_write_runs.cpp (GLM53_DFLASH2_KVW_RUNS=1: the context rows, 8
    layers) and rope_kv with its runs (GLM53_DFLASH2_ROPE_KV_RUNS=1), each traced 10 calls a trace and replayed
    GLM53_KVW_STRESS times (default 200: 2,000 calls), the caches (and rope_kv's q) then bit for bit against the earlier
    forms' (=0) one call from the same initial contents, at U = GLM53_KVW_RUNS_U (default 4,16), BFP8, the ring caches
    as the drafter has them.  The writes are the same each call, so a race shows as a byte that differs.  Prints
    "KVW_STRESS .." lines and "KVW_STRESS ALL SAME".
    """
    mesh = mesh_device
    rep = mapper(mesh, (None, None))
    reps = int(os.environ.get("GLM53_KVW_STRESS", "200"))
    g = torch.Generator().manual_seed(1125)

    def put(t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=rep)

    def replay(fn):
        fn()
        ttnn.synchronize_device(mesh)
        with trace_capture(mesh) as trace:
            for _ in range(10):
                fn()
        for _ in range(reps):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ttnn.release_trace(mesh, trace)

    ok = True
    for u in [int(v) for v in os.environ.get("GLM53_KVW_RUNS_U", "4,16").split(",")]:
        # kv_write's context rows
        t_u = TOKENS[u]
        r = u * t_u
        heads = LAYERS * HEADS
        layer_pages = u * HEADS * PAGES
        kr = put(torch.randn(1, 1, r * heads, HD, generator=g), ttnn.bfloat8_b)
        kv = put(torch.randn(1, 1, r, 2 * heads * HD, generator=g), ttnn.bfloat8_b)
        pos = torch.full((r,), -1, dtype=torch.int32)
        usr = torch.zeros(r, dtype=torch.int32)
        for v in range(u):
            start = int(torch.randint(0, 2 * PAGES * 32 - t_u, (1,), generator=g))
            acc = t_u if v % 3 else int(torch.randint(0, t_u + 1, (1,), generator=g))
            for j in range(t_u):
                usr[v * t_u + j] = v
                if j < acc:
                    pos[v * t_u + j] = start + j
        idx = put(pos.view(1, r), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        users = put(usr.view(1, r), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cache0 = torch.randn(LAYERS * layer_pages + 1, 1, 32, 128, generator=g)
        caches = {name: [put(cache0, ttnn.bfloat8_b) for _ in range(2)] for name in ("k", "v")}

        def write(runs: bool, kc, vc):
            os.environ["GLM53_DFLASH2_KVW_RUNS"] = "1" if runs else "0"
            writes = [(kr, kc, (heads, 0, HEADS, 1), (0, 0, 0)), (kv, vc, (1, 0, 0, 0), (heads, HEADS, 1))]
            ops.kv_write(mesh, writes, idx, users, ring=True, tokens=r, block=1, layers=LAYERS, heads=HEADS,
                         n_users=u, pages=PAGES, layer_pages=layer_pages)  # fmt: skip

        write(False, caches["k"][0], caches["v"][0])
        replay(lambda: write(True, caches["k"][1], caches["v"][1]))
        res = [same(caches["k"][0], caches["k"][1]), same(caches["v"][0], caches["v"][1])]
        print(f"KVW_STRESS kv_write U {u}: {10 * reps} calls, K the {'same' if res[0] else 'DIFFER'}, V the "
              f"{'same' if res[1] else 'DIFFER'}", flush=True)  # fmt: skip
        ok = ok and all(res)
        for t in caches["k"] + caches["v"] + [idx, users, kr, kv]:
            ttnn.deallocate(t)
        # rope_kv with its runs (one layer's block, as dflash2._attention)
        b = {1: 8, 4: 8, 16: 4}[u]
        hq, m, vu = 8, u * b, u * HEADS
        v0 = (hq + HEADS) * HD // 128
        n = put(torch.randn(1, 2 * vu, 32, 128, generator=g), ttnn.bfloat16)
        w = put(torch.randn(1, 1, 64, 128, generator=g), ttnn.bfloat16)
        cos = put(torch.rand(1, vu, 32, 128, generator=g) * 2 - 1, ttnn.float32)
        sin = put(torch.rand(1, vu, 32, 128, generator=g) * 2 - 1, ttnn.float32)
        qkv = put(torch.randn(1, 1, m, (hq + 2 * HEADS) * HD, generator=g), ttnn.bfloat16)
        starts = put(torch.tensor([[(37 * i + 5) % (2 * PAGES * 32) for i in range(u)]], dtype=torch.int32), ttnn.int32,
                     ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        lp = u * HEADS * PAGES
        c0 = torch.randn(2 * lp, 1, 32, 128, generator=g)
        rc = {name: [put(c0, ttnn.bfloat8_b) for _ in range(2)] for name in ("k", "v")}
        qs = []

        def rope(runs: bool, kc, vc):
            os.environ["GLM53_DFLASH2_ROPE_KV_RUNS"] = "1" if runs else "0"
            q = ops.rope_kv(mesh, n, w, cos, sin, vu, qkv, v0, kc, vc, starts, block=b, heads=HEADS, n_users=u,
                            pages=PAGES, page_base=lp, ring=True)  # fmt: skip
            qs.append(q)
            return q

        q0 = rope(False, rc["k"][0], rc["v"][0])
        qs.clear()
        replay(lambda: rope(True, rc["k"][1], rc["v"][1]))
        res = [same(q0, qs[-1]), same(rc["k"][0], rc["k"][1]), same(rc["v"][0], rc["v"][1])]
        print(f"KVW_STRESS rope_kv U {u}: {10 * reps} calls, q the {'same' if res[0] else 'DIFFER'}, K the "
              f"{'same' if res[1] else 'DIFFER'}, V the {'same' if res[2] else 'DIFFER'}", flush=True)  # fmt: skip
        ok = ok and all(res)
        for t in qs + [q0, n, w, cos, sin, qkv, starts] + rc["k"] + rc["v"]:
            ttnn.deallocate(t)
    for k in ("GLM53_DFLASH2_KVW_RUNS", "GLM53_DFLASH2_ROPE_KV_RUNS"):
        os.environ.pop(k, None)
    print(f"KVW_STRESS {'ALL SAME' if ok else 'NOT ALL SAME'}", flush=True)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_ctx_heads(mesh_device):
    """write_context's K (GLM53_DFLASH2_CTX_HEADS): the earlier chain (=0, `today` below: slice, untilize, reshape,
    tilize, rope_rows of 16 R rows, rms_norm, rope with a weight a row) against the head-major one (=1, the default,
    `new`: heads_major, rope_rows of R rows, rms_norm, rope with cos_period / w_div) on the same random kv, positions
    and tables: every row (r, h) of the earlier chain's the same bits as row (h, r) of the head-major one's, on all 32
    chips, at R = 32 / 64 (32 / 128 users); then both chains traced and timed (GLM53_KVW_RUNS_TIME=0: no timing).
    Prints "CTX_HEADS R ..: the same" and "CTX_HEADS ALL SAME"."""
    mesh = mesh_device
    rep = mapper(mesh, (None, None))
    timing = os.environ.get("GLM53_KVW_RUNS_TIME", "1") != "0"
    g = torch.Generator().manual_seed(1126)
    heads, eps = LAYERS * HEADS, 1e-6
    from models.demos.blackhole.glm53_flash.ttnn.common import HIFI4_FP32 as hifi4  # (the drafter's norm config)

    tab = ttnn.from_torch(torch.randn(512, 128, generator=g), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=mesh, mesh_mapper=rep)  # fmt: skip
    norm = torch.randn(heads, 128, generator=g).bfloat16()
    ok = True
    for r in [int(v) for v in os.environ.get("GLM53_CTX_HEADS_R", "32,64").split(",")]:
        kv = ttnn.from_torch(torch.randn(1, 1, r, 2 * heads * HD, generator=g), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)  # fmt: skip
        pos = torch.randint(0, 512, (1, r), generator=g, dtype=torch.int32)
        idx = ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        w_rows = ttnn.from_torch(norm.repeat(r, 1).reshape(1, 1, -1, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, mesh_mapper=rep)  # fmt: skip
        w_heads = ttnn.from_torch(norm.repeat_interleave(32, dim=0).reshape(1, 1, -1, 128), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)  # fmt: skip

        def today():
            k = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, r, heads * HD])
            krm = ttnn.to_layout(k, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(k)
            kt = ttnn.tilize_with_zero_padding(ttnn.reshape(krm, [1, 1, r * heads, HD]))
            ttnn.deallocate(krm)
            cos, sin = ops.rope_rows(mesh, idx, tab, None, r * heads, heads)
            n = ttnn.rms_norm(kt, epsilon=eps, compute_kernel_config=hifi4)
            out = ops.rope(mesh, n, w_rows, cos, sin)
            for t in (kt, cos, sin, n):
                ttnn.deallocate(t)
            return out

        def new():
            kt = ops.heads_major(mesh, kv, heads)
            cos, sin = ops.rope_rows(mesh, idx, tab, None, r, 1)
            n = ttnn.rms_norm(kt, epsilon=eps, compute_kernel_config=hifi4)
            out = ops.rope(mesh, n, w_heads, cos, sin, cos_period=r // 32, w_div=r // 32)
            for t in (kt, cos, sin, n):
                ttnn.deallocate(t)
            return out

        a, b = today(), new()
        xs = [ttnn.to_torch(d).view(r, heads, HD) for d in ttnn.get_device_tensors(a)]
        ys = [ttnn.to_torch(d).view(heads, r, HD).transpose(0, 1) for d in ttnn.get_device_tensors(b)]
        res = len(xs) == len(ys) and all(torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))
                                         for x, y in zip(xs, ys))  # fmt: skip
        print(f"CTX_HEADS R {r}: the {'same' if res else 'DIFFER'} on {len(ys)} chips", flush=True)
        ok = ok and res
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        if timing:
            times = []
            for fn in (today, new):
                ttnn.deallocate(fn())
                ttnn.synchronize_device(mesh)
                with trace_capture(mesh) as trace:
                    outs = [fn() for _ in range(REPS)]
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                times.append((time.perf_counter() - t0) / REPS)
                ttnn.release_trace(mesh, trace)
                for o in outs:
                    ttnn.deallocate(o)
            print(f"CTX_HEADS TIME R {r}: {times[0] * 1e6:.1f} -> {times[1] * 1e6:.1f} us ({(times[1] - times[0]) * 1e6:+.1f})",
                  flush=True)  # fmt: skip
        for t in (kv, idx, w_rows, w_heads):
            ttnn.deallocate(t)
    print(f"CTX_HEADS {'ALL SAME' if ok else 'NOT ALL SAME'}", flush=True)
