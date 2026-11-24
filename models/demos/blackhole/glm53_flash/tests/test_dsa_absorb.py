"""The DSA decode's per-head absorb matmuls by one generic_op over more cores (`dsa_absorb.absorb_matmul`,
GLM53_DSA_ABSORB_OP) against ttnn.matmul with `common.batched_matmul_config` (TtDsa's form), bit for bit on every chip,
and both timed in a trace.  Forms: q~ = q W_kb ([1, 16, B, 256] @ [1, 16, 256, 512], the output in L1 as TtDsa's
heads-first decode) and W_vb^T ([1, 16, B, 512] in L1 @ [1, 16, 512, 256], the output in DRAM), B = 8 / 32 / 64 tokens
(8 / 32 / 128 users' verify steps), the weights BFP8 (the model's) and bf16; the cores a head G from
GLM53_ABS_GROUPS (default "0,1,2": 0 the op's default, then others); GLM53_ABS_FORMS / _ROWS / _DTYPES narrow the
cases; GLM53_ABS_RSPLITS (default "1") the row splits tried too.  Prints only ("ABS ALL SAME" or "ABS DIFFER";
GLM53_ABS_STRICT=1 fails on a difference); a device timeout is raised.  GLM53_ABS_TIME=0: no timing (e.g. under the
watcher).

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_absorb.py -s
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

from models.demos.blackhole.glm53_flash.tests._trace import study  # noqa: E402
from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    WEIGHT_MM,
    batched_matmul_config,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.dsa_absorb import absorb_matmul, groups_for  # noqa: E402

HEADS = 16
FORMS = [("kb", 256, 512, "dram", "l1"), ("vb", 512, 256, "l1", "dram")]  # (name, K, N, a's memory, out's memory)
MEM = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
REP = dict(dims=(None, None))


def timed(mesh, fn, reps: int = 20) -> float:
    out = fn()
    ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    outs = []
    with trace_capture(mesh) as trace:
        for _ in range(reps):
            outs.append(fn())
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return seconds


def chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_absorb(mesh_device):
    mesh = mesh_device
    do_time = os.environ.get("GLM53_ABS_TIME", "1") == "1"
    group_list = [int(x) for x in os.environ.get("GLM53_ABS_GROUPS", "0,1,2").split(",")]
    rows_list = [int(x) for x in os.environ.get("GLM53_ABS_ROWS", "8,32,64").split(",")]
    dtypes = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
    dtype_list = os.environ.get("GLM53_ABS_DTYPES", "bfp8,bf16").split(",")
    form_list = os.environ.get("GLM53_ABS_FORMS", "kb,vb").split(",")
    rs_list = [int(x) for x in os.environ.get("GLM53_ABS_RSPLITS", "1").split(",")]
    cores = mesh.compute_with_storage_grid_size().x * mesh.compute_with_storage_grid_size().y
    bad = []

    def run():
        g = torch.Generator().manual_seed(17)
        for name, k, n, a_mem, o_mem in FORMS:
            if name not in form_list:
                continue
            for dname in dtype_list:
                w_host = (torch.randn(1, HEADS, k, n, generator=g) * k**-0.5).to(torch.bfloat16)
                w = put(mesh, w_host, **REP, dtype=dtypes[dname])
                cfg = batched_matmul_config(mesh, k // 32, n // 32)
                for rows in rows_list:
                    a_host = torch.randn(1, HEADS, rows, k, generator=g).to(torch.bfloat16)
                    a = ttnn.to_memory_config(put(mesh, a_host, **REP), MEM[a_mem])

                    def ref():
                        return ttnn.matmul(
                            a, w, program_config=cfg, compute_kernel_config=WEIGHT_MM, memory_config=MEM[o_mem]
                        )

                    want_t = ref()
                    want = chips(want_t)
                    ttnn.deallocate(want_t)
                    line = []
                    t_ref = timed(mesh, ref) if do_time else None
                    for gsel, rs in [(x, y) for y in rs_list for x in group_list]:
                        gg = gsel or groups_for(HEADS, n // 32, cores)
                        if (n // 32) % gg or HEADS * gg * rs > cores or (rows + 31) // 32 % rs:
                            continue

                        def op(gg=gg, rs=rs):
                            os.environ["GLM53_DSA_ABSORB_RSPLIT"] = str(rs)
                            try:
                                return absorb_matmul(mesh, a, w, memory_config=MEM[o_mem], groups=gg)
                            finally:
                                os.environ.pop("GLM53_DSA_ABSORB_RSPLIT", None)

                        same = True
                        for rep in range(3):  # (the first call compiles; three calls, each against ttnn's)
                            got_t = op()
                            got = chips(got_t)
                            ttnn.deallocate(got_t)
                            for c, (x, y) in enumerate(zip(got, want)):
                                if not torch.equal(x, y):
                                    same = False
                                    d = (x.float() - y.float()).abs()
                                    print(
                                        f"  {name} {dname} B {rows} G {gg} R {rs} call {rep}: chip {c} differs at "
                                        f"{int((d > 0).sum())} values, max {float(d.max()):.3g}",
                                        flush=True,
                                    )
                                    break
                        t_op = timed(mesh, op) if do_time else None
                        if not same:
                            bad.append((name, dname, rows, gg, rs))
                        tm = f"; traced {t_ref * 1e6:6.1f} -> {t_op * 1e6:6.1f} us" if do_time else ""
                        line.append(f"G {gg}" + (f" R {rs}" if rs > 1 else "") + f": same {same}{tm}")
                    pcc = torch.corrcoef(
                        torch.stack([want[0].float().reshape(-1), (a_host.float() @ w_host.float()).reshape(-1)])
                    )[0, 1]
                    print(
                        f"ABS {name} {dname} B {rows:3d} (a {a_mem}, out {o_mem}; ttnn against the host pcc "
                        f"{float(pcc):.5f}): " + " | ".join(line),
                        flush=True,
                    )
                    ttnn.deallocate(a)
                ttnn.deallocate(w)
        print("ABS ALL SAME" if not bad else f"ABS DIFFER {bad}", flush=True)

    study(run, strict_env="GLM53_ABS_STRICT")


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_latent_absorb(mesh_device):
    """`dsa_absorb.latent_absorb` (GLM53_DSA_ABSORB_OP=2: the latent write and q W_kb in one program) against
    `dsa_latent.latent_write` then `absorb_matmul`: the whole cache and q~ bit for bit on every chip, both timed in a
    trace.  Cases (tokens, owners): DFlash2's verify steps at 8 / 32 / 128 users (1 x 8, 4 x 8, 16 x 4 a row), some
    positions -1.  Prints only ("LAB ALL SAME" or "LAB DIFFER")."""
    from models.demos.blackhole.glm53_flash.ttnn.dsa_absorb import latent_absorb
    from models.demos.blackhole.glm53_flash.ttnn.dsa_latent import latent_write

    mesh = mesh_device
    do_time = os.environ.get("GLM53_ABS_TIME", "1") == "1"
    rank, page, pages = 512, 32, 16
    bad = []

    def run():
        g = torch.Generator().manual_seed(23)
        w_host = (torch.randn(1, HEADS, 256, rank, generator=g) * 256**-0.5).to(torch.bfloat16)
        w = put(mesh, w_host, **REP, dtype=ttnn.bfloat8_b)
        for tokens, owners in ((8, 1), (32, 4), (64, 16)):
            k = tokens // owners
            nrows = tuple(mesh.shape)[0]  # (8 on the Galaxy; 1 on a one-row mesh)
            first = torch.randint(0, pages * page - k, (nrows, owners), generator=g)
            pos = torch.cat([first + j for j in range(k)], dim=1)
            pos[torch.rand(nrows, tokens, generator=g) < 0.15] = -1
            cur = put(mesh, pos.to(torch.int32).reshape(-1), dims=(0, None), dtype=ttnn.int32,
                      layout=ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
            cache0 = torch.randn(owners * pages + 1, 1, page, rank, generator=g).to(torch.bfloat16)
            lat = put(mesh, torch.randn(1, 1, tokens, rank, generator=g).to(torch.bfloat16), **REP)
            q = put(mesh, torch.randn(1, HEADS, tokens, 256, generator=g).to(torch.bfloat16), **REP)
            caches = {n: put(mesh, cache0, **REP) for n in ("two", "one")}

            def two():
                latent_write(mesh, lat, caches["two"], cur, owners, pages)
                return absorb_matmul(mesh, q, w, memory_config=ttnn.L1_MEMORY_CONFIG)

            def one():
                return latent_absorb(mesh, lat, caches["one"], cur, owners, pages, q, w,
                                     memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip

            a, b = two(), one()
            same_q = all(torch.equal(x, y) for x, y in zip(chips(a), chips(b)))
            same_c = all(torch.equal(x, y) for x, y in zip(chips(caches["two"]), chips(caches["one"])))
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            tm = ""
            if do_time:
                t2, t1 = timed(mesh, two), timed(mesh, one)
                tm = f"; traced {t2 * 1e6:6.1f} -> {t1 * 1e6:6.1f} us"
            print(f"LAB tokens {tokens:3d} owners {owners:3d}: q~ same {same_q}, cache same {same_c}{tm}", flush=True)
            if not (same_q and same_c):
                bad.append((tokens, owners))
            for t in (cur, lat, q, *caches.values()):
                ttnn.deallocate(t)
        ttnn.deallocate(w)
        print("LAB ALL SAME" if not bad else f"LAB DIFFER {bad}", flush=True)

    study(run, strict_env="GLM53_ABS_STRICT")
