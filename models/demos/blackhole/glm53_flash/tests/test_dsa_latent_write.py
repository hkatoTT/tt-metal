"""The DSA decode's latents into the paged latent cache by one op (`dsa_latent.latent_write`) against the form it
replaced (the old `TtDsa._write_latents`: the reshape to [1, B, 1, 512] and, per group of 16
users, slice, height-sharding and paged_update_cache), bit for bit on the whole cache of two chips of different mesh
rows (each row its own positions), and both timed in a trace.  Cases (tokens B, owners U): MTP's verify step (token v
on user v mod U's pages, the draft one row after the token: neighbouring rows of a page, both parities) at 1 / 4 / 16
users a row, the plain step at 2 / 16 / 64; some positions -1 (no token), some at a page's last row.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_latent_write.py -s
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
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS, put  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dsa_latent import latent_write, q_heads  # noqa: E402

RANK, PAGE, PAGES, GROUP = 512, 32, 16, 16
# (and DFlash2's verify steps: T = 8 at 1 / 4 users a row, T = 4 at 16)
CASES = [(2, 1), (8, 4), (32, 16), (2, 2), (16, 16), (64, 64), (8, 1), (32, 4), (64, 16)]
REP = dict(dims=(None, None))


def groups(batch: int, split: int | None) -> list[tuple[int, int]]:
    """As the old `TtDsa.groups`: users of each paged_update_cache call, none crossing a multiple of split."""
    if split:
        rest = batch - split
        later = groups(rest, split if rest > split else None)
        return groups(split, None) + [(split + u0, split + u1) for u0, u1 in later]
    return [(u, min(u + GROUP, batch)) for u in range(0, batch, GROUP)]


def per_row(mesh, values: torch.Tensor):
    """[ROWS, n] int32 -> each mesh row's n values (row-major, as `TtDsa.alloc_positions`)."""
    return put(mesh, values.to(torch.int32).reshape(-1), dims=(0, None), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def table(mesh, rows: torch.Tensor):
    return put(mesh, rows.to(torch.int32), **REP, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def traced(mesh, fn, reps: int = 20) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    with trace_capture(mesh) as trace:
        for _ in range(reps):
            fn()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    return seconds


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_latent_write(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(5)
    chips = [0, 5 * 4 + 2]  # (row 0, column 0) and (row 5, column 2)
    failures = []
    for tokens, owners in CASES:
        split = owners if owners < tokens else None
        k = tokens // owners
        # positions: per mesh row, each owner's first token at p, its next ones at p + 1, p + 2 (MTP's drafts)
        first = torch.randint(0, PAGES * PAGE - k, (ROWS, owners), generator=g)
        first[:, 0] = PAGE - 1  # a page's last row (a draft then on the next page)
        pos = torch.cat([first + j for j in range(k)], dim=1)
        pos[torch.rand(ROWS, tokens, generator=g) < 0.15] = -1
        cache0 = torch.randn(owners * PAGES + 1, 1, PAGE, RANK, generator=g).to(torch.bfloat16)
        lat = torch.randn(1, 1, tokens, RANK, generator=g).to(torch.bfloat16)

        d_lat = put(mesh, lat, **REP)
        d_pos = per_row(mesh, pos)
        pages = (torch.arange(tokens) % owners)[:, None] * PAGES + torch.arange(PAGES)
        parts = [(per_row(mesh, pos[:, u0:u1]), table(mesh, pages[u0:u1])) for u0, u1 in groups(tokens, split)]
        caches = {name: put(mesh, cache0, **REP) for name in ("op", "ref")}

        def op():
            latent_write(mesh, d_lat, caches["op"], d_pos, owners, PAGES)

        def ref():
            lat4 = ttnn.reshape(d_lat, [1, tokens, 1, RANK])
            for (u0, u1), (cur, tab) in zip(groups(tokens, split), parts):
                new = lat4 if u1 - u0 == tokens else ttnn.slice(lat4, [0, u0, 0, 0], [1, u1, 1, RANK])
                grid = ttnn.num_cores_to_corerangeset(u1 - u0, mesh.compute_with_storage_grid_size(), row_wise=True)
                spec = ttnn.ShardSpec(grid, [32, RANK], ttnn.ShardOrientation.ROW_MAJOR)
                sharded = ttnn.to_memory_config(
                    new, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
                )
                ttnn.experimental.paged_update_cache(caches["ref"], sharded, update_idxs_tensor=cur, page_table=tab)
                ttnn.deallocate(sharded)

        op()
        ref()
        got = [ttnn.to_torch(ttnn.get_device_tensors(caches["op"])[c]) for c in chips]
        want = [ttnn.to_torch(ttnn.get_device_tensors(caches["ref"])[c]) for c in chips]
        # the expected cache on the host too (row r of mesh: chip index // 4)
        exact = []
        for c, w in zip(chips, want):
            host = cache0.clone()
            for v in range(tokens):
                p = int(pos[c // 4, v])
                if p >= 0:
                    host[(v % owners) * PAGES + p // PAGE, 0, p % PAGE] = lat[0, 0, v]
            exact.append(torch.equal(host, w))
        same = all(torch.equal(a, b) for a, b in zip(got, want))
        t_op, t_ref = traced(mesh, op), traced(mesh, ref)
        print(
            f"B {tokens:3d}, owners {owners:3d}: op = paged_update_cache form bit for bit {same} (that form = host "
            f"{all(exact)}); traced {t_ref * 1e6:6.1f} -> {t_op * 1e6:6.1f} us",
            flush=True,
        )
        if not same:
            diff = (got[0].float() - want[0].float()).abs().reshape(-1, PAGE, RANK).amax(dim=-1)
            bad = torch.nonzero(diff)[:8].tolist()
            print(f"  first differing (page, row) on chip {chips[0]}: {bad}", flush=True)
            failures.append((tokens, owners))
        for t in [d_lat, d_pos, *caches.values()] + [x for pair in parts for x in pair]:
            ttnn.deallocate(t)
    assert not failures, failures


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_q_heads(mesh_device):
    """`dsa_latent.q_heads` (q [1, 1, B, 16 x 256] -> [1, 16 R, B, 256]) against the forms it replaces: up to 16 tokens
    the reshape to [1, B, 16, 256] and the transpose, past it nlp_create_qkv_heads; bit for bit on every row of the
    tokens (R = 2: each copy against its head), and both timed in a trace.  Prints only (GLM53_QH_STRICT=1: fails on a
    difference)."""
    mesh = mesh_device
    g = torch.Generator().manual_seed(9)
    heads, d = 16, 256
    failures = []
    for rows in (2, 8, 16, 32, 64):
        q_host = torch.randn(1, 1, rows, heads * d, generator=g).to(torch.bfloat16)
        lat_host = torch.randn(1, 1, rows, 512, generator=g).to(torch.bfloat16)
        dq, dl = put(mesh, q_host, **REP), put(mesh, lat_host, **REP)

        def ref():
            if rows > 16:
                h, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    dq, dl, num_heads=heads, num_kv_heads=1, transpose_k_heads=False
                )
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                return h
            r = ttnn.reshape(dq, [1, rows, heads, d])
            return ttnn.transpose(r, 1, 2)

        want = ttnn.to_torch(ttnn.get_device_tensors(ref())[0]).reshape(1, heads, -1, d)[:, :, :rows]
        same = {}
        for copies in (1, 2):
            got = ttnn.to_torch(ttnn.get_device_tensors(q_heads(mesh, dq, heads, copies))[0])
            got = got.reshape(1, heads * copies, -1, d)[:, :, :rows]
            same[copies] = all(
                torch.equal(got[:, h * copies + s], want[:, h]) for h in range(heads) for s in range(copies)
            )
        t_ref = traced(mesh, ref)
        t_op = traced(mesh, lambda: q_heads(mesh, dq, heads))
        print(
            f"q_heads B {rows:3d}: bit for bit {same[1]} (2 copies {same[2]}), against the host "
            f"{torch.equal(want, q_host.reshape(1, rows, heads, d).transpose(1, 2))}; traced {t_ref * 1e6:6.1f} -> "
            f"{t_op * 1e6:6.1f} us",
            flush=True,
        )
        if not all(same.values()):
            failures.append(rows)
        ttnn.deallocate(dq)
        ttnn.deallocate(dl)
    if os.environ.get("GLM53_QH_STRICT") == "1":
        assert not failures, failures
