"""The prefix store's copy op (`ttnn/prefix_store.copy_pages_many`): segments of pages between
pairs of interleaved DRAM buffers (the same segments for each pair, from one row of a plan of several), each mesh row
its own segments, against torch on every chip, bit for bit.

Page kinds as the store copies them: row-major rows of 512 bytes (the DSA latents' FP8 rows; here bf16 [.., 256]),
BFP4 tiles (576 bytes, the pooled indexer keys), BFP8 tiles (1,088, the drafter's K / V), bf16 tiles (2,048, the KDA
wave histories), fp32 tiles (4,096, the KDA wave states).  Each chip's source and destination hold their own random
values; row r copies its own segments (row 7: none, the destination unchanged), segments of 1 to 300 pages, more
pages than the op's 64 cores x 16 slots in one.  Each kind copies PAIRS buffer pairs in one op by the plan's second row
(the first a decoy of other segments), and the first kind once more as a single pair (`copy_pages`) by the first row.
Then (GLM53_PC_TIME=1) one large copy timed: 128 MiB a chip of 512-byte rows and of fp32 tiles.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_prefix_copy.py -s

Prints "PREFIX COPY ALL SAME" when every chip of every kind matches; exits 0 either way (a study).
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

from models.demos.blackhole.glm53_flash.ttnn.common import COLS, DEVICE_PARAMS, MESH_SHAPE, ROWS, mapper  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.prefix_store import (  # noqa: E402
    copy_pages,
    copy_pages_many,
    make_plan,
    make_plans,
    page_bytes,
)

PAGES = 1200  # a tensor's pages a chip
PAIRS = 3  # buffer pairs an op
KINDS = [  # (name, dtype, layout, a page's shape)
    ("rows bf16 x256 (512 B)", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (1, 256)),
    ("bfp4 tiles (576 B)", ttnn.bfloat4_b, ttnn.TILE_LAYOUT, (32, 32)),
    ("bfp8 tiles (1,088 B)", ttnn.bfloat8_b, ttnn.TILE_LAYOUT, (32, 32)),
    ("bf16 tiles (2,048 B)", ttnn.bfloat16, ttnn.TILE_LAYOUT, (32, 32)),
    ("fp32 tiles (4,096 B)", ttnn.float32, ttnn.TILE_LAYOUT, (32, 32)),
]


def mesh_tensor(mesh, host: torch.Tensor, dtype, layout):
    """host [ROWS * COLS, PAGES, h, w] (chip k = r COLS + c its own) -> per chip [PAGES, 1, h, w] (a page each)."""
    k, p, h, w = host.shape
    full = host.reshape(ROWS, COLS, p, h, w).permute(0, 2, 1, 3, 4).reshape(ROWS * p, COLS, h, w)
    return ttnn.from_torch(
        full.float(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper(mesh, (0, 1)),
    )


def chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(x).float().reshape(PAGES, -1) for x in ttnn.get_device_tensors(t)]


def plans_by_row(g) -> list:
    """Each mesh row's segments (source page, destination page, pages): distinct destinations, row 7 none."""
    rows = []
    for r in range(ROWS - 1):
        lengths = [1, 3, 120, 64 * 16 + 5] if r == 0 else [1, 3 + r, 300 - 17 * r, 40]
        segs, dst = [], 0
        for n in lengths:
            src = int(torch.randint(0, PAGES - n + 1, (1,), generator=g))
            segs.append((src, dst, n))
            dst += n + int(torch.randint(0, 7, (1,), generator=g))
        assert dst <= PAGES, dst
        rows.append(segs)
    rows.append([])
    return rows


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_prefix_copy(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(11)
    verdict = []
    cases = [(kind, PAIRS, 1) for kind in KINDS] + [(KINDS[0], 1, 0)]  # (kind, pairs, the plan's row)
    for (name, dtype, layout, (h, w)), npairs, plan_row in cases:
        hosts = [
            (torch.randn(ROWS * COLS, PAGES, h, w, generator=g), torch.randn(ROWS * COLS, PAGES, h, w, generator=g))
            for _ in range(npairs)
        ]
        pairs = [(mesh_tensor(mesh, a, dtype, layout), mesh_tensor(mesh, b, dtype, layout)) for a, b in hosts]
        wants = [(chips(src), chips(dst)) for src, dst in pairs]  # (the values as the device holds them: BFP rounded)
        rows, decoy = plans_by_row(g), plans_by_row(g)
        t0 = time.perf_counter()
        if npairs == 1:
            plan = make_plan(mesh, rows)
            copy_pages(mesh, *pairs[0], plan)
        else:
            plan = make_plans(mesh, [decoy, rows])
            copy_pages_many(mesh, pairs, plan, plan_row)
        ttnn.synchronize_device(mesh)
        dt = time.perf_counter() - t0
        bad = []
        for q, ((src, dst), (want_src, want_dst)) in enumerate(zip(pairs, wants)):
            got_src, got_dst = chips(src), chips(dst)
            for k in range(ROWS * COLS):
                want = want_dst[k].clone()
                for a, b, n in rows[k // COLS]:
                    want[b : b + n] = want_src[k][a : a + n]
                if not torch.equal(got_dst[k], want) or not torch.equal(got_src[k], want_src[k]):
                    diff = (got_dst[k] != want).any(dim=1).nonzero().flatten().tolist()
                    bad.append((q, k, diff[:6], len(diff)))
        pages = sum(n for segs in rows for _, _, n in segs)
        print(
            f"PREFIX COPY {name} x {npairs} (plan row {plan_row}): page {page_bytes(pairs[0][0])} B, {pages} pages "
            f"over the rows a pair, {dt * 1e3:.2f} ms (first call, with the program's build): "
            f"{'the same on 32 chips' if not bad else f'DIFFERENT {len(bad)} (pair, chip) {bad[:4]}'}",
            flush=True,
        )
        verdict.append(not bad)
        for src, dst in pairs:
            ttnn.deallocate(src)
            ttnn.deallocate(dst)
        ttnn.deallocate(plan)
    print(f"PREFIX COPY {'ALL SAME' if all(verdict) else 'FAILED'} {verdict}", flush=True)

    if os.environ.get("GLM53_PC_TIME", "0") != "1":
        return
    for name, dtype, layout, (h, w), n in (
        ("rows of 512 B", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (1, 256), 262144),
        ("fp32 tiles", ttnn.float32, ttnn.TILE_LAYOUT, (32, 32), 32768),
    ):
        shape = [n * h, w] if layout == ttnn.ROW_MAJOR_LAYOUT else [n, 1, h, w]
        spec = ttnn.TensorSpec(ttnn.Shape(shape), dtype, layout, buffer_type=ttnn.BufferType.DRAM)
        a = ttnn.allocate_tensor_on_device(spec, mesh)
        b = ttnn.allocate_tensor_on_device(spec, mesh)
        plan = make_plan(mesh, [[(0, 0, n)] for _ in range(ROWS)])
        copy_pages(mesh, a, b, plan)
        ttnn.synchronize_device(mesh)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            copy_pages(mesh, a, b, plan)
            ttnn.synchronize_device(mesh)
            times.append(time.perf_counter() - t0)
        size = n * page_bytes(a)
        best = min(times)
        print(
            f"PREFIX COPY TIME {name}: {size / 2**20:.0f} MiB a chip in {best * 1e3:.2f} ms ({size / best / 1e9:.0f} GB/s "
            f"a chip read and written; {[round(t * 1e3, 2) for t in times]} ms)",
            flush=True,
        )
        for t in (a, b, plan):
            ttnn.deallocate(t)
