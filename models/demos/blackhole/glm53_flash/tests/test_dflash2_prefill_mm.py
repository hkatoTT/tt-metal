"""The drafter's prefill matmuls (`TtDFlash2._lin` past 128 rows, `_head_lin` past HEAD_1D_MAX tile rows) under two L1
layouts: ttnn's default program config, whose blocks follow the L1 free for programs at the call
(matmul_program_config.cpp: get_max_l1_space -> available_program_l1_capacity), against GLM53_DFLASH2_PREFILL_MM=fixed's
(`dflash2.fixed_prefill_config`), each chip's output bit for bit.

The layouts: with and without an L1 tensor of GLM53_PMM_HELD_B bytes a core held (176,512: the L1 landing of the next
MoE layer's chosen experts' weights, GLM53_W1, at C = 5), or with GLM53_PMM_FREE ("F1,F2,.." bytes) an L1 tensor held
so that the L1 free for programs is each F in turn (the model's values at the drafter's calls, GLM53_DFLASH2_MM_LOG's
"DRAFTER MM .. free L1"; e.g. 1,326,784 and 1,150,272, where the default's fc at 1,024 rows parted on every chip and
the fixed config did not).

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dflash2_prefill_mm.py -s

Prints "PREFILL MM <shape> <form>: free L1 .. vs .., x / 32 chips bit for bit .." per shape and form.  A probe
(tests/_trace.py `study`: an exception is printed, a device timeout raised).
"""

from __future__ import annotations

import hashlib
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tests._trace import study  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dflash2 as d2  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS, MESH_SHAPE, put  # noqa: E402

# (m, k, n, weight dtype, output fp32): the context's fc (K = 36,864) and K / V at chunks 1,024 / 512, the head at 128
# users' candidate rows
SHAPES = [
    (1024, 36864, 1024, ttnn.bfloat8_b, False),
    (512, 36864, 1024, ttnn.bfloat8_b, False),
    (1024, 4096, 4096, ttnn.bfloat8_b, False),
    (1024, 4096, 4864, ttnn.bfloat8_b, True),
]
HELD_B = int(os.environ.get("GLM53_PMM_HELD_B", "176512"))
FREE = [int(v) for v in os.environ.get("GLM53_PMM_FREE", "").split(",") if v.strip()]


def free_l1(mesh) -> int:
    """The L1 free for programs: the lowest allocated block less the lowest block (the CBs' base), or the whole span."""
    table = ttnn.get_memory_view(mesh, ttnn.BufferType.L1).block_table
    base = min(int(b["address"]) for b in table)
    top = max(int(b["address"]) + int(b["size"]) for b in table)
    used = [int(b["address"]) for b in table if b.get("allocated") == "yes"]
    return (min(used) if used else top) - base


def held(mesh, size: int):
    """An L1 tensor of `size` bytes a core on the 12 x 10 grid (L1 buffers go top-down: the free span shrinks)."""
    words = size // 4
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9))])
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        torch.zeros(120, words, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh, memory_config=mem, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )  # fmt: skip


def hashes(t) -> list[str]:
    return [hashlib.sha1(ttnn.to_torch(c).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
            for c in ttnn.get_device_tensors(t)]  # fmt: skip


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_dflash2_prefill_mm(mesh_device):
    study(lambda: _run(mesh_device))


def _layout(mesh, layout):
    """The layout's held tensor (or None): `layout` False / True (HELD_B held), or a free-L1 value to reach."""
    if FREE:
        size = (free_l1(mesh) - layout) // 64 * 64
        return held(mesh, size) if size > 0 else None
    return held(mesh, HELD_B) if layout else None


def _run(mesh) -> None:
    g = torch.Generator().manual_seed(5)
    layouts = FREE if FREE else [False, True]
    for m, k, n, wdt, f32 in SHAPES:
        x = put(mesh, torch.randn(1, 1, m, k, generator=g).bfloat16(), dims=(None, None), dtype=ttnn.bfloat16)
        w = put(mesh, (torch.randn(1, 1, k, n, generator=g) / k**0.5).bfloat16(), dims=(None, None), dtype=wdt)
        kw = dict(dtype=ttnn.float32) if f32 else {}
        cc = d2.HEAD_MM if f32 else d2.DRAFTER_MM
        cfg = d2.fixed_prefill_config(mesh, m, k, n, 1088 if wdt == ttnn.bfloat8_b else 2048, 4096 if f32 else 2048)
        forms = ["default"] + (["fixed"] if cfg is not None else [])
        if cfg is None:
            print(f"PREFILL MM m {m} k {k} n {n}: no fixed config fits", flush=True)
        got = {}
        for form in forms:
            for layout in layouts:
                h = _layout(mesh, layout)
                seen = free_l1(mesh)
                if form == "default":
                    y = ttnn.linear(x, w, compute_kernel_config=cc, **kw)
                else:
                    y = ttnn.linear(x, w, program_config=cfg, compute_kernel_config=cc, **kw)
                got[(form, layout)] = (hashes(y), ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float(), seen)
                ttnn.deallocate(y)
                if h is not None:
                    ttnn.deallocate(h)
        for form in forms:
            h0, y0, f0 = got[(form, layouts[0])]
            for layout in layouts[1:]:
                h1, y1, f1 = got[(form, layout)]
                same = sum(a == b for a, b in zip(h0, h1))
                d = (y0 - y1).abs()
                print(f"PREFILL MM m {m} k {k} n {n}{' fp32' if f32 else ''} {form}: free L1 {f1} vs {f0}, {same} / "
                      f"{len(h0)} chips bit for bit (chip 0: {int((d > 0).sum())} values differ, max |d| "
                      f"{d.max().item():.3e})", flush=True)  # fmt: skip
        if cfg is not None:
            d = (got[("default", layouts[0])][1] - got[("fixed", layouts[0])][1]).abs()
            print(f"PREFILL MM m {m} k {k} n {n}: fixed against default (the first layout), chip 0 max |d| "
                  f"{d.max().item():.3e}, {int((d > 0).sum())} values differ; fixed K block {cfg.in0_block_w}",
                  flush=True)  # fmt: skip
        ttnn.deallocate(x)
        ttnn.deallocate(w)
