"""First device check on the BH Galaxy: open the 8x4 mesh and run the collectives the model uses on both axes.

Run (a device test; set GLM53_DEVICE_TEST=1):
    pytest models/demos/blackhole/glm53_flash/tests/test_device_smoke.py
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)

ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS  # noqa: E402

MESH = (8, 4)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


def to_mesh(mesh, t: torch.Tensor, dims):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH, dims=dims),
    )


def from_mesh(mesh, t, dims):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MESH, dims=dims))


@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("axis", [0, 1])
def test_all_gather_and_all_reduce(mesh_device, axis):
    rows, cols = MESH
    g = torch.Generator().manual_seed(axis)
    # one [32, 256] block per chip, placed on dims (2, 3) by (row, col)
    full = torch.randn(1, 1, 32 * rows, 256 * cols, generator=g).to(torch.bfloat16)
    x = to_mesh(mesh_device, full, dims=(2, 3))

    gathered = ttnn.all_gather(x, dim=2 if axis == 0 else 3, cluster_axis=axis)
    out = from_mesh(mesh_device, gathered, dims=(2, 3))
    blocks = full.unflatten(2, (rows, 32)).unflatten(4, (cols, 256))  # [1, 1, r, 32, c, 256]
    if axis == 0:  # every chip in column c holds the whole column
        expected = blocks.permute(0, 1, 4, 2, 3, 5).reshape(1, 1, cols, rows * 32, 256)
        expected = expected.unsqueeze(2).expand(1, 1, rows, cols, rows * 32, 256)
        expected = expected.permute(0, 1, 2, 4, 3, 5).reshape(1, 1, rows * rows * 32, cols * 256)
    else:  # every chip in row r holds the whole row
        expected = blocks.reshape(1, 1, rows, 32, cols * 256).unsqueeze(4).expand(1, 1, rows, 32, cols, cols * 256)
        expected = expected.reshape(1, 1, rows * 32, cols * cols * 256)
    assert out.shape == expected.shape
    assert torch.equal(out, expected), f"all_gather axis {axis}: pcc {pcc(out, expected):.6f}"

    reduced = ttnn.all_reduce(x, cluster_axis=axis)
    out = from_mesh(mesh_device, reduced, dims=(2, 3)).float()
    sums = blocks.float().sum(dim=2 if axis == 0 else 4, keepdim=True)
    expected = sums.expand_as(blocks).reshape(full.shape)
    score = pcc(out, expected)
    assert score > 0.9999, f"all_reduce axis {axis}: pcc {score:.6f}"
