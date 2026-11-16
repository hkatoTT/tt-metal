"""The device MLP (TP4 over a row's chips) against `tt/`: a dense MLP (layer 0) and a shared expert (layer 3).

Each of the 8 rows gets different tokens, so the test also checks that the rows stay apart.
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_mlp_device.py
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.config import Glm53Config  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.common import Mlp, MlpWeights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS,
    MESH_SHAPE,
    ROWS,
    gather_rows,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.mlp import TtMlp  # noqa: E402


def pcc(a, b) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "prefix", ["model.language_model.layers.0.mlp.", "model.language_model.layers.3.mlp.shared_experts."]
)
@pytest.mark.parametrize("seq", [int(v) for v in os.environ.get("GLM53_MLP_TEST_SEQS", "32,128").split(",")])
def test_mlp_device(mesh_device, prefix, seq, pcc_thresholds):
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = MlpWeights.from_checkpoint(checkpoint, prefix)
    reference = Mlp(weights, config.swiglu_limit)
    module = TtMlp(mesh_device, weights, config.swiglu_limit)

    g = torch.Generator().manual_seed(seq)
    x = (torch.randn(ROWS, 1, seq, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
    out = gather_rows(mesh_device, module(put(mesh_device, x, dims=(0, None))))
    with torch.no_grad():
        expected = reference(x)
    for r in range(ROWS):
        score = pcc(out[r], expected[r])
        print(f"{prefix} seq {seq} row {r}: pcc {score:.6f}")
        assert score >= pcc_thresholds["test_mlp_device"], f"row {r}: pcc {score:.6f}"
    module.deallocate()
