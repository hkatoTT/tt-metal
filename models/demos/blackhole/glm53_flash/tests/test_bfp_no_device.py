"""The fast BFP rounding equals the bit-exact replay of TT's host packer on bf16 inputs (no device)."""

from __future__ import annotations

import pytest
import torch

from models.demos.blackhole.glm53_flash.tools.bfp_study import bfp_round, bfp_round_exact


@pytest.mark.parametrize("fmt", ["bfp8", "bfp4"])
@pytest.mark.parametrize("scale", [1e-3, 0.02, 3.0])
def test_fast_equals_exact_on_bf16(fmt, scale):
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(256, 1024, generator=g) * scale).to(torch.bfloat16)
    x[0, :16] = 0  # an all-zero group
    x[1, 3] = 0  # a zero inside a group
    x[2, :16] = torch.tensor([2.0**-k for k in range(16)])  # a wide range inside one group
    assert torch.equal(bfp_round(x, fmt), bfp_round_exact(x, fmt))


def test_known_values():
    x = torch.tensor([1.0, 0.5, 0.25, 3.0, -2.0, 1.015625, 0.0078125, 1e-3] + [0.0] * 8)
    assert bfp_round_exact(x, "bfp8").tolist()[:8] == [
        1.0,
        0.5,
        0.25,
        3.0,
        -2.0,
        1.0,
        0.0,
        0.0,
    ]  # step 1/32, ties to even
    assert bfp_round_exact(x, "bfp4").tolist()[:8] == [1.0, 0.5, 0.0, 3.0, -2.0, 1.0, 0.0, 0.0]  # step 1/2
