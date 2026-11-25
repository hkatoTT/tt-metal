"""The constant tiles built for `mhc_split_sinkhorn` reproduce `tt/` mHC weights when the op's arithmetic is replayed in
torch, on real layer weights (no device; needs the checkpoint)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.paths import checkpoint_skip_reason
from models.demos.blackhole.glm53_flash.tt.common import unweighted_rms_norm
from models.demos.blackhole.glm53_flash.tt.mhc import HyperConnection, HyperConnectionWeights
from models.demos.blackhole.glm53_flash.ttnn.mhc_consts import W, emulate_split_sinkhorn, split_sinkhorn_consts

NO_CHECKPOINT = checkpoint_skip_reason()  # (GLM53_CHECKPOINT, else HF_MODEL)
if NO_CHECKPOINT:
    pytest.skip(NO_CHECKPOINT, allow_module_level=True)


@pytest.mark.parametrize("layer, block", [(0, "attn"), (20, "ffn"), (44, "attn")])
def test_consts_reproduce_reference(layer, block):
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = HyperConnectionWeights.from_checkpoint(checkpoint, layer, block)
    reference = HyperConnection(
        weights,
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    g = torch.Generator().manual_seed(layer)
    streams = (torch.randn(1, 64, config.hc_mult, config.hidden_size, generator=g) * 0.02).to(torch.bfloat16)
    post, comb, _ = reference.read(streams)

    flat = unweighted_rms_norm(streams.view(1, 64, -1).float(), config.rms_norm_eps)
    mixes = F.pad(F.linear(flat, weights.fn.float())[0], (0, W - 24))  # [T, 32]
    consts = split_sinkhorn_consts(config.hc_mult, weights.scale, weights.base)
    _, e_post, e_comb = emulate_split_sinkhorn(mixes, consts, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps)
    torch.testing.assert_close(e_post, post[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(e_comb, comb[0].reshape(64, -1), rtol=1e-4, atol=1e-6)
