"""The MTP layer (45, `num_nextn_predict_layers` 1): the draft of MTP self-speculative decode.  HF does not load it.

A plain decoder layer (no mHC: DSA attention and MoE with pre-norm residuals) on eh_proj(cat(enorm(embed(t[i+1])),
hnorm(h[i]))), h[i] the main model's final normed hidden state at position i (what the LM head reads); its output goes
through shared_head.norm and the main LM head and predicts t[i+2].  The concatenation order (the embedding first)
gives a top-1 agreement of 0.89 with the model's own greedy tokens on the acceptance records
(tools/mtp_acceptance.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tt.common import LAYER_PREFIX, bf16, rms_norm
from models.demos.blackhole.glm53_flash.tt.dsa import Dsa, DsaState, DsaWeights
from models.demos.blackhole.glm53_flash.tt.moe import Moe, MoeWeights


class MtpLayer:
    def __init__(self, checkpoint, config: Glm53Config, order: str = "eh") -> None:
        layer = config.num_layers
        p = LAYER_PREFIX.format(layer)
        t = lambda name: bf16(checkpoint.tensor(p + name))  # noqa: E731
        self.eh_proj, self.enorm, self.hnorm = t("eh_proj.weight"), t("enorm.weight"), t("hnorm.weight")
        self.input_norm, self.post_norm = t("input_layernorm.weight"), t("post_attention_layernorm.weight")
        self.head_norm = t("shared_head.norm.weight")
        self.attention = Dsa(DsaWeights.from_checkpoint(checkpoint, layer), config)
        self.mlp = Moe(MoeWeights(checkpoint, layer), config)
        self.eps = config.rms_norm_eps
        self.order = order

    def __call__(
        self, embeds: torch.Tensor, hidden: torch.Tensor, state: DsaState | None = None
    ) -> tuple[torch.Tensor, DsaState]:
        """embeds [B, S, H] (of t[i+1]), hidden [B, S, H] (h[i]) -> (the layer's output [B, S, H] bf16 before
        shared_head.norm, the attention state after these positions)."""
        e, h = rms_norm(embeds, self.enorm, self.eps), rms_norm(hidden, self.hnorm, self.eps)
        x = F.linear(torch.cat([e, h] if self.order == "eh" else [h, e], dim=-1), self.eh_proj)
        a, state, _ = self.attention(rms_norm(x, self.input_norm, self.eps), state)
        x = x + a
        m, _ = self.mlp(rms_norm(x, self.post_norm, self.eps))
        return x + m, state

    def head_input(self, y: torch.Tensor) -> torch.Tensor:
        """The layer's output -> what the main LM head reads."""
        return rms_norm(y, self.head_norm, self.eps)
