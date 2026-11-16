"""The MoE of layers 3-44: 288 routed experts (top 8 per token) and one shared expert, all clamped SwiGLU MLPs.

Routing: sigmoid scores from an fp32 linear; the top 8 are chosen on score + a per-expert correction bias, but weighted
by the plain scores, normalized to sum to 1 and multiplied by 2.5.  The routed outputs are added per token in
ascending expert order in bf16 (as HF's expert loop does), then the shared expert's output is added.

Expert weights are checkpoint views: an expert's pages are read only when a token is routed to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tt.common import LAYER_PREFIX, Mlp, MlpWeights, bf16, clamped_swiglu


@dataclass(frozen=True)
class Routing:
    logits: torch.Tensor  # [N, E] fp32
    indices: torch.Tensor  # [N, k] int64, in torch.topk(sorted=False) order
    weights: torch.Tensor  # [N, k] fp32: normalized and scaled


class Router:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, config: Glm53Config) -> None:
        self.weight = weight  # [E, hidden] bf16
        self.bias = bias  # [E] fp32 (e_score_correction_bias)
        self.k = config.num_experts_per_token
        self.scale = config.routed_scaling_factor

    def __call__(self, x: torch.Tensor) -> Routing:
        x = x.reshape(-1, x.shape[-1])
        logits = F.linear(x.float(), self.weight.float())
        scores = logits.sigmoid()
        choice = scores + self.bias
        indices = torch.topk(choice, k=self.k, dim=-1, sorted=False)[1]
        weights = scores.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
        return Routing(logits=logits, indices=indices, weights=weights * self.scale)


class MoeWeights:
    """Lazy per-expert views of one MoE layer."""

    def __init__(self, checkpoint, layer: int) -> None:
        self.checkpoint = checkpoint
        self.layer = layer
        self.prefix = LAYER_PREFIX.format(layer) + "mlp."
        t = checkpoint.tensor
        self.router = bf16(t(self.prefix + "gate.weight"))
        self.router_bias = t(self.prefix + "gate.e_score_correction_bias").float()
        self.shared = MlpWeights.from_checkpoint(checkpoint, self.prefix + "shared_experts.")

    def expert(self, e: int) -> MlpWeights:
        return MlpWeights.from_checkpoint(self.checkpoint, self.prefix + f"experts.{e}.")


class Moe:
    def __init__(self, weights: MoeWeights, config: Glm53Config) -> None:
        self.w = weights
        self.router = Router(weights.router, weights.router_bias, config)
        self.shared = Mlp(weights.shared, config.swiglu_limit)
        self.limit = config.swiglu_limit

    def routed(self, x: torch.Tensor, routing: Routing) -> torch.Tensor:
        """x [N, hidden] -> sum over each token's experts of weight * expert(x), accumulated in bf16 by expert id."""
        out = torch.zeros_like(x)
        for e in torch.unique(routing.indices).tolist():
            slot, token = torch.where(routing.indices.T == e)
            w = self.w.expert(e)
            h = clamped_swiglu(F.linear(x[token], w.gate), F.linear(x[token], w.up), self.limit)
            h = F.linear(h, w.down) * routing.weights[token, slot, None]
            out.index_add_(0, token, h.to(out.dtype))
        return out

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, Routing]:
        """x [B, S, hidden] bf16 -> (out [B, S, hidden], routing)."""
        routing = self.router(x)
        out = self.routed(x.reshape(-1, x.shape[-1]), routing).view(x.shape)
        return out + self.shared(x), routing
