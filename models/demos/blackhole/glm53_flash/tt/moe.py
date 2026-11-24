"""The MoE of layers 3-44: 288 routed experts (top 8 per token) and one shared expert, all clamped SwiGLU MLPs.

Routing: sigmoid scores from an fp32 linear; the top 8 are chosen on score + a per-expert correction bias, but weighted
by the plain scores, normalized to sum to 1 and multiplied by 2.5.  The routed outputs are added per token in
ascending expert order in bf16 (as HF's expert loop does), then the shared expert's output is added.

Expert weights are checkpoint views: an expert's pages are read only when a token is routed to it.

Reference hook (read at each call; unset / empty: the reference as it is):
    GLM53_REF_MOE_PARTIAL  the device's col8 form of the routed sum: per mesh column c (experts 72 c ..) and chip r of
                           the column (intermediate features 256 r ..), the fp32 partial of the column's experts' outputs
                           rounded to bf16 (as the prefill's combine writes it), then summed over the column's 8 chips
                           and the 4 columns in bf16.  "bf16": the device's default form; a format of
                           tools/kv_round.round_kv (bfp8 / bfp8a): each partial rounded to it once before the column sum
                           (decode's column return with its partial sums in BFP8, GLM53_MOE_RS_BFP8);
                           "ring:<format>": the prefill column ring reduce-scatter sending its running sums in that
                           format (GLM53_CRR_BFP8): two chains into chip 0, 4 -> 3 -> 2 -> 1 and 5 -> 6 -> 7, each
                           running sum rounded to the format at every send, added to chip 0's own partial at the end;
                           "ring32:<format>": the same ring with each hop's add in fp32 and one rounding of that fp32
                           sum to the format (a fp32 Dest packed with bfp8_pack_precise: ties are rare, so the BH
                           packer's ties-away rounding is nearly round to nearest), the partials bf16 as the combine
                           writes them; "ring32c:<format>": also the partials rounded to the format from their fp32
                           sums (the combine packing the format from its fp32 Dest); "ring32cf:<format>": also the
                           column's sum in the format (the ring reduce-scatter's output in it, then added to the
                           shared expert's partial in bf16)
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
        import os

        fp = os.environ.get("GLM53_REF_MOE_PARTIAL", "")
        if fp:
            return self._routed_col8(x, routing, fp)
        out = torch.zeros_like(x)
        for e in torch.unique(routing.indices).tolist():
            slot, token = torch.where(routing.indices.T == e)
            w = self.w.expert(e)
            h = clamped_swiglu(F.linear(x[token], w.gate), F.linear(x[token], w.up), self.limit)
            h = F.linear(h, w.down) * routing.weights[token, slot, None]
            out.index_add_(0, token, h.to(out.dtype))
        return out

    def _routed_col8(self, x: torch.Tensor, routing: Routing, fmt: str) -> torch.Tensor:
        """The routed sum in the device's col8 form (the reference hook GLM53_REF_MOE_PARTIAL)."""
        from models.demos.blackhole.glm53_flash.tools.kv_round import round_kv

        n, d = x.shape
        part = torch.zeros(4, 8, n, d, dtype=torch.float32)
        for e in torch.unique(routing.indices).tolist():
            slot, token = torch.where(routing.indices.T == e)
            w = self.w.expert(e)
            h = clamped_swiglu(F.linear(x[token], w.gate), F.linear(x[token], w.up), self.limit)
            h = h.float() * routing.weights[token, slot, None]
            inter = h.shape[-1] // 8
            for r in range(8):
                y = F.linear(h[:, inter * r : inter * (r + 1)], w.down[:, inter * r : inter * (r + 1)].float())
                part[e // 72, r].index_add_(0, token, y)
        kind, _, f = fmt.partition(":") if ":" in fmt else ("", "", fmt)
        ring = kind == "ring"
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        q = lambda t: t if f == "bf16" else round_kv(t, f)  # noqa: E731  (a bf16 tensor in the format, bf16)
        out = torch.zeros(n, d, dtype=torch.bfloat16)
        for c in range(4):
            if kind in ("ring32", "ring32c", "ring32cf"):  # (fp32 adds, one rounding a hop: round_kv keeps fp32)
                p = [bf(part[c, r]).float() if kind == "ring32" else q(part[c, r]) for r in range(8)]
                a = q(p[4].float())
                for r in (3, 2, 1):
                    a = q(a + p[r].float())
                b2 = q(p[5].float())
                for r in (6, 7):
                    b2 = q(b2 + p[r].float())
                col = bf(q(p[0].float() + a + b2) if kind == "ring32cf" else p[0].float() + a + b2)
                out = bf(out.float() + col.float())
                continue
            p = [bf(part[c, r]) for r in range(8)]
            if ring:  # (each running sum rounded at every send; the last hop lands in chip 0)
                a = p[4]
                for r in (3, 2, 1):
                    a = bf(q(a).float() + p[r].float())
                b = p[5]
                for r in (6, 7):
                    b = bf(q(b).float() + p[r].float())
                col = bf(p[0].float() + q(a).float() + q(b).float())
            else:
                col = None
                for r in range(8):
                    t = q(p[r])
                    col = t if col is None else bf(col.float() + t.float())
            out = bf(out.float() + col.float())
        return out.to(x.dtype)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, Routing]:
        """x [B, S, hidden] bf16 -> (out [B, S, hidden], routing)."""
        routing = self.router(x)
        out = self.routed(x.reshape(-1, x.shape[-1]), routing).view(x.shape)
        return out + self.shared(x), routing
