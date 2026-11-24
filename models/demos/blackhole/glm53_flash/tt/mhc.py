"""Manifold-constrained hyper-connections (mHC): the residual is `hc_mult` streams instead of one.

Before a block (attention or MLP), `read` turns the streams into three sets of weights through one small projection:
`pre` collapses the streams into the block input, `post` spreads the block output back over the streams, and `comb`
(a doubly-stochastic matrix from Sinkhorn iterations) mixes the old streams.  `write` forms
`post * block_out + comb^T @ streams`.  The final collapse before the LM head is a plain mean over the streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.blackhole.glm53_flash.tt.common import LAYER_PREFIX, bf16, unweighted_rms_norm


@dataclass(frozen=True)
class HyperConnectionWeights:
    fn: torch.Tensor  # [(2 + n) * n, n * hidden]
    base: torch.Tensor  # [(2 + n) * n]
    scale: torch.Tensor  # [3]: pre, post, comb

    @classmethod
    def from_checkpoint(cls, checkpoint, layer: int, block: str) -> "HyperConnectionWeights":
        if block not in ("attn", "ffn"):
            raise ValueError(f"block must be attn or ffn, got {block!r}")
        p = LAYER_PREFIX.format(layer) + f"hc_{block}_"
        # The HF model holds these in the model dtype (bf16); `fn` is used in fp32 after that rounding.
        return cls(
            bf16(checkpoint.tensor(p + "fn")), bf16(checkpoint.tensor(p + "base")), bf16(checkpoint.tensor(p + "scale"))
        )


class HyperConnection:
    def __init__(
        self, weights: HyperConnectionWeights, *, hc_mult: int, sinkhorn_iters: int, hc_eps: float, norm_eps: float
    ):
        self.w = weights
        self.n = hc_mult
        self.iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.norm_eps = norm_eps

    def read(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """streams [B, S, n, D] -> (post [B, S, n] fp32, comb [B, S, n, n] fp32, block input [B, S, D])."""
        b, s = streams.shape[:2]
        n = self.n
        flat = unweighted_rms_norm(streams.view(b, s, -1).float(), self.norm_eps)
        mixed = F.linear(flat, self.w.fn.float())
        pre_w, post_w, comb_w = mixed.split([n, n, n * n], dim=-1)
        pre_b, post_b, comb_b = self.w.base.split([n, n, n * n])
        pre_scale, post_scale, comb_scale = self.w.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w.view(b, s, n, n) * comb_scale + comb_b.view(n, n), dim=-1) + self.hc_eps
        # Sinkhorn-Knopp: columns, then (rows, columns) iters - 1 times.
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        block_input = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
        return post, comb, block_input

    @staticmethod
    def write(block_out: torch.Tensor, streams: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """New streams [B, S, n, D] = post * block_out + comb^T @ streams, in the streams' dtype."""
        dtype = streams.dtype
        return post.to(dtype).unsqueeze(-1) * block_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), streams
        )


def expand_streams(embeddings: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """[B, S, D] -> [B, S, n, D]: every stream starts as the token embedding."""
    return embeddings.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()


def collapse_streams(streams: torch.Tensor) -> torch.Tensor:
    """[B, S, n, D] -> [B, S, D]: the unweighted mean before the final norm."""
    return streams.mean(dim=2)
