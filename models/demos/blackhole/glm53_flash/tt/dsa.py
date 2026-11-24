"""DeepSeek sparse attention (DSA): the softmax attention of 11 of the 45 layers (every fourth, from layer 3).

The attention is MLA without RoPE: queries come through a 1536-wide low-rank bottleneck, keys and values are expanded
from one 512-wide normalized latent per token (the only per-token KV state), 64 heads of 256.

The indexer picks which past tokens each query attends to.  It groups the tokens into pools of 4 (from the first
token), compresses each complete pool into one key (a softmax-weighted mean of its tokens' indexer keys, weights from a
learned gate plus a per-slot bias), scores the pools with 32 ReLU heads mixed by per-token weights, keeps the 512 best
pools (2048 tokens) whose last token is visible, and always adds the incomplete tail pool (up to 3 tokens).  With at
most 512 complete pools visible every token is kept, so up to about 2,050 tokens of context DSA is plain causal
attention.

The state between calls is the latent, the indexer key and the indexer gate score of every past token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tools import kv_round
from models.demos.blackhole.glm53_flash.tt.common import LAYER_PREFIX, bf16, rms_norm

INDEXER_NORM_EPS = 1e-6


@dataclass(frozen=True)
class DsaWeights:
    q_a: torch.Tensor  # [q_lora, hidden]
    q_a_norm: torch.Tensor  # [q_lora]
    q_b: torch.Tensor  # [H * Dqk, q_lora]
    kv_a: torch.Tensor  # [kv_lora, hidden]
    kv_a_norm: torch.Tensor  # [kv_lora]
    kv_b: torch.Tensor  # [H * (Dqk + Dv), kv_lora]
    o_proj: torch.Tensor  # [hidden, H * Dv]
    index_wq: torch.Tensor  # [Hi * Di, q_lora]
    index_wk: torch.Tensor  # [Di, hidden]
    index_k_norm_weight: torch.Tensor  # [Di] (LayerNorm)
    index_k_norm_bias: torch.Tensor  # [Di]
    index_head_weights: torch.Tensor  # [Hi, hidden]
    index_pool_bias: torch.Tensor  # [kpool, Di]  (index_kpool_compress_ape)
    index_pool_gate: torch.Tensor  # [Di, hidden] (index_kpool_compress_gate)

    @classmethod
    def from_checkpoint(cls, checkpoint, layer: int) -> "DsaWeights":
        p = LAYER_PREFIX.format(layer) + "self_attn."
        t = lambda name: bf16(checkpoint.tensor(p + name))  # noqa: E731
        return cls(
            q_a=t("q_a_proj.weight"),
            q_a_norm=t("q_a_layernorm.weight"),
            q_b=t("q_b_proj.weight"),
            kv_a=t("kv_a_proj_with_mqa.weight"),
            kv_a_norm=t("kv_a_layernorm.weight"),
            kv_b=t("kv_b_proj.weight"),
            o_proj=t("o_proj.weight"),
            index_wq=t("indexer.wq_b.weight"),
            index_wk=t("indexer.wk.weight"),
            index_k_norm_weight=t("indexer.k_norm.weight"),
            index_k_norm_bias=t("indexer.k_norm.bias"),
            index_head_weights=t("indexer.weights_proj.weight"),
            index_pool_bias=t("indexer.index_kpool_compress_ape"),
            index_pool_gate=t("indexer.index_kpool_compress_gate"),
        )


@dataclass(frozen=True)
class DsaState:
    latent: torch.Tensor  # [B, T, kv_lora] bf16: the normalized KV latent of every past token
    index_key: torch.Tensor  # [B, T, Di] bf16
    index_gate: torch.Tensor  # [B, T, Di] bf16

    @property
    def length(self) -> int:
        return self.latent.shape[1]


@dataclass(frozen=True)
class IndexerResult:
    tokens: torch.Tensor  # [B, S, topk + kpool - 1] int32: selected token positions, -1 for none
    pool_scores: torch.Tensor  # [B, S, P] fp32 over the complete pools, masked pools at finfo.min


class Dsa:
    def __init__(self, weights: DsaWeights, config: Glm53Config) -> None:
        self.w = weights
        self.heads = config.num_heads
        self.qk_dim = config.qk_head_dim
        self.v_dim = config.v_head_dim
        self.eps = config.rms_norm_eps
        self.scale = self.qk_dim**-0.5
        self.index_heads = config.index_num_heads
        self.index_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.kpool = config.index_kpool

    def indexer(self, x, q_resid, index_key, index_gate, start: int) -> IndexerResult:
        """Select the tokens for queries at positions start .. start + S - 1 over keys 0 .. T - 1 (T = start + S)."""
        b, s = x.shape[:2]
        total = index_key.shape[1]
        kp = self.kpool
        q = F.linear(q_resid, self.w.index_wq).view(b, s, -1, self.index_dim)

        # Complete pools only: pool p holds tokens kp*p .. kp*p + kp - 1.
        pools = total // kp
        tokens_of_pool = torch.arange(pools * kp).view(pools, kp)
        keys = index_key[:, tokens_of_pool]  # [B, P, kp, Di]
        logits = index_gate[:, tokens_of_pool].float() + self.w.index_pool_bias.float()[None, None]
        weights = torch.nan_to_num(logits.softmax(dim=2)).to(keys.dtype)
        pool_keys = (weights * keys).sum(dim=2)  # [B, P, Di]

        scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))  # [B, S, Hi, P]
        scores = F.relu(scores * self.index_dim**-0.5)
        head_weights = F.linear(x, self.w.index_head_weights).float() * (self.index_heads**-0.5)
        pool_scores = torch.matmul(head_weights.unsqueeze(-2), scores).squeeze(-2)  # [B, S, P]

        positions = start + torch.arange(s)
        visible = (tokens_of_pool[:, -1][None, :] <= positions[:, None]).expand(b, s, pools)  # [B, S, P]
        pool_scores = pool_scores.masked_fill(~visible, torch.finfo(pool_scores.dtype).min)
        k = min(self.index_topk // kp, pools)
        chosen = pool_scores.topk(k, dim=-1).indices  # [B, S, k]
        chosen_ok = visible.gather(-1, chosen)
        selected = tokens_of_pool[chosen].flatten(-2)  # [B, S, k * kp]
        selected = selected.masked_fill(~chosen_ok[..., None].expand(-1, -1, -1, kp).flatten(-2), -1)

        # The tail: the visible tokens after the last complete visible pool.
        visible_count = positions + 1
        tail_count = visible_count.remainder(kp)
        tail = (visible_count - tail_count)[:, None] + torch.arange(kp - 1)  # [S, kp - 1]
        tail = tail.masked_fill(~(torch.arange(kp - 1)[None] < tail_count[:, None]), -1)
        selected = torch.cat([selected, tail.expand(b, -1, -1)], dim=-1)

        width = self.index_topk + kp - 1
        selected = F.pad(selected, (0, width - selected.shape[-1]), value=-1)[..., :width]
        return IndexerResult(tokens=selected.to(torch.int32), pool_scores=pool_scores)

    def __call__(self, x: torch.Tensor, state: DsaState | None = None) -> tuple[torch.Tensor, DsaState, IndexerResult]:
        """x [B, S, hidden] bf16 -> (out [B, S, hidden], state with these tokens appended, indexer selection)."""
        b, s = x.shape[:2]
        start = 0 if state is None else state.length

        q_resid = rms_norm(F.linear(x, self.w.q_a), self.w.q_a_norm, self.eps)
        q = F.linear(q_resid, self.w.q_b).view(b, s, -1, self.qk_dim).transpose(1, 2)  # [B, H, S, Dqk]
        latent = rms_norm(F.linear(x, self.w.kv_a), self.w.kv_a_norm, self.eps)  # [B, S, kv_lora]
        latent = kv_round.latent(latent)  # (GLM53_REF_KV_LATENT: as a low-precision cache holds it)
        index_key = F.layer_norm(
            F.linear(x, self.w.index_wk),
            (self.index_dim,),
            self.w.index_k_norm_weight,
            self.w.index_k_norm_bias,
            INDEXER_NORM_EPS,
        )
        index_gate = F.linear(x, self.w.index_pool_gate)
        if state is not None:
            latent = torch.cat([state.latent, latent], dim=1)
            index_key = torch.cat([state.index_key, index_key], dim=1)
            index_gate = torch.cat([state.index_gate, index_gate], dim=1)
        total = latent.shape[1]

        kv = F.linear(latent, self.w.kv_b).view(b, total, -1, self.qk_dim + self.v_dim).transpose(1, 2)
        key, value = kv.split([self.qk_dim, self.v_dim], dim=-1)
        key = key.contiguous()

        selection = self.indexer(x, q_resid, index_key, index_gate, start)
        valid = selection.tokens.ge(0) & selection.tokens.lt(total)
        counts = torch.zeros(b, s, total, dtype=torch.int32)
        counts.scatter_add_(-1, selection.tokens.clamp(0, total - 1).long(), valid.to(torch.int32))
        mask = counts.ne(0).unsqueeze(1)  # [B, 1, S, T]

        out = F.scaled_dot_product_attention(q, key, value, attn_mask=mask, scale=self.scale)
        out = out.transpose(1, 2).reshape(b, s, -1)
        out = F.linear(out, self.w.o_proj)
        return out, DsaState(latent=latent, index_key=index_key, index_gate=index_gate), selection
