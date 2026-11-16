"""The DFlash2 block-diffusion draft model (canada-quant/GLM-5.3-Flash-DFlash2-G, Apache-2.0) on the CPU, in bf16, for
measuring its acceptance on our reference (tools/dflash2_acceptance.py).  HF has no class for it; this follows the
checkpoint's config and tensor names and the public description of DFlash / DFlash2 (arXiv 2602.06036, vLLM's
`qwen3_dflash2` drafter and its speculator).

The target side: the mean of the 4 mHC streams after each target layer in `target_layer_ids` (HF's
`hidden_states[layer + 1]`, the residual entering the next layer), for the positions the target has verified.

Drafting at the bonus token t[s] (sampled by the target at s - 1, not yet run through it):
  - context, positions 0 .. s - 1: c = hidden_norm(fc(cat(taps))); in every layer K = rope(k_norm(k_proj(c))) and
    V = v_proj(c).  The context does not go through the layers.
  - query block, positions s .. s + K: embed([t[s], mask x K]) (the mask slots take `mask_embedding.pt`), then 8
    pre-norm layers and the final norm.  Each sublayer (attention, MLP) is wrapped in a grouped dynamic 2-tap
    convolution over the block, causal inside the block and zero across its start: per row i and channel c,
    out[i, c] = sum_t (base[side, t, c] + delta[i, side, t, c // 16]) x[i - t, c], delta = kernel_projection(h) of
    the sublayer's normed input h; side 0 is applied to h, side 1 to the sublayer's output.  The attention is not
    causal: the block's queries see the context and the whole block (GQA 32 / 8, per-head q / k norms, neox RoPE).
  - candidates: the draft LM head's top 16 at each mask slot.  The selector scores the edge from a predecessor p (the
    anchor t[s] for the first slot, else one of the previous slot's candidates) to a candidate c as
    unary[c] + <A[p] * proj(h), B[c]>, A / B the predecessor / successor codebooks (rank 256) and h the slot's final
    hidden state; the draft is the greedy walk from the anchor.  `plain` keeps the head's own top 1 per slot.

The number of mask slots changes the block (the attention sees all of it), so each K is its own forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from models.demos.blackhole.glm53_flash.paths import drafter_dir
from models.demos.blackhole.glm53_flash.tools import kv_round
from models.demos.blackhole.glm53_flash.tt.common import bf16, rms_norm


@dataclass(frozen=True)
class DFlash2Config:
    hidden: int
    intermediate: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    eps: float
    rope_theta: float
    vocab: int
    target_layer_ids: tuple[int, ...]
    block_size: int
    mask_token_id: int
    conv_taps: int
    conv_group: int
    selector_rank: int
    selector_top_k: int

    @classmethod
    def from_dict(cls, c: dict) -> "DFlash2Config":
        d = c["dflash_config"]
        if c.get("is_causal", False):
            raise ValueError("expected a non-causal drafter (is_causal false)")
        if c.get("rope_parameters", {}).get("rope_type", "default") != "default":
            raise ValueError(f"unexpected rope {c['rope_parameters']}")
        return cls(
            hidden=c["hidden_size"],
            intermediate=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            eps=c["rms_norm_eps"],
            rope_theta=c["rope_parameters"]["rope_theta"],
            vocab=c["vocab_size"],
            target_layer_ids=tuple(d["target_layer_ids"]),
            block_size=d["block_size"],
            mask_token_id=d["mask_token_id"],
            conv_taps=d["conv_kernel_size"],
            conv_group=d["conv_group_size"],
            selector_rank=d["selector_rank"],
            selector_top_k=d["selector_top_k"],
        )


@dataclass
class DraftContext:
    """Per layer, the context K / V of positions 0 .. S - 1: k [S, kv heads, head_dim] (normed, rotated), v the same."""

    k: list[torch.Tensor]
    v: list[torch.Tensor]


@dataclass
class Proposal:
    drafts: torch.Tensor  # [A, K] the selector's walk
    plain: torch.Tensor  # [A, K] the head's own top 1 per slot
    in_top_k: torch.Tensor  # [A, K] bool: the slot's candidates hold the given reference token (if any)


def _rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    freqs = positions.to(torch.float64)[..., None] * inv
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float(), emb.sin().float()


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """neox RoPE on [..., heads, head_dim] in fp32, rounded to x's dtype; cos / sin [..., head_dim] (no head axis)."""
    xf = x.float()
    half = xf.shape[-1] // 2
    rotated = torch.cat([-xf[..., half:], xf[..., :half]], dim=-1)
    return (xf * cos.unsqueeze(-2) + rotated * sin.unsqueeze(-2)).to(x.dtype)


class DFlash2Drafter:
    def __init__(self, root: Path | str | None = None) -> None:
        root = drafter_dir(root)  # the argument, else $GLM53_DFLASH2_DRAFTER
        self.config = cfg = DFlash2Config.from_dict(json.loads((root / "config.json").read_text()))
        w = {k: bf16(v) for k, v in load_file(str(root / "model.safetensors")).items()}
        self.w = w
        mask = torch.load(root / "mask_embedding.pt", weights_only=True)
        if isinstance(mask, dict):
            if mask.get("mask_token_id", cfg.mask_token_id) != cfg.mask_token_id:
                raise ValueError(f"mask_embedding.pt is for token {mask['mask_token_id']}, not {cfg.mask_token_id}")
            mask = mask["embedding"]
        self.mask_embedding = bf16(mask).reshape(cfg.hidden)
        self.lm_head32 = w["lm_head.weight"].float()  # the candidate logits in fp32
        n_taps = len(cfg.target_layer_ids)
        if w["fc.weight"].shape != (cfg.hidden, n_taps * cfg.hidden):
            raise ValueError(f"fc {tuple(w['fc.weight'].shape)} does not take {n_taps} taps")
        self.groups = cfg.hidden // cfg.conv_group

    # -- context -------------------------------------------------------------------------------------------------
    def context_features(self, taps: torch.Tensor) -> torch.Tensor:
        """taps [S, n_taps * hidden] bf16 (the target layers in config order) -> hidden_norm(fc(taps)) [S, hidden]."""
        return rms_norm(F.linear(bf16(taps), self.w["fc.weight"]), self.w["hidden_norm.weight"], self.config.eps)

    def context(self, taps: torch.Tensor) -> DraftContext:
        cfg = self.config
        c = self.context_features(taps)
        cos, sin = _rope_cos_sin(torch.arange(c.shape[0]), cfg.head_dim, cfg.rope_theta)
        ks, vs = [], []
        for i in range(cfg.num_layers):
            p = f"layers.{i}.self_attn."
            k = F.linear(c, self.w[p + "k_proj.weight"]).view(-1, cfg.num_kv_heads, cfg.head_dim)
            k = rms_norm(k, self.w[p + "k_norm.weight"], cfg.eps)
            ks.append(kv_round.draft(_rope(k, cos, sin)))  # (GLM53_REF_KV_DRAFT: as a low-precision cache)
            v = F.linear(c, self.w[p + "v_proj.weight"]).view(-1, cfg.num_kv_heads, cfg.head_dim)
            vs.append(kv_round.draft(v))
        return DraftContext(ks, vs)

    # -- query block ---------------------------------------------------------------------------------------------
    def _conv(self, x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """x [A, B, hidden], delta [A, B, taps, groups], base [taps, hidden] -> [A, B, hidden] (fp32 math, x's dtype)."""
        a, b, _ = x.shape
        taps, g = self.config.conv_taps, self.groups
        xf = x.float().view(a, b, g, -1)
        coef = base.float().view(1, 1, taps, g, -1) + delta.float().unsqueeze(-1)  # [A, B, taps, groups, group]
        out = coef[:, :, 0] * xf
        for t in range(1, taps):
            out[:, t:] += coef[:, t:, t] * xf[:, :-t]
        return out.view(a, b, -1).to(x.dtype)

    def _wrap(self, prefix: str, h: torch.Tensor):
        cfg = self.config
        delta = F.linear(h, self.w[prefix + "kernel_projection.weight"])
        delta = delta.view(*h.shape[:2], 2, cfg.conv_taps, self.groups)
        base = self.w[prefix + "base_kernel"]  # [2 sides, taps, hidden]
        return self._conv(h, delta[:, :, 0], base[0]), lambda y: self._conv(y, delta[:, :, 1], base[1])

    def _attention(self, i: int, x: torch.Tensor, ctx: DraftContext, starts: torch.Tensor, cos, sin) -> torch.Tensor:
        cfg = self.config
        p = f"layers.{i}.self_attn."
        a, b, _ = x.shape
        q = F.linear(x, self.w[p + "q_proj.weight"]).view(a, b, cfg.num_heads, cfg.head_dim)
        k = F.linear(x, self.w[p + "k_proj.weight"]).view(a, b, cfg.num_kv_heads, cfg.head_dim)
        v = F.linear(x, self.w[p + "v_proj.weight"]).view(a, b, cfg.num_kv_heads, cfg.head_dim)
        q = _rope(rms_norm(q, self.w[p + "q_norm.weight"], cfg.eps), cos, sin)
        k = kv_round.draft(_rope(rms_norm(k, self.w[p + "k_norm.weight"], cfg.eps), cos, sin))
        v = kv_round.draft(v)  # (the block's K / V go into the device's cache before its attention)
        rep = cfg.num_heads // cfg.num_kv_heads
        qf = q.float().transpose(1, 2)  # [A, heads, B, d]
        kc = ctx.k[i].float().transpose(0, 1).repeat_interleave(rep, dim=0)  # [heads, S, d]
        vc = ctx.v[i].float().transpose(0, 1).repeat_interleave(rep, dim=0)
        kb = k.float().transpose(1, 2).repeat_interleave(rep, dim=1)  # [A, heads, B, d]
        vb = v.float().transpose(1, 2).repeat_interleave(rep, dim=1)
        scale = cfg.head_dim**-0.5
        s_ctx = torch.einsum("ahbd,hsd->ahbs", qf, kc) * scale
        visible = torch.arange(kc.shape[1])[None, :] < starts[:, None]  # [A, S]: the context before the bonus token
        s_ctx = s_ctx.masked_fill(~visible[:, None, None, :], float("-inf"))
        s_blk = torch.einsum("ahbd,ahcd->ahbc", qf, kb) * scale
        prob = torch.softmax(torch.cat([s_ctx, s_blk], dim=-1), dim=-1)
        n = kc.shape[1]
        out = torch.einsum("ahbs,hsd->ahbd", prob[..., :n], vc) + torch.einsum("ahbc,ahcd->ahbd", prob[..., n:], vb)
        out = bf16(out.transpose(1, 2).reshape(a, b, -1))
        return F.linear(out, self.w[p + "o_proj.weight"])

    def _mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        p = f"layers.{i}.mlp."
        g = F.linear(x, self.w[p + "gate_proj.weight"])
        u = F.linear(x, self.w[p + "up_proj.weight"])
        h = bf16(F.silu(g.float()) * u.float())
        return F.linear(h, self.w[p + "down_proj.weight"])

    def forward(self, ctx: DraftContext, starts: torch.Tensor, bonus: torch.Tensor, k: int) -> torch.Tensor:
        """starts [A] the bonus positions s (the context is 0 .. s - 1), bonus [A] the tokens t[s], k mask slots ->
        the final normed hidden states [A, k + 1, hidden] bf16 of positions s .. s + k."""
        cfg = self.config
        a = starts.shape[0]
        x = torch.empty(a, k + 1, cfg.hidden, dtype=torch.bfloat16)
        x[:, 0] = F.embedding(bonus, self.w["embed_tokens.weight"])
        x[:, 1:] = self.mask_embedding
        cos, sin = _rope_cos_sin(starts[:, None] + torch.arange(k + 1)[None, :], cfg.head_dim, cfg.rope_theta)
        for i in range(cfg.num_layers):
            p = f"layers.{i}."
            h = rms_norm(x, self.w[p + "input_layernorm.weight"], cfg.eps)
            h, finish = self._wrap(p + "attention_conv.", h)
            x = bf16(x.float() + finish(self._attention(i, h, ctx, starts, cos, sin)).float())
            h = rms_norm(x, self.w[p + "post_attention_layernorm.weight"], cfg.eps)
            h, finish = self._wrap(p + "mlp_conv.", h)
            x = bf16(x.float() + finish(self._mlp(i, h)).float())
        return rms_norm(x, self.w["norm.weight"], cfg.eps)

    # -- candidates and the selector -----------------------------------------------------------------------------
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden.float(), self.lm_head32)

    def propose(self, hidden: torch.Tensor, bonus: torch.Tensor, reference: torch.Tensor | None = None) -> Proposal:
        """hidden [A, B, hidden] the final hidden states of the block (slot 0 the bonus token), bonus [A] ->
        the K = B - 1 drafts.  reference [A, K] (optional): the tokens to look for among each slot's candidates."""
        cfg = self.config
        h = hidden[:, 1:]  # the mask slots
        a, k, _ = h.shape
        top = cfg.selector_top_k
        unary, cand = self.logits(h).topk(top, dim=-1)  # [A, K, top]
        proj = F.linear(h, self.w["candidate_selector.hidden_projection.weight"]).float()  # [A, K, rank]
        pred_ids = torch.cat([bonus[:, None, None].expand(a, 1, top), cand[:, :-1]], dim=1)
        pred = self.w["candidate_selector.predecessor_codebook"][pred_ids].float()  # [A, K, top, rank]
        succ = self.w["candidate_selector.successor_codebook"][cand].float()
        scores = unary[:, :, None, :] + torch.einsum("alpr,alcr->alpc", pred * proj[:, :, None, :], succ)
        rows = torch.arange(a)
        prev = torch.zeros(a, dtype=torch.long)  # the first slot's predecessors are all the anchor
        drafts = torch.empty(a, k, dtype=torch.long)
        for slot in range(k):
            idx = scores[rows, slot, prev].argmax(-1)
            drafts[:, slot] = cand[rows, slot, idx]
            prev = idx
        in_top = (
            (cand == reference[..., None]).any(-1) if reference is not None else torch.zeros(a, k, dtype=torch.bool)
        )
        return Proposal(drafts=drafts, plain=cand[..., 0], in_top_k=in_top)
