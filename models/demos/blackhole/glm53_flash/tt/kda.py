"""Kimi delta attention (KDA): the linear attention of 34 of the 45 layers.

Per layer: q, k, v projections (64 heads x 128), a causal depthwise conv of width 4 with SiLU over the concatenated
q|k|v channels, a per-channel forget gate g in (lower_bound, 0) (log decay), a per-head write strength beta, and the
delta rule on a [128 x 128] fp32 state per head:

    S <- S * exp(g_t)                    (decay, per key channel)
    S <- S + k_t (beta_t (v_t - k_t^T S))  (delta-rule write)
    o_t = q_t^T S / sqrt(128)            (q and k L2-normalized first)

The output goes through an RMS norm gated by sigmoid(g_b(g_a(x))) and o_proj.  Prefill uses the chunked form (64
tokens per chunk) and decode the per-token recurrence, as HF does; the two are equal up to rounding.

The state between calls is the last 4 conv inputs and the fp32 delta-rule state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tt.common import LAYER_PREFIX, bf16, gated_rms_norm

CHUNK = 64


@dataclass(frozen=True)
class KdaWeights:
    q_proj: torch.Tensor  # [H * Dh, hidden]
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    conv: torch.Tensor  # [3 * H * Dh, kernel] fp32, channels q | k | v
    f_a: torch.Tensor  # [Dh, hidden]
    f_b: torch.Tensor  # [H * Dh, Dh]
    dt_bias: torch.Tensor  # [H * Dh] fp32
    a_log: torch.Tensor  # [H] fp32
    b_proj: torch.Tensor  # [H, hidden]
    g_a: torch.Tensor  # [Dh, hidden]
    g_b: torch.Tensor  # [H * Dh, Dh]
    o_norm: torch.Tensor  # [Dh]
    o_proj: torch.Tensor  # [hidden, H * Dh]

    @classmethod
    def from_checkpoint(cls, checkpoint, layer: int) -> "KdaWeights":
        p = LAYER_PREFIX.format(layer) + "self_attn."
        t = checkpoint.tensor
        convs = [t(p + f"{name}_conv1d.weight") for name in "qkv"]
        kernel = convs[0].shape[-1]
        return cls(
            q_proj=bf16(t(p + "q_proj.weight")),
            k_proj=bf16(t(p + "k_proj.weight")),
            v_proj=bf16(t(p + "v_proj.weight")),
            conv=torch.cat([c.reshape(-1, kernel) for c in convs]).float(),
            f_a=bf16(t(p + "f_a_proj.weight")),
            f_b=bf16(t(p + "f_b_proj.weight")),
            dt_bias=t(p + "dt_bias").float(),
            a_log=t(p + "A_log").float(),
            b_proj=bf16(t(p + "b_proj.weight")),
            g_a=bf16(t(p + "g_a_proj.weight")),
            g_b=bf16(t(p + "g_b_proj.weight")),
            o_norm=bf16(t(p + "o_norm.weight")),
            o_proj=bf16(t(p + "o_proj.weight")),
        )


@dataclass(frozen=True)
class KdaState:
    conv: torch.Tensor  # [B, 3 * H * Dh, kernel] bf16: the last conv inputs (zeros before the first token)
    recurrent: torch.Tensor  # [B, H, Dk, Dv] fp32

    @property
    def batch(self) -> int:
        return self.recurrent.shape[0]


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def kda_recurrent(q, k, v, g, beta, state):
    """Token-by-token delta rule.  q, k, v, g: [B, S, H, D]; beta: [B, S, H]; state [B, H, Dk, Dv] fp32 or None."""
    dtype = q.dtype
    q, k, v, g, beta = (x.to(torch.float32) for x in (q, k, v, g, beta))
    q, k = l2norm(q), l2norm(k)
    b, s, h, dk = k.shape
    dv = v.shape[-1]
    q = q * (1 / (q.shape[-1] ** 0.5))
    out = torch.zeros(b, s, h, dv, dtype=v.dtype)
    s_t = torch.zeros(b, h, dk, dv, dtype=v.dtype) if state is None else state.to(v)
    for i in range(s):
        s_t = s_t * g[:, i][..., None].exp()
        remembered = (s_t * k[:, i][..., None]).sum(dim=-2)
        delta = (v[:, i] - remembered) * beta[:, i][..., None]
        s_t = s_t + k[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (s_t * q[:, i].unsqueeze(-1)).sum(dim=-2)
    return out.to(dtype), s_t


def kda_chunked(q, k, v, g, beta, state, chunk: int = CHUNK):
    """The chunked form of the same recurrence (WY representation within a chunk, the state carried across chunks)."""
    dtype = q.dtype
    q, k, v, beta, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g))
    q, k = l2norm(q), l2norm(k)
    b, h, s, dk = k.shape
    dv = v.shape[-1]
    pad = (chunk - s % chunk) % chunk
    total = s + pad

    q = F.pad(q, (0, 0, 0, pad)) * (1 / (dk**0.5))
    k = F.pad(k, (0, 0, 0, pad))
    v = F.pad(v, (0, 0, 0, pad))
    g = F.pad(g, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, v, g, k_beta, v_beta = (x.reshape(b, h, -1, chunk, x.shape[-1]) for x in (q, k, v, g, k_beta, v_beta))

    g = g.cumsum(dim=-2)  # log decay from the chunk start, per key channel
    upper = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=0)
    strictly_upper = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=1)
    # decay[i, j] = exp(g_i - g_j) per channel for j <= i
    decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(strictly_upper[..., None], float("-inf")).exp().float()
    a = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(dim=-1).masked_fill(upper, 0)
    for i in range(1, chunk):  # forward substitution: (I - A)^-1 row by row
        row = a[..., i, :i].clone()
        sub = a[..., :i, :i].clone()
        a[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    a = a + torch.eye(chunk, dtype=a.dtype)
    u = a @ v_beta
    w = a @ (k_beta * g.exp())

    s_t = torch.zeros(b, h, dk, dv, dtype=u.dtype) if state is None else state.to(u)
    out = torch.zeros_like(u)
    for c in range(total // chunk):
        q_c, k_c, u_c, g_c = q[:, :, c], k[:, :, c], u[:, :, c], g[:, :, c]
        inter = (q_c * g_c.exp()) @ s_t
        intra = (q_c.unsqueeze(-2) * k_c.unsqueeze(-3) * decay[:, :, c]).sum(dim=-1).masked_fill(strictly_upper, 0)
        v_new = u_c - w[:, :, c] @ s_t
        out[:, :, c] = inter + intra @ v_new
        s_t = s_t * g_c[:, :, -1].exp().unsqueeze(-1) + (k_c * (g_c[:, :, -1:] - g_c).exp()).transpose(-1, -2) @ v_new

    out = out.reshape(b, h, -1, dv)[:, :, :s]
    return out.transpose(1, 2).contiguous().to(dtype), s_t


class Kda:
    def __init__(self, weights: KdaWeights, config: Glm53Config) -> None:
        self.w = weights
        self.heads = config.kda_num_heads
        self.head_dim = config.kda_head_dim
        self.kernel = config.kda_conv_kernel
        self.lower_bound = config.kda_gate_lower_bound
        self.eps = config.rms_norm_eps

    def empty_state(self, batch: int) -> KdaState:
        channels = 3 * self.heads * self.head_dim
        return KdaState(
            conv=torch.zeros(batch, channels, self.kernel, dtype=torch.bfloat16),
            recurrent=torch.zeros(batch, self.heads, self.head_dim, self.head_dim, dtype=torch.float32),
        )

    def short_conv(self, mixed: torch.Tensor, previous: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal depthwise conv + SiLU in fp32 over [B, C, S]; returns (out [B, C, S] fp32, the window it read).

        The arithmetic is HF's for each case (CPU conv results depend on the input length and padding, so the window
        shapes are kept as HF has them): the first call pads like `causal_conv1d_fn` (a window shorter than the kernel
        is left-padded to the kernel first, as HF's cache does); a one-token step convolves the last `kernel` inputs and
        the new one without padding, as `causal_conv1d_update`; a multi-token continuation convolves the cached window
        and the new inputs with the causal padding and keeps the new positions.
        """
        s = mixed.shape[-1]
        weight = self.w.conv.unsqueeze(1)
        if previous is None:
            window = F.pad(mixed, (self.kernel - s, 0)) if s < self.kernel else mixed
        else:
            window = torch.cat([previous, mixed], dim=-1)
        if previous is not None and s == 1:
            out = F.conv1d(window.to(torch.float32), weight, groups=window.shape[1])[..., -1:]
        else:
            out = F.conv1d(window.to(torch.float32), weight, padding=self.kernel - 1, groups=window.shape[1])
            out = out[..., : window.shape[-1]][..., -s:]
        return F.silu(out), window

    def forget_gate(self, x: torch.Tensor) -> torch.Tensor:
        """g in (lower_bound, 0): lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)), fp32 [B, S, H, Dh]."""
        b, s = x.shape[:2]
        f = F.linear(F.linear(x, self.w.f_a), self.w.f_b)
        g = (f.float() + self.w.dt_bias.view(1, 1, -1)).view(b, s, -1, self.head_dim)
        rate = torch.exp(self.w.a_log.view(1, 1, self.heads, 1))
        return self.lower_bound * torch.sigmoid(rate * g)

    def __call__(self, x: torch.Tensor, state: KdaState | None = None) -> tuple[torch.Tensor, KdaState]:
        """x [B, S, hidden] bf16 -> (out [B, S, hidden], state after the last token)."""
        b, s = x.shape[:2]
        shape = (b, s, -1, self.head_dim)
        decode = state is not None and s == 1

        mixed = torch.cat([F.linear(x, self.w.q_proj), F.linear(x, self.w.k_proj), F.linear(x, self.w.v_proj)], dim=-1)
        mixed = mixed.transpose(1, 2)  # [B, C, S]
        conv, window = self.short_conv(mixed, None if state is None else state.conv)
        conv = conv.to(x.dtype)
        q, k, v = (t.reshape(shape) for t in conv.transpose(1, 2).split(self.heads * self.head_dim, dim=-1))

        g = self.forget_gate(x)
        beta = torch.sigmoid(F.linear(x, self.w.b_proj))
        initial = None if state is None else state.recurrent
        if decode:
            core, recurrent = kda_recurrent(q, k, v, g, beta, initial)
        else:
            core, recurrent = kda_chunked(q, k, v, g, beta, initial)

        gate = F.linear(F.linear(x, self.w.g_a), self.w.g_b).view(shape)
        out = gated_rms_norm(core, gate, self.w.o_norm, self.eps).reshape(b, s, -1)
        out = F.linear(out, self.w.o_proj)
        return out, KdaState(conv=window[..., -self.kernel :].contiguous(), recurrent=recurrent.float())
