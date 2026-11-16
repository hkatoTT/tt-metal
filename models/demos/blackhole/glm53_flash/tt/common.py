"""Norms and the clamped SwiGLU MLP shared by the components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

LAYER_PREFIX = "model.language_model.layers.{}."


def bf16(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm computed in fp32, rounded to the input dtype, then scaled by the weight in its own dtype."""
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(dtype)


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm without a weight; the reciprocal is computed in fp32 and rounded to the input dtype."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm, weight and sigmoid gate all in fp32, rounded to the input dtype at the end (the KDA output norm)."""
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight.to(torch.float32) * h
    h = h * torch.sigmoid(gate.to(torch.float32))
    return h.to(dtype)


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """silu(min(gate, limit)) * clamp(up, -limit, limit), in the inputs' dtype."""
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


@dataclass(frozen=True)
class MlpWeights:
    gate: torch.Tensor  # [intermediate, hidden]
    up: torch.Tensor  # [intermediate, hidden]
    down: torch.Tensor  # [hidden, intermediate]

    @classmethod
    def from_checkpoint(cls, checkpoint, prefix: str) -> "MlpWeights":
        t = checkpoint.tensor
        return cls(
            gate=bf16(t(prefix + "gate_proj.weight")),
            up=bf16(t(prefix + "up_proj.weight")),
            down=bf16(t(prefix + "down_proj.weight")),
        )


class Mlp:
    """The clamped SwiGLU MLP: the dense MLP of the first layers and the shared expert of the MoE layers."""

    def __init__(self, weights: MlpWeights, swiglu_limit: float) -> None:
        self.w = weights
        self.limit = swiglu_limit

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = clamped_swiglu(F.linear(x, self.w.gate), F.linear(x, self.w.up), self.limit)
        return F.linear(h, self.w.down)
