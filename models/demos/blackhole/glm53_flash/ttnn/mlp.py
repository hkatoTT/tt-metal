"""The clamped SwiGLU MLP (dense layers 0-2 and the shared expert) on the 8x4 mesh.

Tensor parallel over the 4 chips of a row: gate and up split by output feature (each chip holds a quarter of the
intermediate, its gate and up quarters side by side in one weight: one matmul with twice the output tiles, so on twice
the cores at decode), down split by input feature, so each chip's down output is a partial sum; one all-reduce over
the row's chips adds them.  Every row holds the same weights and works on its own users.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.glm53_flash.tt.common import MlpWeights
from models.demos.blackhole.glm53_flash.ttnn import swiglu_split
from models.demos.blackhole.glm53_flash.ttnn.common import (
    COLS,
    HIFI4_FP32,
    ar_input_config,
    decode_linear,
    swiglu_activations,
    tp_all_reduce,
    tp_weight,
)

# the clamped SwiGLU from gate | up in one op (swiglu_split, the same bits as the two slices and the multiply; 36.4 ->
# 17.0 us at 1,024 tokens of the shared expert, 131 -> 65 us of a dense layer): 0 off, 1 the prefill (not a decode
# step; the default), 2 every call
SWIGLU_FUSED = 1


def _fused(gate_up) -> bool:
    if SWIGLU_FUSED == 0 or not swiglu_split.usable(gate_up):
        return False
    if SWIGLU_FUSED == 1:
        from .moe_route import in_decode  # (it imports common, which this module imports)

        return not in_decode()
    return True


class TtMlp:
    def __init__(self, mesh, weights: MlpWeights, swiglu_limit: float, dtype=ttnn.bfloat16) -> None:
        self.mesh = mesh
        self.limit = swiglu_limit
        self.quarter = q = weights.gate.shape[0] // COLS
        # chip c's block of the output features: its gate quarter, then its up quarter
        g, u = weights.gate, weights.up
        both = torch.cat([torch.cat([g[c * q : (c + 1) * q], u[c * q : (c + 1) * q]]) for c in range(COLS)])
        self.gate_up = tp_weight(mesh, both, "out", dtype)
        self.down = tp_weight(mesh, weights.down, "in", dtype)

    def __call__(self, x):
        """x [1, 1, S, hidden] bf16, the same on the 4 chips of a row -> the same layout."""
        partial = self.partial(x, memory_config=ar_input_config(self.mesh, x.shape[-2], x.shape[-1]))
        out = tp_all_reduce(self.mesh, partial)
        ttnn.deallocate(partial)
        return out

    def partial(self, x, memory_config=None):
        """This chip's share of the output before the all-reduce over the row (to fold into another all-reduce);
        `memory_config` (`ar_input_config`): where the down matmul writes it."""
        gate_up = decode_linear(self.mesh, x, self.gate_up, compute_kernel_config=HIFI4_FP32)  # [1, 1, S, 2 q]
        rows, q = gate_up.shape[-2], self.quarter
        if _fused(gate_up):
            h = swiglu_split.swiglu(self.mesh, gate_up, q, self.limit)
            ttnn.deallocate(gate_up)
        else:
            gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, rows, q])
            up = ttnn.slice(gate_up, [0, 0, 0, q], [1, 1, rows, 2 * q])
            ttnn.deallocate(gate_up)
            h = ttnn.multiply(gate, up, **swiglu_activations(self.limit))
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
        partial = decode_linear(self.mesh, h, self.down, compute_kernel_config=HIFI4_FP32, memory_config=memory_config)
        ttnn.deallocate(h)
        return partial

    def deallocate(self) -> None:
        for t in (self.gate_up, self.down):
            ttnn.deallocate(t)
