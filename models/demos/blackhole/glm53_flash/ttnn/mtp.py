"""The MTP layer (45) on the 8x4 mesh: the draft of MTP self-speculative decode (the CPU reference `tt/mtp.py`).

A plain decoder layer (no mHC) on eh_proj(cat(enorm(embed(t[i+1])), hnorm(h[i]))): DSA attention and MoE with pre-norm
residuals, the layers' own parts (`TtDsa`, `TtMoe`, `TtRmsNorm`) with layer 45's weights; its output goes through
shared_head.norm and the main model's LM head.  eh_proj [8192 -> 4096] is split by its output features over a row's
chips and all-gathered (as the embedding is).  The attention keeps its own cache, at the positions the main model's
tokens take.
"""

from __future__ import annotations

import ttnn
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tt.common import LAYER_PREFIX
from models.demos.blackhole.glm53_flash.tt.dsa import DsaWeights
from models.demos.blackhole.glm53_flash.tt.moe import MoeWeights
from models.demos.blackhole.glm53_flash.ttnn.common import HIFI4_FP32, TP_AXIS, decode_linear, tp_weight
from models.demos.blackhole.glm53_flash.ttnn.dsa import TtDsa
from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm
from models.demos.blackhole.glm53_flash.ttnn.moe import TtMoe


class TtMtp:
    def __init__(
        self,
        mesh,
        checkpoint,
        config: Glm53Config,
        max_seq: int,
        expert_dtype=ttnn.bfloat16,
        dtype=ttnn.bfloat16,
    ) -> None:
        layer = config.num_layers
        p = LAYER_PREFIX.format(layer)
        t = checkpoint.tensor
        eps = config.rms_norm_eps
        self.mesh = mesh
        self.enorm = TtRmsNorm(mesh, t(p + "enorm.weight"), eps)
        self.hnorm = TtRmsNorm(mesh, t(p + "hnorm.weight"), eps)
        self.eh_proj = tp_weight(mesh, t(p + "eh_proj.weight"), "out", dtype)  # [8192, 1024] per chip
        self.input_norm = TtRmsNorm(mesh, t(p + "input_layernorm.weight"), eps)
        self.post_norm = TtRmsNorm(mesh, t(p + "post_attention_layernorm.weight"), eps)
        self.head_norm = TtRmsNorm(mesh, t(p + "shared_head.norm.weight"), eps)
        self.attention = TtDsa(
            mesh, DsaWeights.from_checkpoint(checkpoint, layer), config, max_seq=max_seq, dtype=dtype
        )
        self.mlp = TtMoe(mesh, MoeWeights(checkpoint, layer), config, expert_dtype=expert_dtype, dtype=dtype)

    def empty_state(self, batch: int):
        return self.attention.empty_state(batch)

    def _input(self, embeds, hidden):
        """embeds, hidden [1, 1, B, 4096] bf16 (the same on a row's chips) -> eh_proj(cat(enorm(e), hnorm(h)))."""
        e, h = self.enorm(embeds), self.hnorm(hidden)
        cat = ttnn.concat([e, h], dim=-1)  # [1, 1, B, 8192]
        ttnn.deallocate(e)
        ttnn.deallocate(h)
        quarter = decode_linear(self.mesh, cat, self.eh_proj, compute_kernel_config=HIFI4_FP32)  # [1, 1, B, 1024]
        ttnn.deallocate(cat)
        x = ttnn.all_gather(quarter, dim=3, cluster_axis=TP_AXIS)
        ttnn.deallocate(quarter)
        return x

    def step(self, embeds, hidden, state, positions, keep: bool = False):
        """A decode step: embeds [1, 1, B, 4096] (each position's next token's embedding), hidden [1, 1, B, 4096]
        (the main model's final normed hidden state there, or a chained step's: the layer's own output) , positions
        (`TtDsa.alloc_positions`) -> (the layer's output [1, 1, B, 4096] bf16 before shared_head.norm, state); `keep`:
        and the attention's input (for `rewrite`)."""
        x = self._input(embeds, hidden)
        normed = self.input_norm(x)
        a, state = self.attention.step(normed, state, positions)
        if keep:
            kept = normed
        else:
            ttnn.deallocate(normed)
        x2 = ttnn.add(x, a)
        ttnn.deallocate(x)
        ttnn.deallocate(a)
        normed = self.post_norm(x2)
        m = self.mlp(normed)
        ttnn.deallocate(normed)
        y = ttnn.add(x2, m)
        ttnn.deallocate(x2)
        ttnn.deallocate(m)
        if keep:
            return y, state, kept
        return y, state

    def rewrite(self, kept, state, positions) -> None:
        """The cache rows of a step kept with `step(.., keep=True)` written again at its positions (a chained step at
        later positions wrote over them); `kept` is freed."""
        self.attention.rewrite(kept, state, positions)
        ttnn.deallocate(kept)

    def prefill(self, embeds, hidden, state, start: int, tables):
        """A prefill chunk: embeds [1, 1, T, 4096] (each position's next token's embedding), hidden [1, 1, T, 4096]
        (the main model's final normed hidden states there) -> state, the layer's cache written at the chunk's
        positions (`TtDsa.prefill`, the main model's tables).  Only the cache is kept: the layer's output feeds the
        drafts alone, so the MoE is left out here."""
        x = self._input(embeds, hidden)
        normed = self.input_norm(x)
        ttnn.deallocate(x)
        a, state = self.attention.prefill(normed, state, start, tables)
        ttnn.deallocate(normed)
        ttnn.deallocate(a)
        return state

    def head_input(self, y):
        """The layer's output -> what the main LM head reads."""
        return self.head_norm(y)
