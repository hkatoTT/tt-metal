"""One decoder layer: mHC read -> norm -> attention (KDA or DSA) -> mHC write, then the same around the MLP."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.demos.blackhole.glm53_flash.config import KDA, Glm53Config
from models.demos.blackhole.glm53_flash.tt.common import LAYER_PREFIX, Mlp, MlpWeights, bf16, rms_norm
from models.demos.blackhole.glm53_flash.tt.dsa import Dsa, DsaState, DsaWeights, IndexerResult
from models.demos.blackhole.glm53_flash.tt.kda import Kda, KdaState, KdaWeights
from models.demos.blackhole.glm53_flash.tt.mhc import HyperConnection, HyperConnectionWeights
from models.demos.blackhole.glm53_flash.tt.moe import Moe, MoeWeights, Routing


@dataclass(frozen=True)
class LayerAux:
    routing: Routing | None  # None for the dense layers
    selection: IndexerResult | None  # None for the KDA layers


class DecoderLayer:
    def __init__(self, checkpoint, config: Glm53Config, layer: int) -> None:
        if not 0 <= layer < config.num_layers:
            raise ValueError(f"layer {layer} is outside the {config.num_layers}-layer backbone")
        self.layer = layer
        self.kind = config.layer_types[layer]
        p = LAYER_PREFIX.format(layer)
        t = checkpoint.tensor
        hc = dict(
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.hc_sinkhorn_iters,
            hc_eps=config.hc_eps,
            norm_eps=config.rms_norm_eps,
        )
        self.attn_hc = HyperConnection(HyperConnectionWeights.from_checkpoint(checkpoint, layer, "attn"), **hc)
        self.ffn_hc = HyperConnection(HyperConnectionWeights.from_checkpoint(checkpoint, layer, "ffn"), **hc)
        self.input_norm = bf16(t(p + "input_layernorm.weight"))
        self.post_attention_norm = bf16(t(p + "post_attention_layernorm.weight"))
        self.eps = config.rms_norm_eps
        if self.kind == KDA:
            self.attention = Kda(KdaWeights.from_checkpoint(checkpoint, layer), config)
        else:
            self.attention = Dsa(DsaWeights.from_checkpoint(checkpoint, layer), config)
        if config.is_moe(layer):
            self.mlp = Moe(MoeWeights(checkpoint, layer), config)
        else:
            self.mlp = Mlp(MlpWeights.from_checkpoint(checkpoint, p + "mlp."), config.swiglu_limit)

    def __call__(
        self, streams: torch.Tensor, state: KdaState | DsaState | None = None
    ) -> tuple[torch.Tensor, KdaState | DsaState, LayerAux]:
        """streams [B, S, n, hidden] bf16 -> (streams, attention state after these tokens, aux)."""
        post, comb, x = self.attn_hc.read(streams)
        x = rms_norm(x, self.input_norm, self.eps)
        selection = None
        if self.kind == KDA:
            x, state = self.attention(x, state)
        else:
            x, state, selection = self.attention(x, state)
        streams = HyperConnection.write(x, streams, post, comb)

        post, comb, x = self.ffn_hc.read(streams)
        x = rms_norm(x, self.post_attention_norm, self.eps)
        routing = None
        if isinstance(self.mlp, Moe):
            x, routing = self.mlp(x)
        else:
            x = self.mlp(x)
        streams = HyperConnection.write(x, streams, post, comb)
        return streams, state, LayerAux(routing=routing, selection=selection)
