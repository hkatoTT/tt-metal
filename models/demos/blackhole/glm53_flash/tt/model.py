"""The text model: embedding -> 4 mHC streams -> 45 decoder layers -> mean of the streams -> norm -> LM head.

The LM head is computed in fp32 (bf16 hidden state and weights, upcast), the reference precision of the acceptance
records.  The MTP layer (45) is not part of this forward; HF does not load it either.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tt.common import bf16, rms_norm
from models.demos.blackhole.glm53_flash.tt.layer import DecoderLayer, LayerAux
from models.demos.blackhole.glm53_flash.tt.mhc import collapse_streams, expand_streams

PREFIX = "model.language_model."


@dataclass
class ModelState:
    layers: list  # per layer: KdaState | DsaState | None
    length: int = 0


class Glm53TextModel:
    def __init__(self, checkpoint: Glm53Checkpoint, config: Glm53Config | None = None, num_layers: int | None = None):
        self.checkpoint = checkpoint
        self.config = config or Glm53Config.from_dict(checkpoint.config)
        n = self.config.num_layers if num_layers is None else num_layers
        self.embed = bf16(checkpoint.tensor(PREFIX + "embed_tokens.weight"))
        self.layers = [DecoderLayer(checkpoint, self.config, i) for i in range(n)]
        self.norm = bf16(checkpoint.tensor(PREFIX + "norm.weight"))
        self.lm_head = bf16(checkpoint.tensor("lm_head.weight"))

    def empty_state(self) -> ModelState:
        return ModelState(layers=[None] * len(self.layers))

    def hidden(self, input_ids: torch.Tensor, state: ModelState | None = None):
        """input_ids [B, S] -> (final normed hidden [B, S, hidden] bf16, state, per-layer aux)."""
        state = state or self.empty_state()
        streams = expand_streams(F.embedding(input_ids, self.embed), self.config.hc_mult)
        aux: list[LayerAux] = []
        for i, layer in enumerate(self.layers):
            streams, state.layers[i], layer_aux = layer(streams, state.layers[i])
            aux.append(layer_aux)
        state.length += input_ids.shape[1]
        return rms_norm(collapse_streams(streams), self.norm, self.config.rms_norm_eps), state, aux

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden.float(), self.lm_head.float())

    def __call__(self, input_ids: torch.Tensor, state: ModelState | None = None):
        hidden, state, aux = self.hidden(input_ids, state)
        return self.logits(hidden), state, aux
