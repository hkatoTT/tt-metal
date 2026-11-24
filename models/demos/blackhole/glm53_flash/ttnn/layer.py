"""One decoder layer on the 8x4 mesh, one-token decode (v0): mHC around the attention and around the MLP."""

from __future__ import annotations

import contextlib

import ttnn
from models.demos.blackhole.glm53_flash.config import KDA, Glm53Config
from models.demos.blackhole.glm53_flash.tt.common import LAYER_PREFIX, MlpWeights
from models.demos.blackhole.glm53_flash.tt.dsa import DsaWeights
from models.demos.blackhole.glm53_flash.tt.kda import KdaWeights
from models.demos.blackhole.glm53_flash.tt.mhc import HyperConnectionWeights
from models.demos.blackhole.glm53_flash.tt.moe import MoeWeights
from models.demos.blackhole.glm53_flash.ttnn import mhc
from models.demos.blackhole.glm53_flash.ttnn.common import (
    COLS,
    ar_keep_l1,
    row_quarter,
    row_split_on,
    row_tokens_gather,
    to_row_quarter,
)
from models.demos.blackhole.glm53_flash.ttnn.dsa import TtDsa
from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm
from models.demos.blackhole.glm53_flash.ttnn.kda import TtKda
from models.demos.blackhole.glm53_flash.ttnn.mhc import TtHyperConnection, collapse_streams
from models.demos.blackhole.glm53_flash.ttnn.mlp import TtMlp
from models.demos.blackhole.glm53_flash.ttnn.moe import TtMoe, w1_decode


def _by_halves(x, state, halves, run):
    """A pair of sequence-parallel waves' attention: x [1, 1, 2 T, D] (the first wave's
    segment rows, then the second's) through `run(x_half, state, half)` -> (out, state) one wave after the other (the
    second sees the first's state and cache, as a wave after another does) -> (out [1, 1, 2 T, D], state)."""
    t, d = x.shape[-2] // len(halves), x.shape[-1]
    outs = []
    for i, half in enumerate(halves):
        part = ttnn.slice(x, [0, 0, i * t, 0], [1, 1, (i + 1) * t, d])
        y, state = run(part, state, half)
        ttnn.deallocate(part)
        outs.append(y)
    out = ttnn.concat(outs, dim=2)
    for y in outs:
        ttnn.deallocate(y)
    return out, state


class TtDecoderLayer:
    def __init__(
        self,
        mesh,
        checkpoint,
        config: Glm53Config,
        layer: int,
        max_seq: int,
        expert_dtype=ttnn.bfloat16,
        dtype=ttnn.bfloat16,
    ) -> None:
        self.kind = config.layer_types[layer]
        p = LAYER_PREFIX.format(layer)
        hc = dict(
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.hc_sinkhorn_iters,
            hc_eps=config.hc_eps,
            norm_eps=config.rms_norm_eps,
        )
        self.attn_hc = TtHyperConnection(mesh, HyperConnectionWeights.from_checkpoint(checkpoint, layer, "attn"), **hc)
        self.ffn_hc = TtHyperConnection(mesh, HyperConnectionWeights.from_checkpoint(checkpoint, layer, "ffn"), **hc)
        self.input_norm = TtRmsNorm(mesh, checkpoint.tensor(p + "input_layernorm.weight"), config.rms_norm_eps)
        self.post_attention_norm = TtRmsNorm(
            mesh, checkpoint.tensor(p + "post_attention_layernorm.weight"), config.rms_norm_eps
        )
        if self.kind == KDA:
            self.attention = TtKda(mesh, KdaWeights.from_checkpoint(checkpoint, layer), config, dtype=dtype)
        else:
            self.attention = TtDsa(
                mesh, DsaWeights.from_checkpoint(checkpoint, layer), config, max_seq=max_seq, dtype=dtype
            )
        if config.is_moe(layer):
            self.mlp = TtMoe(mesh, MoeWeights(checkpoint, layer), config, expert_dtype=expert_dtype, dtype=dtype)
        else:
            self.mlp = TtMlp(mesh, MlpWeights.from_checkpoint(checkpoint, p + "mlp."), config.swiglu_limit, dtype)

    def empty_state(self, batch: int, spec: int = 0):
        """spec: KDA's state of a verify step with that many drafts (`TtKda.step_spec`)."""
        if self.kind == KDA:
            return self.attention.empty_state(batch, spec=spec)
        return self.attention.empty_state(batch)

    def step(
        self, streams, state, device_positions, tokens: int, mark=None, spec=None, pending=None, defer=False, tap=None
    ):
        """One decode step: streams [1, 1, 4 Tp, 4096] bf16 (ttnn/mhc.py) of the row's `tokens` users (one token
        each) -> (streams, state).

        `mark` (a callable taking a part name), if given, is called after every part (attn_hc.read, input_norm,
        attention, attn_hc.write, ffn_hc.read, post_attention_norm, mlp), e.g. to split a device profile; the rest of
        the step is the ffn_hc.write.  `spec` (`kda.SpecKeep`): a verify step, each user's token and its K drafts ((K
        + 1) B tokens; DSA's from the virtual users of `device_positions`).

        With mhc.WRITE_READ the layers chain their mHC writes into the next reads (`mhc_stream_write_read`): `pending`
        (the previous layer's deferred FFN write: (its mHC, block_out, streams, post, comb)) is written here together
        with this layer's first read (`streams` None: the layer makes and frees its input), and `defer` returns this
        layer's FFN write the same way in place of the new streams (every layer but the last).  `tap` (a list): this
        layer's input streams' per-token mean [1, 1, T, 4096] (the previous layer's output, as the DFlash2 drafter takes
        it) appended to it."""
        kw = dict(decode=True, pending=pending, defer=defer, tap=tap)
        if self.kind == KDA and spec is not None:
            return self._forward(streams, tokens, lambda x: self.attention.step_spec(x, state, spec), mark, **kw)
        if self.kind == KDA:
            return self._forward(streams, tokens, lambda x: self.attention.step(x, state), mark, **kw)
        return self._forward(streams, tokens, lambda x: self.attention.step(x, state, device_positions), mark, **kw)

    def prefill(
        self,
        streams,
        state,
        tokens: int,
        start: int,
        keep,
        mark=None,
        kda_picks=None,
        dsa_tables=None,
        tap=None,
        pending=None,
        defer=False,
        halves=None,
    ):
        """A prefill chunk: streams of `tokens` consecutive tokens (positions start ..) of each mesh row's request, on
        the rows whose `keep` is 1 (the other rows' requests have ended or there is none) -> (streams, state).
        `kda_picks` (`TtKda.prefill_picks`) and `dsa_tables` (`TtDsa.prefill_tables`) are the chunk's per-row inputs,
        made once and shared by the layers.  DSA writes the padded tokens' keys too, past the prompt: decode overwrites
        each position before it attends to it.  `mark`, `tap`, `pending` and `defer` as in `step` (the mHC writes
        chained into the next reads where the prefill's groups form takes them, mhc.WRITE_READ_PREFILL)."""
        moe = {}
        if kda_picks is not None and kda_picks.real is not None:  # the MoE leaves the padding tokens out
            moe["real"] = kda_picks.real
            moe["picks"] = kda_picks  # (the mask gathered once for all the layers)
        kw = dict(tap=tap, pending=pending, defer=defer)
        if halves is not None:  # a pair of sequence-parallel waves (`TtGlm53Model.prefill_sp_pair`)
            if self.kind == KDA:
                # the waves' gated norms, then o_proj and the row all-reduce once over both waves' 2 T rows, in place of
                # once a wave (the same values; 1 x 8K -2.5 to -4 ms)

                def pair(x):
                    run = lambda x, st, h: self.attention.prefill(x, st, start, keep, h[0], project=False)  # noqa: E731
                    o, st = _by_halves(x, state, halves, run)
                    return self.attention.project(o), st

                return self._forward(streams, tokens, pair, mark, moe, **kw)
            run = lambda x, st, h: self.attention.prefill(x, st, start, h[1])  # noqa: E731
            return self._forward(streams, tokens, lambda x: _by_halves(x, state, halves, run), mark, moe, **kw)
        if self.kind == KDA:
            return self._forward(
                streams, tokens, lambda x: self.attention.prefill(x, state, start, keep, kda_picks), mark, moe, **kw
            )
        return self._forward(
            streams, tokens, lambda x: self.attention.prefill(x, state, start, dsa_tables), mark, moe, **kw
        )

    def _forward(self, streams, tokens: int, attend, mark, moe=None, decode=False, pending=None, defer=False, tap=None):
        mark = mark or (lambda part: None)
        # a long prefill wave with mHC over the row's chips by tokens (common.ROW_SPLIT): the streams, the mHC's tokens
        # (`mt`), the sub-layers' outputs and the normed inputs it reads are this chip's quarter of the row's tokens;
        # the inputs go to the sub-layers as the row's `tokens` (`row_tokens_gather`), their outputs come back as the
        # quarter (`row_quarter`, `to_row_quarter`)
        split = not decode and row_split_on(tokens)
        mt = tokens // COLS if split else tokens
        quarter = (lambda: row_quarter(tokens)) if split else contextlib.nullcontext
        mesh = self.attn_hc.mesh
        if pending is not None:  # the previous layer's FFN write and this layer's first read in one op
            hc, block_out, prev, p_post, p_comb = pending
            streams, (post, comb, x) = hc.write_read(
                block_out, prev, p_post, p_comb, self.attn_hc, mt, self.input_norm, prefill=not decode
            )
            for t in (block_out, prev, p_post, p_comb):
                ttnn.deallocate(t)
        else:
            post, comb, x = self.attn_hc.read(streams, mt, self.input_norm, chunked=decode)  # normed
        if split:
            mine, x = x, row_tokens_gather(x)
            ttnn.deallocate(mine)
        if tap is not None:  # the input streams' per-token mean (`step`)
            mean = collapse_streams(mesh, streams, self.attn_hc.n, mt)
            if split:
                mine, mean = mean, row_tokens_gather(mean)
                ttnn.deallocate(mine)
            tap.append(mean)
        mark("attn_hc.read")
        mark("input_norm")
        with ar_keep_l1(decode), quarter():  # (the sub-layer's all-reduce output goes to the mHC write alone)
            y, state = attend(x)
        if split:
            y = to_row_quarter(mesh, y, tokens)
        mark("attention")
        # the write and the next read in one op (mhc.WRITE_READ; in prefill where the groups form takes it)
        fused = decode and (pending is not None or defer)
        if not decode and mhc.WRITE_READ_PREFILL:
            fused = self.ffn_hc.fused_read_slices(mt, streams.shape[-2], prefill=True) is not None
        if fused:
            new, (post2, comb2, h) = self.attn_hc.write_read(
                y, streams, post, comb, self.ffn_hc, mt, self.post_attention_norm, prefill=not decode
            )
            for t in (post, comb, x, y):
                ttnn.deallocate(t)
            if pending is not None:  # (the input this layer made)
                ttnn.deallocate(streams)
            post, comb = post2, comb2
            mark("attn_hc.write")
            mark("ffn_hc.read")
        else:
            new = self.attn_hc.write(y, streams, post, comb)
            for t in (post, comb, x, y):
                ttnn.deallocate(t)
            if pending is not None:
                ttnn.deallocate(streams)
            mark("attn_hc.write")
            post, comb, h = self.ffn_hc.read(new, mt, self.post_attention_norm, chunked=decode)  # normed
            mark("ffn_hc.read")
        if split:
            mine, h = h, row_tokens_gather(h)
            ttnn.deallocate(mine)
        mark("post_attention_norm")
        # (a decode step's MoE takes the next-layer expert weights' L1 landing spec where armed: moe.w1_decode)
        with ar_keep_l1(decode), w1_decode() if decode else contextlib.nullcontext(), quarter():
            y = self.mlp(h, **moe) if moe and isinstance(self.mlp, TtMoe) else self.mlp(h)
        ttnn.deallocate(h)
        if split:
            y = to_row_quarter(mesh, y, tokens)
        mark("mlp")
        if defer:  # the next layer writes it with its first read
            return (self.ffn_hc, y, new, post, comb), state
        out = self.ffn_hc.write(y, new, post, comb)
        for t in (post, comb, y, new):
            ttnn.deallocate(t)
        return out, state
