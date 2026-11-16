"""mHC on the device (v2): the residual as 4 streams per token, packed as rows [1, 1, 4 Tp, hidden] bf16.

Row 4 t + i holds stream i of token t; Tp is the tokens padded to a multiple of 8 (at least 8) so the rows fill whole
tiles, and the padded tokens' rows are explicit zeros (the matmuls below contract over them).  Every token's mixing is
a small matrix applied to its own 4 rows, so each step is one matmul against a block-structured matrix:
  read:  mixes = (sum_i stream_i @ fn_i) / rms(streams): one matmul against fn^T laid out [hidden, 4 * 24] (the i-th
         24 columns are stream i's slice), a mask keeping row i's i-th block, and a 0/1 matmul adding each token's
         4 rows (then its 4 blocks); the squares are summed per row and added per token the same way.
         pre, post, comb come from the fused `ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn` (DeepSeek-V4's
         mHC, whose arithmetic is GLM's: sigmoid + eps for pre, 2 * sigmoid for post, row softmax + eps then a column
         normalisation and 19 row/column rounds for comb).
         block input = P @ streams, P [T, 4 Tp] holding each token's pre on its own 4 rows (fp32 sums, one rounding).
  write: new streams = W @ streams + Q @ block_out, with W [4 Tp, 4 Tp] block-diagonal (each token's comb^T, bf16 as
         HF) and Q [4 Tp, T] each token's post on its own rows: the two bf16 terms HF forms, added in bf16.
Grouped form (GROUP_MIN tokens or more, a multiple of 32: the prefill chunks): every one of those 0/1 matrices is
block diagonal at a tile's granularity (32 rows hold 8 tokens' streams; 128 rows hold 32 tokens'), so the same
products are taken as batched matmuls over the groups instead of one matmul over all T x 4T (or 4T x 4T) entries,
which grew with the square of the chunk.  The nonzero products and their fp32 sums are the
same, so the results are bit for bit the full form's.
v1 kept one token per tile row ([T, 1, 32, D], 4 real rows of 32): the elementwise work and reads were 8x the data, and
moving block inputs / outputs between [1, 1, T, D] and [T, 1, 1, D] needed reshapes (2.1 ms per layer at 256 users).
v0 packed the streams along the last dim and mixed them with per-token columns (228 programs and 1.9 ms per layer at
any batch).

The streams of a row's users are replicated on the row's 4 chips (no tensor parallelism inside mHC).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.glm53_flash.tt.mhc import HyperConnectionWeights
from models.demos.blackhole.glm53_flash.ttnn.common import HIFI4_FP32, ROWS, WAVE_ROWS, gather_rows, put
from models.demos.blackhole.glm53_flash.ttnn.mhc_consts import split_sinkhorn_consts, stream_mix_consts

N_STREAMS = 4
GROUP = 32  # tokens per group in the grouped form (one tile of tokens, 128 stream rows)
# tokens from which the grouped form runs (0: never): bit for bit the full form; prefill chunks of 512 gain 4 %, eager
# chunks of 128 lose 3 %
GROUP_MIN = 512
# the full form's projection streams @ fn [R, D] x [D, n * 24] in this many K slices, each a batch item on its own core
# (`_project`), up to PROJ_SPLIT_ROWS stream rows (32 tokens): its 3 output tiles a tile row left ttnn's matmul on 3
# cores walking all of K (50 us); more rows give it more output tiles.  Decode 1.0 / 5.2 ms faster at 8 / 128 users.
# 0: one matmul
PROJ_SPLIT = 32
PROJ_SPLIT_ROWS = 128
# the full form's read for up to 32 tokens in three ops in place of 28: `mhc_stream_project` (the projection and each
# row's Gram tile in STREAM_SLICES K slices, one core each), `mhc_stream_mix` (one core: the slices summed, each
# token's mixes and RMS scale, the Sinkhorn of `mhc_split_sinkhorn`, the block input's weights P), and P @ streams.
# 0: the composite
STREAM_READ = True
# ... and in decode above 32 tokens up to this many (a multiple of 32), 32 at a time (a token's 4 stream rows are
# together, so 128 rows a call), the results joined: 512 users' 64 tokens a row had taken the composite (its matmuls,
# norms and products ~22 ms a step against ~4 ms for 256 users' 32 by the ops).  Not in prefill: its chunk-128
# first-token KLD 0.00065 / 0.00039 with the composite, 0.00187 / 0.00200 with this.  0: the composite
STREAM_READ_CHUNKED = 128
# ... and those up to this many tokens in one call of each op instead: the projection over all the rows, the mix a core
# a group of 32 tokens (its P the weights of its own 128 rows), and P @ streams batched over the groups (views of the
# streams, no copies), in place of 32 tokens at a time and the joins (512 users' 64 a row: 2 slices, 2 x 3 ops and 3
# concats a read).  0: 32 at a time
STREAM_READ_GROUPS = 128
STREAM_SLICES = 8
# the read and the block's RMS norm as one op for up to 16 tokens (`ttnn.experimental.deepseek_prefill.mhc_stream_read`:
# the project, mix and P @ streams steps above with the partials, P and the norm's sums passed over the NoC, each slice
# core keeping its streams in L1) in place of the three ops and the norm (34 + 11 us at 8 users).  0: separate
READ_FUSED = True
# its slice cores: 8 per row tile of streams (two row tiles: 16), or this many if more (8 or 16)
READ_SLICES = 8
# ... and in prefill, the same op in its groups form (32 tokens a group: a mix core and READ_GROUP_SLICES slice cores
# each, the groups side by side, up to 640 tokens): in place of the composite's ~16 ops (0.88 ms a read at 512 tokens,
# 159 us; the prefill's KLD as with the composite).  0: the composite
READ_GROUPS = True
READ_GROUP_SLICES = 4
# ... and in decode at 32 or 64 tokens a row (128 users' verify step, 256 / 512 users' plain step), its groups form of 8
# tokens a group with this many slice cores a group (more than 5: 8), in place of the chunked project, mix (one core a
# 32 tokens), P @ streams and the separate norm: 32 / 64 tokens 65.9 / 90.4 -> 36.4 / 53.5 us (post and comb bit for
# bit, the normed input within a bf16 step), MTP's verify step at 128 users 60.7 -> 57.55 ms, 256 / 512 users' plain
# step 59.2 / 80.0 -> 56.5 / 76.5 ms; 8,000 traced steps without a hang.  0: those
READ_DECODE_G8 = 8
# ... and in decode past 64 tokens a row (128 users' verify step of 6 or 8 tokens a user: 96 / 128 a row), the
# prefill's groups form (READ_GROUPS: 32 tokens a group, READ_GROUP_SLICES slice cores each) in place of the chunked
# project, mix, P @ streams and the separate norm (a read ~170 us at 128 tokens): 128 users' verify step of T = 6 / 8
# 103.5 / 120.4 -> 96.5 / 111.9 ms (synthetic weights; the read 81 us at 128 tokens).
# False: those
READ_DECODE_GROUPS = True
# decode: a sub-layer's write and the next sub-layer's read as one op (`mhc_stream_write_read`) where the read is the
# fused single form (at most 16 tokens a row) or the groups of 8 at 32 tokens: the write's kernels on the cores the read
# leaves free, each new streams tile handed to the read's slice core in L1 (and written to DRAM, the next write's
# input), the slices waiting for their tiles in place of reading them; the same values as the two ops.  The decoder
# layers chain the last write of a layer with the next layer's first read (`TtDecoderLayer.step`'s `pending`).  Op alone
# (1 / 8 / 16 / 32 tokens): 31.9 / 36.4 / 48.5 / 48.7 -> 29.1 / 33.3 / 42.6 / 39.7 us; the model's outputs bit for bit
# at 8 / 32 / 128 users; the verify step at 8 / 32 / 128 users 24.87 / 39.78 / 50.75 -> 24.92 / 39.35 / 49.80 ms, the
# plain step 19.51 / 29.91 / 45.42 -> 19.44 / 29.56 / 45.11 ms; stresses clean.  False: the two ops.  Read at each call
# (tests switch it)
WRITE_READ = True
# ... and in prefill, a layer's attention write and its FFN read as one op in the groups form of 32 tokens (the
# write's kernels on the cores the read's groups leave: 40 at 512 tokens a row, 100 at 128; each new streams tile
# handed to its group's slice core in L1).  The same values as the two ops.  Read at each call.  False: the two ops
WRITE_READ_PREFILL = True
# ... at 64 tokens a row too (128 users' verify step at T = 4, 512 users' plain step: the groups of 8 on 8 grid rows,
# 72 read cores, the write on the other 48).  Op alone 61.2 -> 49.7 us, the same values; the verify step at 128 users
# x T = 4 60.19 / 60.23 -> 59.30 / 59.33 ms, 512 users' plain step 63.0 -> 62.1 ms (synthetic weights); stresses clean
# (2 x 2,000 verify steps, 2,000 plain steps at 512 users).
# False: the two ops there.  Read at each call (tests switch it)
WRITE_READ_64 = True


def padded_tokens(tokens: int) -> int:
    return max(8, -(-tokens // 8) * 8)


def _pad(t, rows: int | None = None, cols: int | None = None, value: float = 0.0):
    """Explicit padding of the last two dims (to contract over them, or to reduce them, safely)."""
    r = 0 if rows is None else rows - t.shape[-2]
    c = 0 if cols is None else cols - t.shape[-1]
    if r == 0 and c == 0:
        return t
    return ttnn.pad(t, [(0, 0)] * (len(t.shape) - 2) + [(0, r), (0, c)], value)


@dataclass
class PackConsts:
    """0/1 matrices for T tokens (Tp padded, R = 4 Tp rows, Tk = T padded to a tile)."""

    tokens: int
    rows: int
    token_sum: object  # [T, R] fp32: 1 at a token's own rows
    token_mean: object  # [T, R] fp32: 1/4 there
    spread: object  # [R, Tk] bf16: row (t, i) <- token t
    spread32: object  # [R, Tk] fp32: the same
    repeat: object  # [32, R] fp32: entry i of a token's 4 -> every token's column i (rows 4.. zero)
    same_token: object  # [R, R] fp32: 1 where row and column belong to one token
    own_block: object  # [R, 96] fp32: row (t, i) keeps columns 24 i .. 24 i + 23


@dataclass
class GroupConsts:
    """The grouped form's 0/1 matrices for T = 32 G tokens (each group: 32 tokens, 128 stream rows)."""

    groups: int
    token_sum: object  # [G, 1, 32, 128] fp32: token t's own 4 rows (the same block per group)
    repeat_rows: object  # [1, 1, 32, 128] fp32: entry i of a token's 4 -> column (t', i) for every t'
    repeat_cols: object  # [1, 1, 32, 32] fp32: entry i -> column (t', i) of an 8-token block
    same_token: object  # [B, 1, 32, 32] fp32: an 8-token block's own 4 x 4 blocks (B = T / 8 blocks)
    spread: object  # [G, 1, 128, 32] fp32: row (t, j) <- token t


_PACK: dict = {}


def grouped(tokens: int) -> bool:
    return GROUP_MIN > 0 and tokens >= GROUP_MIN and tokens % GROUP == 0


_GROUP: dict = {}


def group_consts(mesh, tokens: int, n: int = N_STREAMS) -> GroupConsts:
    key = (tokens, n)
    if key not in _GROUP or _GROUP[key][0] is not mesh:
        g, rows = tokens // GROUP, n * GROUP
        tok = torch.arange(rows) // n
        lane = torch.arange(rows) % n
        token_sum = (torch.arange(GROUP)[:, None] == tok[None, :]).float()  # [32, 128]
        repeat_rows = torch.zeros(32, rows)
        repeat_rows[:n] = (torch.arange(n)[:, None] == lane[None, :]).float()
        repeat_cols = repeat_rows[:, :32].contiguous()
        same_token = (tok[:32, None] == tok[None, :32]).float()
        spread = (tok[:, None] == torch.arange(GROUP)[None, :]).float()  # [128, 32]
        f32 = dict(dims=(None, None), dtype=ttnn.float32)
        _GROUP[key] = (
            mesh,
            GroupConsts(
                groups=g,
                token_sum=put(mesh, token_sum.expand(g, 1, GROUP, rows).contiguous(), **f32),
                repeat_rows=put(mesh, repeat_rows[None, None], **f32),
                repeat_cols=put(mesh, repeat_cols[None, None], **f32),
                same_token=put(mesh, same_token.expand(tokens * n // 32, 1, 32, 32).contiguous(), **f32),
                spread=put(mesh, spread.expand(g, 1, rows, GROUP).contiguous(), **f32),
            ),
        )
    return _GROUP[key][1]


def pack_consts(mesh, tokens: int, n: int = N_STREAMS, mix: int = 24) -> PackConsts:
    key = (tokens, n, mix)
    if key not in _PACK or _PACK[key][0] is not mesh:
        tp = padded_tokens(tokens)
        rows, tk = n * tp, -(-tokens // 32) * 32
        tok = torch.arange(rows) // n  # each row's token
        lane = torch.arange(rows) % n  # each row's stream
        token_sum = (torch.arange(tokens)[:, None] == tok[None, :]).float()
        spread = torch.zeros(rows, tk)
        spread[: n * tokens] = (tok[: n * tokens, None] == torch.arange(tk)[None, :]).float()
        repeat = torch.zeros(32, rows)
        repeat[:n] = (torch.arange(n)[:, None] == lane[None, :]).float()
        same_token = (tok[:, None] == tok[None, :]).float()
        own_block = (lane[:, None] == (torch.arange(n * mix) // mix)[None, :]).float()
        f32 = dict(dims=(None, None), dtype=ttnn.float32)
        _PACK[key] = (
            mesh,
            PackConsts(
                tokens=tokens,
                rows=rows,
                token_sum=put(mesh, token_sum[None, None], **f32),
                token_mean=put(mesh, (token_sum / n)[None, None], **f32),
                spread=put(mesh, spread[None, None], dims=(None, None)),
                spread32=put(mesh, spread[None, None], **f32),
                repeat=put(mesh, repeat[None, None], **f32),
                same_token=put(mesh, same_token[None, None], **f32),
                own_block=put(mesh, own_block[None, None], **f32),
            ),
        )
    return _PACK[key][1]


class TtHyperConnection:
    def __init__(
        self,
        mesh,
        weights: HyperConnectionWeights,
        *,
        hc_mult: int,
        sinkhorn_iters: int,
        hc_eps: float,
        norm_eps: float,
    ):
        if hc_mult != N_STREAMS:
            raise ValueError(f"the packed layout assumes {N_STREAMS} streams, got {hc_mult}")
        self.mesh = mesh
        self.n = n = hc_mult
        self.iters = sinkhorn_iters
        self.eps = hc_eps
        self.norm_eps = norm_eps
        self.mix = mix = weights.fn.shape[0]  # (2 + n) n = 24
        hidden = weights.fn.shape[1] // n
        self.hidden = hidden
        # fn is used after HF's bf16 rounding, so bf16 holds it exactly; base and scale as HF holds them.
        fn_t = weights.fn.float().T  # [n hidden, 24]
        blocks = torch.cat([fn_t[i * hidden : (i + 1) * hidden] for i in range(n)], dim=1)  # [hidden, n * 24]
        self.fn_blocks = put(mesh, blocks[None, None], dims=(None, None))
        if PROJ_SPLIT:  # fn^T in K slices: [1, S, n * 24, D / S]
            sliced = blocks.T.reshape(n * mix, PROJ_SPLIT, hidden // PROJ_SPLIT).permute(1, 0, 2)
            self.fn_slices = put(mesh, sliced[None].contiguous(), dims=(None, None))
        self.block_sum = put(mesh, torch.cat([torch.eye(mix)] * n)[None, None], dims=(None, None), dtype=ttnn.float32)
        scale, base = weights.scale, weights.base
        if scale.is_meta:  # shape-only weights (speed experiments): neutral constants
            scale, base = torch.ones(scale.shape), torch.zeros(base.shape)
        consts = split_sinkhorn_consts(n, scale, base)
        self.consts = put(mesh, consts, dims=(None, None), dtype=ttnn.float32)
        self.mix_consts = put(mesh, stream_mix_consts(n, mix), dims=(None, None), dtype=ttnn.float32)

    def _project(self, streams):
        """streams [1, 1, R, D] bf16 -> (streams @ fn [D, n * 24] as [1, 1, R, n * 24] fp32, each row's sum of squares
        [1, 1, R, 1] fp32), K in PROJ_SPLIT slices on their own cores: streams transposed to [D, R] makes each K slice
        whole tile rows (a reshape, no copy); the slices' products fn^T_s @ streams^T_s are summed in fp32 (another
        order than one matmul's fp32 sum) and transposed back.  The squares come from the same transposed slices (a sum
        over their rows on many cores; one row sum over [R, D] ran on 2 cores, 30 us).
        """
        rows, d = streams.shape[-2], streams.shape[-1]
        st = ttnn.reshape(ttnn.transpose(streams, -2, -1), [1, PROJ_SPLIT, d // PROJ_SPLIT, rows])
        grid = self.mesh.compute_with_storage_grid_size()
        program = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=d // PROJ_SPLIT // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=self.fn_slices.shape[-2] // 32,
            per_core_N=rows // 32,
        )
        parts = ttnn.matmul(
            self.fn_slices, st, dtype=ttnn.float32, program_config=program, compute_kernel_config=HIFI4_FP32
        )  # [1, S, n * 24, R]
        x = ttnn.typecast(st, ttnn.float32)
        ttnn.deallocate(st)
        slice_squares = ttnn.sum(ttnn.multiply(x, x), dim=-2, keepdim=True)  # [1, S, 1, R]
        ttnn.deallocate(x)
        squares = ttnn.sum(slice_squares, dim=1, keepdim=True)  # [1, 1, 1, R]
        ttnn.deallocate(slice_squares)
        row_squares = ttnn.transpose(squares, -2, -1)  # [1, 1, R, 1]
        ttnn.deallocate(squares)
        summed = ttnn.sum(parts, dim=1, keepdim=True)  # [1, 1, n * 24, R]
        ttnn.deallocate(parts)
        proj = ttnn.transpose(summed, -2, -1)  # [1, 1, R, n * 24]
        ttnn.deallocate(summed)
        return proj, row_squares

    def read(self, streams, tokens: int, norm=None, chunked: bool = False):
        """streams [1, 1, 4 Tp, D] bf16 -> (post [T, n] fp32, comb [T, n*n] fp32, block input [1, 1, T, D] bf16), the
        block input through `norm` (a `TtRmsNorm`) when given.  `chunked`: above 32 tokens 32 at a time by the ops
        (decode, STREAM_READ_CHUNKED)."""
        rows = streams.shape[-2]
        if norm is not None and READ_GROUPS and not chunked and tokens > WAVE_ROWS and tokens % WAVE_ROWS == 0:
            return self._read_pieces(streams, tokens, norm)  # (a long prefill wave)
        groups_ok = not chunked and rows > 64 and tokens % 32 == 0 and rows == self.n * tokens <= 4 * 640  # prefill
        if chunked and READ_DECODE_GROUPS and tokens > 64:  # (decode past 64 tokens a row: the same form)
            groups_ok = tokens % 32 == 0 and rows == self.n * tokens <= 4 * 640
        if norm is not None and READ_GROUPS and groups_ok:
            return ttnn.experimental.deepseek_prefill.mhc_stream_read(
                streams,
                self.fn_blocks,
                self.mix_consts,
                self.consts,
                norm.weight_tile,
                tokens,
                self.n,
                self.iters,
                READ_GROUP_SLICES,
                self.eps,
                self.norm_eps,
                norm.eps,
            )
        g8_ok = chunked and tokens in (32, 64) and rows == self.n * tokens and READ_DECODE_G8 > 5
        if norm is not None and READ_FUSED and g8_ok:
            return ttnn.experimental.deepseek_prefill.mhc_stream_read(
                streams,
                self.fn_blocks,
                self.mix_consts,
                self.consts,
                norm.weight_tile,
                tokens,
                self.n,
                self.iters,
                READ_DECODE_G8,
                self.eps,
                self.norm_eps,
                norm.eps,
            )
        if norm is not None and READ_FUSED and not grouped(tokens) and rows <= 64:
            return ttnn.experimental.deepseek_prefill.mhc_stream_read(
                streams,
                self.fn_blocks,
                self.mix_consts,
                self.consts,
                norm.weight_tile,
                tokens,
                self.n,
                self.iters,
                min(16, max(READ_SLICES, 8 * (rows // 32))),
                self.eps,
                self.norm_eps,
                norm.eps,
            )
        post, comb, x = self._read(streams, tokens, chunked)
        if norm is not None:
            normed = norm(x)
            ttnn.deallocate(x)
            x = normed
        return post, comb, x

    def _read_pieces(self, streams, tokens: int, norm):
        """A long prefill wave's read: the groups form takes up to 640 tokens a call,
        so the op in calls of WAVE_ROWS tokens (`offset`), each writing its rows of the first call's outputs (the groups
        are independent: the values of waves of WAVE_ROWS)."""
        outs = None
        for i in range(0, tokens, WAVE_ROWS):
            outs = ttnn.experimental.deepseek_prefill.mhc_stream_read(
                streams,
                self.fn_blocks,
                self.mix_consts,
                self.consts,
                norm.weight_tile,
                WAVE_ROWS,
                self.n,
                self.iters,
                READ_GROUP_SLICES,
                self.eps,
                self.norm_eps,
                norm.eps,
                offset=i,
                outputs=outs,
            )
        return tuple(outs)

    def _read(self, streams, tokens: int, chunked: bool = False):
        if grouped(tokens):
            return self._read_grouped(streams, tokens)
        rows = streams.shape[-2]
        within = 32 < tokens <= STREAM_READ_CHUNKED and tokens % 32 == 0 and rows == self.n * tokens
        if chunked and STREAM_READ and within and tokens <= STREAM_READ_GROUPS:
            d, g = streams.shape[-1], tokens // 32
            part = ttnn.experimental.deepseek_prefill.mhc_stream_project(streams, self.fn_blocks, STREAM_SLICES)
            post, comb, weights = ttnn.experimental.deepseek_prefill.mhc_stream_mix(
                part, self.mix_consts, self.consts, tokens, self.n, self.iters, self.eps, self.norm_eps, self.hidden
            )  # weights [T, 128]: group j's rows 32 j .. 32 j + 31, over its own 128 stream rows
            ttnn.deallocate(part)
            block_input = ttnn.matmul(
                ttnn.reshape(weights, [g, 1, 32, self.n * 32]),
                ttnn.reshape(streams, [g, 1, self.n * 32, d]),
                dtype=ttnn.bfloat16,
                compute_kernel_config=HIFI4_FP32,
            )
            ttnn.deallocate(weights)
            return post, comb, ttnn.reshape(block_input, [1, 1, tokens, d])
        if chunked and STREAM_READ and within:
            d, step = streams.shape[-1], self.n * 32
            parts = []
            for r0 in range(0, rows, step):
                chunk = ttnn.slice(streams, [0, 0, r0, 0], [1, 1, r0 + step, d])
                parts.append(self._read(chunk, 32))
                ttnn.deallocate(chunk)
            joined = tuple(ttnn.concat([p[i] for p in parts], dim=-2) for i in range(3))
            for p in parts:
                for t in p:
                    ttnn.deallocate(t)
            return joined
        if STREAM_READ and tokens <= 32:
            part = ttnn.experimental.deepseek_prefill.mhc_stream_project(streams, self.fn_blocks, STREAM_SLICES)
            post, comb, weights = ttnn.experimental.deepseek_prefill.mhc_stream_mix(
                part, self.mix_consts, self.consts, tokens, self.n, self.iters, self.eps, self.norm_eps, self.hidden
            )
            ttnn.deallocate(part)
            rows = streams.shape[-2]
            block_input = ttnn.matmul(
                ttnn.reshape(weights, [1, 1, tokens, rows]),
                streams,
                dtype=ttnn.bfloat16,
                compute_kernel_config=HIFI4_FP32,
            )
            ttnn.deallocate(weights)
            return post, comb, block_input
        c = pack_consts(self.mesh, tokens, self.n, self.mix)
        mm = dict(compute_kernel_config=HIFI4_FP32)
        # bf16 x bf16 products are exact and the matmul accumulates in fp32
        if PROJ_SPLIT and streams.shape[-2] <= PROJ_SPLIT_ROWS:
            proj, row_squares = self._project(streams)
        else:
            proj = ttnn.matmul(streams, self.fn_blocks, dtype=ttnn.float32, **mm)  # [R, n * 24]
            # RMSNorm over a token's 4 x D values has no weight: its factor is applied after the projection
            x = ttnn.typecast(streams, ttnn.float32)
            row_squares = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)  # [R, 1]
            ttnn.deallocate(x)
        own = ttnn.matmul(c.token_sum, ttnn.multiply(proj, c.own_block), **mm)  # [T, n * 24]
        ttnn.deallocate(proj)
        mixes = ttnn.matmul(own, self.block_sum, **mm)  # [T, 24]
        ttnn.deallocate(own)
        square_sum = ttnn.matmul(c.token_sum, row_squares, **mm)  # [T, 1]
        scale = ttnn.rsqrt(ttnn.add(ttnn.multiply(square_sum, 1.0 / (self.n * self.hidden)), self.norm_eps))
        mixes = ttnn.multiply(mixes, scale)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.iters, self.eps
        )  # [T, n], [T, n], [T, n*n]
        ttnn.deallocate(mixes)
        # P [T, R]: token t's pre on its own 4 rows; fp32 weights times the bf16 streams, summed in fp32, rounded once
        pre = ttnn.reshape(pre, [1, 1, tokens, self.n])
        weights = ttnn.multiply(ttnn.matmul(_pad(pre, cols=32), c.repeat, **mm), c.token_sum)
        ttnn.deallocate(pre)
        block_input = ttnn.matmul(weights, streams, dtype=ttnn.bfloat16, **mm)  # [1, 1, T, D]
        ttnn.deallocate(weights)
        return post, comb, block_input

    def fused_read_slices(self, tokens: int, rows: int, prefill: bool = False):
        """The slice cores of `read`'s one-op form for a decode step's tokens (rows of streams), or None: the groups of 8
        at 32 or 64 tokens (READ_DECODE_G8, WRITE_READ_64), else the single form up to 16 tokens; what `write_read` can
        take.  `prefill`: the groups form of 32 tokens (READ_GROUPS, WRITE_READ_PREFILL, up to 512 tokens a row)."""
        if prefill:  # (a long wave: in calls of WAVE_ROWS tokens, `write_read`)
            fits = tokens <= WAVE_ROWS or tokens % WAVE_ROWS == 0
            ok = READ_GROUPS and WRITE_READ_PREFILL and tokens % 32 == 0 and 64 < rows == self.n * tokens and fits
            return READ_GROUP_SLICES if ok else None
        g8 = tokens == 32 or (tokens == 64 and WRITE_READ_64)
        if READ_FUSED and g8 and rows == self.n * tokens and READ_DECODE_G8 > 5:
            return READ_DECODE_G8
        if READ_FUSED and not grouped(tokens) and rows <= 64:
            return min(16, max(READ_SLICES, 8 * (rows // 32)))
        return None

    def write_read(self, block_out, streams, post, comb, nxt: "TtHyperConnection", tokens: int, norm, prefill=False):
        """This sub-layer's `write` and then `nxt`'s decode `read` (`prefill`: its prefill read) of the new streams
        through `norm` -> (new streams, (post, comb, normed block input)): one op (`mhc_stream_write_read`) with
        WRITE_READ where the read takes it, else the two calls (the same values)."""
        slices = nxt.fused_read_slices(tokens, streams.shape[-2], prefill) if WRITE_READ else None
        if slices is not None:
            # (a long prefill wave: in calls of WAVE_ROWS tokens, each writing its rows of the first call's outputs)
            piece = WAVE_ROWS if prefill and tokens > WAVE_ROWS else tokens
            outs = None
            for i in range(0, tokens, piece):
                outs = ttnn.experimental.deepseek_prefill.mhc_stream_write_read(
                    streams,
                    block_out,
                    post,
                    comb,
                    nxt.fn_blocks,
                    nxt.mix_consts,
                    nxt.consts,
                    norm.weight_tile,
                    piece,
                    nxt.n,
                    nxt.iters,
                    slices,
                    nxt.eps,
                    nxt.norm_eps,
                    norm.eps,
                    offset=i,
                    outputs=outs,
                )
            new, post2, comb2, x = outs
            return new, (post2, comb2, x)
        new = self.write(block_out, streams, post, comb)
        return new, nxt.read(new, tokens, norm, chunked=not prefill)

    def write(self, block_out, streams, post, comb):
        """new streams [1, 1, 4 Tp, D] = per token comb^T @ its streams + post * block_out (block_out [1, 1, T, D]): one
        op for any number of tokens (`mhc_stream_write`: per 32-row tile of streams the 8 tokens' comb^T blocks and
        posts as two bf16 tiles, then bf16(W @ S) + bf16(Q @ block_out) per output tile; 62 / 68 -> 15 / 39 us at
        1 / 16 tokens a row against the 14 ttnn ops it replaced, the same bits)."""
        return ttnn.experimental.deepseek_prefill.mhc_stream_write(streams, block_out, post, comb, self.n)

    def _read_grouped(self, streams, tokens: int):
        """`read` as batched matmuls over groups of 32 tokens (the same products and sums)."""
        c = pack_consts(self.mesh, tokens, self.n, self.mix)
        gc = group_consts(self.mesh, tokens, self.n)
        g, rows, d = gc.groups, self.n * GROUP, self.hidden
        mm = dict(compute_kernel_config=HIFI4_FP32)
        proj = ttnn.matmul(streams, self.fn_blocks, dtype=ttnn.float32, **mm)  # [R, n * 24]
        masked = ttnn.reshape(ttnn.multiply(proj, c.own_block), [g, 1, rows, self.n * self.mix])
        ttnn.deallocate(proj)
        own = ttnn.reshape(ttnn.matmul(gc.token_sum, masked, **mm), [1, 1, tokens, self.n * self.mix])
        ttnn.deallocate(masked)
        mixes = ttnn.matmul(own, self.block_sum, **mm)  # [T, 24]
        ttnn.deallocate(own)
        x = ttnn.typecast(streams, ttnn.float32)
        row_squares = ttnn.reshape(ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True), [g, 1, rows, 1])
        ttnn.deallocate(x)
        square_sum = ttnn.reshape(ttnn.matmul(gc.token_sum, row_squares, **mm), [1, 1, tokens, 1])
        scale = ttnn.rsqrt(ttnn.add(ttnn.multiply(square_sum, 1.0 / (self.n * self.hidden)), self.norm_eps))
        mixes = ttnn.multiply(mixes, scale)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.iters, self.eps
        )
        ttnn.deallocate(mixes)
        pre = ttnn.reshape(_pad(ttnn.reshape(pre, [1, 1, tokens, self.n]), cols=32), [g, 1, GROUP, 32])
        weights = ttnn.multiply(ttnn.matmul(pre, gc.repeat_rows, **mm), gc.token_sum)  # [G, 1, 32, 128]
        ttnn.deallocate(pre)
        block_input = ttnn.matmul(weights, ttnn.reshape(streams, [g, 1, rows, d]), dtype=ttnn.bfloat16, **mm)
        ttnn.deallocate(weights)
        return post, comb, ttnn.reshape(block_input, [1, 1, tokens, d])


def expand_streams(mesh, embedding, n: int):
    """[1, 1, T, D] -> [1, 1, 4 Tp, D]: every stream starts as the token embedding (padded tokens zero)."""
    c = pack_consts(mesh, embedding.shape[-2], n)
    return ttnn.matmul(
        c.spread, _pad(embedding, rows=c.spread.shape[-1]), dtype=ttnn.bfloat16, compute_kernel_config=HIFI4_FP32
    )


def collapse_streams(mesh, streams, n: int, tokens: int):
    """[1, 1, 4 Tp, D] -> [1, 1, T, D]: the plain mean of each token's streams (fp32 sums, rounded to bf16)."""
    c = pack_consts(mesh, tokens, n)
    return ttnn.matmul(c.token_mean, streams, dtype=ttnn.bfloat16, compute_kernel_config=HIFI4_FP32)


def streams_to_device(mesh, streams: torch.Tensor):
    """Host streams [ROWS, T, n, D] (each mesh row its own users) -> the device layout [1, 1, 4 Tp, D] per chip."""
    rows, t, n, d = streams.shape
    packed = torch.zeros(rows, 1, n * padded_tokens(t), d, dtype=streams.dtype)
    packed[:, 0, : n * t] = streams.reshape(rows, n * t, d)
    return put(mesh, packed, dims=(0, None))


def streams_from_device(mesh, streams, n: int, tokens: int) -> torch.Tensor:
    """The device streams -> [ROWS, T, n, D] on the host (chip 0 of each row)."""
    full = gather_rows(mesh, streams)  # [ROWS, 1, 4 Tp, D]
    return full[:, 0, : n * tokens].reshape(ROWS, tokens, n, full.shape[-1])
