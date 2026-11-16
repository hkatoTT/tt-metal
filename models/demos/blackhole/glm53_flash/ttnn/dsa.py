"""DSA one-token decode on the 8x4 mesh: dense MLA for contexts up to index_topk + kpool - 1 tokens, the indexer's
sparse attention beyond.

Up to 2,051 tokens of context the indexer keeps every causal token (tests/test_reference_vs_hf.py), so the layer is
plain causal attention.  A longer cache (`max_seq`) takes the indexer (`_step_indexer`): the latents in one row-major
cache of every user [1, 1, U T, 512], the pools' keys (4 tokens each, compressed by the gated softmax) in a paged
cache like the latent's (`dsa_indexer.index_update` in decode), the pool scores of the step's queries
(`dsa_indexer.pool_scores`), the top 512 pools (topk_large_indices), their tokens and the tail
(`dsa_indexer.select_tokens`), and the attention over them (`dsa_indexer.sparse_split` / `sparse_combine` in decode,
sparse_sdpa in prefill).  Its prefill (`_prefill_indexer`, chunks of a multiple of 128 tokens at a multiple of 128)
writes the chunk's latents and its pools' keys, and keeps the 4 tokens of the pool after the request's last token for
decode; a first chunk attends within itself (it keeps every token), a later one picks each query's tokens as decode
does.

Tensor parallel over a row's 4 chips by head (16 of 64 per chip).  q_a and kv_a (4096 -> 1536, 512; one matmul, as
they read the same input) with their norms are replicated; q_b and kv_b are split by head; o_proj is split by input
feature and all-reduced over the row.

The cache is paged, 32 positions a page, for U users of P pages (user u's position s is row s % 32 of page u P + s // 32),
the last page a scratch page.  It holds each token's normed kv latent, [U P + 1, 1, 32, 512] bf16 per chip, the
same on a row's chips (16 times less than expanded keys and values: at a 2,048-token context 512 and 1,024 users
fit).  Prefill writes and reads it through a page table per mesh row (`prefill_tables`), so each row's request goes to
its own user (a row without one writes the scratch page): paged_fill_cache, then for a chunk after the first, chunked
attention over its user's pages.  Decode writes each new latent by one op (`dsa_latent.latent_write`) and attends with
one paged decode call for every user up to each user's position (one user per core: a row with more users than
cores takes two or more calls), through constant page tables (`alloc_positions`).

The attention is MQA over the latent (absorbed MLA, the model has no RoPE): head h's key is W_kb_h c and its value
W_vb_h c (kv_b's 256 key rows and 256 value rows of the head), so its scores are (q_h W_kb_h) . c and its output
W_vb_h (sum_t p_t c_t).  Decode and the chunks after the first: q~ = q W_kb per head (batched matmuls), paged flash MLA
(decode or chunked prefill) over the latent pages, then W_vb^T per head.  A first chunk attends within itself, keys
and values expanded by kv_b (ABSORBED_FIRST: absorbed as the later ones).  Rounding: q~ and the latent output are bf16
(the attention output 1.8e-3 from fp32 against 1.2e-3 with expanded bf16 keys and values, CPU check on layer 3).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import NamedTuple

import torch

import ttnn
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tt.dsa import DsaWeights
from models.demos.blackhole.glm53_flash.ttnn import prefill_tables
from models.demos.blackhole.glm53_flash.ttnn.common import (
    COLS,
    HIFI4_FP32,
    ROW_AXIS,
    ROWS,
    WAVE_ROWS,
    WEIGHT_MM,
    ar_input_config,
    batched_matmul_config,
    decode_linear,
    mapper,
    put,
    tp_all_reduce,
    tp_weight,
    upload_shards,
    wave_linear,
    write_rows,
    zeros,
)
from models.demos.blackhole.glm53_flash.ttnn.dsa_latent import latent_write, q_heads
from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm, norm_weight

CACHE_BLOCK = 256
# the latent cache's per-head matmuls (q W_kb, W_vb^T) on one tile row of tokens with a config of one head per core:
# ttnn's default ran them on 24 / 8 cores in 59 / 100 us at 8 users
ABSORB_CONFIG = True
# ... for up to this many tokens (512 users' 64 a row: two tile rows a head, each on its own core)
ABSORB_ROWS = 64
INDEX_TOPK_WIDTH = 2176  # token rows a query for sparse_sdpa: 2,048 + the tail's 3, to the k chunk (128)
INDEX_ALIGN = 128  # a long cache's prefill chunks: whole pages of pools (32 pools of 4 tokens)
# the long cache decode's attention with each query's selected keys split over SPLIT_K cores (dsa_indexer.sparse_split:
# sparse_sdpa computes a query on one core, ~200 us a layer for any number of users) and the splits combined
# (sparse_combine), when the step's queries times 2 fit the grid (G lowered to fit), else sparse_sdpa: the op
# 225 -> 69 / 100 / 139 us at 1 / 8 / 16 queries, the whole model's verify step at 8 / 32 / 128 users -3.9 / -1.9 /
# -0.7 ms, the KLD from the CPU reference the same.  The queries as tiles
# ([1, S, 16, 512] by a transpose)
SPLIT_K = 8
# the long cache's decode reads only `params` of its positions' buffers: written alone each step (the dense form's
# cur_pos and attend; rows: rows_write reads base + position from params), each write ~0.25 ms of host (the long
# form's writes 3.4 / 3.8 ms at 8 / 32 users against the dense form's 1.6 / 1.9).  False (a test's): every buffer
TRIM_WRITES = True
# test only: with TRIM_WRITES, the buffers it skips written each step with the values of positions 5 on (in range,
# wrong): a reader of one changes the outputs
POISON_SKIPPED = os.environ.get("GLM53_DSA_POISON_SKIPPED", "0") == "1"
PAGE = 32  # cache positions per page
# a dense latent cache's first prefill chunk attends as the later ones do (absorbed: q W_kb over the latents) rather than
# over keys and values expanded by kv_b: closer to tt/ at the layer (3 prompts, pcc 0.99994-0.999997 against
# 0.99983-0.99992 expanded).  0: expanded
ABSORBED_FIRST = os.environ.get("GLM53_DSA_ABSORBED_FIRST", "0") == "1"
# one long request's prefill over the mesh rows (sequence parallel: `TtKda._prefill_sp`,
# `_prefill_sp`): the dense latent cache takes one more user per row, where the rows other than the request's own keep
# its latents and attend.  1 x 2,048 2,502 -> 7,699 tok/s, 1 x 8,192 2,174 -> 12,552; 8 requests at once
# unchanged; the accuracy as one row's (72 teacher-forced positions against the CPU reference: KLD 0.0206-0.0231
# against 0.0202); about 3,400 traced waves without a hang
PREFILL_SP = True
# a long cache's prefill: the sparse attention split by queries over the row's 4 chips (each chip all 64 heads of a
# quarter of the queries: sparse_sdpa reads each query's 2,176 selected latents once a chip instead of four times, and
# no padded heads; 64 heads x 128 queries 975 us against 16 padded to 32 x 512 2,560 us, the rows bit for bit),
# the absorbed q and the raw outputs moved by all_to_all_async_generic (each chip's quarter only): 1 x 32,768 over the
# rows 13,986 -> 14,929 tok/s, stress 600 rounds clean, the logits bit for bit.
# 0 (a test's, and a chunk not of whole tiles a chip): by heads
PREFILL_QSPLIT = 3
# with the split by queries (PREFILL_QSPLIT): each chip's indexer takes only its quarter of the chunk's queries (the
# tokens its own sparse_sdpa attends: the head weights, the params rows and the queries' tiles taken per chip, then
# pool_scores, topk and select_tokens on the quarter: per-query ops, the same values)
PREFILL_INDEX_QSPLIT = True
# a prefill chunk's pools' keys in one op (dsa_indexer.pool_keys: the two pads to a pool a tile, `_compress`'s 10 ops and
# the reshape; the same values, 8 pools a tile).  False (a test's): the ttnn ops
PREFILL_POOL_KEYS_OP = True
# quality forms of a low-precision KV cache (the values only, no speed): the long cache's values rounded as a
# low-precision cache would hold them, still stored in bf16.  KV_ROUND_LATENT 8: the latents in FP8 e4m3 as the FP8
# cache writes them (`fp8_nudge`, then ttnn.typecast there and back: to nearest, ties away), 7: ttnn.typecast alone
# (BH truncates toward zero), before their rows are written (prefill and decode; a first chunk attends within
# itself on its own latents, as an FP8 cache would).  0: off
KV_ROUND_LATENT = int(os.environ.get("GLM53_KV_ROUND_LATENT", "0"))
# ... and the storage forms (read at empty_state): KV_LATENT_FP8 the long cache's latents stored in FP8 e4m3 (row-major
# [1, 1, U T, 512] fp8: written through ttnn.typecast, read by sparse_sdpa's FP8_E4M3 kv form in prefill and by
# sparse_split in decode).  0: bf16.  The default is the long-context form: this and the three below on (the latents
# in FP8, the pools' keys in BFP4, the split scores and latents from a cache of 8,192 tokens).  A cache of up to 2,051
# positions (the dense form) is the same in both forms.  Set the four back (1 -> 0, bfp4 -> bf16, 8192 -> 0) for the
# bf16, unsplit cache
KV_LATENT_FP8 = os.environ.get("GLM53_KV_LATENT_FP8", "1") == "1"
# ... the long cache's pools' keys (the indexer's, [U PP + 1, 1, 32, 128] tiles; read at empty_state): bf16, or "bfp8" /
# "bfp4" stored in BFP8_b / BFP4_b tiles (1,088 / 576 bytes a tile against 2,048: the indexer's part of a user's context
# memory 0.70 -> 0.37 / 0.20 KB a position over the 11 layers), each face row of 16 values rounded to nearest (ties
# away from zero) on its largest exponent by the same code in prefill (dsa_indexer.bfp_pack before paged_fill_cache)
# and decode (dsa_indexer.index_update's owner core), read by pool_scores in the cache's format.  "bf16": bf16 tiles.
# "bfp4" by default (the long-context form, above)
POOLS_DTYPE = os.environ.get("GLM53_DSA_POOLS_DTYPE", "bfp4")
# the split scores: a long cache's decode indexer split over the row's 4 chips (the pools' scores of
# the blocks 4 j + c on mesh column c, its top 512 of them, the candidates gathered over the row and merged:
# `dsa_indexer.shard_candidates`, `shard_pick`, select_tokens' map; the same pools as the top 512 of the whole row,
# ties aside: topk_large_indices' order of equal scores is its own), for a
# cache of at least SCORES_SHARD tokens (whole pool pages a chip, at most 65,536 pools a chip: up to a 1M cache).
# 8,192 by default (the long-context form, above); 0: off
SCORES_SHARD = int(os.environ.get("GLM53_DSA_SCORES_SHARD", "8192"))
# the split latents: a long cache's latents split over the row's 4 chips by token (token t of a user
# on mesh column t mod 4, at row t / 4 of its user's L / 4: `rows_write(shard=True)`), for a cache of at least
# SHARD_KV tokens; decode attends each chip's share with all 64 heads (the absorbed q gathered over the row
# row-major, two 32-head units a query, `sparse_split(upq=2)`, or one 64-head unit, `upq=1`), the partials exchanged
# (`shard_repack`, all_to_all) and combined for the chip's 16 heads (`sparse_combine(xchip=...)`).  8,192 by default
# (the long-context form, above); 0: off
SHARD_KV = int(os.environ.get("GLM53_DSA_SHARD_KV", "8192"))
SHARD_WIDTH = 640  # a chip's token rows a query: a token of each of 512 pools and of the tail (<= 1), to 128
# the gathered prefix (prefill over the split latents): a chunk whose pools reach at most SHARD_PREFIX tokens gathers
# its user's prefix over the row (`dsa_indexer.prefix_rows`, all_gather) and attends as an unsplit cache does (QSPLIT,
# sparse_sdpa over the same latents in the same order: the same values); a longer one exchanges the queries and partials
# (`_shard_attend`).  The gather is 3/4 of the prefix a chip (bf16 1 KB a token): 6 MB at 8K, 96 MB at 128K.  0: always
# the exchange
SHARD_PREFIX = int(os.environ.get("GLM53_DSA_SHARD_PREFIX", "131072"))
_POOLS_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def pools_dtype():
    """The pools cache's ttnn dtype (POOLS_DTYPE, read when the states are made)."""
    if POOLS_DTYPE not in _POOLS_DTYPES:
        raise ValueError(f"GLM53_DSA_POOLS_DTYPE={POOLS_DTYPE!r}: one of {', '.join(_POOLS_DTYPES)}")
    return _POOLS_DTYPES[POOLS_DTYPE]


def _zeros(mesh, shape, dtype, layout=ttnn.TILE_LAYOUT):
    """`zeros` for a bf16 cache, dsa_indexer.zeros_bytes for a low-precision one (FP8 / BFP: ttnn.zeros makes FP8
    nowhere and BFP on the host)."""
    if dtype == ttnn.bfloat16:
        return zeros(mesh, shape, layout=layout)
    from .dsa_indexer import zeros_bytes

    return zeros_bytes(mesh, shape, dtype, layout)


def fp8_nudge(t):
    """bf16 t (deallocated) with 8 added to each value's bits (its magnitude's, sign-magnitude): BH's typecast to FP8
    e4m3 truncates the 4 low mantissa bits, so after the nudge it rounds to nearest, ties away from zero
    (tools/kv_round.fp8_e4m3 "rna"; the subnormals below 2^-6 still truncated)."""
    u = ttnn.bitcast(t, ttnn.uint16)
    ttnn.deallocate(t)
    v = ttnn.add(u, 8)
    ttnn.deallocate(u)
    out = ttnn.bitcast(v, ttnn.bfloat16)
    ttnn.deallocate(v)
    return out


def to_fp8(t, nudge: bool = True):
    """bf16 t (deallocated) as FP8 e4m3: nudged to nearest (`fp8_nudge`), or truncated (nudge False)."""
    if nudge:
        t = fp8_nudge(t)
    low = ttnn.typecast(t, ttnn.fp8_e4m3)
    ttnn.deallocate(t)
    return low


def round_kv(t, code: int):
    """t (deallocated) through a low-precision dtype and back to bf16: code 8 with a row-major t FP8 e4m3 to nearest
    (7: truncated), with a tiled one BFP8_b (the drafter's K / V, dflash2.KV_ROUND); 4 BFP4_b; 0: t itself."""
    if not code:
        return t
    if code in (7, 8) and t.layout == ttnn.ROW_MAJOR_LAYOUT:
        low = to_fp8(t, nudge=code == 8)
    elif code in (4, 8):
        low = ttnn.typecast(t, ttnn.bfloat8_b if code == 8 else ttnn.bfloat4_b)
        ttnn.deallocate(t)
    else:
        raise ValueError(f"a KV rounding code {code}: 0, 8, 7 (row-major) or 4")
    out = ttnn.typecast(low, ttnn.bfloat16)
    ttnn.deallocate(low)
    return out


# a captured later chunk's K chunk (the flexible chunked SDPA, start-independent): what the eager form takes at a start
# that is a multiple of 128, so the dense cache's later chunks of a multiple of 128 tokens have a traced form
TRACED_K_CHUNK = 128


def _free(t, *shared) -> None:
    """Deallocate t unless it is one of `shared` (a slice of the whole tensor, or a view, is the tensor itself)."""
    if t.is_allocated() and all(t.buffer_address() != s.buffer_address() for s in shared):
        ttnn.deallocate(t)


class DsaPrefillTables(NamedTuple):
    """A long cache's prefill chunk inputs per mesh row (`TtDsa.prefill_tables`)."""

    rows: object  # [1, 1, 1, T] uint32: each token's latent row in the row-major cache (0xFFFFFFFF: no request)
    params: object  # [1, 1, T, 16] uint32: (position, its user's first row, its first pool page) a query
    pool_written: object  # [1, T / 128] int32: the pages of the chunk's pools (the scratch page: no request)
    pool_read: object  # [1, n] int32: its user's first n pool pages (32 n: a power of 2 >= 512 and the chunk's pools)
    # [1, 1, 32, T] bf16: the tokens of the pool after the request's last token, each picked into its position's
    # row of the recent pages (`dsa_indexer.recent_row`)
    recent_pick: object
    recent_table: object  # [1, 1] int32: its user's page of rkey / rgate (the scratch page: no request)
    traced: bool = False  # a captured chunk: the pool scores over the whole bucket (the same values, any start)
    gathered: bool = False  # sequence parallel (`sp_tables`): rows, pool_written name every row's segment (gathered)


class DsaSpTables(NamedTuple):
    """A sequence-parallel chunk's inputs per mesh row (`TtDsa.sp_tables`): the rows hold consecutive segments of one
    request."""

    written: object  # [1, 8 T / 32] int32: the pages of every row's segment (row r's latents at r T of the gathered
    # block) in this row's target user (the scratch page: a row without a segment)
    attended: object  # [1, P] int32: the target user's pages
    start: object  # [1] int32: this row's segment's first position (0 without one)


@dataclass
class TtDsaState:
    # per chip, user u's pages u P .. (u + 1) P - 1, the last page scratch: the latent [U P + 1, 1, PAGE, 512] bf16
    keys: object
    # the indexer (a cache past the dense limit): every user's latents row-major [1, 1, U T, 512] (sparse_sdpa's), the
    # pools' keys paged [U PP + 1, 1, 32, 128] (pool p of user u at row p of its pages), and the recent tokens'
    # indexer keys and gates [U + 1, 1, 32, 128] (row `dsa_indexer.recent_row(pos)` of page u: a ring of 4 pools)
    rm: object = None
    pools: object = None
    rkey: object = None
    rgate: object = None

    def deallocate(self):
        for t in (self.keys, self.rm, self.pools, self.rkey, self.rgate):
            if t is not None:
                ttnn.deallocate(t)


@dataclass
class DsaPositions:
    """A decode step's positions at fixed addresses (`TtDsa.alloc_positions`; written each step with
    `write_positions`), with constant page tables."""

    cur_pos: object  # [B] int32 per chip: each user's position (sdpa decode, latent_write)
    pages: object  # [B, P] int32: each user's pages (sdpa decode)
    attend: list  # per sdpa decode call: (its users' positions [b] int32, their pages [b, P]); one call: the above
    # the indexer: each user's latent row [1, 1, 1, R] uint32 (TRIM_WRITES: unused), (pos, row base, pool page base)
    # [1, 1, B, 16] uint32
    rows: object = None
    params: object = None
    # MTP's verify step: k U virtual users, the first U each user's token, the next U its draft (user u
    # again, one position on), with two drafts the next U its second
    split: int | None = None


class TtDsa:
    def __init__(
        self,
        mesh,
        w: DsaWeights,
        config: Glm53Config,
        max_seq: int = 2048,
        sdpa_compute_config=None,
        dtype=ttnn.bfloat16,
    ) -> None:
        dense_limit = config.index_topk + config.index_kpool - 1
        # sdpa decode reads the cache in chunks of a multiple of 32 rows; whole 256-row blocks keep that true
        max_seq = -(-max_seq // CACHE_BLOCK) * CACHE_BLOCK
        self.long = max_seq > min(dense_limit, 2048)
        self.mesh = mesh
        self.heads = config.num_heads // COLS
        self.dk, self.dv = config.qk_head_dim, config.v_head_dim
        self.eps = config.rms_norm_eps
        self.max_seq = max_seq
        self.pages = max_seq // PAGE  # per user
        self.scale = config.qk_head_dim**-0.5
        # q_a and kv_a read the same input: one matmul against [4096, 1536 + 512], split after
        self.q_rank = w.q_a.shape[0]
        self.a_proj = put(mesh, torch.cat([w.q_a, w.kv_a]).T.contiguous(), dims=(None, None), dtype=dtype)
        # width-sharded in decode (ttnn's interleaved norm ran q's 1,536 features on one core: 25 us at 8 users);
        # kv's 512 stay interleaved (9 us, about what the sharded form's moves in and out cost)
        self.q_a_norm = TtRmsNorm(mesh, w.q_a_norm, self.eps)
        self.q_b = tp_weight(mesh, w.q_b, "out", dtype)
        self.kv_a_norm = norm_weight(mesh, w.kv_a_norm)
        self.kv_b = tp_weight(mesh, w.kv_b, "out", dtype)  # per head: 256 key features then 256 value features
        self.o_proj = tp_weight(mesh, w.o_proj, "in", dtype)
        self.rank = w.kv_b.shape[1]  # 512
        # the latent cache (absorbed MLA), per chip its 16 heads: W_kb [1, 16, 256, 512] (q~ = q W_kb) and W_vb^T
        # [1, 16, 512, 256]
        per_head = w.kv_b.view(-1, self.dk + self.dv, self.rank)
        h = self.heads
        self.w_kb = upload_shards(mesh, lambda r, c: per_head[c * h : (c + 1) * h, : self.dk][None], dtype=dtype)
        self.w_vb_t = upload_shards(
            mesh,
            lambda r, c: per_head[c * h : (c + 1) * h, self.dk :].transpose(1, 2).contiguous()[None],
            dtype=dtype,
        )
        self.kb_config = batched_matmul_config(mesh, k_tiles=self.dk // 32, n_tiles=self.rank // 32)
        self.vb_config = batched_matmul_config(mesh, k_tiles=self.rank // 32, n_tiles=self.dv // 32)
        size = mesh.compute_with_storage_grid_size()
        self.grid = size
        # K/V read in 128-row chunks: with ~1 core per (user, head) the default chunk is the whole cache and the
        # circular buffers outgrow L1.  The compute config is left to the op by default (another BH model notes that
        # custom configs gave wrong decode output); tests compare both.
        self.sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=size, q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False
        )
        self.sdpa_compute = sdpa_compute_config
        if self.long:
            self._init_indexer(w, config)

    def _init_indexer(self, w: DsaWeights, config: Glm53Config) -> None:
        """The indexer's weights (the same on every chip: its 32 heads on each) and constants."""
        mesh = self.mesh
        self.kpool, self.index_heads, self.index_dim = config.index_kpool, config.index_num_heads, config.index_head_dim
        self.index_k = config.index_topk // config.index_kpool  # pools kept: 512
        self.pool_pages = self.max_seq // self.kpool // PAGE  # per user
        rep = dict(dims=(None, None))
        # x -> the indexer key (before its LayerNorm), the pool gate and the head weights, one matmul [4096, 288]; the
        # head weights' scale 32^-0.5 and the scores' 128^-0.5 folded into theirs (1 / 64: a power of 2, the same
        # rounding)
        self.head_scale = float((self.index_heads**-0.5) * (self.index_dim**-0.5))
        self.idx_proj = put(
            mesh,
            torch.cat([w.index_wk, w.index_pool_gate, w.index_head_weights * self.head_scale]).T.contiguous(),
            **rep,
        )
        self.idx_norm_w = put(mesh, w.index_k_norm_weight.reshape(1, 1, 1, -1), **rep)
        self.idx_norm_b = put(mesh, w.index_k_norm_bias.reshape(1, 1, 1, -1), **rep)
        self.idx_wq = put(mesh, w.index_wq.T.contiguous(), **rep)  # q_resid -> 32 heads x 128
        # the pool's slot bias, rows 0-3 of a tile, -inf below (the softmax over the 4 rows)
        bias = torch.full((1, 1, PAGE, self.index_dim), float("-inf"))
        bias[0, 0, : self.kpool] = 0.0 if w.index_pool_bias.is_meta else w.index_pool_bias.float()  # (synthetic: 0)
        self.pool_bias = put(mesh, bias, dtype=ttnn.float32, **rep)
        from .dsa_indexer import pool_keys_bias

        # pool_keys' slot bias: row r of a tile the bias of slot r mod 4 (the pools 8 a tile)
        self.pool_rows_bias = pool_keys_bias(mesh, None if w.index_pool_bias.is_meta else w.index_pool_bias)
        self.sparse_format = getattr(ttnn.transformer, "SparseKVFormat", None) or getattr(ttnn, "SparseKVFormat")

    def attend_parts(self, batch: int) -> list[tuple[int, int]]:
        """The users of each sdpa decode call: it takes one user per core, so 128 users per row go in two calls of 64."""
        calls = -(-batch // (self.grid.x * self.grid.y))
        size = -(-batch // calls)
        return [(u, min(u + size, batch)) for u in range(0, batch, size)]

    def empty_state(self, batch: int) -> TtDsaState:
        if self.long:  # (PREFILL_SP: user `batch`, past the decode's, for the sequence-parallel rows)
            users = batch + 1 if PREFILL_SP else batch
            rm_rows = users * self.max_seq // (COLS if self.shard_kv() else 1)  # (split latents: a chip's quarter)
            return TtDsaState(
                keys=None,
                rm=_zeros(
                    self.mesh,
                    [1, 1, rm_rows, self.rank],
                    ttnn.fp8_e4m3 if KV_LATENT_FP8 else ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                pools=_zeros(self.mesh, [users * self.pool_pages + 1, 1, PAGE, self.index_dim], pools_dtype()),
                rkey=zeros(self.mesh, [users + 1, 1, PAGE, self.index_dim]),
                rgate=zeros(self.mesh, [users + 1, 1, PAGE, self.index_dim]),
            )
        users = batch + 1 if PREFILL_SP else batch  # (user `batch`, past the decode's, for the sequence-parallel rows)
        return TtDsaState(keys=zeros(self.mesh, [users * self.pages + 1, 1, PAGE, self.rank]))

    def users(self, state: TtDsaState) -> int:
        if state.rkey is not None:
            return state.rkey.shape[0] - 1
        return (state.keys.shape[0] - 1) // self.pages

    def _table(self, rows):
        """A page table (int32 row-major), the same on every chip."""
        rows = torch.tensor(rows, dtype=torch.int32)
        return put(self.mesh, rows, dims=(None, None), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def traced_bucket(self, start: int, t: int) -> int | None:
        """A later chunk's (start > 0) trace key part (`TtGlm53Model.prefill_key`): a long cache's pool bucket (the
        pool pages it scores, `_indexer_tables`), 0 for a dense cache whose chunk has the start-independent form (the
        latent cache, start and T multiples of 128: the flexible chunked SDPA then takes the K and Q chunks the eager
        form takes), else None (eager)."""
        if self.long:
            return self.pool_bucket(start, t)
        return 0 if start % TRACED_K_CHUNK == 0 and t % TRACED_K_CHUNK == 0 else None

    def pool_bucket(self, start: int, t: int) -> int:
        """A long cache's chunk scores its user's first n pool pages: 32 n a power of 2 from 512 pools up to its end
        (a few shapes to compile, whatever the chunk's start)."""
        return min(self.pool_pages, (1 << (max(self.index_k, (start + t) // self.kpool) - 1).bit_length()) // PAGE)

    def prefill_tables(
        self, place, users: int, slots: list, start: int, t: int, valid: list | None = None, traced: bool = False
    ) -> tuple:
        """A prefill chunk's page tables per mesh row, made once per chunk and shared by the DSA layers, put on the
        device by `place` (`common.eager_place` or `FixedInputs.place`): for the positions start .. start + T - 1
        (written) and 0 .. start + T - 1 (attended; None when start is 0), [1, n] int32 each, of user `slots[row]`
        (`users` per row), or the scratch page for a row whose slot is None.  `traced` (a captured later chunk,
        `traced_bucket` 0): attended is that user's whole table, followed by the start [1] int32 (the flexible chunked
        SDPA's).  A long cache: `DsaPrefillTables` (each row's real tokens `valid`, default all)."""
        if self.long:
            return self._indexer_tables(place, users, slots, start, t, valid, traced)
        scratch = users * self.pages

        def table(name: str, first: int, last: int):
            rows = [
                [scratch] * (last - first) if s is None else [s * self.pages + p for p in range(first, last)]
                for s in slots
            ]
            return place(name, torch.tensor(rows, dtype=torch.int32), (0, None), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        first, last = start // PAGE, (start + t) // PAGE
        written = table("dsa_written", first, last)
        if not start:  # (a first chunk over WAVE_ROWS: its rest absorbed, `_first_wave`)
            return (written, table("dsa_attended", 0, last)) if ABSORBED_FIRST or t > WAVE_ROWS else (written, None)
        if traced and self.traced_bucket(start, t) == 0:
            at = torch.tensor([start], dtype=torch.int32)
            return (
                written,
                table("dsa_attended_all", 0, self.pages),
                place("dsa_start", at, (None, None), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            )
        return written, table("dsa_attended", 0, last)

    def sp_tables(self, place, users: int, groups, t: int):
        """A sequence-parallel chunk's tables: each request's segments of T tokens on its rows (`common.SpGroup`, from
        its position `start`).  Its latents go to its user on its own row and to the extra user (`users` - 1,
        PREFILL_SP) on its other rows; each row's queries attend over its target user's positions up to its segment's
        end.  A dense cache: `DsaSpTables`; a long one: `DsaPrefillTables` (gathered, `_sp_indexer_tables`)."""
        if not PREFILL_SP:
            raise ValueError("a sequence-parallel prefill with PREFILL_SP off")
        for grp in groups:
            if grp.start % TRACED_K_CHUNK or t % TRACED_K_CHUNK or grp.start + len(grp.rows) * t > self.max_seq:
                raise ValueError(
                    f"segments of {t} from {grp.start}: multiples of {TRACED_K_CHUNK} within {self.max_seq}"
                )
        member = {r: (grp, i) for grp in groups for i, r in enumerate(grp.rows)}
        if self.long:
            return self._sp_indexer_tables(place, users, groups, member, t)
        scratch, span = users * self.pages, t // PAGE
        written, attended, at = [], [], []
        for r in range(ROWS):
            grp, i = member.get(r, (None, 0))
            user = grp.user if grp is not None and r == grp.owner else users - 1
            pages = []
            for src in range(ROWS):
                if grp is not None and src in grp.rows:
                    first_page = user * self.pages + (grp.start + grp.rows.index(src) * t) // PAGE
                    pages += list(range(first_page, first_page + span))
                else:
                    pages += [scratch] * span
            written.append(pages)
            attended.append([user * self.pages + p for p in range(self.pages)])
            at.append(grp.start + i * t if grp is not None else 0)
        i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return DsaSpTables(
            written=place("dsa_sp_written", torch.tensor(written, dtype=torch.int32), (0, None), **i32),
            attended=place("dsa_sp_attended", torch.tensor(attended, dtype=torch.int32), (0, None), **i32),
            start=place("dsa_sp_start", torch.tensor(at, dtype=torch.int32), (0, None), **i32),
        )

    def sp_bucket(self, start: int, t: int) -> int:
        """A long cache's sequence-parallel wave from `start` scores the pool pages up to a whole wave's end (8 T), so
        that every wave from one start has one captured form."""
        return self.pool_bucket(start, ROWS * t)

    def _sp_indexer_tables(self, place, users: int, groups, member: dict, t: int):
        """`sp_tables` for a long cache: `DsaPrefillTables` (gathered) whose rows and pool pages name every row's
        segment of the row's own request (row r's at r T of the gathered latents, r T / 4 of the gathered pool keys)
        in the row's target user, whose queries go as a later chunk's do (each row's own positions, the scores over
        the wave's bucket: the largest a group's start takes), and each request's recent pool (after its last token)
        written by its own row."""
        from .dsa_indexer import PARAMS, RECENT_RING, SENT

        n_read = max(self.sp_bucket(grp.start, t) for grp in groups)
        tab = prefill_tables.dsa_sp_arrays(
            ROWS, member, groups, users, t, **self._table_sizes(n_read), sent=SENT, n_params=PARAMS, ring=RECENT_RING
        )
        u32 = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return DsaPrefillTables(
            rows=place("dsa_sp_rows", tab["rows"].to(torch.int32), (0, None), **u32),
            params=place("dsa_params", tab["params"].to(torch.int32), (0, None), **u32),
            pool_written=place("dsa_sp_pool_written", tab["pool_written"], (0, None), **i32),
            pool_read=place(f"dsa_pool_read_{n_read}", tab["pool_read"], (0, None), **i32),
            recent_pick=place("dsa_recent_pick", tab["recent_pick"], (0, None)),
            recent_table=place("dsa_recent_table", tab["recent_table"], (0, None), **i32),
            traced=True,
            gathered=True,
        )

    def _table_sizes(self, n_read: int) -> dict:
        """`prefill_tables`' sizes of this cache's indexer tables (a chunk scoring `n_read` pool pages)."""
        return dict(kp=self.kpool, page=PAGE, max_seq=self.max_seq, pool_pages=self.pool_pages, n_read=n_read)

    def _indexer_tables(
        self, place, users: int, slots: list, start: int, t: int, valid, traced: bool = False
    ) -> DsaPrefillTables:
        from .dsa_indexer import PARAMS, RECENT_RING, SENT

        if t % INDEX_ALIGN or start % INDEX_ALIGN or start + t > self.max_seq:
            raise ValueError(
                f"chunk [{start}, {start + t}): a long cache's prefill takes multiples of {INDEX_ALIGN} tokens at a "
                f"multiple of {INDEX_ALIGN}, within the {self.max_seq}-token cache"
            )
        valid = [t] * ROWS if valid is None else list(valid)
        n_read = self.pool_bucket(start, t)  # the pools a chunk scores
        sizes = dict(self._table_sizes(n_read), sent=SENT, n_params=PARAMS, ring=RECENT_RING)
        tab = prefill_tables.dsa_row_arrays(ROWS, list(slots), start, t, valid, users, **sizes)
        rows, params, pool_written, pool_read = tab["rows"], tab["params"], tab["pool_written"], tab["pool_read"]
        pick, recent = tab["recent_pick"], tab["recent_table"]
        u32 = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return DsaPrefillTables(
            rows=place("dsa_rows", rows.to(torch.int32), (0, None), **u32),
            params=place("dsa_params", params.to(torch.int32), (0, None), **u32),
            pool_written=place("dsa_pool_written", pool_written, (0, None), **i32),
            pool_read=place(f"dsa_pool_read_{n_read}", pool_read, (0, None), **i32),  # (a trace's per bucket)
            recent_pick=place("dsa_recent_pick", pick, (0, None)),
            recent_table=place("dsa_recent_table", recent, (0, None), **i32),
            traced=traced,
        )

    def positions_on_device(self, positions: torch.Tensor) -> DsaPositions:
        """positions [ROWS, B] (each row's users) -> `DsaPositions`, newly allocated (eager use).  For a captured trace
        use alloc_positions / write_positions instead."""
        bufs = self.alloc_positions(positions.shape[1])
        self.write_positions(bufs, positions)
        return bufs

    def _host(self, t: torch.Tensor):
        """A 1-D host tensor per chip, [ROWS, n] sharded over the rows."""
        t = t.to(torch.int32).reshape(-1)
        return ttnn.from_torch(
            t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper(self.mesh, (0, None))
        )

    def alloc_positions(self, batch: int, users: int | None = None) -> DsaPositions:
        """Fixed-address device tensors for the positions (written each step with write_positions), with the constant
        page tables per user [B, P] (sdpa).  `users` < B (MTP's verify step): B = k users virtual users, virtual user v
        on user v mod users's pages."""

        def zeros(n):
            return put(
                self.mesh,
                torch.zeros(ROWS * n, dtype=torch.int32),
                dims=(0, None),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        split = users if users is not None and users < batch else None
        owner = torch.arange(batch) % (users or batch)
        pages = owner[:, None] * self.pages + torch.arange(self.pages)  # [B, P]
        cur_pos, table = zeros(batch), self._table(pages.tolist())
        parts = self.attend_parts(batch)
        if len(parts) == 1:
            attend = [(cur_pos, table)]
        else:
            attend = [(zeros(u1 - u0), self._table(pages[u0:u1].tolist())) for u0, u1 in parts]
        bufs = DsaPositions(cur_pos=cur_pos, pages=table, attend=attend, split=split)
        if self.long:
            self._alloc_indexer_positions(bufs, batch)
        return bufs

    def _indexer_values(self, positions: torch.Tensor, users: int | None = None) -> dict:
        """The indexer's per-step inputs from positions [ROWS, B] (`DsaPositions` fields, torch; sharded over rows);
        `users` < B (MTP's verify step, `DsaPositions.split`): virtual user v takes user v mod users's rows and pools.
        """
        from .dsa_indexer import PARAMS

        rows_n, b = positions.shape
        width = max(32, -(-b // 32) * 32)
        owner = torch.arange(b, dtype=torch.int64) % (users or b)
        base = owner * self.max_seq
        rows = torch.full((rows_n, 1, 1, width), -1, dtype=torch.int64)
        rows[:, 0, 0, :b] = base[None] + positions
        params = torch.zeros(rows_n, 1, b, PARAMS, dtype=torch.int64)
        params[:, 0, :, 0] = positions
        params[:, 0, :, 1] = base[None]
        params[:, 0, :, 2] = owner[None] * self.pool_pages
        return dict(rows=rows, params=params)

    def _indexer_host(self, positions: torch.Tensor, users: int | None = None) -> dict:
        """`_indexer_values` as host tensors (sharded over rows)."""
        values = self._indexer_values(positions, users)
        rm = dict(layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper(self.mesh, (0, None)))
        return dict(
            rows=ttnn.from_torch(values["rows"].to(torch.int32), dtype=ttnn.uint32, **rm),
            params=ttnn.from_torch(values["params"].to(torch.int32), dtype=ttnn.uint32, **rm),
        )

    def _alloc_indexer_positions(self, bufs: DsaPositions, batch: int) -> None:
        host = self._indexer_host(torch.zeros(ROWS, batch, dtype=torch.int64), bufs.split)
        bufs.rows = ttnn.to_device(host["rows"], self.mesh)
        bufs.params = ttnn.to_device(host["params"], self.mesh)

    def write_positions(self, bufs: DsaPositions, positions: torch.Tensor, pack=None) -> None:
        """positions [ROWS, B] -> the buffers the decode reads; `pack` (an `InputPack` holding the row-major ones):
        their values into it (its one write and the device's unpack follow, the caller's), else a write each."""
        if int(positions.max()) >= self.max_seq:
            raise ValueError(f"position {int(positions.max())} is past the {self.max_seq}-token cache")
        self._pack = pack
        try:
            trim = TRIM_WRITES and self.long
            if not trim:
                self._write_dense_fields(bufs, positions)
            if not self.long:
                return
            self._put(self._params_host(bufs, positions), bufs.params, ttnn.uint32)
            values = functools.lru_cache(maxsize=1)(
                lambda: self._indexer_values(positions.to(torch.int64), bufs.split)
            )  # (the other fields' values, made only when one of them is written)
            self._write_indexer_fields(bufs, positions, values, "needed" if trim else "all")
            if trim and POISON_SKIPPED:  # (a test's check that nothing reads the buffers skipped)
                wrong = (positions + 5) % self.max_seq
                self._write_dense_fields(bufs, wrong)
                wrong_values = functools.lru_cache(maxsize=1)(
                    lambda: self._indexer_values(wrong.to(torch.int64), bufs.split)
                )
                self._write_indexer_fields(bufs, wrong, wrong_values, "skipped")
        finally:
            self._pack = None

    def _params_host(self, bufs: DsaPositions, positions: torch.Tensor) -> torch.Tensor:
        """The params' host values [ROWS, 1, B, PARAMS] for these positions: each user's first rows and pool pages made
        once a positions object (`_indexer_values`), the positions set in place each step."""
        cached = getattr(bufs, "params_host", None)
        if cached is None or tuple(cached.shape[:3]) != (positions.shape[0], 1, positions.shape[1]):
            cached = self._indexer_values(positions.to(torch.int64), bufs.split)["params"].clone()
            bufs.params_host = cached
        cached[:, 0, :, 0] = positions
        return cached

    def _put(self, values: torch.Tensor, buf, dtype=ttnn.int32) -> None:
        """A row-major position buffer's values (dim 0 over the mesh rows): into the pack being filled, else written."""
        pack = getattr(self, "_pack", None)
        if pack is not None and pack.has(buf):
            pack.set(buf, values)
        else:
            write_rows(self.mesh, values, buf, dtype)

    @staticmethod
    def position_buffers(bufs: DsaPositions) -> list:
        """A positions object's row-major buffers written each step (for `InputPack`; visible, a fp32 tile, apart)."""
        out = [bufs.cur_pos] + [cur for cur, _ in bufs.attend] + [bufs.rows, bufs.params]
        return [t for t in out if t is not None]

    def _write_dense_fields(self, bufs: DsaPositions, positions: torch.Tensor) -> None:
        """The dense form's decode inputs: cur_pos (sdpa decode, latent_write), each attend part's."""
        put_rows = lambda t, buf: self._put(t.reshape(-1), buf)  # noqa: E731  (as `_host`)
        put_rows(positions, bufs.cur_pos)
        if len(bufs.attend) > 1:
            for (u0, u1), (cur, _) in zip(self.attend_parts(positions.shape[1]), bufs.attend):
                put_rows(positions[:, u0:u1], cur)

    def _write_indexer_fields(self, bufs: DsaPositions, positions: torch.Tensor, values, which: str) -> None:
        """The long form's inputs besides params (`values()`: `_indexer_values`): rows, which rows_write reads without
        TRIM_WRITES; `which` "all", "needed" (as the decode reads them) or "skipped" (the others)."""
        needed = not TRIM_WRITES
        if which == "all" or needed == (which == "needed"):
            self._put(values()["rows"], bufs.rows, ttnn.uint32)

    def _project(self, x, with_resid: bool = False):
        """x [1, 1, S, 4096] -> (q [1, 1, S, 16 * 256], the normed kv latent [1, 1, S, 512]; with_resid: and the
        normed q_resid [1, 1, S, 1536])."""
        rows = x.shape[-2]
        mm = dict(compute_kernel_config=HIFI4_FP32)
        a = decode_linear(self.mesh, x, self.a_proj, **mm)  # [1, 1, S, 1536 + 512]
        q_resid = ttnn.slice(a, [0, 0, 0, 0], [1, 1, rows, self.q_rank])
        latent = ttnn.slice(a, [0, 0, 0, self.q_rank], [1, 1, rows, a.shape[-1]])
        ttnn.deallocate(a)
        q_resid = self.q_a_norm(q_resid)
        q = decode_linear(self.mesh, q_resid, self.q_b, **mm)
        latent = ttnn.rms_norm(latent, weight=self.kv_a_norm, epsilon=self.eps, **mm)
        if with_resid:
            return q, latent, q_resid
        return q, latent

    def _expand(self, latent):
        """The latent [1, 1, S, 512] -> [1, 1, S, 16 * 512]: per head 256 key, 256 value features."""
        return ttnn.linear(latent, self.kv_b, compute_kernel_config=WEIGHT_MM)

    def _absorb(self, q):
        """q per head [1, 16, S, 256] -> q~ = q W_kb [1, 16, S, 512] (the head's scores are q~ . c)."""
        cfg = {"program_config": self.kb_config} if ABSORB_CONFIG and q.shape[-2] <= max(32, ABSORB_ROWS) else {}
        return ttnn.matmul(q, self.w_kb, compute_kernel_config=WEIGHT_MM, **cfg)

    def _values(self, out):
        """The weighted latents per head [1, 16, S, 512] -> the heads' outputs W_vb (sum p c) [1, 16, S, 256]."""
        cfg = {"program_config": self.vb_config} if ABSORB_CONFIG and out.shape[-2] <= max(32, ABSORB_ROWS) else {}
        return ttnn.matmul(out, self.w_vb_t, compute_kernel_config=WEIGHT_MM, **cfg)

    def prefill(self, x, state: TtDsaState, start: int, tables):
        """x [1, 1, T, 4096] bf16: T tokens (positions start .. start + T - 1) of each mesh row's request, into the
        pages `tables` gives (`prefill_tables`: a row without a request writes the scratch page) -> (out [1, 1, T,
        4096] bf16, state).  T and start are multiples of 32; start + T <= max_seq.

        The chunk's keys and values go into the cache first; its queries then attend causally over its user's
        positions 0 .. start + T - 1 (plain causal attention within the chunk when start is 0)."""
        if isinstance(tables, DsaSpTables):
            return self._prefill_sp(x, state, tables)
        if self.long:
            return self._prefill_indexer(x, state, start, tables)
        t, end = x.shape[-2], start + x.shape[-2]
        if t % PAGE or start % PAGE or end > self.max_seq:
            raise ValueError(f"chunk [{start}, {end}) must be tile-aligned and within the {self.max_seq}-row cache")
        q, latent = self._project(x)
        q = self._q_heads(q)  # [1, 16, T, 256]
        written, attended = tables[:2]
        at = tables[2] if len(tables) > 2 else None  # a captured later chunk's start (`prefill_tables`)
        ttnn.experimental.paged_fill_cache(state.keys, ttnn.reshape(latent, [1, 1, t, self.rank]), written)
        if start == 0 and not ABSORBED_FIRST and t > WAVE_ROWS:  # a long wave's first chunk
            attn = self._first_wave(q, latent, lambda qb: self._absorbed_attend(qb, state, attended, None, WAVE_ROWS))
        elif start == 0 and not ABSORBED_FIRST:  # the first chunk: keys and values expanded by kv_b, causal within it
            attn = self._first_attend(q, latent)
        else:  # absorbed: q~ over its user's latent pages
            attn = self._absorbed_attend(q, state, attended, at, start)
        ttnn.deallocate(q)
        ttnn.deallocate(latent)
        flat = self._join_heads(attn)
        partial = wave_linear(self.mesh, flat, self.o_proj, compute_kernel_config=WEIGHT_MM)
        return tp_all_reduce(self.mesh, partial), state

    def _sdpa(self) -> dict:
        return dict(scale=self.scale, compute_kernel_config=self.sdpa_compute or HIFI4_FP32)

    def _first_attend(self, q, latent):
        """A first chunk's attention: keys and values expanded by kv_b from its latents, causal within it."""
        kv = self._expand(latent)
        key, value = self._kv_heads(kv)
        ttnn.deallocate(kv)
        attn = ttnn.transformer.scaled_dot_product_attention(q, key, value, is_causal=True, **self._sdpa())
        for tensor in (key, value):
            ttnn.deallocate(tensor)
        return attn

    def _absorbed_attend(self, q, state: TtDsaState, attended, at, start: int):
        """A later chunk's attention (q [1, 16, T, 256] at `start`): q~ over its user's latent pages (`attended`; `at`
        the start on the device, a captured chunk's, else None)."""
        t = q.shape[-2]
        q_t = self._absorb(q)
        # the op takes the chunk's start in whole K chunks: the largest of 128 / 64 / 32 rows that divides it
        k_chunk = TRACED_K_CHUNK if at is not None else next(c for c in (128, 64, PAGE) if start % c == 0)
        q_chunk = next(c for c in (128, 64, PAGE) if t % c == 0)
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        if at is not None:  # the start from the device: one program for every start (the same values)
            out = ttnn.transformer.chunked_scaled_dot_product_attention(
                q_t, state.keys, state.keys, attended, chunk_start_idx_tensor=at, program_config=program, **self._sdpa()
            )
        else:
            out = ttnn.transformer.chunked_flash_mla_prefill(
                q_t, state.keys, self.rank, attended, start, program_config=program, **self._sdpa()
            )  # [1, 16, T, 512]
        ttnn.deallocate(q_t)
        attn = self._values(out)
        ttnn.deallocate(out)
        return attn

    def _first_wave(self, q, latent, rest):
        """A first chunk of more than WAVE_ROWS tokens (a long prefill wave): its first
        WAVE_ROWS tokens as a first chunk (`_first_attend`), the others by `rest` (their q -> attention) as the chunk at
        WAVE_ROWS: the values of waves of WAVE_ROWS.  q [1, 16, T, 256], latent [1, 1, T, 512] -> [1, 16, T, 256]."""
        t, h, heads, dq = q.shape[-2], WAVE_ROWS, q.shape[1], q.shape[-1]
        qa = ttnn.slice(q, [0, 0, 0, 0], [1, heads, h, dq])
        la = ttnn.slice(latent, [0, 0, 0, 0], [1, 1, h, latent.shape[-1]])
        first = self._first_attend(qa, la)
        ttnn.deallocate(qa)
        ttnn.deallocate(la)
        qb = ttnn.slice(q, [0, 0, h, 0], [1, heads, t, dq])
        later = rest(qb)
        ttnn.deallocate(qb)
        attn = ttnn.concat([first, later], dim=2)
        ttnn.deallocate(first)
        ttnn.deallocate(later)
        return attn

    def _prefill_sp(self, x, state: TtDsaState, tables: DsaSpTables):
        """A sequence-parallel chunk (`sp_tables`): x [1, 1, T, 4096] per mesh row, its segment of one request ->
        (out, state).  Every row's latents, gathered over the rows, go into each row's target user; each row's
        queries then attend causally from its segment's start (a device tensor per row) over that user's latents, as
        a captured later chunk does (the flexible chunked SDPA)."""
        t = x.shape[-2]
        q, latent = self._project(x)
        q = self._q_heads(q)  # [1, 16, T, 256]
        every = ttnn.all_gather(ttnn.reshape(latent, [1, 1, t, self.rank]), dim=2, cluster_axis=ROW_AXIS)
        ttnn.deallocate(latent)
        ttnn.experimental.paged_fill_cache(state.keys, every, tables.written)  # [1, 1, 8 T, 512]: row r's at r T
        ttnn.deallocate(every)
        q_t = self._absorb(q)
        ttnn.deallocate(q)
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=next(c for c in (128, 64, PAGE) if t % c == 0),
            k_chunk_size=TRACED_K_CHUNK,
            exp_approx_mode=False,
        )
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_t,
            state.keys,
            state.keys,
            tables.attended,
            chunk_start_idx_tensor=tables.start,
            program_config=program,
            scale=self.scale,
            compute_kernel_config=self.sdpa_compute or HIFI4_FP32,
        )  # [1, 16, T, 512]
        ttnn.deallocate(q_t)
        attn = self._values(out)
        ttnn.deallocate(out)
        flat = self._join_heads(attn)
        partial = wave_linear(self.mesh, flat, self.o_proj, compute_kernel_config=WEIGHT_MM)
        return tp_all_reduce(self.mesh, partial), state

    def step(self, x, state: TtDsaState, device_positions: DsaPositions):
        """x [1, 1, B, 4096] bf16 (the same on a row's chips); device_positions from `positions_on_device` (made once
        per step and shared by the DSA layers) -> (out [1, 1, B, 4096] bf16, state)."""
        if self.long:
            return self._step_indexer(x, state, device_positions)
        q, latent = self._project(x)
        heads = q_heads(self.mesh, q, self.heads)  # [1, 16, B, 256] by one op of whole-tile copies
        ttnn.deallocate(q)
        return self._step_latent(heads, latent, state, device_positions)

    def rewrite(self, x, state: TtDsaState, device_positions: DsaPositions) -> None:
        """x's latents [1, 1, B, 4096] written into the cache at the positions again, with no attention (MTP's verify
        step with two drafts: the first level's rows back after the second level wrote over them).  The same matmul
        and norm as `_project`'s, so the same values."""
        if self.long:
            raise ValueError("rewrite: the dense latent cache only")
        batch = x.shape[-2]
        mm = dict(compute_kernel_config=HIFI4_FP32)
        a = decode_linear(self.mesh, x, self.a_proj, **mm)  # [1, 1, B, 1536 + 512]
        latent = ttnn.slice(a, [0, 0, 0, self.q_rank], [1, 1, batch, a.shape[-1]])
        ttnn.deallocate(a)
        normed = ttnn.rms_norm(latent, weight=self.kv_a_norm, epsilon=self.eps, **mm)
        ttnn.deallocate(latent)
        self._write_latents(normed, state, device_positions)

    def _write_latents(self, latent, state: TtDsaState, device_positions: DsaPositions) -> None:
        """The new latents [1, 1, B, 512] into the cache at the positions (deallocated)."""
        owners = device_positions.split or latent.shape[-2]
        latent_write(self.mesh, latent, state.keys, device_positions.cur_pos, owners, self.pages)
        ttnn.deallocate(latent)

    def _step_latent(self, heads, latent, state: TtDsaState, device_positions: DsaPositions):
        """Decode with the latent cache: the heads [1, 16, B, 256] (`dsa_latent.q_heads`), the new latents
        [1, 1, B, 512] -> (out, state)."""
        batch = heads.shape[2]
        self._write_latents(latent, state, device_positions)
        q_t = ttnn.transpose(self._absorb(heads), 1, 2)  # [1, B, 16, 512]
        ttnn.deallocate(heads)
        parts = []
        for (u0, u1), (cur, pages) in zip(self.attend_parts(batch), device_positions.attend):
            part = q_t if u1 - u0 == batch else ttnn.slice(q_t, [0, u0, 0, 0], [1, u1, self.heads, self.rank])
            parts.append(
                ttnn.transformer.paged_flash_multi_latent_attention_decode(
                    part,
                    state.keys,
                    head_dim_v=self.rank,
                    page_table_tensor=pages,
                    is_causal=True,
                    cur_pos_tensor=cur,
                    scale=self.scale,
                    program_config=self.sdpa_program,
                    compute_kernel_config=self.sdpa_compute,
                )
            )  # [1, b, 16, 512]
        ttnn.deallocate(q_t)
        out = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=1)
        heads = self._values(ttnn.transpose(out, 1, 2))  # [1, 16, B, 256]
        ttnn.deallocate(out)
        # [1, 1, B, 16 * 256] in one move (a transpose and a reshape were 4 + 22 us at 128 users)
        flat = ttnn.experimental.nlp_concat_heads(heads)
        ttnn.deallocate(heads)
        out_cfg = ar_input_config(self.mesh, flat.shape[-2], self.o_proj.shape[-1])
        partial = decode_linear(self.mesh, flat, self.o_proj, compute_kernel_config=WEIGHT_MM, memory_config=out_cfg)
        return tp_all_reduce(self.mesh, partial), state

    def _compress(self, rk, rg):
        """Pools' keys from their 4 tokens' indexer keys and gates, rows 0-3 of each [N, 1, 32, 128] (rows 4-31 left
        out by pool_bias's -inf): per feature, softmax over the 4 of (gate + slot bias) in fp32, the weights rounded to
        bf16, times the keys, summed (the reference's `Dsa.indexer`) -> [N, 1, 1, 128] bf16."""
        t = ttnn.add(ttnn.typecast(rg, ttnn.float32), self.pool_bias)
        e = ttnn.exp(ttnn.subtract(t, ttnn.max(t, dim=-2, keepdim=True)))
        ttnn.deallocate(t)
        w = ttnn.typecast(ttnn.divide(e, ttnn.sum(e, dim=-2, keepdim=True)), ttnn.bfloat16)
        ttnn.deallocate(e)
        prod = ttnn.multiply(w, rk)
        ttnn.deallocate(w)
        key = ttnn.sum(prod, dim=-2, keepdim=True)
        ttnn.deallocate(prod)
        return key

    def _chunk_pool_keys(self, ik, ig):
        """A prefill chunk's pools' keys from its tokens' indexer keys and gates [1, 1, T, 128] -> [1, 1, T / 4, 128]."""
        t, d, kp = ik.shape[-2], self.index_dim, self.kpool
        if PREFILL_POOL_KEYS_OP and t % (kp * PAGE) == 0 and kp == 4 and d == 128:
            from .dsa_indexer import pool_keys

            return pool_keys(self.mesh, ik, ig, self.pool_rows_bias)

        def pools(v):  # [1, 1, T, d] -> [T / 4, 1, 32, d]: pool p's tokens in rows 0-3 of page p, zeros below
            rm = ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT)
            padded = ttnn.pad(ttnn.reshape(rm, [t // kp, 1, kp, d]), [(0, 0), (0, 0), (0, PAGE - kp), (0, 0)], 0.0)
            ttnn.deallocate(rm)
            out = ttnn.to_layout(padded, ttnn.TILE_LAYOUT)
            ttnn.deallocate(padded)
            return out

        rk, rg = pools(ik), pools(ig)
        key = self._compress(rk, rg)  # [T / 4, 1, 1, d]
        ttnn.deallocate(rk)
        ttnn.deallocate(rg)
        return ttnn.reshape(key, [1, 1, t // kp, d])

    def _q_heads(self, q):
        """[1, 1, T, 16 x 256] -> [1, 16, T, 256]."""
        lo, hi = ttnn.experimental.nlp_create_q_heads_split(q, num_heads=self.heads, split_head_dim=self.dk // 2)
        heads = ttnn.concat([lo, hi], dim=-1)
        ttnn.deallocate(lo)
        ttnn.deallocate(hi)
        return heads

    def _kv_heads(self, kv):
        """kv_b's [1, 1, T, 16 x (256 + 256)] -> (key, value) [1, 16, T, 256] each."""
        return ttnn.experimental.nlp_create_q_heads_split(kv, num_heads=self.heads, split_head_dim=self.dk)

    def _join_heads(self, attn):
        """[1, 16, T, 256] -> [1, 1, T, 16 x 256]."""
        return ttnn.experimental.nlp_concat_heads(attn)

    @staticmethod
    def _rm_store(latent_rm, state: TtDsaState):
        """Latent rows [1, 1, n, 512] bf16 row-major (deallocated) as the long cache stores them: FP8 e4m3 for an fp8
        cache (KV_LATENT_FP8), else bf16 (KV_ROUND_LATENT: rounded as FP8 there and back)."""
        if state.rm.dtype == ttnn.fp8_e4m3:
            return to_fp8(latent_rm)
        return round_kv(latent_rm, KV_ROUND_LATENT)

    def _kv_format(self, state: TtDsaState):
        """sparse_sdpa's form of the long cache's latents: bf16, or FP8_E4M3 (KV_LATENT_FP8)."""
        return self.sparse_format.FP8_E4M3 if state.rm.dtype == ttnn.fp8_e4m3 else self.sparse_format.BF16

    def _sparse_attend_split(self, q, state: TtDsaState, tokens, own: bool = False):
        """PREFILL_QSPLIT: `_sparse_attend` with each chip of the row taking all 64 heads of its quarter of the queries
        (chip c the queries c S / 4 ..), the q and the outputs moved over the row by all_to_all_async_generic, the
        selected tokens of its queries taken (`row_split.take`; `own`: `tokens` are already its quarter's,
        PREFILL_INDEX_QSPLIT) -> [1, 16, S, 256] (this chip's heads, every query)."""
        from .common import TP_AXIS
        from .row_split import take

        s = q.shape[-2]
        quarter = s // COLS
        # chip c's quarter of every chip's heads, concatenated by source chip: its 64 heads
        own_t = self._absorb(q)  # [1, 16, S, 512]
        q_t = ttnn.experimental.all_to_all_async_generic(own_t, in_dim=1, out_dim=2, cluster_axis=TP_AXIS)
        ttnn.deallocate(own_t)  # q_t [1, 64, S / 4, 512]
        q_rm = ttnn.to_layout(q_t, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(q_t)
        width = tokens.shape[-1]
        if own:
            mine = tokens
        else:
            offsets = [c * quarter for c in range(COLS)]
            mine = take(self.mesh, tokens, [1, 1, quarter, width], 1, quarter, quarter, offsets)
        out = ttnn.transformer.sparse_sdpa(
            q_rm,
            state.rm,
            mine,
            v_dim=self.rank,
            kv_format=self._kv_format(state),
            scale=self.scale,
            k_chunk_size=128,
            compute_kernel_config=HIFI4_FP32,
        )  # [1, 64, S / 4, 512] row-major
        ttnn.deallocate(q_rm)
        if not own:
            ttnn.deallocate(mine)
        tiled = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        ttnn.deallocate(out)
        # chip c's 16 heads of every chip's quarter, concatenated by source chip: all queries
        own = ttnn.experimental.all_to_all_async_generic(tiled, in_dim=2, out_dim=1, cluster_axis=TP_AXIS)
        ttnn.deallocate(tiled)  # own [1, 16, S, 512]
        heads = self._values(own)
        ttnn.deallocate(own)
        return heads

    def _split_attend(self, q, state: TtDsaState, tokens, splits: int):
        """SPLIT_K: `_sparse_attend` with each query's keys over `splits` cores (dsa_indexer.sparse_split) and the
        partials combined (sparse_combine -> [1, S, 16, 512] tiles, transposed to the heads' [1, 16, S, 512])."""
        from .dsa_indexer import sparse_combine, sparse_split

        q_t = self._absorb(q)  # [1, 16, S, 512]
        q_s = ttnn.transpose(q_t, 1, 2)  # [1, S, 16, 512]: a token's heads as its tile row, read as tiles by the split
        ttnn.deallocate(q_t)
        parts = sparse_split(self.mesh, q_s, state.rm, tokens, splits, self.scale)
        ttnn.deallocate(q_s)
        heads16 = sparse_combine(self.mesh, parts, tokens, splits, self.scale)  # [1, S, 16, 512]
        ttnn.deallocate(parts)
        out = ttnn.transpose(heads16, 1, 2)  # [1, 16, S, 512]
        ttnn.deallocate(heads16)
        heads = self._values(out)  # [1, 16, S, 256]
        ttnn.deallocate(out)
        return heads

    def _shard_attend(self, q, state: TtDsaState, tokens):
        """The split latents' attention: q per head [1, 16, S, 256], this chip's token rows [1, 1, S, SHARD_WIDTH] of
        its latents -> [1, 16, S, 256]: q~ = q W_kb gathered over the row row-major (64 heads), each query's two 32-head
        units (or one 64-head unit) over this chip's share (sparse_split, empty shares neutral), the partials repacked
        by destination chip, exchanged (all_to_all) and combined for this chip's 16 heads over the 4 chips x G splits.
        Every query of the step or chunk in one sparse_split, one exchange and one combine (the kernels take more units
        than cores, a core's in turn; an earlier form's blocks of 60 / 120 queries: the prefill of 1 x 8K +19.5 %)."""
        from .common import TP_AXIS
        from .dsa_indexer import PART_TILES, shard_repack, sparse_combine, sparse_split

        n = q.shape[-2]
        q_t = self._absorb(q)  # [1, 16, S, 512]
        q_r = ttnn.to_layout(q_t, ttnn.ROW_MAJOR_LAYOUT)  # (a query's head rows read by the split: no transpose)
        ttnn.deallocate(q_t)
        q_all = ttnn.all_gather(q_r, dim=1, cluster_axis=TP_AXIS)  # [1, 64, S, 512] row-major: unit 2 s + h query
        ttnn.deallocate(q_r)  # s's heads 32 h .. (sparse_split's rows (32 h + r) S + s)
        cores = self.mesh.compute_with_storage_grid_size()
        cores = cores.x * cores.y
        # the splits of a query's 32-head unit (sparse_split's G): 1 from 16 queries a chip, else 2; fewer when the
        # units outnumber the cores (a core's in turn)
        g = 1 if n >= 16 else 2
        g = max(1, min(g, cores // (2 * n)))
        if (n * g) % 2:
            raise ValueError(f"the split latents' attention: {n} queries x {g} splits (even)")
        # q~ gathered row-major and read by the split as a query's head rows (no transpose: an earlier form's
        # tiles transposed to a query's tile rows cost 940 us a prefill segment of 512 queries); a query's 64 heads
        # one unit (two tile rows over its keys read once, `sparse_split` H = 64) when its two 32-head units would not
        # fit the cores at once (128 users' Q = 64, a prefill chunk: the split 235 -> 160 / 1,426 -> 939 us)
        upq = 1 if 2 * n * g > cores else 2
        parts = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(
                ttnn.Shape([1, 1, 2 * n * g * 32, PART_TILES * 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                buffer_type=ttnn.BufferType.DRAM,
            ),
            self.mesh,
        )  # fmt: skip
        sparse_split(self.mesh, q_all, state.rm, tokens, g, self.scale, upq=upq, write_empty=True, out=parts)
        x = shard_repack(self.mesh, parts, n, g)
        ttnn.deallocate(parts)
        y = ttnn.experimental.all_to_all_async_generic(x, in_dim=1, out_dim=2, cluster_axis=TP_AXIS)  # [1, 4, R, ..]
        ttnn.deallocate(x)
        heads16 = sparse_combine(self.mesh, y, tokens, COLS * g, self.scale, xchip=(n, g))  # [1, S, 16, 512]
        ttnn.deallocate(y)
        ttnn.deallocate(q_all)
        out = ttnn.transpose(heads16, 1, 2)  # [1, 16, S, 512]
        ttnn.deallocate(heads16)
        heads = self._values(out)  # [1, 16, S, 256]
        ttnn.deallocate(out)
        return heads

    def _sparse_attend(self, q, state: TtDsaState, tokens, prefill: bool = False):
        """q per head [1, 16, S, 256] over each query's tokens (`select_tokens` [1, 1, S, 2176]) of the row-major
        latents: q~ = q W_kb, padded to 32 heads, sparse_sdpa, W_vb^T -> [1, 16, S, 256].  `prefill`: the padding and
        the cut row-major (the 16 zero heads a cached constant), or the split by queries (PREFILL_QSPLIT); decode with
        SPLIT_K: the keys split over cores (`_split_attend`)."""
        s = q.shape[-2]
        if prefill and PREFILL_QSPLIT and s % (32 * COLS) == 0:
            return self._sparse_attend_split(q, state, tokens)
        if not prefill and SPLIT_K > 1:
            grid = self.mesh.compute_with_storage_grid_size()
            splits = min(SPLIT_K, (grid.x * grid.y) // s)
            if splits >= 2:
                return self._split_attend(q, state, tokens, splits)
        q_t = self._absorb(q)  # [1, 16, S, 512]
        rm_moves = prefill
        if rm_moves:
            q_rm = ttnn.to_layout(q_t, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(q_t)
            from .common import const_zeros

            zeros16 = const_zeros(self.mesh, [1, self.heads, s, self.rank], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            q32 = ttnn.concat([q_rm, zeros16], dim=1)  # [1, 32, S, 512] row-major
            ttnn.deallocate(q_rm)
        else:
            pad = ttnn.zeros_like(q_t)
            q32 = ttnn.to_layout(ttnn.concat([q_t, pad], dim=1), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(q_t)
            ttnn.deallocate(pad)
        out = ttnn.transformer.sparse_sdpa(
            q32,
            state.rm,
            tokens,
            v_dim=self.rank,
            kv_format=self._kv_format(state),
            scale=self.scale,
            k_chunk_size=128,
            compute_kernel_config=HIFI4_FP32,
        )  # [1, 32, S, 512]
        ttnn.deallocate(q32)
        if rm_moves:
            half = ttnn.slice(out, [0, 0, 0, 0], [1, self.heads, s, self.rank])  # row-major (maybe a view)
            tiled = ttnn.to_layout(half, ttnn.TILE_LAYOUT)
            _free(half, out)
            ttnn.deallocate(out)
            heads = self._values(tiled)  # [1, 16, S, 256]
            ttnn.deallocate(tiled)
            return heads
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        heads = self._values(ttnn.slice(out, [0, 0, 0, 0], [1, self.heads, s, self.rank]))  # [1, 16, S, 256]
        ttnn.deallocate(out)
        return heads

    def _indexed_attend(self, q, q_resid, hw, params, state: TtDsaState, tb: DsaPrefillTables, start: int):
        """A long cache's chunk's attention over the tokens its indexer picks: q [1, 16, T, 256], the normed q_resid
        [1, 1, T, 1536], the indexer's head weights `hw` and the chunk's `params` rows (all deallocated here), at
        `start` -> [1, 16, T, 256]."""
        from .dsa_indexer import pool_scores, select_tokens

        shard = self.shard_kv()  # split latents: every chip all the queries over its tokens' latents
        params_full = params
        t = q.shape[-2]
        qi = ttnn.linear(q_resid, self.idx_wq, compute_kernel_config=HIFI4_FP32)  # [1, 1, T, 32 * 128]
        ttnn.deallocate(q_resid)
        own = PREFILL_INDEX_QSPLIT and PREFILL_QSPLIT and t % (32 * COLS) == 0  # (`_sparse_attend_split`'s queries)
        if own:  # each chip its quarter of the queries: [1, 1, T / 4, ..] of qi, hw and params
            from .row_split import take

            n = t // COLS
            quarter = take(self.mesh, qi, [1, 1, n, qi.shape[-1]], 1, n // 32 * (qi.shape[-1] // 32), 0,
                           [c * (n // 32) * (qi.shape[-1] // 32) for c in range(COLS)])  # fmt: skip
            ttnn.deallocate(qi)
            qi = quarter
            hw_q = take(self.mesh, hw, [1, 1, n, hw.shape[-1]], 1, n // 32, 0, [c * (n // 32) for c in range(COLS)])
            ttnn.deallocate(hw)
            hw = hw_q
            ps = params.shape
            params_q = take(self.mesh, params, [ps[0], ps[1], n, ps[-1]], 1, n, 0, [c * n for c in range(COLS)])
            if params is not tb.params and not shard:
                ttnn.deallocate(params)
            params = params_q
            t_rows = n
        else:
            t_rows = t
        # [T, 4096] (tiled) to the scores op's [T, 1, 32, 128] tiles by untilize, a row-major view and tilize (~40 us
        # at T = 512, against a 380 us ReshapeView on the tiles)
        qi_rm = ttnn.to_layout(qi, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(qi)
        qv = ttnn.reshape(qi_rm, [t_rows, 1, self.index_heads, self.index_dim])  # (a view)
        q4 = ttnn.to_layout(qv, ttnn.TILE_LAYOUT)
        _free(qv, qi_rm)
        ttnn.deallocate(qi_rm)
        p = tb.pool_read.shape[-1] * PAGE
        # pools past the chunk's end are -inf either way: a captured chunk computes the whole bucket (any start)
        computed = None if tb.traced else (start + t) // self.kpool
        rows = pool_scores(self.mesh, q4, state.pools, hw, params, p, True, computed)
        ttnn.deallocate(q4)
        ttnn.deallocate(hw)
        top = ttnn.experimental.topk_large_indices(rows, k=self.index_k)  # [1, 1, T, 512] uint32
        ttnn.deallocate(rows)
        span = tb.pool_read.shape[-1] * PAGE * self.kpool  # the tokens the chunk's pools can reach (its bucket)
        if shard and span <= SHARD_PREFIX:  # the gathered prefix: its user's prefix over the row, the unsplit attention
            from .common import TP_AXIS
            from .dsa_indexer import prefix_rows

            n = span // COLS
            mine = prefix_rows(self.mesh, state.rm, params_full, n)  # this chip's rows of the prefix
            every_rm = ttnn.all_gather(
                mine, dim=2, cluster_axis=TP_AXIS
            )  # [1, 1, span, 512]: token t at (t % 4) n + t / 4
            ttnn.deallocate(mine)
            tokens = select_tokens(self.mesh, top, params, INDEX_TOPK_WIDTH, self.kpool, prefix_q=n)
            ttnn.deallocate(top)
            if params is not tb.params:
                ttnn.deallocate(params)
            if params_full is not params and params_full is not tb.params:
                ttnn.deallocate(params_full)
            view = TtDsaState(keys=None, rm=every_rm)
            if own:
                attn = self._sparse_attend_split(q, view, tokens, own=True)
            else:
                attn = self._sparse_attend(q, view, tokens, prefill=True)
            ttnn.deallocate(tokens)
            ttnn.deallocate(every_rm)
            return attn
        if shard:  # every query's top 512 on every chip (each chip its quarter's, gathered), its tokens' rows
            from .common import TP_AXIS

            if own:
                every = ttnn.all_gather(top, dim=2, cluster_axis=TP_AXIS)  # [1, 1, T, 512]: chip c's quarter at c T / 4
                ttnn.deallocate(top)
                top = every
                if params is not tb.params:
                    ttnn.deallocate(params)
            tokens = select_tokens(self.mesh, top, params_full, SHARD_WIDTH, self.kpool, shard=True)
            ttnn.deallocate(top)
            if params_full is not tb.params:
                ttnn.deallocate(params_full)
            attn = self._shard_attend(q, state, tokens)
            ttnn.deallocate(tokens)
            return attn
        tokens = select_tokens(self.mesh, top, params, INDEX_TOPK_WIDTH, self.kpool)
        ttnn.deallocate(top)
        if params is not tb.params:
            ttnn.deallocate(params)
        if own:
            attn = self._sparse_attend_split(q, state, tokens, own=True)
        else:
            attn = self._sparse_attend(q, state, tokens, prefill=True)
        ttnn.deallocate(tokens)
        return attn

    def _index_project(self, x):
        """x [1, 1, S, 4096] -> the indexer key (LayerNormed), the pool gate [1, 1, S, 128] each and the head weights
        [1, 1, S, 32]."""
        rows, d = x.shape[-2], self.index_dim
        mm = dict(compute_kernel_config=HIFI4_FP32)
        ip = decode_linear(self.mesh, x, self.idx_proj, **mm)  # [1, 1, S, 128 + 128 + 32]
        ik = ttnn.layer_norm(
            ttnn.slice(ip, [0, 0, 0, 0], [1, 1, rows, d]),
            weight=self.idx_norm_w,
            bias=self.idx_norm_b,
            epsilon=1e-6,
            **mm,
        )
        ig = ttnn.slice(ip, [0, 0, 0, d], [1, 1, rows, 2 * d])
        hw = ttnn.slice(ip, [0, 0, 0, 2 * d], [1, 1, rows, 2 * d + self.index_heads])
        ttnn.deallocate(ip)
        return ik, ig, hw

    def _prefill_indexer(self, x, state: TtDsaState, start: int, tb: DsaPrefillTables):
        """A long cache's prefill chunk (arguments as `prefill`, `tb` from `prefill_tables`)."""
        from .dsa_indexer import rows_write

        t = x.shape[-2]
        if t > min(self.index_k * self.kpool, 2048):
            raise ValueError(f"a {t}-token chunk: a long cache's prefill chunks are up to 2,048 tokens")
        mm = dict(compute_kernel_config=HIFI4_FP32)
        q, latent, q_resid = self._project(x, with_resid=True)
        q = self._q_heads(q)  # [1, 16, T, 256]
        latent_rm = ttnn.to_layout(latent, ttnn.ROW_MAJOR_LAYOUT)
        if tb.gathered:  # sequence parallel: every row's segment, [1, 1, 8 T, 512] (row r's at r T)
            every = ttnn.all_gather(latent_rm, dim=2, cluster_axis=ROW_AXIS)
            ttnn.deallocate(latent_rm)
            latent_rm = every
        latent_rm = self._rm_store(latent_rm, state)
        rows_write(self.mesh, latent_rm, state.rm, tb.rows, shard=self.shard_kv())  # (shard: each chip its tokens')
        ttnn.deallocate(latent_rm)
        ik, ig, hw = self._index_project(x)
        # the 4 tokens of the pool after the request's last real token (decode completes that pool), then the chunk's
        # pools' keys
        for cache, new in ((state.rkey, ik), (state.rgate, ig)):
            page = ttnn.matmul(tb.recent_pick, new, **mm)  # [1, 1, 32, 128]: a one-hot row each, exact
            ttnn.experimental.paged_fill_cache(cache, page, tb.recent_table)
            ttnn.deallocate(page)
        keys = self._chunk_pool_keys(ik, ig)
        ttnn.deallocate(ik)
        ttnn.deallocate(ig)
        if tb.gathered:  # every row's segment's pools, [1, 1, 8 T / 4, 128]
            every = ttnn.all_gather(keys, dim=2, cluster_axis=ROW_AXIS)
            ttnn.deallocate(keys)
            keys = every
        if state.pools.dtype != ttnn.bfloat16:  # (POOLS_DTYPE: the BFP cache's tiles, rounded as decode rounds them)
            from .dsa_indexer import bfp_pack

            low = bfp_pack(self.mesh, keys, state.pools.dtype)
            ttnn.deallocate(keys)
            keys = low
        ttnn.experimental.paged_fill_cache(state.pools, keys, tb.pool_written)
        ttnn.deallocate(keys)

        if start == 0 and not tb.gathered and t > WAVE_ROWS:  # a long wave's first chunk (`_first_wave`)

            def rest(qb):
                h = WAVE_ROWS
                qr = ttnn.slice(q_resid, [0, 0, h, 0], [1, 1, t, q_resid.shape[-1]])
                hb = ttnn.slice(hw, [0, 0, h, 0], [hw.shape[0], hw.shape[1], t, hw.shape[-1]])
                ps = tb.params.shape
                pb = ttnn.slice(tb.params, [0, 0, h, 0], [ps[0], ps[1], t, ps[-1]])
                return self._indexed_attend(qb, qr, hb, pb, state, tb, h)

            attn = self._first_wave(q, latent, rest)
            ttnn.deallocate(q_resid)
            ttnn.deallocate(hw)
        elif start == 0 and not tb.gathered:  # every token of the chunk kept (T <= 2,048): causal attention within it
            ttnn.deallocate(q_resid)
            ttnn.deallocate(hw)
            attn = self._first_attend(q, latent)
        else:
            attn = self._indexed_attend(q, q_resid, hw, tb.params, state, tb, start)
        ttnn.deallocate(q)
        ttnn.deallocate(latent)
        flat = self._join_heads(attn)
        ttnn.deallocate(attn)
        partial = wave_linear(self.mesh, flat, self.o_proj, compute_kernel_config=WEIGHT_MM)
        return tp_all_reduce(self.mesh, partial), state

    def _step_indexer(self, x, state: TtDsaState, pos: DsaPositions):
        """Decode past the dense limit: the indexer picks each user's 512 best visible pools and its tail, and the 16
        heads attend over those tokens' latents (`_sparse_attend`) -> (out [1, 1, B, 4096], state)."""
        from .dsa_indexer import index_update, pool_scores, rows_write

        batch = x.shape[-2]
        mm = dict(compute_kernel_config=HIFI4_FP32)
        q, latent, q_resid = self._project(x, with_resid=True)
        heads = q_heads(self.mesh, q, self.heads)  # [1, 16, B, 256] by one op of whole-tile copies
        ttnn.deallocate(q)
        q = heads
        # this token's latent into the row-major cache, its indexer key and gate into its pool's rows
        latent_rm = self._rm_store(ttnn.to_layout(latent, ttnn.ROW_MAJOR_LAYOUT), state)
        ttnn.deallocate(latent)
        shard = self.shard_kv()  # (split latents: each chip its tokens' rows only)
        if TRIM_WRITES:  # (the row from params: base + position)
            rows_write(self.mesh, latent_rm, state.rm, None, params=pos.params, shard=shard)
        else:
            rows_write(self.mesh, latent_rm, state.rm, pos.rows, shard=shard)
        ttnn.deallocate(latent_rm)
        ik, ig, hw = self._index_project(x)
        # each segment's recent rows of the key and gate, the pool's key and its row in the pool cache in one op: MTP's
        # virtual users (pos.split) a segment at a time (the tokens, then the drafts one position on), so that a
        # draft's slot (a new pool's first) never enters its token's (just complete) pool's key
        users = pos.split or batch
        index_update(
            self.mesh, ik, ig, pos.params, state.rkey, state.rgate, state.pools, self.pool_bias, users, batch // users
        )
        ttnn.deallocate(ik)
        ttnn.deallocate(ig)

        # pool scores of each user's query over its pools, the invisible ones -inf; each query's heads as its
        # row-major row (pool_scores tilizes it)
        proj = decode_linear(self.mesh, q_resid, self.idx_wq, **mm)  # [1, 1, B, 32 x 128]
        ttnn.deallocate(q_resid)
        qi = ttnn.untilize(proj)
        ttnn.deallocate(proj)
        shard = self.scores_shard()
        rows = pool_scores(
            self.mesh,
            qi,
            state.pools,
            hw,
            pos.params,
            self.pool_pages * PAGE,
            False,
            group=(users, batch // users),
            shard=shard,
        )  # [1, 1, B, P] (shard: [1, 1, B, P / 4], this chip's blocks)
        ttnn.deallocate(qi)
        ttnn.deallocate(hw)
        # (over the cache's whole width; the pools past a query's position are -inf.  Bounding it by each row's
        # visible pools gave the same picks.)
        top = ttnn.experimental.topk_large_indices(rows, k=self.index_k)  # [1, 1, B, 512] uint32
        every = None
        if shard:  # the row's top 512 among its chips' (split scores): picks in the gathered candidates
            from .common import TP_AXIS
            from .dsa_indexer import shard_candidates, shard_pick

            cand = shard_candidates(self.mesh, rows, top)
            ttnn.deallocate(top)
            every = ttnn.all_gather(cand, dim=3, cluster_axis=TP_AXIS)  # [1, 2, B, 4 x 512]
            ttnn.deallocate(cand)
            top = shard_pick(self.mesh, every, self.index_k)
        ttnn.deallocate(rows)
        return self._attend_selected(q, top, state, pos, every)

    def scores_shard(self) -> bool:
        """The split scores (SCORES_SHARD) for this cache: on from SCORES_SHARD tokens, whole pool pages a chip, 512 to
        65,536 pools a chip (a cache of 8K to 1M)."""
        return bool(
            SCORES_SHARD
            and self.max_seq >= SCORES_SHARD
            and self.pool_pages % COLS == 0
            and self.index_k <= self.pool_pages * PAGE // COLS <= 65536
        )

    def shard_kv(self) -> bool:
        """The split latents (SHARD_KV) for this cache: a long cache of at least SHARD_KV tokens."""
        return bool(SHARD_KV and self.long and self.max_seq >= SHARD_KV and self.max_seq % COLS == 0)

    def _attend_selected(self, q, top, state: TtDsaState, pos: DsaPositions, cand=None):
        """Decode's end: q's heads [1, 16, B, 256], the top 512 pools [1, 1, B, 512] uint32 (`cand`, split scores: picks
        in the row's gathered candidates) -> their tokens and the tail, the attention over them, o_proj -> (out [1, 1,
        B, 4096], state)."""
        from .dsa_indexer import select_tokens

        shard = self.shard_kv()
        width = SHARD_WIDTH if shard else INDEX_TOPK_WIDTH
        tokens = select_tokens(self.mesh, top, pos.params, width, self.kpool, cand=cand, shard=shard)
        ttnn.deallocate(top)
        if cand is not None:
            ttnn.deallocate(cand)
        heads = self._shard_attend(q, state, tokens) if shard else self._sparse_attend(q, state, tokens)
        ttnn.deallocate(q)
        ttnn.deallocate(tokens)
        flat = ttnn.experimental.nlp_concat_heads(heads)
        ttnn.deallocate(heads)
        out_cfg = ar_input_config(self.mesh, flat.shape[-2], self.o_proj.shape[-1])
        partial = decode_linear(self.mesh, flat, self.o_proj, compute_kernel_config=WEIGHT_MM, memory_config=out_cfg)
        return tp_all_reduce(self.mesh, partial), state

    def load_reference_state(self, latent: torch.Tensor, index_key: torch.Tensor, index_gate: torch.Tensor, w):
        """A test's start: the reference's state (`tt/dsa.py` DsaState tensors [U, T, ..], every row of the mesh the
        same) into a fresh indexer state: the latents row-major, the complete pools' keys, the current pool's tokens'
        keys and gates (at their rows of the recent pages)."""
        from .dsa_indexer import recent_row

        u, t = latent.shape[:2]
        d, kp = self.index_dim, self.kpool
        rep = dict(dims=(None, None))
        rm = torch.zeros(u, self.max_seq, self.rank, dtype=torch.bfloat16)
        rm[:, :t] = latent
        pools = torch.zeros(u, self.pool_pages * PAGE, d, dtype=torch.bfloat16)
        complete = t // kp
        logits = index_gate[:, : complete * kp].float().view(u, complete, kp, d) + w.index_pool_bias.float()[None, None]
        weights = torch.nan_to_num(logits.softmax(dim=2)).to(torch.bfloat16)
        pools[:, :complete] = (weights * index_key[:, : complete * kp].view(u, complete, kp, d)).sum(dim=2)
        recent_k = torch.zeros(u + 1, 1, PAGE, d, dtype=torch.bfloat16)
        recent_g = torch.zeros(u + 1, 1, PAGE, d, dtype=torch.bfloat16)
        tail = t - complete * kp
        for j in range(tail):
            recent_k[:u, 0, recent_row(complete * kp + j)] = index_key[:, complete * kp + j]
            recent_g[:u, 0, recent_row(complete * kp + j)] = index_gate[:, complete * kp + j]
        scratch = torch.zeros(1, PAGE, d, dtype=torch.bfloat16)
        pages = torch.cat([pools.view(u * self.pool_pages, PAGE, d), scratch]).view(-1, 1, PAGE, d)
        pdt = pools_dtype()
        if pdt != ttnn.bfloat16:  # (rounded as the device rounds them: the host's packer then keeps the values)
            from models.demos.blackhole.glm53_flash.tools.kv_round import bfp_rna

            pages = bfp_rna(pages.float(), "bfp8" if pdt == ttnn.bfloat8_b else "bfp4")
        rm_dtype = ttnn.fp8_e4m3 if KV_LATENT_FP8 else ttnn.bfloat16
        if self.shard_kv():  # split latents: chip c's latents of the tokens t = c mod 4, at t / 4
            shards = []
            for _r in range(ROWS):
                for c in range(COLS):
                    part = rm[:, c::COLS].reshape(1, 1, -1, self.rank)
                    shards.append(
                        ttnn.from_torch(
                            part.float() if KV_LATENT_FP8 else part.contiguous(),
                            dtype=rm_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        )
                    )
            rm_dev = ttnn.to_device(ttnn.from_host_shards(shards, self.mesh.shape), self.mesh)
        else:
            rm_dev = put(
                self.mesh,
                rm.reshape(1, 1, u * self.max_seq, self.rank).float()
                if KV_LATENT_FP8
                else rm.reshape(1, 1, -1, self.rank),
                dtype=rm_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                **rep,
            )
        return TtDsaState(
            keys=None,
            rm=rm_dev,
            pools=put(self.mesh, pages, dtype=pdt, **rep),
            rkey=put(self.mesh, recent_k, **rep),
            rgate=put(self.mesh, recent_g, **rep),
        )
