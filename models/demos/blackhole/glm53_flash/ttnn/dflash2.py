"""The DFlash2 block drafter (canada-quant/GLM-5.3-Flash-DFlash2-G; the CPU reference `tt/dflash2.py`) on the 8x4 mesh.

Layout: the mesh rows hold their own users (data parallel, as the target's decode), a row's 4 chips split the drafter
(tensor parallel, as the target's attention): chip c the q heads 8 c .. 8 c + 7 and the KV heads 2 c, 2 c + 1, a quarter
of the MLP's intermediate, a quarter of fc's outputs; o_proj and down split by input, all-reduced over the row.  The
dynamic convolutions' kernel projections and the norms are on every chip.  The LM head is the target's (the same
weights, ttnn/head.py's mesh form: chip k = 4 r + c a 1/32 of the vocabulary for every row's tokens): each chip's top
16 of its part (`dflash2_ops.top16`), gathered over the mesh, merged for each row's users (`top16_merge`).

Context KV (`write_context`): for each written token (taps of the target's 9 layers), c = hidden_norm(fc(taps)) and per
layer K = rope(k_norm(k_proj c)), V = v_proj c, into a paged cache per chip, [8 layers x 2 U virtual users x P pages
+ 1, 1, 32, 128] bf16 for K and for V: virtual user v = (user v // 2, KV head v % 2), position p at row p % 32 of page
l 2 U P + v P + p // 32.  The drafter's block (`draft`): [bonus, mask x (B - 1)] per user, the 8 layers, the head's
candidates and the selector's walk; its K / V are written at positions s .. s + B - 1 before its attention (paged sdpa
decode over the virtual users, the 32 query rows of a virtual user its 4 q heads x B positions, cur_pos s + B - 1;
non-causal in the block as the reference).  A later `write_context` writes the accepted positions over them; positions
past those are written by the next block before they are read.

Rounding as the reference: bf16 activations, fp32 sums; the convolutions' coefficients and the RoPE in fp32, rounded
once; the norms' weights after the bf16 normalized value (q / k norms); the candidates' logits fp32; the selector's
pairwise terms exact products (the fp32 product pred * proj split in two bf16 parts) summed in fp32.

The KV writes, the RoPE rows, the candidates and the walk, and (in place of several ttnn ops per use) the
convolutions, the heads' regrouping and the RoPE are the drafter's own ops (`dflash2_ops`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.glm53_flash.paths import drafter_dir
from models.demos.blackhole.glm53_flash.tt.dflash2 import DFlash2Config, _rope_cos_sin
from models.demos.blackhole.glm53_flash.ttnn import dflash2_ops as ops
from models.demos.blackhole.glm53_flash.ttnn import fc_ksplit
from models.demos.blackhole.glm53_flash.ttnn.common import (
    COLS,
    HIFI4_FP32,
    ROW_AXIS,
    ROWS,
    TP_AXIS,
    WAVE_GRID_Y,
    WEIGHT_DTYPES,
    ar_input_config,
    ar_input_dtype,
    ar_keep_l1,
    decode_linear,
    fidelity,
    put,
    tp_all_reduce,
    tp_weight,
)
from models.demos.blackhole.glm53_flash.ttnn.dsa import round_kv
from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm

KV_HEADS_CHIP = 2  # a chip's KV heads (8 over a row's 4 chips)
# The drafter's own ops (dflash2_ops; the same values as the ttnn ops they replaced, which were removed): the heads'
# regrouping for the paged sdpa decode (heads_split / heads_merge), the dynamic convolutions (conv), the q / k norms'
# weight and the RoPE (rope: q and k in one tensor from heads_split, q's virtual users then k's, one rms_norm and one
# rope op for both), the selector's pairwise terms (sel_scores) and the MLP's gate up to 128 rows (silu_mul).
# o_proj and down write straight into the row all-reduce's L1 shards, and its sums stay there for the convolution op
# (the target's AR_L1_IN / AR_L1_OUT: two moves an all-reduce fewer), when the all-reduce is all_reduce_async (up to
# 32 rows a mesh row); False: through DRAM
AR_L1 = True
# weights of at least this many tiles by decode_matmul up to 64 rows (decode_linear's own threshold is 6,000: the
# kernel projections and o_proj, 4,096 tiles, ran its 1D config on 32 cores at ~280 GB/s)
DM_MIN_TILES = 4096


def conv_fused_on(kp=None) -> bool:
    """GLM53_DFLASH2_CONV_FUSED (read at each call; on by default, 0: the conv op as before): each sublayer's
    convolution before its body (side 0) made inside the kernel projection's decode_matmul (its conv_* arguments: on
    the cores its blocks leave free, the conv op's arithmetic tile for tile, the same bits), in place of the conv op.
    Only where decode_matmul's one compute config takes HiFi4 as the conv op does (DECODE_MATMUL_FIDELITY unset or 4;
    1, LoFi on BFP4 weights only, unless kp itself is BFP4): the fused conv's products (delta x E, S x) run at the kp
    matmul's fidelity (as kda.fg_fused_on)."""
    f = os.environ.get("DECODE_MATMUL_FIDELITY", "4")
    return os.environ.get("GLM53_DFLASH2_CONV_FUSED", "1") == "1" and (
        f == "4" or (f == "1" and kp is not None and kp.dtype != ttnn.bfloat4_b)
    )


def norm_shard_on() -> bool:
    """GLM53_DFLASH2_NORM_SHARD (read at each call; on by default): the sublayers' residual stream kept in the
    rms_norm's width-sharded form (16 cores' L1) from the first sublayer's sum on: the norm on the shards
    (`TtRmsNorm.on_shard`, the same op and configs as its `__call__`), the kernel projection's decode_matmul reading the
    normed shards as its x, the residual sum written straight into the shards (`ttnn.add`'s memory config): no I2S / S2I
    around the norms; the values the same (moves only); 0: as before."""
    return os.environ.get("GLM53_DFLASH2_NORM_SHARD", "1") == "1"


# a study of the KV cache's precision, a quality form (no speed): the K / V the drafter writes (its context's and its
# block's) rounded as a BFP8 (8) / BFP4 (4) cache would hold them (ttnn.typecast there and back),
# still stored in bf16.  0: off.  Read at each call
KV_ROUND = int(os.environ.get("GLM53_DFLASH2_KV_ROUND", "0"))
# ... the storage form (read when a TtDFlash2 is made): the K / V caches in BFP8_b / BFP4_b ("bfp8" / "bfp4"; their
# writes cast by dsa_indexer.bfp_cast, the packer's precise form, then kv_write's BFP form; read by the paged sdpa
# decode as it reads any BFP cache).  BFP8 by default (the target's text is the same under the exact acceptance, the
# acceptance of the 11 reference records the same; DFlash2's step at a 32K cache 8 / 32 / 128 users -0.7 / -2.3 /
# -6.5 %; long traced stress runs and the seat test clean).  bf16: the earlier form.  (Under a relaxed acceptance,
# delta > 0, the drafts it takes may change with the drafter's K / V: not measured.)
KV_DTYPE = os.environ.get("GLM53_DFLASH2_KV", "bfp8")
_KV_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
CTX_BLOCK = 256  # the cache's positions per user round up to this (sdpa decode reads 128-row chunks)
# the paged sdpa decode's K chunk (rows; it splits a virtual user's context over cores by chunks)
SDPA_K_CHUNK = 128
# the head's matmul with the 1D multicast config up to these tile rows (the compact head's 56 / 112 rows at 1 / 2 users
# a row), ttnn's default past them (the head's rows are at most 4 tile rows compact, else 8 a tile row of slots a mesh
# row: 8, 16, ..; the 0/1 pick of the real rows and the 2D config from 10 tile rows, GLM53_DFLASH2_REPACK /
# GLM53_DFLASH2_HEAD_2D, -10 us at 128 users, were removed)
HEAD_1D_MAX = 4
# 1-2 users a row: the head gathers the slots' real rows only (row-major), not whole-tile blocks
COMPACT_HEAD = True


# GLM53_DFLASH2_SEL_FUSED (read at each call, so that the speed test's arms can set it): 1 / 0 on / off; unset, on
# at the drafter's (users a mesh row, drafts) of SEL_FUSED_SHAPES (128 users K = 3, with head_pick v3:
# head_pick.OWN_SHAPES): the selector's pairwise terms gather the two codebooks' rows themselves
# (dflash2_ops.sel_scores_ids, in place of two ttnn.embedding ops and their reshapes; the same bits)
SEL_FUSED_SHAPES = {(16, 3)}


def sel_fused_on(shape: tuple | None = None) -> bool:
    v = os.environ.get("GLM53_DFLASH2_SEL_FUSED", "")
    return v == "1" if v else shape in SEL_FUSED_SHAPES


# the drafter's own weights (the context's fc and k / v, the kernel projections, qkv, o_proj, gate | up, down; not the
# target's head and embedding it shares, nor the selector's bf16 projection) when TtDFlash2 is given no dtype:
# GLM53_DFLASH2_WEIGHT_DTYPE bf16 / bfp8 / bfp4, bfp8 by default.  bfp4 is a low-precision trial (with
# DECODE_MATMUL_FIDELITY=1 its decode_matmuls run LoFi, BFP4's 3 mantissa bits whole)
WEIGHT_DTYPE = os.environ.get("GLM53_DFLASH2_WEIGHT_DTYPE", "bfp8")
# the head's matmul fidelity in the drafter's own call (the target's head weight; the verify step's head keeps its
# own): GLM53_DFLASH2_HEAD_FIDELITY lofi / hifi2 / hifi3 / hifi4, hifi4 by default (a low-precision trial)
HEAD_FIDELITY = fidelity("GLM53_DFLASH2_HEAD_FIDELITY")
HEAD_MM = (
    HIFI4_FP32
    if HEAD_FIDELITY == ttnn.MathFidelity.HiFi4
    else ttnn.WormholeComputeKernelConfig(
        math_fidelity=HEAD_FIDELITY, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
)
_HEAD_MMS: dict = {}


def head_mm():
    """The drafter's head matmul's compute config, GLM53_DFLASH2_HEAD_FIDELITY read at each call (the speed test's
    arms set it per arm; HEAD_MM, its value at the module's load, as before by default; the drafts' head only, the
    target's tokens under the exact acceptance the same)."""
    f = fidelity("GLM53_DFLASH2_HEAD_FIDELITY")
    if f == HEAD_FIDELITY:
        return HEAD_MM
    if f not in _HEAD_MMS:
        _HEAD_MMS[f] = (
            HIFI4_FP32
            if f == ttnn.MathFidelity.HiFi4
            else ttnn.WormholeComputeKernelConfig(
                math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        )
    return _HEAD_MMS[f]


# the drafter's own ttnn matmuls on its weights (decode_linear and ttnn.linear; decode_matmul has
# DECODE_MATMUL_FIDELITY): GLM53_DFLASH2_MM_FIDELITY, hifi4 by default, apart from the target's GLM53_MM_FIDELITY (its
# low-precision forms leave the drafter as it is; common.WEIGHT_MM's values by default, a config decode_linear keeps)
DRAFTER_MM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=fidelity("GLM53_DFLASH2_MM_FIDELITY"), math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)  # fmt: skip
# GLM53_DFLASH2_PREFILL_MM (read at each call; "fixed", the default, or "0": ttnn's default as before): the drafter's
# matmuls that have no config of their own - `_lin` past 128 rows (a prefill's chunks: the context's fc, K = 36,864,
# and its K / V) and `_head_lin` past HEAD_1D_MAX tile rows (the candidates' head: 256 rows at 32 users) - with
# `fixed_prefill_config` (2D multicast, K in blocks of up to 16 tiles, the largest whose CBs fit 768 KB) in place of
# ttnn's default, whose blocks follow the L1 free for programs at the call (matmul_program_config.cpp:
# get_max_l1_space -> available_program_l1_capacity).  The default's fc at 1,024 rows took other K blocks at 1,150,272 B
# free (an L1 tensor the size of the GLM53_W1 landing held) than at 1,326,784 (without): its values apart on every chip,
# the context K / V rounded apart in ~8 % of the ring's rows, the drafts and from there the verified tokens apart (133 /
# 140 requests the same with the landing as without it; with the fixed config 140 / 140).  The values move against
# ttnn's default's (127 / 140 requests apart from ttnn's default without the landing).
DFLASH2_PREFILL_MM_DEFAULT = "fixed"
DRAFTER_KBLOCK_L1 = 768 * 1024


def fixed_prefill_config(mesh, m: int, k: int, n: int, w_tile: int, out_tile: int):
    """GLM53_DFLASH2_PREFILL_MM=fixed: a 2D multicast config for x [.., m, k] @ w [k, n] (m past 128 rows) with K in
    blocks of up to 16 tiles (a divisor of K's tiles, as large as the CBs fit DRAFTER_KBLOCK_L1: in0 and in1 blocks
    double-buffered and the output block), a fixed choice: the same values whatever L1 the call finds free; or None."""
    key = (m, k, n, w_tile, out_tile)
    if key not in _FIXED_MM:
        grid = mesh.compute_with_storage_grid_size()
        mt, kt, nt = -(-m // 32), k // 32, -(-n // 32)
        gy = max(y for y in range(1, min(WAVE_GRID_Y, grid.y) + 1) if mt % y == 0)
        gx = min(grid.x, nt)
        pm, pn = mt // gy, -(-nt // gx)
        cfg = None
        for bw in range(min(16, kt), 0, -1):
            if kt % bw or 2 * pm * bw * 2048 + 2 * bw * pn * w_tile + pm * pn * out_tile > DRAFTER_KBLOCK_L1:
                continue
            sub = max(
                ((h, w_) for h in range(1, 5) for w_ in range(1, 5) if h * w_ <= 4 and pm % h == 0 and pn % w_ == 0),
                key=lambda hw: (hw[0] * hw[1], hw[1]),
            )
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=bw,
                out_subblock_h=sub[0],
                out_subblock_w=sub[1],
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
            )
            break
        _FIXED_MM[key] = cfg
    return _FIXED_MM[key]


_FIXED_MM: dict = {}
_MM_SEEN: set = set()


def mm_log(mesh, what: str, x, w, fixed) -> None:
    """GLM53_DFLASH2_MM_LOG=1 (a diagnostic; read at each call): a drafter matmul past its own configs, each (shape,
    form, the L1 free for programs) once - "DRAFTER MM .. free L1 F (lowest allocated A, base B)": ttnn's default config
    picks its blocks against F (available_program_l1_capacity: the lowest occupied L1 less the CBs' base)."""
    if os.environ.get("GLM53_DFLASH2_MM_LOG") != "1":
        return
    try:
        table = ttnn.get_memory_view(mesh, ttnn.BufferType.L1).block_table
        base = min(int(b["address"]) for b in table)
        used = [int(b["address"]) for b in table if b.get("allocated") == "yes"]
        low = min(used) if used else None
    except Exception as e:  # (no view)
        base, low = None, f"? {str(e)[:60]}"
    free = (low - base) if isinstance(low, int) and base is not None else None
    key = (what, tuple(x.shape), tuple(w.shape), fixed is not None, free)
    if key in _MM_SEEN:
        return
    _MM_SEEN.add(key)
    form = f"fixed K block {fixed.in0_block_w}" if fixed is not None else "ttnn's default"
    lo = low if not isinstance(low, int) else hex(low)
    print(f"DRAFTER MM {what} x {tuple(x.shape)} w {tuple(w.shape)} {w.dtype}: {form}; free L1 {free} (lowest "
          f"allocated {lo}, base {hex(base) if base is not None else None})", flush=True)  # fmt: skip


# the drafter's K / V of its last positions only (bounds the drafter's memory at long contexts): each
# virtual user's cache RING = M positions (a multiple of 32), position p at p mod M (kv_write's ring; the paged sdpa
# decode's cache_position_modulo M), the attention over the last M positions up to the block's last (its
# sliding_window_size M: the block's B among them).  A position is written before any query whose window holds it
# reads it, and no two positions of one window share a slot (every write is at most the block's last position, the
# windows' ends only grow); a prefill call's tokens more than M before its last are not written (the generator: they
# would share slots within the call).  4,096 (the drafter trained on generations up to 4,096 tokens accepts more past
# ~32K with a window of 4-8K; its step -2 to -10 %; its K / V a user 17 MiB whatever the context).  A cache of at most
# M positions (max_ctx) holds every position, no ring (the same attention).  (Other M, or every position in a longer
# cache, were removed.)
RING = 4096


def ctx_heads_on() -> bool:
    """GLM53_DFLASH2_CTX_HEADS (read at each call; 1 by default): write_context's K in head-major rows (row h R + r,
    `dflash2_ops.heads_major`: a tile copy of kv's K columns) at R a multiple of 32, the cos / sin of its R rows and the
    norm weight a tile row a head (`dflash2_ops.rope` cos_period / w_div), in place of the slice, untilize, reshape and
    tilize and the cos / sin of 16 R rows.  Every row's values the same (the norm and RoPE are a row's), the caches the
    same bytes (kv_write's rows (1, 0, 2 R, R))."""
    return os.environ.get("GLM53_DFLASH2_CTX_HEADS", "1") == "1"


def rope_kv_on() -> bool:
    """GLM53_DFLASH2_ROPE_KV (read at each call; on by default, 0: rope, two bfp_casts and kv_write as before): the
    block's RoPE, its K / V cast to the BFP8_b caches and their writes in one op (`dflash2_ops.rope_kv`; the same values
    and rows); BFP8_b caches only."""
    return os.environ.get("GLM53_DFLASH2_ROPE_KV", "1") == "1"


def window() -> int:
    """GLM53_DFLASH2_WINDOW (read at each call; a study's knob): the drafter's attention over the last W positions only
    (the block's included: the paged sdpa decode's sliding_window_size, W positions up to its cur_pos s + B - 1), 0: the
    whole context (default).  The cache still holds every position."""
    return int(os.environ.get("GLM53_DFLASH2_WINDOW", "0"))


def load_weights(root: Path | str | None = None, synthetic: bool | None = None) -> tuple[DFlash2Config, dict, object]:
    """(config, {name: tensor}, mask_embedding): bf16 torch tensors from the checkpoint, or meta tensors of its shapes
    (synthetic: GLM53_SYNTHETIC_WEIGHTS=1; the device makes random values)."""
    from safetensors import safe_open

    root = drafter_dir(root)  # the argument, else $GLM53_DFLASH2_DRAFTER
    cfg = DFlash2Config.from_dict(json.loads((root / "config.json").read_text()))
    if synthetic is None:
        synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS", "0") == "1"
    w = {}
    with safe_open(str(root / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if synthetic:
                shape = f.get_slice(k).get_shape()
                w[k] = torch.empty(shape, dtype=torch.bfloat16, device="meta")
            else:
                w[k] = f.get_tensor(k).to(torch.bfloat16)
    if synthetic:
        mask = torch.empty(cfg.hidden, dtype=torch.bfloat16, device="meta")
    else:
        m = torch.load(root / "mask_embedding.pt", weights_only=True)
        mask = (m["embedding"] if isinstance(m, dict) else m).reshape(cfg.hidden).to(torch.bfloat16)
    return cfg, w, mask


@dataclass
class _Layer:
    in_norm: TtRmsNorm
    post_norm: TtRmsNorm
    kp: list  # [attention, mlp]: the kernel projections [4096, 1024], replicated
    conv_base: list  # [attention, mlp][side]: [1, taps, 32, 4096] fp32 (the conv op's)
    qkv: object
    # q's and k's norm weights [1, 1, 32, 128] (the rope op's: the row repeated); unused (the rope op takes qk_norm),
    # made so that the synthetic weights' seeds (one a tensor made, in order) stay those of the tensors after them
    q_norm: object
    k_norm: object
    qk_norm: object  # the rope op's two: [1, 1, 64, 128], q's 32 rows then k's
    o_proj: object
    gate_up: object
    down: object


class TtDFlash2:
    def __init__(
        self,
        mesh,
        cfg: DFlash2Config,
        w: dict,
        mask_embedding: torch.Tensor,
        users: int,
        max_ctx: int = 2048,
        block: int = 8,
        dtype=None,
        layers: int | None = None,
        head_weight=None,
        embedding=None,
        head_dtype=None,
    ) -> None:
        """users: a mesh row's users (U); max_ctx: positions a user (the block's last included); block B (B - 1
        drafts); dtype: the drafter's weights (WEIGHT_DTYPE's when None); layers: the first this many (tests);
        head_weight: the target's TtLmHead.mesh_weight (else made from w["lm_head.weight"] in that form, head_dtype or
        dtype); embedding: the target's TtEmbedding (else made from w["embed_tokens.weight"]).
        """
        if dtype is None:  # (read when made: a tasks test's form may set it with "@")
            name = os.environ.get("GLM53_DFLASH2_WEIGHT_DTYPE", WEIGHT_DTYPE)
            if name not in WEIGHT_DTYPES:
                raise ValueError(f"GLM53_DFLASH2_WEIGHT_DTYPE={name}: one of {sorted(WEIGHT_DTYPES)}")
            dtype = WEIGHT_DTYPES[name]
        head_dtype = dtype if head_dtype is None else head_dtype
        assert 2 <= block <= 8 and cfg.num_kv_heads == KV_HEADS_CHIP * COLS, (block, cfg.num_kv_heads)
        self.mesh, self.cfg, self.users, self.block = mesh, cfg, users, block
        self.slots = block - 1
        self.n_layers = cfg.num_layers if layers is None else layers
        self.heads_chip = cfg.num_heads // COLS  # 8
        self.group = cfg.num_heads // cfg.num_kv_heads  # q heads a KV head: 4
        assert self.group * block <= 32, (self.group, block)
        self.vu = users * KV_HEADS_CHIP  # virtual users a chip
        self.max_ctx = -(-max_ctx // CTX_BLOCK) * CTX_BLOCK
        # the ring (RING), or 0 for a cache no longer than it: every position, the same attention
        self.ring = RING if RING < self.max_ctx else 0
        self.pages = (self.ring or self.max_ctx) // 32
        self.layer_pages = self.vu * self.pages
        self.m = users * block  # the block's rows a mesh row
        self.r7 = users * self.slots  # the mask slots' rows a mesh row
        self.p7 = -(-self.r7 // 32) * 32  # .. in whole tiles (the head's gathered blocks)
        # (7 real rows of 32 at 1 user: the head's matmul 4.5x the work)
        self.compact = COMPACT_HEAD and self.p7 >= 2 * self.r7
        self.h_rows = self.r7 if self.compact else self.p7  # the slots' rows a mesh row in the head's gathered rows
        self.eps = cfg.eps
        self.scale = cfg.head_dim**-0.5
        rep = dict(dims=(None, None))
        h, hd = cfg.hidden, cfg.head_dim

        # -- constants
        # the RoPE's rows (sized for the drafter's memory at long contexts): one table of the first halves of cos and
        # sin (cos's columns 0-63, sin's 64-127: the two halves of each are the same values, cat(freqs, freqs))
        # [max_ctx, 128] fp32 on every chip, 512 B a position.  (Its other forms, cos and sin whole, 1 KB a position,
        # and none on the device with the host writing each call's rows, 0 B, the same values, were removed.)
        cos, sin = _rope_cos_sin(torch.arange(self.max_ctx), hd, cfg.rope_theta)
        half = hd // 2
        if not (torch.equal(cos[:, :half], cos[:, half:]) and torch.equal(sin[:, :half], sin[:, half:])):
            raise ValueError("the RoPE table's two halves differ")
        halves = torch.cat([cos[:, :half], sin[:, :half]], dim=1).contiguous()  # [max_ctx, 128]
        self.cos_tab = put(mesh, halves, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, **rep)
        self.sin_tab = None  # (the halves' one table: rope_rows's sin_tab None)
        del cos, sin
        bonus_rows = torch.zeros(self.m, users)
        bonus_rows[torch.arange(users) * block, torch.arange(users)] = 1.0
        self.bonus_rows = put(
            mesh, bonus_rows.bfloat16().view(1, 1, self.m, users), **rep
        )  # [M, U]: the bonus embeddings to rows u B
        mask_rows = torch.zeros(self.m, h, dtype=torch.bfloat16, device="meta" if mask_embedding.is_meta else "cpu")
        if not mask_embedding.is_meta:
            for u in range(users):
                mask_rows[u * block + 1 : (u + 1) * block] = mask_embedding
        self.mask_rows = put(mesh, mask_rows.view(1, 1, self.m, h), **rep)
        pick = torch.zeros(self.h_rows, self.m)
        for u in range(users):
            for s in range(self.slots):
                pick[u * self.slots + s, u * block + 1 + s] = 1.0
        self.pick = put(mesh, pick.bfloat16().view(1, 1, self.h_rows, self.m), **rep)
        spread = torch.zeros(self.r7 * 32, self.h_rows)
        spread[torch.arange(self.r7 * 32), torch.arange(self.r7 * 32) // 32] = 1.0
        self.spread = put(mesh, spread.bfloat16().view(1, 1, self.r7 * 32, self.h_rows), **rep)  # [U K 32, h_rows]
        self.tables = []
        for l in range(self.n_layers):
            t = l * self.layer_pages + torch.arange(self.vu)[:, None] * self.pages + torch.arange(self.pages)[None]
            self.tables.append(put(mesh, t.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **rep))
        self.conv_consts = ops.conv_constants(mesh, block, cfg.conv_group)
        self.sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=SDPA_K_CHUNK,
            exp_approx_mode=False,
        )

        # -- the caches
        total = self.n_layers * self.layer_pages + 1
        # (GLM53_DFLASH2_KV read again here: a tasks test's form may set it with "@", as the weights' dtype)
        self.kv_dtype = _KV_DTYPES[os.environ.get("GLM53_DFLASH2_KV", KV_DTYPE)]
        if self.kv_dtype == ttnn.bfloat16:
            self.k_cache = ttnn.zeros(
                [total, 1, 32, hd], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # fmt: skip
            self.v_cache = ttnn.zeros(
                [total, 1, 32, hd], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # fmt: skip
        else:  # (BFP zeros made on the device: ttnn.zeros builds them on the host)
            from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import zeros_bytes

            self.k_cache = zeros_bytes(mesh, [total, 1, 32, hd], self.kv_dtype, ttnn.TILE_LAYOUT)
            self.v_cache = zeros_bytes(mesh, [total, 1, 32, hd], self.kv_dtype, ttnn.TILE_LAYOUT)

        # -- context: fc (column parallel), hidden_norm, the 8 layers' k / v (each chip its 2 KV heads)
        self.fc = tp_weight(mesh, w["fc.weight"], "out", dtype)
        self.hidden_norm = TtRmsNorm(mesh, w["hidden_norm.weight"], self.eps)
        kv_rows = []
        for c in range(COLS):
            ks, vs = [], []
            for l in range(self.n_layers):
                p = f"layers.{l}.self_attn."
                for j in range(KV_HEADS_CHIP):
                    head = KV_HEADS_CHIP * c + j
                    ks.append(w[p + "k_proj.weight"][head * hd : (head + 1) * hd])
                    vs.append(w[p + "v_proj.weight"][head * hd : (head + 1) * hd])
            kv_rows.append(torch.cat(ks + vs))  # [16 x 128 k | 16 x 128 v, 4096]: block (l, j) at 2 l + j
        self.ctx_kv = tp_weight(mesh, torch.cat(kv_rows), "out", dtype)
        self.ctx_k_norm = torch.stack(
            [w[f"layers.{l}.self_attn.k_norm.weight"] for l in range(self.n_layers) for _ in range(KV_HEADS_CHIP)]
        )  # [L 2, 128]: row (l, j)
        self._ctx_norm_cache: dict = {}
        self._mm: dict = {}

        # -- the layers
        self.layers = []
        for l in range(self.n_layers):
            p = f"layers.{l}."
            a = p + "self_attn."
            qkv = []
            for c in range(COLS):
                hq = self.heads_chip
                kv = slice(KV_HEADS_CHIP * c * hd, KV_HEADS_CHIP * (c + 1) * hd)
                qkv += [
                    w[a + "q_proj.weight"][hq * c * hd : hq * (c + 1) * hd],
                    w[a + "k_proj.weight"][kv],
                    w[a + "v_proj.weight"][kv],
                ]
            inter = cfg.intermediate // COLS
            gu = []
            for c in range(COLS):
                gu += [
                    w[p + "mlp.gate_proj.weight"][c * inter : (c + 1) * inter],
                    w[p + "mlp.up_proj.weight"][c * inter : (c + 1) * inter],
                ]
            conv_bases = []
            for conv in ("attention_conv.", "mlp_conv."):
                b = w[p + conv + "base_kernel"]  # [2 sides, taps, hidden]
                conv_bases.append([ops.conv_base(mesh, b[s]) for s in range(2)])
            self.layers.append(
                _Layer(
                    in_norm=TtRmsNorm(mesh, w[p + "input_layernorm.weight"], self.eps),
                    post_norm=TtRmsNorm(mesh, w[p + "post_attention_layernorm.weight"], self.eps),
                    kp=[
                        put(mesh, w[p + conv + "kernel_projection.weight"].T.contiguous(), dtype=dtype, **rep)
                        for conv in ("attention_conv.", "mlp_conv.")
                    ],
                    conv_base=conv_bases,
                    qkv=tp_weight(mesh, torch.cat(qkv), "out", dtype),
                    q_norm=put(mesh, self._norm_rows(w[a + "q_norm.weight"]), **rep),
                    k_norm=put(mesh, self._norm_rows(w[a + "k_norm.weight"]), **rep),
                    qk_norm=put(
                        mesh,
                        torch.cat([self._norm_rows(w[a + n + "_norm.weight"]) for n in ("q", "k")], dim=2),
                        **rep,
                    ),
                    o_proj=tp_weight(mesh, w[a + "o_proj.weight"], "in", dtype),
                    gate_up=tp_weight(mesh, torch.cat(gu), "out", dtype),
                    down=tp_weight(mesh, w[p + "mlp.down_proj.weight"], "in", dtype),
                )
            )
        self.norm = TtRmsNorm(mesh, w["norm.weight"], self.eps)

        # -- the head (the target's mesh form), the selector, the embedding
        chips = ROWS * COLS
        vocab = cfg.vocab
        self.id_width = -(-vocab // (chips * 32)) * 32
        self.valid = [max(0, min(self.id_width, vocab - k * self.id_width)) for k in range(chips)]
        if head_weight is None:
            hw = w["lm_head.weight"]
            if hw.is_meta:
                padded = torch.empty(chips * self.id_width, h, dtype=hw.dtype, device="meta")
            else:
                padded = torch.zeros(chips * self.id_width, h, dtype=hw.dtype)
                padded[:vocab] = hw
            per_chip = padded.T.contiguous().view(h, ROWS, COLS * self.id_width).permute(1, 0, 2)
            head_weight = put(mesh, per_chip.unsqueeze(1), dims=(0, 3), dtype=head_dtype)
        self.head = head_weight
        self.hp = put(mesh, w["candidate_selector.hidden_projection.weight"].T.contiguous(), **rep)  # bf16
        cb = dict(layout=ttnn.ROW_MAJOR_LAYOUT, **rep)
        self.pred_cb = put(mesh, w["candidate_selector.predecessor_codebook"], **cb)
        self.succ_cb = put(mesh, w["candidate_selector.successor_codebook"], **cb)
        if embedding is None:
            table = w["embed_tokens.weight"]
            self.embed_table = put(mesh, table[None, None], dims=(None, 3), layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            self.embed_table = embedding.table

    # -- small helpers ---------------------------------------------------------------------------------------------
    @staticmethod
    def _norm_rows(weight: torch.Tensor) -> torch.Tensor:
        """A head norm's weight [128] as the rope op's [1, 1, 32, 128]."""
        row = weight.reshape(1, 1, 1, -1)
        return row.expand(1, 1, 32, row.shape[-1]).contiguous()

    def _lin(self, x, w, big_k: bool = False, **kw):
        """A weight's matmul on the block's rows: decode_linear up to 64 rows (decode_matmul or its 1D config; with
        big_k its 1D config only: decode_matmul keeps x's K whole in each core's L1); to 128 rows (128 users' 128 a
        mesh row) the 1D multicast config with every row on each core (ttnn's default ran the drafter's projections
        at 128 rows 3-5x slower); past it (prefill chunks) ttnn's default."""
        m = x.shape[-2]
        if m <= 64:
            tiles = (w.shape[-1] // 32) * (w.shape[-2] // 32)
            if not big_k and tiles >= DM_MIN_TILES and x.is_sharded() is False:
                return ttnn.experimental.deepseek_prefill.decode_matmul(
                    x, w, output_dtype=kw.get("dtype"), memory_config=kw.get("memory_config")
                )
            extra = dict(program_config=None) if big_k else {}
            return decode_linear(self.mesh, x, w, compute_kernel_config=DRAFTER_MM, **extra, **kw)
        if m > 128:
            if os.environ.get("GLM53_DFLASH2_PREFILL_MM", DFLASH2_PREFILL_MM_DEFAULT) == "fixed":
                w_tile = 1088 if w.dtype == ttnn.bfloat8_b else (576 if w.dtype == ttnn.bfloat4_b else 2048)
                out_tile = 4096 if kw.get("dtype") == ttnn.float32 else 2048
                cfg = fixed_prefill_config(self.mesh, m, w.shape[-2], w.shape[-1], w_tile, out_tile)
                if cfg is not None:
                    mm_log(self.mesh, "lin", x, w, cfg)
                    kw.pop("program_config", None)
                    return ttnn.linear(x, w, program_config=cfg, compute_kernel_config=DRAFTER_MM, **kw)
            mm_log(self.mesh, "lin", x, w, None)
            return ttnn.linear(x, w, compute_kernel_config=DRAFTER_MM, **kw)
        n, k, mt = w.shape[-1] // 32, w.shape[-2] // 32, -(-m // 32)
        key = (n, k, mt)
        if key not in self._mm:
            grid = self.mesh.compute_with_storage_grid_size()
            per = -(-n // (grid.x * grid.y))
            block = 8
            while k % block:
                block //= 2
            self._mm[key] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=max(v for v in (4, 2, 1) if per % v == 0),
                out_block_h=mt,
                out_block_w=per,
                per_core_M=mt,
                per_core_N=per,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        kw.pop("program_config", None)
        return ttnn.linear(x, w, program_config=self._mm[key], compute_kernel_config=DRAFTER_MM, **kw)

    def _norm(self, norm: TtRmsNorm, x):
        """TtRmsNorm, width-sharded also past its 64 rows up to 128 (its interleaved form ran 128 rows on 4 cores:
        67 us)."""
        m = x.shape[-2]
        if m <= 64 or m > 128 or not norm.shard_cores:  # (past 128 rows, prefill chunks: interleaved, many tile rows)
            return norm(x)
        memory, program = norm._shard_configs(norm.shard_cores, -(-m // 32))
        sharded = ttnn.to_memory_config(x, memory)
        y = ttnn.rms_norm(
            sharded, weight=norm.weight, epsilon=norm.eps, compute_kernel_config=HIFI4_FP32,
            program_config=program, memory_config=memory,
        )  # fmt: skip
        ttnn.deallocate(sharded)
        out = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y)
        return out

    def _norm_rope(self, x, weight, cos, sin):
        """bf16(weight * bf16(rms_norm(x))) and its RoPE (fp32, rounded to bf16), per row of 128."""
        n = ttnn.rms_norm(x, epsilon=self.eps, compute_kernel_config=HIFI4_FP32)
        out = ops.rope(self.mesh, n, weight, cos, sin)
        ttnn.deallocate(n)
        return out

    def _rope_rows(self, idx, rows: int, div: int, mod: int = 0, cur=None):
        """ops.rope_rows from the device's table of the halves."""
        return ops.rope_rows(self.mesh, idx, self.cos_tab, self.sin_tab, rows, div, mod=mod, cur=cur)

    def _ctx_norm_heads(self):
        """[1, 1, 16 x 32, 128] bf16: tile row (l, j) the k_norm weight of layer l, KV head j, over its 32 rows (the
        head-major K of GLM53_DFLASH2_CTX_HEADS)."""
        if "heads" not in self._ctx_norm_cache:
            rows = self.ctx_k_norm.repeat_interleave(32, dim=0)
            self._ctx_norm_cache["heads"] = put(self.mesh, rows.reshape(1, 1, -1, rows.shape[-1]), dims=(None, None))
        return self._ctx_norm_cache["heads"]

    def _ctx_norm_rows(self, tokens: int):
        """[1, 1, R 16, 128] bf16: row (r, l, j) the k_norm weight of layer l (R tokens)."""
        if tokens not in self._ctx_norm_cache:
            rows = self.ctx_k_norm.repeat(tokens, 1)
            self._ctx_norm_cache[tokens] = put(self.mesh, rows.reshape(1, 1, -1, rows.shape[-1]), dims=(None, None))
        return self._ctx_norm_cache[tokens]

    # -- the context ------------------------------------------------------------------------------------------------
    def write_context(self, taps, positions, users, gathered=None) -> None:
        """taps [1, 1, R, 9 x 4096] bf16 (or the 9 [1, 1, R, 4096]: the target's layers in config order, each the mean
        of its 4 streams; a mesh row's own tokens, the same on its 4 chips), positions and users [1, R] int32 row-major
        (each token's position, -1: not written, and its user on the mesh row) -> the drafter's context K / V at those
        positions, every layer.

        gathered (the sequence-parallel prefill, a row's tokens some other row's request's): (positions, users)
        [1, 8 R] int32 row-major a row, every row's R tokens in row order, the position where this row's request owns
        the token (else -1) and its user here: the K / V (each row's own, `positions` for RoPE) are all-gathered over
        the rows and each row writes the ones it owns (`users` unused)."""
        if isinstance(taps, (list, tuple)):
            taps = ttnn.concat(list(taps), dim=-1)
            own = True
        else:
            own = False
        r = taps.shape[-2]
        assert (r * 16) % 32 == 0, r
        if fc_ksplit.on() and fc_ksplit.takes(self.mesh, taps, self.fc):
            # (GLM53_DFLASH2_FC_KSPLIT=1: K over 3 cores a column, the same values as `_lin`'s 1D matmul; up to 64 rows)
            part = fc_ksplit.fc_ksplit(self.mesh, taps, self.fc, fidelity=DRAFTER_MM.math_fidelity)
        else:
            part = self._lin(taps, self.fc, big_k=True)  # [1, 1, R, 1024]: chip c's features (K = 36,864)
        if own:
            ttnn.deallocate(taps)
        c = ttnn.all_gather(part, dim=3, cluster_axis=TP_AXIS)
        ttnn.deallocate(part)
        cn = self._norm(self.hidden_norm, c)
        ttnn.deallocate(c)
        kv = self._lin(cn, self.ctx_kv)  # [1, 1, R, 2048 k | 2048 v]
        ttnn.deallocate(cn)
        hd = self.cfg.head_dim
        heads = self.n_layers * KV_HEADS_CHIP
        if ctx_heads_on() and gathered is None and r % 32 == 0:
            # (GLM53_DFLASH2_CTX_HEADS=1: K in head-major rows, row h R + r: a tile copy in place of the slice, the
            # untilize, the reshape and the tilize; the cos / sin of R rows; the same values a row, the same bytes in
            # the caches)
            kt = ops.heads_major(self.mesh, kv, heads)
            cos, sin = self._rope_rows(positions, r, div=1)
            n = ttnn.rms_norm(kt, epsilon=self.eps, compute_kernel_config=HIFI4_FP32)
            kr = ops.rope(self.mesh, n, self._ctx_norm_heads(), cos, sin, cos_period=r // 32, w_div=r // 32)
            for t in (n, kt, cos, sin):
                ttnn.deallocate(t)
            kr, kv = self._kv_store(kr), self._kv_store(kv)
            writes = [
                (kr, self.k_cache, (1, 0, KV_HEADS_CHIP * r, r), (0, 0, 0)),  # k rows (l, j, r)
                (kv, self.v_cache, (1, 0, 0, 0), (heads, KV_HEADS_CHIP, 1)),  # v columns (l, j) of row r
            ]
            ops.kv_write(self.mesh, writes, positions, users, ring=bool(self.ring), tokens=r, block=1,
                         layers=self.n_layers, heads=KV_HEADS_CHIP, n_users=self.users, pages=self.pages,
                         layer_pages=self.layer_pages)  # fmt: skip
            ttnn.deallocate(kr)
            ttnn.deallocate(kv)
            return
        k = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, r, heads * hd])
        krm = ttnn.to_layout(k, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(k)
        kflat = ttnn.reshape(krm, [1, 1, r * heads, hd])
        kt = ttnn.tilize_with_zero_padding(kflat)
        ttnn.deallocate(krm)
        cos, sin = self._rope_rows(positions, r * heads, div=heads)
        kr = self._norm_rope(kt, self._ctx_norm_rows(r), cos, sin)
        for t in (kt, cos, sin):
            ttnn.deallocate(t)
        if gathered is not None:  # (each row's K / V, in row order: token s R + i is row s's token i)
            positions, users = gathered
            kr_all = ttnn.all_gather(kr, dim=2, cluster_axis=ROW_AXIS)
            kv_all = ttnn.all_gather(kv, dim=2, cluster_axis=ROW_AXIS)
            for t in (kr, kv):
                ttnn.deallocate(t)
            kr, kv, r = kr_all, kv_all, r * ROWS
        kr, kv = self._kv_store(kr), self._kv_store(kv)
        common = dict(
            tokens=r, block=1, layers=self.n_layers, heads=KV_HEADS_CHIP, n_users=self.users, pages=self.pages,
            layer_pages=self.layer_pages,
        )  # fmt: skip
        writes = [
            (kr, self.k_cache, (heads, 0, KV_HEADS_CHIP, 1), (0, 0, 0)),  # k rows (r, l, j)
            (kv, self.v_cache, (1, 0, 0, 0), (heads, KV_HEADS_CHIP, 1)),  # v columns (l, j) of row r
        ]
        ops.kv_write(self.mesh, writes, positions, users, ring=bool(self.ring), **common)
        ttnn.deallocate(kr)
        ttnn.deallocate(kv)

    def _kv_store(self, t):
        """K / V rows to write (tiled bf16, deallocated) as the caches take them: BFP (KV_DTYPE, dsa_indexer.bfp_cast),
        else bf16 (KV_ROUND: rounded there and back)."""
        if self.kv_dtype == ttnn.bfloat16:
            return round_kv(t, KV_ROUND)
        from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import bfp_cast

        low = bfp_cast(self.mesh, t, self.kv_dtype)
        ttnn.deallocate(t)
        return low

    # -- the block --------------------------------------------------------------------------------------------------
    def _attention(self, i: int, layer: _Layer, h, starts, cos, sin, cur):
        hd = self.cfg.head_dim
        hq = self.heads_chip
        qkv = self._lin(h, layer.qkv)  # [1, 1, M, 1024 + 256 + 256]
        # q and k one tensor: one norm, one rope op
        qk = ops.heads_split(
            self.mesh, qkv, self.users, self.block, self.group, KV_HEADS_CHIP, hq * hd // 128, combined=True
        )
        n = ttnn.rms_norm(qk, epsilon=self.eps, compute_kernel_config=HIFI4_FP32)
        ttnn.deallocate(qk)
        if rope_kv_on() and self.kv_dtype == ttnn.bfloat8_b:
            q = ops.rope_kv(
                self.mesh, n, layer.qk_norm, cos, sin, self.vu, qkv, (hq + KV_HEADS_CHIP) * hd // 128, self.k_cache,
                self.v_cache, starts, block=self.block, heads=KV_HEADS_CHIP, n_users=self.users, pages=self.pages,
                page_base=i * self.layer_pages, ring=bool(self.ring),
            )  # fmt: skip
            ttnn.deallocate(n)
            ttnn.deallocate(qkv)
            return self._attend(q, i, cur, layer)
        q, k = ops.rope(self.mesh, n, layer.qk_norm, cos, sin, halves=self.vu)
        ttnn.deallocate(n)
        k, qkv = self._kv_store(k), self._kv_store(qkv)  # (q is its own tensor by now)
        common = dict(
            tokens=self.m, block=self.block, layers=1, heads=KV_HEADS_CHIP, n_users=self.users, pages=self.pages,
            layer_pages=self.layer_pages, page_base=i * self.layer_pages,
        )  # fmt: skip
        v0 = (hq + KV_HEADS_CHIP) * hd // 128
        writes = [
            (k, self.k_cache, (2 * 32, 1, 0, 32), (0, 0, 0)),  # k in the virtual users' tiles
            (qkv, self.v_cache, (self.block, 1, 0, 0), (v0, 0, 1)),  # v straight from the projection
        ]
        ops.kv_write(self.mesh, writes, starts, None, ring=bool(self.ring), **common)
        ttnn.deallocate(k)
        ttnn.deallocate(qkv)
        return self._attend(q, i, cur, layer)

    def _attend(self, q, i: int, cur, layer):
        """`_attention`'s paged SDPA (q deallocated), heads_merge and o_proj."""
        win = self.ring or window()
        ring = {"cache_position_modulo": self.ring} if self.ring else {}
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            self.tables[i],
            is_causal=True,
            cur_pos_tensor=cur,
            scale=self.scale,
            program_config=self.sdpa_program,
            **({"sliding_window_size": win} if win else {}),
            **ring,
        )  # [1, 2 U, 32, 128]
        ttnn.deallocate(q)
        flat = ops.heads_merge(self.mesh, attn, self.users, self.block, self.group, KV_HEADS_CHIP)
        ttnn.deallocate(attn)
        return self._row_sum(flat, layer.o_proj)

    def _row_sum(self, x, w):
        """x @ w (a row-parallel weight) summed over the row's 4 chips; x deallocated.  With AR_L1 and the all_reduce_async
        path, the matmul writes into its input shards and its output stays sharded in L1 (the conv op reads it there).
        """
        out_cfg = ar_input_config(self.mesh, x.shape[-2], w.shape[-1]) if AR_L1 else None
        kw = {} if out_cfg is None else {"memory_config": out_cfg}
        if out_cfg is not None and ar_input_dtype(x.shape[-2]) is not None:  # (GLM53_AR_BFP8)
            kw["dtype"] = ar_input_dtype(x.shape[-2])
        partial = self._lin(x, w, **kw)
        ttnn.deallocate(x)
        with ar_keep_l1(AR_L1 and out_cfg is not None):
            out = tp_all_reduce(self.mesh, partial)
        ttnn.deallocate(partial)
        return out

    def _mlp(self, layer: _Layer, h):
        inter = self.cfg.intermediate // COLS
        gu = self._lin(h, layer.gate_up)
        m = h.shape[-2]
        if m <= 128:  # (one op: `dflash2_ops.silu_mul`; the same values)
            act = ops.silu_mul(self.mesh, gu, inter)
            ttnn.deallocate(gu)
            return self._row_sum(act, layer.down)
        g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, m, inter])
        u = ttnn.slice(gu, [0, 0, 0, inter], [1, 1, m, 2 * inter])
        ttnn.deallocate(gu)
        act = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)])
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        return self._row_sum(act, layer.down)

    def _conv(self, x, delta, base, side: int):
        """The grouped dynamic 2-tap convolution over the block (`dflash2_ops.conv`): x [1, 1, M, 4096] bf16, delta
        [1, 1, M, 1024] (its sublayer's kernel projection), base[side] (the conv op's) -> bf16."""
        return ops.conv(self.mesh, x, delta, base[side], self.conv_consts, side)

    def _sublayer(self, x, norm, kp, base, body):
        tiles = (kp.shape[-1] // 32) * (kp.shape[-2] // 32)
        cfgs = norm.sharded_configs(x.shape[-2]) if norm_shard_on() else None
        shard = (
            cfgs is not None and conv_fused_on(kp) and x.shape[-2] <= 64 and tiles >= DM_MIN_TILES
        )  # (GLM53_DFLASH2_NORM_SHARD: the form below only)
        if shard:
            memory, program = cfgs
            xs = x if x.is_sharded() and x.memory_config() == memory else ttnn.to_memory_config(x, memory)
            h = norm.on_shard(xs, memory, program)
            if xs is not x:
                ttnn.deallocate(xs)
        else:
            h = self._norm(norm, x) if not x.is_sharded() else self._norm_any(norm, x)
        if conv_fused_on(kp) and h.shape[-2] <= 64 and tiles >= DM_MIN_TILES and (shard or not h.is_sharded()):
            # (`_lin`'s decode_matmul for kp, with side 0's convolution of h on its free cores)
            spec = ttnn.TensorSpec(
                ttnn.Shape(list(h.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
            )
            hc = ttnn.allocate_tensor_on_device(spec, self.mesh)
            e, s = self.conv_consts
            delta = ttnn.experimental.deepseek_prefill.decode_matmul(
                h, kp, conv_side=0, conv_base=base[0], conv_e=e, conv_s=s, conv_out=hc
            )  # [1, 1, M, 1024]
        else:
            delta = self._lin(h, kp)  # [1, 1, M, 1024]
            hc = self._conv(h, delta, base, 0)
        ttnn.deallocate(h)
        y = body(hc)
        ttnn.deallocate(hc)
        yc = self._conv(y, delta, base, 1)
        ttnn.deallocate(y)
        ttnn.deallocate(delta)
        out = ttnn.add(x, yc, memory_config=cfgs[0]) if shard else ttnn.add(x, yc)
        ttnn.deallocate(yc)
        return out

    def _norm_any(self, norm: TtRmsNorm, x):
        """`_norm` of x in DRAM, or of x already in the norm's sharded form (GLM53_DFLASH2_NORM_SHARD's residual
        stream): the norm on its shards, its output moved to DRAM as `TtRmsNorm.__call__` leaves it."""
        if not x.is_sharded():
            return self._norm(norm, x)
        cfgs = norm.sharded_configs(x.shape[-2])
        if cfgs is None or x.memory_config() != cfgs[0]:  # (another sharded form: through DRAM as before)
            moved = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            out = self._norm(norm, moved)
            ttnn.deallocate(moved)
            return out
        memory, program = cfgs
        y = norm.on_shard(x, memory, program)
        out = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y)
        return out

    def hidden(self, bonus, starts):
        """bonus [1, U] uint32, starts [1, U] int32 (row-major, a mesh row's users) -> the block's final normed hidden
        states [1, 1, U B, 4096] bf16 (rows (u, pos)); writes the block's K / V into the caches."""
        # row-major through the gather, tiled after on many cores (the embedding's own tiled output was a 1-core tilize:
        # 36 us)
        quarter = ttnn.embedding(bonus, self.embed_table, layout=ttnn.ROW_MAJOR_LAYOUT)
        quarter = ttnn.reshape(quarter, [1, 1, quarter.shape[-2], quarter.shape[-1]])
        gathered = ttnn.all_gather(quarter, dim=3, cluster_axis=TP_AXIS)  # [1, 1, U, 4096] row-major
        ttnn.deallocate(quarter)
        emb = ttnn.tilize_with_zero_padding(gathered)
        ttnn.deallocate(gathered)
        placed = ttnn.matmul(self.bonus_rows, emb, dtype=ttnn.bfloat16, compute_kernel_config=HIFI4_FP32)
        ttnn.deallocate(emb)
        x = ttnn.add(placed, self.mask_rows)
        ttnn.deallocate(placed)
        cos, sin, cur = self._rope_rows(
            starts, self.vu * 32, div=2 * 32, mod=self.block, cur=(self.vu, KV_HEADS_CHIP, self.block - 1)
        )
        cos4 = ttnn.reshape(cos, [1, self.vu, 32, self.cfg.head_dim])
        sin4 = ttnn.reshape(sin, [1, self.vu, 32, self.cfg.head_dim])
        for i, layer in enumerate(self.layers):
            bases = layer.conv_base
            x1 = self._sublayer(
                x, layer.in_norm, layer.kp[0], bases[0],
                lambda h, i=i, layer=layer: self._attention(i, layer, h, starts, cos4, sin4, cur),
            )  # fmt: skip
            ttnn.deallocate(x)
            x = self._sublayer(x1, layer.post_norm, layer.kp[1], bases[1], lambda h, layer=layer: self._mlp(layer, h))
            ttnn.deallocate(x1)
        for t in (cos, sin, cur):
            ttnn.deallocate(t)
        out = self._norm_any(self.norm, x)
        ttnn.deallocate(x)
        return out

    def _head_lin(self, x):
        """The mesh head's matmul [n, 4096] @ [4096, 4,864] -> fp32: up to HEAD_1D_MAX tile rows the 1D multicast
        config (every row on each core, the columns over the cores; ttnn's default took 166 us at 256 rows),
        past it ttnn's default."""
        mt = -(-x.shape[-2] // 32)
        n, k = self.head.shape[-1] // 32, self.head.shape[-2] // 32
        if mt > HEAD_1D_MAX:
            if os.environ.get("GLM53_DFLASH2_PREFILL_MM", DFLASH2_PREFILL_MM_DEFAULT) == "fixed":  # (as `_lin`'s)
                dt = self.head.dtype
                w_tile = 1088 if dt == ttnn.bfloat8_b else (576 if dt == ttnn.bfloat4_b else 2048)
                k_, n_ = self.head.shape[-2], self.head.shape[-1]
                cfg = fixed_prefill_config(self.mesh, x.shape[-2], k_, n_, w_tile, 4096)
                if cfg is not None:
                    mm_log(self.mesh, "head", x, self.head, cfg)
                    return ttnn.linear(
                        x, self.head, dtype=ttnn.float32, program_config=cfg, compute_kernel_config=head_mm()
                    )
            mm_log(self.mesh, "head", x, self.head, None)
            from .head import head_mm_config  # (GLM53_HEAD_MM_2D: the same blocks as ttnn's default, the same bits)

            cfg = head_mm_config(self.mesh, x.shape[-2], self.head.shape[-1])
            kw = {} if cfg is None else {"program_config": cfg}
            return ttnn.linear(x, self.head, dtype=ttnn.float32, compute_kernel_config=head_mm(), **kw)
        key = ("head", n, k, mt)
        if key not in self._mm:
            grid = self.mesh.compute_with_storage_grid_size()
            per = -(-n // (grid.x * grid.y))
            self._mm[key] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=8 if k % 8 == 0 else 4,
                out_subblock_h=1,
                out_subblock_w=max(v for v in (4, 2, 1) if per % v == 0 and v <= 4),
                out_block_h=mt,
                out_block_w=per,
                per_core_M=mt,
                per_core_N=per,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        return ttnn.linear(
            x, self.head, dtype=ttnn.float32, program_config=self._mm[key], compute_kernel_config=head_mm()
        )

    def candidates(self, hidden, bonus):
        """The mask slots' candidates: the head's top 16 over the mesh -> (sel, cand, pred) (`dflash2_ops.top16_merge`)
        and the slots' hidden states [1, 1, h_rows, 4096] (U K rows; padded to whole tiles a mesh row unless compact:
        the gather moves tiles)."""
        h7 = ttnn.matmul(self.pick, hidden, dtype=ttnn.bfloat16, compute_kernel_config=HIFI4_FP32)
        if self.compact:  # the real rows only, row-major through the gather
            flat = ttnn.to_layout(h7, ttnn.ROW_MAJOR_LAYOUT)
            gathered = ttnn.all_gather(flat, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8 U K, 4096]
            ttnn.deallocate(flat)
            x_all = ttnn.tilize_with_zero_padding(gathered)
            ttnn.deallocate(gathered)
        else:
            x_all = ttnn.all_gather(h7, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8 P, 4096]: row r's at r P
        logits = self._head_lin(x_all)
        ttnn.deallocate(x_all)
        n = ROWS * self.h_rows
        lists = ops.top16(self.mesh, logits, n, self.valid, self.id_width, self.h_rows, self.r7)  # [1, 1, n, 32]
        ttnn.deallocate(logits)
        g1 = ttnn.all_gather(lists, dim=1, cluster_axis=TP_AXIS)
        ttnn.deallocate(lists)
        g2 = ttnn.all_gather(g1, dim=0, cluster_axis=ROW_AXIS)  # [8, 4, n, 32]
        ttnn.deallocate(g1)
        block = -(-n // 32) * 32
        sel, cand, pred = ops.top16_merge(self.mesh, g2, bonus, self.users, self.slots, self.h_rows, block)
        ttnn.deallocate(g2)
        return sel, cand, pred, h7

    def scores(self, h7, cand, pred):
        """The selector's pairwise terms [1, U K, 32, 32] fp32: row p (predecessor), column c (candidate)
        <pred[p] * proj, succ[c]> (16 x 16 real)."""
        r7 = self.r7
        # [1, 1, P, 256] (the 1D multicast config: ttnn's default read x and hp on 32 cores, 52 / 91 us at 32 / 128
        # users)
        proj = self._lin(h7, self.hp, dtype=ttnn.bfloat16)
        rank = proj.shape[-1]
        # each (user, slot)'s proj on its 32 rows by a 0/1 matmul (exact): [1, 1, U K 32, 256]
        p4 = ttnn.matmul(self.spread, proj, dtype=ttnn.bfloat16, compute_kernel_config=HIFI4_FP32)
        ttnn.deallocate(proj)
        # (the codebook rows gathered in the op: the two embeddings' 20 / 30 us at 32 / 128 users)
        if sel_fused_on((self.users, self.slots)):
            s = ops.sel_scores_ids(self.mesh, pred, p4, cand, self.pred_cb, self.succ_cb, r7)
            ttnn.deallocate(p4)
            return s
        rows = []
        for ids, table in ((pred, self.pred_cb), (cand, self.succ_cb)):
            e = ttnn.embedding(ids, table, layout=ttnn.TILE_LAYOUT)  # [1, U K 32, 256]
            rows.append(ttnn.reshape(e, [1, 1, r7 * 32, rank]))
        pr, su = rows
        s = ops.sel_scores(self.mesh, pr, p4, su, r7)  # (the product, its hi / lo split and the two matmuls)
        for t in (pr, p4, su):
            ttnn.deallocate(t)
        return s

    def draft(self, bonus, starts, debug: bool = False):
        """bonus [1, U] uint32 (each user's bonus token t[s]), starts [1, U] int32 (its position s), a mesh row's users,
        the same on the row's 4 chips -> drafts [U, 8] uint32 row-major (t[s], the B - 1 drafts, zeros)."""
        hidden = self.hidden(bonus, starts)
        sel, cand, pred, h7 = self.candidates(hidden, bonus)
        s = self.scores(h7, cand, pred)
        drafts = ops.walk(self.mesh, s, sel, bonus, self.users, self.slots)
        if debug:
            return drafts, dict(hidden=hidden, sel=sel, cand=cand, pred=pred, h7=h7, scores=s)
        for t in (hidden, sel, cand, pred, h7, s):
            ttnn.deallocate(t)
        return drafts
