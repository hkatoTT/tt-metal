"""Embedding, final norm and LM head on the 8x4 mesh (v0).

The embedding table is split by hidden feature over a row's 4 chips (row-major, as ttnn.embedding takes it) and the
looked-up quarters are gathered over the row.  The LM head is split by vocabulary over the row's chips; the logits
are gathered over the row.  Norm weights are replicated in the [1, 1, D/32, 32] row-major form ttnn.rms_norm takes.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.glm53_flash.ttnn.common import COLS, HIFI4_FP32, ROW_AXIS, ROWS, TP_AXIS, put, tp_weight

# the greedy decode's head over the whole mesh (`TtLmHead.greedy_mesh`): each chip a 1/32 of the vocabulary for every
# row's tokens, rather than each row's 4 chips a quarter for its own (the 8 rows read the same 168 MB a step, BFP8;
# that form was removed); and the pick over the mesh (the 32 chips' (value, id) pairs to the top's id, on every chip)
# in one op (`head_pick.head_pick`: one-shot over the fabric, the mesh row's then the column's), in place of
# all-gathering the values and the ids over the row and the column (4 all_broadcast + concat pairs, ~140 us a head) and
# the max, compare and min over the 32.  The same ids (exact compares; a tie keeps the smallest id): MTP's verify step
# and the plain step at 8 users -0.25 ms, at 32 users -0.1 / -0.15 ms; 16,000 traced steps at 8 and 32 users without
# a hang.  0: those ops
HEAD_PICK = True
# ... up to this many rows of tokens over the mesh (the op's limit).  64 at first: at 32 a row the op's folds of
# 256 rows took longer than the all-gathers, the RISC-Vs comparing floats by library calls; the folds now
# compare the floats' bits as ordered integers.  Read at each call (tests switch it)
HEAD_PICK_MAX_ROWS = 256
# ... and with the pick's v2 (head_pick.py; the default, GLM53_HEAD_PICK_V2=0 for v1 and the limit above): up to this
# many (128 users' verify step: 512 rows, in place of the all_broadcast + concat pairs and the max / compare / min,
# alone 172.1 -> 90.2 us; 64 / 256 rows v1 35.9 / 109.4 -> 23.3 / 53.4 us)
HEAD_PICK_V2_MAX_ROWS = 1024


def head_pick_max_rows() -> int:
    """The pick op's rows: HEAD_PICK_V2_MAX_ROWS with the v2 pick (the default), else HEAD_PICK_MAX_ROWS."""
    from .head_pick import v2_on

    return HEAD_PICK_V2_MAX_ROWS if v2_on() else HEAD_PICK_MAX_ROWS


# The mesh head's matmul [M, 4096] @ [4096, 4,864 a chip] (fp32 out, HiFi4, fp32 accumulation, the packer's L1
# accumulation) at 8 tile rows and more (32 / 128 users' verify step: 256 / 512 rows; the drafter's candidates' head
# past its 1D form, `dflash2._head_lin`): ttnn's default for this narrow shape is the 1D in0 multicast with K blocks of
# 2 tiles, per core all M rows x 2 column tiles (76 of the 120 cores: 166 / 291 us at 256 / 512 rows); past 16
# tile rows (not narrow) its 2D path, left as it is.
# GLM53_HEAD_MM_2D=<per_core_N> (read at each call; 13 by default, 0: ttnn's default): a fixed 2D multicast config -
# the M tile rows over 8 grid rows, per_core_N column tiles a core over ceil(152 / per_core_N) grid columns, K blocks
# of 2 tiles as ttnn's default (the packer's L1 accumulation adds a block's sum at a time: the same blocks, the same
# sums; each output tile's K order is the core's own either way), so the same bits (the fp32 logits of 32 chips,
# tests/test_head_device.py test_head_path_v2).  Alone at 256 / 512 rows 175.2 / 308.3 -> 147.6 / 249.3 us
# (per_core_N 14: 147.9 / 257.8).
HEAD_MM_2D_MIN_TILES = 8
HEAD_MM_2D_DEFAULT = "13"
_HEAD_MM: dict = {}
# GLM53_HEAD_MM_1D=<per_core_N> (read at each call; 0, ttnn's default, by default): below 8 tile rows (8 users' verify
# step: 64 rows), ttnn's default's 1D in0 multicast form with its K blocks of 2 tiles, per core per_core_N column tiles
# in place of its 2 (the same blocks, the same sums: each output tile's K order is the core's own).  At 64 rows 4 (38
# cores): 95.5 -> 82.8 us alone, the fp32 logits of 32 chips bit for bit (3: 84.8, 5: 92.5, 6: 105.2).  The shapes
# it is taken at: HEAD_MM_1D_TILES.
HEAD_MM_1D_TILES = (1, 2, 4)
# Value-changing switches (read at each call, off by default): GLM53_HEAD_MM_KB=<K tiles a block> for the configs
# above (the 2D one, and the 1D one with GLM53_HEAD_MM_1D): other blocks, other sums (the packer's L1 accumulation adds
# a block's sum at a time); 16: the verify head alone at 256 / 512 rows 131.6 / 229.9 -> 108.4 / 204.1 us.
# GLM53_HEAD_MM_FIDELITY (lofi / hifi2 / hifi3 / hifi4): the verify head's matmul's fidelity (on the BFP8 weight);
# hifi2 at 64 / 256 / 512 rows 95.5 / 131.6 / 229.8 -> 94.8 / 127.0 / 163.9 us (only the 512 rows' are bound by
# HiFi4's compute).


def head_mm_kb() -> int:
    return int(os.environ.get("GLM53_HEAD_MM_KB", "2") or 2)


def head_mm_config(mesh, rows: int, n: int):
    """The fixed 2D config for the head's matmul of `rows` rows and n columns, or None (off, or the shape not taken);
    below 8 tile rows, GLM53_HEAD_MM_1D's 1D config (or None)."""
    mt, nt = -(-rows // 32), n // 32
    p1 = int(os.environ.get("GLM53_HEAD_MM_1D", "0") or 0)
    if mt < HEAD_MM_2D_MIN_TILES:
        if p1 <= 0 or mt not in HEAD_MM_1D_TILES:
            return None
        kb = head_mm_kb()
        key = ("1d", mt, nt, p1, kb)
        if key not in _HEAD_MM:
            grid = mesh.compute_with_storage_grid_size()
            cfg = None
            if -(-nt // p1) <= grid.x * grid.y:
                cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(grid.x, grid.y),
                    in0_block_w=kb,  # (2: ttnn's default's for K = 4096 at these shapes, the same blocks)
                    out_subblock_h=1,
                    out_subblock_w=max(v for v in (4, 2, 1) if p1 % v == 0),
                    out_block_h=mt,
                    out_block_w=p1,
                    per_core_M=mt,
                    per_core_N=p1,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            _HEAD_MM[key] = cfg
        return _HEAD_MM[key]
    pn = int(os.environ.get("GLM53_HEAD_MM_2D", HEAD_MM_2D_DEFAULT) or 0)
    # only where ttnn's default is the 1D form with K blocks of 2 (its narrow shapes: width // height > 8, i.e. up to
    # 16 tile rows here): past it the default is its 2D path, whose K blocks follow the grid (Kt / grid.x or 1) and the
    # L1 left - other blocks, other sums (1,024 rows differed)
    if pn <= 0 or mt < HEAD_MM_2D_MIN_TILES or mt % 8 != 0 or n // (32 * mt) <= 8:
        return None
    kb = head_mm_kb()
    key = (mt, nt, pn, kb)
    if key not in _HEAD_MM:
        grid = mesh.compute_with_storage_grid_size()
        gx, gy = -(-nt // pn), 8
        cfg = None
        if gx <= grid.x and gy <= grid.y:
            pm = mt // gy
            sub_w = max(w for w in (4, 2, 1) if pn % w == 0 and pm * w <= 4)
            sub_h = max(h for h in (4, 2, 1) if pm % h == 0 and h * sub_w <= 4)
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=kb,  # (2: ttnn's default's for K = 4096, the same blocks)
                out_subblock_h=sub_h,
                out_subblock_w=sub_w,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
            )
        _HEAD_MM[key] = cfg
    return _HEAD_MM[key]


_HEAD_FIDS: dict = {}


def head_linear(mesh, x, w, compute_kernel_config):
    """The mesh head's matmul x @ w -> fp32 (ttnn's default config, or `head_mm_config`'s; GLM53_HEAD_MM_FIDELITY,
    value-changing, in place of the given config's fidelity)."""
    cfg = head_mm_config(mesh, x.shape[-2], w.shape[-1])
    kw = {} if cfg is None else {"program_config": cfg}
    fid = os.environ.get("GLM53_HEAD_MM_FIDELITY")
    if fid:
        from .common import fidelity  # (the names: lofi / hifi2 / hifi3 / hifi4)

        f = fidelity("GLM53_HEAD_MM_FIDELITY")
        if f not in _HEAD_FIDS:
            _HEAD_FIDS[f] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        compute_kernel_config = _HEAD_FIDS[f]
    return ttnn.linear(x, w, dtype=ttnn.float32, compute_kernel_config=compute_kernel_config, **kw)


# the rows' hidden states gathered as tiles when a row's tokens fill whole tile rows (32 a row and more: no untilize
# and tilize around the gather, 3 + 55 us at 32 a row), else row-major and tilized after.  False: always row-major
TILED_GATHER = True


# A looser acceptance of a verify step's drafts: where a draft's logit is within this many nats of the top logit at the
# position before it (the same as its log-probability's gap: one softmax), the greedy head hands back the draft as its
# id there (`TtLmHead.greedy_mesh`'s `relax`), so the acceptance (DFlash2's `verify_accept.accept_device`, MTP's on the
# host or in vLLM's plugin) takes it as it takes an equal one, and the bonus after it is the target's own greedy id
# given the drafts taken.  0 (the default): off, the plain greedy head (no extra op).  On, the gap is a device scalar
# (`TtLmHead.set_relax`: changed between steps without a new capture); < 0 (e.g. -1: on, at the exact acceptance) takes
# none, the exact acceptance's ids bit for bit
ACCEPT_RELAX = float(os.environ.get("GLM53_ACCEPT_RELAX", "0"))
# ... its decision inside the greedy head's pick over the mesh (`head_relax.head_pick_relax`: the next token's logit
# read where a chip holds it, folded with the (value, id) pairs, the substitution at the end; the same ids) for up to
# head_relax.MAX_ROWS rows of tokens, in place of the ttnn ops of `_relaxed`: the head +116 / +121 / +131 -> +10 / +18 /
# +38 us at 64 / 128 / 256 rows, and one all-gather of the step's tokens for the next tokens' column's 9 ops; the
# same ids.  False (a test's): those ops
RELAX_FUSED = True
BIG = 1.0e30  # (a masked logit: -BIG)


def norm_weight(mesh, w: torch.Tensor):
    return put(mesh, w.reshape(1, 1, -1, 32), dims=(None, None), layout=ttnn.ROW_MAJOR_LAYOUT)


class TtEmbedding:
    def __init__(self, mesh, table: torch.Tensor) -> None:
        self.table = put(mesh, table[None, None], dims=(None, 3), layout=ttnn.ROW_MAJOR_LAYOUT)

    def __call__(self, ids):
        """ids [1, S] uint32 row-major (each row its own tokens) -> [1, 1, S, D] bf16, the same on a row's chips."""
        quarter = ttnn.embedding(ids, self.table, layout=ttnn.TILE_LAYOUT)
        quarter = ttnn.reshape(quarter, [1, 1, quarter.shape[-2], quarter.shape[-1]])
        return ttnn.all_gather(quarter, dim=3, cluster_axis=TP_AXIS)


# a decode step's norms (at most one tile row of tokens) on this many cores, the features split over them
# (width-sharded): ttnn's interleaved norm splits by tile rows, so it ran on one core (58 us a norm). 16 / 32 / 64
# cores: decode 4 % faster at 8 and 128 users alike; 0: never
NORM_SHARD_CORES = 16
# ... for up to this many tokens (whole tile rows a shard: 512 users' 64 a row took the interleaved norm, 2 tile rows on
# 2 cores, 67 us a call against 7.6 for 32 tokens sharded, 90 calls a step).  32: one tile row only (the earlier form)
NORM_SHARD_ROWS = 64


class TtRmsNorm:
    def __init__(self, mesh, weight: torch.Tensor, eps: float, shard_cores: int | None = None) -> None:
        self.mesh = mesh
        self.weight = norm_weight(mesh, weight)
        self.weight_tile = put(mesh, weight.reshape(1, 1, 1, -1), dims=(None, None))  # row 0 (`mhc_stream_read`)
        self.eps = eps
        self.features = weight.numel()
        self.shard_cores = NORM_SHARD_CORES if shard_cores is None else shard_cores
        self._sharded: dict = {}  # (cores, tile rows) -> (memory config, program config)

    def __call__(self, x):
        norm = dict(weight=self.weight, epsilon=self.eps, compute_kernel_config=HIFI4_FP32)
        if not self.shard_cores or x.shape[-2] > max(32, NORM_SHARD_ROWS):
            return ttnn.rms_norm(x, **norm)
        memory, program = self._shard_configs(self.shard_cores, -(-x.shape[-2] // 32))
        sharded = ttnn.to_memory_config(x, memory)
        y = ttnn.rms_norm(sharded, program_config=program, memory_config=memory, **norm)
        ttnn.deallocate(sharded)
        out = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y)
        return out

    def sharded_configs(self, rows: int):
        """(memory config, program config) of the width-sharded form `__call__` takes for `rows` tokens, or None (it
        takes the interleaved norm there)."""
        if not self.shard_cores or rows > max(32, NORM_SHARD_ROWS):
            return None
        return self._shard_configs(self.shard_cores, -(-rows // 32))

    def on_shard(self, sharded, memory, program):
        """The norm of an input already in the width-sharded form (`sharded_configs`: e.g. written there by the matmul
        before it), its output left there: `__call__`'s norm without the moves in and out."""
        norm = dict(weight=self.weight, epsilon=self.eps, compute_kernel_config=HIFI4_FP32)
        return ttnn.rms_norm(sharded, program_config=program, memory_config=memory, **norm)

    def _shard_configs(self, cores: int, tile_rows: int = 1) -> tuple:
        if (cores, tile_rows) not in self._sharded:
            grid = self.mesh.compute_with_storage_grid_size()
            x = max(x for x in range(1, grid.x + 1) if cores % x == 0 and cores // x <= grid.y)
            block_w = self.features // 32 // cores
            memory = ttnn.create_sharded_memory_config(
                shape=(32 * tile_rows, self.features),
                core_grid=ttnn.CoreGrid(y=cores // x, x=x),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(x, cores // x),
                subblock_w=max(w for w in (4, 2, 1) if block_w % w == 0),
                block_h=tile_rows,
                block_w=block_w,
                inplace=False,
            )
            self._sharded[(cores, tile_rows)] = (memory, program)
        return self._sharded[(cores, tile_rows)]


class TtLmHead:
    def __init__(self, mesh, weight: torch.Tensor, dtype=ttnn.bfloat16) -> None:
        self.weight = tp_weight(mesh, weight, "out", dtype)  # [D, V/4] per chip (the logits)
        self.mesh = mesh
        # greedy_mesh's: chip k = 4 r + c: ids k w .. k w + w - 1 (w whole tiles; past V zero weights, masked)
        vocab, d, chips = weight.shape[0], weight.shape[1], ROWS * COLS
        w = -(-vocab // (chips * 32)) * 32
        if weight.is_meta:  # the synthetic weights: the shape only (the ids past V are masked anyway)
            padded = torch.empty(chips * w, d, dtype=weight.dtype, device="meta")
        else:
            padded = torch.zeros(chips * w, d, dtype=weight.dtype)
            padded[:vocab] = weight
        per_chip = padded.T.contiguous().view(d, ROWS, COLS * w).permute(1, 0, 2)  # [8, D, 4 w]
        self.mesh_weight = put(mesh, per_chip.unsqueeze(1), dims=(0, 3), dtype=dtype)
        self.mesh_ids_width = w
        self.mesh_valid = [max(0, min(w, vocab - k * w)) for k in range(chips)]
        self.vocab = vocab
        self.relax = None  # the looser acceptance's gap, a device scalar [1, 1, 1, 1] fp32 (`prepare_relax`)
        self.relax_value = None
        self._consts = None  # `prepare_scoring`'s
        if ACCEPT_RELAX != 0:
            self.prepare_relax(ACCEPT_RELAX)

    # -- the looser acceptance and the scoring ------------------------------------------------------------------------
    def prepare_scoring(self) -> None:
        """The device constants of `score_mesh` and the relaxed `greedy_mesh`, made once: before any capture (a buffer
        made after one may sit in its scratch, and a replay writes over it).  Each chip's global ids [1, 1, 1, w] fp32
        (chip k = 4 r + c: k w ..), the bias of its padded columns (0 at the real ids, -BIG past the vocabulary), and
        the biases that keep the even / odd columns of the relaxed head's 32 gathered (top, draft logit) pairs."""
        if self._consts is not None:
            return
        chips, w = ROWS * COLS, self.mesh_ids_width
        ids = torch.arange(chips * w, dtype=torch.float32)
        per_chip = lambda t: t.view(ROWS, 1, 1, COLS * w)  # noqa: E731  (dims (0, 3): chip (r, c)'s w columns)
        pairs = torch.zeros(1, 1, 1, 2 * chips)
        odd, even = pairs.clone(), pairs.clone()
        odd[..., 1::2] = -BIG  # keeps the even columns (the tops)
        even[..., 0::2] = -BIG  # keeps the odd ones (the drafts' logits)
        self._consts = {
            "ids": put(self.mesh, per_chip(ids), dims=(0, 3), dtype=ttnn.float32),
            "pad": put(self.mesh, per_chip(torch.where(ids < self.vocab, 0.0, -BIG)), dims=(0, 3), dtype=ttnn.float32),
            "tops": put(self.mesh, odd, dims=(None, None), dtype=ttnn.float32),
            "drafts": put(self.mesh, even, dims=(None, None), dtype=ttnn.float32),
        }

    def prepare_relax(self, delta: float) -> None:
        """The looser acceptance on (its device constants and the gap scalar, before any capture; the verify steps'
        inputs made after it carry the shifted-token pad, `TtGlm53Model.alloc_spec_io`), with the gap `delta`."""
        self.prepare_scoring()
        if RELAX_FUSED:  # (its landing areas and semaphores, before any capture)
            from .head_relax import prealloc as relax_prealloc

            relax_prealloc(self.mesh)
        if self.relax is None:
            self.relax = put(self.mesh, torch.full((1, 1, 1, 1), float(delta)), dims=(None, None), dtype=ttnn.float32)
            self.relax_value = float(delta)
        self.set_relax(delta)

    def set_relax(self, delta: float) -> None:
        """The gap in nats (a host write when it changes; between steps, never inside a capture).  < 0: no draft is
        taken but an equal one (the exact acceptance's ids, the same bits)."""
        assert self.relax is not None, "prepare_relax first"
        if float(delta) == self.relax_value:
            return
        host = ttnn.from_torch(
            torch.full((1, 1, 1, 1), float(delta)),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, self.relax)
        self.relax_value = float(delta)

    def _next_logits(self, part, nxt):
        """part [1, 1, M, w] fp32 (this chip's logits), nxt [1, 1, M, 1] fp32 (each row's next token; -1: none) -> the
        next token's logit where this chip holds it, else -BIG, [1, 1, M, 1] fp32 (a max over 0/1-picked values:
        exact)."""
        mask = ttnn.eq(self._consts["ids"], nxt)  # [1, 1, M, w]: 1 at the next token's column
        picked = ttnn.where(mask, part, -BIG)
        ttnn.deallocate(mask)
        out = ttnn.max(picked, dim=3, keepdim=True)
        ttnn.deallocate(picked)
        return out

    def _relaxed(self, part, value, ids, relax):
        """The relaxed head's ids: the mesh's greedy ids `ids` [1, 1, M, 1], except the next token's where its logit is
        within the gap of the top (the 32 chips' tops `value` and next logits gathered: their maxima exact)."""
        nxt, delta = relax
        mine = self._next_logits(part, nxt)
        pair = ttnn.concat([value, mine], dim=3)  # [1, 1, M, 2]
        ttnn.deallocate(mine)
        row = ttnn.all_gather(pair, dim=3, cluster_axis=TP_AXIS)
        ttnn.deallocate(pair)
        every = ttnn.all_gather(row, dim=3, cluster_axis=ROW_AXIS)  # [1, 1, M, 64]: chip k's (top, next) at 2 k
        ttnn.deallocate(row)
        out = []
        for key in ("tops", "drafts"):
            kept = ttnn.add(every, self._consts[key])
            out.append(ttnn.max(kept, dim=3, keepdim=True))
            ttnn.deallocate(kept)
        ttnn.deallocate(every)
        top, nxt_logit = out
        gap = ttnn.sub(top, nxt_logit)
        for x in out:
            ttnn.deallocate(x)
        near = ttnn.le(gap, delta)
        ttnn.deallocate(gap)
        relaxed = ttnn.where(near, nxt, ids)
        ttnn.deallocate(near)
        return relaxed

    def score_mesh(self, x, nxt):
        """Teacher forcing's scores (`TtGlm53Model.prefill_score`): x [1, 1, T, D] bf16 (each row its T positions), nxt
        [1, 1, 8 T, 1] fp32 (every row's positions' next tokens; -1: none) -> [1, 1, 8 T, 3] fp32 a chip, its own: the
        top of its real ids' logits m, the sum of exp(logit - m) over them, and the next token's logit (-BIG where
        another chip holds it).  The host joins the 32 (`score_join`)."""
        self.prepare_scoring()
        x_all = self._gather_rows(x)
        part = ttnn.linear(x_all, self.mesh_weight, dtype=ttnn.float32, compute_kernel_config=HIFI4_FP32)
        ttnn.deallocate(x_all)
        real = ttnn.add(part, self._consts["pad"])
        top = ttnn.max(real, dim=3, keepdim=True)
        shifted = ttnn.sub(real, top)
        ttnn.deallocate(real)
        e = ttnn.exp(shifted)
        ttnn.deallocate(shifted)
        total = ttnn.sum(e, dim=3, keepdim=True, compute_kernel_config=HIFI4_FP32)
        ttnn.deallocate(e)
        mine = self._next_logits(part, nxt)
        ttnn.deallocate(part)
        out = ttnn.concat([top, total, mine], dim=3)
        for t in (top, total, mine):
            ttnn.deallocate(t)
        return out

    @staticmethod
    def score_join(out, rows: int) -> torch.Tensor:
        """`score_mesh`'s tensor -> [rows, 3] float64 on the host: the next token's log-probability, its logit's gap
        below the top (0: it is the top), the top's log-probability."""
        chips = torch.stack(
            [ttnn.to_torch(c).reshape(-1, 3)[:rows].double() for c in ttnn.get_device_tensors(out)]
        )  # [32, rows, 3]
        m, s, nl = chips[..., 0], chips[..., 1], chips[..., 2]
        top = m.max(0).values
        lse = top + torch.log((s * torch.exp(m - top)).sum(0))
        nxt = nl.max(0).values
        return torch.stack([nxt - lse, top - nxt, top - lse], dim=-1)

    def __call__(self, x):
        """[1, 1, S, D] bf16 -> logits [1, 1, S, V] fp32, the same on a row's chips."""
        part = ttnn.linear(x, self.weight, dtype=ttnn.float32, compute_kernel_config=HIFI4_FP32)
        return ttnn.all_gather(part, dim=3, cluster_axis=TP_AXIS)

    def greedy_mesh(self, x, relax=None, keep_greedy: bool = False, own: bool = False):
        """[1, 1, T, D] bf16 (each row its T tokens) -> the argmax ids of all the rows' tokens [1, 1, 8 T, 1] fp32 (row
        r's at r T), the same on every chip: the tokens gathered over the rows (row-major), each chip's top logit of
        its 1/32 of the vocabulary and its smallest global id by one op (`head_argmax.head_argmax`: the RISC-Vs of the
        grid scan the logits once; 120 / 110 / 231 -> 43 / 58 / 100 us at 16 / 64 / 256 rows against the six ttnn
        passes it replaced, the same bits), the (value, id) pairs of the 32 chips to the top (ties: the
        smallest id, as torch's argmax).

        `relax` (the looser acceptance, ACCEPT_RELAX): (nxt [1, 1, 8 T, 1] fp32, each token's next one in the verify
        step, -1 for none; the gap scalar) -> at each token whose next one's logit is within the gap of the top, the
        next one in place of the argmax (`_relaxed`); with keep_greedy (relax given) -> (those ids, the argmax ids).
        Fused (RELAX_FUSED, `TtGlm53Model._relax`): (the step's tokens, every row's [1, 1, 8, b] uint32 row-major, the
        gap, U), the same in one op (`head_relax.head_pick_relax`).

        `own` (no `relax`; GLM53_HEAD_PICK_OWN, the DFlash2 step with its fused acceptance): each chip's own mesh
        row's ids only (`head_pick` v3; the other rows undefined), where head_pick takes the rows (else as without)."""
        x_all = self._gather_rows(x)
        part = head_linear(self.mesh, x_all, self.mesh_weight, HIFI4_FP32)
        ttnn.deallocate(x_all)
        from .head_argmax import head_argmax  # (it imports common)

        value, ids = head_argmax(self.mesh, part, self.mesh_valid, self.mesh_ids_width)
        if relax is None:
            ttnn.deallocate(part)
            return self._pick(x, value, ids, own=own)
        assert not own, "head: own rows only without the looser acceptance"
        if len(relax) == 3:  # the fused form: (every row's tokens [1, 1, 8, b] uint32 row-major, the gap, U)
            from .head_relax import head_pick_relax

            tokens, delta, users = relax
            if value.dtype != ttnn.float32:
                value = ttnn.typecast(value, ttnn.float32)
            out, top = head_pick_relax(
                self.mesh, value, ids, part, tokens, delta, users, self.mesh_valid, self.mesh_ids_width
            )
            for t in (part, value, ids) + (() if keep_greedy else (top,)):
                ttnn.deallocate(t)
            return (out, top) if keep_greedy else out
        top = self._pick(x, value, ids, keep_value=True)
        out = self._relaxed(part, value, top, relax)
        for t in (part, value) + (() if keep_greedy else (top,)):
            ttnn.deallocate(t)
        return (out, top) if keep_greedy else out

    def sample_mesh(self, x, sampling, seeds):
        """As `greedy_mesh` (no looser acceptance), with sampling (`sampling.StepSampling`): the
        greedy ids at the greedy rows, a draw from the target's distribution at the others (`seeds` [n] uint32: the
        step's, `StepSampling.seeds`) -> [1, 1, 8 T, 1] fp32 on every chip."""
        x_all = self._gather_rows(x)
        part = head_linear(self.mesh, x_all, self.mesh_weight, HIFI4_FP32)
        ttnn.deallocate(x_all)
        from .head_argmax import head_argmax  # (it imports common)

        value, ids = head_argmax(self.mesh, part, self.mesh_valid, self.mesh_ids_width)
        greedy = self._pick(x, value, ids)
        out = sampling.draw(part, greedy, seeds)
        for t in (part, greedy):
            ttnn.deallocate(t)
        return out

    @staticmethod
    def _gather_rows(x):
        """[1, 1, t, D] a row -> every row's [1, 1, 8 t, D] (row r's at r t), tiled."""
        if TILED_GATHER and x.shape[-2] % 32 == 0:
            return ttnn.all_gather(x, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8 t, D], the same rows
        flat = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        gathered = ttnn.all_gather(flat, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8 t, D]
        ttnn.deallocate(flat)
        x_all = ttnn.tilize_with_zero_padding(gathered)
        ttnn.deallocate(gathered)
        return x_all

    def _pick(self, x, value, ids, keep_value: bool = False, own: bool = False):
        """Each chip's (value, id) [1, 1, 8 t, 1] -> the mesh's top ids [1, 1, 8 t, 1] fp32, on every chip (`value`
        left allocated with keep_value).  own: the own mesh row's t rows only (head_pick v3, t a multiple of 8)."""
        if HEAD_PICK and ROWS * x.shape[-2] <= head_pick_max_rows():
            from .head_pick import head_pick  # (it imports common)

            if value.dtype != ttnn.float32:
                value = ttnn.typecast(value, ttnn.float32)
            out = head_pick(self.mesh, value, ids, own=own and value.shape[-2] % 64 == 0)
            if not keep_value:
                ttnn.deallocate(value)
            ttnn.deallocate(ids)
            return out
        # gathered over the row's chips, then the mesh rows: [1, 1, 8 t, 32], chip 4 r + c's in column 4 r + c
        gather = lambda v: ttnn.all_gather(  # noqa: E731
            ttnn.all_gather(v, dim=3, cluster_axis=TP_AXIS), dim=3, cluster_axis=ROW_AXIS
        )
        values, ids = gather(value), gather(ids)
        top = ttnn.max(values, dim=-1, keepdim=True)
        others = ttnn.multiply(ttnn.ne(values, top), 1.0e9)
        return ttnn.min(ttnn.add(ids, others), dim=-1, keepdim=True)
