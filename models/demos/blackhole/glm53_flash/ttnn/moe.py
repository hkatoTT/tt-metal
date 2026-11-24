"""The MoE on the 8x4 mesh: the routed experts sliced by mesh column, run by the experts op.

Placement: mesh column c holds the column's 72 experts 72 c .. 72 c + 71, chip (row r, column c) their intermediate
columns 256 r .. 256 r + 255 (and the shared expert's, as the column's slot 72: SHARED_IN_OP_MAX), in the experts
op's layout (`_init_experts_op`), so a column's chips do the same work.
Flow for one layer (`__call__`):
  1. the rows' tokens gathered over the mesh column (`_gather_rows`): every chip sees the column's N = 8 T tokens;
  2. the router (fp32) and the routing op (`moe_route.route_only` in a decode step, else `moe_decode_route`): each
     token's 8 experts among the column's 72, their normalized weights scaled by 2.5;
  3. the experts op (`moe_decode_experts`) on up to EXPERTS_OP_CALL tokens a call, or the prefill's
     `moe_prefill.experts` (each expert's weights read once a chunk) past that: the chip's columns of every chosen
     expert, weighted and summed;
  4. back to the rows (`_scatter_rows`, or the one-shot column scatter and row all-reduce, `_scatter_ar`), plus the
     shared expert's TP4 partial where the experts op did not compute it, summed over the row's 4 chips.
(An earlier placement held the experts whole on the chips, 9 a chip, with dense and sparse_matmul paths for the
decode and prefill; those were removed.)
"""

from __future__ import annotations

import contextlib
import os

import torch

import ttnn
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tt.moe import MoeWeights
from models.demos.blackhole.glm53_flash.ttnn import common, moe_route, w1_dense
from models.demos.blackhole.glm53_flash.ttnn.common import (
    COLS,
    HIFI4_FP32,
    ROW_AXIS,
    ROWS,
    ar_input_config,
    ar_input_dtype,
    host_threads,
    put,
    synthetic,
    tp_all_reduce,
    upload_shards,
    weight_dtype,
    zeros,
)
from models.demos.blackhole.glm53_flash.ttnn.mlp import TtMlp

# GLM53_MOE_PACK_SLICES (default 1; 0: the cut, then from_torch): the real experts'
# shards packed into block-float tiles straight from the column's 72 experts' slices, transposed, in one pass
# (ttnn bfp_utils.pack_transposed_slices: tt-metal's bfp_pack_fast), with no bf16 copy of each 151 MB shard made first.
# The same bytes as from_torch(torch.cat(..)) (the CPU check against the general form's copy, then the weights' bytes)
PACK_SLICES = os.environ.get("GLM53_MOE_PACK_SLICES", "1") == "1"

SLICE = 256  # a chip's intermediate columns of each of its column's experts
# a row's tokens, when fewer than this, go over the row collectives padded to it: ttnn picks slower algorithms for
# small tensors (8 tokens: all-gather 117 us, reduce-scatter 192 us; 32 tokens: 41, 45 us).  Decode 5 % faster at 8
# and 128 users
CCL_PAD_ROWS = 32
# the padded rows' tokens taken out and put back by 0/1 matmuls (`_row_picks`; exact copies; in place of slice +
# reshape and reshape + pad, 45 and 46 us a layer at 128 users): decode 3.1 / 2.0 ms faster at 8 / 128 users
# up to 4 tokens a row: the row gather on the way in and the column sum on the way back as one-shot fabric writes
# (`col_collectives.py`, generic_op; the sums in fp32, max |d| from the exact sum 0.0039 against 0.0075-0.0096), in
# place of the untilize, row-major all_gather and tilize, and of the all_reduce and the 0/1 pick.  Alone (trace, 100
# calls) the gather 22.4 / 32.1 -> 12.9 / 16.2 us, the scatter 38.7 / 38.6 -> 14.0 / 22.9 us at 1 / 4 tokens a row;
# with the row all-reduce op, decode 8 / 32 / 128 users 26.3 / 37.8 / 57.5 -> 23.8 / 36.0 / 57.7 ms (synthetic), real
# weights 21.9 / 29.4 / 52.5 ms (the row op only: 23.5 / 30.7 / 52.4); 4 x 2,000 stress steps at 8
# and 32 users clean; its persistent L1 only semaphores (DRAM landing areas).  The teacher-forcing KLD 0.00228 /
# 0.00254 -> 0.00268 / 0.00282, from a few positions where near ties flip (the median the same, a bootstrap 95 %
# interval of the difference around 0); per layer the sums are closer to float64 and to tt/ than the bf16
# all_reduce's, which run +1e-5 high.  On by default (as the router config and the SFPU Sinkhorn).  0: ttnn's padded
# reduce-scatter and all-reduce (their sums' order: kept for comparing the accuracy)
MOE_COL_CUSTOM = os.environ.get("GLM53_MOE_COL_CUSTOM", "1") == "1"
# ... its gather whatever MOE_COL_CUSTOM says (exact copies, no adds: the KLD question is the scatter's): decode 8 / 32
# / 128 users 24.5 / 36.6 / 56.6 -> 24.1 / 36.0 / 56.6 ms, 2 x 2,000 traced steps at 32 and at 8 users without a
# hang; up to this many tokens a row (above 4 with 32 workers: at 16, 40.5 us against the pad, all-gather and 0/1
# matmul's ~58; decode 128 users 56.6 -> 55.9 ms; 2 x 2,000 traced steps without a hang)
COL_GATHER_MAX = 16
# the experts op's padded y (`row_tokens`: 8 or 16 tokens a row, each row's in its own tile row) back to the rows, up to
# this many tokens a row, by the one-shot column scatter (`col_collectives.col_scatter_padded`: fp32 sums in the chips'
# order, as `col_scatter`; closer to the exact sum than ttnn's) in place of ttnn.reduce_scatter and the slice.  Alone
# (traced) 47.9 -> 39.1 us at 8 tokens a row, but 47.4 -> 66.2 at 16 (each slice crosses up to 4 links: bound by the
# links); MTP's verify step at 32 users and the plain step at 64 -0.5 ms.  With the row's all-reduce in it
# (`_scatter_ar`)
MOE_COL_SCATTER_PADDED = 8
# (the experts op writes its y for 8 or 16 tokens a row, 64 / 128 users, by mesh row into the reduce-scatter's padded
# layout: each row's tokens at the top of a tile row of its own, zeros after, `row_tokens`, in place of the 0/1 matmul
# that spread them, 20 us a layer at 128 users; the same values, decode 128 / 64 users 55.9 / 46.3 -> 54.9 / 45.9 ms)
# the way back for a few tokens a row in one op (`col_collectives.col_scatter_ar`: the column's one-shot scatter, then
# the row's all-reduce as `row_all_reduce` makes it, its slices sent on as soon as the column's sums are there) where it
# took col_scatter (t <= 4) or col_scatter_padded (t = 8, the experts op's padded y) and then tp_all_reduce, with the
# shared expert in the experts op: the same sums as col_scatter and row_all_reduce (t <= 4); at t = 8 row_all_reduce's
# sums in place of all_reduce_async's (the same bits over 1M test values).  Alone (traced) t = 1 / 2 / 4 / 8: 23.6 /
# 27.0 / 34.8 / 57.3 -> 18.6 / 22.1 / 29.5 / 42.0 us; MTP's verify step at 32 users 43.4 -> 42.85 ms, the plain step at
# 32 users 31.55 -> 31.3 ms, 8 users -0.1 ms; 16,000 traced steps at 8 and 32 users without a hang
# the router's matmul ([N, 4096] @ [4096, 288] -> fp32) with a program config up to ROUTER_CONFIG_ROWS tokens
# (`_router_config`): ttnn's default read K in small blocks on 9 / 36 cores, 54 / 95 us at 32 / 128 tokens, and 19 / 21
# us with these (trace-timed; K in slices on more cores was slower, 47 / 130 us, from the transposes and the sum).
# Decode 1.7 / 2.6 ms faster at 8 / 128 users.  (256 until the experts op took 512 tokens: 512 users' router matmul ~4x
# slower.)  1,024 for the prefill's waves of 1,024 tokens a row (a long prefill wave: ttnn's default above 512 rows
# blocks K otherwise, other bits; this config's 2D form at 32 tile rows, 8 rows of cores, the same bits as two waves of
# 512).  More tokens: ttnn's default
ROUTER_CONFIG_ROWS = 1024
# random weights (`ShapeOnlyCheckpoint`): "spread" (the default) zeroes the random selection bias, so each token picks
# its own experts, uniformly (the real weights with random tokens pick fewer distinct ones: spread is 3-7 ms pessimistic
# at 8-128 users); "bias" keeps it, and since it outweighs the random router's logits ~10x every token picks about the
# same 8 experts (4-17 ms optimistic)
SYNTHETIC_ROUTING = os.environ.get("GLM53_SYNTHETIC_ROUTING", "spread")
# mesh tokens a layer takes through the experts op and moe_prefill.experts (more: an error), in calls of
# EXPERTS_OP_CALL (the op's packed units: 16 tile rows, once the down cores alone held the h slots; 512 users 142.9 ->
# 125.6 ms against calls of 256; 1,024 users in two calls of 512: 334.6 -> 229.0 ms against the dense path).
# 8,192: the prefill's waves of 1,024 tokens a row (moe_prefill's kernels take any N, their L1 independent of it; one
# wave of 8,192 the bits of two of 4,096; the op 2.94 ms against 2 x 1.82)
EXPERTS_OP_COLUMN_MAX = 8192
EXPERTS_OP_CALL = 512
# the prefill's chunks of more than EXPERTS_OP_CALL mesh tokens through moe_prefill.experts (each expert's weights read
# once a chunk, the tokens gathered by expert) in place of the experts op's calls of EXPERTS_OP_CALL tokens, each
# reading the column's weights again (the prefill's KLD as before), with BFP8 or BFP4 experts (its kernels' formats):
# `_routed_once_local` for T >= 32 a row in multiples of 32 (the router and the routing op on the row's own T tokens,
# their rows and the tokens, row-major, gathered over the column: the router's work an eighth, no untilize of the
# column's tokens), else in `_routed_sliced`.  The chunk's padding tokens (past a row's prompt, or a row without a
# request: `KdaPicks.real` 0) get no pairs from the plan (their partial sums 0): a wave with few requests had every
# empty row's tokens routed to the same experts (one request of 512 at 0.56 s against 0.41 for eight; the real tokens'
# sums the same bit for bit); the mask gathered over the column once a chunk and shared by the MoE layers
# (`KdaPicks.derived`)
# decode: the shared expert as one more expert of the experts op, in mesh column `layer % 4` (the column's 73rd, local
# 72: its 2048 intermediate columns 256 a chip as a routed expert's), chosen by every token with weight 1.0 by the
# routing on that column's chips (`local_base` [.., 1]) and summed by the same reduce-scatter and all-reduce, in place
# of the separate TP4 shared MLP (6 ops, 38 us a layer at 8 users), which stays for the prefill and more tokens: up to
# this many tokens of the mesh (one tile row: 8 / 32 users -1.6 / -1.4 ms; two: the verify step at 32 users and the
# plain step at 64 users -1.3 / -1.2 ms, the KLD the same); at 128 users the shared expert's 4 units in one column cost
# more than the TP4 MLP (+0.6 ms), so it stays there, unless spread over the columns, a quarter of the tokens each
# (moe_route.SHARED_SPLIT: -0.9 .. -1.1 ms a step at 128 .. 512 users; an earlier form before route_only: 128 / 512
# users 56.5 / 125.2 -> 56.2 / 126.0 ms, not kept)
SHARED_IN_OP_MAX = 64
# The progressive form (GLM53_MOE_PROG_F, a quality-changing trial, off by default):
# the routed experts' (and the in-op shared expert's) weights each a BFP4 base, BFP4(W), and a BFP4 correction,
# BFP4(W - base), both made on the device from bf16 uploads by the packer (`_prog_split`), in the experts op's layout
# with twice the slots (the corrections after the bases); the experts op (moe_decode_experts' program factory reads
# GLM53_MOE_PROG_F) reads the correction too for the column's experts with the largest routing weight sums, within F
# of the column's BFP8 bytes; moe_prefill reads both for every expert.  Its value: F (0 .. 1.06; 2: every expert);
# 10 + F: F's and the in-op shared expert's correction always (at F = 0.5 the busiest column reads none, and the shared
# expert's went missing: verify KLD -30 / -18 % at 8 / 32 users, a step +0.4 %; value-changing); 9: the shared
# expert's only (route_build.cpp)
PROG = os.environ.get("GLM53_MOE_PROG_F") is not None
# (the experts op's w1 by a tile copy, GLM53_MOE_W1_COPY, and the fewer experts a token, GLM53_MOE_TOPK / _KEEP, were
# removed)


# (H1's router matmul and routing run in one program, router_mm.router_route; the same values bit for bit, the model
# too; op alone 29.0 / 29.7 / 31.3 -> 25.7 / 27.2 / 31.2 us at 8 / 32 / 64 tokens a row; the DFlash2 step at 8 / 32
# users K = 7 -0.17 / -0.17 ms, 128 users K = 3 the same; 2 x 2,000-step stresses at 128 / 32 / 8 users clean.  The two
# ops before that, GLM53_MOE_ROUTER_FUSED=0, were removed)


# the MoE's way back at 32 tokens a row (the reduce-scatter, then the row all-reduce): the reduce-scatter's output in
# all_reduce_async's L1 shards (`ar_input_config`), in place of the move in.  The outputs bit for bit at 16 and 32 users a
# row; synthetic weights, traced, off / on twice: the verify step at 128 users 4,739 / 4,746 -> 4,756 / 4,754 tok/s;
# 2,000 traced steps there clean.  (False: the move, as before; tests)
MOE_RS_L1 = True


# the MoE's way back at 32 / 64 tokens a row (ttnn's ring reduce_scatter over the column) with these tuning knobs:
# "<tokens a row>:w<workers a link>c<chunks a sync>b<buffers a channel>" comma list (any of w / c / b), e.g. "64:w8c8".
# Alone (traced, back to back; the same bits): 64 tokens 72.0 -> 64.0 us (w8c2 .. w8c16, w4c2), 32 tokens 41.3 -> 40.5
# at best; the verify step at 128 users x T = 4 59.3 -> 59.0 ms (synthetic, traced).  The default "64:w4c2" (ttnn's own
# choice at 64 tokens a row: 2 workers a link and direction, a sync every 4 or fewer chunks; its threshold for more
# workers on Blackhole is 1 MB a link and direction, 0.875 here), and "128:c4" for the prefill's chunks of 128 (8 MB a
# chip: alone 135.5 -> 116.5 us, the same bits; a sync every 4 chunks, the 8 workers ttnn picks; 512 tokens a row gained
# nothing from any knob).  Read at each call (tests switch it).  "": ttnn's own choice at every t
RS_KNOBS = "64:w4c2,128:c4"


def _rs_knobs(t: int) -> dict:
    import re

    keys = {"w": "num_workers_per_link", "c": "chunks_per_sync", "b": "num_buffers_per_channel"}
    for item in RS_KNOBS.split(","):
        if ":" in item and int(item.split(":")[0]) == t:
            return {keys[a]: int(b) for a, b in re.findall(r"([wcb])(\d+)", item.split(":")[1])}
    return {}


# H1 (the MoE's column gather inside the experts op): in a decode step of H1_TOKENS tokens a mesh row, the router and
# route_only on the row's own tokens (route_only writing its rows into the column's landing w_all, `h1_landing`), and
# the experts op gathering the column's routing and x itself over the fabric (its sender core) while it starts, in place
# of the column gather before the router: the same values bit for bit.  The MoE layer alone, 64 / 128 / 256 / 512 tokens
# a column: 644.6 / 733.6 / 766.2 / 901.4 -> 617.8 / 696.6 / 726.8 / 854.6 us; decode (synthetic, traced): the verify
# step at 32 / 128 users 39.94 / 51.43 -> 39.16 / 50.29 ms, the plain step at 128 / 256 / 512 users 45.55 / 49.78 /
# 67.70 -> 44.46 / 49.16 / 66.06 ms; stress clean; with the mHC write and read fused: the verify step at 32 / 128 users
# 39.16 / 50.13 -> 38.70 / 48.70 ms, the plain step at 128 / 256 / 512 users 45.39 / 48.79 / 66.27 -> 44.27 / 47.91 /
# 64.65 ms, the outputs bit for bit.  Read at each call (tests switch it); its persistent state is made at the model's
# load (`prealloc_h1`).  On by default (2,000-step stresses with it); False: the column gather before the router, as
# before (tests)
H1 = True
# the prefill's routing (`_routed_once_local`) by moe_route.route_only, which gives moe_decode_route's rows bit for bit
# and faster (a wave of 1,024 tokens a row: 101.4 -> 32.3 us a layer; prefill 8 x 2,048 0.659 -> 0.651 s, 1 x 8K 0.400
# -> 0.395 s; the same values, two stresses clean), where it takes the call (a column's experts, its rows per core).
# Read at call time (tests switch it).  False: moe_decode_route
PREFILL_ROUTE_ONLY = True
# tokens a mesh row it takes: the experts op's packed units with builders (8 .. 64; at 1 .. 4 the experts op's one
# tile row form, whose x would have to be put together from 64-byte pieces: not done)
H1_TOKENS = (8, 16, 32, 64)
# GLM53_W1 (the L1 landing of the next MoE layer's chosen experts' weights, ttnn/w1_landing.py): a TtMoe's `w1_spec`
# (W1Landing.arm_model) goes to its experts op only inside the model's decode steps (`w1_decode`, TtGlm53Model.forward /
# forward_spec / forward_verify), on the H1 path: never a prefill's (its programs are made without it, and do not signal
# the DRISCs)
W1_DECODE = False


@contextlib.contextmanager
def w1_decode():
    """The model's decode steps: the armed TtMoe's `w1_spec` given to their experts op."""
    global W1_DECODE
    before, W1_DECODE = W1_DECODE, True
    try:
        yield
    finally:
        W1_DECODE = before


H1_LANDING = (8, 16, 32, 64)  # (the landing buffers made at the load)
# the column's way back in the experts op (with H1 at 32 / 64 tokens a mesh row): the op runs the ring reduce-scatter
# over the column itself (reducer and return-sender cores, the ring's sums with a 16-bit Dst as
# reduce_scatter_minimal_async: the same bits) and returns this mesh row's [t, D], in place of y and ttnn.reduce_scatter
# after it.  The same bits (the op; the model, test_model_bits); the verify step with the real weights 32 users K = 7
# 41.81 -> 41.06 ms, 128 users K = 3 56.23 -> 55.25 ms; 2 x 2,000-step stresses at 128 users T = 4 and 32 users T = 8
# clean.  On by default (False: ttnn.reduce_scatter after the op, as before; tests); its two more global semaphores are
# made at the load with H1's when on.  Read at each call (tests switch it; the semaphores then come at the first call,
# before any trace)
MOE_COL_RS = True
COL_RS_TOKENS = (32, 64)
# the prefill's x all-gather over the column (`_routed_once_local`) in two dimensions (x2d_gather.py mode 3: half of K
# down the column, relayed by two relay cores a chip to the row neighbours, who send the other half back; a copy, the
# same bits) for the chunks whose persistent landing the model took at its load (`prealloc_fabric_ops`: x2d_gather.TS,
# 128 / 512 / 1,024 tokens a row), else ttnn.all_gather.  The op alone at 1,024 / 512 tokens a row 706 / 363 -> 514 /
# 266 us; prefill 8 x 2,048 0.652 -> 0.636 s, 1 x 8K 0.398 -> 0.390 s (synthetic weights); the model bit for bit.  On by
# default (False: ttnn.all_gather, as before; tests).  Its landings are made at the model's load.  Read at each call
# (tests switch it)
X2D_GATHER = True
_X2D_SEEN: set = set()  # (the chunk sizes it took, printed once each)
# The prefill's way back as one op (`col_row_rs`): the column's reduce-scatter one traversal a batch, the shared
# expert's add and the row all-reduce's reduce-scatter fused behind it, then the row's all-gather; the same bits as
# `_finish`.  For the sizes whose buffers the generator took when made (col_row_rs.ts_for its chunks: 512 and 1,024
# tokens a row, 45 + 90 MiB a chip of DRAM; a chunk list without 512 or 1,024 takes none).  Read at each call (tests
# switch it).  On by default (8 x 2,048 -1.9 %, 1 x 8K -1.5 %); False: the ttnn ops one after the other (tests)
COL_ROW_RS = True
# mHC over the row's chips by tokens (common.ROW_SPLIT): the one-op way back without its row all-gather and an
# all-to-all to this chip's quarter of the tokens (1, default: a wave -2.8 ms at 1,024 tokens a row against 0), or with
# the all-gather and the quarter taken after it (0).  Read at each call
ROW_SPLIT_MOE_A2A = os.environ.get("GLM53_MHC_ROW_SPLIT_MOE_A2A", "1") == "1"
_CRR_SEEN: set = set()
# The prefill MoE's plan (moe_prefill.plan: its counts ~42 us and its push ~125 us at 1,024 tokens a row) in the x
# all-gather's program (x2d_gather), on the cores it leaves free: the two at once (the plan needs the routing and the
# mask only, not x): the counts and the push (the count cores signal one gathering push core, which tells the others: 32
# atomics a layer).  The same tables and values (test_model_bits 12 / 12 at chunks 512 / 1,024; the op's 20,000-call
# race stress clean); 8 x 2,048 -12.4 / -13.4 ms (-2.0 %), 1 x 8K -6.4 / -7.5 ms (-1.8 %).  (The push alone in it, the
# counts before, was removed.)  Read at each call (tests switch it).  False: the plan after the gather, as before
PLAN_WITH_X2D = True
_ROW_PICKS: dict = {}  # tokens per row -> (mesh, (pick, spread, real))
_H1: dict = {}  # (mesh, D) -> {"sems": (x, r) addresses, "keep": the semaphores, t: (x_all, w_all)}


def x_bfp8_mode() -> int:
    """GLM53_MOE_X_BFP8 (read at each call; 0 by default; value-changing): with H1 from 32 tokens a mesh row, the
    experts op's x (the column's, over the fabric and the row relay, and the shared expert's in the op) in BFP8_b,
    passed as x_row: half the bytes on the column's and the rows' links.  1: the BFP8 copy written by the router's
    program (router_mm RM_X8, after the logits); 2: by a ttnn.typecast before the op (a program more).  The router and
    the TP4 shared MLP keep bf16."""
    v = os.environ.get("GLM53_MOE_X_BFP8", "0")
    return int(v) if v in ("1", "2") else 0


def prealloc_h1(mesh, d: int) -> None:
    """H1's persistent state now, before any program is built or trace captured: two global semaphores (a word on
    every core) and, for each of H1_LANDING, the column's landing w_all [8 t, 32] fp32 (DRAM, interleaved, the same
    address on every chip) and x_all [8 t, D] bf16 (the experts op's x: only its shape is used, x lands in L1)."""
    for t in H1_LANDING:
        h1_landing(mesh, t, d)
    if MOE_COL_RS:
        col_rs_sems(mesh, d)


def col_rs_sems(mesh, d: int):
    """-> (rs_sem, rs_mail): the way back in the op's two global semaphores (a word on every core; made on first use)."""
    h1_landing(mesh, H1_LANDING[0], d)
    st = _H1[(id(mesh), d)]
    if "rs" not in st:
        grid = mesh.compute_with_storage_grid_size()
        full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        keep = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(2)]
        st["rs_keep"] = keep
        st["rs"] = tuple(ttnn.get_global_semaphore_address(g) for g in keep)
    return st["rs"]


def h1_landing(mesh, t: int, d: int):
    """-> (x_all, w_all, sem_x, sem_r) for t tokens a mesh row (made on first use: `prealloc_h1` first)."""
    key = (id(mesh), d)
    if key not in _H1 or _H1[key]["mesh"] is not mesh:
        grid = mesh.compute_with_storage_grid_size()
        full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        keep = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(2)]
        _H1[key] = {"mesh": mesh, "keep": keep, "sems": tuple(ttnn.get_global_semaphore_address(g) for g in keep)}
    st = _H1[key]
    if t not in st:

        def land(width, dtype):
            spec = ttnn.TensorSpec(
                ttnn.Shape([1, 1, ROWS * t, width]), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
            )
            return ttnn.allocate_tensor_on_device(spec, mesh)

        st[t] = (land(d, ttnn.bfloat16), land(32, ttnn.float32))
    return (*st[t], *st["sems"])


def _row_picks(mesh, t: int) -> tuple:
    """For t < CCL_PAD_ROWS tokens per row, the row collectives' padded layout (row r's token i at 32 r + i of
    8 x 32): `pick` [1, 1, n, 8 x 32] 0/1 takes the 8 t tokens out (token r t + i), `spread` (its transpose) puts n rows
    back; n is 8 t rounded up to a tile, the rows past 8 t zero tokens, and `real` [1, 1, n, 1] is 1 on the real ones
    (None when n = 8 t)."""
    if t not in _ROW_PICKS or _ROW_PICKS[t][0] is not mesh:
        n = -(-ROWS * t // 32) * 32
        pick = torch.zeros(n, ROWS * CCL_PAD_ROWS)
        for r in range(ROWS):
            for i in range(t):
                pick[r * t + i, r * CCL_PAD_ROWS + i] = 1.0
        rep = dict(dims=(None, None))
        real = None
        if n != ROWS * t:
            real = put(mesh, (torch.arange(n) < ROWS * t).float().view(1, 1, n, 1), **rep)
        _ROW_PICKS[t] = (
            mesh,
            (put(mesh, pick[None, None], **rep), put(mesh, pick.T.contiguous()[None, None], **rep), real),
        )
    return _ROW_PICKS[t][1]


def _prog_split(t):
    """A dense bf16 weight copy (tiled, DRAM) -> (base, correction), both BFP4 packed as the host packer would:
    base = BFP4(t), correction = BFP4(t - base) (`prog_split`, the device's RISC-Vs; ttnn.typecast's BFP4 truncates)."""
    from models.demos.blackhole.glm53_flash.ttnn import prog_split

    return prog_split.split(t.device(), t)


class TtMoe:
    def __init__(
        self,
        mesh,
        weights: MoeWeights,
        config: Glm53Config,
        expert_dtype=ttnn.bfloat16,
        dtype=ttnn.bfloat16,
    ) -> None:
        inter = config.moe_intermediate_size
        if config.num_experts % COLS or inter != SLICE * ROWS:
            raise ValueError(
                f"{config.num_experts} experts of I = {inter} do not split by mesh column, {SLICE} columns a chip, on "
                f"{ROWS}x{COLS}"
            )
        host_threads()
        self.mesh = mesh
        self.k = config.num_experts_per_token  # each token's experts
        self.scale = config.routed_scaling_factor
        self.limit = config.swiglu_limit
        # the routed experts' down projection in another precision than gate | up: GLM53_EXPERT_DOWN_DTYPE (a
        # low-precision pattern; the decode and prefill ops take each matrix's own format)
        down = os.environ.get("GLM53_EXPERT_DOWN_DTYPE")
        self.down_dtype = weight_dtype(down) if down else expert_dtype
        self.prog = PROG  # (the progressive form: BFP4 base and correction, made on the device from bf16, below)
        if self.prog:
            expert_dtype = self.down_dtype = ttnn.bfloat4_b
        upload_dtype = ttnn.bfloat16 if self.prog else expert_dtype
        # moe_prefill's kernels take BFP8 or BFP4 weight tiles (mp_gu5 and down); bf16 goes through moe_decode_experts
        low = (ttnn.bfloat8_b, ttnn.bfloat4_b)
        self.prefill_once = expert_dtype in low and self.down_dtype in low
        self.op_experts = config.num_experts // COLS  # a column's experts (72 c + j, j < 72)
        # the shared expert in the weights' slot 72 of every column, chosen in column `layer % 4` only
        self.shared_in_op = weights.shared.gate.shape[0] == SLICE * ROWS
        self.op_slots = self.op_experts + (1 if self.shared_in_op else 0)
        self.shared_col = getattr(weights, "layer", 0) % COLS
        self.layer = getattr(weights, "layer", None)  # (the error past the op's paths)
        self.w1_spec = None  # GLM53_W1's landing spec for this layer's decode calls (W1Landing.arm_model), else None
        # bf16 (the checkpoint's values): bf16 x bf16 products are exact and the matmul accumulates in fp32, as HF's
        # fp32 router; the fp32 copy took 57 us on 9 cores
        self.router_wt = put(mesh, weights.router.T.contiguous()[None, None], dims=(None, None))
        self.router_bias = put(mesh, weights.router_bias.view(1, 1, 1, -1), dims=(None, None), dtype=ttnn.float32)
        if weights.router_bias.is_meta and SYNTHETIC_ROUTING == "spread":
            ttnn.deallocate(self.router_bias)
            self.router_bias = zeros(mesh, [1, 1, 1, config.num_experts], dtype=ttnn.float32)
        # the routing's id range: the column's first expert (and its slot 1: this column's chips add the shared expert,
        # slot op_experts, weight 1, to every token)
        base = torch.zeros(ROWS, COLS, 1, 8, dtype=torch.int32)
        base[:, :, 0, 0] = torch.arange(COLS, dtype=torch.int32) * self.op_experts
        if self.shared_in_op:
            base[:, self.shared_col, 0, 1] = 1
        self.local_base = put(mesh, base, dims=(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        # the shared expert spread over the columns (moe_route.SHARED_SPLIT): column c's chips add it to part c of
        # COLS of the tokens (the base page's [1] 2, [2] the part, [3] the parts: kernels/route/rt_common.hpp)
        self.local_base_split = None
        if self.shared_in_op:
            split = base.clone()
            split[:, :, 0, 1] = 2
            split[:, :, 0, 2] = torch.arange(COLS, dtype=torch.int32)
            split[:, :, 0, 3] = COLS
            self.local_base_split = put(mesh, split, dims=(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._router_cfgs: dict = {}
        self.shared = TtMlp(mesh, weights.shared, config.swiglu_limit, dtype)

        # the dense copies, per chip: [1, 1, D, 72 256] (the column's experts' gates, columns 256 r .. of each, side by
        # side; the ups likewise) and [1, 1, 72 256, D] (their downs along K): the experts op's copies are made from
        # them on the device (`_init_experts_op`), then they are freed (~240 MB a layer in BFP8, ~10 GB a chip).
        # Random weights make them too (the seeds per tensor run in order)
        meta = weights.expert(0).gate.is_meta  # (`ShapeOnlyCheckpoint`: random values made on the device)
        # the layer's expert views made once for its 96 shards (common.UPLOAD_ONCE; they were made for each shard: 6,912
        # views of 72 experts' 3 tensors a layer, GIL-bound in the pool's threads)
        views = [weights.expert(e) for e in range(config.num_experts)] if common.UPLOAD_ONCE and not meta else None

        def column_weights(r, c):  # the chip's experts' weights, its first column
            span = range(c * self.op_experts, (c + 1) * self.op_experts)
            return ([views[e] for e in span] if views is not None else [weights.expert(e) for e in span]), r * SLICE

        down_dtype = upload_dtype if self.prog else self.down_dtype

        def packed(dtype):  # (PACK_SLICES: the slices straight into the dtype's tiles)
            return PACK_SLICES and not meta and dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)

        def side_by_side(name):
            direct = packed(upload_dtype)

            def shard(r, c, out=None):  # (out: a reused buffer, common.UPLOAD_SCRATCH)
                ws, s = column_weights(r, c)
                if direct:
                    return ttnn._ttnn.bfp_utils.pack_transposed_slices(
                        [getattr(w, name)[s : s + SLICE] for w in ws], 1, upload_dtype
                    )
                return torch.cat([getattr(w, name)[s : s + SLICE].T for w in ws], dim=1, out=out)[None, None]

            return shard

        def down(r, c, out=None):
            ws, s = column_weights(r, c)
            if packed(down_dtype):
                return ttnn._ttnn.bfp_utils.pack_transposed_slices(
                    [w.down[:, s : s + SLICE] for w in ws], 0, down_dtype
                )
            return torch.cat([w.down[:, s : s + SLICE].T for w in ws], dim=0, out=out)[None, None]

        hidden, wide = (
            config.hidden_size,
            self.op_experts * SLICE,
        )  # (the shards' shapes: [hidden, wide], [wide, hidden])
        gu_shape = None if packed(upload_dtype) else (hidden, wide)  # (the cut's reused buffers: none when packed)
        dn_shape = None if packed(down_dtype) else (wide, hidden)
        self.gate = upload_shards(mesh, side_by_side("gate"), dtype=upload_dtype, meta=meta, out_shape=gu_shape)
        self.up = upload_shards(mesh, side_by_side("up"), dtype=upload_dtype, meta=meta, out_shape=gu_shape)
        self.down = upload_shards(mesh, down, dtype=down_dtype, meta=meta, out_shape=dn_shape)
        views = None
        from_dense = not meta
        if self.shared_in_op and from_dense:  # the shared expert's slice of each chip, for the experts op's slot 72
            # only (the dense copies above hold the column's 72 experts)
            def shared_slice(name):
                def shard(r, c):
                    w = getattr(weights.shared, name)
                    piece = w[:, r * SLICE : (r + 1) * SLICE] if name == "down" else w[r * SLICE : (r + 1) * SLICE]
                    return piece.T.contiguous()[None, None]

                return shard

            self._shared_slices = [
                upload_shards(
                    mesh,
                    shared_slice(n),
                    dtype=upload_dtype if self.prog else (self.down_dtype if n == "down" else expert_dtype),
                    by="r",  # (the row's slice, the same on its 4 chips)
                    meta=False,
                )
                for n in ("gate", "up", "down")
            ]
        self._init_experts_op(mesh, config, expert_dtype, from_dense=from_dense)
        for t in (self.gate, self.up, self.down):
            ttnn.deallocate(t)
        self.gate = self.up = self.down = None

    def _init_experts_op(self, mesh, config: Glm53Config, expert_dtype, from_dense: bool) -> None:
        """The experts in moe_decode_experts' layout, by column (I = 256 of the column's 72 experts and the shared
        slot, E = op_slots; T = I / 32): w1 [T E D, 64], expert e's (gate, up) column tiles c at tile rows
        (T e + c) D / 32 + k, a block of 16 tile rows a shard (a column's 8 blocks in the 8 banks); w2 [32 E I, D / 32],
        down core q's columns of expert e at rows (E q + e) I + i, a down core's slice a shard.  DRAM ND-sharded, so
        each core reads from one bank, a block at a time (74 against 82 us an expert with a tile row a shard).
        From the dense copies on the device, whole tiles moved by slices and concats (BFP8 as it is; through row-major
        and a permute, 64-byte rows, it took ~70 s a layer in the model's load), or random values."""
        hidden = config.hidden_size
        self.inter = SLICE
        dram = mesh.dram_grid_size()

        def slices(height: int, width_tiles: int):
            return ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.DRAM,
                nd_shard_spec=ttnn.NdShardSpec(
                    shard_shape=ttnn.Shape([height, width_tiles * 32]),
                    grid=ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))]
                    ),
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

        def make(shape, height, width_tiles, dtype=expert_dtype):
            flat = synthetic(mesh, shape, dtype)
            out = ttnn.to_memory_config(flat, slices(height, width_tiles))
            ttnn.deallocate(flat)
            return out

        q = 32  # down cores
        n = self.op_slots
        sets = 2 if self.prog else 1  # (the progressive form: the bases' slots, then the corrections')
        if not from_dense:
            self.e_w1 = make([1, 1, SLICE // 32 * sets * n * hidden, 64], 16 * 32, 2)
            self.e_w2 = make(
                [1, 1, q * sets * n * SLICE, hidden // q], sets * n * SLICE, hidden // q // 32, self.down_dtype
            )
            return

        interleaved = ttnn.DRAM_MEMORY_CONFIG

        def cols(w, start, width):  # whole tiles of the dense copy, as they are
            return ttnn.slice(w, [0, 0, 0, start], [1, 1, w.shape[-2], start + width], memory_config=interleaved)

        def stack(parts):  # along the rows, at most 8 at a time, freeing the parts
            while len(parts) > 1:
                merged = []
                for i in range(0, len(parts), 8):
                    group = parts[i : i + 8]
                    if len(group) == 1:
                        merged.append(group[0])
                        continue
                    merged.append(ttnn.concat(group, dim=-2, memory_config=interleaved))
                    for t in group:
                        ttnn.deallocate(t)
                parts = merged
            return parts[0]

        def place(t, height, width_tiles):
            out = ttnn.to_memory_config(t, slices(height, width_tiles))
            ttnn.deallocate(t)
            return out

        # w1: for each slot e and column tile c of its 256, gate column tile c | up column tile c ([D, 64]);
        # w2: down core q's columns of all the slots' rows ([E 256, D / Q] each)
        # (the slot past the column's experts: the shared expert's slices)
        shared = getattr(self, "_shared_slices", None)
        from . import prog_split

        def empty(shape, height, width_tiles):  # (the layout's ND-sharded BFP4 tensor, unwritten)
            spec = ttnn.TensorSpec(
                ttnn.Shape(shape),
                ttnn.bfloat4_b,
                ttnn.TILE_LAYOUT,
                slices(height, width_tiles).nd_shard_spec,
                ttnn.BufferType.DRAM,
            )
            return ttnn.allocate_tensor_on_device(spec, mesh)

        if self.prog:  # the progressive form: the split writes the bases and corrections at their places in w1 / w2
            # (ttnn.concat along the width re-rounds BFP4 / BFP8 values); the layout below, sets 2
            t_tiles, d_tiles, wq = SLICE // 32, hidden // 32, hidden // q // 32
            self.e_w1 = empty([1, 1, t_tiles * 2 * n * hidden, 64], 16 * 32, 2)
            self.e_w2 = empty([1, 1, q * 2 * n * SLICE, hidden // q], 2 * n * SLICE, wq)
            w1 = dict(out=self.e_w1, mode=1, slots=n, d_tiles=d_tiles, corr_off=n * t_tiles * d_tiles * 2)
            w2 = dict(out=self.e_w2, mode=2, slots=n, d_tiles=wq, corr_off=n * t_tiles * wq)
            for half, t in ((0, self.gate), (1, self.up)):
                prog_split.split(mesh, t, e_off=0, half=half, **w1)
                if shared is not None:
                    prog_split.split(mesh, shared[half], e_off=self.op_experts, half=half, **w1)
            prog_split.split(mesh, self.down, e_off=0, **w2)
            if shared is not None:
                prog_split.split(mesh, shared[2], e_off=self.op_experts, **w2)
            if os.environ.get("GLM53_TEST_PROG_KEEP") == "1":  # (a test: the plain split copies kept for a check)
                bases, corrs = zip(*[_prog_split(t) for t in (self.gate, self.up, self.down)])
                sb, sc = zip(*[_prog_split(t) for t in shared]) if shared is not None else (None, None)
                self._prog_dense = [(*bases, sb), (*corrs, sc)]
            if shared is not None:
                for t in shared:
                    ttnn.deallocate(t)
                del self._shared_slices
            return
        dense = [(self.gate, self.up, self.down, shared)]
        # (ttnn.concat along the width re-rounds ~7 % of the BFP8 values with another face row's exponent: the experts'
        # w1 as it has always been made; a tile copy that kept the host's packing, GLM53_MOE_W1_COPY, was not taken and
        # was removed)
        pairs = []
        for gate_d, up_d, _, shared_d in dense:
            for e in range(self.op_slots):
                gate, up, e0 = (gate_d, up_d, e) if e < self.op_experts else (shared_d[0], shared_d[1], 0)
                for c in range(SLICE // 32):
                    g = cols(gate, e0 * SLICE + c * 32, 32)
                    u = cols(up, e0 * SLICE + c * 32, 32)
                    pairs.append(ttnn.concat([g, u], dim=-1, memory_config=interleaved))
                    ttnn.deallocate(g)
                    ttnn.deallocate(u)
        self.e_w1 = place(stack(pairs), 16 * 32, 2)
        width = hidden // q

        def core_rows(core):  # down core q's columns of every slot's rows (the progressive form: bases', corrections')
            parts = []
            for _, _, down_d, shared_d in dense:
                parts.append(cols(down_d, core * width, width))
                if self.op_slots > self.op_experts:
                    parts.append(cols(shared_d[2], core * width, width))
            if len(parts) == 1:
                return parts[0]
            out = ttnn.concat(parts, dim=-2, memory_config=interleaved)
            for t in parts:
                ttnn.deallocate(t)
            return out

        self.e_w2 = place(
            stack([core_rows(core) for core in range(q)]), len(dense) * self.op_slots * SLICE, width // 32
        )
        if shared is not None:
            for t in shared:
                ttnn.deallocate(t)
            del self._shared_slices

    def route(self, x_all):
        """[1, 1, N, D] bf16 -> dense routing weights [1, 1, N, E] fp32 (zero outside each token's top k), by ttnn ops:
        tests/test_moe_device.py's check of the routing (the model routes by `moe_route` and moe_decode_route).

        DeepSeek's fused `moe_grouped_topk` (one group) was tried in place of this: it chose another expert set for
        20-30 % of the tokens, 8th-9th gaps up to 1e-2 (its internal precision), so it is not used."""
        logits = self.router_logits(x_all)
        scores = ttnn.sigmoid(logits)
        choice = ttnn.add(scores, self.router_bias)
        # (padding the 288 scores to 1024 for ttnn's multi-core topk saved 5 us at 8 tokens and cost 160 us at 256)
        top, _ = ttnn.topk(choice, k=self.k, dim=-1, largest=True, sorted=True)
        n = choice.shape[-2]
        kth = ttnn.slice(top, [0, 0, 0, self.k - 1], [1, 1, n, self.k])
        chosen = ttnn.multiply(scores, ttnn.ge(choice, kth))
        total = ttnn.add(ttnn.sum(chosen, dim=-1, keepdim=True), 1e-20)
        return ttnn.multiply(ttnn.multiply(chosen, ttnn.reciprocal(total)), self.scale)

    def router_logits(self, x_all):
        """[1, 1, N, D] bf16 -> [1, 1, N, E] fp32."""
        mm = dict(dtype=ttnn.float32, compute_kernel_config=HIFI4_FP32)
        if x_all.shape[-2] <= ROUTER_CONFIG_ROWS:
            mm["program_config"] = self._router_config(x_all.shape[-2])
        return ttnn.matmul(x_all, self.router_wt, **mm)

    def _router_config(self, n: int):
        """K in blocks of 16 tiles, each output column tile (9) on its own core; the in0 rows multicast (1D) for one
        tile row of tokens, else a column of cores per tile row (2D)."""
        if n not in self._router_cfgs:
            m, e, w = -(-n // 32), self.router_wt.shape[-1] // 32, 16
            if m == 1:
                cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(e, 1),
                    in0_block_w=w,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    out_block_h=1,
                    out_block_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            else:
                rows = max(y for y in range(1, self.mesh.compute_with_storage_grid_size().y + 1) if m % y == 0)
                cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(e, rows),
                    in0_block_w=w,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    out_block_h=m // rows,
                    out_block_w=1,
                    per_core_M=m // rows,
                    per_core_N=1,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )
            self._router_cfgs[n] = cfg
        return self._router_cfgs[n]

    def __call__(self, x, real=None, picks=None):
        """x [1, 1, T, D] bf16, the same on a row's 4 chips (each row its own T tokens) -> the same layout.

        `real` (prefill: `KdaPicks.real`, [1, T, 1] per row, or None): the padding tokens' routed pairs are left out
        (`_routed_once_local`'s path); `picks` (`KdaPicks`, with `real`): where the gathered mask is kept for the
        chunk's other layers.  More than EXPERTS_OP_COLUMN_MAX tokens of the mesh (chunks over 1,024 a row): an
        error."""
        t = x.shape[-2]
        if (
            self.prefill_once
            and t >= CCL_PAD_ROWS
            and t % 32 == 0
            and ROWS * t > EXPERTS_OP_CALL
            and ROWS * t <= EXPERTS_OP_COLUMN_MAX
        ):
            from . import col_row_rs

            b = col_row_rs.batches_for(self.mesh, t) if COL_ROW_RS else 1
            if COL_ROW_RS and b == 1 and t in col_row_rs.TS and ("missing", t) not in _CRR_SEEN:
                _CRR_SEEN.add(("missing", t))  # (once a size: a measurement must not take the old path unknowingly)
                print(
                    f"WARNING moe: no col_row_rs buffers for {t} tokens a row (Glm53Generator or col_row_rs.prealloc "
                    "takes them before any trace): the way back as the ttnn ops (the same values, slower)",
                    flush=True,
                )
            # (col_row_rs.CARRY: the combine carried in the way back's op, `_finish_crr`)
            carry = col_row_rs.carry_for(self.mesh, t, self._crr_ag(t)) if b > 1 and COL_ROW_RS else None
            partial = self._routed_once_local(x, real, picks, batches=b, carry=carry)
            if b > 1:
                return self._finish_crr(x, partial, t)
            return self._finish(x, partial, t)
        if self._h1_fits(x):
            shared = self._shared_mode(ROWS * t, ROWS * t) == 0  # (else the experts op computes it)
            w1 = self.w1_spec if W1_DECODE and self.w1_spec else []  # (a decode step's only)
            if MOE_COL_RS and t in COL_RS_TOKENS:  # (the way back in the op)
                rows = self._routed_h1(x, col_rs=True, w1=w1, share8=not shared)  # (GLM53_AR_BFP8: not before an add)
                return self._finish_rows(x, rows, shared=shared)
            return self._finish(x, self._routed_h1(x, w1=w1), t, shared=shared)
        x_all = self._gather_rows(x)  # [1, 1, N, D]: the 8 T tokens (then zero tokens up to a tile, `_row_picks`)
        n = x_all.shape[-2]
        if n % 32 == 0 and n <= EXPERTS_OP_COLUMN_MAX:
            shared = self._shared_mode(n, ROWS * t) == 0  # (else the experts op computes it)
            return self._finish(x, self._routed_sliced(x_all, ROWS * t), t, shared=shared)
        raise RuntimeError(
            f"layer {self.layer}: {t} tokens a row, {n} of the mesh: past the experts op's paths (at most "
            f"{EXPERTS_OP_COLUMN_MAX} tokens of the mesh)"
        )

    def _h1_fits(self, x) -> bool:
        """Whether this call takes H1 (`_routed_h1`): a decode step's route_only, H1_TOKENS tokens a mesh row, the
        column layout."""
        t = x.shape[-2]
        return (
            H1
            and t in H1_TOKENS
            and ROWS == 8
            and ROWS * t <= EXPERTS_OP_CALL
            and x.dtype == ttnn.bfloat16
            and moe_route.fits(self, t)
            and moe_route.fits(self, ROWS * t)
        )

    def _routed_h1(self, x, col_rs: bool = False, w1=None, share8: bool = False):
        """The row's [1, 1, t, D] -> this chip's partial sums of the column's 8 t tokens, as `_routed_sliced` gives them
        from `_gather_rows`: the router and route_only on the row's t tokens (its rows into the column's landing w_all),
        then the experts op on the landing x_all and w_all, which it fills over the fabric (H1).  `col_rs` (MOE_COL_RS):
        the row's t tokens summed over the column [1, 1, t, D] (the way back in the op)."""
        t = x.shape[-2]
        rs = col_rs_sems(self.mesh, x.shape[-1]) if col_rs else (0, 0)
        # (the way back's output straight into the row all-reduce's L1 shards, as the reduce-scatter's: MOE_RS_L1)
        rs_cfg = ar_input_config(self.mesh, t, x.shape[-1]) if col_rs and MOE_RS_L1 else None
        x_all, w_all, sem_x, sem_r = h1_landing(self.mesh, t, x.shape[-1])
        from .router_mm import (  # (the router and the routing in one program, as router_logits and route_only)
            router_route,
        )

        x_row, x8 = x, x_bfp8_mode() if t >= 32 else 0  # (GLM53_MOE_X_BFP8: the experts op's x in BFP8_b)
        if x8 == 1:
            spec = ttnn.TensorSpec(
                ttnn.Shape(list(x.shape)), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
            )
            x_row = ttnn.allocate_tensor_on_device(spec, self.mesh)
        with w1_dense.host("router"):  # (GLM53_W1D: riders for the next layer's x_proj)
            router_route(self.mesh, self, x, t, h1=w_all, x8=x_row if x8 == 1 else None)
        if x8 == 2:
            x_row = ttnn.typecast(x, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.experimental.deepseek_prefill.moe_decode_experts(
            x_all,
            w_all,
            self.e_w1,
            self.e_w2,
            self.op_slots,
            SLICE,
            float(self.limit),
            row_tokens=t if t in (8, 16) else 0,  # (as `_routed_sliced`: the padded y of 8 / 16 tokens a row)
            x_row=x_row,
            h1_sem_x=sem_x,
            h1_sem_r=sem_r,
            rs_sem=rs[0],
            rs_mail=rs[1],
            rs_memory_config=rs_cfg,
            w1_recv=list(w1) if w1 else [],
            rs_dtype=ar_input_dtype(t) if col_rs and share8 else None,  # (GLM53_AR_BFP8: the row AR's share in BFP8_b)
        )
        if x_row is not x:
            ttnn.deallocate(x_row)
        return out

    def _shared_in_op(self, n: int) -> bool:
        """Whether the experts op computes the shared expert for these n tokens of the mesh (SHARED_IN_OP_MAX)."""
        return self.shared_in_op and n <= SHARED_IN_OP_MAX

    def _shared_mode(self, n: int, tokens: int) -> int:
        """Where the shared expert goes for n tokens of the mesh (`tokens` of them real): 1 the experts op of column
        `shared_col`, every token (SHARED_IN_OP_MAX); 2 the experts op of every column, a quarter of the tokens each
        (moe_route.SHARED_SPLIT: a decode step's route_only, SHARED_SPLIT_MIN tokens or more); 0 the TP4 shared MLP."""
        if not self.shared_in_op:
            return 0
        if n <= SHARED_IN_OP_MAX:
            return 1
        split = moe_route.SHARED_SPLIT and n >= moe_route.SHARED_SPLIT_MIN and moe_route.fits(self, tokens)
        return 2 if split else 0

    def _finish(self, x, partial, t: int, shared: bool = True):
        """This chip's experts' [1, 1, N, D] partial sums -> the layer's output: the row's tokens summed over the rows,
        plus the shared expert (unless the partial sums have it: SHARED_IN_OP_MAX), summed over the row's chips."""
        if not shared:
            with w1_dense.host("moe_ar"):  # (GLM53_W1D: riders for the next layer's x_proj)
                out = self._scatter_ar(partial, t)
            if out is not None:
                return out
        # the reduce-scatter writes straight into the row all-reduce's L1 shards when nothing comes between them
        # (MOE_RS_L1, `ar_input_config`)
        cfg = ar_input_config(self.mesh, t, partial.shape[-1]) if MOE_RS_L1 and not shared else None
        rows = self._scatter_rows(partial, t, cfg)  # [1, 1, T, D]: this row's tokens
        if shared:
            part = self.shared.partial(x)
            total = ttnn.add(rows, part)
            ttnn.deallocate(rows)
            ttnn.deallocate(part)
        else:
            total = rows
        out = tp_all_reduce(self.mesh, total)
        ttnn.deallocate(total)
        return out

    @staticmethod
    def _crr_ag(t: int) -> bool:
        """Whether the way back's op ends with the row's all-gather at t tokens a row (mHC over the row by tokens,
        common.ROW_SPLIT: this chip's quarter by the op's reduce-scatter and an all-to-all, ROW_SPLIT_MOE_A2A; else the
        op's all-gather and the quarter taken by the caller)."""
        return not (ROW_SPLIT_MOE_A2A and common.ROW_SPLIT_A2A and common.quarter_of_rows(t))

    def _finish_crr(self, x, partial, t: int):
        """`_finish` (shared) as one op (`col_row_rs.run`): partial [1, 1, 8 t, D] batch-major (or the combine's
        `moe_prefill.Carried` form, col_row_rs.CARRY: the combine then in the op's program, the shared expert's
        partial made before it) -> the layer's output."""
        from . import col_row_rs, moe_prefill

        carried = isinstance(partial, moe_prefill.Carried)
        if (t, carried) not in _CRR_SEEN:
            _CRR_SEEN.add((t, carried))
            how = " with the combine in its program" if carried else ""
            print(f"moe: the way back at {t} tokens a row as one op (col_row_rs){how}", flush=True)
        part = self.shared.partial(x)
        ag = self._crr_ag(t)
        if not ag and ("a2a", t) not in _CRR_SEEN:
            _CRR_SEEN.add(("a2a", t))
            print(f"moe: the way back at {t} tokens a row without the row all-gather (an all-to-all)", flush=True)
        out = col_row_rs.run(self.mesh, partial, part, t, ag=ag)
        ttnn.deallocate(part)
        if carried:
            for z in list(partial.tensors) + [partial.out]:
                ttnn.deallocate(z)
        else:
            ttnn.deallocate(partial)
        return out

    def _finish_rows(self, x, rows, shared: bool = True):
        """The row's tokens summed over the column [1, 1, t, D] (the way back in the experts op) -> the layer's output:
        plus the shared expert's TP4 partial (unless the experts op made it), summed over the row's chips, as `_finish`.
        """
        if shared:
            part = self.shared.partial(x)
            total = ttnn.add(rows, part)
            ttnn.deallocate(rows)
            ttnn.deallocate(part)
        else:
            total = rows
        with w1_dense.host("moe_ar"):  # (GLM53_W1D: riders for the next layer's x_proj)
            out = tp_all_reduce(self.mesh, total)
        ttnn.deallocate(total)
        return out

    def _scatter_ar(self, partial, t: int):
        """`_scatter_rows` and `tp_all_reduce` in one op (`col_collectives.col_scatter_ar`) for the few tokens a row
        a one-shot column scatter takes (t <= 4, and the experts op's padded y at t = 6 / 8), else None."""
        from .common import TP_AR_CUSTOM, TP_AR_CUSTOM_MAX

        d = partial.shape[-1]
        if partial.dtype != ttnn.bfloat16 or ROWS != 8 or not t < CCL_PAD_ROWS:
            return None
        plain = MOE_COL_CUSTOM and t <= 4 and partial.shape[-2] == 32
        plain = plain and TP_AR_CUSTOM and t <= TP_AR_CUSTOM_MAX and d % (32 * 8) == 0
        padded = partial.shape[-2] == ROWS * CCL_PAD_ROWS == ROWS * 32 and t % 2 == 0 and 4 < t <= 8
        padded = padded and t <= MOE_COL_SCATTER_PADDED and d % (32 * 32) == 0
        if not (plain or padded):
            return None
        from .col_collectives import col_scatter_ar

        out = col_scatter_ar(self.mesh, partial, t)  # [1, 1, t, D]
        ttnn.deallocate(partial)
        return out

    def _picks(self, t: int):
        """`_row_picks` for t tokens per row when the row collectives pad them (the 0/1 matmuls move them), else None."""
        return _row_picks(self.mesh, t) if t < CCL_PAD_ROWS else None

    def _gather_rows(self, x):
        """[1, 1, T, D] per row -> [1, 1, N, D]: every row's tokens (8 T), padded to CCL_PAD_ROWS rows for the
        transfer; with the 0/1 matmuls (`_row_picks`) N is 8 T rounded up to a tile, the rest zero tokens."""
        t, d = x.shape[-2], x.shape[-1]
        if not t < CCL_PAD_ROWS:
            return ttnn.all_gather(x, dim=2, cluster_axis=ROW_AXIS)
        if t <= COL_GATHER_MAX and ROWS == 8 and x.dtype == ttnn.bfloat16 and d % (32 * 32) == 0:
            from .col_collectives import col_gather  # (it imports common)

            return col_gather(self.mesh, x)  # [1, 1, 8 t up to a tile, D]
        # pad within the tile row returns a tensor on x's own buffer: never deallocate it here
        padded = ttnn.all_gather(
            ttnn.pad(x, [(0, 0), (0, 0), (0, CCL_PAD_ROWS - t), (0, 0)], 0.0), dim=2, cluster_axis=ROW_AXIS
        )
        x_all = ttnn.matmul(self._picks(t)[0], padded, compute_kernel_config=HIFI4_FP32)  # exact copies
        ttnn.deallocate(padded)
        return x_all

    def _scatter_rows(self, partial, t: int, memory_config=None):
        """[1, 1, N, D] partial sums (N as `_gather_rows` gives) -> [1, 1, T, D]: this row's tokens summed over the
        rows (padded as `_gather_rows` pads); `memory_config`: where a whole tile row's reduce-scatter writes them."""
        d = partial.shape[-1]
        if not t < CCL_PAD_ROWS:
            rows = ttnn.reduce_scatter(
                partial, dim=2, cluster_axis=ROW_AXIS, memory_config=memory_config, **_rs_knobs(t)
            )
            ttnn.deallocate(partial)
            return rows
        if partial.shape[-2] == ROWS * CCL_PAD_ROWS and ROWS * t < partial.shape[-2]:
            padded = partial  # already by mesh row (the experts op's row_tokens)
        else:
            padded = ttnn.matmul(self._picks(t)[1], partial, compute_kernel_config=HIFI4_FP32)  # exact copies
        rows = ttnn.reduce_scatter(padded, dim=2, cluster_axis=ROW_AXIS)
        ttnn.deallocate(partial)
        if padded is not partial:
            ttnn.deallocate(padded)
        return ttnn.slice(rows, [0, 0, 0, 0], [1, 1, t, d])  # may be a view of rows: rows is not freed

    def _routed_sliced(self, x_all, tokens: int):
        """[1, 1, N, D] (N a multiple of 32 up to EXPERTS_OP_COLUMN_MAX; rows from `tokens` on choose nothing) -> this
        chip's columns of every chosen expert (256 of each of the column's), weighted and summed, [1, 1, N, D]:
        moe_decode_experts with I = SLICE on the routing's chosen ids and weights (the column's 72 from its first one),
        up to EXPERTS_OP_CALL tokens a call (the packed units; more: several calls, each reading the weights of its own
        rows' experts), or moe_prefill.experts past that (`prefill_once`)."""
        logits = self.router_logits(x_all)
        if moe_route.fits(self, tokens):  # (a decode step: the same rows, faster, moe_route.py)
            w = moe_route.route_only(self.mesh, self, logits, tokens)
        else:
            w = ttnn.experimental.deepseek_prefill.moe_decode_route(
                logits,
                self.router_bias,
                self.local_base,
                tokens,
                self.k,
                self.op_experts,
                float(self.scale),
                shared=self._shared_mode(x_all.shape[-2], tokens) == 1,
            )
        ttnn.deallocate(logits)
        run = lambda xs, ws, rt=0: ttnn.experimental.deepseek_prefill.moe_decode_experts(  # noqa: E731
            xs, ws, self.e_w1, self.e_w2, self.op_slots, SLICE, float(self.limit), row_tokens=rt
        )
        n, d = x_all.shape[-2], x_all.shape[-1]
        call = EXPERTS_OP_CALL
        t = tokens // ROWS
        if self.prefill_once and n > call:
            from . import moe_prefill

            partial = moe_prefill.experts(self.mesh, x_all, w, self.e_w1, self.e_w2, self.op_slots, float(self.limit))
            ttnn.deallocate(w)
            ttnn.deallocate(x_all)
            return partial
        if n <= call and t in (8, 16) and n == tokens and self._picks(t):
            partial = run(x_all, w, t)  # [1, 1, 8 x 32, D]: _scatter_rows' padded layout
        elif n <= call:
            partial = run(x_all, w)
        else:
            parts = []
            for r0 in range(0, n, call):
                r1 = min(n, r0 + call)
                xs = ttnn.slice(x_all, [0, 0, r0, 0], [1, 1, r1, d])
                ws = ttnn.slice(w, [0, 0, r0, 0], [1, 1, r1, 32])
                parts.append(run(xs, ws))
                ttnn.deallocate(xs)
                ttnn.deallocate(ws)
            partial = ttnn.concat(parts, dim=2)
            for part in parts:
                ttnn.deallocate(part)
        ttnn.deallocate(w)
        ttnn.deallocate(x_all)
        return partial

    def _routed_once_local(self, x, real=None, picks=None, batches: int = 1, carry=None):
        """The row's [1, 1, T, D] -> this chip's partial sums of the column's 8 T tokens, as
        `_routed_sliced` gives them through moe_prefill.experts, the routing made on the row's own tokens; `real`
        ([1, T, 1] per row): the tokens to give pairs (gathered over the column with the routing); `carry`
        (col_row_rs.carry_for): the combine not run but returned (`moe_prefill.Carried`)."""
        from . import moe_prefill

        t = x.shape[-2]
        logits = self.router_logits(x)
        if PREFILL_ROUTE_ONLY and moe_route.fits_prefill(self, t):
            w = moe_route.route_only(self.mesh, self, logits, t)  # (no shared expert: `_shared_mode` 0 here)
        else:
            w = ttnn.experimental.deepseek_prefill.moe_decode_route(
                logits, self.router_bias, self.local_base, t, self.k, self.op_experts, float(self.scale), shared=False
            )
        ttnn.deallocate(logits)
        w_all = ttnn.all_gather(w, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8 T, 32]: row r's at r T
        ttnn.deallocate(w)
        mask, shared = None, picks is not None
        if real is not None:  # (before the x gather: the plan beside it needs it)
            gather = lambda: ttnn.all_gather(
                ttnn.reshape(real, [1, 1, t, 1]), dim=2, cluster_axis=ROW_AXIS
            )  # noqa: E731
            mask = picks.derived("moe_real_all", gather) if shared else gather()  # [1, 1, 8 T, 1]
        flat = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        from . import x2d_gather

        x2d = X2D_GATHER and x2d_gather.fits(self.mesh, flat)
        planned = None
        if x2d:  # (its persistent landing: not deallocated)
            if t not in _X2D_SEEN:
                _X2D_SEEN.add(t)
                print(f"moe: the x all-gather at {t} tokens a row in two dimensions (x2d_gather)", flush=True)
            if PLAN_WITH_X2D:  # (the plan's counts now, its push in the gather's program on the cores it leaves)
                busy = x2d_gather.busy_cores(self.mesh, flat.shape[-1])
                *planned, push = moe_prefill.plan(self.mesh, w_all, mask, avoid=busy, defer_push=True)
                x_rm = x2d_gather.x2d_gather(self.mesh, flat, extra=push)
                push.done()
            else:
                x_rm = x2d_gather.x2d_gather(self.mesh, flat)
        else:
            x_rm = ttnn.all_gather(flat, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8 T, D] row-major
        ttnn.deallocate(flat)
        partial = moe_prefill.experts(
            self.mesh,
            x_rm,
            w_all,
            self.e_w1,
            self.e_w2,
            self.op_slots,
            float(self.limit),
            row_major=True,
            mask=mask,
            batches=batches,
            planned=planned,
            carry=carry,
        )
        if mask is not None and not shared:
            ttnn.deallocate(mask)
        ttnn.deallocate(w_all)
        if not x2d:
            ttnn.deallocate(x_rm)
        return partial
