"""KDA on the 8x4 mesh: one-token decode (ttnn.experimental.kda's decode ops) and chunked prefill (its prefill ops).

Tensor parallel over a row's 4 chips by head: 16 of the 64 heads per chip.  q/k/v, f_b, g_b, b_proj, dt_bias and
A_log are split by head; f_a and g_a (4096 -> 128) are replicated; o_proj is split by input feature and its partial
outputs are all-reduced over the row.  Per chip and user the state is the last 3 inputs of the width-4 conv (fp32,
[1, 1, B, 6144], channels (q | k | v, head, d)) and the delta-rule state [B, 16, 128, 128] (bf16).  A
prefill runs one request per mesh row (a "wave", each row's request going to its own user) and carries it from chunk
to chunk in a wave state per row (fp32); the wave's last chunk writes each row's into its user (`prefill`).

One step, as `tt/kda.py` (`kda_recurrent`):
    conv + SiLU in fp32 -> q, k, v (rounded to bf16); q, k L2-normalized in fp32, q scaled by 1/sqrt(128)
    g = -5 sigmoid(exp(A_log) (f_b(f_a(x)) + dt_bias)); S <- S * exp(g) (per key row)
    S <- S + k^T (beta (v - k S));  o = q S
    o -> RMS norm gated by sigmoid(g_b(g_a(x))) -> o_proj
Decode takes three ops a layer after the x projection (one matmul, per chip [q | k | v of its 16 heads (6144), b_proj
(16, padded to 32), f_a (128), g_a (128)]) and f_b, g_b (one block-diagonal matmul): `decode_prepare` (the conv, SiLU,
the L2 norms, the decay and beta, the history shift; the per-head vectors as rows [1, 1, B * 16, 128]),
`recurrent_decode_step` (the delta rule, the state read and written once, its update in fp32) and `decode_finish` (o
rounded to bf16, the gated RMS norm, in o_proj's layout).  They replaced a composite of ttnn ops (107 programs a
layer at first).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple

import torch

import ttnn
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tt.kda import KdaWeights
from models.demos.blackhole.glm53_flash.ttnn import prefill_tables
from models.demos.blackhole.glm53_flash.ttnn.common import (
    COLS,
    HIFI4_FP32,
    ROW_AXIS,
    ROWS,
    WEIGHT_MM,
    ar_input_config,
    const_tensor,
    const_zeros,
    decode_linear,
    put,
    sp_first,
    tp_all_reduce,
    tp_weight,
    upload_shards,
    wave_linear,
    zeros,
)

BETA_COLS = 32  # b_proj's 16 outputs per chip, padded to a tile
# The delta-rule states (`recurrent`) are bf16: recurrent_decode_step reads and writes bf16 tiles, its update in fp32
# (Dest), S' rounded to bf16 once a step (the DRAM traffic halved).  The prefill's wave state stays fp32, rounded into
# its user's state once (`_commit`).  (The fp32 states, GLM53_KDA_STATE_BF16=0, the earlier form, were removed.)
# The plain step's recurrent_decode_step form: 384 (128 a pair's value blocks in a per-core rotated order, 256 the top
# rows' reads on NoC 1; op alone the better of the forms at every shape; a long traced stress clean) + 1024, its units
# over the cores in proportion to their grid row's speed and the top 5 rows' reads on NoC 1 in place of 3
# (`recurrent_decode_step_program_factory.cpp` STEP_ROW_PCT_NOC1: the same values; with the prepare's `row_split` (the
# top 6 grid rows' reads on NoC 1 and the units by grid row speed) the plain step at 512 / 256 users 60.89 / 46.87 ->
# 56.76 / 45.36 ms; long traced stress runs clean)
PLAIN_FORM = 384 | 1024
# A verify step of K drafts (MTP's K = 1 / 2 and DFlash2's, `TtKda.step_spec`) in the chunked form
# (recurrent_decode_step's `saved`): decode_prepare and recurrent_decode_step for the T = K + 1 tokens a user; the
# state holds the start state of the last step and `spec_rows` the step's T tokens' rows, from which a user's accepted
# tokens (all T when all are accepted) are redone at the start of the next step's chunk; the step writes its own start
# over the state (one state written a step).  The prefill's commit puts the request's rows to do nothing
# (`_commit`).  So `recurrent` is the start of the last verify step, not the state after its tokens.  decode_prepare's
# outputs by pair (`pair_major`: the pair's T tokens one run of rows, row (u H + h) T + k).  (The checkpoint form,
# GLM53_KDA_VERIFY_CHUNK=0, and MTP's own two- and three-token ops, GLM53_KDA_VERIFY_GENERAL_FROM=3, were removed.)
# Its hand-offs in L1 interleaved: decode_prepare's outputs (`l1_outputs`) and the delta rule's o (read by
# decode_finish next), in place of DRAM (the same bits; with PREP_SPLIT below the verify step 8 / 32 / 128 users
# -0.15 / -0.40 / -1.71 ms, a 2,000-step stress clean).  Tests switch it: False DRAM
HANDOFF_L1 = True
# ... its decode_prepare over twice the cores (`split_channels`: q / k / v's units by channel halves; the same
# values).  Tests switch it: False a unit a core's share
PREP_SPLIT = True


def _conv_block(t: int) -> int:
    """The prefill conv's channel blocks (`qkv_causal_conv1d_silu`'s channel_chunk_size): any block gives the same
    values (a channel's taps and rows only); 768 left 8 of the 120 cores two work items at a chunk of 512 (128 items)
    and used 32 cores at 128; 32: the conv 128.8 -> 54.9 us at 512 tokens.  64 from 1,024 tokens (with the mask
    folded in 121.5 -> 108.2 us at 1,024, the same at 512), else 32."""
    return 64 if t >= 1024 else 32


def chunk_slot_floats(tokens: int) -> int:
    """The floats of a unit's slot of the chunked form's saved rows (a batch's in its first unit's: k and the decays,
    4 tiles x 2 faces x T rows of 64 bytes each, beta 64 bytes, the batch's up to 4 blocks' v, 2 faces x T rows)."""
    return (1536 * tokens + 64) // 4


def chunk_slot_tiles(tokens: int) -> int:
    """The fp32 tiles of a unit's slot (its own whole tiles: the op reads and writes their bytes as they are)."""
    return (chunk_slot_floats(tokens) + 1023) // 1024


def chunk_saved_host(users: int, heads: int, tokens: int) -> torch.Tensor:
    """Saved rows that do nothing (a fresh state's, and a new request's after its prefill: the decays 1, the rest 0,
    so redoing them is the identity) for `users` users of `heads` pairs, [users, 1, heads 4 nt 32, 32] fp32 for the TILE
    layout (nt = chunk_slot_tiles: unit u's slot in tiles u nt ..): each slot's floats as the tiles' bytes in order, so
    laid out here as their faces (16 x 16, row-major, faces 0 .. 3 a tile) for the tilizing to put them back."""
    nt = chunk_slot_tiles(tokens)
    units = users * heads * 4
    raw = torch.zeros(units, nt * 1024)
    raw[:, 128 * tokens : 256 * tokens] = 1.0
    # raw tile [face row-block fr, face column-block fc, row in face, column in face] -> rows fr 16 + r, columns
    # fc 16 + c
    tiles = raw.reshape(units * nt, 2, 2, 16, 16).permute(0, 1, 3, 2, 4).reshape(units * nt * 32, 32)
    return tiles.reshape(users, 1, heads * 4 * nt * 32, 32)


# the sequence-parallel prefill (`_prefill_sp`) gathers every row's last 3 inputs over the mesh column by the one-shot
# fabric gather (`col_collectives.col_gather`) with this many workers for the 6,144 conv channels (a 1,152-byte slice
# each)
SP_GATHER_WORKERS = 32


@dataclass
class TtKdaState:
    history: list  # the last 3 conv inputs, oldest first, each [1, 1, B, 6144] fp32 (updated in place)
    recurrent: object  # [B, 16, 128, 128] bf16 (updated in place; with a verify state, the start of its last step)
    wave_history: object  # prefill: the mesh row's request's last 3 conv inputs so far, [1, 3, 6144] bf16
    wave_state: object  # prefill: its delta-rule state so far, [16, 128, 128] fp32
    # a verify step of K drafts (`TtKda.step_spec`): the conv history's K checkpoint rows, [1, K, B, 6144] fp32; the
    # step's tokens' rows, [B, 1, 16 4 nt 32, 32] fp32 TILE (chunk_saved_host), and one user's rows doing nothing [1, 1,
    # 16 4 nt 32, 32] (the prefill's commit puts them in the request's user)
    spec_saved: object = None
    spec_rows: object = None
    spec_inert: object = None
    # ... as a ping-pong (spec_pingpong()): the other set of (history, recurrent, spec_saved, spec_rows), a SpecSet.  A
    # verify step reads the fields above and writes this set; `spec_flip` then swaps them (the caller, after each step
    # run or launched); taking the step back is a flip back, and the step run again reads what it read the first time
    # (bit for bit the same).  Everything else (the prefill's commit, ...) takes the fields above: the set the next step
    # reads
    spec_alt: object = None
    spec_parity: int = 0  # (the ping-pong) which of the two sets the fields hold: 0 the first made, 1 the other
    # (the ping-pong) the last verify step issued (`TtGlm53Model.kda_spec_launched`) may still be taken back: no
    # prefill's commit since (`_commit`, the model's replay_commit); depth 1
    spec_pending: bool = False

    def deallocate(self):
        for t in [*self.history, self.recurrent, self.wave_history, self.wave_state]:
            ttnn.deallocate(t)
        for t in (self.spec_saved, self.spec_rows, self.spec_inert):
            if t is not None:
                ttnn.deallocate(t)
        if self.spec_alt is not None:
            alt = self.spec_alt
            for t in [*alt.history, alt.recurrent, alt.spec_saved, alt.spec_rows]:
                ttnn.deallocate(t)


@dataclass
class SpecSet:
    """A ping-pong's other set of a KDA layer's verify state (TtKdaState.spec_alt)."""

    history: list
    recurrent: object
    spec_saved: object
    spec_rows: object


def spec_pingpong() -> bool:
    """Whether new verify states are made as a ping-pong of two sets
    (GLM53_KDA_SPEC_PINGPONG=1; read at each empty_state: tests switch it).  Off: one set, in place."""
    return os.environ.get("GLM53_KDA_SPEC_PINGPONG", "0") == "1"


def spec_flip(state: TtKdaState) -> None:
    """A ping-pong's sets swapped (TtKdaState.spec_alt): after a verify step (run or launched) the set it wrote becomes
    the one the next step reads; again, to take that step back.  No device work: Python references only."""
    alt = state.spec_alt
    if alt is None:
        return
    state.history, alt.history = alt.history, state.history
    state.recurrent, alt.recurrent = alt.recurrent, state.recurrent
    state.spec_saved, alt.spec_saved = alt.spec_saved, state.spec_saved
    state.spec_rows, alt.spec_rows = alt.spec_rows, state.spec_rows
    state.spec_parity ^= 1


@dataclass
class SpecKeep:
    """Per user of a row, how many of the last verify step's K drafts were accepted (0 .. K: the state after the last
    accepted token goes on), written by the host before each step and shared by the KDA layers (`TtKda.step_spec`)."""

    flags: object  # [1, B] uint32 row-major: the decode ops' `keep`

    @staticmethod
    def host(keep: torch.Tensor) -> dict:
        """keep [ROWS, B] (the drafts accepted) -> the host tensors of the fields."""
        rows, b = keep.shape
        return {"flags": keep.to(torch.int32).reshape(rows, b)}


@dataclass
class KdaPicks:
    """A prefill chunk's inputs made once per chunk and shared by the KDA layers (`TtKda.prefill_picks`)."""

    history_pick: object  # [1, 3, 32 + T] bf16 per mesh row (`TtKda._last_inputs`)
    real: object  # [1, T, 1] per row: 1 on the real tokens; None when every row is full (eager)
    # a wave's last chunk (else None): per row the one-hot of its user, [1, 1, B, 1] fp32 (zeros on a row without a
    # request), and either (eager, and a captured commit apart from the chunk: `commit_picks`) that user as a page table
    # [1, 1] int32 (-1 without a request: nothing written), or (a traced chunk: the same programs for any rows) the
    # one-hot again as [B, 1, 1, 1] fp32
    commit_rows: object = None
    commit_table: object = None
    commit_users: object = None
    # per row 0 where it computes the chunk, else -1 ([1, 1] int32): the wave state's paged copy (`prefill`)
    wave_page: object = None
    all_kept: bool = False  # every row with a request computes this chunk (none has ended)
    owned: bool = True  # new tensors, deallocated after the chunk (not the fixed buffers of a trace)
    sp: "KdaSpPicks | None" = None  # one request over the mesh rows (`TtKda.sp_picks`)
    # made from the fields above on first use within a chunk and shared by its layers (`derived`), freed at the chunk's
    # end (`release_derived`)
    extra: dict | None = None

    def derived(self, name: str, make):
        """A tensor made from the picks once a chunk (the first layer that asks makes it; inside a captured chunk too,
        freed at its end by `release_derived`)."""
        if self.extra is None:
            self.extra = {}
        if name not in self.extra:
            self.extra[name] = make()
        return self.extra[name]

    def release_derived(self):
        for t in (self.extra or {}).values():
            if isinstance(t, (tuple, list)):
                for x in t:
                    ttnn.deallocate(x)
            else:
                ttnn.deallocate(t)
        self.extra = None

    def history_parts(self):
        """`history_pick` split at its column 32: (the history block's pick [1, 3, 32], the chunk's [1, 3, T])."""

        def make():
            t = self.history_pick.shape[-1]
            k = self.history_pick.shape[-2]
            head = ttnn.slice(self.history_pick, [0, 0, 0], [1, k, 32])
            inputs = ttnn.slice(self.history_pick, [0, 0, 32], [1, k, t])
            return head, inputs

        return self.derived("history_parts", make)

    def deallocate(self):
        self.release_derived()
        if self.owned:
            for t in (
                self.history_pick,
                self.real,
                self.commit_rows,
                self.commit_table,
                self.commit_users,
                self.wave_page,
                *(self.sp or ()),
            ):
                if t is not None:
                    ttnn.deallocate(t)


class KdaSpPicks(NamedTuple):
    """A sequence-parallel chunk's KDA inputs (`TtKda.sp_picks`): requests' consecutive segments on the mesh rows,
    each request on its own block of rows (`common.SpGroup`).  The conv histories are picked (0/1 products: exact)
    from the block [1, 1, 64, 6144] of the row's wave history (rows 0-2) and every row's last 3 inputs gathered over
    the rows (row r's at 32 + 3 r)."""

    in_pick: object  # [1, 1, 3, 64] per row: its segment's conv history (its wave history for a request's first, else
    # the previous segment's last inputs)
    end_pick: object  # [1, 1, 3, 64] per row: the last 3 inputs of its request's last segment (the next history)
    start: object  # [1] uint32: the wave's first row f as the ops' chronology (f T; T rows a row)
    keep16: object  # [1, 1, 1] bf16 per row: 0 where a request begins (its state and history from zeros), else 1
    keep32: object  # the same, fp32
    owner: object  # [1, 1, 1] fp32 per row: 1 on a request's last segment's row (its scan's state is the wave state)
    others: object  # 1 - owner
    rank_mask: object  # [1, 8] uint32 row-major per row: chain_affine_transforms' rank_mask, 1 on its own request's
    # rows (the other rows' maps read as the identity)


def sp_identity(mesh, dh: int):
    """[1, 1, 128, 256] bf16 [I | 0]: a state map S -> S (`TtKda._prefill_sp`'s maps of other requests' rows)."""
    return const_tensor(
        mesh, "kda_sp_identity", lambda: torch.cat([torch.eye(dh), torch.zeros(dh, dh)], dim=1).view(1, 1, dh, 2 * dh)
    )


class TtKda:
    def __init__(self, mesh, w: KdaWeights, config: Glm53Config, dtype=ttnn.bfloat16) -> None:
        self.mesh = mesh
        self.heads = heads = config.kda_num_heads // COLS  # per chip
        self.dh = dh = config.kda_head_dim
        self.kernel = config.kda_conv_kernel
        self.lower = config.kda_gate_lower_bound
        self.eps = config.rms_norm_eps
        self.channels = 3 * heads * dh  # q, k, v per chip
        width = config.kda_num_heads * dh  # 8192

        def channels(c: int) -> torch.Tensor:
            """Chip c's q, k, v channels in the order (q | k | v, head, d) (ttnn.experimental.kda's conv order), as
            indices into q | k | v."""
            idx = [
                part * width + (c * heads + h) * dh + d for part in range(3) for h in range(heads) for d in range(dh)
            ]
            return torch.tensor(idx)

        qkv = torch.cat([w.q_proj, w.k_proj, w.v_proj])  # [3 * 8192, hidden]

        def x_proj(r, c):
            beta = torch.zeros(BETA_COLS, w.b_proj.shape[1], dtype=w.b_proj.dtype, device=w.b_proj.device)
            beta[:heads] = w.b_proj[c * heads : (c + 1) * heads]
            return torch.cat([qkv[channels(c)], beta, w.f_a, w.g_a]).T.contiguous()[None, None]

        def fg_proj(r, c):
            own = slice(c * heads * dh, (c + 1) * heads * dh)
            m = torch.zeros(2 * dh, 2 * heads * dh, dtype=torch.bfloat16, device=w.f_b.device)
            m[:dh, : heads * dh] = w.f_b[own].T
            m[dh:, heads * dh :] = w.g_b[own].T
            return m[None, None]

        self.x_proj = upload_shards(mesh, x_proj, dtype=dtype)  # [hidden, 6144 + 32 + 256]
        self.fg_proj = upload_shards(mesh, fg_proj, dtype=dtype)  # [256, 2 * 2048]: f | gate
        # prefill (ttnn.experimental.kda): bf16 taps [1, 1, 6144], the norm weight [128], a zero start position
        self.taps_bf16 = [
            upload_shards(mesh, lambda r, c, t=t: w.conv[channels(c), t].reshape(1, 1, -1).to(torch.bfloat16))
            for t in range(self.kernel)
        ]
        self.o_norm_bf16 = put(mesh, w.o_norm.to(torch.bfloat16).reshape(-1), dims=(None, None))
        self.zero = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.op_config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        rate = torch.exp(w.a_log.float()).repeat_interleave(dh)  # per channel
        from .kda_prefill_ops import gate_constant

        # the prefill's decay gate (`kda_prefill_ops.gate`): dt_bias and rate a row in all 32 rows
        self.gate_bcast = (gate_constant(mesh, w.dt_bias, width), gate_constant(mesh, rate, width))
        self.o_norm = put(mesh, w.o_norm.float().reshape(1, 1, 1, -1), dims=(None, None), dtype=ttnn.float32)
        # decode_prepare: the 4 taps as rows [4, 6144] and dt_bias | rate as rows [2, 2048], per chip
        self.taps_rows = upload_shards(
            mesh,
            lambda r, c: torch.stack([w.conv[channels(c), t] for t in range(self.kernel)]).float()[None, None],
            dtype=ttnn.float32,
        )
        gate_rows = torch.cat([w.dt_bias.float().reshape(1, width), rate.reshape(1, width)])
        self.gate_rows = put(mesh, gate_rows[None, None], dims=(None, 3), dtype=ttnn.float32)
        self.o_proj = tp_weight(mesh, w.o_proj, "in", dtype)

    def empty_state(self, batch: int, spec: int = 0) -> TtKdaState:
        """spec: with the state of a verify step of that many drafts (`step_spec`: the conv history's checkpoint rows,
        zeros, and the step's rows doing nothing)."""
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        history = [zeros(self.mesh, [1, 1, batch, self.channels], f32) for _ in range(self.kernel - 1)]
        recurrent = zeros(self.mesh, [batch, self.heads, self.dh, self.dh], bf16)
        wave_history = zeros(self.mesh, [1, self.kernel - 1, self.channels])
        wave_state = zeros(self.mesh, [self.heads, self.dh, self.dh], f32)
        state = TtKdaState(history=history, recurrent=recurrent, wave_history=wave_history, wave_state=wave_state)
        if not spec:
            return state
        state.spec_saved = zeros(self.mesh, [1, spec, batch, self.channels], f32)
        inert = chunk_saved_host(1, self.heads, spec + 1)
        state.spec_inert = put(self.mesh, inert, dims=(None, None), dtype=f32)

        def rows():  # (a tensor of its own: a repeat of one user is the input itself; the steps write it)
            rep = ttnn.repeat(state.spec_inert, ttnn.Shape([batch, 1, 1, 1])) if batch > 1 else None
            return rep if rep is not None else ttnn.clone(state.spec_inert)

        state.spec_rows = rows()
        if spec_pingpong():  # the other set, as this one starts (zeros, the rows doing nothing)
            state.spec_alt = SpecSet(
                history=[zeros(self.mesh, [1, 1, batch, self.channels], f32) for _ in range(self.kernel - 1)],
                recurrent=zeros(self.mesh, [batch, self.heads, self.dh, self.dh], bf16),
                spec_saved=zeros(self.mesh, [1, spec, batch, self.channels], f32),
                spec_rows=rows(),
            )
        return state

    def step(self, x, state: TtKdaState):
        """x [1, 1, B, 4096] bf16 (the same on a row's chips) -> (out [1, 1, B, 4096] bf16, state)."""
        b, heads, dh, chans = x.shape[-2], self.heads, self.dh, self.channels
        proj = decode_linear(self.mesh, x, self.x_proj, compute_kernel_config=WEIGHT_MM)  # [1, 1, B, 6432] bf16
        fg_in = ttnn.slice(proj, [0, 0, 0, chans + BETA_COLS], [1, 1, b, chans + BETA_COLS + 2 * dh])
        fg = ttnn.linear(fg_in, self.fg_proj, compute_kernel_config=WEIGHT_MM)  # [1, 1, B, 2 * 2048] bf16
        ttnn.deallocate(fg_in)
        hist = state.history
        q, k, v, e, beta = ttnn.experimental.kda.decode_prepare(
            proj,
            fg,
            hist[0],
            hist[1],
            hist[2],
            self.taps_rows,
            self.gate_rows,
            heads,
            self.lower,
            1.0 / dh**0.5,
            row_split=True,
        )
        ttnn.deallocate(proj)
        o = ttnn.experimental.kda.recurrent_decode_step(state.recurrent, q, k, v, e, beta, form=PLAIN_FORM)
        return self._output(o, fg, b), state

    def step_spec(self, x, state: TtKdaState, keep: SpecKeep):
        """A verify step of K drafts (MTP's K = 1 / 2, DFlash2's): x [1, 1, (K + 1)
        B, 4096], each user's token, then its drafts (draft j at rows j B ..), one position on each -> (out [1, 1, (K +
        1) B, 4096], state).  The projections, the gated norm, o_proj and the all-reduce take the (K + 1) B tokens at
        once.  One decode_prepare of the K + 1 tokens from the history the keep codes pick (code n: the last step's n
        drafts accepted, its history n tokens on), the earlier inputs into the K checkpoint rows; one
        recurrent_decode_step in the chunked form (the last step's accepted tokens redone from its start state, then
        this step's K + 1 tokens; the step's start state and rows written for the next)."""
        rows, heads, dh, chans = x.shape[-2], self.heads, self.dh, self.channels
        n = state.spec_saved.shape[1] + 1
        # a ping-pong (spec_alt): this step reads the fields and writes the other set (spec_flip swaps them after)
        alt = state.spec_alt
        pp_prep = {}
        if alt is not None:
            pp_prep = dict(
                history0_out=alt.history[0],
                history1_out=alt.history[1],
                history2_out=alt.history[2],
                saved_out=alt.spec_saved,
            )
        b = rows // n
        proj = decode_linear(self.mesh, x, self.x_proj, compute_kernel_config=WEIGHT_MM)  # [1, 1, n B, 6432] bf16
        fg_in = ttnn.slice(proj, [0, 0, 0, chans + BETA_COLS], [1, 1, rows, chans + BETA_COLS + 2 * dh])
        fg = ttnn.linear(fg_in, self.fg_proj, compute_kernel_config=WEIGHT_MM)  # [1, 1, n B, 2 * 2048] bf16
        ttnn.deallocate(fg_in)
        mid = ttnn.experimental.kda.decode_prepare(
            proj,
            fg,
            *state.history,
            self.taps_rows,
            self.gate_rows,
            heads,
            self.lower,
            1.0 / dh**0.5,
            users=b,
            keep=keep.flags,
            saved=state.spec_saved,
            tokens=n,
            pair_major=True,
            l1_outputs=HANDOFF_L1,
            split_channels=PREP_SPLIT,
            **pp_prep,
        )
        ttnn.deallocate(proj)
        chunk = dict(saved=state.spec_rows)
        if alt is not None:  # (the step's start state and rows into the other set, the last step's read from here)
            chunk.update(saved=alt.spec_rows, source=state.recurrent, saved_source=state.spec_rows)
        if HANDOFF_L1:  # (o into L1: decode_finish reads it next)
            chunk["output"] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows * heads, dh]),
                ttnn.float32,
                ttnn.TILE_LAYOUT,
                self.mesh,
                ttnn.L1_MEMORY_CONFIG,
            )
        o = ttnn.experimental.kda.recurrent_decode_step(
            alt.recurrent if alt is not None else state.recurrent,
            *mid,
            keep=keep.flags,
            heads=heads,
            tokens=n,
            pair_major=True,
            **chunk,
        )
        for t in mid:
            ttnn.deallocate(t)
        return self._output(o, fg, rows), state

    def _output(self, o, fg, b: int):
        """The delta rule's output o [1, 1, B H, 128] fp32 and fg -> out [1, 1, B, 4096] bf16: decode_finish (the gated
        RMS norm in o_proj's layout), o_proj and the row's all-reduce."""
        flat = ttnn.experimental.kda.decode_finish(o, fg, self.o_norm, self.heads, self.eps)
        ttnn.deallocate(o)
        ttnn.deallocate(fg)
        out_cfg = ar_input_config(self.mesh, flat.shape[-2], self.o_proj.shape[-1])
        partial = decode_linear(self.mesh, flat, self.o_proj, compute_kernel_config=WEIGHT_MM, memory_config=out_cfg)
        return tp_all_reduce(self.mesh, partial)

    @staticmethod
    def prefill_picks(
        place,
        t: int,
        valid: list[int],
        kernel: int,
        users: int,
        slots: list | None = None,
        rows: list | None = None,
        traced: bool = False,
    ) -> KdaPicks:
        """A prefill chunk's inputs made once per chunk and shared by the KDA layers (`KdaPicks`), put on the device
        by `place` (`common.eager_place` or `FixedInputs.place`): the 0/1 matrix [1, 3, 32 + T] per mesh row that
        picks the conv history's next inputs from its count of real tokens (`_last_inputs`), the real-token mask
        [1, T, 1] per row (None when every row is full, unless `traced`), and with `slots` (a wave's last chunk: per
        row its request's user, of `users`, or None for a row without one) what writes the rows' states
        into those users.  `rows`: the rows that compute this chunk (default all).  `traced`: the same tensors
        whatever the rows (a captured prefill)."""
        k = kernel - 1
        pick = prefill_tables.history_pick(ROWS, valid, k, t)
        picks = KdaPicks(history_pick=place("kda_history_pick", pick, (0, None)), real=None, owned=not traced)
        if traced or min(valid) < t or (rows is not None and not all(rows)):
            mask = torch.arange(t)[None] < torch.tensor(valid)[:, None]
            if rows is not None:  # a row without a request has no real token (the MoE leaves them out: PREFILL_MASK)
                mask &= torch.tensor(rows, dtype=torch.bool)[:, None]
            picks.real = place("kda_real", mask.float().view(ROWS, t, 1), (0, None))
        page = torch.tensor([0 if rows is None or rows[r] else -1 for r in range(ROWS)], dtype=torch.int32)
        picks.wave_page = place("kda_wave_page", page.view(ROWS, 1), (0, None), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        if slots is not None:
            one_hot = torch.stack(
                [(torch.arange(users) == s).float() if s is not None else torch.zeros(users) for s in slots]
            )
            picks.commit_rows = place("kda_commit_rows", one_hot.view(ROWS, 1, users, 1), (0, None), ttnn.float32)
            if traced:
                picks.commit_users = place(
                    "kda_commit_users", one_hot.view(ROWS * users, 1, 1, 1), (0, None), ttnn.float32
                )
            else:  # every row, as `commit_picks` (a row without a request: the table's -1, nothing written): the same
                # program for any rows (the rows' coordinates made one per set of rows, compiled after a server's
                # captures: a buffer allocated after them, alive at a replay)
                table = torch.tensor([-1 if s is None else s for s in slots], dtype=torch.int32).view(ROWS, 1)
                picks.commit_table = place("kda_commit_table", table, (0, None), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            picks.all_kept = rows is None or all(rows[r] for r, s in enumerate(slots) if s is not None)
        return picks

    @staticmethod
    def sp_picks(place, t: int, groups, kernel: int, users: int, traced: bool = False) -> KdaPicks:
        """A sequence-parallel chunk's inputs (`KdaPicks` with `sp`): each request's segments of T tokens on its rows
        (`common.SpGroup`: consecutive in the ring order from the wave's first row, the groups one after another, maybe
        with rows between them, the last row the request's own), only a request's last segment partial.  A group's `last` wave writes its state
        into its user on its own row (`commit`; `traced`: apart, `replay_commit`).  The chain of the segments' state
        maps (`_prefill_sp`) runs in the ring order from the first group's first row, each row seeing its own
        request's maps only."""
        k = kernel - 1
        first = sp_first(groups)
        order = [(first + c) % ROWS for c in range(ROWS)]
        member: dict = {}  # row -> (group, index)
        at = 0  # the ring position after the previous group (rows between groups have no segment)
        for grp in groups:
            if len(grp.rows) != len(grp.valid) or any(v != t for v in grp.valid[:-1]) or not 0 < grp.valid[-1] <= t:
                raise ValueError(f"a group's segments {grp.valid}: full but the last, of {t}")
            pos = [order.index(r) for r in grp.rows]
            if pos[0] < at or pos != list(range(pos[0], pos[0] + len(pos))):
                raise ValueError(f"groups {[g.rows for g in groups]}: not in the ring order from row {first}")
            for i, r in enumerate(grp.rows):
                member[r] = (grp, i)
            at = pos[-1] + 1
        valid = [member[r][0].valid[member[r][1]] if r in member else 0 for r in range(ROWS)]
        # rows 0-2 of the last-inputs pick as `prefill_picks`' (from the row's wave history and its segment's tokens)
        hist = prefill_tables.history_pick(ROWS, valid, k, t)
        chain = prefill_tables.kda_sp_arrays(ROWS, member, k)  # (the chain's picks, rank mask, keep and owner)
        inc, end, mask, keep, owner = (chain[n] for n in ("inc", "end", "mask", "keep", "owner"))
        real = torch.arange(t)[None] < torch.tensor(valid)[:, None]
        sp = KdaSpPicks(
            in_pick=place("kda_sp_in", inc.view(ROWS, 1, k, 64), (0, None)),
            end_pick=place("kda_sp_end", end.view(ROWS, 1, k, 64), (0, None)),
            start=place("kda_sp_start", torch.tensor([first * t]), (None, None), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            keep16=place("kda_sp_keep16", keep, (0, None)),
            keep32=place("kda_sp_keep32", keep, (0, None), ttnn.float32),
            owner=place("kda_sp_owner", owner, (0, None), ttnn.float32),
            others=place("kda_sp_others", 1.0 - owner, (0, None), ttnn.float32),
            rank_mask=place(
                "kda_sp_rank_mask", mask.to(torch.int32).view(ROWS, ROWS), (0, None), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            ),
        )
        picks = KdaPicks(
            history_pick=place("kda_history_pick", hist, (0, None)),
            real=place("kda_real", real.float().view(ROWS, t, 1), (0, None)),
            sp=sp,
            owned=not traced,
        )
        slots = [None] * ROWS
        for grp in groups:
            if grp.last:
                slots[grp.owner] = grp.user
        if not traced and any(s is not None for s in slots):
            one_hot = torch.stack(
                [(torch.arange(users) == s).float() if s is not None else torch.zeros(users) for s in slots]
            )
            picks.commit_rows = place("kda_commit_rows", one_hot.view(ROWS, 1, users, 1), (0, None), ttnn.float32)
            table = torch.tensor([-1 if s is None else s for s in slots], dtype=torch.int32).view(ROWS, 1)
            picks.commit_table = place("kda_commit_table", table, (0, None), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        return picks

    @staticmethod
    def commit_picks(place, users: int, slots: list) -> KdaPicks:
        """What writes each mesh row's wave state into its user (`slots[row]`, of `users`, or None: no request) apart
        from a chunk, the same programs for any rows: a captured prefill's commit (`TtGlm53Model.capture_prefills`).
        The delta-rule state goes by a paged copy on every chip, the row's user as the page table (`_commit`); a row
        without a request has -1 in the table (nothing written), so the cost does not grow with the number of users (the
        one-hot over all of them, as a captured one-chunk wave commits, took 25 ms a wave at 32 users a row)."""
        one_hot = torch.stack(
            [(torch.arange(users) == s).float() if s is not None else torch.zeros(users) for s in slots]
        )
        table = torch.tensor([-1 if s is None else s for s in slots], dtype=torch.int32).view(ROWS, 1)
        picks = KdaPicks(history_pick=None, real=None, owned=False)
        picks.commit_rows = place("kda_commit_rows", one_hot.view(ROWS, 1, users, 1), (0, None), ttnn.float32)
        picks.commit_table = place("kda_commit_table", table, (0, None), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        return picks

    def commit(self, state: TtKdaState, picks: KdaPicks) -> None:
        """Each mesh row's wave state (its delta-rule state and conv history, as the wave's chunks left them) into its
        user (`commit_picks`): what a wave's last chunk does after its wave-state update when a row's prompt ended
        before it."""
        self._commit(state, state.wave_state, state.wave_history, picks)

    def _history_head(self, hist):
        """The conv history [1, 3, 6144] (row-major bf16) in rows 0..2 of a tiled [1, 32, 6144] block, zeros below."""
        k, chans = self.kernel - 1, self.channels
        pad = const_zeros(self.mesh, [1, 32 - k, chans], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.concat([hist, pad], dim=1), ttnn.TILE_LAYOUT)

    def _last_inputs(self, proj, picks: KdaPicks, hist=None, head=None):
        """The request's next conv history on each mesh row: the last 3 inputs of (its history, the chunk's first
        valid[row] tokens), [1, 3, 6144] bf16.  The rows' counts differ, so a 0/1 matrix per row (`picks.history_pick`,
        from `prefill_picks`) picks them from the history [1, 3, 6144] (row-major bf16) in rows 0..2 of a block of 32
        (the rest zero) and the chunk's inputs (input n, oldest history -3, is block row n + 3 or the chunk's row n):
        one product of each part, added (each output one copy plus a zero: exact).  `head`: the history block already
        made (`_history_head`, kept), else made from `hist`."""
        k, chans = self.kernel - 1, self.channels
        own_head = head is None
        if own_head:
            head = self._history_head(hist)
        pick_head, pick_inputs = picks.history_parts()
        from_inputs = ttnn.matmul(pick_inputs, proj, compute_kernel_config=HIFI4_FP32)  # [1, 3, 6432]
        from_head = ttnn.matmul(pick_head, head, compute_kernel_config=HIFI4_FP32)  # [1, 3, 6144]
        if own_head:
            ttnn.deallocate(head)
        conv = ttnn.slice(from_inputs, [0, 0, 0], [1, k, chans])
        ttnn.deallocate(from_inputs)
        last = ttnn.add(conv, from_head)
        ttnn.deallocate(conv)
        ttnn.deallocate(from_head)
        return last

    def prefill(self, x, state: TtKdaState, start: int, keep, picks: KdaPicks, project: bool = True):
        """A prefill chunk with ttnn.experimental.kda (chunks of 32 tokens): x [1, 1, T, 4096] bf16, T tokens
        (positions start ..) of each mesh row's request (T a multiple of 32; only the first valid[row] are real, the
        rest are made identity steps), on the rows whose `keep` ([1, 1, 1, 1] per row) is 1; elsewhere the row's wave
        state stays as it was (a prompt that has ended).  `picks`: `prefill_picks` for the chunk, made once per chunk
        by the caller.  `start` 0 begins a request from an empty state; otherwise the chunk continues from the row's
        wave state.  On a wave's last chunk (`picks.commit_rows`) every row with a request writes its wave state into its
        user.  -> (out [1, 1, T, 4096] bf16, state).

        The ops run per chip on one request: conv + SiLU (`qkv_causal_conv1d_silu`, from the wave's conv history),
        `prepare_chunk_recurrence` (l2 norms, the 1/sqrt(128) scale, sigmoid(beta), the gate's cumulative sum and
        exponentials) and `recurrent_chunk_scan` from the wave state, then `sigmoid_gated_rms_norm`.  The mesh rows
        are the ops' sequence-parallel axis; with the start at 0 every row computes on its own (no exchange)."""
        if picks.sp is not None:
            return self._prefill_sp(x, state, picks, project)
        assert project, "the gated norm's output alone: the sequence-parallel form's (TtDecoderLayer.prefill's pair)"
        t, heads, dh, chans = x.shape[-2], self.heads, self.dh, self.channels
        if t % 32:
            raise ValueError(f"a prefill chunk of {t} rows must be a multiple of 32")
        width = heads * dh
        proj = wave_linear(self.mesh, x, self.x_proj, compute_kernel_config=WEIGHT_MM)  # [1, 1, T, 6432] bf16
        proj = ttnn.reshape(proj, [1, t, proj.shape[-1]])

        # the request's conv history, oldest first: [1, 3, 6144] bf16 row-major (shared zeros at the start)
        if start == 0:
            hist = const_zeros(self.mesh, [1, self.kernel - 1, chans], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        else:
            hist = ttnn.to_layout(state.wave_history, ttnn.ROW_MAJOR_LAYOUT)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            proj,
            hist,
            *self.taps_bf16,
            width,
            width,
            width,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=_conv_block(proj.shape[-2])),
            actual_start=self.zero,
            predecessor_carry=hist,
            compute_kernel_config=self.op_config,
            real=picks.real,
        )
        # a request's first chunk: the history block is the shared zero tile (in place of zeros padded and tilized)
        head = const_zeros(self.mesh, [1, 32, chans], ttnn.bfloat16) if start == 0 else None
        last = self._last_inputs(proj, picks, hist=hist, head=head)
        if start:
            ttnn.deallocate(hist)
        fg, terms = self._terms(proj, q, k, v, picks.real)
        s0 = const_zeros(self.mesh, [heads, dh, dh], ttnn.float32) if start == 0 else state.wave_state
        out, s_end = ttnn.experimental.kda.recurrent_chunk_scan(
            *terms, s0, actual_start=self.zero, tail_entry_states=s0, compute_kernel_config=self.op_config
        )
        for z in terms:
            ttnn.deallocate(z)
        o = self._gated_norm(out, fg)
        ttnn.deallocate(proj)

        if picks.commit_rows is not None and picks.all_kept:  # the wave ends with every request: into their users
            self._commit(state, s_end, last, picks)
        else:
            # the wave state: the scan's final state copied on the rows that compute (the page -1 elsewhere: skipped;
            # a copy: exact); the new history where keep is 1, as it was elsewhere (x 0 / 1, x + 0: exact), in place.
            # (Copying the final state on every row instead, a row without a request taking identity steps, moved the
            # decode logits by up to 3.5e-4: not exact.)
            whole = [1, heads, dh, dh]
            ttnn.experimental.paged_fill_cache(
                ttnn.reshape(state.wave_state, whole), ttnn.reshape(s_end, whole), picks.wave_page
            )
            drop = ttnn.rsub(keep, 1.0)
            keep16, drop16 = (ttnn.reshape(z, [1, 1, 1]) for z in (keep, drop))
            ttnn.multiply(state.wave_history, drop16, output_tensor=state.wave_history)
            ttnn.addcmul(state.wave_history, last, keep16, output_tensor=state.wave_history)
            ttnn.deallocate(drop)
            if picks.commit_rows is not None:
                self._commit(state, state.wave_state, state.wave_history, picks)
        ttnn.deallocate(s_end)  # (s0 is the shared zeros or the wave state)
        ttnn.deallocate(last)

        partial = wave_linear(
            self.mesh, ttnn.reshape(o, [1, 1, t, width]), self.o_proj, compute_kernel_config=WEIGHT_MM
        )
        ttnn.deallocate(o)
        return tp_all_reduce(self.mesh, partial), state

    def _prefill_sp(self, x, state: TtKdaState, picks: KdaPicks, project: bool = True):
        """A sequence-parallel chunk (`sp_picks`): x [1, 1, T, 4096] per mesh row, the row's segment of its request ->
        (out, state).

        The conv histories come from a gather of every row's last 3 inputs over the rows (a request's first segment's
        from the row's wave history, zeros where the request begins).  Each row prepares its segment as `prefill`
        does and summarizes it as one affine map of the state, S -> A S + B (`summarize_chunk_recurrence`: [A | B] in
        bf16).  The maps, gathered over the rows, are applied in the ring order from the wave state
        (`chain_affine_transforms`, fp32 accumulation), each row seeing those of its own request's rows (the others
        the identity: x 1 + 0, x 0 + [I | 0], exact), which gives each row its segment's entry state and the state
        after its request's last segment; each row then scans its segment from its entry state.  The wave state
        becomes the last segment's scan's final state on its row and the chain's on the request's other rows (the
        same state, the chain's through the bf16 maps), and the wave history the last segment's last inputs; a row
        without a segment keeps both.  A group's last wave (`sp_picks`' commit) writes them into its user."""
        sp = picks.sp
        t, heads, dh, chans, k = x.shape[-2], self.heads, self.dh, self.channels, self.kernel - 1
        width = heads * dh
        proj = wave_linear(self.mesh, x, self.x_proj, compute_kernel_config=WEIGHT_MM)  # [1, 1, T, 6432] bf16
        proj = ttnn.reshape(proj, [1, t, proj.shape[-1]])
        kept = ttnn.multiply(state.wave_history, sp.keep16)  # x 0 / 1: exact
        hist = ttnn.to_layout(kept, ttnn.ROW_MAJOR_LAYOUT)  # [1, 3, 6144]
        ttnn.deallocate(kept)
        from .col_collectives import col_gather  # (it imports common)

        head = self._history_head(hist)  # [1, 32, 6144]: the wave history in rows 0..2
        ttnn.deallocate(hist)
        own = self._last_inputs(proj, picks, head=head)  # [1, 3, 6144]
        gathered = col_gather(self.mesh, ttnn.reshape(own, [1, 1, k, chans]), SP_GATHER_WORKERS)
        ttnn.deallocate(own)  # gathered: [1, 1, 32, 6144], row r's at 3 r, zeros from 24
        block = ttnn.concat([ttnn.reshape(head, [1, 1, 32, chans]), gathered], dim=2)  # [1, 1, 64, 6144]
        ttnn.deallocate(head)
        ttnn.deallocate(gathered)
        incoming = ttnn.matmul(sp.in_pick, block, compute_kernel_config=HIFI4_FP32)  # [1, 1, 3, 6144]
        end = ttnn.matmul(sp.end_pick, block, compute_kernel_config=HIFI4_FP32)
        ttnn.deallocate(block)
        inc = ttnn.to_layout(ttnn.reshape(incoming, [1, k, chans]), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(incoming)
        q, kk, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            proj,
            inc,
            *self.taps_bf16,
            width,
            width,
            width,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=_conv_block(proj.shape[-2])),
            actual_start=self.zero,
            predecessor_carry=inc,
            compute_kernel_config=self.op_config,
            real=picks.real,
        )
        ttnn.deallocate(inc)
        fg, terms = self._terms(proj, q, kk, v, picks.real)
        ttnn.deallocate(proj)
        (packed,) = ttnn.experimental.kda.summarize_chunk_recurrence(
            *terms,
            groups_per_head=1,
            actual_start=sp.start,
            sequence_parallel_axis=ROW_AXIS,
            compute_kernel_config=self.op_config,
            packed_head=True,
        )  # [1, 16, 128, 256] bf16: [A | B]
        maps = ttnn.all_gather(packed, dim=0, cluster_axis=ROW_AXIS)  # [8, 16, 128, 256]
        ttnn.deallocate(packed)
        s0 = ttnn.multiply(state.wave_state, sp.keep32)
        # the other requests' rows' maps read as the identity in the op (rank_mask)
        entry, after = ttnn.experimental.kda.chain_affine_transforms(
            maps,
            s0,
            actual_start=sp.start,
            local_rows=t,
            compute_kernel_config=self.op_config,
            sequence_parallel_axis=ROW_AXIS,
            rank_mask=sp.rank_mask,
            identity=sp_identity(self.mesh, dh),
        )
        for z in (maps, s0):
            ttnn.deallocate(z)
        out, s_end = ttnn.experimental.kda.recurrent_chunk_scan(
            *terms, entry, actual_start=self.zero, tail_entry_states=entry, compute_kernel_config=self.op_config
        )
        for z in (*terms, entry):
            ttnn.deallocate(z)
        o = self._gated_norm(out, fg)
        # the wave state (x 1 + y x 0: exact), in place: the last segment's scan's on its row, the chain's elsewhere
        ttnn.multiply(after, sp.others, output_tensor=state.wave_state)
        ttnn.addcmul(state.wave_state, s_end, sp.owner, output_tensor=state.wave_state)
        ttnn.copy(ttnn.reshape(end, [1, k, chans]), state.wave_history)
        for z in (after, s_end, end):
            ttnn.deallocate(z)
        if picks.commit_rows is not None:
            self._commit(state, state.wave_state, state.wave_history, picks)
        o = ttnn.reshape(o, [1, 1, t, width])
        if not project:  # (the pair's o_proj and all-reduce once over both waves: `project`, TtDecoderLayer.prefill)
            return o, state
        return self.project(o), state

    def project(self, o):
        """The gated norm's output [1, 1, T, 2048] bf16 (deallocated) -> o_proj and the row all-reduce [1, 1, T, 4096].
        Per token: a pair of sequence-parallel waves takes both waves' at once (TtDecoderLayer.prefill; o_proj's K
        blocks are 16 at 512 and 1,024 rows, the all-reduce sums by column: the same values)."""
        partial = wave_linear(self.mesh, o, self.o_proj, compute_kernel_config=WEIGHT_MM)
        ttnn.deallocate(o)
        return tp_all_reduce(self.mesh, partial)

    def _terms(self, proj, q, k, v, real):
        """A chunk's decay gate and prepared recurrence terms from its projection [1, T, 6432] and conv outputs (q, k,
        v deallocated) -> (fg [1, T, 4096]: f | gate, the 7 `prepare_chunk_recurrence` terms).  `real` ([1, T, 1] per
        row, or None): the padded tokens' k = v = g = 0, an identity step, the mask folded into the ops that make k, v
        (the conv's writer, `real`) and the gate (its writer) rather than three multiplies (the same values: x * 0
        keeps x's sign; the prefill +1.7 / +2.2 / +1.8 % at 8 x 128 / 8 x 2,048 / 1 x 8K)."""
        t, heads, dh, chans = proj.shape[1], self.heads, self.dh, self.channels
        width = heads * dh
        # the decay gate in [0, 1] (prepare scales it by -5): sigmoid(exp(A_log) (f_b(f_a(x)) + dt_bias))
        fg_in = ttnn.slice(proj, [0, 0, chans + BETA_COLS], [1, t, chans + BETA_COLS + 2 * dh])
        fg = wave_linear(self.mesh, fg_in, self.fg_proj, compute_kernel_config=WEIGHT_MM)  # [1, 1, T, 4096]
        fg = ttnn.reshape(fg, [1, t, fg.shape[-1]])
        ttnn.deallocate(fg_in)
        if t % 32:
            raise ValueError(f"a prefill chunk of {t} rows must be a multiple of 32")
        from .kda_prefill_ops import gate

        # bf16(sigmoid((fp32(f) + dt_bias) * rate)) in one op (the ttnn ops' SFPU steps and rounding)
        gate01 = gate(self.mesh, fg, *self.gate_bcast, width, real=real)
        terms = ttnn.experimental.kda.prepare_chunk_recurrence(
            q,
            k,
            v,
            gate01,
            proj,
            heads,
            compute_kernel_config=self.op_config,
            output_bf16_mask=0,
            actual_start=self.zero,
            gate_scale=float(self.lower),
            beta_logits_column_offset=chans,
        )
        for z in (q, k, v, gate01):
            ttnn.deallocate(z)
        return fg, terms

    def _gated_norm(self, out, fg):
        """The scan's output [16, T, 128] (deallocated, with fg) -> the gated RMS norm [1, T, 2048] bf16."""
        t = fg.shape[1]
        o = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            ttnn.reshape(out, [self.heads, t, self.dh]),
            fg,
            self.o_norm_bf16,
            self.heads,
            epsilon=self.eps,
            compute_kernel_config=self.op_config,
            output_dtype=ttnn.bfloat16,
            gate_column_offset=self.heads * self.dh,
        )  # [1, T, 2048]
        ttnn.deallocate(out)
        ttnn.deallocate(fg)
        return o

    def _commit_rows(self, state: TtKdaState, picks: KdaPicks) -> None:
        """The request's user's saved rows of the verify step to do nothing (`spec_inert`), so that its first verify
        step's redo of "all T" (keep K) starts from the prefill's state as it is; the same picks as the state's commit
        (`_commit`)."""
        rows, inert = state.spec_rows, state.spec_inert
        if picks.commit_table is not None:  # (the row's user, -1: skipped)
            ttnn.experimental.paged_fill_cache(rows, inert, picks.commit_table)
        elif picks.commit_users is not None:  # (the row's user's one-hot over all B: x 0 / 1, + 0, exact)
            others = ttnn.rsub(picks.commit_users, 1.0)
            ttnn.multiply(rows, others, output_tensor=rows)
            ttnn.addcmul(rows, picks.commit_users, inert, output_tensor=rows)
            ttnn.deallocate(others)

    def _commit(self, state: TtKdaState, recurrent, history, picks: KdaPicks) -> None:
        """Each mesh row with a request writes its request's final state into its user: the delta-rule state
        (`recurrent` [16, 128, 128] fp32, rounded to bf16 once on its way in) with paged_fill_cache (the states [B, 16,
        128, 128] seen as B pages of 128 rows, the row's user as the page table, -1 without a request: a tile copy), or
        in a one-chunk wave's trace by that user's one-hot over all B states (x 0 / 1, + 0: exact); the conv history
        (`history` [1, 3, 6144] bf16) rows by one op of copies into the row's user (the page table; in the one-chunk
        wave's trace by the one-hot)."""
        heads, dh = self.heads, self.dh
        state.spec_pending = False  # (the ping-pong: a step before this commit is kept; it writes the fields' set)
        final = ttnn.reshape(recurrent, [1, heads, dh, dh])

        def to_state(t):  # t in the states' dtype (bf16, a new tensor), t deallocated
            out = ttnn.typecast(t, ttnn.bfloat16)
            ttnn.deallocate(t)
            return out

        if picks.commit_table is not None:  # the final state into the row's user (-1: skipped)
            src = ttnn.typecast(final, ttnn.bfloat16)
            ttnn.experimental.paged_fill_cache(state.recurrent, src, picks.commit_table)
            ttnn.deallocate(src)
        elif picks.commit_users is not None:
            # [B, 16, 128, 128]: the final state at the row's user, else 0
            new = to_state(ttnn.multiply(final, picks.commit_users))
            others = to_state(ttnn.rsub(picks.commit_users, 1.0))
            ttnn.multiply(state.recurrent, others, output_tensor=state.recurrent)
            ttnn.add(state.recurrent, new, output_tensor=state.recurrent)
            ttnn.deallocate(new)
            ttnn.deallocate(others)
        if state.spec_rows is not None:
            self._commit_rows(state, picks)
        if picks.commit_table is not None:  # the history rows into the row's user (-1: none)
            from .kda_prefill_ops import history_commit

            history_commit(self.mesh, history, state.history, picks.commit_table)
            return
        one_hot = picks.commit_rows
        stay = ttnn.rsub(one_hot, 1.0)
        for i, h in enumerate(state.history):
            row = ttnn.slice(history, [0, i, 0], [1, i + 1, self.channels])
            new = ttnn.reshape(ttnn.typecast(row, ttnn.float32), [1, 1, 1, self.channels])
            ttnn.multiply(h, stay, output_tensor=h)
            ttnn.addcmul(h, one_hot, new, output_tensor=h)  # + the new row at the row's user (column times row)
            ttnn.deallocate(row)
            ttnn.deallocate(new)
        ttnn.deallocate(stay)
