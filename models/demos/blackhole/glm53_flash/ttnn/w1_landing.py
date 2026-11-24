# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The L1 landing of the next MoE layer's chosen experts' weights (GLM53_W1): the experts op reads its layer's C chosen
experts' weight blocks from an L1 landing that a DRISC a bank fills in the layer before's end window, in place of DRAM.
The values are the same: the landing holds the DRAM's bytes, read as a seqlock (seal, block, seal: the layer's tag | the
expert << 16 both times), and a block whose seals are not (its layer, its expert) is read from DRAM again
(stream_active.cpp MOE_W1_RECV, kernels/w1_landing/drisc_w1.cpp).

The landing is made at the model's load (`TtGlm53Model`: by default for the DFlash2 verify step's shapes, `landing_c`;
GLM53_W1=0 off, GLM53_W1=C that many slots; never with TT_METAL_STREAMING_PROFILER) and held for the process, so every
program is made with it in L1 (a static CB over it fails at the program's allocation; the SDPA keeps its K and V in one
ring of 2 slots, TT_SDPA_KV_SHARED (set by the landing) / TT_SDPA_KV_SLOTS_MAX: the values the same).  Each arming
seals with its own tags (`arm`: an epoch in them), so no block left by a former arming passes.  The decode calls of the
experts op take their layer's spec (`arm_model`: each TtMoe's `w1_spec`, given to the op only inside the model's decode
steps, `moe.w1_decode`), the prefill's never.  The DRISCs run from the end of the generator's warm-up (`start`, the
device idle at a step's start) to its release (`stop`, also at the process's exit: atexit; a process killed with them
running leaves them polling, harmless, until the next device open puts the DRAM cores' RISCs in reset).

    land = W1Landing(mesh, C)                        # the landing (every core, lock-step): C slots of a block + seal
    land.arm_model(model, mesh_tokens)               # each MoE layer's spec on its TtMoe (the decode step's order)
    ... the warm-up and the captures ...
    land.start()                                     # the DRISCs, the device idle at a step's start
    ... the replays ...
    stats = land.stop()                              # windows, skipped, a window's cycles (chip 0)
    land.disarm_model(model); land.release()

C up to MAX_C (24): the landing as large as the CBs allow (C = 10 at 8 / 32 users, 13 with GLM53_W1_SHALLOW_TOKENS=32
and TT_SDPA_DECODE_TREE_CB=1); the DRISC's copy then runs on past the end window, beside the next layer's chain (the
chain's DRAM-reading ops slow a little there).  Switches (the values the same either way):
  GLM53_W1_CUT=1             the DRISC stops a window at the next experts op's start (its gate | up streams' signal,
                             mailbox word 12; read when the experts op's programs are made and at `start`; off by
                             default)
  GLM53_W1_SHALLOW_TOKENS=N  `arm_model`'s down ring 2 / h slots 2 from N tokens a mesh row (64 by default: 128 users
                             K = 3; 32: 32 users K = 7 too, ~196 KB more room for the landing on the down cores)
  GLM53_W1_FILL=items         the slots past the popular table's (8 a column) filled so that each of the experts op's
                             items (its units in expert order, 8 a group) gets as few landing experts as the others, in
                             place of the lowest ids (which all fall in item 0, whose reads then hit the item's floor;
                             synthetic weights: 32 users -0.48 ms at C 12).  "low": the lowest ids.  Only C past the
                             popular table's 8 differs.  "items" is the default (the real weights at C 12, 4
                             alternating pairs: 32 users -0.31 ms (se 0.10), 8 users -0.07 (se 0.05))

Which experts (`chosen_columns`): a list per mesh column (each column holds 72 experts, 72 c .. 72 c + 71: local ids
0 .. 71, the shared expert 72): the shared expert's slot where this call's experts op computes it (TtMoe._shared_mode:
1 the shared column's, 2 every column's), then the column's most popular routed experts of the layer
(GLM53_W1_POPULAR: a JSON of {layer: [column 0's ids by popularity, ..]}, w1_popular.json beside this file by default:
real-weight routing of technical text and the model's own answers), or the lowest local ids (GLM53_W1_POPULAR=0).  The
DRISCs of a chip fill its column's; the seals name the expert, so a list that does not match the op's only costs the
reads.

The DRISC counts windows from its start (window w fills layer (w + 1) mod len(layers)): a count out of step only
costs speed (the seals send the reads to DRAM).  Only the armed layers' decode calls signal it, in the step's order.
"""

from __future__ import annotations

import atexit
import json
import os
import statistics
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

import ttnn

DRISC = ttnn._ttnn.operations.experimental  # glm_drisc_* (tt-metalium/experimental/glm_drisc.hpp)
KERNEL = "models/demos/blackhole/glm53_flash/ttnn/kernels/w1_landing/drisc_w1.cpp"
BLOCK = 34816  # a W1 block (16 K tiles x (gate, up)) and a W2 block (8 rows x 4 column tiles), BFP8
SLOT = BLOCK + 64  # a block and its seal
# a block's bytes by the experts' format (32 tiles): BFP8 34,816, BFP4 18,432 (the progressive form's base and
# correction, and all-BFP4 experts: GLM53_W1_PROG)
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
EXTRA = 2048 + 64  # after the slots: the streams' seals (256 B a ring place, 8 places), then their counts
RING = 128  # windows recorded a DRISC
COLS, PER = 4, 72  # mesh columns, a column's routed experts
MAX_C = 24  # slots (drisc_w1.cpp's seals; the experts op's common args take C up to 24)
# The landing by default: the DFlash2 verify step's shapes measured and stressed, by the step's tokens a mesh row (the
# experts op's form): 8 and 32 (8 / 32 users K = 7, and 64 users K = 3 as 32 tokens) C = 5, 64 (128 users K = 3, 64
# users K = 7) C = 3; every other shape (plain decode, MTP, 16 tokens a row, 128 or more) none
DEFAULT_C = {8: 5, 32: 5, 64: 3}
# ... and for the fixed points measured again on the real weights, by (users a mesh row, tokens a user): 128 users
# K = 3 C = 5 (with the experts op's gate | up ring 5 deep at 64 tokens a mesh row, its factory's default), with
# GLM53_W1_CUT=1 and TT_SDPA_DECODE_TREE_CB=1 as measured (`landing_c` sets them; alone they change nothing): the
# DFlash2 step against the earlier default (C 3, the ring 3 deep) -1.38 ms (the values the same).  At 8 / 32 users
# K = 7, C 12 alone (-0.49 / -0.14 ms) clashes with the prefill's circular buffers (the mHC stream read's end at
# 1,140,096 B against the C 12 landing's foot at ~1.11 MB: the generator's warm-up stopped), which the speed test (no
# prefill) never ran.  Other shapes of the same tokens a mesh row (e.g. 64 users K = 3) keep DEFAULT_C
# ... 8 / 32 users K = 7 at C 10: the DFlash2 step against C 5 -0.80 / -0.78 ms on the real weights (two other
# processes: -0.67 / -0.89), C 12 alone no better.  The prefill's programs then fit under the landing with three L1
# switches, each the same values and within the prefill's noise (MEASURED_ENV_SHAPE): GLM53_MHC_PART_MIX (the mHC
# groups forms' partials CB on the mix cores only: the slice cores' CBs 256 KB lower), TT_SDPA_Q_SLOTS_MAX=1 (the dense
# chunked SDPA's Q in one slot where a core has two chunks: 128 KB) and GLM53_LN_GAMMA_STREAM=1 (the drafter's context
# norm's gamma a width block at a time: 240 KB; GLM53_MHC_PART_MIX alone met that norm at 32 seats).  The highest CB
# end left is the prefill MoE's gate | up (mp_gu5, 1,101,760 B) under the landing's foot less all_reduce_async's 64 KB
# landing areas (~1,110,900 B on their 64 cores)
MEASURED_C = {(16, 4): 5, (4, 8): 12, (1, 8): 12}
MEASURED_ENV = {"GLM53_W1_CUT": "1", "TT_SDPA_DECODE_TREE_CB": "1"}
_L1_SWITCHES = {"GLM53_MHC_PART_MIX": "1", "TT_SDPA_Q_SLOTS_MAX": "1", "GLM53_LN_GAMMA_STREAM": "1"}
# ... and C 12 there: the two slots past C 10 are experts of the op's last items, copied at their reader's start
# (GLM53_W1_LATE=10: the DRAM idle there, x not yet in) in place of the window before's end, where C 12's copy ran into
# the next layer's chain and the drafter (the DFlash2 step at 32 users against C 10 -0.195 / -0.255 ms on the real
# weights; all at that end window +0.085).  The prefill's programs fit
# under the C 12 landing with mp_gu5's tilized x one row deep (moe_prefill.py; the same values)
_C12_FORM = {**_L1_SWITCHES, "GLM53_W1_LATE": "10"}
MEASURED_ENV_SHAPE = {(4, 8): _C12_FORM, (1, 8): _C12_FORM}
# The progressive form at a budget that reads (almost) no corrections (GLM53_MOE_PROG_F <= 0.5: the fast preset C; its
# busiest column's budget m is always 0), by default (`w1_prog`): its BFP4 base blocks (18,496 B a slot with the seal)
# at C 10 for 8 / 32 users K = 7, the most the serving flow's prefill programs leave (the mHC write helper's CBs end at
# 1,248,640 B under 1,441,152 - 18,496 C), no correction slot, no GLM53_W1_LATE (C 10 without it the faster: the
# DFlash2 step on the real weights 31.19 -> 30.24 ms at 32 users and 24.64 -> 24.21 ms at 8 with
# GLM53_MOE_PROG_FAST=2, against C 22 + GLM53_W1_LATE=14's -0.73 at 32); at 128 users K = 3 (16 users a mesh row) C 9
# with its slots 6 - 8 copied at their reader's start (GLM53_W1_LATE=6), the down ring 3 in its spec (`arm_model`) and
# the down control's early h release and faster 0/1 tiles (moe_decode_experts' dn_early) as without a landing: the
# DFlash2 step on the real weights 40.190 -> 39.850 ms (10,015 tok/s).  Without that down control a landing there
# gained nothing (C 9 + the down ring 3 +0.24 ms; C 6 / C 9 + GLM53_W1_LATE=6 +0.09 / +0.01 ms), nor C 6 with it
# (-0.15 ms, within the noise).  Other shapes none
PROG_MEASURED_C = {(4, 8): 10, (1, 8): 10, (16, 4): 9}
PROG_MEASURED_ENV_SHAPE = {(4, 8): _L1_SWITCHES, (1, 8): _L1_SWITCHES, (16, 4): {**_L1_SWITCHES, "GLM53_W1_LATE": "6"}}
POPULAR = Path(__file__).with_name("w1_popular.json")
# MoeDecodeExpertsParams::w1_recv's places (moe_decode_experts_device_operation_types.hpp W1_SPEC_*)
SPEC_DRISCS, SPEC_EXPERTS = 5, 23


def default_c(users_per_row: int, tokens_a_user: int) -> int:
    """The landing's slots by default for a DFlash2 verify step of `users_per_row` users a mesh row, `tokens_a_user`
    (K + 1) tokens each (MEASURED_C for the measured fixed points, else DEFAULT_C by its tokens a mesh row; 0: no
    landing)."""
    return MEASURED_C.get((users_per_row, tokens_a_user), DEFAULT_C.get(users_per_row * tokens_a_user, 0))


def streaming_profiler() -> bool:
    """TT_METAL_STREAMING_PROFILER on, as tt-metal's rtoptions read it (a value starting with "1"): its relays hold
    the DRAM cores' DRISCs for the process, the DRISCs the landing's copies run on."""
    return os.environ.get("TT_METAL_STREAMING_PROFILER", "")[:1] == "1"


def prog_budget() -> Optional[float]:
    """GLM53_MOE_PROG_F as a number (the progressive form's budget; None: not the progressive form)."""
    v = os.environ.get("GLM53_MOE_PROG_F")
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def w1_prog() -> bool:
    """The landing for the progressive form (the fast presets: each slot's id a base, the expert, or a correction,
    op_slots + the expert, at the weights' own places; the experts op's MOE_PROG branches read either from the landing)
    and for all-BFP4 experts, with slots of the format's blocks (`TILE_BYTES`: BFP4's 18,432 B, about twice BFP8's slots
    in the same L1).  GLM53_W1_PROG=1: on, 0: off; unset: on for the progressive form at a budget of 0.5 or less (preset
    C, `PROG_MEASURED_C`), off otherwise (presets A / B, all-BFP4 experts)."""
    v = os.environ.get("GLM53_W1_PROG", "")
    if v != "":
        return v == "1"
    f = prog_budget()
    return f is not None and f <= 0.5


def landing_dtype(experts=None):
    """The routed experts' weight blocks' format the landing copies: BFP4 for the progressive form (its base and
    correction), else the model's routed experts' (`experts`; None: BFP8)."""
    if os.environ.get("GLM53_MOE_PROG_F") is not None:
        return ttnn.bfloat4_b
    return ttnn.bfloat8_b if experts is None else experts


def w1_blocked(experts=None) -> str:
    """Why this process's experts op cannot take the landing ("" when it can).  The DRISCs' copies and the op's landing
    reads take a slot's block, then its seal after it, of the weights' format (`TILE_BYTES`).  Without GLM53_W1_PROG
    only BFP8 experts in the plain layout: the progressive form (GLM53_MOE_PROG_F, the fast presets: each slot a BFP4
    base and a BFP4 correction, twice the slots) and experts in another format (`experts`, the model's:
    GLM53_EXPERT_DTYPE / GLM53_WEIGHT_DTYPE) run without it; with GLM53_W1_PROG=1 the progressive form and all-BFP4
    experts too (`w1_prog`).  Experts of mixed formats (GLM53_EXPERT_LAYERS) never: one landing, one block size."""
    if os.environ.get("GLM53_EXPERT_LAYERS"):
        return "routed experts of mixed formats (GLM53_EXPERT_LAYERS): one landing holds one format's blocks"
    if os.environ.get("GLM53_MOE_PROG_F") is not None and not w1_prog():
        return (
            "the progressive form (GLM53_MOE_PROG_F: the fast presets) at a budget above 0.5, which the experts op "
            "takes with the landing only with GLM53_W1_PROG=1"
        )
    if os.environ.get("GLM53_MOE_PROG_F") is None and experts is not None and experts != ttnn.bfloat8_b:
        if experts not in TILE_BYTES or not w1_prog():
            return f"routed experts in {experts}; the landing's blocks are BFP8's (BFP4 too with GLM53_W1_PROG=1)"
    return ""


def landing_c(dflash_shape: Optional[Tuple[int, int]] = None, experts=None) -> int:
    """The landing's slots at the model's load: GLM53_W1 when set (0: off, C: that many, up to MAX_C); else the default
    for the DFlash2 verify step the process will serve (`dflash_shape`: (users a mesh row, tokens a user),
    TtGlm53Model's), none without one.  None with the streaming profiler on (its relays hold the DRISCs) or with routed
    experts the landing cannot copy (`w1_blocked`: the fast presets' progressive form, experts not in BFP8; `experts`:
    the model's routed experts' dtype): the process then runs as GLM53_W1=0, the reason printed, and the measured
    shapes' switches (MEASURED_ENV) are not set."""
    given = os.environ.get("GLM53_W1", "")
    prog = os.environ.get("GLM53_MOE_PROG_F") is not None  # (the progressive form: its own measured shapes)
    measured = PROG_MEASURED_C if prog else MEASURED_C
    shapes = PROG_MEASURED_ENV_SHAPE if prog else MEASURED_ENV_SHAPE
    if given != "":
        c = int(given)
    elif not dflash_shape:
        c = 0
    else:
        c = measured.get(tuple(dflash_shape), 0) if prog else default_c(*dflash_shape)
    why = w1_blocked(experts) if c else ""
    if why:
        how = f"GLM53_W1={given}" if given != "" else f"the default for the DFlash2 step {tuple(dflash_shape)}"
        print(f"GLM53_W1 off: {why}; no landing (C {c} by {how}): this process runs as GLM53_W1=0", flush=True)
        return 0
    if given == "" and dflash_shape and c and tuple(dflash_shape) in measured and not streaming_profiler():
        for k, v in {**MEASURED_ENV, **shapes.get(tuple(dflash_shape), {})}.items():  # (read when the
            os.environ.setdefault(k, v)  # programs are made, after this: the form measured)
    if c and streaming_profiler():
        how = f"GLM53_W1={given}" if given != "" else f"the default for the DFlash2 step {tuple(dflash_shape)}"
        print(f"GLM53_W1 off: TT_METAL_STREAMING_PROFILER is on and its relays hold the DRAM cores' DRISCs, which the "
              f"landing's copies run on; no landing (C {c} by {how}): this process runs as GLM53_W1=0",
              flush=True)  # fmt: skip
        return 0
    return c


def popular_table(path: Optional[str] = None) -> Optional[dict]:
    """GLM53_W1_POPULAR (a path; "0": none, the lowest ids) -> {layer: [[ids by popularity] per column]}."""
    where = os.environ.get("GLM53_W1_POPULAR", str(POPULAR)) if path is None else path
    if where in ("", "0") or not Path(where).exists():
        return None
    raw = json.loads(Path(where).read_text())
    raw = raw.get("popular_local_ids", raw)
    return {int(k): v for k, v in raw.items()}


def late_slots() -> int:
    """GLM53_W1_LATE=k (0 / unset: none): slots k .. of a layer are its experts of the op's last items, copied whole
    once the experts op that reads them has started (its start, x not yet in: the DRAM idle there), in place of the
    window before's end beside the next layer's chain; with GLM53_W1_CUT only (that op's start is its cut signal).  The
    DRISCs' mailbox word 11 (drisc_w1.cpp); the values the same.  (The progressive form's correction slots go before k:
    only the routed bases from k on are taken from the last items.)"""
    return int(os.environ.get("GLM53_W1_LATE", "0") or 0)


def chosen_columns(module, mesh_tokens: int, c: int, popular: Optional[dict] = None) -> List[List[int]]:
    """A layer's C experts (local ids) per mesh column: the shared expert's slot (op_slots - 1) where this call's
    experts op computes it, then the column's most popular routed ones (`popular`, the layer's), else the lowest.
    The progressive form (`module.prog`, GLM53_W1_PROG): the ids are the slots' places in the weights, a base the
    expert, a correction op_slots + the expert (a step reads a correction only for the column's experts with the
    largest routing weight sums: the shared expert's, weight 1 for every token, first, then the popular ones'):
    GLM53_W1_PROG_CORR (default 0: preset C reads almost no corrections, its busiest column's budget 0) of the slots
    hold corrections, the shared expert's (when present) and the most popular experts', each right after its base; the
    rest bases as above."""
    mode = module._shared_mode(mesh_tokens, mesh_tokens) if module.op_slots > module.op_experts else 0
    ranked = popular.get(int(module.layer)) if popular is not None and module.layer is not None else None
    items = os.environ.get("GLM53_W1_FILL", "items") == "items"
    prog = bool(getattr(module, "prog", False))
    corr = min(c // 2, int(os.environ.get("GLM53_W1_PROG_CORR", "0") or 0)) if prog else 0
    # GLM53_W1_LATE=k (`late_slots`, its slots copied at their reader's start): slots k .. from the last items
    late = late_slots() or c
    out = []
    for col in range(COLS):
        has_shared = mode == 2 or (mode == 1 and col == module.shared_col)
        ids = [module.op_slots - 1] if has_shared else []
        order = list(ranked[col]) if ranked is not None else []
        if corr:  # (the progressive form: corrections right after their bases, the shared expert's first)
            heads = ids + [e for e in (order or range(PER)) if e not in ids]  # (no table: the lowest)
            ids, k = [], 0
            for e in heads:
                want = 2 if k < corr else 1
                if len(ids) + want > c:
                    break
                ids.append(e)
                if want == 2:
                    ids.append(module.op_slots + e)
                    k += 1
        if items:  # the ranked ones, then the items with the fewest landing experts first
            ids += [e for e in order if e not in ids][: c - len(ids)]
            while len(ids) < c:
                count = [sum(1 for e in ids if e < PER and e // 8 == i) for i in range(PER // 8)]
                free = [i for i in range(PER // 8) if any(e not in ids for e in range(8 * i, 8 * i + 8))]
                i = min(free, key=lambda i: (count[i], -i if len(ids) >= late else i))
                ids.append(next(e for e in range(8 * i, 8 * i + 8) if e not in ids))
        else:
            order += [e for e in range(PER) if e not in order]
            ids += [e for e in order if e not in ids][: c - len(ids)]
        out.append(ids[:c])
    return out


class W1Landing:
    def __init__(self, mesh, c: int, dtype=None):
        """C slots of the weights' format's blocks (`dtype`: the routed experts' weights the op reads, BFP8 by default;
        BFP4, the progressive form's and all-BFP4 experts', with GLM53_W1_PROG: `TILE_BYTES`)."""
        assert 1 <= c <= MAX_C, c
        self.dtype = ttnn.bfloat8_b if dtype is None else dtype
        assert self.dtype in TILE_BYTES, f"w1_landing: no blocks of {self.dtype}"
        self.block = 32 * TILE_BYTES[self.dtype]
        self.slot = self.block + 64
        if streaming_profiler():  # (a test's own landing: the model's is never made then, `landing_c`)
            raise RuntimeError("w1_landing: TT_METAL_STREAMING_PROFILER's relays hold the DRAM cores' DRISCs")
        self.mesh, self.c = mesh, c
        self.words = (c * self.slot + EXTRA) // 4
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9))])
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [1, self.words], ttnn.ShardOrientation.ROW_MAJOR),
        )
        # zeros: no slot sealed with any (layer, expert) (tags are 1 ..), the counts 0
        self.tensor = ttnn.from_torch(
            torch.zeros(120, self.words, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.addr = self.tensor.buffer_address()
        assert self.addr % 64 == 0, f"the landing at {self.addr:#x} is not 64-byte aligned"
        base, size = DRISC.glm_drisc_region(mesh)
        senders = DRISC.glm_drisc_sender_cores(mesh)
        self.cores = [tuple(senders[b][0]) for b in range(len(senders))]  # role 0 a bank
        self.nocs = DRISC.glm_drisc_noc_coords(mesh, self.cores)
        self.mailbox, self.stage = base, base + 4096
        self.table = self.stage + 2 * self.block
        self.room = base + size - self.table  # the table's bytes
        assert len(self.cores) == 8 and self.room > 64, (len(self.cores), base, size)
        self.cols = int(tuple(mesh.shape)[1])
        assert self.cols == COLS, f"w1_landing: a mesh of {COLS} columns, not {tuple(mesh.shape)}"
        self.running = False
        self.gen = 0
        # the armings' epoch (in the tags, `arm`): from a random value, so that not even another process's landing at
        # this address (L1 is not cleared between processes; this one's starts zeroed) shares this one's tags
        self.epoch = int.from_bytes(os.urandom(2), "little") % (1 << 10)
        # the SDPA's K / V in one ring (the paged chunked form, V the same bytes as K: DSA's prefill; values the same)
        # for every program made from here on: its CBs then end below the landing at C up to 5
        os.environ.setdefault("TT_SDPA_KV_SHARED", "1")
        # ... and the prefill's K-blocked matmuls (GLM53_PREFILL_KBLOCK) within 900 KB of CBs, 1,024 rows as blocks of
        # 512 (the values of the default 1000 KB, whose CBs would reach into the landing: `common.kblock_for_w1`)
        from .common import kblock_for_w1

        kblock_for_w1()
        self.tables = None  # per mesh column: the DRISCs' table words
        self.specs = None  # per MoE layer: the op's spec
        print(f"GLM53_W1 landing: C {c} of {self.dtype} blocks, {self.words * 4} B a core on the 12 x 10 grid at "
              f"{self.addr:#x} (to {self.addr + self.words * 4:#x}); DRISCs {self.cores}, mailbox {base:#x}",
              flush=True)  # fmt: skip

    def tag(self, i: int) -> int:
        """Layer i's seal tag under the current arming (16 bits: the epoch's 10, then i + 1 in 6; never 0, the
        landing's zeros)."""
        return (self.epoch << 6) | (i + 1)

    def spec(self, tag: int, columns: Sequence[Sequence[int]], dn_depth: int = 0, h_slots: int = 0) -> List[int]:
        """The experts op's landing spec (MoeDecodeExpertsParams::w1_recv): tag, C, the landing, a slot's bytes, the
        DRISCs' mailbox and NoC x, y, the down cores' W2 ring depth and h slots (0: the op's), the experts by column."""
        out = (
            [tag, self.c, self.addr, self.slot, self.mailbox]
            + [v for xy in self.nocs for v in xy]
            + [dn_depth, h_slots]
        )
        assert len(out) == SPEC_EXPERTS, len(out)
        for ids in columns:
            assert len(ids) == self.c, (tag, columns)
            out += list(ids)
        return out

    def arm(self, layers: Sequence[Tuple[object, object, Sequence[Sequence[int]]]], dn_depth: int = 0,
            h_slots: int = 0, epoch: Optional[int] = None) -> List[List[int]]:  # fmt: skip
        """`layers`: the decode step's MoE layers in order, (w1, w2, the experts per mesh column) each -> the op's spec
        per layer, and the DRISCs' tables per column.  Layer i's tag: this arming's epoch << 6 | i + 1 (`tag`), so
        that no block a former arming left in the landing passes a seal check: the slots keep their bytes and seals
        until a window rewrites them, and the same (tag, expert) could name other bytes after a re-arming (another
        layer list, or other sources: a test's later step read an earlier step's blocks of the next layer under the same
        tags).  `epoch` (tests): this arming's epoch as given (the DRISCs' tables of an arming whose specs the
        op already has, e.g. its experts rearranged), else the next."""
        assert len(layers) < 1 << 6, f"w1_landing: {len(layers)} MoE layers (a tag holds 63)"
        self.epoch = (self.epoch + 1) % (1 << 10) if epoch is None else epoch
        tables = [[len(layers), self.c, self.slot, self.block, 0] for _ in range(self.cols)]
        specs, w2_shard = [], None
        for i, (w1, w2, columns) in enumerate(layers):
            # w2 [32 E I, D / 32], a down core's slice a shard (bank q mod 8, its (q div 8)-th): E blocks of BLOCK
            assert (
                w1.dtype == w2.dtype == self.dtype
            ), f"w1_landing: a landing of {self.dtype} blocks, w1 {w1.dtype} w2 {w2.dtype}"
            shard = (int(w2.shape[-2]) // 32 // 32) * (int(w2.shape[-1]) // 32) * TILE_BYTES[self.dtype]
            assert w2_shard in (None, shard), (w2_shard, shard)
            w2_shard = shard
            tag = self.tag(i)
            assert tag < 1 << 16 and all(e < 1 << 16 for ids in columns for e in ids)
            for col in range(self.cols):
                ids = list(columns[col])
                pad = [0] * (max(8, self.c) - len(ids))  # (a layer's experts: 8 words, or C above 8)
                tables[col] += [tag, w1.buffer_address(), w2.buffer_address()] + ids + pad
            specs.append(self.spec(tag, columns, dn_depth, h_slots))
        for t in tables:
            t[4] = w2_shard
            assert 4 * len(t) <= self.room, (len(t), self.room)
        self.tables, self.specs = tables, specs
        return specs

    def arm_model(self, model, mesh_tokens: int, popular: Optional[dict] = None, dn_depth: Optional[int] = None,
                  h_slots: Optional[int] = None) -> List[List[int]]:  # fmt: skip
        """Each MoE layer's spec on its TtMoe (`w1_spec`, taken by its decode calls: the programs made from here on).
        dn_depth / h_slots None: 2 / 2 at 64 tokens or more a mesh row (128 users K = 3: room for the landing on the
        down cores; the down ring 3 for the fast preset C, GLM53_MOE_PROG_F <= 0.5), else the op's."""
        from .moe import TtMoe

        if popular is None:
            popular = popular_table()
        mods = [layer.mlp for layer in model.layers if isinstance(layer.mlp, TtMoe)]
        wide = mesh_tokens // int(tuple(self.mesh.shape)[0]) >= int(os.environ.get("GLM53_W1_SHALLOW_TOKENS", "64"))
        # (the progressive form at a budget of 0.5 or less, the fast preset C: the down ring 3 with its landing, as its
        # experts op takes without one, moe_decode_experts' prog_ring3)
        budget = prog_budget()
        wide_dn = 3 if budget is not None and budget <= 0.5 else 2
        dn = (wide_dn if wide else 0) if dn_depth is None else dn_depth
        hs = (2 if wide else 0) if h_slots is None else h_slots
        specs = self.arm([(m.e_w1, m.e_w2, chosen_columns(m, mesh_tokens, self.c, popular)) for m in mods], dn, hs)
        for m, s in zip(mods, specs):
            m.w1_spec = s
        return specs

    @staticmethod
    def disarm_model(model) -> None:
        from .moe import TtMoe

        for layer in model.layers:
            if isinstance(layer.mlp, TtMoe):
                layer.mlp.w1_spec = None

    def start(self) -> None:
        """The DRISCs (8 a chip), counting windows from here: call with the device idle at a step's start."""
        assert not self.running and self.tables
        rt = []
        chips = self.mesh.get_num_devices()
        for _ in range(chips):
            for b in range(len(self.cores)):
                gu = [ttnn.CoreCoord(4 + b, c) for c in range(8)]
                dn = [ttnn.CoreCoord(q % 4, q // 4) for q in (b, b + 8, b + 16, b + 24)]
                coords = []
                for core in gu + dn:
                    v = self.mesh.worker_core_from_logical_core(core)
                    coords += [v.x, v.y]
                rt.append([self.addr] + coords)
        self.gen += 1
        for core in self.cores:
            DRISC.glm_drisc_write_l1(self.mesh, -1, core, self.mailbox, [0] * (16 + 4 * RING))  # (and the records)
            DRISC.glm_drisc_write_l1(self.mesh, -1, core, self.mailbox + 36, [self.gen])  # word 9: the generation
            cut = 1 if os.environ.get("GLM53_W1_CUT", "0") not in ("", "0") else 0
            DRISC.glm_drisc_write_l1(self.mesh, -1, core, self.mailbox + 56, [cut])  # word 14: the cut switch
            # word 11: GLM53_W1_LATE (`late_slots`; 0: none; the kernel reads it only with the cut switch)
            DRISC.glm_drisc_write_l1(self.mesh, -1, core, self.mailbox + 44, [late_slots()])
            for chip in range(chips):  # (the mesh view's order: row-major, column chip mod COLS)
                DRISC.glm_drisc_write_l1(self.mesh, chip, core, self.table, self.tables[chip % self.cols])
        DRISC.glm_drisc_start(self.mesh, KERNEL, self.cores, [self.stage, self.mailbox, self.table], rt)
        self.running = True
        atexit.register(self._at_exit)

    def _at_exit(self) -> None:
        try:
            self.stop()
        except Exception as e:  # (the device may be gone: tt-metal's teardown resets the DRAM cores anyway)
            print(f"w1_landing: stopping the DRISCs at exit: {e}", flush=True)

    def stop(self) -> dict:
        """Stop the DRISCs (each returns once its window in hand is done); chip 0's records."""
        if not self.running:
            return {}
        for core in self.cores:
            DRISC.glm_drisc_write_l1(self.mesh, -1, core, self.mailbox, [1])
        DRISC.glm_drisc_wait(self.mesh)
        self.running = False
        atexit.unregister(self._at_exit)
        wins, skipped, w1c, w2c, cuts = [], [], [], [], []
        for core in self.cores:
            box = DRISC.glm_drisc_read_l1(self.mesh, 0, core, self.mailbox, 16 + 4 * RING)
            wins.append(box[1])
            skipped.append(box[8])
            cuts.append(box[13])
            for i in range(min(box[1], RING)):
                if box[16 + 4 * i] or box[16 + 4 * i + 1]:  # (skipped windows leave no record)
                    w1c.append(box[16 + 4 * i + 1])
                    w2c.append(box[16 + 4 * i + 2])
        med = lambda v: int(statistics.median(v)) if v else 0  # noqa: E731
        return {
            "windows": (min(wins), max(wins)),
            "skipped": max(skipped),
            "cut": max(cuts),
            "w1_cycles": med(w1c),
            "w2_cycles": med(w2c),
        }

    def counts(self, chips: Sequence[int] = (0,)) -> Tuple[int, int, int, int]:
        """(gate | up blocks from the landing, gate | up read again, down halves from the landing, down read again)
        on those chips (the mesh view's order: 0 .. 3 the first mesh row's columns), summed over their op cores,
        since the landing was made.  (Each chip's whole landing read back: ~17 MB at C 4.)"""
        at = (self.c * self.slot + 2048) // 4
        gu = [12 * y + x for y in range(8) for x in range(4, 12)]
        dn = [12 * y + x for y in range(8) for x in range(4)]
        out = [0, 0, 0, 0]
        devs = ttnn.get_device_tensors(self.tensor)
        for d in chips:
            t = ttnn.to_torch(devs[d]).to(torch.int64) & 0xFFFFFFFF
            out[0] += int(t[gu, at].sum())
            out[1] += int(t[gu, at + 1].sum())
            out[2] += int(t[dn, at].sum())
            out[3] += int(t[dn, at + 1].sum())
        return tuple(out)

    def release(self) -> None:
        if self.running:
            self.stop()
        ttnn.deallocate(self.tensor)
