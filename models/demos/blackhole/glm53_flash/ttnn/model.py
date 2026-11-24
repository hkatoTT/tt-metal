"""The text model on the 8x4 mesh: each of the 8 rows decodes its own B users (one token each per step, capturable as
a trace) and prefills one request at a time in chunks (`prefill`).  The logits come back to the host in fp32.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from typing import NamedTuple

import torch

import ttnn
from models.demos.blackhole.glm53_flash.config import DSA, KDA, Glm53Config
from models.demos.blackhole.glm53_flash.ttnn import mhc as mhc_mod
from models.demos.blackhole.glm53_flash.ttnn import moe_route, w1_dense
from models.demos.blackhole.glm53_flash.ttnn.common import (
    COLS,
    HIFI4_FP32,
    ROW_AXIS,
    ROWS,
    WAVE_ROWS,
    FixedInputs,
    SpGroup,
    drop_scratch,
    eager_place,
    gather_rows,
    host_threads,
    mapper,
    put,
    row_quarter_of,
    row_split_on,
    row_tokens_gather,
    weight_dtype,
    write_rows,
)
from models.demos.blackhole.glm53_flash.ttnn.head import TtEmbedding, TtLmHead, TtRmsNorm
from models.demos.blackhole.glm53_flash.ttnn.input_pack import InputPack
from models.demos.blackhole.glm53_flash.ttnn.kda import KdaPicks, SpecKeep, TtKda, spec_flip
from models.demos.blackhole.glm53_flash.ttnn.layer import TtDecoderLayer
from models.demos.blackhole.glm53_flash.ttnn.mhc import collapse_streams, expand_streams
from models.demos.blackhole.glm53_flash.ttnn.mtp import TtMtp

# a decode / verify step's small inputs (ids, the DSA positions' buffers, MTP's keep flags) in one host write and one
# device op that copies them to their tensors (`input_pack.InputPack`, the step's first op), in place of a write each
# (~0.12-0.25 ms of host a write).  The same bits; MTP's verify step at 8 / 32 / 128 users -0.4 / -0.3 / -0.6 ms, the
# plain step -0.1 to -0.3 ms, the long cache -0.3 ms; traced steps without a hang.
# The prefill's kept hidden state (MTP's first drafts' input) sits in one buffer made with the model, before any capture
# (`_kept_hidden`): each prefill (eager, or captured: the copy inside its trace) writes it there.  The kept tensor used
# to be the chunk's own output: a captured prefill's output buffer could sit in the scratch of a trace captured before
# it, and the KDA commit's trace (captured among the chunk length 128's, before the 512's) replayed after a wave's last
# chunk wrote over it before the first drafts read it (values ~7e19, every slot's first draft wrong, depending on
# the allocations' layout); a sequence-parallel replay did not keep its hidden at all (the first drafts read an older
# one)


P = "model.language_model."
_ROW_SPLIT_SEEN: set = set()  # (the wave sizes that ran mHC over the row by tokens, printed once each)

# every prefill chunk the generator issues replays a captured trace (`capture_prefills`), not only a one-chunk wave's
# chunk: one trace per `PrefillKey` (chunk length, first or later, a long cache's pool bucket), the start in the inputs,
# and one more that writes the rows' KDA wave states into their users after a wave's last chunk (`COMMIT`).  The parts
# that depended on the start take start-independent forms with the same values bit for bit: a dense DSA's later
# chunk attends through the flexible chunked SDPA (the start in a device tensor, its user's whole page table) instead
# of chunked_flash_mla_prefill, and a long cache's pool scores are computed over the whole power-of-2 bucket (then
# masked as before) rather than up to the chunk's end.  A captured
# chunk always updates the rows' wave states and the commit follows the wave's last chunk: eager's form when a row's
# prompt ended before the last chunk; when none did, eager commits the last chunk's states directly (the same values:
# x 0 + y x 1).  Traced and eager bit for bit.  0: only a one-chunk
# wave's chunk, its commit inside
PREFILL_TRACE_ALL = os.environ.get("GLM53_PREFILL_TRACE_ALL", "1") == "1"
# two consecutive sequence-parallel waves of WAVE_ROWS tokens a row as one wave of 2 WAVE_ROWS (`prefill_sp_pair`, the
# generator's `_prefill_sp_round`): the attention layers one wave after the other (the
# values of the two waves), the rest at once.  False (a test's): the waves apart
SP_PAIR = True
# the MTP layer (45) loaded for self-speculative decode (`forward_spec`).  0: without it
MTP = os.environ.get("GLM53_MTP", "0") == "1"
# a one-row prefill's argmax of its logits on the device as well (`_device_ids`: ttnn.argmax over the same fp32
# logits, the first index of the maximum as torch's), so that a caller wanting only each request's greedy first token
# (`greedy_ids`: the generator's tasks, DFlash2's and MTP's first drafts) reads 8 ids instead of the 8 rows' logits
# (620 KB a row, every wave).  The same ids.  GLM53_PREFILL_IDS=0: the host's argmax of the logits, as before (a captured prefill then has no argmax op)
PREFILL_IDS = os.environ.get("GLM53_PREFILL_IDS", "1") == "1"
# GLM53_PREFILL_IDS_CHECK=1 (a test's): with PREFILL_IDS, every device argmax checked against the host's argmax of the
# 8 rows' logits read as before (`ids_check`: [rows checked, rows differing])
PREFILL_IDS_CHECK = os.environ.get("GLM53_PREFILL_IDS_CHECK", "0") == "1"
# a diagnostic: `capture_prefills` reports, form by form, the programs and DRAM its eager run
# before the captures added (none: that run only made what the warm-up had; its seconds the most skipping it saves)
PRERUN_STUDY = os.environ.get("GLM53_CAPTURE_PRERUN_STUDY", "0") == "1"


def _programs_and_dram(mesh) -> tuple:
    """(the mesh's program cache entries, one chip's DRAM allocated bytes), the device idle, dropped tensors freed."""
    import gc

    gc.collect()
    ttnn.synchronize_device(mesh)
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return mesh.num_program_cache_entries(), int(view.total_bytes_allocated_per_bank) * int(view.num_banks)


class PrefillKey(NamedTuple):
    """The captured prefill a chunk replays (`TtGlm53Model.prefill_key`)."""

    t: int  # the chunk length
    first: bool  # at position 0 (a request begins: KDA from zeros, DSA within the chunk)
    last: bool  # the trace writes the rows' KDA states into their users (a one-chunk wave's, without TRACE_ALL)
    bucket: int = 0  # a long cache's later chunk: the pool pages it scores (`TtDsa.pool_bucket`)
    sp: bool = False  # one request over the mesh rows (`prefill_sp`; first: its first wave)
    # a pair of sequence-parallel waves in one (`prefill_sp_pair`, t their two segments a row): the second wave's
    # bucket (`bucket` the first's); -1 otherwise
    pair: int = -1


COMMIT = PrefillKey(0, False, True)  # (GLM53_PREFILL_TRACE_ALL) the commit after a wave's last captured chunk


def l1_probe(mesh, label: str) -> None:
    """GLM53_L1_PROBE=1: print where a new L1 buffer lands (just below the persistent L1 buffers; every program's
    static CBs must end below it)."""
    if os.environ.get("GLM53_L1_PROBE") != "1":
        return
    probe = ttnn.from_torch(
        torch.zeros(32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    print(f"L1 probe, {label}: a new L1 buffer at {probe.buffer_address()}", flush=True)
    ttnn.deallocate(probe)


def prealloc_fabric_ops(mesh, d: int) -> None:
    """The persistent state of the one-shot fabric ops that are on (the row all-reduce, the MoE's column collectives:
    DRAM landing areas and a few semaphores in L1), before any program is built or trace captured."""
    from models.demos.blackhole.glm53_flash.ttnn import common

    if common.TP_AR_CUSTOM:
        from models.demos.blackhole.glm53_flash.ttnn.row_all_reduce import prealloc

        prealloc(mesh, d)
    from models.demos.blackhole.glm53_flash.ttnn import head

    if head.HEAD_PICK:
        from models.demos.blackhole.glm53_flash.ttnn.head_pick import prealloc as head_prealloc

        head_prealloc(mesh)
    from models.demos.blackhole.glm53_flash.ttnn.col_collectives import prealloc as col_prealloc

    col_prealloc(mesh, d, ar=True)
    from models.demos.blackhole.glm53_flash.ttnn import moe

    moe.prealloc_h1(mesh, d)  # (moe.H1, the column gather in the experts op: landing buffers, two global semaphores)
    from models.demos.blackhole.glm53_flash.ttnn.x2d_gather import prealloc as x2d_prealloc

    x2d_prealloc(mesh, d)  # (the prefill's 2D x all-gather: its landings and global semaphores)


def prealloc_sp_ops(mesh, channels: int, heads: int | None = None, dh: int | None = None) -> None:
    """The sequence-parallel prefill's one-shot fabric gather of the KDA conv inputs (`TtKda._prefill_sp`), when on;
    and its maps exchange's landing and semaphores (`kda.MAPS_X`, `heads` a chip of `dh`)."""
    from models.demos.blackhole.glm53_flash.ttnn import dsa, kda

    if dsa.PREFILL_SP:
        from models.demos.blackhole.glm53_flash.ttnn.col_collectives import prealloc_gather

        prealloc_gather(mesh, channels, kda.SP_GATHER_WORKERS)
        if kda.maps_x_any() and heads is not None:
            from models.demos.blackhole.glm53_flash.ttnn.kda_maps_x import prealloc as maps_x_prealloc

            maps_x_prealloc(mesh, heads, dh)


def decode_step(fn):
    """A decode step (`forward`, `forward_spec`): its MoE layers route by moe_route.route_only (ROUTE_FAST), its io's
    idle tokens choosing no expert (`moe_route.LiveTokens`)."""

    @functools.wraps(fn)
    def step(self, io, *args, **kwargs):
        with moe_route.decoding(getattr(io, "live", None)):
            return fn(self, io, *args, **kwargs)

    return step


class TtGlm53Model:
    def __init__(
        self,
        mesh,
        checkpoint,
        config: Glm53Config | None = None,
        num_layers: int | None = None,
        max_seq: int = 2048,
        expert_dtype=None,
        dtype=None,
        head_dtype=None,
        mtp: bool | None = None,
        dflash_shape: tuple[int, int] | None = None,
    ):
        """dtype: the large matrices' precision (`common.weight_dtype`, GLM53_WEIGHT_DTYPE); expert_dtype
        (GLM53_EXPERT_DTYPE: bfp4 is a lower-precision trial) and head_dtype
        (GLM53_HEAD_DTYPE) override it for the routed experts and the LM head.  dflash_shape: (users a mesh row,
        tokens a user) of the DFlash2 verify step this model will serve (its Glm53Generator's users_per_row and
        K + 1), which sizes the experts' L1 landing by default (GLM53_W1, `w1_landing.landing_c`); None: no landing
        unless GLM53_W1 asks."""
        host_threads()
        # The L1 landing of the next MoE layer's chosen experts' weights (GLM53_W1; on by default for the DFlash2
        # verify step's measured shapes: `w1_landing.landing_c` from `dflash_shape`, GLM53_W1=0 off, GLM53_W1=C that
        # many slots; off with TT_METAL_STREAMING_PROFILER, whose relays hold the DRISCs): the landing held from here
        # for the process, before every program, so that each is made with it in L1 (a static CB over it fails at its
        # program's allocation); the SDPA (DSA's prefill) with its K and V in one ring (TT_SDPA_KV_SHARED, set by the
        # landing) at 2 slots (TT_SDPA_KV_SLOTS_MAX; both read at each SDPA program's creation, the values the same: its
        # CBs end below the landing at C up to 5).  A generator arms the MoE layers for its decode step and runs the
        # DRISCs from its warm-up's end (Glm53Generator)
        self.w1 = None
        self.dflash_shape = tuple(dflash_shape) if dflash_shape else None
        dtype = dtype or weight_dtype()
        if expert_dtype is None:
            experts = os.environ.get("GLM53_EXPERT_DTYPE")
            expert_dtype = weight_dtype(experts) if experts else dtype
        from .w1_landing import landing_c

        # (none for routed experts the landing cannot copy: the fast presets' progressive form, experts not in BFP8)
        w1_c = landing_c(self.dflash_shape, experts=expert_dtype)
        # the row all-reduce's narrow landing areas by default for this decode shape (`common.ar_land_narrow_default`)
        from .common import ar_land_narrow_default

        ar_land_narrow_default(self.dflash_shape)
        if w1_c:
            os.environ.setdefault("TT_SDPA_KV_SLOTS_MAX", "2")
            from .w1_landing import W1Landing, landing_dtype

            self.w1 = W1Landing(mesh, w1_c, dtype=landing_dtype(expert_dtype))  # (BFP4's blocks: GLM53_W1_PROG)
        if head_dtype is None:
            head = os.environ.get("GLM53_HEAD_DTYPE")
            head_dtype = weight_dtype(head) if head else dtype
        self.dtypes = {"weights": dtype, "experts": expert_dtype, "head": head_dtype}
        # GLM53_EXPERT_LAYERS="bfp4:3-20,30": those layers' routed experts in that precision, the others' expert_dtype
        # (a low-precision pattern mixed by layer)
        layer_experts = {}
        spec = os.environ.get("GLM53_EXPERT_LAYERS")
        if spec:
            name, ranges = spec.split(":")
            for part in ranges.split(","):
                lo, _, hi = part.partition("-")
                for i in range(int(lo), int(hi or lo) + 1):
                    layer_experts[i] = weight_dtype(name)
            self.dtypes["experts " + spec] = weight_dtype(name)
        self.mesh = mesh
        self.config = config or Glm53Config.from_dict(checkpoint.config)
        n = self.config.num_layers if num_layers is None else num_layers
        # the fabric ops' persistent state first, before any program is built or trace captured
        l1_probe(mesh, "before the fabric ops' L1")
        prealloc_fabric_ops(mesh, self.config.hidden_size)
        if any(kind == KDA for kind in self.config.layer_types[:n]):
            prealloc_sp_ops(
                mesh,
                3 * self.config.kda_num_heads // COLS * self.config.kda_head_dim,
                self.config.kda_num_heads // COLS,
                self.config.kda_head_dim,
            )
        l1_probe(mesh, "after the fabric ops' L1")
        names = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}
        print("weights: " + ", ".join(f"{k} {names.get(v, v)}" for k, v in self.dtypes.items()), flush=True)
        self.embed = TtEmbedding(mesh, checkpoint.tensor(P + "embed_tokens.weight"))
        self.layers = []
        for i in range(n):
            self.layers.append(
                TtDecoderLayer(
                    mesh,
                    checkpoint,
                    self.config,
                    i,
                    max_seq,
                    expert_dtype=layer_experts.get(i, expert_dtype),
                    dtype=dtype,
                )
            )
            if hasattr(checkpoint, "clear"):  # a transformed-weight view caches per layer
                checkpoint.clear()
            if hasattr(checkpoint, "drop_layer"):  # (the layer's bytes out of the page cache: checkpoint.DROP_BEHIND)
                checkpoint.drop_layer(i)
            print(f"device layer {i} loaded", flush=True)
        self.norm = TtRmsNorm(mesh, checkpoint.tensor(P + "norm.weight"), self.config.rms_norm_eps)
        self.lm_head = TtLmHead(mesh, checkpoint.tensor("lm_head.weight"), dtype=head_dtype)
        self.mtp = None
        if MTP if mtp is None else mtp:
            self.mtp = TtMtp(mesh, checkpoint, self.config, max_seq, expert_dtype=expert_dtype, dtype=dtype)
            print("device MTP layer loaded", flush=True)
            if hasattr(checkpoint, "drop_layer"):  # (the MTP layer's, after the decoder layers)
                checkpoint.drop_layer(self.config.num_layers)
        drop_scratch()  # (the upload's reused host buffers, common.UPLOAD_SCRATCH)
        self.ids_check = [0, 0]  # PREFILL_IDS_CHECK: [rows checked, rows differing]
        # a test's (tests/test_model_device_wikitext.py, GLM53_WT_MODE=prefill_all): a list -> every eager prefill wave
        # (`prefill`, `prefill_sp`, `prefill_sp_pair`) leaves its logits at every position there, per chip its quarter
        # of the vocabulary [1, 1, T, V / 4] fp32 (the head's matmul without the row gather), with each row's segments
        # (row, offset in the row, first position, count, the request's own row); the logits returned are zeros
        self.logits_all = None
        # the prefill's kept hidden state, [1, 1, 1, hidden] bf16 per chip, before any capture
        self._kept_hidden = (
            put(mesh, torch.zeros(1, 1, 1, self.config.hidden_size), dims=(None, None), dtype=ttnn.bfloat16)
            if self.mtp is not None
            else None
        )
        # DFlash2's taps: with `taps` a list, `forward_spec` and the prefill append to it the per-token
        # mean of the streams entering each layer of `tap_layers` (the drafter's target layers + 1), in order
        self.taps: list | None = None
        self.capture_taps = False  # the prefill traces keep their taps as outputs (`capture_prefills`, DFlash2)
        self.tap_layers: frozenset = frozenset()
        self.dsa = next((layer.attention for layer in self.layers if layer.kind == DSA), None)
        self.has_kda = any(layer.kind == KDA for layer in self.layers)
        self._io: dict = {}
        self._first: dict = {}  # users -> FirstIO (`mtp_first_drafts`'s fixed inputs and its trace)
        self._plain_traces: list = []  # (the KDA ping-pong) (trace, states, parity) of `capture`'s plain decodes
        l1_probe(mesh, "the model loaded")

    def empty_state(self, batch: int, spec: int = 0) -> list:
        """The layers' states, then (with the MTP layer) its attention's.  spec: KDA's checkpoints for MTP's verify
        step with that many drafts (`forward_spec`)."""
        states = [layer.empty_state(batch, spec=spec) for layer in self.layers]
        if self.mtp is not None:
            states.append(self.mtp.empty_state(batch))
        return states

    # -- the KDA verify states' ping-pong (GLM53_KDA_SPEC_PINGPONG) ----------------------------------------------------
    # With it (the verify form, `empty_state` with spec >= 1) each KDA layer holds two sets of its verify state (the
    # conv history, the start state, the checkpoint rows and the step's rows); a verify step reads the set its fields
    # hold (the parity p) and writes the other.  So a verify trace is captured once per parity (`kda_spec_set_parity`
    # before each capture), and every verify step issued, eager or replayed, is followed by `kda_spec_launched` (the
    # fields then hold what it wrote); `kda_spec_undo` takes the last one back (the fields back to what it read,
    # unchanged: running it again with other inputs reads the same bits).  A prefill's commit writes the fields' set
    # (its traced commit captured once per parity, the replay picks); after it the step before can't be taken back
    # (depth 1).  Plain decode steps work on the fields' set in place: a captured one replays at the parity of its
    # capture only (`replay` checks).
    def _kda_pp(self, states: list) -> list:
        return [st for st, layer in zip(states, self.layers) if layer.kind == KDA and st.spec_alt is not None]

    def kda_spec_parity(self, states: list) -> int | None:
        """The set the KDA layers' fields hold (0 / 1), or None without the ping-pong."""
        pp = self._kda_pp(states)
        return pp[0].spec_parity if pp else None

    def kda_spec_set_parity(self, states: list, parity: int) -> None:
        """The fields as set `parity` (no device work): before capturing a verify step for that parity."""
        for st in self._kda_pp(states):
            if st.spec_parity != parity:
                spec_flip(st)

    def kda_spec_launched(self, states: list) -> None:
        """After a verify step is issued (eager, or a replay of the trace of the current parity): the set it wrote is
        the one the next step reads.  Until a prefill's commit, `kda_spec_undo` may take it back."""
        for st in self._kda_pp(states):
            spec_flip(st)
            st.spec_pending = True

    def kda_spec_undo(self, states: list) -> None:
        """Take back the last verify step issued (`kda_spec_launched`): the fields back to the set it read, which it
        left unchanged.  Then (after a prefill's commit into it, if any) the step again with the real inputs and the
        trace of the same parity reads the same state bits.  Only once a step, and not after a prefill's commit."""
        pp = self._kda_pp(states)
        if not pp:
            raise RuntimeError(
                "kda_spec_undo: the KDA states have no ping-pong (GLM53_KDA_SPEC_PINGPONG=1 at empty_state)"
            )
        if not all(st.spec_pending for st in pp):
            raise RuntimeError(
                "kda_spec_undo: no verify step to take back (none issued, already undone, or a prefill's commit since)"
            )
        for st in pp:
            spec_flip(st)
            st.spec_pending = False

    def _kda_committed(self, states: list) -> None:
        """A traced prefill's commit replayed (eager commits mark it in `TtKda._commit`)."""
        for st in self._kda_pp(states):
            st.spec_pending = False

    def _by_parity(self, captured: "PrefillTrace", states: list) -> "PrefillTrace":
        """A traced commit's capture for the fields' set (the ping-pong's: one capture a parity)."""
        if captured.other is None:
            return captured
        return captured if self.kda_spec_parity(states) == captured.parity else captured.other

    # -- inputs at fixed addresses: written from the host each step, read by the device-only forward (trace) ------
    def alloc_io(self, batch: int) -> "DecodeIO":
        ids = put(
            self.mesh,
            torch.zeros(ROWS, batch, dtype=torch.int32),
            dims=(0, None),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        positions = self.dsa.alloc_positions(batch) if self.dsa is not None else None
        live = moe_route.LiveTokens.alloc(self.mesh, batch)
        io = DecodeIO(batch=batch, ids=ids, positions=positions, live=live)
        buffers = [ids, live.tensor] + (self.dsa.position_buffers(positions) if positions else [])
        io.pack = InputPack(self.mesh, buffers)
        io.pack.set(live.tensor, live.last)  # (all live)
        return io

    def write_inputs(self, io: "DecodeIO", tokens: torch.Tensor, positions: torch.Tensor, live=None) -> None:
        """tokens, positions [ROWS, B]; live [ROWS, B] bool (the seats with a request; None: as last written, all live
        as made) -> the io buffers (host to device, outside any trace; with a pack its one write: the step's `_unpack`
        copies them on the device)."""
        pack = io.pack
        if pack is not None:
            pack.set(io.ids, tokens)
        else:
            write_rows(self.mesh, tokens, io.ids, ttnn.uint32)
        if self.dsa is not None:
            self.dsa.write_positions(io.positions, positions, pack=pack)
        if live is not None and io.live is not None:
            io.live.write(self.mesh, live, pack)
        if pack is not None:
            pack.write()

    @staticmethod
    def _unpack(io) -> None:
        """A packed io's inputs to their tensors (the step's first device op: inside a capture too)."""
        if io.pack is not None:
            io.pack.unpack()

    @decode_step
    def forward(
        self,
        io: "DecodeIO",
        states: list,
        profile: dict | None = None,
        greedy: bool = False,
        after_stage=None,
    ):
        """The device-only decode step (capturable as a trace): io -> logits [1, 1, B, V] fp32, the same on a row's chips
        (or, with greedy, the argmax ids [1, 1, B, 1] fp32 chosen on the device).

        With `profile` (a dict), the device is synchronized after every stage and the seconds are appended per stage
        name (embed, kda+dense / kda+moe / dsa+moe ..., head): slower, for the breakdown only (never in a trace).
        `after_stage` (a callable taking the stage name) is called after every stage and every part of a layer ("<i>
        <part>", see TtDecoderLayer.step), e.g. to read the device profiler before its buffers overflow (never in a
        trace).
        """

        def mark(name, start):
            if after_stage is not None:
                after_stage(name)
            if profile is None:
                return start
            ttnn.synchronize_device(self.mesh)
            now = time.perf_counter()
            profile.setdefault(name, []).append(now - start)
            return now

        t = mark("start", time.perf_counter()) if profile is not None else None
        self._unpack(io)
        streams = expand_streams(self.mesh, self.embed(io.ids), self.config.hc_mult)
        t = mark("embed", t)
        chain = mhc_mod.WRITE_READ  # the layers' mHC writes chained into the next reads (mhc.WRITE_READ)
        pending = None
        for i, layer in enumerate(self.layers):
            part = None if after_stage is None else (lambda name, i=i: after_stage(f"{i} {name}"))
            defer = chain and i + 1 < len(self.layers)
            new, states[i] = layer.step(
                streams, states[i], io.positions, io.batch, mark=part, pending=pending, defer=defer
            )
            if pending is None:
                ttnn.deallocate(streams)
            pending, streams = (new, None) if defer else (None, new)
            t = mark(f"{layer.kind}+{'moe' if self.config.is_moe(i) else 'dense'}", t)
        hidden = self.norm(collapse_streams(self.mesh, streams, self.config.hc_mult, io.batch))
        ttnn.deallocate(streams)
        if greedy:
            out = self.lm_head.greedy_mesh(hidden)
        else:
            out = self.lm_head(hidden)
        ttnn.deallocate(hidden)
        mark("head", t)
        return out

    def logits_to_host(self, logits, batch: int) -> torch.Tensor:
        return gather_rows(self.mesh, logits).reshape(ROWS, batch, -1)[..., : self.config.vocab_size]

    def prefill_logits_to_host(self, logits, rows=None) -> torch.Tensor:
        """A prefill's logits (`_prefill`: [1, 1, 1, V] row-major per chip, untilized without the tile's 31 padding rows
        on the device, the same on a row's chips) -> [ROWS, V] fp32: each mesh row's first chip's (V fp32 a row: 5 MB in
        all, in place of every chip's 32-row tile padding, 634 MB, ~0.15 s a wave).  `rows` (the mesh rows whose
        logits the caller reads, indices or one bool a row): only those read, the others zero (620 KB a row: a single
        request's wave reads 1 row, not 8)."""
        chips = ttnn.get_device_tensors(logits)
        if rows is None:
            want = set(range(ROWS))
        elif len(rows) == ROWS and all(isinstance(r, bool) for r in rows):
            want = {r for r in range(ROWS) if rows[r]}
        else:
            want = {int(r) for r in rows}
        vocab = self.config.vocab_size
        out = torch.zeros(ROWS, vocab)
        for r in sorted(want):
            out[r] = ttnn.to_torch(chips[r * COLS]).reshape(-1)[:vocab].float()
        return out

    def ids_to_host(self, ids, batch: int) -> torch.Tensor:
        """`greedy_mesh`'s ids (every row's on every chip: the first chip's) -> [ROWS, batch] int64."""
        return ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(ROWS, batch).round().to(torch.int64)

    def step(
        self, tokens: torch.Tensor, positions: torch.Tensor, states: list, profile: dict | None = None
    ) -> torch.Tensor:
        """Eager decode step: tokens, positions [ROWS, B] (each row's users) -> logits [ROWS, B, V] fp32 on the host."""
        batch = tokens.shape[1]
        if batch not in self._io:
            self._io[batch] = self.alloc_io(batch)
        io = self._io[batch]
        self.write_inputs(io, tokens, positions)
        logits = self.forward(io, states, profile)
        t = time.perf_counter()
        out = self.logits_to_host(logits, batch)
        ttnn.deallocate(logits)
        if profile is not None:
            profile.setdefault("readback", []).append(time.perf_counter() - t)
        return out

    # -- MTP self-speculative decode ---------------------------------------------------------------------------------
    def _keep_prefill_hidden(self, hidden, owned: bool) -> None:
        """The last prefill chunk's final normed hidden state at each row's last real token, [1, 1, 1, 4096] per row:
        a new tensor (`owned`, freed when replaced) or a captured prefill's output buffer."""
        old = getattr(self, "_prefill_hidden", None)
        if old is not None and old[1] and old[0] is not hidden:
            ttnn.deallocate(old[0])
        self._prefill_hidden = (hidden, owned)

    def mtp_first_drafts(
        self, states: list, first: torch.Tensor, slots: list, users: int, drafts: int = 1
    ) -> torch.Tensor:
        """After a prefill chunk in which requests' prompts end (eager): the MTP layer at each such request's last
        prompt position P - 1, on its final normed hidden state there (the chunk's, `_keep_prefill_hidden`) and its
        first generated token g_P -> the first drafts (the predictions at P + 1), [ROWS] (-1 on the other rows); with
        two `drafts` also the second, the layer chained at P on its own output and the first draft -> [ROWS, 2].
        first [ROWS]: g_P on the rows whose request ended (< 0 elsewhere); slots[row]: (its user, P - 1) or None.  The
        other users write the layer's cache at its last position, which each writes itself before attending there."""
        hidden = self._prefill_hidden[0]
        ids = torch.zeros(ROWS, users, dtype=torch.int64)
        positions = torch.full((ROWS, users), self.dsa.max_seq - 1, dtype=torch.int64)
        pick = torch.zeros(ROWS, 1, users, 1)
        for row, slot in enumerate(slots):
            if slot is None or int(first[row]) < 0:
                continue
            user, pos = slot
            ids[row, user], positions[row, user], pick[row, 0, user, 0] = int(first[row]), pos, 1.0
        if users not in self._io:
            self._io[users] = self.alloc_io(users)
        io = self._io[users]
        self.write_inputs(io, ids, positions)
        fio = self._first_io(users, hidden)
        rows = mapper(self.mesh, (0, None))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(pick, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rows), fio.pick
        )
        if hidden is not fio.hidden:
            ttnn.copy(hidden, fio.hidden)
        if drafts == 1:  # one level: replayed when captured (`capture_first_drafts`), else eager
            if fio.trace is not None:
                ttnn.execute_trace(self.mesh, fio.trace, cq_id=0, blocking=False)
                ids = fio.out
            else:
                ids = self._first_step(io, fio, states)
            out = self.ids_to_host(ids, users)
            if fio.trace is None:
                ttnn.deallocate(ids)
            return torch.tensor([int(out[r, s[0]]) if s is not None else -1 for r, s in enumerate(slots)])
        n = len(self.layers)
        y = self._first_step(io, fio, states, head=False)
        outs = []
        for level in range(drafts):
            # chained: at P on the layer's own output and the last level's drafts (the others again at the end)
            if level:
                # (a prompt that fills the cache gets no usable second draft: its position stays in the cache)
                ahead = torch.clamp(
                    torch.where(pick[:, 0, :, 0] > 0, positions + 1, positions), max=self.dsa.max_seq - 1
                )
                self.write_inputs(io, outs[-1], ahead)
                self._unpack(io)
                embeds = self.embed(io.ids)
                y2, states[n] = self.mtp.step(embeds, y, states[n], io.positions)
                for t in (embeds, y):
                    ttnn.deallocate(t)
                y = y2
            head = self.mtp.head_input(y)
            ids = self.lm_head.greedy_mesh(head)
            ttnn.deallocate(head)
            outs.append(self.ids_to_host(ids, users))
            ttnn.deallocate(ids)
        ttnn.deallocate(y)
        got = torch.stack(
            [torch.tensor([int(o[r, s[0]]) if s is not None else -1 for r, s in enumerate(slots)]) for o in outs], dim=1
        )
        return got[:, 0] if drafts == 1 else got

    def _first_io(self, users: int, hidden) -> "FirstIO":
        """`mtp_first_drafts`'s fixed inputs for `users` users a row: each row's kept hidden state (as the prefill keeps
        it) and its request's user as a 0/1 column [1, 1, U, 1] bf16; made on the first (eager) call, before any capture.
        """
        if users not in self._first:
            pick = put(self.mesh, torch.zeros(ROWS, 1, users, 1), dims=(0, None), dtype=ttnn.bfloat16)
            kept = hidden if hidden is self._kept_hidden else ttnn.clone(hidden)  # (the fixed buffer: read in place)
            self._first[users] = FirstIO(hidden=kept, pick=pick)
        return self._first[users]

    def _first_step(self, io: "DecodeIO", fio: "FirstIO", states: list, head: bool = True):
        """The first drafts' device work (capturable): each row's hidden state at its request's user (x 1 / 0: exact),
        the MTP layer on (it, the first tokens' embeddings) at the written positions -> the greedy ids [1, 1, 8 U, 1]
        (`head`), else the layer's output."""
        n, users = len(self.layers), fio.pick.shape[-2]
        # [1, 1, U, 4096]: the row's at every user (at 1 user a row repeat would return the buffer itself: kept)
        spread = ttnn.repeat(fio.hidden, ttnn.Shape([1, 1, users, 1])) if users > 1 else fio.hidden
        h = ttnn.multiply(spread, fio.pick)  # the row's at its request's user, zeros elsewhere
        if users > 1:
            ttnn.deallocate(spread)
        self._unpack(io)
        embeds = self.embed(io.ids)
        y, states[n] = self.mtp.step(embeds, h, states[n], io.positions)
        for t in (embeds, h):
            ttnn.deallocate(t)
        if not head:
            return y
        x = self.mtp.head_input(y)
        ttnn.deallocate(y)
        ids = self.lm_head.greedy_mesh(x)
        ttnn.deallocate(x)
        return ids

    def capture_first_drafts(self, users: int, states: list) -> None:
        """Capture `mtp_first_drafts`' one-level step for `users` users a row (after an eager call made its inputs and
        constants): the layer and the head replayed per arriving request instead of ~100 eager ops."""
        fio = self._first.get(users)
        if fio is None or fio.trace is not None or users not in self._io:
            return
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        out = self._first_step(self._io[users], fio, states)
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        fio.trace, fio.out = trace, out

    def release_first_drafts(self) -> None:
        for fio in self._first.values():
            if fio.trace is not None:
                ttnn.release_trace(self.mesh, fio.trace)
                ttnn.deallocate(fio.out)
                fio.trace = fio.out = None

    def alloc_spec_io(self, users: int, drafts: int = 1, mtp: bool = True) -> "SpecIO":
        """Fixed-address inputs of the verify step for `users` users a row: (drafts + 1) users tokens (each user's
        token, then its drafts), DSA's virtual users (with two drafts again one position on, for the MTP layer's second
        level), the KDA keep masks (all kept) and the MTP inputs' host overrides (none).  `mtp` False: the drafts come
        from elsewhere (`forward_verify`, any number of drafts): no MTP layer's positions."""
        b = (drafts + 1) * users
        f32 = ttnn.float32
        ids = put(
            self.mesh,
            torch.zeros(ROWS, b, dtype=torch.int32),
            dims=(0, None),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        flags = SpecKeep.host(torch.ones(ROWS, users))["flags"]
        flags = put(self.mesh, flags, dims=(0, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        keep = SpecKeep(flags=flags)
        others = torch.full((ROWS, 1, ROWS, 1), 1.0e9)
        others[torch.arange(ROWS), 0, torch.arange(ROWS), 0] = 0.0  # row r: 0 at its own ids
        zeros = torch.zeros(ROWS, 1, 1, b)
        io = SpecIO(
            users=users,
            drafts=drafts,
            ids=ids,
            positions=self.dsa.alloc_positions(b, users=users) if self.dsa is not None else None,
            positions2=self.dsa.alloc_positions(b, users=users)
            if drafts > 1 and mtp and self.dsa is not None
            else None,
            keep=keep,
            others=put(self.mesh, others, dims=(0, None), dtype=f32),
            forced=put(self.mesh, zeros, dims=(0, None), dtype=f32),
            free=put(self.mesh, zeros + 1, dims=(0, None), dtype=f32),
        )
        if self.lm_head.relax is not None:  # (the looser acceptance: the last level's tokens have no next one)
            io.no_next = put(self.mesh, torch.full((1, 1, 1, users), -1.0), dims=(None, None), dtype=f32)
        io.live = moe_route.LiveTokens.alloc(self.mesh, b)
        buffers = [ids, flags, io.live.tensor] + (self.dsa.position_buffers(io.positions) if io.positions else [])
        if io.positions2 is not None:
            buffers += self.dsa.position_buffers(io.positions2)
        io.pack = InputPack(self.mesh, buffers)
        io.pack.set(io.live.tensor, io.live.last)  # (all live)
        return io

    def write_spec_inputs(self, io: "SpecIO", tokens, positions, keep, forced=None, live=None) -> None:
        """tokens, positions [ROWS, k U] (each user's token at p, then its drafts at p + 1 ..); keep [ROWS, U] (the
        user's drafts accepted in the last step); forced [ROWS, k U] (the MTP layer's next token at a slot, in place of
        the step's own greedy id; < 0: none); live [ROWS, k U] bool (the tokens of seats with a request; None: as last
        written, all live as made) -> the io buffers (host to device, outside any trace)."""
        rows = mapper(self.mesh, (0, None))
        up = lambda t, dtype, layout: ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=rows)  # noqa: E731
        pack = io.pack
        if pack is not None:
            pack.set(io.ids, tokens)
        else:
            write_rows(self.mesh, tokens, io.ids, ttnn.uint32)
        if self.dsa is not None:  # (a model of only KDA layers, tests: none)
            self.dsa.write_positions(io.positions, positions, pack=pack)
        if io.positions2 is not None:  # the MTP layer's second level, one position on
            self.dsa.write_positions(io.positions2, positions + 1, pack=pack)
        flags = SpecKeep.host(keep)["flags"]
        if pack is not None:
            pack.set(io.keep.flags, flags)
        else:
            write_rows(self.mesh, flags, io.keep.flags, ttnn.uint32)
        if live is not None and io.live is not None:
            io.live.write(self.mesh, live, pack)
        if pack is not None:  # (forced / free, fp32 tiles, are written apart when given)
            pack.write()
        if forced is None or not bool((forced >= 0).any()):
            if io.forced_none:  # the buffers already say none (serving: every step; two host writes of ~0.4 ms each)
                return
            forced = torch.full(tokens.shape, -1)
        on = forced >= 0
        b = tokens.shape[1]
        ttnn.copy_host_to_device_tensor(
            up(torch.where(on, forced, 0).float().reshape(ROWS, 1, 1, b), ttnn.float32, ttnn.TILE_LAYOUT), io.forced
        )
        ttnn.copy_host_to_device_tensor(
            up((~on).float().reshape(ROWS, 1, 1, b), ttnn.float32, ttnn.TILE_LAYOUT), io.free
        )
        io.forced_none = not bool(on.any())

    def _relax(self, io: "SpecIO"):
        """The looser acceptance's input to the head (`TtLmHead.greedy_mesh`'s `relax`), or None
        when it is off: each of the step's tokens' next one, every row's on every chip [1, 1, 8 b, 1] fp32 (virtual user
        j U + u's: user u's token j + 1, its next draft; the last level's -1), made on the device from the step's own
        tokens, and the gap scalar.  Fused (head.RELAX_FUSED, up to head_relax.MAX_ROWS rows): the step's tokens,
        every row's [1, 1, 8, b] uint32 row-major, the gap scalar and U (the head picks the next ones itself)."""
        if self.lm_head.relax is None:
            return None
        b, u = (io.drafts + 1) * io.users, io.users
        from models.demos.blackhole.glm53_flash.ttnn import head as head_mod
        from models.demos.blackhole.glm53_flash.ttnn.head_relax import MAX_ROWS as RELAX_ROWS

        if head_mod.RELAX_FUSED and ROWS * b <= RELAX_ROWS:  # the fused form: every row's tokens as they are
            every = ttnn.all_gather(ttnn.reshape(io.ids, [1, 1, 1, b]), dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8, b]
            return every, self.lm_head.relax, u
        flat = ttnn.to_layout(ttnn.reshape(io.ids, [1, 1, 1, b]), ttnn.TILE_LAYOUT)
        tok = ttnn.typecast(flat, ttnn.float32)
        ttnn.deallocate(flat)
        later = ttnn.slice(tok, [0, 0, 0, u], [1, 1, 1, b])
        ttnn.deallocate(tok)
        nxt = ttnn.concat([later, io.no_next], dim=3)  # [1, 1, 1, b]
        ttnn.deallocate(later)
        rm = ttnn.to_layout(nxt, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(nxt)
        every = ttnn.all_gather(rm, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8, b]
        ttnn.deallocate(rm)
        col = ttnn.to_layout(ttnn.reshape(every, [1, 1, ROWS * b, 1]), ttnn.TILE_LAYOUT)
        ttnn.deallocate(every)
        return col, self.lm_head.relax

    def _row_ids(self, ids, io: "SpecIO", forced: bool = True):
        """greedy_mesh's ids of all the rows [1, 1, 8 b, 1] fp32 -> this row's [1, b] uint32 row-major, a host
        override at the forced slots (x 0 / 1, + 0: exact; not with `forced` False): the smallest of (row r's id + 1e9
        off the own row)."""
        b = (io.drafts + 1) * io.users
        grid = ttnn.reshape(ids, [1, 1, ROWS, b])
        own = ttnn.min(ttnn.add(grid, io.others), dim=2, keepdim=True)  # [1, 1, 1, b]
        ttnn.deallocate(grid)
        if forced:
            mixed = ttnn.addcmul(io.forced, own, io.free)
            ttnn.deallocate(own)
        else:
            mixed = own
        flat = ttnn.to_layout(ttnn.typecast(mixed, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(mixed)
        return ttnn.reshape(flat, [1, b])

    @decode_step
    def forward_spec(self, io: "SpecIO", states: list, logits: bool = False):
        """MTP's verify step, device only (capturable): each user's token t (at p) and draft d (at p + 1) through the
        model, then the MTP layer on (its hidden there, its greedy id there) -> (ids, drafts), both [1, 1, 8 2U, 1]
        fp32 on every chip (row r's 2U at 2U r, t's first): the greedy ids g_t, g_d and the MTP layer's next drafts
        A (from t: used if d is rejected) and B (from d: used if accepted).

        With two drafts (`io.drafts` 2): t, d1, d2 at p .. p + 2 (3U), ids g_t, g_d1, g_d2 and the MTP layer's first
        level A1, B1, C1 there (`drafts`), then its second level, chained on its own outputs and A1, B1, C1 one position
        on -> A2, B2, C2 (`drafts2`): (ids, drafts, drafts2).  The second level writes the layer's cache at p + 1 ..
        p + 3; the first level's rows are written back after it.

        `logits` (one draft, eager: a test's check): also the logits there, (ids, drafts, the model's [1, 1, 2U, V]
        sharded as `forward`'s, the MTP layer's)"""
        if logits and io.drafts > 1:
            raise ValueError("forward_spec: logits with one draft only")
        if self.kda_spec_parity(states) is not None:  # (its callers don't flip the sets: DFlash2's verify only)
            raise NotImplementedError("forward_spec (MTP) with the KDA ping-pong (GLM53_KDA_SPEC_PINGPONG)")
        b = (io.drafts + 1) * io.users
        mtp_state = states[len(self.layers)]
        self._unpack(io)
        streams = expand_streams(self.mesh, self.embed(io.ids), self.config.hc_mult)
        chain = mhc_mod.WRITE_READ  # (as `forward`)
        pending = None
        for i, layer in enumerate(self.layers):
            defer = chain and i + 1 < len(self.layers)
            tap = self.taps if self.taps is not None and i in self.tap_layers else None
            new, states[i] = layer.step(
                streams, states[i], io.positions, b, spec=io.keep, pending=pending, defer=defer, tap=tap
            )
            if pending is None:
                ttnn.deallocate(streams)
            pending, streams = (new, None) if defer else (None, new)
        hidden = self.norm(collapse_streams(self.mesh, streams, self.config.hc_mult, b))
        ttnn.deallocate(streams)
        relax = self._relax(io)  # (the looser acceptance: a near draft as the id, the MTP layer's input there too)
        ids = self.lm_head.greedy_mesh(hidden, relax=relax)
        if relax is not None:
            ttnn.deallocate(relax[0])
        main_logits = self.lm_head(hidden) if logits else None
        own = self._row_ids(ids, io)
        embeds = self.embed(own)
        ttnn.deallocate(own)
        two = io.drafts > 1
        y, _, *kept = self.mtp.step(embeds, hidden, mtp_state, io.positions, keep=two)
        ttnn.deallocate(embeds)
        ttnn.deallocate(hidden)
        head = self.mtp.head_input(y)
        drafts = self.lm_head.greedy_mesh(head)
        draft_logits = self.lm_head(head) if logits else None
        ttnn.deallocate(head)
        if not two:
            ttnn.deallocate(y)
            return (ids, drafts, main_logits, draft_logits) if logits else (ids, drafts)
        own = self._row_ids(drafts, io, forced=False)
        embeds = self.embed(own)
        ttnn.deallocate(own)
        y2, _ = self.mtp.step(embeds, y, mtp_state, io.positions2)
        for t in (embeds, y):
            ttnn.deallocate(t)
        self.mtp.rewrite(kept[0], mtp_state, io.positions)
        head = self.mtp.head_input(y2)
        ttnn.deallocate(y2)
        drafts2 = self.lm_head.greedy_mesh(head)
        ttnn.deallocate(head)
        return ids, drafts, drafts2

    @decode_step
    def forward_verify(
        self, io: "SpecIO", states: list, logits: bool = False, raw: bool = False, sampling=None, own: bool = False
    ):
        """The verify step of drafts from elsewhere (a draft model's, DFlash2), device only
        (capturable): each user's token t (at p) and its K drafts (at p + 1 .. p + K, rows j U ..) through the model ->
        the greedy ids there, [1, 1, 8 (K + 1) U, 1] fp32 on every chip (row r's (K + 1) U at (K + 1) U r, the tokens'
        first).  The KDA layers come back to the state after each user's accepted tokens (`io.keep`: the drafts the
        last step accepted); the attention caches are written at every position, the later steps overwrite the
        rejected ones before they read them.  `logits` (a looser acceptance's tests): also the logits there,
        (ids, the model's [1, 1, (K + 1) U, V] sharded as `forward`'s).  `raw` (not with `logits`): (ids, the plain
        argmax ids there with the looser acceptance on, else None; `accept_device` counts the drafts taken that were
        not the argmax).  `sampling` ((`sampling.StepSampling`, the step's seeds)): draws in place of the greedy ids at
        the sampled users (no looser acceptance with it).  `own` (greedy, no looser acceptance; else ignored): each
        chip's own mesh row's ids only, the other rows undefined (`TtLmHead.greedy_mesh`)."""
        b = (io.drafts + 1) * io.users
        self._unpack(io)
        streams = expand_streams(self.mesh, self.embed(io.ids), self.config.hc_mult)
        chain = mhc_mod.WRITE_READ  # (as `forward`)
        pending = None
        with w1_dense.step(self, b) as w1d:  # (GLM53_W1D: the next x_proj's riders)
            for i, layer in enumerate(self.layers):
                if w1d is not None:
                    w1d.at_layer(i)
                defer = chain and i + 1 < len(self.layers)
                tap = self.taps if self.taps is not None and i in self.tap_layers else None  # (DFlash2's taps)
                new, states[i] = layer.step(
                    streams, states[i], io.positions, b, spec=io.keep, pending=pending, defer=defer, tap=tap
                )
                if pending is None:
                    ttnn.deallocate(streams)
                pending, streams = (new, None) if defer else (None, new)
        hidden = self.norm(collapse_streams(self.mesh, streams, self.config.hc_mult, b))
        ttnn.deallocate(streams)
        relax = None if sampling is not None else self._relax(io)  # (the looser acceptance: a near draft as the id)
        plain = None
        if sampling is not None:
            ids = self.lm_head.sample_mesh(hidden, *sampling)
        elif relax is not None and raw:
            ids, plain = self.lm_head.greedy_mesh(hidden, relax=relax, keep_greedy=True)
        else:
            ids = self.lm_head.greedy_mesh(hidden, relax=relax, own=own and relax is None and not logits)
        if relax is not None:
            ttnn.deallocate(relax[0])
        out = (ids, self.lm_head(hidden)) if logits else (ids, plain) if raw else ids
        ttnn.deallocate(hidden)
        return out

    def capture_spec(self, io: "SpecIO", states: list):
        """As `capture`: one eager run (it advances the states), then the capture -> (trace, the outputs): the ids,
        drafts[, drafts2] of `forward_spec` as one row-major [1, 1, k 8 b, 1] fp32 tensor, read back once a step
        (`spec_outputs_to_host`; each read ~0.5 ms of host)."""

        def run():
            outs = self.forward_spec(io, states)
            flat = [ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) for t in outs]
            for t in outs:
                ttnn.deallocate(t)
            both = ttnn.concat(flat, dim=2)
            for t in flat:
                ttnn.deallocate(t)
            return (both,)

        for t in run():  # (eager first: every program of the capture, the joining ops too, in the cache)
            ttnn.deallocate(t)
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        outs = run()
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        return (trace, *outs)

    def spec_outputs_to_host(self, both, count: int, batch: int) -> list:
        """`capture_spec`'s tensor -> its `count` outputs, each [ROWS, batch] int64 (as `ids_to_host`: every row's ids
        on every chip, the first chip's read)."""
        flat = ttnn.to_torch(ttnn.get_device_tensors(both)[0]).reshape(count, ROWS, batch).round().to(torch.int64)
        return [flat[i] for i in range(count)]

    # -- prefill ---------------------------------------------------------------------------------------------------
    def prefill(
        self,
        tokens: torch.Tensor,
        start: int,
        states: list,
        slots: list,
        rows: list[bool] | None = None,
        valid: int | list[int] | None = None,
        logits: bool = True,
        last: bool = True,
        after_stage=None,
        next_tokens: torch.Tensor | None = None,
        greedy_ids: bool = False,
    ) -> torch.Tensor | None:
        """One prefill chunk of a wave (one request per mesh row), eager: tokens [ROWS, T] (positions start .. start +
        T - 1), row r's request going to its user slots[r] (None: no request on that row).  The rows in `rows` (default
        those with a request) compute their chunk; the others run on their (ignored) tokens and change nothing (a
        prompt that has ended keeps its KDA wave state; DSA writes a scratch page).  The first `valid` tokens are real
        (one count for every row or one per row, default all).  `last`: the wave's last chunk, after which every
        request's KDA state is in its user.  T and start are multiples of 32.  -> the logits after each row's last real
        token, [ROWS, V] fp32 on the host (meaningful on the `rows` rows), or None without `logits` (nothing read back:
        a chunk no prompt ends in).  `after_stage` is called as in `forward` (embed, "<i> <part>" and each layer's
        kind, head), e.g. to read the device profiler.  `next_tokens` [ROWS, T]: the MTP layer's inputs
        (`prefill_inputs`).  `greedy_ids`: each row's greedy token instead of its logits, [ROWS] int64 (PREFILL_IDS:
        argmax on the device, 8 ids read back)."""
        place = eager_place(self.mesh)
        inputs = self.prefill_inputs(
            place, tokens, start, states, slots, rows, valid, last, logits, next_tokens=next_tokens
        )
        out = self._prefill(inputs, states, start, after_stage)
        inputs.deallocate()
        if self.logits_all is not None:  # (a test's: every position's logits kept, zeros returned)
            computing = [s is not None for s in slots] if rows is None else rows
            counts = valid if isinstance(valid, list) else [valid if valid is not None else tokens.shape[1]] * ROWS
            segs = [(r, 0, start, counts[r], r) for r in range(ROWS) if computing[r]]
            self.logits_all.append((out, tokens.shape[1], segs))
            return torch.zeros(ROWS, self.config.vocab_size) if logits else None
        if out is None:
            return None
        computing = [s is not None for s in slots] if rows is None else rows  # (the others' logits unread)
        if greedy_ids:
            ids = self._device_ids(out)
            if ids is not None:
                result = self.prefill_ids_to_host(ids, out, computing)
                ttnn.deallocate(ids)
            else:
                result = self.prefill_logits_to_host(out, computing).argmax(-1)
        else:
            result = self.prefill_logits_to_host(out, computing)
        ttnn.deallocate(out)
        return result

    def _device_ids(self, out):
        """PREFILL_IDS: a prefill's logits ([1, 1, 1, V] fp32 row-major per chip) -> their argmax [1, 1, 1, 1] uint32
        per chip, on the device (the first index of the maximum, as torch's); else None."""
        return ttnn.argmax(out, dim=3, keepdim=True) if PREFILL_IDS and out is not None else None

    def prefill_ids_to_host(self, ids, logits=None, rows=None) -> torch.Tensor:
        """`_device_ids`' ids (the same on a row's chips) -> [ROWS] int64: each mesh row's first chip's.  With
        PREFILL_IDS_CHECK and the `logits` they came from: checked against the host's argmax of all 8 rows' logits (the
        read before PREFILL_IDS) on the computing `rows` (`ids_check`)."""
        chips = ttnn.get_device_tensors(ids)
        out = torch.tensor([int(ttnn.to_torch(chips[r * COLS]).reshape(-1)[0]) for r in range(ROWS)], dtype=torch.int64)
        if PREFILL_IDS_CHECK and logits is not None:
            want = self.prefill_logits_to_host(logits).argmax(-1)
            check = [r for r in range(ROWS) if rows is None or rows[r]]
            self.ids_check[0] += len(check)
            self.ids_check[1] += sum(int(out[r]) != int(want[r]) for r in check)
        return out

    def prefill_score(
        self,
        tokens: torch.Tensor,
        start: int,
        states: list,
        slots: list,
        next_tokens: torch.Tensor,
        rows: list[bool] | None = None,
        valid: int | list[int] | None = None,
        last: bool = True,
    ) -> torch.Tensor:
        """Teacher forcing: one prefill chunk as `prefill` (eager), scoring every position's next
        token instead of the logits: next_tokens [ROWS, T] (-1: none) -> [ROWS, T, 3] float64 on the host: the next
        token's log-probability under the model, its logit's gap below the top (0: the greedy token), the top's
        log-probability (`TtLmHead.score_mesh`, the logits over the whole vocabulary on the 32 chips)."""
        self.lm_head.prepare_scoring()
        place = eager_place(self.mesh)
        mtp_next = next_tokens.clamp(min=0) if self.mtp is not None else None
        inputs = self.prefill_inputs(
            place, tokens, start, states, slots, rows, valid, last, logits=False, next_tokens=mtp_next
        )
        t = tokens.shape[1]
        inputs.score_next = put(
            self.mesh, next_tokens.float().reshape(1, 1, ROWS * t, 1), dims=(None, None), dtype=ttnn.float32
        )
        out = self._prefill(inputs, states, start)
        inputs.deallocate()
        result = self.lm_head.score_join(out, ROWS * t).view(ROWS, t, 3)
        ttnn.deallocate(out)
        return result

    def prefill_sp(self, tokens: torch.Tensor, states: list, groups, logits: bool = True, after_stage=None) -> dict:
        """One wave of requests' prompts over the mesh rows (sequence parallel), eager: tokens
        [ROWS, T], each request's segments on its rows (`common.SpGroup`: rows[i] holding positions start + i T ..,
        valid[i] real tokens).  A group's `last` wave puts its KDA state into its user.  -> {request's row: the
        logits after its last token [V] fp32} for the groups whose last wave it is ({} without `logits`).  Needs
        `dsa.PREFILL_SP` (on: the DSA cache's extra user, `sp_ready`)."""
        inputs = self.sp_inputs(eager_place(self.mesh), tokens, states, groups)
        out = self._prefill(inputs, states, 0, after_stage)
        inputs.deallocate()
        if self.logits_all is not None:  # (a test's: every position's logits kept, zeros returned)
            t = tokens.shape[1]
            segs = [
                (r, 0, g.start + i * t, v, g.owner) for g in groups for i, (r, v) in enumerate(zip(g.rows, g.valid))
            ]
            self.logits_all.append((out, t, segs))
            return {g.owner: torch.zeros(self.config.vocab_size) for g in groups if g.last} if logits else {}
        if not logits or not any(g.last for g in groups):
            ttnn.deallocate(out)
            return {}
        result = self.prefill_logits_to_host(out, [g.owner for g in groups if g.last])
        ttnn.deallocate(out)
        return {g.owner: result[g.owner] for g in groups if g.last}

    def sp_inputs(
        self, place, tokens: torch.Tensor, states: list, groups, traced: bool = False, own: bool = True
    ) -> "PrefillInputs":
        """A sequence-parallel wave's inputs (arguments as `prefill_sp`; `traced`: without the KDA commit, which
        `replay_commit` does apart; not `own`: without the ids, keep and logits' one-hot, a pair's wave's)."""
        t = tokens.shape[1]
        valid = [0] * ROWS
        for grp in groups:
            for r, v in zip(grp.rows, grp.valid):
                valid[r] = v
        keep = place("keep", torch.ones(ROWS, 1, 1, 1), (0, None)) if own else None
        ids = place("ids", tokens.to(torch.int32), (0, None), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if own else None
        picks = tables = None
        if self.has_kda:
            kernel = self.config.kda_conv_kernel
            picks = TtKda.sp_picks(place, t, groups, kernel, self._kda_users(states), traced)
        if self.dsa is not None:
            users = next(self.dsa.users(st) for st, layer in zip(states, self.layers) if layer.kind == DSA)
            tables = self.dsa.sp_tables(place, users, groups, t)
        one_hot = torch.nn.functional.one_hot((torch.tensor(valid) - 1).clamp(min=0), t).float().view(ROWS, 1, 1, t)
        last_token = place("last_token", one_hot, (0, None)) if own else None
        nxt = None
        if self.mtp is not None:  # a segment's last next token: the next segment's first (the wave's last: 0)
            nxt = torch.cat([tokens[:, 1:], torch.zeros_like(tokens[:, :1])], dim=1)
            for grp in groups:
                for a, b in zip(grp.rows, grp.rows[1:]):
                    nxt[a, t - 1] = tokens[b, 0]
        return PrefillInputs(
            t=t,
            ids=ids,
            keep=keep,
            kda=picks,
            dsa=tables,
            last_token=last_token,
            owned=not traced,
            next_ids=self._next_ids(place, tokens, nxt),
        )

    def replay_prefill_sp(
        self, captured: "PrefillTrace", tokens: torch.Tensor, states: list, groups, logits: bool = True
    ) -> dict:
        """A sequence-parallel wave through its captured prefill (`prefill_key(.., sp=True)`; arguments as
        `prefill_sp`; the KDA commit apart, `replay_commit`) -> as `prefill_sp`."""
        captured.inputs.begin()
        self.sp_inputs(captured.inputs.place, tokens, states, groups, traced=True)
        captured.check_packed(captured.inputs.end())
        ttnn.execute_trace(self.mesh, captured.trace, cq_id=0, blocking=False)
        if captured.hidden is not None and captured.hidden_shared:  # (its trace writes the kept hidden there)
            self._keep_prefill_hidden(captured.hidden, owned=False)
        if captured.taps is not None and self.taps is not None:  # (as `replay_prefill`: the trace's own buffers)
            self.taps.extend(captured.taps)
        if not logits or not any(g.last for g in groups):
            return {}
        result = self.prefill_logits_to_host(captured.out, [g.owner for g in groups if g.last])
        return {g.owner: result[g.owner] for g in groups if g.last}

    def prefill_sp_pair(self, tokens: torch.Tensor, states: list, groups_a, groups_b, logits: bool = True) -> dict:
        """Two consecutive sequence-parallel waves (as `prefill_sp`'s, each of T tokens a row) as one wave of 2 T
        tokens a row, eager: tokens [ROWS, 2 T], each row the first wave's segment and
        then the second's; the attention layers (KDA, DSA, the MTP layer) take the two waves one after the other as
        `prefill_sp` would, the other layers all 2 T tokens at once (per-token: the same values).  -> {request's row:
        the logits after its last token} for the groups whose last wave is either."""
        inputs = self.sp_pair_inputs(eager_place(self.mesh), tokens, states, groups_a, groups_b)
        out = self._prefill(inputs, states, 0)
        inputs.deallocate()
        ended = [g for g in (*groups_a, *groups_b) if g.last]
        if self.logits_all is not None:  # (a test's: every position's logits kept, zeros returned)
            t = tokens.shape[1] // 2
            segs = [
                (r, w * t, g.start + i * t, v, g.owner)
                for w, groups in enumerate((groups_a, groups_b))
                for g in groups
                for i, (r, v) in enumerate(zip(g.rows, g.valid))
            ]
            self.logits_all.append((out, 2 * t, segs))
            return {g.owner: torch.zeros(self.config.vocab_size) for g in ended} if logits else {}
        if not logits or not ended:
            ttnn.deallocate(out)
            return {}
        result = self.prefill_logits_to_host(out, [g.owner for g in ended])
        ttnn.deallocate(out)
        return {g.owner: result[g.owner] for g in ended}

    def sp_pair_inputs(self, place, tokens: torch.Tensor, states: list, groups_a, groups_b, traced=False):
        """A pair of sequence-parallel waves' inputs (`prefill_sp_pair`): each wave's `sp_inputs` in buffers of its
        own (`place` names prefixed "a:" / "b:"), with the pair's ids, keep, the one-hot of each row's last real token
        over the 2 T rows (a row's request ends in one wave at most: the other's rows are its block's) and the MoE's
        real-token mask of both waves."""
        t2 = tokens.shape[1]
        t = t2 // 2
        prefixed = lambda tag: lambda name, *a, **k: place(tag + name, *a, **k)  # noqa: E731
        a = self.sp_inputs(prefixed("a:"), tokens[:, :t], states, groups_a, traced, own=False)
        b = self.sp_inputs(prefixed("b:"), tokens[:, t:], states, groups_b, traced, own=False)
        valid = torch.zeros(ROWS, 2, dtype=torch.int64)
        last = torch.full((ROWS,), t2 - 1, dtype=torch.int64)
        for w, groups in enumerate((groups_a, groups_b)):
            for grp in groups:
                for r, v in zip(grp.rows, grp.valid):
                    valid[r, w] = v
                if grp.last:
                    last[grp.owner] = w * t + grp.valid[-1] - 1
        real = torch.cat([torch.arange(t)[None] < valid[:, :1], torch.arange(t)[None] < valid[:, 1:]], dim=1)
        one_hot = torch.nn.functional.one_hot(last, t2).float().view(ROWS, 1, 1, t2)
        moe = KdaPicks(
            history_pick=None,
            real=place("sp2_real", real.float().view(ROWS, t2, 1), (0, None)) if self.has_kda else None,
            owned=not traced,
        )
        return PrefillInputs(
            t=t2,
            ids=place("ids", tokens.to(torch.int32), (0, None), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            keep=place("keep", torch.ones(ROWS, 1, 1, 1), (0, None)),
            kda=moe if self.has_kda else None,
            dsa=None,
            last_token=place("last_token", one_hot, (0, None)),
            owned=not traced,
            halves=[a, b],
        )

    def replay_prefill_sp_pair(
        self, captured: "PrefillTrace", tokens: torch.Tensor, states: list, groups_a, groups_b, logits: bool = True
    ) -> dict:
        """A pair of sequence-parallel waves through its captured prefill (`pair_key`; the KDA commit apart,
        `replay_commit`) -> as `prefill_sp_pair`."""
        captured.inputs.begin()
        self.sp_pair_inputs(captured.inputs.place, tokens, states, groups_a, groups_b, traced=True)
        captured.check_packed(captured.inputs.end())
        ttnn.execute_trace(self.mesh, captured.trace, cq_id=0, blocking=False)
        if captured.hidden is not None and captured.hidden_shared:  # (its trace writes the kept hidden there)
            self._keep_prefill_hidden(captured.hidden, owned=False)
        if captured.taps is not None and self.taps is not None:  # (as `replay_prefill`: the trace's own buffers)
            self.taps.extend(captured.taps)
        ended = [g for g in (*groups_a, *groups_b) if g.last]
        if not logits or not ended:
            return {}
        result = self.prefill_logits_to_host(captured.out, [g.owner for g in ended])
        return {g.owner: result[g.owner] for g in ended}

    def pair_key(self, start_a: int, start_b: int, t: int) -> PrefillKey | None:
        """The captured form of a pair of sequence-parallel waves of T tokens a row from `start_a` and `start_b` (their
        groups' largest starts), or None (GLM53_PREFILL_TRACE_ALL off): one form for any starts, a long cache's by the
        waves' buckets (`TtDsa.sp_bucket`)."""
        if not PREFILL_TRACE_ALL:
            return None
        long = self.dsa is not None and self.dsa.long
        bucket = (lambda s: self.dsa.sp_bucket(s, t)) if long else (lambda s: 0)  # noqa: E731
        return PrefillKey(2 * t, False, False, bucket(start_a), True, bucket(start_b))

    def _pair_capture_groups(self, key: PrefillKey, start: int) -> tuple | None:
        """Capture groups for a pair form: one request on the first b rows (8, 4, 2 or 1: the largest the cache takes)
        from `start`, its next wave from start + b T, whose buckets are the key's; None when none fits."""
        t = key.t // 2
        cache = self.dsa.max_seq if self.dsa is not None else 1 << 30
        for b in (ROWS, 4, 2, 1):
            if start + 2 * b * t > cache:
                continue
            if self.pair_key(start, start + b * t, t) != key:
                continue
            rows = tuple(range(b))
            return [SpGroup(rows, (t,) * b, 0, start, False)], [SpGroup(rows, (t,) * b, 0, start + b * t, False)]
        return None

    @property
    def sp_ready(self) -> bool:
        """The sequence-parallel prefill is possible (`dsa.PREFILL_SP`, on; a test may turn it off)."""
        from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_module

        return dsa_module.PREFILL_SP

    def prefill_inputs(
        self,
        place,
        tokens: torch.Tensor,
        start: int,
        states: list,
        slots: list,
        rows: list[bool] | None = None,
        valid: int | list[int] | None = None,
        last: bool = True,
        logits: bool = True,
        traced: bool = False,
        next_tokens: torch.Tensor | None = None,
    ) -> "PrefillInputs":
        """A chunk's per-row inputs (arguments as `prefill`), put on the device by `place` (`common.eager_place`, or a
        trace's `FixedInputs.place`): they depend only on the rows' requests and real-token counts, and go up once
        for all the layers.  `traced`: the same tensors whatever the rows (`TtKda.prefill_picks`).  `next_tokens`
        [ROWS, T]: with the MTP layer, each position's next token (default: the chunk's own, shifted; the last 0)."""
        t = tokens.shape[1]
        rows = [s is not None for s in slots] if rows is None else rows
        valid = [t] * ROWS if valid is None else [valid] * ROWS if isinstance(valid, int) else list(valid)
        valid = [v if r else t for v, r in zip(valid, rows)]
        keep = place("keep", torch.tensor([1.0 if r else 0.0 for r in rows]).view(ROWS, 1, 1, 1), (0, None))
        ids = place("ids", tokens.to(torch.int32), (0, None), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        picks = tables = None
        if self.has_kda:
            users = next(st.recurrent.shape[0] for st, layer in zip(states, self.layers) if layer.kind == KDA)
            commit = slots if last else None
            picks = TtKda.prefill_picks(place, t, valid, self.config.kda_conv_kernel, users, commit, rows, traced)
        if self.dsa is not None:
            users = next(self.dsa.users(st) for st, layer in zip(states, self.layers) if layer.kind == DSA)
            written = [s if r else None for s, r in zip(slots, rows)]
            tables = self.dsa.prefill_tables(place, users, written, start, t, valid, traced=traced)
        last_token = None
        if (
            logits or traced
        ):  # each row's last real token: a 0/1 row per mesh row picks it (one product per output: exact)
            one_hot = torch.nn.functional.one_hot(torch.tensor(valid) - 1, t).float().view(ROWS, 1, 1, t)
            last_token = place("last_token", one_hot, (0, None))
        return PrefillInputs(
            t=t,
            ids=ids,
            keep=keep,
            kda=picks,
            dsa=tables,
            last_token=last_token,
            owned=not traced,
            next_ids=self._next_ids(place, tokens, next_tokens),
        )

    def _next_ids(self, place, tokens: torch.Tensor, next_tokens: torch.Tensor | None):
        """The MTP layer's prefill input: each position's next token [1, T] uint32 per row (None without the layer)."""
        if self.mtp is None:
            return None
        if next_tokens is None:
            next_tokens = torch.cat([tokens[:, 1:], torch.zeros_like(tokens[:, :1])], dim=1)
        return place("next_ids", next_tokens.to(torch.int32), (0, None), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def _prefill(self, inputs: "PrefillInputs", states: list, start: int, after_stage=None):
        """The device part of a prefill chunk (only device tensors in and out: a trace can capture it) -> the logits
        after each row's last real token, [1, 1, 1, V] fp32 per row, or None without `inputs.last_token`."""
        try:
            return self._prefill_layers(inputs, states, start, after_stage)
        finally:  # the tensors the layers made once from the chunk's inputs and shared (`KdaPicks.derived`)
            for one in [inputs, *(inputs.halves or [])]:
                if one.kda is not None:
                    one.kda.release_derived()

    def _prefill_layers(self, inputs, states, start, after_stage):
        mark = after_stage or (lambda name: None)
        t = inputs.t
        # mHC over the row's chips by tokens (common.ROW_SPLIT): the streams of this chip's quarter of the row's tokens
        # (`mt` of them) from the start to the end, the sub-layers on the row's t tokens (TtDecoderLayer._forward)
        split = row_split_on(t)
        mt = t // COLS if split else t
        if split and t not in _ROW_SPLIT_SEEN:  # (once a size: a measurement shows which form it took)
            _ROW_SPLIT_SEEN.add(t)
            print(f"model: mHC over the row's chips by tokens at {t} tokens a row ({mt} a chip)", flush=True)
        embedded = self.embed(inputs.ids)
        if split:
            mine = row_quarter_of(self.mesh, embedded)
            ttnn.deallocate(embedded)
            embedded = mine
        streams = expand_streams(self.mesh, embedded, self.config.hc_mult)
        if split:
            ttnn.deallocate(embedded)
        mark("embed")
        # the layers' mHC writes chained into the next layers' reads where the prefill's groups form takes them
        # (mhc.WRITE_READ_PREFILL), as the decode step chains them
        hc = self.layers[0].attn_hc
        chain = mhc_mod.WRITE_READ and hc.fused_read_slices(mt, hc.n * mt, prefill=True) is not None
        pending = None
        for i, layer in enumerate(self.layers):
            part = None if after_stage is None else (lambda name, i=i: after_stage(f"{i} {name}"))
            defer = chain and i + 1 < len(self.layers)
            new, states[i] = layer.prefill(
                streams,
                states[i],
                t,
                start,
                inputs.keep,
                mark=part,
                kda_picks=inputs.kda,
                dsa_tables=inputs.dsa,
                tap=self.taps if self.taps is not None and i in self.tap_layers else None,
                pending=pending,
                defer=defer,
                halves=[(h.kda, h.dsa) for h in inputs.halves] if inputs.halves else None,
            )
            if pending is None:
                ttnn.deallocate(streams)
            pending, streams = (new, None) if defer else (None, new)
            mark(f"{layer.kind}+{'moe' if self.config.is_moe(i) else 'dense'}")
        mtp = len(states) > len(self.layers) and (
            inputs.next_ids is not None or any(h.next_ids is not None for h in inputs.halves or [])
        )
        if inputs.last_token is None and not mtp and inputs.score_next is None and self.logits_all is None:
            ttnn.deallocate(streams)
            mark("head")
            return None
        hidden = collapse_streams(self.mesh, streams, self.config.hc_mult, mt)
        ttnn.deallocate(streams)
        if split:  # (the row's t tokens again)
            mine, hidden = hidden, row_tokens_gather(hidden)
            ttnn.deallocate(mine)
        if mtp and inputs.halves:  # a pair of waves: the MTP layer's cache one wave after the other
            normed = self.norm(hidden)
            n, h, d = len(self.layers), t // 2, normed.shape[-1]
            for i, half in enumerate(inputs.halves):
                part = ttnn.slice(normed, [0, 0, i * h, 0], [1, 1, (i + 1) * h, d])
                embeds = self.embed(half.next_ids)
                states[n] = self.mtp.prefill(embeds, part, states[n], start, half.dsa)
                ttnn.deallocate(embeds)
                ttnn.deallocate(part)
            ttnn.deallocate(normed)
            mark("mtp")
        elif mtp:  # the MTP layer's cache at the chunk's positions
            normed = self.norm(hidden)
            embeds = self.embed(inputs.next_ids)
            n = len(self.layers)
            states[n] = self.mtp.prefill(embeds, normed, states[n], start, inputs.dsa)
            ttnn.deallocate(embeds)
            ttnn.deallocate(normed)
            mark("mtp")
        if self.logits_all is not None:  # a test's: the logits at every position (each chip its quarter of the vocab)
            normed = self.norm(hidden)
            ttnn.deallocate(hidden)
            out = ttnn.linear(normed, self.lm_head.weight, dtype=ttnn.float32, compute_kernel_config=HIFI4_FP32)
            ttnn.deallocate(normed)
            mark("head")
            return out
        if inputs.score_next is not None:  # teacher forcing's scores of every position (`prefill_score`)
            normed = self.norm(hidden)
            ttnn.deallocate(hidden)
            out = self.lm_head.score_mesh(normed, inputs.score_next)
            ttnn.deallocate(normed)
            mark("head")
            return out
        if inputs.last_token is None:
            ttnn.deallocate(hidden)
            mark("head")
            return None
        last = ttnn.matmul(inputs.last_token, hidden, compute_kernel_config=HIFI4_FP32)  # [1, 1, 1, 4096]
        ttnn.deallocate(hidden)
        normed = self.norm(last)
        ttnn.deallocate(last)
        out = self.lm_head(normed)
        if self.mtp is not None:  # MTP's input at each row's last token, in the fixed buffer (`mtp_first_drafts`)
            ttnn.copy(normed, self._kept_hidden)
            ttnn.deallocate(normed)
            self._keep_prefill_hidden(self._kept_hidden, owned=False)
        else:
            ttnn.deallocate(normed)
        rm = ttnn.untilize_with_unpadding(out, ttnn.Shape([0, 0, 0, out.shape[-1] - 1]))  # (`prefill_logits_to_host`)
        ttnn.deallocate(out)
        out = rm
        mark("head")
        return out

    def prefill_key(self, start: int, t: int, last: bool, sp: bool = False) -> PrefillKey | None:
        """The captured prefill (`capture_prefills`) a chunk of T tokens from `start` replays, the wave's `last` chunk
        or not: without GLM53_PREFILL_TRACE_ALL a one-chunk wave's (start 0, last; its commit inside); with it any
        chunk's, the commit apart (`COMMIT`, replayed after the wave's last chunk).  None: no captured form (a later
        chunk of a dense cache not at a multiple of 128 or with the expanded cache: eager)."""
        if not PREFILL_TRACE_ALL:
            return PrefillKey(t, True, True) if start == 0 and last else None
        if sp:  # one form for any start (a long cache's: the pool pages a wave from `start`, its groups' last, scores)
            bucket = self.dsa.sp_bucket(start, t) if self.dsa is not None and self.dsa.long else 0
            return PrefillKey(t, False, False, bucket, True)
        if start == 0 or self.dsa is None:
            return PrefillKey(t, start == 0, False)
        bucket = self.dsa.traced_bucket(start, t)
        return None if bucket is None else PrefillKey(t, False, False, bucket)

    def prefill_trace_specs(self, t: int, tokens: int) -> list[tuple[PrefillKey, int]]:
        """The captures chunks of T tokens need, (key, a start that takes it), for the starts at multiples of T up to
        `tokens` (a schedule's chunks of one length start at multiples of it); with GLM53_PREFILL_TRACE_ALL and KDA
        layers, the commit too."""
        specs: dict = {}
        for start in range(0, max(t, tokens - t + 1), t):
            key = self.prefill_key(start, t, True)
            if key is not None and key not in specs:
                specs[key] = start
        if PREFILL_TRACE_ALL and self.sp_ready and t % 128 == 0 and t <= WAVE_ROWS:  # the SP waves (a group's start)
            for start in range(0, max(1, tokens - t + 1), t):
                specs.setdefault(self.prefill_key(start, t, False, sp=True), start)
        h = t // 2
        if PREFILL_TRACE_ALL and self.sp_ready and SP_PAIR and h == WAVE_ROWS:  # pairs of SP waves of h (from 0 on)
            for start in range(0, max(1, tokens - 2 * h + 1), h):
                for b in (ROWS, 4, 2, 1):
                    if start + 2 * b * h <= tokens:
                        specs.setdefault(self.pair_key(start, start + b * h, h), start)
                        break
        if PREFILL_TRACE_ALL and self.has_kda:
            specs[COMMIT] = 0
        return list(specs.items())

    def _trace_inputs(self, fixed: FixedInputs, states: list, key: PrefillKey, start: int) -> PrefillInputs:
        """A capture's inputs in its fixed buffers: user 0 on every row, all rows computing (`key.last`: the wave's
        last chunk, every row's prompt ending in it, its commit inside)."""
        tokens = torch.zeros(ROWS, key.t, dtype=torch.int64)
        if key.pair >= 0:  # a pair of waves (`_pair_capture_groups`)
            groups_a, groups_b = self._pair_capture_groups(key, start)
            return self.sp_pair_inputs(fixed.place, tokens, states, groups_a, groups_b, traced=True)
        if key.sp:  # one request: as many full segments from row 0 as the cache takes, user 0 of the last one's row
            n = min(ROWS, (self.dsa.max_seq - start) // key.t) if self.dsa is not None else ROWS
            group = SpGroup(tuple(range(n)), (key.t,) * n, 0, start, False)
            return self.sp_inputs(fixed.place, tokens, states, [group], traced=True)
        return self.prefill_inputs(fixed.place, tokens, start, states, [0] * ROWS, last=key.last, traced=True)

    def _kda_users(self, states: list) -> int:
        return next(st.recurrent.shape[0] for st, layer in zip(states, self.layers) if layer.kind == KDA)

    def commit_waves(self, states: list, picks) -> None:
        """Every KDA layer's rows' wave states into their users (`TtKda.commit`; `picks` from `TtKda.commit_picks`)."""
        for layer, state in zip(self.layers, states):
            if layer.kind == KDA:
                layer.attention.commit(state, picks)

    def capture_prefills(
        self, states: list, specs: list[tuple[PrefillKey, int]], prerun: bool = True, made: frozenset = frozenset()
    ) -> dict:
        """Capture the prefill of every (key, start) in `specs` (`prefill_trace_specs`) as a trace whose inputs sit in
        fixed buffers, one set per chunk length (`replay_prefill` rewrites them), and the commit (`COMMIT`,
        `replay_commit`) -> {key: PrefillTrace}.  Every form runs once eagerly first, and the fixed buffers are made
        then, all before the first capture (a buffer made after a capture may sit where that trace keeps its scratch:
        replaying it overwrites the buffer).  The runs write user 0 on every row (states are left meaningless,
        as after the generator's warm-up).  `prerun` False (the generator's GLM53_CAPTURE_PRERUN=0): the keys in `made`
        (the generator's warm-up ran an eager form with the captured form's programs, its `warm_keys`) have their fixed
        buffers made as before but not their eager run (their programs and constants are there: none added at 33K);
        every other key runs as before."""
        fixed: dict = {}  # chunk length (0: the commit) -> FixedInputs
        self.capture_programs_inside = 0  # programs made between a capture's begin and end (none expected)

        def inputs_of(key, start):  # (host writes: never inside a capture) -> (inputs, the packed layout or None)
            fx = fixed[key.t]
            fx.begin()
            if key == COMMIT:
                inputs = TtKda.commit_picks(fx.place, self._kda_users(states), [0] * ROWS)
            else:
                inputs = self._trace_inputs(fx, states, key, start)
            return inputs, fx.end()

        def run(key, start, inputs):
            if key == COMMIT:
                self.commit_waves(states, inputs)
                return None, None
            out = self._prefill(inputs, states, start)
            return out, self._device_ids(out)  # (PREFILL_IDS: the logits' argmax too)

        want_taps = self.capture_taps and bool(self.tap_layers)
        study = [] if PRERUN_STUDY else None  # (GLM53_CAPTURE_PRERUN_STUDY: what each form's eager run added)
        for key, start in specs:  # every constant the forms make comes before the first capture
            before = _programs_and_dram(self.mesh) if study is not None else None
            t0 = time.time()
            fixed.setdefault(key.t, FixedInputs(self.mesh))
            inputs, _ = inputs_of(key, start)
            if not prerun and key in made:  # (the buffers made; the eager run left out: the warm-up made its programs)
                continue
            self.taps = [] if want_taps and key != COMMIT else None
            out, ids = run(key, start, inputs)
            for t in self.taps or []:
                ttnn.deallocate(t)
            self.taps = None
            for t in (out, ids):
                if t is not None:
                    ttnn.deallocate(t)
            if study is not None:
                after = _programs_and_dram(self.mesh)
                study.append((tuple(key), start, after[0] - before[0], after[1] - before[1], time.time() - t0))
        ttnn.synchronize_device(self.mesh)
        if study:
            idle = [x for x in study if x[2] == 0 and x[3] == 0]
            print(
                f"capture pre-run study: {len(study)} forms, {sum(x[4] for x in study):.1f} s; {len(idle)} added no "
                f"program and no DRAM ({sum(x[4] for x in idle):.1f} s)",
                flush=True,
            )
            for key, start, programs, dram, seconds in study:
                print(f"  pre-run {key} @ {start}: +{programs} programs, {dram / 2**20:+.1f} MiB, {seconds:.2f} s",
                      flush=True)  # fmt: skip
        for fx in fixed.values():  # the packed rows (`prefill_pack`), made before any capture
            fx.make_pack()

        def capture(key, start, label=""):
            inputs, layout = inputs_of(key, start)
            fixed[key.t].unpack(layout)  # (its program compiled and loaded before the capture: the same values)
            used = trace_bytes(self.mesh)
            self.taps = [] if want_taps and key != COMMIT else None  # (kept: the trace's outputs, DFlash2's taps)
            programs = self.mesh.num_program_cache_entries()
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                fixed[key.t].unpack(layout)  # (the trace's first op: its inputs from the packed row)
                out, ids = run(key, start, inputs)
            except Exception:  # (the capture ended and released: a mesh left capturing can't close its fabric)
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                ttnn.release_trace(self.mesh, trace)
                raise
            ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
            self.capture_programs_inside += self.mesh.num_program_cache_entries() - programs
            taps, self.taps = self.taps, None
            size = trace_bytes(self.mesh) - used
            print(f"prefill trace {tuple(key)}{label}: {size / 2**20:.1f} MB of the trace region", flush=True)
            one = PrefillTrace(
                t=key.t, trace=trace, inputs=fixed[key.t], out=out, key=key, bytes=size, taps=taps, ids=ids
            )
            one.packed = _pack_sig(layout)
            if self.mtp is not None and out is not None:  # the hidden state it writes (`_keep_prefill_hidden`)
                one.hidden = self._prefill_hidden[0]
                one.hidden_shared = one.hidden is self._kept_hidden  # (the model's buffer)
                self._prefill_hidden = (one.hidden, False)
            return one

        parity = self.kda_spec_parity(states)
        captured = {}
        for key, start in specs:
            if parity is not None and key.last and key != COMMIT:
                raise NotImplementedError(
                    "the KDA ping-pong with a one-chunk wave's prefill trace (its commit inside): only "
                    "GLM53_PREFILL_TRACE_ALL's commit apart is captured per parity"
                )
            one = capture(key, start)
            if parity is not None and key == COMMIT:  # the ping-pong: the commit into the other set, its own capture
                self.kda_spec_set_parity(states, 1 - parity)
                try:
                    other = capture(key, start, f" (KDA set {1 - parity})")
                finally:
                    self.kda_spec_set_parity(states, parity)
                one.parity, one.other = parity, other
                one.bytes += other.bytes
            captured[key] = one
        return captured

    def capture_prefill(self, states: list, t: int) -> "PrefillTrace":
        """Capture the prefill of a one-chunk wave (T tokens from position 0, every request ending in it): see
        `capture_prefills`."""
        key = PrefillKey(t, True, True)
        return self.capture_prefills(states, [(key, 0)])[key]

    def replay_prefill(
        self,
        captured: "PrefillTrace",
        tokens: torch.Tensor,
        states: list,
        slots: list,
        rows: list[bool] | None = None,
        valid: int | list[int] | None = None,
        start: int = 0,
        logits: bool = True,
        next_tokens: torch.Tensor | None = None,
        greedy_ids: bool = False,
    ) -> torch.Tensor | None:
        """A chunk through its captured prefill (`capture_prefills`, `prefill_key`; arguments as `prefill`, the
        start and whether it is the wave's last chunk given by the key) -> the logits [ROWS, V] fp32 on the host, or
        None without `logits` (the trace computes them anyway: nothing read back); `greedy_ids`: each row's greedy
        token [ROWS] int64 instead (the trace's argmax with PREFILL_IDS at its capture, else the host's)."""
        key = captured.key or PrefillKey(captured.t, True, True)
        captured.inputs.begin()
        self.prefill_inputs(
            captured.inputs.place,
            tokens,
            start,
            states,
            slots,
            rows,
            valid,
            last=key.last,
            traced=True,
            next_tokens=next_tokens,
        )
        captured.check_packed(captured.inputs.end())
        ttnn.execute_trace(self.mesh, captured.trace, cq_id=0, blocking=False)
        if captured.hidden is not None:
            self._keep_prefill_hidden(captured.hidden, owned=False)
        if captured.taps is not None and self.taps is not None:  # (the trace's own buffers: read them, keep them)
            self.taps.extend(captured.taps)
        if not logits:
            return None
        return self.prefill_read(captured, slots, rows, greedy_ids)

    def prefill_read(self, captured: "PrefillTrace", slots: list, rows=None, greedy_ids: bool = False) -> torch.Tensor:
        """A replayed chunk's logits [ROWS, V] fp32 (or with `greedy_ids` its ids [ROWS] int64) read back, the rows
        computing only (`replay_prefill`'s read; generator.CTX_EARLY reads it after the drafter's context write)."""
        computing = [s is not None for s in slots] if rows is None else rows  # (the others' logits unread)
        if greedy_ids:
            if captured.ids is not None:
                return self.prefill_ids_to_host(captured.ids, captured.out, computing)
            return self.prefill_logits_to_host(captured.out, computing).argmax(-1)
        return self.prefill_logits_to_host(captured.out, computing)

    def replay_commit(self, captured: "PrefillTrace", states: list, slots: list) -> None:
        """After a wave's last chunk replayed from a trace without its commit (GLM53_PREFILL_TRACE_ALL): each mesh row's
        wave state into its user `slots[row]` (None: no request) through the captured commit (`COMMIT`)."""
        captured.inputs.begin()
        TtKda.commit_picks(captured.inputs.place, self._kda_users(states), slots)
        captured.check_packed(captured.inputs.end())
        ttnn.execute_trace(self.mesh, self._by_parity(captured, states).trace, cq_id=0, blocking=False)
        self._kda_committed(states)

    # -- trace ---------------------------------------------------------------------------------------------------
    def capture(self, io: "DecodeIO", states: list, greedy: bool = False):
        """Compile the forward once (it runs: the states advance one step), then capture it.  Returns (trace id, the
        output tensor the trace writes: logits, or ids with greedy).  The capture itself does not run the step."""
        out = self.forward(io, states, greedy=greedy)
        ttnn.deallocate(out)
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        out = self.forward(io, states, greedy=greedy)
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        if self.kda_spec_parity(states) is not None:  # (the ping-pong: it works on the set of its capture's parity)
            self._plain_traces.append((trace, states, self.kda_spec_parity(states)))
        return trace, out

    def replay(self, trace, logits, batch: int) -> torch.Tensor:
        for t, states, parity in self._plain_traces:
            if t is trace and self.kda_spec_parity(states) != parity:
                raise RuntimeError(
                    f"a plain decode trace captured at KDA set {parity} replayed at set {self.kda_spec_parity(states)}"
                )
        ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)
        return self.logits_to_host(logits, batch)


@dataclass
class PrefillInputs:
    """A prefill chunk's per-row inputs on the device (`TtGlm53Model.prefill_inputs`)."""

    t: int
    ids: object  # [1, T] uint32 per mesh row
    keep: object  # [1, 1, 1, 1] per row: 1 where the row computes its chunk
    kda: object  # TtKda.prefill_picks (None without KDA layers)
    dsa: tuple | None  # TtDsa.prefill_tables (None without DSA layers)
    last_token: object  # [1, 1, 1, T] per row: the one-hot of its last real token (None: no logits)
    owned: bool = True  # new tensors (not a trace's fixed buffers)
    next_ids: object = None  # [1, T] uint32 per row: each position's next token (the MTP layer's prefill)
    score_next: object = None  # `prefill_score`'s: every row's positions' next tokens [1, 1, 8 T, 1] fp32 (-1: none)
    # a pair of sequence-parallel waves (`sp_pair_inputs`): each wave's inputs (its half of the rows' tokens), whose
    # KDA, DSA and MTP inputs the attention layers take one wave after the other; `kda` then holds the MoE's real-token
    # mask of both
    halves: list | None = None

    def deallocate(self):
        if not self.owned:
            return
        for t in (self.ids, self.keep, self.last_token, self.next_ids, self.score_next, *(self.dsa or ())):
            if isinstance(t, ttnn.Tensor):  # (the long cache's tables carry a flag)
                ttnn.deallocate(t)
        if self.kda is not None:
            self.kda.deallocate()
        for half in self.halves or []:
            half.deallocate()


def _pack_sig(layout) -> tuple | None:
    """A packed layout's fields, pages and offsets (`prefill_pack.PackSession.layout`), or None."""
    return None if layout is None else tuple((name, page, pages, at) for name, _, page, pages, _, at in layout)


def trace_bytes(mesh) -> int:
    """The bytes of the trace region taken (every chip alike), over all DRAM banks."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
    return int(view.total_bytes_allocated_per_bank) * int(view.num_banks)


@dataclass
class PrefillTrace:
    t: int
    trace: object
    inputs: FixedInputs  # the buffers the trace reads (shared by a chunk length's traces)
    out: object  # the logits [1, 1, 1, V] fp32 per row it writes
    key: PrefillKey | None = None  # what it captured (None: a one-chunk wave's, PrefillKey(t, True, True))
    bytes: int = 0  # of the trace region
    hidden: object = None  # with the MTP layer: the final normed hidden state at each row's last token it writes
    hidden_shared: bool = False  # `hidden` is the model's kept buffer (`_kept_hidden`): not freed with the trace
    taps: list | None = None  # with `capture_taps`: the taps it writes (DFlash2's drafter's inputs)
    packed: tuple | None = None  # its inputs' packed layout (`prefill_pack`: its first op unpacks them)
    ids: object = None  # PREFILL_IDS at its capture: the logits' argmax [1, 1, 1, 1] uint32 per row it writes
    # the KDA ping-pong's commit (`kda_spec_parity`): the KDA set this capture writes, and the capture for the other
    # (`other`, the same inputs; `_by_parity` picks)
    parity: int | None = None
    other: object = None

    def check_packed(self, layout) -> None:
        """A replay's packed layout is the capture's (the trace's unpack copies the capture's layout)."""
        if _pack_sig(layout) != self.packed:
            raise RuntimeError(f"a prefill replay's packed inputs differ from its capture's ({self.key})")

    def release(self, mesh) -> None:
        ttnn.release_trace(mesh, self.trace)
        if self.other is not None:  # (the ping-pong's commit for the other set: its trace only, the inputs shared)
            ttnn.release_trace(mesh, self.other.trace)
            self.other = None
        if self.out is not None:  # (the commit's: none)
            ttnn.deallocate(self.out)
        if self.ids is not None:
            ttnn.deallocate(self.ids)
        if self.hidden is not None and not self.hidden_shared:
            ttnn.deallocate(self.hidden)
        for t in self.taps or []:
            ttnn.deallocate(t)
        self.inputs.deallocate()


@dataclass
class FirstIO:
    """`mtp_first_drafts`' inputs at fixed addresses and its captured step (`TtGlm53Model.capture_first_drafts`)."""

    hidden: object  # [1, 1, 1, 4096] per row: the row's kept hidden state, copied in before each step
    pick: object  # [1, 1, U, 1] bf16 per row: 1 at its request's user
    trace: object = None
    out: object = None  # the trace's ids [1, 1, 8 U, 1]


@dataclass
class SpecIO:
    """The verify step's inputs at fixed addresses (`TtGlm53Model.alloc_spec_io`)."""

    users: int  # a row's; the step takes (drafts + 1) users tokens
    drafts: int  # a user: 1 or 2 with the MTP layer (`forward_spec`), any with drafts from elsewhere (`forward_verify`)
    ids: object  # [1, kU] uint32 row-major per chip: the U users' tokens, then their drafts
    positions: object  # TtDsa.alloc_positions(kU, users=U): virtual user jU + u is user u j positions on
    positions2: object  # with two drafts the MTP layer's second level's: positions + 1 (else None)
    keep: SpecKeep
    others: object  # [1, 1, 8, 1] fp32 per row: 0 at its own row, 1e9 at the others (constant, `_row_ids`)
    forced: object  # [1, 1, 1, kU] fp32: the MTP layer's host-given next tokens (0 where none)
    free: object  # [1, 1, 1, kU] fp32: 1 where none is given
    forced_none: bool = True  # forced and free hold none (as made): `write_spec_inputs` leaves them
    pack: object = None  # ids, the positions' row-major buffers, keep's flags in one write (`InputPack`)
    live: object = None  # moe_route.LiveTokens of the kU tokens a row
    no_next: object = None  # the looser acceptance's: [1, 1, 1, U] fp32 -1 (the last level's next tokens, `_relax`)


@dataclass
class DecodeIO:
    batch: int
    ids: object  # [1, B] uint32 row-major per chip, fixed address
    positions: object  # TtDsa.alloc_positions (DsaPositions; None without DSA layers), fixed addresses
    pack: object = None  # ids and the positions' row-major buffers in one write (`InputPack`)
    live: object = None  # moe_route.LiveTokens of the B tokens a row
