"""Serving on the 8x4 mesh: request slots, chunked prefill per slot, one traced decode step for every slot.

Physical slot s (0 .. 8 U - 1) is user s % U of mesh row s // U.  A request's prompt is prefilled into its slot
(eager, in chunks chosen per prompt, `schedule`), up to 8 requests at once, one per mesh row (`prefill_many`; the
other slots keep their state); starting at position 0 begins the slot afresh (KDA from an empty state; DSA writes the
cache rows as the prompt goes, and decode writes each later position before attending to it), so a finished request's
slot needs no clearing.  Decode runs every slot each step from a captured
trace, returning the greedy ids chosen on the device or the full logits.  A slot without a request (position -1) runs
token 0 at position 0, choosing no expert (`moe_route.LiveTokens`): it changes only its own state, which the next
prefill into it rewrites.

The traces are captured once, before the first request and after the prefill's warm-up (`warmup`): a capture's
compile run advances every slot.  A wave that fits one chunk of a captured length (the prefill traces, e.g. 128-token
prompts) replays its trace; the prefill is otherwise eager.  With `dsa.PREFILL_SP` (on) a round of requests goes over the
8 rows when that takes fewer waves (`sp_plan`: each request's consecutive segments on the rows that end at its own,
sequence parallel), e.g. a single prompt longer than a chunk.  With GLM53_PREFILL_TRACE_ALL=1 every chunk of a captured
length replays a trace (`TtGlm53Model.prefill_key`: first or later, a long cache's pool bucket), and a wave whose
last chunk did so writes its rows' KDA states into their users with one more (`TtGlm53Model.replay_commit`).

With `dflash` the speculative decode is DFlash2's in place of MTP's: each prefill chunk's taps
(the drafter's target layers' stream means) go into the drafter's context (`TtDFlash2.write_context`), a finished
prompt's first drafts come from the drafter (`first`), and `dflash_verify` replays the whole step (verify, acceptance,
context write and the next block, `dflash2_step.DFlash2Step`).  One request a row (no SP); a prefill trace keeps its
taps as outputs (`TtGlm53Model.capture_taps`).
"""

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.blackhole.glm53_flash.ttnn import model as model_module
from models.demos.blackhole.glm53_flash.ttnn.common import COLS, ROWS, WAVE_ROWS, SpGroup, arm_trace_alloc_guard, put
from models.demos.blackhole.glm53_flash.ttnn.dflash2_step import DFlash2Step, FirstDrafts, verify_logits_by_slot
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model

PREFILL_ALIGN = 32  # a chunk's rows (KDA's chunk ops and the cache writes need multiples of 32)
SP_ALIGN = 128  # a sequence-parallel segment's (the DSA's K chunk, a long cache's pool pages)
# ... and its longest: a segment's KDA summary (bf16 maps) and its rows' order change the values with its length, so a
# wave of more than WAVE_ROWS tokens a row stays one row each
SP_MAX = WAVE_ROWS
SP_WAVE = (323, 1)  # a prefill wave's cost a + b T, T its chunk length (relative: 0.14 s at 128, 0.27 s at 512)
# DFlash2 with the sequence-parallel prefill: each row's segment's drafter K / V all-gathered over the rows, the
# request's own row writes them (`TtDFlash2.write_context`'s gathered form; serving with 1K+ prompts was slower than
# MTP's on the one-row prefill; the vLLM flow's acceptance as before).  False: one request a row
DFLASH2_SP = True
# ... and its first drafts after a prefill replayed from a trace (`dflash2_step.FirstDrafts`).  False: eager
FIRST_TRACE = True
# The warm-up's eager prefill runs each program a chunk start can take once (`_warm_forms`: a long cache's later chunks
# by the pool pages they score, the sequence-parallel waves and pairs too), in place of a one-row chunk at every start
# up to the cache's end (that took ~2.4 s per 1K tokens and chunk length at a long cache, and the sequence-parallel
# forms compiled at their first use, after the captures)


def _first_id(x, vocab: int) -> int:
    """A prefill's result for a request -> its greedy first token: an id (`prefill_many(greedy_ids=True)`: an int or a
    one-element tensor) as it is, else the argmax of its logits [V] fp32 (the first index of the maximum)."""
    if isinstance(x, int):
        return x
    if x.numel() == 1:
        return int(x.reshape(-1)[0])
    return int(x.reshape(-1)[:vocab].argmax())


def _rows(x, lo: int, hi: int):
    """Rows lo .. hi - 1 of x [.., R, D] (a new tensor)."""
    r = len(x.shape)
    return ttnn.slice(x, [0] * (r - 2) + [lo, 0], [x.shape[i] for i in range(r - 2)] + [hi, x.shape[-1]])


def _wave_cost(t: int) -> int:
    return SP_WAVE[0] + t * SP_WAVE[1]


class Glm53Generator:
    def __init__(
        self,
        model: TtGlm53Model,
        users_per_row: int,
        chunk: int | tuple[int, ...] = 128,
        spec: int = 0,
        dflash: tuple | None = None,
    ) -> None:
        """`chunk`: the prefill chunk length, or several (each dividing the next), chosen per prompt (`schedule`).
        `spec`: MTP self-speculative decode with that many drafts a step (1 or 2; the model loaded with its MTP layer;
        `verify`).  `dflash`: (TtDFlash2 of users_per_row users a row, K): DFlash2's with K drafts a step
        instead (`dflash_verify`)."""
        if spec and model.mtp is None:
            raise ValueError("speculative decode needs the model's MTP layer (GLM53_MTP=1)")
        if spec and dflash is not None:
            raise ValueError("MTP's speculative decode or DFlash2's, not both")
        chunks = tuple(sorted({chunk} if isinstance(chunk, int) else set(chunk)))
        if any(c % PREFILL_ALIGN for c in chunks) or any(b % a for a, b in zip(chunks, chunks[1:])):
            raise ValueError(f"prefill chunks {chunks}: multiples of {PREFILL_ALIGN}, each dividing the next")
        self.model = model
        self.mesh = model.mesh
        self.users = users_per_row
        self.chunks = chunks
        self.chunk = chunks[-1]  # the longest
        from models.demos.blackhole.glm53_flash.ttnn import moe as moe_module

        if moe_module.COL_ROW_RS:  # (the prefill's way back as one op: its buffers for the tokens a row it will take)
            from models.demos.blackhole.glm53_flash.ttnn import col_row_rs

            col_row_rs.prealloc(self.mesh, ts=col_row_rs.ts_for(chunks, SP_MAX if model_module.SP_PAIR else None))
        self.max_seq = model.dsa.max_seq if model.dsa is not None else None
        self.spec = int(spec)
        self.drafter, self.dk = dflash if dflash is not None else (None, 0)
        if self.drafter is not None and self.drafter.users != users_per_row:
            raise ValueError(f"the drafter's {self.drafter.users} users a row, not {users_per_row}")
        need = (model.dsa.max_seq if model.dsa is not None else 0) + (self.drafter.block if self.drafter else 0)
        if self.drafter is not None and self.drafter.max_ctx < need:
            raise ValueError(
                f"the drafter's {self.drafter.max_ctx}-position cache: the model's cache + its block, {need}"
            )
        self.states = model.empty_state(users_per_row, spec=self.spec or self.dk)
        self.io = model.alloc_io(users_per_row)
        self.sio = model.alloc_spec_io(users_per_row, drafts=self.spec) if spec else None
        self.spec_trace = None  # (trace id, ids, drafts[, drafts2]): the captured verify step
        # slot -> the first draft after its prompt (`_first_drafts`; with two drafts the pair), taken by the caller
        self.first = {}
        # each physical slot's position for a verify step that it sits out: 0 (as `next_inputs` makes it), but from a
        # prefill into it to its first step with a request (a prompt partway between its chunks, tools/glm53_vllm.py
        # CHUNKED) the end of that prompt so far, where its writes (DSA's latent row, indexer key and recent row, the
        # drafter's context) land on rows the next chunk writes again (at 0 they overwrote the prompt's first rows)
        self.idle_pos = torch.zeros(self.slots, dtype=torch.int64)
        self.verify_seconds = (0.0, 0.0)  # the last `verify` / `decode`: its host writes, its replay with the readback
        self.traces: dict[bool, tuple] = {}  # greedy -> (trace id, its output tensor)
        self.prefill_traces: dict = {}  # PrefillKey -> the captured prefill (TtGlm53Model.capture_prefills)
        self.traced_chunks: set = set()  # the chunk lengths captured
        # prefill_many's, since the start (traced_waves: every chunk replayed)
        self.counts = {"waves": 0, "traced_waves": 0, "chunks": 0, "traced_chunks": 0, "sp_pairs": 0}
        self.sp = True  # (with `dsa.PREFILL_SP`) a round of requests may go over the rows (`sp_plan`)
        self.dstep = None  # DFlash2's step (`dflash`)
        # dflash2_step.next_inputs_on(): the inputs the last step made on the device for the next one
        # (`NextInputs.predict`), and how many steps took them (the host's write skipped) or wrote their own
        self.dflash_next = None
        self.dflash_writes = {"skipped": 0, "written": 0}
        # a step launched ahead (`dflash_verify(.., ahead=True)`): None, "live" (its
        # results for the next call if that call's inputs are the predicted ones) or "void" (thrown away: a prefill came
        # after it, `dflash_void`)
        self.dflash_ahead = None
        self.dflash_logits = None  # (the last `dflash_verify(.., logits=True)`'s [slots, K + 1, V] fp32)
        if self.drafter is not None:
            self.dstep = DFlash2Step(model, self.drafter, users_per_row, self.dk)
            self.dfirst = FirstDrafts(self.drafter)  # (the first drafts after a prefill, captured after the step)
            model.tap_layers = frozenset(layer + 1 for layer in self.drafter.cfg.target_layer_ids)
            model.capture_taps = True
            self.sp = DFLASH2_SP  # (else one request a row, each row's taps its own request's)
            # the block of a slot without a bonus yet (a first drafts' call for others): past every context it reads.
            # With the drafter's ring (dflash2.RING, a cache longer than it) -1: its K / V not written (kv_write skips a
            # position < 0); the scratch block's slots would be live positions of the row's other requests
            self.scratch = -1 if self.drafter.ring else self.drafter.max_ctx - self.drafter.block

    @property
    def slots(self) -> int:
        return ROWS * self.users

    # -- set-up ----------------------------------------------------------------------------------------------------
    def warmup(
        self,
        greedy: bool = True,
        logits: bool = True,
        chunks: tuple | None = None,
        tokens: int | None = None,
        traced: tuple = (),
    ) -> dict:
        """Compile the prefill (the chunk starts up to `tokens`, default the cache length, for each chunk length in
        `chunks`, default this generator's: a schedule's chunks of one length start at multiples of it, so this
        covers them: one start a program, and the sequence-parallel forms, `_warm_forms`), capture the
        one-chunk prefill for the chunk lengths in `traced` (GLM53_PREFILL_TRACE_ALL: every
        form their chunks take up to the cache length, `TtGlm53Model.prefill_trace_specs`), and then capture the
        decode traces (greedy ids and / or full logits).

        The order matters: the prefill and the decode make device constants on first use (the mHC matrices per chunk
        length and per batch) and a prefill trace its input buffers, all of which stay allocated, and a buffer
        allocated after a capture may sit where that trace keeps its scratch tensors (replaying it overwrites them:
        a decode after a prefill trace hung); so every capture comes after them.  After the captures, prefill
        allocates only temporaries.  Every slot's state is left meaningless (a request's prefill from position 0
        starts its slot afresh).

        The same holds across calls: a call that captures anything new first releases what earlier calls captured
        (their buffers would precede this call's allocations) and captures it again with its own, after them.  The
        vLLM adapter warms up the prefill and MTP's verify step first, then the decode with the prefill's captures:
        with the verify step's and the first drafts' traces kept from the first call, the plain decode read zeros
        once the first drafts' trace had run)."""
        arm_trace_alloc_guard(self.mesh, False)  # (GLM53_TRACE_ALLOC_GUARD: watches the replays after the warm-up)
        seconds = {}
        new = [t for t in dict.fromkeys(traced) if t not in self.traced_chunks]
        kinds = [kind for want, kind in ((logits, False), (greedy, True)) if want]
        more = bool(new) or any(k not in self.traces for k in kinds) or bool(self.spec and self.spec_trace is None)
        more = more or bool(self.dstep is not None and self.dstep.trace is None)
        if more and (self.traces or self.spec_trace is not None or self.prefill_traces or self._dflash_captured()):
            traced = tuple(dict.fromkeys(tuple(sorted(self.traced_chunks)) + tuple(traced)))
            logits, greedy = logits or False in self.traces, greedy or True in self.traces
            self.release()
            seconds["recaptured"] = True
            new = [t for t in dict.fromkeys(traced) if t not in self.traced_chunks]
        tokens = tokens or self.max_seq or self.chunk
        seconds.update(self._warm_forms(chunks or self.chunks, tokens))
        zeros = torch.zeros(ROWS, self.users, dtype=torch.int64)
        if more:  # the decode makes constants on first use too (the mHC matrices for its
            # batch): before any capture
            if any(t not in self.chunks for t in new) or self.traces:
                raise ValueError(f"prefill traces of {new} tokens: not all of {self.chunks}, or after the decode's")
            t0 = time.time()
            self.model.write_inputs(self.io, zeros, zeros)
            for kind in (False, True):
                ttnn.deallocate(self.model.forward(self.io, self.states, greedy=kind))
            if self.spec:
                self._spec_compile()
            if self.dstep is not None:  # DFlash2's step and first drafts: their constants before any capture
                self._dflash_first({})
                self.dstep.write(*self._dflash_inputs(*self._dflash_idle()))
                self.dstep.compile(self.states)
                if self.sp:  # the sequence-parallel waves' context write (nothing written: no groups)
                    for t in (c for c in self.chunks if c % SP_ALIGN == 0 and c <= SP_MAX):
                        zero = put(
                            self.mesh, torch.zeros(1, 1, t, len(self.model.tap_layers) * 4096), dims=(None, None)
                        )
                        self._dflash_context_sp(zero, [], t, owned=False)
                        ttnn.deallocate(zero)
            seconds["decode_compile"] = round(time.time() - t0, 1)
        if new:
            t0 = time.time()
            cover = self.max_seq or max(new)
            specs = dict(spec for t in new for spec in self.model.prefill_trace_specs(t, cover))
            captured = self.model.capture_prefills(self.states, list(specs.items()))
            self.prefill_traces.update(captured)
            self.traced_chunks.update(new)
            seconds["capture_prefill_" + "_".join(map(str, new))] = round(time.time() - t0, 1)
            seconds["prefill_traces"] = len(captured)
            seconds["prefill_trace_mb"] = round(sum(c.bytes for c in captured.values()) / 2**20, 1)
        if self.spec and self.spec_trace is None:  # its constants before any capture, then its own capture
            self._spec_compile()
            t0 = time.time()
            used = model_module.trace_bytes(self.mesh)
            self.spec_trace = self.model.capture_spec(self.sio, self.states)
            seconds["capture_verify"] = round(time.time() - t0, 1)
            seconds["verify_trace_mb"] = round((model_module.trace_bytes(self.mesh) - used) / 2**20, 1)
            # the first drafts' step after a prefill (`TtGlm53Model.mtp_first_drafts`: the MTP layer and the head at
            # each arriving request's last prompt position) replayed from a trace too, not eager (~100 ops a request:
            # serving at 256 requests paid it per arrival); made by _spec_compile's eager call
            if self.spec == 1:
                used = model_module.trace_bytes(self.mesh)
                self.model.capture_first_drafts(self.users, self.states)
                seconds["first_drafts_trace_mb"] = round((model_module.trace_bytes(self.mesh) - used) / 2**20, 1)
        if (
            self.dstep is not None and self.dstep.trace is None
        ):  # DFlash2's step: compiled above, captured after the rest
            t0 = time.time()
            self.dstep.capture(self.states)
            if FIRST_TRACE:
                self.dfirst.capture()
            seconds["dflash_step_capture"] = round(time.time() - t0, 1)
        for want, kind in ((logits, False), (greedy, True)):
            if want and kind not in self.traces:
                t0 = time.time()
                self.model.write_inputs(self.io, zeros, zeros)
                used = model_module.trace_bytes(self.mesh)
                self.traces[kind] = self.model.capture(self.io, self.states, greedy=kind)
                name = "greedy" if kind else "logits"
                seconds[f"capture_{name}"] = round(time.time() - t0, 1)
                seconds[f"decode_trace_{name}_mb"] = round((model_module.trace_bytes(self.mesh) - used) / 2**20, 1)
        self.idle_pos.zero_()  # (the warm-up's prompts are not partway)
        arm_trace_alloc_guard(self.mesh)
        return seconds

    def _warm_forms(self, chunks, tokens: int) -> dict:
        """The warm-up's eager prefill up to `tokens`: for each chunk length, a one-row wave of one
        request (row 0) whose chunks run only at `_row_starts`, and one of a request on every row (every chunk full:
        the forms without the real-token mask, `TtKda.prefill_picks`: KDA's conv and gate and the MoE's plan, which do
        not change with the pool bucket; else a wave of 8 equal prompts compiled them after the captures) at its
        first, second and last chunks, then, when a round may go over the rows, the sequence-parallel forms
        (`_warm_sp`) -> the seconds and the chunks / forms run, a chunk length each.  The states are left meaningless
        (as before: a request's prefill from position 0 starts its slot afresh)."""
        seconds = {}
        sp, self.sp = self.sp, False  # (the one-row form first)
        try:
            for chunk in chunks:
                t0 = time.time()
                starts = self._row_starts(chunk, tokens)
                # (greedy_ids: the last chunk's ids by the device's argmax as well, `model.PREFILL_IDS`; else an eager
                # wave's argmax compiled after the captures, its program's buffer alive at a replay)
                self.prefill_many([(0, [0] * tokens)], chunk=chunk, traced=False, starts=starts, greedy_ids=True)
                every = [s for s, _ in self.schedule(tokens, chunk)]
                full = {every[0], every[min(1, len(every) - 1)], every[-1]}
                self.prefill_many(
                    [(r * self.users, [0] * tokens) for r in range(ROWS)],
                    chunk=chunk,
                    traced=False,
                    starts=full,
                    greedy_ids=True,
                )
                seconds[f"prefill_{chunk}_compile"] = round(time.time() - t0, 1)
                seconds[f"prefill_{chunk}_starts"] = len(starts) + len(full)
        finally:
            self.sp = sp
        if self.sp and self.model.sp_ready:
            for chunk in (c for c in chunks if c % SP_ALIGN == 0 and c <= SP_MAX):
                t0 = time.time()
                seconds[f"prefill_sp_{chunk}_forms"] = self._warm_sp(chunk, tokens)
                seconds[f"prefill_sp_{chunk}_compile"] = round(time.time() - t0, 1)
        return seconds

    def _row_starts(self, t: int, tokens: int) -> set:
        """The starts of a one-row wave's chunks of T (`schedule`'s, up to `tokens`) whose eager runs compile every
        program the others take: a long cache's first chunk, of the later ones the first at each pool bucket (the
        pool pages its indexer scores, `TtDsa.pool_bucket`: the one part of a later chunk's programs its start
        changes; without it 5 buckets' compiled after the captures), and the last (the wave's last chunk: the KDA commit
        and the logits); every start for a dense cache or none (its later chunks' attended pages and K chunks follow
        the start)."""
        starts = [s for s, _ in self.schedule(tokens, t)]
        dsa = self.model.dsa
        if dsa is None or not dsa.long:
            return set(starts)
        seen, keep = set(), {starts[-1]}
        for s in starts:
            form = dsa.pool_bucket(s, t) if s else -1  # (-1: the first chunk's form)
            if form not in seen:
                seen.add(form)
                keep.add(s)
        return keep

    def _sp_forms(self, t: int, tokens: int) -> list:
        """The sequence-parallel forms a round of segments of T can take up to `tokens` -> [(start, rows, start_b,
        last)]: a wave of one request from `start` on rows 0 .. rows - 1 (start_b None), or a pair of waves
        (`prefill_sp_pair`: T = SP_MAX, SP_PAIR) of a row each, the second from start_b; one a program: a long cache's
        waves by the pool pages they score (`TtDsa.sp_bucket`: a wave's groups' largest), a pair by its two waves' (the
        second's groups are the first's that go on, each from at most 8 T further: up to the bucket 8 T past the
        first's, any below it), a dense cache's one whatever the starts.  The first wave and the first pair end their
        request (`last`: the KDA commit inside and the logits)."""
        dsa = self.model.dsa
        bucket = (lambda s: dsa.sp_bucket(s, t)) if dsa is not None and dsa.long else (lambda s: 0)  # noqa: E731
        end = min(tokens, self.max_seq or tokens)
        at: dict = {}  # a bucket -> the first start in it
        for s in range(0, end - t + 1, t):
            at.setdefault(bucket(s), s)
        forms = [(s, max(1, min(ROWS, (end - s) // t)), None, i == 0) for i, s in enumerate(at.values())]
        if not (model_module.SP_PAIR and t == SP_MAX):
            return forms
        seen = set()
        for s in range(0, end - t + 1, t):
            reach = bucket(min(s + ROWS * t, end - t))
            for b, s_b in at.items():
                if b <= reach and (bucket(s), b) not in seen:
                    seen.add((bucket(s), b))
                    forms.append((s, 1, s_b, len(seen) == 1))
        return forms

    def _warm_sp(self, t: int, tokens: int) -> int:
        """`_warm_forms`' sequence-parallel part for segments of T: each of `_sp_forms` once, eagerly as an eager
        round runs it (`_prefill_sp_round`: the wave or the pair, its taps into the drafter's context), request 0's
        user 0 on the rows -> the forms run."""
        forms = self._sp_forms(t, tokens)
        for start, b, start_b, last in forms:
            rows = tuple(range(b))
            waves = [[SpGroup(rows, (t,) * b, 0, start, last and start_b is None)]]
            if start_b is not None:
                waves.append([SpGroup(rows, (t,) * b, 0, start_b, last)])
            if self.drafter is not None:
                self.model.taps = []
            ids = torch.zeros(ROWS, t * len(waves), dtype=torch.int64)
            if start_b is None:
                self.model.prefill_sp(ids, self.states, waves[0])
            else:
                self.model.prefill_sp_pair(ids, self.states, waves[0], waves[1])
            if self.drafter is not None:  # (as `_prefill_sp_round`'s eager wave and pair)
                taps, self.model.taps = self.model.taps, None
                if start_b is None:
                    self._dflash_context_sp(taps, waves[0], t, owned=True)
                else:
                    for i, groups in enumerate(waves):
                        self._dflash_context_sp([_rows(x, i * t, (i + 1) * t) for x in taps], groups, t, owned=True)
                    for x in taps:
                        ttnn.deallocate(x)
        return len(forms)

    def _spec_compile(self) -> None:
        """MTP's verify step and the first drafts' step once eagerly (their device constants), states meaningless."""
        if not self.spec:
            return
        k = self.spec
        zeros = torch.zeros(ROWS, (k + 1) * self.users, dtype=torch.int64)
        self.model.write_spec_inputs(self.sio, zeros, zeros, torch.full((ROWS, self.users), k))
        for t in self.model.forward_spec(self.sio, self.states):
            ttnn.deallocate(t)
        if getattr(self.model, "_prefill_hidden", None) is not None:
            self.model.mtp_first_drafts(
                self.states, torch.zeros(ROWS, dtype=torch.int64), [(0, 0)] * ROWS, self.users, drafts=k
            )

    def _dflash_captured(self) -> bool:
        return self.dstep is not None and self.dstep.trace is not None

    def _dflash_idle(self) -> tuple:
        """Every slot without a request: the inputs of `dflash_verify`."""
        n = self.dk + 1
        return (
            torch.zeros(self.slots, n, dtype=torch.int64),
            torch.full((self.slots, n), -1, dtype=torch.int64),
            torch.full((self.slots,), self.dk, dtype=torch.int64),
        )

    def release(self) -> None:
        if (
            self.dstep is not None and self.dflash_ahead == "live"
        ):  # (a step launched ahead: done before its trace goes)
            self.dflash_void()
            ttnn.synchronize_device(self.mesh)
        self.dflash_ahead = None
        if self.dstep is not None:
            self.dfirst.release()
        if self.dstep is not None and self.dstep.trace is not None:  # (its buffers kept: a later warm-up recaptures)
            self.dstep.release_traces()
        for trace, out in self.traces.values():
            ttnn.release_trace(self.mesh, trace)
            ttnn.deallocate(out)
        self.traces.clear()
        if self.spec_trace is not None:
            ttnn.release_trace(self.mesh, self.spec_trace[0])
            for out in self.spec_trace[1:]:
                ttnn.deallocate(out)
            self.spec_trace = None
        self.model.release_first_drafts()
        for captured in self.prefill_traces.values():
            captured.release(self.mesh)
        self.prefill_traces.clear()
        self.traced_chunks.clear()
        hidden = getattr(self.model, "_prefill_hidden", None)
        if hidden is not None and not hidden[1]:  # a released capture's output: gone with it
            self.model._prefill_hidden = None

    # -- requests --------------------------------------------------------------------------------------------------
    def schedule(self, length: int, chunk: int | None = None) -> list[tuple[int, int]]:
        """The chunks (start, length) a prompt of `length` tokens goes in: `chunk` throughout if given, else the
        longest of this generator's chunk lengths while more than half of it is left (a long chunk costs less per
        token, a short one pads less), so the longer chunks come first, each at a multiple of its length."""
        sizes = (chunk,) if chunk else self.chunks[::-1]
        out, start = [], 0
        while start < length:
            t = next((c for c in sizes if length - start > c // 2), sizes[-1])
            out.append((start, t))
            start += t
        if self.max_seq is not None and start > self.max_seq:
            raise ValueError(f"a {length}-token prompt does not fit the {self.max_seq}-row cache")
        return out

    def prefill(self, slot: int, tokens: list[int], chunk: int | None = None, traced: bool = True) -> torch.Tensor:
        """A new request's prompt (positions 0 ..) into physical slot `slot`, in chunks of `chunk` tokens (default
        this generator's `schedule`; warmed up before the capture) -> the logits after its last token [V] fp32 on
        the host."""
        return self.prefill_many([(slot, tokens)], chunk, traced)[0]

    def prefill_many(
        self,
        requests: list[tuple[int, list[int]]],
        chunk: int | None = None,
        traced: bool = True,
        greedy_ids: bool = False,
        starts: set | None = None,
        begin: int = 0,
    ) -> list:
        """New requests' prompts, (physical slot, tokens) each -> the logits after each prompt [V] fp32, in order;
        `greedy_ids`: each one's greedy first token (an int) instead, the one-row waves' from the device's argmax
        (`model.PREFILL_IDS`: 8 ids read back a wave in place of the rows' logits), the others' from the logits.
        `starts` (the warm-up's, `_warm_forms`): of the one-row waves' chunks only those from these starts run, the
        wave's last among them (the states are left meaningless).
        `begin` (a chunked prefill's, tools/glm53_vllm.py CHUNKED): a single request's prompt so far, of which the
        tokens before `begin` went in by earlier calls with no other prefill between them (decode steps leave a row's
        wave state and the DSA cache's extra user alone): only its chunks / segments from `begin` run, as a call with
        the whole prompt would run them (the same choice of rows and lengths, from the prompt so far), so that the
        values are those of one call when `begin` is on the grid of every chunk and segment length (a multiple of
        8,192: one wave pair, `_sp_pair`).  Each call ends as a whole prompt does (the KDA commit into its user, the
        first drafts: written again by the next call).

        A slot lives on its mesh row, and each row computes one prefill chunk at a time, so the requests go in waves
        of up to one per row (each row's longest first, so that a wave's prompts are of similar lengths), chunk by
        chunk from position 0 (the wave's longest prompt's `schedule`); a row whose prompt has ended waits for the
        wave's last chunk.  A chunk with a captured form replays its trace (unless not `traced`): a one-chunk wave's
        chunk of a captured length, or with GLM53_PREFILL_TRACE_ALL every chunk of a captured length.
        Consecutive slot numbers share a row, so callers spread their requests over the rows (tools/glm53_vllm.py
        interleaves its slots over the rows).  With `dsa.PREFILL_SP` (on) a round of requests goes over the rows instead
        when that takes fewer waves (`sp_plan`, `_prefill_sp_round`)."""
        if begin and (len(requests) != 1 or not 0 < begin < len(requests[0][1])):
            raise ValueError(f"a prefill from position {begin}: one request, with tokens past it")
        for slot, tokens in requests:  # (a verify step it sits out writes there: `idle_pos`)
            if 0 <= slot < self.slots:
                self.idle_pos[slot] = min(len(tokens), self.max_seq - 2) if self.max_seq is not None else len(tokens)
        per_row: dict[int, list[int]] = {}  # mesh row -> its requests
        for i, (slot, tokens) in enumerate(requests):
            if not 0 <= slot < self.slots:
                raise ValueError(f"slot {slot} out of 0 .. {self.slots - 1}")
            if not tokens:
                raise ValueError("an empty prompt")
            self.schedule(len(tokens), chunk)  # fits the cache
            per_row.setdefault(slot // self.users, []).append(i)
        if len({slot for slot, _ in requests}) < len(requests):
            raise ValueError("a slot is given twice")
        for members in per_row.values():
            members.sort(key=lambda i: -len(requests[i][1]))
        out: list = [None] * len(requests)
        for wave in range(max((len(m) for m in per_row.values()), default=0)):
            members = {row: m[wave] for row, m in per_row.items() if wave < len(m)}  # mesh row -> request
            rounds = self.sp_rounds({row: len(requests[i][1]) for row, i in members.items()}, chunk)
            if rounds is not None:  # over the rows: together on blocks of rows, or one request at a time on all
                for owners, t, blocks in rounds:
                    self._prefill_sp_round(requests, {o: members[o] for o in owners}, t, blocks, traced, out, begin)
                continue
            slots = [None] * ROWS
            for row, i in members.items():
                slots[row] = requests[i][0] % self.users
            prompts = {row: requests[i][1] for row, i in members.items()}
            length = max(len(p) for p in prompts.values())
            chunks = self.schedule(length, chunk)
            if starts is not None:  # (the warm-up's: the wave's last chunk is the last kept, `_row_starts`)
                chunks = [(s, t) for s, t in chunks if s in starts]
            if begin:  # (a chunked prefill's: the chunks from `begin`, which must start one)
                chunks = [(s, t) for s, t in chunks if s >= begin]
                if chunks[0][0] != begin:
                    raise ValueError(
                        f"a prefill from position {begin}: not a chunk start of {self.schedule(length, chunk)}"
                    )
            self.counts["waves"] += 1
            self.counts["chunks"] += len(chunks)
            replayed = 0
            for n, (start, t) in enumerate(chunks):
                last = n == len(chunks) - 1
                key = self.model.prefill_key(start, t, last) if traced and self.prefill_traces else None
                captured = self.prefill_traces.get(key) if key is not None else None
                ids = torch.zeros(ROWS, t, dtype=torch.int64)
                nxt = torch.zeros(ROWS, t, dtype=torch.int64)  # each position's next token (the MTP layer's)
                rows, valid, ending = [False] * ROWS, [t] * ROWS, []
                for row, prompt in prompts.items():
                    k = min(t, len(prompt) - start)
                    if k <= 0:
                        continue
                    rows[row], valid[row] = True, k
                    ids[row, :k] = torch.tensor(prompt[start : start + k])
                    ahead = prompt[start + 1 : start + k + 1]
                    nxt[row, : len(ahead)] = torch.tensor(ahead, dtype=torch.int64)
                    if start + k == len(prompt):
                        ending.append(row)
                if self.drafter is not None:
                    self.model.taps = []
                more = {"greedy_ids": True} if greedy_ids else {}  # (each ending row's greedy id, [ROWS] int64)
                if captured is not None:
                    logits = self.model.replay_prefill(
                        captured,
                        ids,
                        self.states,
                        slots,
                        rows=rows,
                        valid=valid,
                        start=start,
                        logits=bool(ending),
                        next_tokens=nxt,
                        **more,
                    )
                    replayed += 1
                    if last and not captured.key.last and model_module.COMMIT in self.prefill_traces:
                        self.model.replay_commit(self.prefill_traces[model_module.COMMIT], self.states, slots)
                else:
                    logits = self.model.prefill(
                        ids,
                        start,
                        self.states,
                        slots,
                        rows=rows,
                        valid=valid,
                        logits=bool(ending),
                        last=last,
                        next_tokens=nxt,
                        **more,
                    )
                if self.drafter is not None:  # the chunk's taps -> the drafter's context, each row's own request's
                    taps, self.model.taps = self.model.taps, None
                    pos = torch.full((ROWS, t), -1, dtype=torch.int32)
                    usr = torch.zeros(ROWS, t, dtype=torch.int32)
                    for row in range(ROWS):
                        if rows[row]:
                            pos[row, : valid[row]] = torch.arange(start, start + valid[row])
                            usr[row] = slots[row]
                    self._dflash_context(taps, pos, usr, owned=captured is None)
                for row in ending:
                    out[members[row]] = int(logits[row]) if greedy_ids else logits[row]
                if self.drafter is not None and ending:
                    self._dflash_first({row: (slots[row], len(prompts[row]) - 1, logits[row]) for row in ending})
                if self.spec and ending:
                    self._first_drafts({row: (slots[row], len(prompts[row]) - 1, logits[row]) for row in ending})
            self.counts["traced_chunks"] += replayed
            self.counts["traced_waves"] += replayed == len(chunks)
        if greedy_ids:  # (the rounds over the rows give logits)
            out = [x if isinstance(x, int) else _first_id(x, self.model.config.vocab_size) for x in out]
        return out

    @staticmethod
    def sp_blocks(owners) -> dict[int, tuple]:
        """The rows each request of a round takes over the rows: the block that ends at its own row and begins after
        the previous request's (in the ring order), all 8 for a single request -> {own row: its rows in order}."""
        owners = sorted(owners)
        blocks = {}
        for i, o in enumerate(owners):
            size = (o - owners[i - 1]) % ROWS or ROWS
            blocks[o] = tuple((o - size + 1 + j) % ROWS for j in range(size))
        return blocks

    def sp_plan(self, lengths: dict[int, int], chunk: int | None = None):
        """Whether a round of requests (own row -> prompt length) goes over the rows -> (segment length, the rows of
        each, `sp_blocks`), or None for one row each.  The segment length is `chunk` if given, else the one that
        costs the least: a wave costs about the same whatever its rows hold, a little less with shorter segments
        (`SP_WAVE`: 128-token waves 0.13-0.14 s, 512-token ones 0.27 s); over the rows when that costs less
        than the one-row chunks (`schedule`), e.g. a single prompt of more than a segment, or a few requests on
        blocks of 2 or more rows."""
        best = self._sp_cost(lengths, chunk)
        return best[1:] if best is not None and best[0] < self._one_row_cost(lengths, chunk) else None

    def sp_rounds(self, lengths: dict[int, int], chunk: int | None = None):
        """The rounds over the rows of a wave of requests (own row -> prompt length) -> [(own rows, segment length,
        blocks)] in order, or None for one row each: the requests together (`sp_plan`), or one after another, each on
        all 8 rows, when that costs less (requests on neighbouring rows: the blocks end at the owners' rows, so one
        gets a block of a row; 2 x 8K 3.00 s one row each, a single 8K over the rows 0.47 s)."""
        together = self._sp_cost(lengths, chunk)
        if together is None:
            return None
        best, rounds = self._one_row_cost(lengths, chunk), None
        if together[0] < best:
            best, rounds = together[0], [(tuple(sorted(lengths)), together[1], together[2])]
        if len(lengths) > 1:
            alone = {o: self._sp_cost({o: n}, chunk) for o, n in lengths.items()}
            if sum(a[0] for a in alone.values()) < best:
                rounds = [((o,), a[1], a[2]) for o, a in sorted(alone.items())]
        return rounds

    def _sp_cost(self, lengths: dict[int, int], chunk: int | None):
        """A round over the rows (own row -> prompt length) -> (its cost in `SP_WAVE` units, the segment length,
        `sp_blocks`), or None when it cannot go over the rows."""
        if not (self.sp and self.model.sp_ready):
            return None
        sizes = [c for c in ((chunk,) if chunk else self.chunks) if c % SP_ALIGN == 0 and c <= SP_MAX]
        if not chunk and self.traced_chunks:  # a length without a captured form goes eager (~4x a wave's time)
            sizes = [c for c in sizes if c in self.traced_chunks] or sizes
        if not sizes:
            return None
        blocks = self.sp_blocks(lengths)
        ceil = lambda a, b: -(-a // b)  # noqa: E731
        waves = lambda t: max(ceil(ceil(n, t), len(blocks[o])) for o, n in lengths.items())  # noqa: E731
        t = min(sizes, key=lambda c: (waves(c) * _wave_cost(c), -c))
        return waves(t) * _wave_cost(t), t, blocks

    def _one_row_cost(self, lengths: dict[int, int], chunk: int | None) -> int:
        """One row each: the longest prompt's chunks (`schedule`), in `SP_WAVE` units."""
        return sum(_wave_cost(c) for _, c in self.schedule(max(lengths.values()), chunk))

    def _prefill_sp_round(
        self, requests, members: dict, t: int, blocks: dict, traced: bool, out: list, begin: int = 0
    ) -> None:
        """A round of requests (own row -> request) over the rows in segments of T (`sp_plan`): wave k gives each
        request its next segments, as many as its block has rows, on the block's last rows (so a request's rows in a
        later wave held its state in the one before); a request whose prompt has ended leaves its rows idle.  Each
        wave replays its captured form when there is one (`TtGlm53Model.prefill_key(.., sp=True)`), then the commit
        when a request ends in it.  The logits after each prompt go into `out`.  `begin` (`prefill_many`'s, a single
        request's): its segments from there (a multiple of T), the first wave continuing the state as a later wave."""
        if begin % t:
            raise ValueError(f"a prefill from position {begin}: not a multiple of the {t}-token segments")
        segs = {}
        for row, i in members.items():
            n = len(requests[i][1])
            segs[row] = [(s0, min(t, n - s0)) for s0 in range(begin, n, t)]
        count = max(-(-len(segs[o]) // len(blocks[o])) for o in segs)
        self.counts["waves"] += 1
        self.counts["chunks"] += count
        waves = []
        for k in range(count):
            groups, ids = [], torch.zeros(ROWS, t, dtype=torch.int64)
            for o in sorted(segs):  # (the blocks' order is the ring order from the first's first row)
                block = blocks[o]
                part = segs[o][k * len(block) : (k + 1) * len(block)]
                if not part:
                    continue
                rows = block[len(block) - len(part) :]
                tokens = requests[members[o]][1]
                for r, (s0, v) in zip(rows, part):
                    ids[r, :v] = torch.tensor(tokens[s0 : s0 + v])
                last = (k + 1) * len(block) >= len(segs[o])
                user = requests[members[o]][0] % self.users
                groups.append(SpGroup(rows, tuple(v for _, v in part), user, part[0][0], last))
            waves.append((groups, ids))
        replayed, k = 0, 0
        while k < count:
            pair = self._sp_pair(waves, k, t, traced)
            if pair is not None:  # two waves as one of 2 T tokens a row (`TtGlm53Model.prefill_sp_pair`)
                (groups_a, ids_a), (groups_b, ids_b) = waves[k], waves[k + 1]
                groups, ids, captured = groups_a + groups_b, torch.cat([ids_a, ids_b], dim=1), pair
            else:
                groups, ids = waves[k]
                start = max(g.start for g in groups)
                key = self.model.prefill_key(start, t, False, sp=True) if traced and self.prefill_traces else None
                captured = self.prefill_traces.get(key) if key is not None else None
            if self.drafter is not None:
                self.model.taps = []
            if pair is not None and captured is not False:
                logits = self.model.replay_prefill_sp_pair(captured, ids, self.states, groups_a, groups_b)
                replayed += 2
            elif pair is not None:
                logits = self.model.prefill_sp_pair(ids, self.states, groups_a, groups_b)
            elif captured is not None:
                logits = self.model.replay_prefill_sp(captured, ids, self.states, groups)
                replayed += 1
            else:
                logits = self.model.prefill_sp(ids, self.states, groups)
            traced_now = captured is not None and captured is not False
            if traced_now and any(g.last for g in groups):
                slots = [None] * ROWS
                for g in groups:
                    if g.last:
                        slots[g.owner] = g.user
                self.model.replay_commit(self.prefill_traces[model_module.COMMIT], self.states, slots)
            if self.drafter is not None:  # the wave's taps -> the drafter's context on each request's own row
                taps, self.model.taps = self.model.taps, None
                if pair is None:
                    self._dflash_context_sp(taps, groups, t, owned=not traced_now)
                else:  # each wave's rows of the taps, in its order
                    for i, wave_groups in enumerate((groups_a, groups_b)):
                        part = [_rows(x, i * t, (i + 1) * t) for x in taps]
                        self._dflash_context_sp(part, wave_groups, t, owned=True)
                    for x in taps if not traced_now else []:
                        ttnn.deallocate(x)
            for g in groups:
                if g.last:
                    out[members[g.owner]] = logits[g.owner]
            if self.drafter is not None:
                ended = {
                    g.owner: (g.user, len(requests[members[g.owner]][1]) - 1, logits[g.owner]) for g in groups if g.last
                }
                if ended:
                    self._dflash_first(ended)
            if self.spec:
                ended = {
                    g.owner: (g.user, len(requests[members[g.owner]][1]) - 1, logits[g.owner]) for g in groups if g.last
                }
                if ended:
                    self._first_drafts(ended)
            self.counts["sp_pairs"] += pair is not None
            k += 2 if pair is not None else 1
        self.counts["traced_chunks"] += replayed
        self.counts["traced_waves"] += replayed == count

    def _sp_pair(self, waves: list, k: int, t: int, traced: bool):
        """Waves k and k + 1 of a sequence-parallel round as one pair (`TtGlm53Model.prefill_sp_pair`; SP_PAIR, segments
        of SP_MAX): its captured form (`pair_key`) when the round is traced, False to run it eagerly (an eager
        round), or None (waves apart: a single wave left, or a traced round without that pair's form)."""
        if not (model_module.SP_PAIR and t == SP_MAX and k + 1 < len(waves) and self.model.sp_ready):
            return None
        if not (traced and self.prefill_traces):
            return False
        start_a, start_b = (max(g.start for g in waves[i][0]) for i in (k, k + 1))
        key = self.model.pair_key(start_a, start_b, t)
        return self.prefill_traces.get(key) if key is not None else None

    def _first_drafts(self, ended: dict) -> None:
        """After a prefill chunk: {row: (user, P - 1, its logits)} -> each such slot's first draft in `self.first`
        (the MTP layer's step at P - 1 on the greedy first token, `TtGlm53Model.mtp_first_drafts`)."""
        first = torch.full((ROWS,), -1, dtype=torch.int64)
        where = [None] * ROWS
        for row, (user, last, logits) in ended.items():  # (logits, or with greedy_ids the id)
            first[row], where[row] = _first_id(logits, self.model.config.vocab_size), (user, last)
        drafts = self.model.mtp_first_drafts(self.states, first, where, self.users, drafts=self.spec)
        for row, (user, _, _) in ended.items():
            d = drafts[row]
            self.first[row * self.users + user] = int(d) if self.spec == 1 else [int(x) for x in d]

    def verify(self, tokens: torch.Tensor, positions: torch.Tensor, keep: torch.Tensor) -> tuple:
        """MTP's verify step for every physical slot with k drafts (`spec`): tokens, positions [slots, k + 1] (each
        slot's committed token at p and its drafts at p + 1 ..; position -1: no request), keep [slots] (the slot's
        drafts accepted in its last step) -> (the greedy ids [slots, k + 1]: after the token and after each draft, the
        MTP layer's next drafts from each of those slots [slots, k + 1], with two drafts also the second level's
        [slots, k + 1]), int64 on the host: with a drafts accepted the next step's drafts are those at column a."""
        n = self.spec + 1
        active = positions[:, 0] >= 0
        u = self.users
        rows = lambda t: t.view(ROWS, u, n).transpose(1, 2).reshape(ROWS, n * u)  # noqa: E731
        tokens = rows(torch.where(active[:, None], tokens, 0))
        positions = rows(torch.where(active[:, None], positions, self.idle_pos[:, None]))
        self.idle_pos[active] = 0  # (its prompt is in: sitting out later, it writes at 0 again)
        if self.max_seq is not None and int(positions.max()) + (n > 2) >= self.max_seq:
            raise ValueError(f"position {int(positions.max())} is past the {self.max_seq}-row cache")
        keep = torch.where(active, keep, self.spec).view(ROWS, u).float()
        trace, *outs = self.spec_trace
        t0 = time.perf_counter()
        live = rows(active[:, None].repeat(1, n))  # (seats without a request choose no expert: moe_route.LiveTokens)
        self.model.write_spec_inputs(self.sio, tokens, positions, keep, live=live)
        t1 = time.perf_counter()
        ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)
        slots = lambda t: t.view(ROWS, n, u).transpose(1, 2).reshape(-1, n)  # noqa: E731
        # the outputs joined in the trace: one read (~0.5 ms of host a read)
        got = tuple(slots(t) for t in self.model.spec_outputs_to_host(outs[0], self.spec + 1, n * u))
        self.verify_seconds = (t1 - t0, time.perf_counter() - t1)
        return got

    # -- DFlash2 ------------------------------------------------------------------------------------------------------
    def _dflash_context(self, taps: list, positions: torch.Tensor, users: torch.Tensor, owned: bool = True) -> None:
        """A prefill chunk's taps (9 x [1, 1, T, 4096] a row) at positions / users [ROWS, T] (-1: not written) -> the
        drafter's context (`owned` False: a prefill trace's outputs, kept)."""
        rm = dict(dims=(0, None), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        positions = self._ring_drop(positions)
        dp, du = put(self.mesh, positions, **rm), put(self.mesh, users, **rm)
        self.drafter.write_context(taps, dp, du)
        for x in (list(taps) if owned else []) + [dp, du]:
            ttnn.deallocate(x)

    def _dflash_context_sp(self, taps: list, groups, t: int, owned: bool = True) -> None:
        """A sequence-parallel wave's taps (each row its segment's, `SpGroup`) -> the drafter's context on each
        request's own row (`TtDFlash2.write_context` gathered: row s's token i is token s T + i of the gathered)."""
        pos = torch.full((ROWS, t), -1, dtype=torch.int32)  # each row's tokens' positions (RoPE)
        owner = torch.full((ROWS,), -1, dtype=torch.int64)
        user = torch.zeros(ROWS, dtype=torch.int32)
        for g in groups:
            for i, (row, v) in enumerate(zip(g.rows, g.valid)):
                pos[row, :v] = torch.arange(g.start + i * t, g.start + i * t + v)
                owner[row], user[row] = g.owner, g.user
        flat = pos.reshape(1, ROWS * t)
        mine = owner.repeat_interleave(t)[None, :] == torch.arange(ROWS)[:, None]  # [row, gathered token]
        gpos = torch.where(mine, flat, torch.full_like(flat, -1))
        gusr = user.repeat_interleave(t)[None, :].expand(ROWS, -1).contiguous()
        rm = dict(dims=(0, None), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        gpos = self._ring_drop(gpos)
        dp, gp, gu = put(self.mesh, pos, **rm), put(self.mesh, gpos, **rm), put(self.mesh, gusr, **rm)
        self.drafter.write_context(taps, dp, None, gathered=(gp, gu))
        for x in (list(taps) if owned else []) + [dp, gp, gu]:
            ttnn.deallocate(x)

    def _ring_drop(self, positions: torch.Tensor) -> torch.Tensor:
        """A prefill call's context positions [ROWS, T] (-1: not written) with the drafter's ring (dflash2.RING = M, a
        cache longer than it): those more than M - 1 before each row's last set to -1 (they would share a slot with a
        later one of the call, written in parallel; no window reads them)."""
        m = self.drafter.ring
        if not m:
            return positions
        top = positions.max(dim=1, keepdim=True).values
        return torch.where(positions > top - m, positions, torch.full_like(positions, -1))

    def _dflash_first(self, ended: dict) -> None:
        """After a prefill chunk: {row: (user, P - 1, its logits)} -> each such slot's first K drafts in `self.first`
        (the drafter's block at P from the greedy first token; the row's other users draft at the scratch block,
        which nothing reads before it is written again)."""
        bonus = torch.zeros(ROWS, self.users, dtype=torch.int32)
        starts = torch.full((ROWS, self.users), self.scratch, dtype=torch.int32)
        for row, (user, last, logits) in ended.items():  # (logits, or with greedy_ids the id)
            bonus[row, user] = _first_id(logits, self.model.config.vocab_size)
            starts[row, user] = last + 1
        drafts = self.dfirst.run(bonus, starts)  # [ROWS, U, 8]
        for row, (user, _, _) in ended.items():
            self.first[row * self.users + user] = drafts[row, user, 1 : 1 + self.dk].tolist()

    def _dflash_inputs(self, tokens: torch.Tensor, positions: torch.Tensor, keep: torch.Tensor) -> tuple:
        """`dflash_verify`'s inputs by slot -> `DFlash2Step.write`'s (tokens, positions, keep, live) by mesh row."""
        n, u = self.dk + 1, self.users
        active = positions[:, 0] >= 0
        rows = lambda t: t.view(ROWS, u, n).transpose(1, 2).reshape(ROWS, n * u)  # noqa: E731
        tokens = rows(torch.where(active[:, None], tokens, 0))
        positions = torch.where(active[:, None], positions, self.idle_pos[:, None])
        self.idle_pos[active] = 0  # (its prompt is in: sitting out later, it writes at 0 again)
        if self.max_seq is not None:
            if int(positions[:, 0].max()) >= self.max_seq:
                raise ValueError(f"position {int(positions[:, 0].max())} is past the {self.max_seq}-row cache")
            # a request's last steps: its drafts past the cache (their outputs never read: the request ends at the
            # cache's last row) write that row, which no query before it reads (e.g. 1K + 1K at max_model_len 2048)
            positions = positions.clamp(max=self.max_seq - 1)
        positions = rows(positions)
        top = int(positions.max()) + 1 + self.drafter.block  # (the next block's end)
        if top > self.drafter.max_ctx:
            raise ValueError(f"position {int(positions.max())} is past the drafter's {self.drafter.max_ctx}-row cache")
        keep = torch.where(active, keep, self.dk).view(ROWS, u)
        return tokens, positions, keep, rows(active[:, None].repeat(1, n))

    def dflash_void(self) -> None:
        """A step launched ahead is thrown away (a prefill comes before the next verify, which then has another seat):
        KDA's state back to before it (`kda_spec_undo`: the ping-pong state's parity; nothing to wait for, the device's
        queue runs the launched step before the prefill), its results not read."""
        if self.dflash_ahead == "live":
            self._dflash_undo()
            self.dflash_ahead = "void"

    def dflash_can_undo(self) -> bool:
        """Whether a step launched ahead can be thrown away: KDA's ping-pong state (GLM53_KDA_SPEC_PINGPONG).
        GLM53_DFLASH2_AHEAD_UNSAFE=1 (a timing test's, never serving): launched without it, a step thrown away left in
        KDA's state (counted in dflash_writes["unsafe"]; the tokens after it are then wrong)."""
        if os.environ.get("GLM53_DFLASH2_AHEAD_UNSAFE") == "1":
            return True
        parity = getattr(self.model, "kda_spec_parity", None)
        return parity is not None and parity(self.states) is not None

    def _dflash_undo(self) -> None:
        if os.environ.get("GLM53_DFLASH2_AHEAD_UNSAFE") == "1" and (
            getattr(self.model, "kda_spec_parity", None) is None or self.model.kda_spec_parity(self.states) is None
        ):
            self.dflash_writes["unsafe"] = self.dflash_writes.get("unsafe", 0) + 1
            return
        if not self.dflash_can_undo():
            raise RuntimeError(
                "a step launched ahead cannot be undone: KDA's ping-pong state (GLM53_KDA_SPEC_PINGPONG)"
            )
        self.model.kda_spec_undo(self.states)
        self.dflash_writes["undone"] = self.dflash_writes.get("undone", 0) + 1

    def dflash_launch_ahead(self) -> bool:
        """The next step launched now on the inputs the last one made on the device (NEXT_INPUTS), not waited for: the
        next `dflash_verify` takes its results if its inputs are those (`dflash_next`), else throws it away (KDA's
        state back by the ping-pong's undo) and runs its own.  False when there is nothing to launch on."""
        if self.dstep is None or self.dstep.next is None or self.dflash_next is None or self.dflash_ahead is not None:
            return False
        if not self.dflash_can_undo():  # (it could not be thrown away)
            return False
        self.dstep.launch(self.states)
        self.dflash_ahead = "live"
        return True

    def dflash_sampling(self, k: torch.Tensor, p: torch.Tensor, temperature: torch.Tensor, seed: torch.Tensor) -> None:
        """Each physical slot's sampling parameters [slots] (`DFlash2Step.set_sampling`'s: k 0 no top-k, temperature 0
        or k 1 greedy, seed an int a request), for the steps from the next one (GLM53_DFLASH2_SAMPLING); a step
        launched ahead on the old ones thrown away."""
        if self.dflash_ahead == "live":
            self.dflash_void()
        self.dstep.set_sampling(*(t.view(ROWS, self.users) for t in (k, p, temperature, seed)))

    def dflash_verify(
        self, tokens: torch.Tensor, positions: torch.Tensor, keep: torch.Tensor, logits: bool = False
    ) -> tuple:
        """DFlash2's step for every physical slot (`dflash`): tokens, positions [slots, K + 1] (each slot's token at
        p and its K drafts at p + 1 ..; position -1: no request), keep [slots] (its drafts accepted in its last step;
        K for a new request) -> (accepted [slots], the bonus tokens [slots], the next drafts [slots, K]) int64: the
        slot takes its first `accepted` drafts and the bonus, its next token at p + accepted + 1, those drafts after
        it.  `warmup` compiles the step (before the prefill's captures) and captures it (after them).  `logits`
        (GLM53_DFLASH2_LOGITS): the step that also returns the target's logits, in `dflash_logits` [slots, K + 1, V]
        fp32 (None after the other step); never a step launched ahead."""
        u = self.users
        t0 = time.perf_counter()
        inputs = self._dflash_inputs(tokens, positions, keep)
        nxt = self.dstep.next
        same = (
            nxt is not None
            and self.dflash_next is not None
            and all(torch.equal(a, b) for a, b in zip(inputs, self.dflash_next))
        )
        pending, self.dflash_ahead = self.dflash_ahead, None
        if logits:  # (a step launched ahead is the other step: thrown away)
            if pending == "live":
                self.dstep.collect()
                self._dflash_undo()
            if not same or pending is not None:
                self.dstep.write(*inputs)
                self.dflash_writes["written"] += 1
            t1 = time.perf_counter()
            self.dstep.launch(self.states, logits=True)
            accepted, bonus, drafts = self.dstep.collect()
        elif pending == "live" and same:  # the step launched ahead is this one: its results
            self.dflash_writes["ahead"] = self.dflash_writes.get("ahead", 0) + 1
            t1 = time.perf_counter()
            accepted, bonus, drafts = self.dstep.collect()
        else:
            if pending == "live":  # launched ahead on other inputs (a request ended, a cut proposal): thrown away
                self.dstep.collect()
                self._dflash_undo()
            if pending is not None:  # (the device's packed inputs are now the launched step's next ones)
                same = False
            if same:
                self.dflash_writes["skipped"] += 1  # (the device made these inputs at the last step's end)
                if os.environ.get("GLM53_DFLASH2_NEXT_CHECK") == "1":  # (a test's: the device's packed row, the host's)
                    self._check_next(inputs)
            else:
                self.dstep.write(*inputs)
                self.dflash_writes["written"] += 1
            t1 = time.perf_counter()
            accepted, bonus, drafts = self.dstep.step(self.states)
        if nxt is not None:  # (the inputs the step just made for the next one)
            self.dflash_next = nxt.predict(accepted, bonus, drafts, inputs[1], inputs[3])
        self.verify_seconds = (t1 - t0, time.perf_counter() - t1)
        # the looser acceptance's count of the accepted drafts that were not the argmax [slots] (None when it is off)
        self.dflash_changed = self.dstep.changed.reshape(-1) if self.dstep.changed is not None else None
        got = self.dstep.logits
        self.dflash_logits = None if got is None else verify_logits_by_slot(got, self.dk, u)
        return accepted.reshape(-1), bonus.reshape(-1), drafts.reshape(ROWS * u, self.dk)

    def _check_next(self, inputs: tuple) -> None:
        """GLM53_DFLASH2_NEXT_CHECK: the packed input row the device made (`NextInputs.run`) against the one the host
        would write for these inputs (its pack filled, not written), every chip; counted in dflash_writes["bad"]."""
        pack = self.dstep.io.pack
        real = pack.write
        pack.write = lambda: None
        try:
            self.dstep.write(*inputs)
        finally:
            pack.write = real
        want = pack.host.to(torch.int64) & 0xFFFFFFFF
        bad = []
        for k, chip in enumerate(ttnn.get_device_tensors(pack.device)):
            got = ttnn.to_torch(chip).reshape(-1).to(torch.int64) & 0xFFFFFFFF
            row = want[k // COLS]
            if not torch.equal(got[: row.numel()], row):
                at = int((got[: row.numel()] != row).nonzero()[0])
                bad.append((k, at, int(got[at]), int(row[at])))
        if bad:
            self.dflash_writes["bad"] = self.dflash_writes.get("bad", 0) + 1
            if self.dflash_writes["bad"] <= 3:
                print(f"NEXT CHECK: the device's packed row differs on {len(bad)} chips: {bad[:4]}", flush=True)
        else:
            self.dflash_writes["checked"] = self.dflash_writes.get("checked", 0) + 1

    def decode(self, tokens: torch.Tensor, positions: torch.Tensor, greedy: bool) -> torch.Tensor:
        """One step for every physical slot: tokens, positions [slots] (position -1: no request) -> the greedy ids
        [slots] int64, or the logits [slots, V] fp32."""
        active = positions >= 0
        tokens = torch.where(active, tokens, 0).view(ROWS, self.users)
        positions = torch.where(active, positions, 0).view(ROWS, self.users)
        if self.max_seq is not None and int(positions.max()) >= self.max_seq:
            raise ValueError(f"position {int(positions.max())} is past the {self.max_seq}-row cache")
        trace, out = self.traces[greedy]
        t0 = time.perf_counter()
        self.model.write_inputs(self.io, tokens, positions, live=active.view(ROWS, self.users))
        t1 = time.perf_counter()
        ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)
        if greedy:
            got = self.model.ids_to_host(out, self.users).reshape(-1)
        else:
            got = self.model.logits_to_host(out, self.users).reshape(self.slots, -1)
        self.verify_seconds = (t1 - t0, time.perf_counter() - t1)
        return got
