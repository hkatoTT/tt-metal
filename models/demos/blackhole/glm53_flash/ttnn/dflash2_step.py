"""The DFlash2 speculative step on the device, one trace: the target's verify step of each
user's token and its K drafts (`TtGlm53Model.forward_verify`), its acceptance (`verify_accept.accept_device`), the
drafter's context write of the step's taps (`TtDFlash2.write_context`, all K + 1 positions: the rejected ones are
written over by the next block before they are read) and the next block from the accepted bonus tokens
(`TtDFlash2.draft`).  The host writes the step's inputs (the tokens, the positions, the last step's accepted counts)
and reads back the accepted counts, the bonus tokens and the next drafts: one read of one chip (the drafts made fp32
and gathered over the rows with the acceptance, as `accept_device` does, in place of a read a row)."""

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.blackhole.glm53_flash.ttnn.common import COLS, ROW_AXIS, ROWS, put, write_rows
from models.demos.blackhole.glm53_flash.ttnn.head_pick import own_on
from models.demos.blackhole.glm53_flash.ttnn.verify_accept import (
    accept_device,
    accept_fused_on,
    accept_fused_takes,
    accept_inputs,
    results_row,
)

# the drafter's context positions and bases in the step's input pack (`InputPack`, with the token ids, DSA's positions,
# KDA's keep and the live mask): one host write a step in place of three (each ~0.12-0.25 ms of host for the mesh's 32
# chips, input_pack.py).  The same values.  Read when the step is made.  0: written apart
PACK_CTX = os.environ.get("GLM53_DFLASH2_PACK_CTX", "0") == "1"

# With GLM53_DFLASH2_SAMPLING: the verify step captured both ways, with the greedy head (`forward_verify` as without
# sampling) and with the draws (`StepSampling`), a step replaying the drawing one only while some seat draws (its
# top-k not 1, `set_sampling`): a greedy-only step then costs what it does without sampling (the draws: +0.35 / +0.55 /
# +0.97 ms at 8 / 32 / 128 users) with the same ids (`TtLmHead.sample_mesh` takes the greedy rows' ids by
# `greedy_mesh`'s ops and blends them in exactly).  The logits step (GLM53_DFLASH2_LOGITS) draws as before.  Read when
# the step is made.  0: every step the drawing one (the form before)
SAMPLING_SPLIT = "GLM53_DFLASH2_SAMPLING_SPLIT"


# ... and the next step's inputs made on the device at the step's end (`next_inputs.NextInputs`: each seat continued
# from its acceptance and next drafts), the host's write skipped when the adapter's inputs are those (the generator
# compares them, `Glm53Generator.dflash_verify`).  Implies PACK_CTX.  Read when the step is made (DFlash2's serving sets
# it before, `glm53_vllm_rows.serving_defaults`).  0 (default but in DFlash2's serving): written each step
def next_inputs_on() -> bool:
    return os.environ.get("GLM53_DFLASH2_NEXT_INPUTS", "0") == "1"


def drafter_inputs(accepted, others, base1, users: int):
    """accept_device's [1, 1, 8, 2 U] fp32 row-major (every row's n, then its bonus tokens), `others` (SpecIO's: 0 at
    the chip's own row, 1e9 at the others), base1 [1, 1, 1, U] int32 row-major (each user's position p + 1) -> the
    drafter's (bonus [1, U] uint32, starts [1, U] int32 = p + 1 + n) row-major, this row's."""
    b1 = ttnn.to_layout(base1, ttnn.TILE_LAYOUT)
    base = ttnn.typecast(b1, ttnn.float32)
    ttnn.deallocate(b1)
    grid = ttnn.to_layout(accepted, ttnn.TILE_LAYOUT)
    picked = ttnn.add(grid, others)
    ttnn.deallocate(grid)
    own = ttnn.min(picked, dim=2, keepdim=True)  # [1, 1, 1, 2 U]
    ttnn.deallocate(picked)
    n = ttnn.slice(own, [0, 0, 0, 0], [1, 1, 1, users])
    b = ttnn.slice(own, [0, 0, 0, users], [1, 1, 1, 2 * users])
    ttnn.deallocate(own)
    s = ttnn.add(n, base)
    ttnn.deallocate(n)
    ttnn.deallocate(base)
    out = []
    for x, dtype in ((b, ttnn.uint32), (s, ttnn.int32)):
        c = ttnn.typecast(x, dtype)
        ttnn.deallocate(x)
        rm = ttnn.to_layout(c, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(c)
        out.append(ttnn.reshape(rm, [1, users]))
    return tuple(out)


def gathered_results(accepted, drafts, users: int):
    """accept_device's [1, 1, 8, 2 U] fp32 row-major and the drafter's [U, 8] uint32 row-major (this row's) -> [1, 1, 8,
    10 U] fp32 row-major on every chip: row r's n (U), its bonus tokens (U), its drafts rows (8 U: t[s], the drafts,
    zeros).  (With the looser acceptance's count, accept_device's 3 U: 11 U, the count after the bonus tokens.)"""
    flat = ttnn.reshape(drafts, [1, 1, 1, 8 * users])
    tiled = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
    f32 = ttnn.typecast(tiled, ttnn.float32)
    ttnn.deallocate(tiled)
    rm = ttnn.to_layout(f32, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(f32)
    if os.environ.get("GLM53_DFLASH2_SMALL_GATHER", "0") == "1":  # (read at each call; off by default)
        # one-shot over the fabric, the same bytes (small_gather.py): the ring's all_gather of these 32 U bytes took
        # ~68 us
        from models.demos.blackhole.glm53_flash.ttnn.small_gather import small_gather

        every = small_gather(drafts.device(), rm)  # [1, 1, 8, 8 U]
    else:
        every = ttnn.all_gather(rm, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8, 8 U]
    ttnn.deallocate(rm)
    both = ttnn.concat([accepted, every], dim=3)
    ttnn.deallocate(every)
    return both


def _drafts_to_host(out, users: int) -> torch.Tensor:
    """The drafter's block [U, 8] uint32 row-major a row -> [ROWS, U, 8] int64 on the host."""
    chips = ttnn.get_device_tensors(out)
    rows = [ttnn.to_torch(chips[COLS * r]).reshape(users, -1) for r in range(ROWS)]
    rows = [x.view(torch.int32).long() & 0xFFFFFFFF if x.element_size() == 4 else x.long() for x in rows]
    return torch.stack(rows)


def _logits_to_host(out, b: int, vocab: int) -> torch.Tensor:
    """The logits step's logits ([1, 1, b, V] fp32 a chip, the same on a row's chips: `TtLmHead.__call__`) -> [ROWS, b,
    V] fp32 on the host, one chip of each row read (b = (K + 1) U rows of 620 KB each: 20 MB a row at 32 seats, K = 7).
    """
    chips = ttnn.get_device_tensors(out)
    return torch.stack([ttnn.to_torch(chips[COLS * r]).reshape(b, -1)[:, :vocab].float() for r in range(ROWS)])


def verify_logits_by_slot(logits: torch.Tensor, k: int, users: int) -> torch.Tensor:
    """The logits step's [ROWS, (K + 1) U, V] (row r's virtual user j U + u: user u's position j) -> [ROWS U, K + 1, V]
    by physical slot (r U + u): each slot's distribution after its token (column 0) and after each draft."""
    rows, _, vocab = logits.shape
    return logits.view(rows, k + 1, users, vocab).transpose(1, 2).reshape(rows * users, k + 1, vocab)


def draft_to_host(drafter, bonus: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    """bonus, starts [ROWS, U] int32 (each row's users) -> the drafter's eager block [ROWS, U, 8] int64 on the host
    (t[s], the drafts, zeros)."""
    rm = dict(dims=(0, None), layout=ttnn.ROW_MAJOR_LAYOUT)
    db = put(drafter.mesh, bonus, dtype=ttnn.uint32, **rm)
    ds = put(drafter.mesh, starts, dtype=ttnn.int32, **rm)
    out = drafter.draft(db, ds)
    result = _drafts_to_host(out, bonus.shape[1])
    for x in (out, db, ds):
        ttnn.deallocate(x)
    return result


class FirstDrafts:
    """The drafter's block after a prefill (each arriving request's first drafts, `Glm53Generator._dflash_first`) at
    fixed input addresses: eager until `capture`, then replayed (eager, the drafter is ~300 programs from the host a
    prefill call: a 1K request's prefill call 0.30 s against MTP's 0.13)."""

    def __init__(self, drafter) -> None:
        self.drafter, self.mesh, self.users = drafter, drafter.mesh, drafter.users
        rm = dict(dims=(0, None), layout=ttnn.ROW_MAJOR_LAYOUT)
        zeros = torch.zeros(ROWS, self.users, dtype=torch.int32)
        self.bonus = put(self.mesh, zeros, dtype=ttnn.uint32, **rm)
        self.starts = put(self.mesh, zeros, dtype=ttnn.int32, **rm)
        self.trace = self.out = None

    def run(self, bonus: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        """bonus, starts [ROWS, U] int32 -> the block [ROWS, U, 8] int64 on the host (as `draft_to_host`)."""
        write_rows(self.mesh, bonus.to(torch.int32), self.bonus, ttnn.uint32)
        write_rows(self.mesh, starts.to(torch.int32), self.starts, ttnn.int32)
        if self.trace is None:
            out = self.drafter.draft(self.bonus, self.starts)
            result = _drafts_to_host(out, self.users)
            ttnn.deallocate(out)
            return result
        ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
        return _drafts_to_host(self.out, self.users)

    def capture(self) -> None:
        """After an eager `run` made the drafter's constants, and after the other captures."""
        assert self.trace is None
        self.trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            self.out = self.drafter.draft(self.bonus, self.starts)
        finally:
            ttnn.end_trace_capture(self.mesh, self.trace, cq_id=0)

    def release(self) -> None:
        if self.trace is not None:
            ttnn.release_trace(self.mesh, self.trace)
            ttnn.deallocate(self.out)
            self.trace = self.out = None


class DFlash2Step:
    """The step at fixed input and output addresses: `write` the inputs, then `step` (eager the first time, captured
    the second and replayed from then on).  A server's warm-up runs `compile` before its other captures (the step's
    constants: made after a capture they may sit where that trace keeps its scratch, and a replay writes over them)
    and `capture` after."""

    def __init__(self, model, drafter, users: int, k: int) -> None:
        assert drafter.users == users and 1 <= k <= drafter.slots, (drafter.users, users, k, drafter.slots)
        self.model, self.drafter, self.users, self.k = model, drafter, users, k
        mesh = self.mesh = model.mesh
        self.io = model.alloc_spec_io(users, drafts=k, mtp=False)
        n = (k + 1) * users
        rm = dict(dims=(0, None), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.ctx_pos = put(mesh, torch.full((ROWS, n), -1, dtype=torch.int32), dtype=ttnn.int32, **rm)
        self.ctx_users = put(mesh, (torch.arange(n) % users).repeat(ROWS, 1).int(), dtype=ttnn.int32, **rm)
        self.base1 = put(mesh, torch.zeros(ROWS, 1, 1, users, dtype=torch.int32), dtype=ttnn.int32, **rm)
        # GLM53_TEST_FORCE_ACCEPT=1 (tests only): each user's n pseudo-random in 0 .. K from its position and seat
        # (`verify_accept.forced_accept`), not the drafts' match: with the synthetic weights (no draft accepted) a
        # test's steps then keep 0 .. K drafts (KDA's keep, the next inputs, the drafter's context) alike in the runs
        # it compares.  The seats' numbers [1, 1, 1, U] fp32 a row
        self.forced = None
        if os.environ.get("GLM53_TEST_FORCE_ACCEPT", "0") == "1":
            seats = torch.arange(ROWS * users, dtype=torch.float32).reshape(ROWS, 1, 1, users)
            self.forced = put(mesh, seats, dims=(0, None), dtype=ttnn.float32)
        self.steps = 0
        self.compiled = False
        self.trace = None
        self.outs = None
        self.traces: dict = {}  # KDA's set read (None without the ping-pong state) -> (trace, outs)
        self._outs, self._eager = None, False
        # the looser acceptance (GLM53_ACCEPT_RELAX): accept_device also counts each user's accepted drafts that were
        # not the target's argmax (read back with the rest: `changed`, [ROWS, U] int64 after each step)
        self.relaxed = model.lm_head.relax is not None
        self.changed = None
        self.seconds: list = []  # per step (device, read): the replay to its end, the read back after it
        self.write_seconds: list = []  # per step (the verify step's inputs, the drafter's two)
        # PACK_CTX / NEXT_INPUTS: the drafter's context positions and bases in the io's input pack (made again with
        # them), and the next inputs' op
        next_inputs = next_inputs_on()
        self.packed = (PACK_CTX or next_inputs) and self.io.pack is not None
        if self.packed:
            from models.demos.blackhole.glm53_flash.ttnn.input_pack import InputPack

            old_pack = self.io.pack
            tensors = [t for t, *_ in old_pack.fields.values()] + [self.ctx_pos, self.base1]
            pack = InputPack(mesh, tensors)
            pack.host[:] = 0
            for t, offset, words, pages, slot in old_pack.fields.values():  # (the old pack's values: the live mask)
                field = old_pack.host[:, offset : offset + pages * slot].view(ROWS, pages, slot)[:, :, :words]
                pack.set(t, field.reshape(ROWS, -1))
            old_pack.deallocate()
            self.io.pack = pack
        # GLM53_DFLASH2_SAMPLING=1: the verify step draws from the target's distribution (speculative sampling,
        # `sampling.StepSampling`; the parameters per user by `set_sampling`, greedy until then)
        self.sampling = None
        if os.environ.get("GLM53_DFLASH2_SAMPLING", "0") == "1":
            from models.demos.blackhole.glm53_flash.ttnn.sampling import StepSampling

            assert not self.relaxed, "sampling with the looser acceptance"
            self.sampling = StepSampling(mesh, model.lm_head, n)
        # SAMPLING_SPLIT: the greedy step captured too, replayed while no seat draws (`drawing`, from `set_sampling`)
        self.split = self.sampling is not None and os.environ.get(SAMPLING_SPLIT, "1") == "1"
        self.drawing = False
        # GLM53_DFLASH2_LOGITS=1: also the step that returns the target's logits at every position (vllm-tt-plugin's
        # `logits` accept mode: the host's exact acceptance for what the device does not draw
        # with, penalties and the like), its own traces: `launch(states, logits=True)`, `collect` reads them into
        # `logits` [ROWS, (K + 1) U, V] fp32 (virtual user j U + u), None after the other step
        self.with_logits = os.environ.get("GLM53_DFLASH2_LOGITS", "0") == "1"
        assert not (self.with_logits and self.relaxed), "the logits step with the looser acceptance"
        self.logits = None
        self.next = None
        if next_inputs and self.packed:
            from models.demos.blackhole.glm53_flash.ttnn.next_inputs import NextInputs

            self.next = NextInputs(model, self)

    def write(self, tokens: torch.Tensor, positions: torch.Tensor, keep: torch.Tensor, live=None) -> None:
        """tokens, positions [ROWS, (K + 1) U] (virtual user j U + u: user u's token j, j = 0 its bonus at p, then its
        drafts at p + 1 ..), keep [ROWS, U] (the drafts accepted in the last step), live [ROWS, (K + 1) U] bool (the
        tokens of seats with a request; None: as last written) -> the step's input buffers."""
        u = self.users
        t0 = time.perf_counter()
        if self.packed:  # (into the pack before its one write, write_spec_inputs')
            self.io.pack.set(self.ctx_pos, positions.to(torch.int32))
            self.io.pack.set(self.base1, (positions[:, :u] + 1).to(torch.int32))
        self.model.write_spec_inputs(self.io, tokens, positions, keep, live=live)
        t1 = time.perf_counter()
        if not self.packed:
            write_rows(self.mesh, positions.to(torch.int32), self.ctx_pos, ttnn.int32)
            write_rows(self.mesh, (positions[:, :u] + 1).to(torch.int32).reshape(ROWS, 1, 1, u), self.base1, ttnn.int32)
        self.write_seconds.append((t1 - t0, time.perf_counter() - t1))

    def _run(self, states, logits: bool = False, sampled: bool | None = None):
        """The step's ops (eager, or recorded by a capture): `sampled` the drawing verify step (StepSampling), else the
        greedy head's; None: drawing with sampling (GLM53_DFLASH2_SAMPLING), the form before SAMPLING_SPLIT."""
        model, dev = self.model, self.drafter
        model.taps = []
        full = None  # (the logits step's: the target's logits)
        if sampled is None:
            sampled = self.sampling is not None
        assert self.sampling is not None or not sampled, "a drawing step needs GLM53_DFLASH2_SAMPLING=1"
        # GLM53_HEAD_PICK_OWN (with the fused acceptance, greedy): the head's pick folds each chip's own mesh row's
        # ids only (head_pick v3); the fused acceptance reads that row's span alone (verify_accept.accept_inputs)
        own = (
            not sampled
            and not logits
            and accept_fused_on()
            and own_on((self.users, self.k))
            and self.io.ids.dtype == ttnn.uint32
            and self.io.ids.layout == ttnn.ROW_MAJOR_LAYOUT
        )
        if sampled:
            seeds = self.sampling.seeds(self.ctx_pos)
            got = model.forward_verify(self.io, states, logits=logits, sampling=(self.sampling, seeds))
            (ids, full), raw = (got if logits else (got, None)), None
            ttnn.deallocate(seeds)
        elif logits:
            (ids, full), raw = model.forward_verify(self.io, states, logits=True), None
        else:
            ids, raw = model.forward_verify(self.io, states, raw=True, own=own)
        taps, model.taps = model.taps, None
        forced = None if self.forced is None else (self.base1, self.forced)
        fused = accept_fused_on() and raw is None and accept_fused_takes(ids, self.io)
        if accept_fused_on() and not getattr(self, "_fused_said", False):  # (once a step object: the path taken)
            self._fused_said = True
            print(
                f"[accept fused] {'taken' if fused else 'not taken'}: raw {raw is not None}, ids {tuple(ids.shape)} "
                f"{ids.dtype} {ids.layout}, io.ids {tuple(self.io.ids.shape)} {self.io.ids.dtype}, head pick own rows {own}",
                flush=True,
            )
        # (own and not fused: the ids' other rows undefined, accept_device would read them)
        assert fused or not own or raw is not None, "head pick own rows: the fused acceptance not taken"
        if fused:  # (GLM53_DFLASH2_ACCEPT_FUSED: the acceptance and the drafter's inputs in one op, the same values)
            # (the own row kept to the end: `results_row` gathers it with the drafter's block in one all-gather)
            accepted, bonus, starts = accept_inputs(self.mesh, ids, self.io, self.base1, forced=forced, gather=False)
        else:
            accepted = accept_device(self.mesh, ids, self.io, raw=raw, forced=forced)
        ttnn.deallocate(ids)
        if raw is not None:
            ttnn.deallocate(raw)
        dev.write_context(taps, self.ctx_pos, self.ctx_users)
        for t in taps:
            ttnn.deallocate(t)
        if not fused:
            bonus, starts = drafter_inputs(accepted, self.io.others, self.base1, self.users)
        drafts = dev.draft(bonus, starts)
        ttnn.deallocate(bonus)
        ttnn.deallocate(starts)
        if fused:
            both = results_row(self.mesh, accepted, drafts, self.users)
        else:
            both = gathered_results(accepted, drafts, self.users)
        ttnn.deallocate(accepted)
        ttnn.deallocate(drafts)
        if self.next is not None:  # the next step's inputs over the input pack (NEXT_INPUTS)
            self.next.run(both)
        return (both,) if full is None else (both, full)

    def set_sampling(self, k, p, temperature, seed) -> None:
        """Each user's sampling parameters [ROWS, U] (k: 0 no top-k, temperature 0 or k 1 greedy; seed: an int a
        request), written between steps (GLM53_DFLASH2_SAMPLING)."""
        assert self.sampling is not None, "GLM53_DFLASH2_SAMPLING=1"
        self.sampling.write(self.sampling.host_rows(k, p, temperature, seed, self.k))
        self.drawing = bool((torch.as_tensor(k) != 1).any())  # (top-k 1: the greedy ids, `StepSampling.host_rows`)

    def _variants(self) -> tuple:
        """The steps captured, as (logits, sampled): the plain one (with sampling the drawing one, and with
        SAMPLING_SPLIT the greedy one too), and the logits one with GLM53_DFLASH2_LOGITS (drawing with sampling, as
        before)."""
        drawing = self.sampling is not None
        out = [(False, False)] if (not drawing or self.split) else []
        if drawing:
            out.append((False, True))
        if self.with_logits:
            out.append((True, drawing))
        return tuple(out)

    def _sampled(self, logits: bool) -> bool:
        """The variant a step now replays: drawing with sampling, unless SAMPLING_SPLIT and no seat draws (the logits
        step always as captured)."""
        if self.sampling is None:
            return False
        return True if (logits or not self.split) else self.drawing

    def compile(self, states) -> None:
        """The step once eagerly on the written inputs (each variant), its results dropped (its programs and
        constants)."""
        for logits, sampled in self._variants():
            for x in self._run(states, logits=logits, sampled=sampled):
                ttnn.deallocate(x)
        self.compiled = True

    def _parity(self, states):
        """KDA's ping-pong state's set the next step reads (GLM53_KDA_SPEC_PINGPONG), or None without it."""
        f = getattr(self.model, "kda_spec_parity", None)
        return f(states) if f is not None else None

    def capture(self, states) -> None:
        """The step captured (the capture records it, it does not run it), after `compile`; with KDA's ping-pong state
        one trace for each set it reads, each variant (`traces`, by (set, logits, sampled)), the set back to 0 after."""
        assert self.compiled and self.trace is None
        sets = (None,) if self._parity(states) is None else (0, 1)
        self.traces = {}
        for p in sets:
            if p is not None:
                self.model.kda_spec_set_parity(states, p)
            for logits, sampled in self._variants():
                trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                try:
                    outs = self._run(states, logits=logits, sampled=sampled)
                finally:
                    ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                self.traces[(p, logits, sampled)] = (trace, outs)
        if sets != (None,):
            self.model.kda_spec_set_parity(states, 0)
        self.trace, self.outs = self.traces[(sets[0],) + self._variants()[0]]

    def release_traces(self) -> None:
        """The captured traces released and their outputs freed (a later warm-up captures again)."""
        for trace, outs in getattr(self, "traces", {}).values():
            ttnn.release_trace(self.mesh, trace)
            for x in outs:
                ttnn.deallocate(x)
        self.traces = {}
        self.trace, self.outs = None, None

    def launch(self, states, logits: bool = False) -> None:
        """The step on the written inputs (or, NEXT_INPUTS, the ones the step before made), not waited for: eager if not
        compiled yet (it compiles), else replayed (captured first if not yet; with KDA's ping-pong state the trace of
        the set it reads now, and the state told, `kda_spec_launched`; with SAMPLING_SPLIT the greedy one while no seat
        draws); `collect` reads it.  `logits`: the step that also returns the target's logits (GLM53_DFLASH2_LOGITS)."""
        assert self.with_logits or not logits, "the logits step needs GLM53_DFLASH2_LOGITS=1"
        self.steps += 1
        self._t0 = time.perf_counter()
        p = self._parity(states)
        sampled = self._sampled(logits)
        if not self.compiled:  # eager: compiles, makes the constants (before the capture)
            self._outs, self._eager = self._run(states, logits=logits, sampled=sampled), True
            self.compiled = True
        else:
            if self.trace is None:
                self.capture(states)
            trace, outs = self.traces[(p, logits, sampled)]
            ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)
            self._outs, self._eager = outs, False
        if p is not None:
            self.model.kda_spec_launched(states)

    def collect(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The launched step's results -> (n, bonus) [ROWS, U] int64 and the next drafts [ROWS, U, K] int64."""
        outs = self._outs
        ttnn.synchronize_device(self.mesh)
        t1 = time.perf_counter()
        u = self.users
        w = 3 if self.relaxed else 2  # (accept_device's: n, bonus[, the count])
        flat = ttnn.to_torch(ttnn.get_device_tensors(outs[0])[0]).reshape(ROWS, (8 + w) * u).round().to(torch.int64)
        n, bonus = flat[:, :u], flat[:, u : 2 * u]
        self.changed = flat[:, 2 * u : 3 * u] if self.relaxed else None
        nxt = flat[:, w * u :].reshape(ROWS, u, 8)[:, :, 1 : 1 + self.k]
        self.logits = (
            _logits_to_host(outs[1], (self.k + 1) * u, self.model.config.vocab_size) if len(outs) > 1 else None
        )
        self.seconds.append((t1 - self._t0, time.perf_counter() - t1))
        if self._eager:
            for x in outs:
                ttnn.deallocate(x)
        self._outs = None
        return n, bonus, nxt

    def step(self, states) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step on the written inputs (`launch`, then `collect`)."""
        self.launch(states)
        return self.collect()

    def release(self) -> None:
        if self.trace is not None:
            self.release_traces()
        for x in (self.ctx_pos, self.ctx_users, self.base1) + (() if self.forced is None else (self.forced,)):
            ttnn.deallocate(x)
