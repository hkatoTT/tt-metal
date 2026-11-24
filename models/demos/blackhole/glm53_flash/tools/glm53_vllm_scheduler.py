"""A TT scheduler for GLM-5.3-Flash's vLLM serving that gathers new requests into fewer prefills: the TT plugin's
scheduler (`vllm_tt_plugin.scheduler.TTScheduler`) takes a prefill-only step as soon as any request waits, so in
serving a freed seat is refilled one request at a time (measured in one serving run: 213 of 224 prefill calls with one
request, 148 ms each against 53 ms a request in a call of 8), and a GLM prefill wave costs about the same whatever its
rows hold (one request a mesh row).

    --scheduler-cls models.demos.blackhole.glm53_flash.tools.glm53_vllm_scheduler.Glm53Scheduler

GLM53_PREFILL_BATCH = m (default 4 in DFlash2's serving, `glm53_vllm_rows.prefill_batch`; else 0: off, the plugin's
    own policy unchanged; 0 sets it off): while decodes run, no prefill is pending already (a partial one) and from 1
    to m - 1 new requests can be admitted (the waiting requests, at most the free seats and, when the adapter publishes
    them (GLM53_VLLM_SPREAD_ROWS=1, `glm53_vllm_rows`), the mesh rows with a free seat), the step is decode-only
    instead; once m can, or the hold has lasted GLM53_PREFILL_BATCH_WAIT_MS (default 1,500 in DFlash2's serving, else
    200) since its first held step, the plugin's prefill step runs, admitting at most one request a free row when the
    rows are published (the adapter then puts them on distinct rows: one wave).  A hold starts only when a request
    could be admitted (requests waiting for a seat to free do not run its clock).  `glm53_vllm_rows.gather` decides.
GLM53_PREFILL_BATCH_ADAPT (unset or "1": all three rules, the default; a comma list of fit, age, cost; "0": off, the
    gathering's plain form): with the gathering on (GLM53_PREFILL_BATCH > 1), `glm53_vllm_rows.GatherClock` decides in
    place of `gather` and the one hold clock: the batch is what one prefill step's token budget takes (fit: 8K prompts
    are not held), each waiting request keeps its own clock across a release that left it behind (age), and the hold
    ends once the held requests' waits add up to the running decodes x GLM53_PREFILL_BATCH_CALL_MS (cost: at low
    concurrency a hold costs more than the prefill call it saves).
GLM53_VLLM_SCHED_LOG = a file: one JSON line a prefill step that ends a hold or follows waiting requests (time,
    admitted, waiting, free seats and rows, held steps and ms), to see what the holds add to the time to the first
    token (the serving benchmark's TTFT shows their sum).
GLM53_VLLM_CHUNKED=1 (the adapter's CHUNKED, vLLM's chunked prefill; on by default with DFlash2): while a prompt is
    partway (a running request with `is_prefill_chunk`), no new request is admitted, so that its next chunk comes alone
    (the generator resumes it from its rows' wave state and the DSA cache's extra user, which another prompt's prefill
    would overwrite); the plugin's decode interleave puts decode steps between its chunks.
GLM53_PREFIX_ENTRIES ("auto" by default, 0 off; with DFlash2's chunked prefill: the adapter's PREFIX), when the adapter
    has a store: for the request at the head of the waiting queue, the longest prefix of its prompt (a multiple of 8,192
    tokens, at most its length - 1) in the adapter's prefix store whose mesh row has a free seat
    (`glm53_vllm_rows.prefix_hit`, the index and the free seats as the adapter published them): its computed tokens set
    to that length (vLLM then takes it as a request whose first tokens arrived by a KV transfer: blocks for all its
    tokens, the prefill from there), and it is admitted alone in a prefill step (the adapter resumes one request a
    call); once admitted, the hit is published by its first KV block (`glm53_vllm_rows.publish_hit`) for the adapter's
    prefill call to restore the entry.
GLM53_VLLM_AHEAD ("1" or "dry"; on by default in DFlash2's serving, `glm53_vllm_rows.serving_defaults`, applied here
    and by the adapter): each step, whether a new request could be admitted and each
    running request's end (`glm53_vllm_rows.publish_admission` / `publish_budgets`) for the adapter's choice to launch
    the next verify step ahead; while one is launched and decodes run, a decode step (a prefill would throw it away),
    unless GLM53_VLLM_AHEAD_HOLD=0.
"""

from __future__ import annotations

import itertools
import json
import os
import time

from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode

from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows
from models.demos.blackhole.glm53_flash.ttnn import prefix_keys as prefix_keys_module

BATCH, WAIT_MS = glm53_vllm_rows.prefill_batch()  # (4 and 1,500 ms in DFlash2's serving unless set, else 0 and 200)
# GLM53_PREFILL_BATCH_ADAPT's rules (all three by default; empty: off, `gather`'s plain hold) and the "cost" rule's
# call (ms)
ADAPT, CALL_MS = glm53_vllm_rows.batch_adapt(), glm53_vllm_rows.batch_call_ms()
LOG = os.environ.get("GLM53_VLLM_SCHED_LOG")
# GLM53_PREFILL_BATCH > 1 with the run-ahead (GLM53_VLLM_AHEAD, on in DFlash2's serving): while a gathering hold lasts,
# the admission told to the adapter says no prefill comes next, so that the held decode steps run ahead as any other
# (told as an admission, every held step would run on the host's critical path, about 60 ms a request at 4 a
# prefill), unless the hold ends at the next step (its wait within HOLD_STEP_MS of the end).  A hold that ends early (a
# request arrives that fills the batch) waits one decode step more behind a launched one.  GLM53_PREFILL_BATCH_AHEAD=0:
# the hold told as an admission
BATCH_AHEAD = os.environ.get("GLM53_PREFILL_BATCH_AHEAD", "1") == "1"
HOLD_STEP_MS = 60.0
# GLM53_VLLM_AHEAD_SEATS=1 (the adapter's switch, read at each step; on in DFlash2's serving unless set,
# `glm53_vllm_rows.RUN_AHEAD_VALUES`): after a prefill step, the admission for the step after it is told again: none
# when the plugin's decode interleave takes a decode step next (or no request could be admitted), so that the adapter
# may launch that verify at the prefill's end (glm53_vllm.AHEAD_SEATS)
AHEAD_SEATS = "GLM53_VLLM_AHEAD_SEATS"
# (on by default with DFlash2, as the adapter reads it: tools/glm53_vllm.py CHUNKED)
CHUNKED = os.environ.get("GLM53_VLLM_CHUNKED", "1" if os.environ.get("GLM53_DFLASH2", "0") == "1" else "0") == "1"
# the prefix store's hits (as the adapter reads it: tools/glm53_vllm.py PREFIX; on unless GLM53_PREFIX_ENTRIES=0).  Only
# what the adapter published counts: with no store (its size "auto" found no room) nothing is ever found
PREFIX = os.environ.get("GLM53_DFLASH2", "0") == "1" and CHUNKED and prefix_keys_module.entries_setting("auto") != 0


class Glm53Scheduler(TTScheduler):
    """`TTScheduler` with new requests' prefills gathered (GLM53_PREFILL_BATCH; off: TTScheduler as it is)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        glm53_vllm_rows.serving_defaults()  # (DFlash2's serving: GLM53_VLLM_AHEAD on unless set, as the adapter reads)
        self._ahead = os.environ.get("GLM53_VLLM_AHEAD", "0") in ("1", "dry")
        self._ahead_hold = os.environ.get("GLM53_VLLM_AHEAD_HOLD", "1") == "1"
        self._hold_since: float | None = None  # time.monotonic() of the current hold's first held step
        # GLM53_PREFILL_BATCH_ADAPT: the hold decided by its rules, each waiting request's clock (None: off,
        # `glm53_vllm_rows.gather` and _hold_since)
        self._clock = glm53_vllm_rows.GatherClock(BATCH, WAIT_MS, ADAPT, CALL_MS) if ADAPT and BATCH > 1 else None
        self._held_steps = 0
        self._first_block: dict = {}  # request id -> its first KV block (GLM53_VLLM_AHEAD's budgets' key)
        self._prefix_keys: tuple | None = None  # PREFIX: (request id, its prompt's prefix_keys) of the waiting head
        self._prefix_pending: tuple | None = None  # PREFIX: (request id, length, key) of a hit not admitted yet

    def _publish_budgets(self) -> None:
        """GLM53_VLLM_AHEAD: each running decode's end as `check_stop` will see it after this step (its tokens left to
        max_tokens or max_model_len, its stop token ids), keyed by its first KV block (the adapter's page table row's
        first entry), for the adapter's choice to launch the next step ahead (`glm53_vllm_rows.publish_budgets`).  Stop
        strings (the front end's) are not foreseen: such an end throws a launched step away (the adapter's undo)."""
        budgets, first = {}, {}
        for r in self.running:
            if r.is_prefill_chunk:
                continue
            key = self._first_block.get(r.request_id)
            if key is None:
                ids = self.kv_cache_manager.get_block_ids(r.request_id)
                if not ids or not ids[0]:
                    continue
                key = int(ids[0][0])
            first[r.request_id] = key
            left = min(r.max_tokens - r.num_output_tokens, self.max_model_len - r.num_tokens)
            sp = r.sampling_params
            stops = set(getattr(sp, "stop_token_ids", None) or ())
            eos = getattr(sp, "eos_token_id", None)
            if eos is not None and not getattr(sp, "ignore_eos", False):
                stops.add(int(eos))
            if key in budgets:  # (a block shared by prefix caching: the earlier end, every stop)
                left, stops = min(left, budgets[key][0]), stops | budgets[key][1]
            budgets[key] = (int(left), frozenset(stops))
        self._first_block = first
        glm53_vllm_rows.publish_budgets(budgets)

    def _admission(self, free: int, partway: bool, decodes: int) -> bool:
        """Whether a prefill may come at the next step, for the adapter's choice to launch one ahead: a request could be
        admitted (or a prompt's next chunk is pending); with a gathering hold (BATCH, BATCH_AHEAD) not while it holds,
        its clock started here (a held step behind a launched one counts), unless it ends within HOLD_STEP_MS."""
        possible = (len(self.waiting) > 0 and free > 0) or partway
        if not possible or partway or BATCH <= 1 or not BATCH_AHEAD or self._forced_mode != TTSchedulingMode.DEFAULT:
            return possible
        now = time.monotonic()
        if self._clock is not None:  # (GLM53_PREFILL_BATCH_ADAPT)
            choice, left = self._decide(now, free, decodes, bool(getattr(self, "skipped_waiting", None)), peek=True)
            return choice != "hold" or left <= HOLD_STEP_MS
        held_ms = None if self._hold_since is None else (now - self._hold_since) * 1e3
        choice = glm53_vllm_rows.gather(
            waiting=len(self.waiting),
            free_seats=free,
            rows=glm53_vllm_rows.free_rows(),
            decodes=decodes,
            partial=bool(getattr(self, "skipped_waiting", None)),
            held_ms=held_ms,
            batch=BATCH,
            wait_ms=WAIT_MS,
        )
        if choice != "hold":
            return True
        if self._hold_since is None:
            self._hold_since = now
        return held_ms is not None and held_ms + HOLD_STEP_MS >= WAIT_MS

    def schedule(self, throttle_prefills: bool = False):
        result = self._schedule(throttle_prefills)
        if self._ahead and os.environ.get(AHEAD_SEATS, "0") == "1" and getattr(result, "scheduled_new_reqs", None):
            # (a prefill step: the admission for the step after it, AHEAD_SEATS)
            free = max(0, self.max_num_running_reqs - len(self.running))
            partway = any(r.is_prefill_chunk for r in self.running)
            decodes = sum(not r.is_prefill_chunk for r in self.running)
            admit = self._admission(free, partway, decodes)
            decode_next = decodes > 0 and self._decode_interleave.wants_decode_step(
                has_pending_prefill=True, has_running_decode=True
            )
            glm53_vllm_rows.publish_admission(admit and (partway or not decode_next))
            # the new requests whose first token the plugin draws as the logits' argmax, by their first KV block
            keys = set()
            for new in result.scheduled_new_reqs:
                blocks = getattr(new, "block_ids", None)
                if blocks and blocks[0] and glm53_vllm_rows.first_is_argmax(getattr(new, "sampling_params", None)):
                    keys.add(int(blocks[0][0]))
            glm53_vllm_rows.publish_first_argmax(keys)
        return result

    def _schedule(self, throttle_prefills: bool = False):
        free = max(0, self.max_num_running_reqs - len(self.running))
        partway = any(r.is_prefill_chunk for r in self.running)  # (CHUNKED: a prompt's next chunk pending)
        decodes = sum(not r.is_prefill_chunk for r in self.running)
        # the adapter's step launched ahead (GLM53_VLLM_AHEAD): whether a request could be
        # admitted now (or a prompt's next chunk is pending: a prefill step comes soon), for its next choice; while one
        # is launched and decodes run, a decode step (a prefill now would throw it away), unless GLM53_VLLM_AHEAD_HOLD=0
        glm53_vllm_rows.publish_admission(self._admission(free, partway, decodes))
        if self._ahead:
            self._publish_budgets()
        behind_ahead = glm53_vllm_rows.ahead() and self._ahead_hold and decodes > 0
        if behind_ahead and self._forced_mode == TTSchedulingMode.DEFAULT:
            self._decode_interleave.record_step(is_decode=True, prefill_pending=False)
            result = self._schedule_decode_only()
            return self._finalize_scheduler_output(result, is_decode=True)
        if CHUNKED and partway:  # its next chunk alone (prefill steps), or the decodes (the plugin's interleave)
            return self._capped(len(self.running), throttle_prefills)  # (no seat free for a waiting request)
        if PREFIX and self._forced_mode == TTSchedulingMode.DEFAULT and self._prefix_head(free) is not None:
            return self._schedule_hit(throttle_prefills)
        if BATCH <= 1 or self._forced_mode != TTSchedulingMode.DEFAULT:
            return super().schedule(throttle_prefills)
        rows = glm53_vllm_rows.free_rows()
        now = time.monotonic()
        held_ms = None if self._hold_since is None else (now - self._hold_since) * 1e3
        if self._clock is not None:  # (GLM53_PREFILL_BATCH_ADAPT: its rules and each waiting request's clock)
            partial = any(r.is_prefill_chunk for r in self.running) or bool(getattr(self, "skipped_waiting", None))
            choice, _ = self._decide(now, free, decodes, partial)
            held_ms = self._clock.held_ms
        else:
            choice = glm53_vllm_rows.gather(
                waiting=len(self.waiting),
                free_seats=free,
                rows=rows,
                decodes=sum(not r.is_prefill_chunk for r in self.running),
                partial=any(r.is_prefill_chunk for r in self.running) or bool(getattr(self, "skipped_waiting", None)),
                held_ms=held_ms,
                batch=BATCH,
                wait_ms=WAIT_MS,
            )
        if choice == "hold":  # a decode step; the interleave policy's counters cleared (a hold is no prefill run)
            if self._hold_since is None:
                self._hold_since = now
            self._held_steps += 1
            self._decode_interleave.record_step(is_decode=True, prefill_pending=False)
            result = self._schedule_decode_only()
            return self._finalize_scheduler_output(result, is_decode=True)
        waiting, held = len(self.waiting), self._held_steps
        self._hold_since, self._held_steps = None, 0
        if choice == "plain":
            return super().schedule(throttle_prefills)
        # at most one new request a free row (the adapter puts them on distinct rows)
        result = self._capped(None if rows is None else len(self.running) + rows, throttle_prefills)
        if LOG:
            line = dict(
                t=round(time.time(), 4),
                admitted=len(getattr(result, "scheduled_new_reqs", None) or []),
                waiting=waiting,
                free_seats=free,
                free_rows=rows,
                held_steps=held,
                held_ms=round(held_ms or 0.0, 2),
            )
            if self._clock is not None:
                line["why"] = self._clock.why  # (GLM53_PREFILL_BATCH_ADAPT: batch, fit, wait or cost)
            with open(LOG, "a") as f:
                f.write(json.dumps(line) + "\n")
        return result

    def _decide(self, now: float, free: int, decodes: int, partial: bool, peek: bool = False) -> tuple:
        """GLM53_PREFILL_BATCH_ADAPT: the clock's choice for this step (`glm53_vllm_rows.GatherClock.decide`) from the
        waiting requests in the order vLLM admits them (their tokens to compute) and the step's token budget; peek:
        the admission's look, which does not end the hold.  Only the queue's head counts (the requests a seat is free
        for, and the batch + 1 for the budget)."""
        head = itertools.islice(self.waiting, max(free, BATCH) + 1)
        waiting = [(r.request_id, r.num_tokens - r.num_computed_tokens) for r in head]
        return self._clock.decide(
            now, waiting, free, glm53_vllm_rows.free_rows(), decodes, partial, budget=self._step_budget(), peek=peek
        )

    def _step_budget(self) -> int | None:
        """The prompt tokens one step can admit, as vLLM 0.31's waiting loop bounds a new request (its token budget,
        and the batched tokens less the drafting slots); None when it is not known."""
        budget = getattr(self, "max_num_scheduled_tokens", None)
        config = getattr(self, "scheduler_config", None)
        batched = getattr(config, "max_num_batched_tokens", None)
        if batched:
            spec = getattr(getattr(self, "vllm_config", None), "speculative_config", None)
            try:
                slots = int(getattr(spec, "max_num_new_slots_for_drafting", 0) or 0) if spec is not None else 0
            except (AttributeError, TypeError, ValueError):  # (a speculative config that cannot say: no bound from it)
                slots = 0
            budget = batched - slots if budget is None else min(budget, batched - slots)
        return None if budget is None else int(budget)

    def _capped(self, room, throttle_prefills: bool):
        """The plugin's step with at most `room` requests running after it (None: as configured).  vLLM 0.31 admits a
        waiting request against max_num_active_reqs (fixed at its start, running + paused streaming sessions) and the
        plugin then asserts the running ones against max_num_running_reqs, so both caps come down together: lowering
        only the second still admits as many and stops the engine at that assert (seen with GLM53_PREFILL_BATCH)."""
        if room is None:
            return super().schedule(throttle_prefills)
        saved = (self.max_num_running_reqs, getattr(self, "max_num_active_reqs", None))
        self.max_num_running_reqs = min(saved[0], room)
        if saved[1] is not None:
            self.max_num_active_reqs = min(saved[1], room + getattr(self, "num_waiting_for_streaming_input", 0))
        try:
            return super().schedule(throttle_prefills)
        finally:
            self.max_num_running_reqs = saved[0]
            if saved[1] is not None:
                self.max_num_active_reqs = saved[1]

    # -- the prefix store's hits (PREFIX) -----------------------------------------------------------------------------
    def _waiting_head(self):
        """The request the waiting loop takes first (vLLM's: those holding KV blocks before the others), or None."""
        for queue in (getattr(self, "kv_holding_waiting", None), self.waiting):
            if queue:
                return queue.peek_request()
        return None

    def _prefix_head(self, free: int):
        """PREFIX: the hit to admit, (request id, length, key): a waiting one found at an earlier step, else the
        waiting head's, set on the request (its computed tokens) when it is found; None when there is none (or nothing
        waits, or no seat is free)."""
        from vllm.v1.request import RequestStatus

        from models.demos.blackhole.glm53_flash.ttnn.prefix_keys import prefix_keys

        pending = self._prefix_pending
        if pending is not None:  # found at an earlier step, not admitted yet: it still goes alone when admitted
            request = self.requests.get(pending[0])
            if request is not None and request.status == RequestStatus.WAITING:
                return pending
            self._prefix_pending = None
        head = self._waiting_head()
        if head is None or free <= 0:
            return None
        if head.num_computed_tokens or head.status != RequestStatus.WAITING or getattr(head, "num_preemptions", 0):
            return None
        if getattr(head, "skip_reading_prefix_cache", False) or head.prompt_token_ids is None:
            return None
        state = glm53_vllm_rows.prefix_state()
        if state is None or not state[0]:
            return None
        if self._prefix_keys is None or self._prefix_keys[0] != head.request_id:
            self._prefix_keys = (head.request_id, prefix_keys(head.prompt_token_ids))
        got = glm53_vllm_rows.prefix_hit(self._prefix_keys[1], head.num_prompt_tokens, *state)
        if got is None:
            return None
        length, key, _ = got
        head.num_computed_tokens = length
        self._prefix_pending = (head.request_id, length, key)
        return self._prefix_pending

    def _schedule_hit(self, throttle_prefills: bool):
        """The plugin's step with at most one request admitted (the waiting head, a hit); once it is admitted its hit
        is published by its first KV block for the adapter."""
        from vllm.v1.request import RequestStatus

        req_id, length, key = self._prefix_pending
        result = self._capped(len(self.running) + 1, throttle_prefills)
        request = self.requests.get(req_id)
        if request is not None and request.status == RequestStatus.RUNNING:
            ids = self.kv_cache_manager.get_block_ids(req_id)
            if ids and ids[0]:
                glm53_vllm_rows.publish_hit(int(ids[0][0]), length, key)
            self._prefix_pending = None
            if LOG:
                with open(LOG, "a") as f:
                    f.write(json.dumps(dict(t=round(time.time(), 4), prefix_hit=length, request=str(req_id))) + "\n")
        return result
