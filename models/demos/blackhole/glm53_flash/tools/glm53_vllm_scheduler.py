"""A TT scheduler for GLM-5.3-Flash's vLLM serving that gathers new requests into fewer prefills: the TT plugin's
scheduler (`vllm_tt_plugin.scheduler.TTScheduler`) takes a prefill-only step as soon as any request waits, so in
serving a freed seat is refilled one request at a time (measured in one serving run: 213 of 224 prefill calls with one
request, 148 ms each against 53 ms a request in a call of 8), and a GLM prefill wave costs about the same whatever its
rows hold (one request a mesh row).

    --scheduler-cls models.demos.blackhole.glm53_flash.tools.glm53_vllm_scheduler.Glm53Scheduler

GLM53_PREFILL_BATCH = m (default 0: off, the plugin's own policy unchanged): while decodes run, no prefill is pending
    already (a partial one) and from 1 to m - 1 new requests can be admitted (the waiting requests, at most the free
    seats and, when the adapter publishes them (GLM53_VLLM_SPREAD_ROWS=1, `glm53_vllm_rows`), the mesh rows with a free
    seat), the step is decode-only instead; once m can, or the hold has lasted GLM53_PREFILL_BATCH_WAIT_MS (default
    200) since its first held step, the plugin's prefill step runs, admitting at most one request a free row when the
    rows are published (the adapter then puts them on distinct rows: one wave).  A hold starts only when a request
    could be admitted (requests waiting for a seat to free do not run its clock).  `glm53_vllm_rows.gather` decides.
GLM53_VLLM_SCHED_LOG = a file: one JSON line a prefill step that ends a hold or follows waiting requests (time,
    admitted, waiting, free seats and rows, held steps and ms), to see what the holds add to the time to the first
    token (the serving benchmark's TTFT shows their sum).
GLM53_VLLM_CHUNKED=1 (the adapter's CHUNKED, vLLM's chunked prefill; on by default with DFlash2): while a prompt is
    partway (a running request with `is_prefill_chunk`), no new request is admitted, so that its next chunk comes alone
    (the generator resumes it from its rows' wave state and the DSA cache's extra user, which another prompt's prefill
    would overwrite); the plugin's decode interleave puts decode steps between its chunks.
GLM53_VLLM_AHEAD ("1" or "dry"; on by default in DFlash2's serving, `glm53_vllm_rows.serving_defaults`, applied here
    and by the adapter): each step, whether a new request could be admitted and each
    running request's end (`glm53_vllm_rows.publish_admission` / `publish_budgets`) for the adapter's choice to launch
    the next verify step ahead; while one is launched and decodes run, a decode step (a prefill would throw it away),
    unless GLM53_VLLM_AHEAD_HOLD=0.
"""

from __future__ import annotations

import json
import os
import time

from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode

from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows

BATCH = int(os.environ.get("GLM53_PREFILL_BATCH", "0"))
WAIT_MS = float(os.environ.get("GLM53_PREFILL_BATCH_WAIT_MS", "200"))
LOG = os.environ.get("GLM53_VLLM_SCHED_LOG")
# (on by default with DFlash2, as the adapter reads it: tools/glm53_vllm.py CHUNKED)
CHUNKED = os.environ.get("GLM53_VLLM_CHUNKED", "1" if os.environ.get("GLM53_DFLASH2", "0") == "1" else "0") == "1"


class Glm53Scheduler(TTScheduler):
    """`TTScheduler` with new requests' prefills gathered (GLM53_PREFILL_BATCH; off: TTScheduler as it is)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        glm53_vllm_rows.serving_defaults()  # (DFlash2's serving: GLM53_VLLM_AHEAD on unless set, as the adapter reads)
        self._ahead = os.environ.get("GLM53_VLLM_AHEAD", "0") in ("1", "dry")
        self._ahead_hold = os.environ.get("GLM53_VLLM_AHEAD_HOLD", "1") == "1"
        self._hold_since: float | None = None  # time.monotonic() of the current hold's first held step
        self._held_steps = 0
        self._first_block: dict = {}  # request id -> its first KV block (GLM53_VLLM_AHEAD's budgets' key)

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

    def schedule(self, throttle_prefills: bool = False):
        free = max(0, self.max_num_running_reqs - len(self.running))
        partway = any(r.is_prefill_chunk for r in self.running)  # (CHUNKED: a prompt's next chunk pending)
        # the adapter's step launched ahead (GLM53_VLLM_AHEAD): whether a request could be
        # admitted now (or a prompt's next chunk is pending: a prefill step comes soon), for its next choice; while one
        # is launched and decodes run, a decode step (a prefill now would throw it away), unless GLM53_VLLM_AHEAD_HOLD=0
        glm53_vllm_rows.publish_admission((len(self.waiting) > 0 and free > 0) or partway)
        if self._ahead:
            self._publish_budgets()
        decodes = sum(not r.is_prefill_chunk for r in self.running)
        behind_ahead = glm53_vllm_rows.ahead() and self._ahead_hold and decodes > 0
        if behind_ahead and self._forced_mode == TTSchedulingMode.DEFAULT:
            self._decode_interleave.record_step(is_decode=True, prefill_pending=False)
            result = self._schedule_decode_only()
            return self._finalize_scheduler_output(result, is_decode=True)
        if CHUNKED and partway:  # its next chunk alone (prefill steps), or the decodes (the plugin's interleave)
            saved = self.max_num_running_reqs
            self.max_num_running_reqs = len(self.running)  # (no seat free for a waiting request)
            try:
                return super().schedule(throttle_prefills)
            finally:
                self.max_num_running_reqs = saved
        if BATCH <= 1 or self._forced_mode != TTSchedulingMode.DEFAULT:
            return super().schedule(throttle_prefills)
        rows = glm53_vllm_rows.free_rows()
        now = time.monotonic()
        held_ms = None if self._hold_since is None else (now - self._hold_since) * 1e3
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
        saved = self.max_num_running_reqs
        if rows is not None:  # at most one new request a free row (the adapter puts them on distinct rows)
            self.max_num_running_reqs = min(saved, len(self.running) + rows)
        try:
            result = super().schedule(throttle_prefills)
        finally:
            self.max_num_running_reqs = saved
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
            with open(LOG, "a") as f:
                f.write(json.dumps(line) + "\n")
        return result
