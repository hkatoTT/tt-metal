"""What the GLM-5.3-Flash vLLM adapter (`glm53_vllm.py`) tells its scheduler (`glm53_vllm_scheduler.py`): the mesh
rows with a free seat now, which bound how many new requests one prefill wave takes (one a row).  Both run in vLLM's
engine-core process (the TT plugin's "uni" executor), so a module global carries it; no
vLLM or ttnn import here.
"""

from __future__ import annotations

import os

_free_rows: int | None = None  # None: not published (the scheduler then counts free seats only)


def publish(free_rows: int) -> None:
    global _free_rows
    _free_rows = int(free_rows)


def free_rows() -> int | None:
    return _free_rows


def reset() -> None:
    global _free_rows, _ahead, _admission, _budgets, _prefix, _first_argmax
    _free_rows = None
    _ahead = False
    _admission = None
    _budgets = None
    _prefix = None
    _first_argmax = None
    _hits.clear()


# The serving step launched ahead (glm53_vllm.py, GLM53_VLLM_AHEAD): the adapter says when it
# launched one (`set_ahead`), so that the scheduler takes a decode step next (a prefill would throw it away); the
# scheduler says at each step whether a new request could be admitted (`publish_admission`: requests wait and a seat
# is free), so that the adapter does not launch one before a prefill.
_ahead = False
_admission: bool | None = None  # None: not published


def set_ahead(on: bool) -> None:
    global _ahead
    _ahead = bool(on)


def ahead() -> bool:
    return _ahead


def publish_admission(possible: bool) -> None:
    global _admission
    _admission = bool(possible)


def admission() -> bool | None:
    return _admission


# Each running request's end as the scheduler knows it before a step (`publish_budgets`), so that the adapter does not
# launch the next step ahead when one ends at this step (its seat is then empty in the next one: the launched step
# would be thrown away).  key (the request's first KV block: its page table row's first entry; a test without page
# tables keys by the batch row) -> (its tokens left: it ends once a step appends this many, its stop token ids).
_budgets: dict | None = None  # None: not published (no end is foreseen)


def publish_budgets(budgets: dict | None) -> None:
    global _budgets
    _budgets = budgets


def budgets() -> dict | None:
    return _budgets


# The new requests of a prefill step whose first token the plugin draws on the host as the argmax of the logits the
# adapter returns (greedy, no penalty, grammar, token filter or min_tokens: `first_is_argmax`), keyed as the budgets
# (their first KV block; a test without page tables: the vLLM slot), published by the scheduler after the step
# (`publish_first_argmax`), so that the adapter knows those requests' first ids at the prefill's end and may launch the
# next verify there (glm53_vllm.AHEAD_SEATS with sample_on_device_mode "decode_only").
_first_argmax: set | None = None  # None: not published (no first token foreseen)


def first_is_argmax(params) -> bool:
    """Whether a request's first token, drawn on the host from the prefill's logits, is their argmax: a temperature
    under vLLM's SAMPLING_EPS and nothing that changes the logits before the pick (vLLM's SamplingParams)."""
    if params is None or float(getattr(params, "temperature", 0.0) or 0.0) >= 1e-5:
        return False
    for name, neutral in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
        value = getattr(params, name, None)
        if value is not None and float(value) != neutral:
            return False
    if getattr(params, "structured_outputs", None) is not None or int(getattr(params, "min_tokens", 0) or 0) > 0:
        return False
    return not any(getattr(params, name, None) for name in ("allowed_token_ids", "bad_words", "logit_bias"))


def publish_first_argmax(keys) -> None:
    global _first_argmax
    _first_argmax = None if keys is None else {int(k) for k in keys}


def first_argmax() -> set | None:
    return _first_argmax


def gather(
    waiting: int,
    free_seats: int,
    rows: int | None,
    decodes: int,
    partial: bool,
    held_ms: float | None,
    batch: int,
    wait_ms: float,
) -> str:
    """The scheduler's choice for a step (`glm53_vllm_scheduler.Glm53Scheduler`): "plain" (the plugin's own step, no
    hold: gathering off, nothing waits, a prefill under way (`partial`, or one blocked), nothing runs meanwhile, or no
    request could be admitted), "hold" (a decode step, gathering: fewer than `batch` new requests could be admitted --
    the waiting ones, at most the free seats and the free rows when known -- and the hold, `held_ms` since its first
    held step (None: none yet), under `wait_ms`), else "release" (the prefill step)."""
    if batch <= 1 or waiting <= 0 or partial or decodes <= 0:
        return "plain"
    cap = free_seats if rows is None else min(free_seats, rows)
    admissible = min(waiting, cap)
    if admissible <= 0:
        return "plain"
    if admissible < batch and (held_ms is None or held_ms < wait_ms):
        return "hold"
    return "release"


# The serving run-ahead's default: DFlash2's serving (GLM53_DFLASH2=1) runs with these on (each "1" unless the variable
# is set), the scheduler being `glm53_vllm_scheduler.Glm53Scheduler` (its admission, budgets and hold).
# GLM53_VLLM_RUN_AHEAD=0: all of them at their own defaults (off).  MTP's serving is not touched (KDA's ping-pong state
# has no MTP step).
RUN_AHEAD_SWITCHES = ("GLM53_DFLASH2_NEXT_INPUTS", "GLM53_VLLM_VECTOR", "GLM53_VLLM_AHEAD", "GLM53_KDA_SPEC_PINGPONG")
# ... and with them: a step also launched ahead where requests end and at a prefill's end (GLM53_VLLM_AHEAD_SEATS),
# launches stopped only where vLLM cuts the drafts (GLM53_VLLM_AHEAD_EDGE=vllm), each as given here unless the variable
# is set
RUN_AHEAD_VALUES = {"GLM53_VLLM_AHEAD_SEATS": "1", "GLM53_VLLM_AHEAD_EDGE": "vllm"}
# DFlash2's serving prefill calls: the captured wave's commit and the drafter's context write issued before the
# logits / ids are read back (generator.CTX_EARLY), and no first drafts at a prefill's end (generator.FIRST_DRAFTS: the
# plugin never proposes after a prefill step), each as given here unless the variable is set; the same values (a call
# 16-23 ms shorter).  ... and the interleaved sequence-parallel blocks (generator.SP_INTERLEAVE): a call of 3-4
# requests on rows next to each other one SP wave of 512 tokens a row in place of the one-row form's wave of 1,024
# (which form a request is prefilled in changes its rounding, as the rows it lands on do).  Not tied to
# GLM53_VLLM_RUN_AHEAD
SERVING_PREFILL_VALUES = {"GLM53_PREFILL_CTX_EARLY": "1", "GLM53_DFLASH2_FIRST_DRAFTS": "0", "GLM53_SP_INTERLEAVE": "1"}


def prefill_batch() -> tuple:
    """The scheduler's gathering of new requests (GLM53_PREFILL_BATCH, its wait GLM53_PREFILL_BATCH_WAIT_MS; read by
    `glm53_vllm_scheduler` and by the adapter's SPREAD_ROWS at import): unset, 4 requests and 1,500 ms in DFlash2's
    serving (with the two above, 1K / 1K at 32 seats 2,276 -> 2,676 tok/s on random ids and 1,855 -> 1,966 on
    wikitext, the median time to the first token 0.22 -> 0.94 / 1.07 s), else 0 (off) and 200 ms.
    GLM53_PREFILL_BATCH=0: the plugin's own policy (no gathering)."""
    dflash = os.environ.get("GLM53_DFLASH2", "0") == "1"
    batch = int(os.environ.get("GLM53_PREFILL_BATCH", "4" if dflash else "0") or 0)
    return batch, float(os.environ.get("GLM53_PREFILL_BATCH_WAIT_MS", "1500" if dflash else "200"))


# GLM53_PREFILL_BATCH_ADAPT (on by default; "0": off, the gathering's plain form: `gather` and one hold clock): rules
# that end the gathering's hold where it cannot pay off, the scheduler deciding with `GatherClock`.
#   fit:  the batch is what one prefill step can take: the waiting requests, in the order vLLM admits them, whose
#         tokens fit the step's token budget (vLLM's, 8,192 in DFlash2's chunked serving), at least one.  An 8K prompt
#         fills a step alone, so 8K prompts are not held (without "fit" each is held 1.5 s, the release admitting one
#         and the next one held again: at 8 concurrent 8K requests, a median time to the first token of about 3 s).
#   age:  a waiting request's clock runs from when it first could be admitted (a seat free for it) until it is, not
#         from the hold's first step: a release that leaves some behind (one request a free row, the budget) starts no
#         new hold for them.
#   cost: the hold also ends once what it has cost the held requests (the sum of their waits, request x ms) reaches
#         what one more prefill call costs the running decodes (decodes x GLM53_PREFILL_BATCH_CALL_MS, default 100: a
#         1K request's call).  With few decodes a hold costs more than the call it saves (at 4 concurrent requests the
#         4th request is the one decoding, so the hold never fills and every request waited the whole 1.5 s); the
#         rent-or-buy rule costs at most twice what the better choice would, made knowing when requests arrive.
# Unset or "1": all three; else a comma list of them.  Nothing computes otherwise; what changes is when a prefill call
# runs and which waiting requests it takes, at most one a free row as without the rules (which requests share a call
# changes the form they are prefilled in and so their rounding, as their arrival times do).
ADAPT_RULES = ("fit", "age", "cost")


def batch_adapt() -> frozenset:
    """GLM53_PREFILL_BATCH_ADAPT's rules: unset (or empty) and "1" all three, the default; "0" none (an empty set: off,
    the gathering's plain form)."""
    raw = os.environ.get("GLM53_PREFILL_BATCH_ADAPT", "1").strip().lower()
    if raw == "0":
        return frozenset()
    if raw in ("", "1"):
        return frozenset(ADAPT_RULES)
    rules = frozenset(r.strip() for r in raw.split(",") if r.strip())
    unknown = sorted(rules - set(ADAPT_RULES))
    if unknown:
        names = ", ".join(ADAPT_RULES)
        raise ValueError(f"GLM53_PREFILL_BATCH_ADAPT={raw!r}: '1' or a comma list of {names} ({unknown})")
    return rules


def batch_call_ms() -> float:
    """GLM53_PREFILL_BATCH_CALL_MS: what one more prefill call costs each running decode (the "cost" rule), ms."""
    return float(os.environ.get("GLM53_PREFILL_BATCH_CALL_MS", "100") or 100)


def fits(waiting: list, budget: int | None) -> int | None:
    """How many of the waiting requests [(request id, its tokens to compute)], from the head, one prefill step takes
    when its token budget is what stops it (the next one does not fit, or nothing is left for another), at least the
    head (a longer one takes the step's whole budget); None when the budget would take more than are waiting."""
    if budget is None:
        return None
    n = used = 0
    for _, tokens in waiting:
        used += max(1, int(tokens))
        if n and used > budget:
            return n
        n += 1
        if used >= budget:
            return n
    return None


class GatherClock:
    """The gathering's hold with GLM53_PREFILL_BATCH_ADAPT's rules: `glm53_vllm_scheduler.Glm53Scheduler` asks it in
    place of `gather` and its one hold clock when a rule is on.  Its
    state: when each waiting request first could be admitted (`since`); without "age" forgotten at each release, so that
    the empty rule set decides as `gather` with the scheduler's clock did."""

    def __init__(self, batch: int, wait_ms: float, rules=frozenset(), call_ms: float = 100.0):
        self.batch, self.wait_ms = int(batch), float(wait_ms)
        self.rules, self.call_ms = frozenset(rules), float(call_ms)
        self.since: dict = {}  # request id -> the time (s) it first could be admitted while it waited
        self.held_ms = 0.0  # the last decision's longest wait among the requests a seat is free for (for a log)
        self.why = ""  # the last decision's reason: plain / batch / fit / wait / cost / hold

    def _done(self, choice: str, why: str, peek: bool) -> tuple:
        if "age" not in self.rules and not peek:
            self.since.clear()
        self.why = why
        return choice, 0.0

    def decide(
        self,
        now: float,
        waiting: list,
        free_seats: int,
        rows: int | None,
        decodes: int,
        partial: bool,
        budget: int | None = None,
        peek: bool = False,
    ) -> tuple:
        """One step's choice -> (choice, ms_left).  waiting: [(request id, its tokens to compute)] in the order vLLM
        admits them (the queue's head is enough: at least the free seats and the batch + 1 when there are as many);
        now: time.monotonic() (s); budget: the step's token budget ("fit"; None: no bound); peek: a look that does not
        end the hold (the scheduler's admission for the run-ahead, before the step's own choice: as its clock was reset
        only by `_schedule`).  choice as `gather`'s: "plain", "hold" (a decode step) or "release" (the prefill step);
        ms_left: with "hold", how long the hold lasts if no request arrives or ends (else 0)."""
        if self.since:
            ids = {rid for rid, _ in waiting}
            for rid in [r for r in self.since if r not in ids]:  # admitted (or aborted)
                del self.since[rid]
        self.held_ms = 0.0
        n = len(waiting)
        if self.batch <= 1 or n <= 0 or partial or decodes <= 0:
            return self._done("plain", "plain", peek)
        seated = min(n, free_seats)  # (the requests a seat is free for: their clocks run)
        admissible = seated if rows is None else min(seated, rows)  # (one prefill call: at most one a free row)
        if admissible <= 0:
            return self._done("plain", "plain", peek)
        target = self.batch
        if "fit" in self.rules:
            full = fits(waiting[: target + 1], budget)
            if full is not None:
                target = min(target, full)
        for rid, _ in waiting[:seated]:
            self.since.setdefault(rid, now)
        ages = [(now - self.since[rid]) * 1e3 for rid, _ in waiting[:seated]]
        self.held_ms = max(ages)
        if admissible >= target:
            return self._done("release", "batch" if target == self.batch else "fit", peek)
        left = self.wait_ms - self.held_ms
        why = "wait"
        if "cost" in self.rules:
            spare = (decodes * self.call_ms - sum(ages)) / seated  # (the waits grow `seated` request x ms a ms)
            if spare < left:
                left, why = spare, "cost"
        if left <= 0:
            return self._done("release", why, peek)
        self.why = "hold"
        return "hold", left


def serving_defaults() -> bool:
    """The run-ahead's switches set to their serving default where unset (RUN_AHEAD_SWITCHES, RUN_AHEAD_VALUES); called
    by the adapter before it makes the model (`Glm53ForCausalLM.initialize_vllm_model`) and by the scheduler
    (`Glm53Scheduler`), both in the engine-core process, so that they read the same.  Whether the run-ahead's defaults
    applied.  DFlash2's serving prefill switches (SERVING_PREFILL_VALUES) are set where unset whatever
    GLM53_VLLM_RUN_AHEAD says."""
    if os.environ.get("GLM53_DFLASH2", "0") != "1":
        return False
    for var, value in SERVING_PREFILL_VALUES.items():
        os.environ.setdefault(var, value)
    if os.environ.get("GLM53_VLLM_RUN_AHEAD", "1") != "1":
        return False
    for var in RUN_AHEAD_SWITCHES:
        os.environ.setdefault(var, "1")
    for var, value in RUN_AHEAD_VALUES.items():
        os.environ.setdefault(var, value)
    return True


# The prefix store (GLM53_PREFIX_ENTRIES, the adapter's PREFIX): the adapter publishes the store's index (each entry's
# key -> its prefix's length and mesh row) and each mesh row's free seats (`publish_prefix`); the scheduler, for the
# request at the head of the waiting queue, finds the longest prefix in the index whose row has a free seat
# (`prefix_hit`), sets the request's computed tokens to its length, and once the request is admitted publishes the hit
# by the request's first KV block (`publish_hit`); the adapter's prefill call of that request (start_pos = the length)
# takes it (`take_hit`: the page table's first entry) and restores the entry.
_prefix: tuple | None = None  # (index {key: (length, row)}, free seats a mesh row [ROWS]); None: not published
_hits: dict = {}  # first KV block -> (length, key)


def publish_prefix(index: dict, free: list) -> None:
    global _prefix
    _prefix = (dict(index), list(free))


def prefix_state() -> tuple | None:
    return _prefix


def prefix_hit(keys: list, prompt_len: int, index: dict, free: list, grid: int | None = None) -> tuple | None:
    """A prompt's longest prefix in the store's index (keys: its `prefix_keys`), of at most prompt_len - 1 tokens (one
    token at least is computed), whose entry's mesh row has a free seat -> (length, key, row), or None."""
    from models.demos.blackhole.glm53_flash.ttnn.prefix_keys import GRID

    grid = grid or GRID
    for j in range(min(len(keys), (prompt_len - 1) // grid) - 1, -1, -1):
        got = index.get(keys[j])
        if got is not None and got[0] == (j + 1) * grid and 0 <= got[1] < len(free) and free[got[1]] > 0:
            return got[0], keys[j], got[1]
    return None


def publish_hit(first_block: int, length: int, key: str) -> None:
    _hits[int(first_block)] = (int(length), key)


def take_hit(first_block: int | None) -> tuple | None:
    """The hit the scheduler published for the request whose first KV block this is -> (length, key), or None."""
    if first_block is None:
        return None
    return _hits.pop(int(first_block), None)
