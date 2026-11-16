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
    global _free_rows, _ahead, _admission, _budgets
    _free_rows = None
    _ahead = False
    _admission = None
    _budgets = None


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


def serving_defaults() -> bool:
    """The run-ahead's switches set to their serving default where unset (RUN_AHEAD_SWITCHES); called by the adapter
    before it makes the model (`Glm53ForCausalLM.initialize_vllm_model`) and by the scheduler (`Glm53Scheduler`), both
    in the engine-core process, so that they read the same.  Whether the defaults applied."""
    if os.environ.get("GLM53_DFLASH2", "0") != "1" or os.environ.get("GLM53_VLLM_RUN_AHEAD", "1") != "1":
        return False
    for var in RUN_AHEAD_SWITCHES:
        os.environ.setdefault(var, "1")
    return True
