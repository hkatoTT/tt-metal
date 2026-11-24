"""The vLLM scheduler's gathering of new requests (`tools/glm53_vllm_rows.gather`: at most one new request a free
row): no device, no vLLM.

    pytest models/demos/blackhole/glm53_flash/tests/test_vllm_gather_no_device.py \
        --confcutdir=models/demos/blackhole/glm53_flash -p no:cacheprovider
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows
from models.demos.blackhole.glm53_flash.tools.glm53_vllm_rows import ADAPT_RULES, GatherClock, fits, gather

KW = dict(rows=None, decodes=20, partial=False, held_ms=None, batch=6, wait_ms=200.0)


def call(**over):
    return gather(**{**dict(waiting=3, free_seats=3), **KW, **over})


def test_off_and_nothing_to_gather():
    assert call(batch=0) == "plain"  # off: the plugin's own step
    assert call(batch=1) == "plain"
    assert call(waiting=0) == "plain"  # nothing waits
    assert call(partial=True) == "plain"  # a prefill under way (or a blocked one)
    assert call(decodes=0) == "plain"  # nothing would run meanwhile: prefill now
    assert call(free_seats=0) == "plain"  # no seat: nothing could be admitted, no hold's clock
    assert call(rows=0) == "plain"


def test_hold_until_enough_or_too_long():
    assert call() == "hold"  # 3 could be admitted, 6 wanted, no hold yet
    assert call(held_ms=150.0) == "hold"
    assert call(held_ms=200.0) == "release"  # held long enough: the prefill with what there is
    assert call(waiting=6, free_seats=6) == "release"  # 6 admissible at once
    assert call(waiting=9, free_seats=6) == "release"
    assert call(waiting=9, free_seats=5) == "hold"  # the seats bound what one prefill admits


def test_free_rows_bound_admission():
    # 8 free seats but only 4 rows with a free seat: one wave takes 4 (one a row), so the hold goes on
    assert call(waiting=8, free_seats=8, rows=4) == "hold"
    assert call(waiting=8, free_seats=8, rows=6) == "release"
    assert call(waiting=8, free_seats=8, rows=4, held_ms=250.0) == "release"


def test_rows_published():
    glm53_vllm_rows.reset()
    assert glm53_vllm_rows.free_rows() is None
    glm53_vllm_rows.publish(5)
    assert glm53_vllm_rows.free_rows() == 5
    glm53_vllm_rows.reset()


# -- GLM53_PREFILL_BATCH_ADAPT ----------------------------------------------------------------------------------
K1, K8 = 1024, 8192


def wl(*tokens):
    return [(f"r{i}", t) for i, t in enumerate(tokens)]


def test_adapt_setting(monkeypatch, expect_error):
    every = set(ADAPT_RULES)
    cases = (("", every), ("0", set()), ("1", every), ("fit", {"fit"}), (" Cost, fit ", {"fit", "cost"}))
    for raw, want in cases:
        monkeypatch.setenv("GLM53_PREFILL_BATCH_ADAPT", raw)
        assert glm53_vllm_rows.batch_adapt() == want, raw
    monkeypatch.delenv("GLM53_PREFILL_BATCH_ADAPT")
    assert glm53_vllm_rows.batch_adapt() == set(ADAPT_RULES)  # (on by default: all three)
    monkeypatch.setenv("GLM53_PREFILL_BATCH_ADAPT", "fit,soon")
    with expect_error(ValueError, "comma list"):
        glm53_vllm_rows.batch_adapt()
    monkeypatch.delenv("GLM53_PREFILL_BATCH_CALL_MS", raising=False)
    assert glm53_vllm_rows.batch_call_ms() == 100.0


def test_fits():
    assert fits(wl(K8), K8) == 1  # (an 8K prompt takes the whole step)
    assert fits(wl(K8, K8, K8), K8) == 1
    assert fits(wl(9000, 10), K8) == 1  # (a longer one: its first chunk takes the step)
    assert fits(wl(4096, 4096, 1), K8) == 2
    assert fits(wl(*[K1] * 9), K8) == 8
    assert fits(wl(K1, K1, K1), K8) is None  # (room for more than wait: the budget does not bound the batch)
    assert fits(wl(7000), K8) is None
    assert fits(wl(K8), None) is None


def test_clock_without_rules_is_gather():
    """`GatherClock` with no rule decides as `gather` with the scheduler's one hold clock (`Glm53Scheduler._admission`
    starts it on a hold, `_schedule` starts it on a hold and ends it otherwise), over random steps."""
    rng = random.Random(7)
    for trial in range(200):
        clock = GatherClock(4, 300.0)
        hold_since = None
        waiting, nxt, t = [], 0, 0.0
        for _ in range(60):
            t += rng.choice((0.0, 0.01, 0.04, 0.2))
            for _ in range(rng.choice((0, 0, 0, 1, 2))):
                waiting.append((f"q{nxt}", rng.choice((128, K1, K8))))
                nxt += 1
            free, rows, decodes = rng.randint(0, 6), rng.choice((None, 0, 1, 2, 8)), rng.choice((0, 3, 20))
            for peek in (True, False):  # (the admission's look, then the step's choice)
                held = None if hold_since is None else (t - hold_since) * 1e3
                want = gather(len(waiting), free, rows, decodes, False, held, 4, 300.0)
                if want == "hold" and hold_since is None:
                    hold_since = t
                elif want != "hold" and not peek:
                    hold_since = None
                got, left = clock.decide(t, list(waiting), free, rows, decodes, False, budget=K8, peek=peek)
                assert got == want, (trial, t, peek)
                if want == "hold" and held is not None:
                    assert abs(left - (300.0 - held)) < 1e-6
            if want == "release":
                cap = free if rows is None else min(free, rows)
                del waiting[: rng.randint(1, max(1, min(cap, len(waiting))))]


def test_fit_rule():
    clock = GatherClock(4, 1500.0, {"fit"})
    assert clock.decide(0.0, wl(K8, K8, K8), 3, 8, 29, False, budget=K8) == ("release", 0.0) and clock.why == "fit"
    assert clock.decide(0.0, wl(K1, K1, K1), 3, 8, 29, False, budget=K8)[0] == "hold"  # (room for a 4th)
    assert GatherClock(4, 1500.0).decide(0.0, wl(K8, K8, K8), 3, 8, 29, False, budget=K8)[0] == "hold"  # (off)


def test_age_rule():
    """A release that leaves a request behind (here the rows) does not start a new hold for it with "age"."""
    for rules, after in (({"age"}, "release"), (set(), "hold")):
        clock = GatherClock(4, 1500.0, rules)
        assert clock.decide(0.0, wl(K1, K1, K1), 3, 2, 20, False)[0] == "hold"
        assert clock.decide(1.6, wl(K1, K1, K1), 3, 2, 20, False) == ("release", 0.0)  # (waited 1.5 s: 2 admitted)
        assert clock.decide(1.65, wl(K1, K1, K1)[2:], 1, 1, 22, False)[0] == after, rules


def test_cost_rule():
    """The hold ends once the held requests' waits add up to the running decodes x the call (request x ms)."""
    clock = GatherClock(4, 1500.0, {"cost"}, call_ms=100.0)
    choice, left = clock.decide(0.0, wl(K1, K1, K1), 3, 8, 1, False)
    assert choice == "hold" and abs(left - 100.0 / 3) < 1e-6  # (1 decode: 100 request x ms, 3 waiting)
    assert clock.decide(0.02, wl(K1, K1, K1), 3, 8, 1, False)[0] == "hold"
    assert clock.decide(0.034, wl(K1, K1, K1), 3, 8, 1, False) == ("release", 0.0) and clock.why == "cost"
    clock = GatherClock(4, 1500.0, {"cost"}, call_ms=100.0)
    choice, left = clock.decide(0.0, wl(K1), 31, 8, 30, False)
    assert choice == "hold" and left == 1500.0  # (30 decodes: the wait binds first)


def test_peek_does_not_end_the_hold():
    clock = GatherClock(4, 300.0)
    assert clock.decide(0.0, wl(K1), 3, 8, 5, False)[0] == "hold"
    assert clock.decide(0.31, wl(K1), 3, 8, 5, False, peek=True)[0] == "release"  # (the admission's look)
    assert clock.decide(0.31, wl(K1), 3, 8, 5, False)[0] == "release"  # (the step's choice: still the same hold)
    assert clock.decide(0.32, wl(K1, K1)[1:], 3, 8, 6, False)[0] == "hold"  # (a new one)


@pytest.fixture
def stand_in(monkeypatch):
    """`Glm53Scheduler` on stand-in parts: the plugin's step admits the waiting requests in order within 8,192 tokens
    and its caps (`_capped`), a decode-only step admits none; a fake clock; ADAPT's clock with all three rules."""
    pytest.importorskip("vllm_tt_plugin.scheduler")
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm_scheduler as sched

    now = [100.0]
    steps = []
    monkeypatch.setattr(sched.time, "monotonic", lambda: now[0])
    for name, value in (("BATCH", 4), ("WAIT_MS", 1500.0), ("BATCH_AHEAD", True), ("CHUNKED", False)):
        monkeypatch.setattr(sched, name, value)
    monkeypatch.setattr(sched, "PREFIX", False)
    monkeypatch.setattr(sched, "LOG", None)

    def plugin_step(self, throttle_prefills=False):
        took, used = [], 0
        while self.waiting and len(self.running) + len(took) < self.max_num_active_reqs:
            need = self.waiting[0].num_tokens - self.waiting[0].num_computed_tokens
            if took and used + need > 8192:
                break
            took.append(self.waiting.pop(0))
            used += need
        assert len(self.running) + len(took) <= self.max_num_running_reqs
        self.running += took
        steps.append(("prefill", [r.request_id for r in took]))
        return SimpleNamespace(scheduled_new_reqs=took)

    monkeypatch.setattr(sched.TTScheduler, "schedule", plugin_step)
    s = object.__new__(sched.Glm53Scheduler)
    s._schedule_decode_only = lambda: steps.append(("decode",)) or SimpleNamespace(scheduled_new_reqs=[])
    s._finalize_scheduler_output = lambda result, is_decode: result
    s._decode_interleave = SimpleNamespace(record_step=lambda **kw: None, wants_decode_step=lambda **kw: False)
    s._forced_mode = sched.TTSchedulingMode.DEFAULT
    s._ahead, s._ahead_hold, s._hold_since, s._held_steps = False, True, None, 0
    s.max_num_running_reqs = s.max_num_active_reqs = 32
    s.num_waiting_for_streaming_input = 0
    s.max_num_scheduled_tokens = 8192
    s.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192)
    s.vllm_config = SimpleNamespace(speculative_config=None)
    s._clock = GatherClock(4, 1500.0, set(ADAPT_RULES), 100.0)
    s.running, s.waiting = [], []
    glm53_vllm_rows.reset()
    glm53_vllm_rows.publish(8)
    yield SimpleNamespace(s=s, now=now, steps=steps, sched=sched)
    glm53_vllm_rows.reset()


def requests(*tokens, start=0):
    return [SimpleNamespace(request_id=f"w{start + i}", num_tokens=t, num_computed_tokens=0, is_prefill_chunk=False)
            for i, t in enumerate(tokens)]  # fmt: skip


def decoding(n):
    return [SimpleNamespace(request_id=f"d{i}", is_prefill_chunk=False) for i in range(n)]


def test_scheduler_adapt_low_concurrency(stand_in):
    """4 concurrent: 1 decoding, 3 waiting 1K.  Off, the hold would last 1.5 s (the 4th is the one decoding); with
    "cost" it ends once the 3 have waited 100 ms between them, and the admission says a prefill comes next."""
    s, steps = stand_in.s, stand_in.steps
    s.running, s.waiting = decoding(1), requests(K1, K1, K1)
    s._schedule()
    assert steps == [("decode",)] and glm53_vllm_rows.admission() is True  # (ends within a step: nothing launched)
    stand_in.now[0] += 0.04
    s._schedule()
    assert steps[-1] == ("prefill", ["w0", "w1", "w2"])


def test_scheduler_adapt_8k(stand_in):
    """30 decoding, 3 waiting 8K: one 8K prompt fills a step, so each goes at once ("fit"); off, held 1.5 s each."""
    s, steps = stand_in.s, stand_in.steps
    s.running, s.waiting = decoding(30), requests(K8, K8)
    s._schedule()
    s._schedule()
    assert steps == [("prefill", ["w0"]), ("prefill", ["w1"])]
    s._clock = None  # (off: `gather` and one hold clock)
    s.running, s.waiting = decoding(30), requests(K8, K8, start=2)
    s._schedule()
    assert steps[-1] == ("decode",) and glm53_vllm_rows.admission() is False


def test_scheduler_adapt_left_behind(stand_in):
    """2 free rows, 3 waiting for 1.6 s: the release takes 2 (one a row), the third goes at the next step ("age")."""
    s, steps = stand_in.s, stand_in.steps
    glm53_vllm_rows.publish(2)
    s.running, s.waiting = decoding(20), requests(K1, K1, K1)
    s._schedule()
    stand_in.now[0] += 1.6
    s._schedule()
    assert steps == [("decode",), ("prefill", ["w0", "w1"])]
    glm53_vllm_rows.publish(1)
    stand_in.now[0] += 0.04
    s._schedule()
    assert steps[-1] == ("prefill", ["w2"])


def test_scheduler_step_budget(stand_in):
    s = stand_in.s
    assert s._step_budget() == 8192
    s.vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(max_num_new_slots_for_drafting=8))
    assert s._step_budget() == 8184  # (vLLM 0.31: a new request's tokens within the batched tokens less the drafts')
    s.max_num_scheduled_tokens = 4096
    assert s._step_budget() == 4096
