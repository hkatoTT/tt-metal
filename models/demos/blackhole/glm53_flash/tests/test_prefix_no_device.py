"""The prefix store's host side: its keys (`ttnn/prefix_keys.py`) and setting, the store's
size from the free DRAM (`ttnn/prefix_store.plan_entries`), the scheduler's choice of a hit
(`tools/glm53_vllm_rows.prefix_hit`) and the adapter's prefill calls with the store on (tools/glm53_vllm.py PREFIX:
`_prefix_call`, `_prefix_seat`, the index it publishes) and off (a partway prompt's next chunk on its own seat,
`_resume`) on a stand-in generator: no device, no vLLM.

    pytest models/demos/blackhole/glm53_flash/tests/test_prefix_no_device.py \
        --confcutdir=models/demos/blackhole/glm53_flash -p no:cacheprovider
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.prefix_keys import AUTO, GRID, entries_setting, prefix_keys  # noqa: E402

ROWS, USERS = 8, 4


def test_prefix_keys():
    a = list(range(3 * GRID + 5))
    b = a[: 2 * GRID] + [7] * (GRID + 5)
    ka, kb = prefix_keys(a), prefix_keys(b)
    assert len(ka) == 3 and len(prefix_keys(a[: GRID - 1])) == 0
    assert ka[:2] == kb[:2] and ka[2] != kb[2]  # (chained: equal while the blocks so far are equal)
    assert prefix_keys(a[: 2 * GRID]) == ka[:2]
    assert prefix_keys(torch.tensor(a)) == ka  # (a tensor of ids as a list)
    assert prefix_keys(a, salt="x")[0] != ka[0]


def test_entries_setting(monkeypatch, expect_error):
    monkeypatch.delenv("GLM53_PREFIX_ENTRIES", raising=False)
    assert entries_setting("auto") == AUTO and entries_setting("0") == 0
    for raw, want in (("auto", AUTO), (" AUTO ", AUTO), ("0", 0), ("16", 16), ("", AUTO)):
        monkeypatch.setenv("GLM53_PREFIX_ENTRIES", raw)
        assert entries_setting("auto") == want, raw
    for raw in ("-1", "lots"):
        monkeypatch.setenv("GLM53_PREFIX_ENTRIES", raw)
        with expect_error(ValueError, "'auto' or a number"):
            entries_setting("auto")


def test_plan_entries(monkeypatch):
    """The store's size from the free DRAM: the free less the reserve, at most a fraction of it and the largest block,
    at most two a seat; none (off, with the reason) when not one entry fits."""
    from models.demos.blackhole.glm53_flash.ttnn import prefix_store as ps

    monkeypatch.setattr(ps, "RESERVE_GIB", 4.0)
    monkeypatch.setattr(ps, "DRAM_FRACTION", 0.5)
    monkeypatch.setattr(ps, "MAX_ENTRIES", None)
    gib, mib = 2**30, 2**20
    n, why = ps.plan_entries(120 * mib, 15 * gib, 15 * gib, 32)  # (36K: two a seat)
    assert n == 64 and why.startswith("prefix store: 64 entries")
    n, _ = ps.plan_entries(265 * mib, 14 * gib, 14 * gib, 32)  # (128K: half of the free, 7 GiB)
    assert n == int(7 * gib // (265 * mib)) == 27
    n, _ = ps.plan_entries(1710 * mib, 9.5 * gib, 9.5 * gib, 32)  # (1M: 4.75 GiB)
    assert n == 2
    n, _ = ps.plan_entries(1710 * mib, 9.5 * gib, 2 * gib, 32)  # (the largest free block)
    assert n == 1
    n, why = ps.plan_entries(1710 * mib, 3.3 * gib, 3.3 * gib, 64)  # (less than the reserve)
    assert n == 0 and why.startswith("prefix store off: not one entry fits")
    monkeypatch.setattr(ps, "MAX_ENTRIES", "5")
    assert ps.plan_entries(120 * mib, 15 * gib, 15 * gib, 32)[0] == 5


def test_prefix_hit():
    keys = prefix_keys(list(range(3 * GRID + 1)))
    free = [1] * ROWS
    index = {keys[0]: (GRID, 2), keys[1]: (2 * GRID, 5)}
    assert glm53_vllm_rows.prefix_hit(keys, 3 * GRID + 1, index, free) == (2 * GRID, keys[1], 5)  # the longest
    assert glm53_vllm_rows.prefix_hit(keys, 2 * GRID, index, free) == (GRID, keys[0], 2)  # at most length - 1
    free[5] = 0
    assert glm53_vllm_rows.prefix_hit(keys, 3 * GRID + 1, index, free) == (GRID, keys[0], 2)  # its row full
    free[2] = 0
    assert glm53_vllm_rows.prefix_hit(keys, 3 * GRID + 1, index, free) is None
    assert glm53_vllm_rows.prefix_hit(keys, 3 * GRID + 1, {keys[1]: (GRID, 1)}, [1] * ROWS) is None  # (a stale key)


class FakeStore:
    def __init__(self, n: int):
        self.entries = [SimpleNamespace(length=0, row=-1, key=None) for _ in range(n)]


class FakeGenerator:
    """Records the prefill calls and the store's snapshots / restores; the seats' state is (prompt so far) only."""

    def __init__(self):
        self.slots = ROWS * USERS
        self.chunks = (128, 512, 1024)
        self.chunk = 1024
        self.traces = {}
        self.dflash_ahead = None
        self.prefix = FakeStore(3)
        self.prefill_end = None
        self.calls, self.snaps, self.restores = [], [], []
        self.first = {}

    def prefill_many(self, requests, chunk=None, greedy_ids=False, begin=0):
        self.calls.append([(slot, len(tokens), begin) for slot, tokens in requests])
        self.prefill_end = (requests[0][0], len(requests[0][1])) if len(requests) == 1 else None
        return [torch.zeros(5) for _ in requests]

    def prefix_snapshot(self, slot, length, keys, entry=None):
        assert self.prefill_end == (slot, length)
        if entry is None:
            entry = next(i for i, e in enumerate(self.prefix.entries) if not e.length)
        e = self.prefix.entries[entry]
        e.length, e.row, e.key = length, slot // USERS, keys[length // GRID - 1]
        self.snaps.append((slot, length, entry))
        return entry

    def prefix_restore(self, entry, slot):
        e = self.prefix.entries[entry]
        assert slot // USERS == e.row, (slot, e.row)
        self.restores.append((entry, slot))
        self.prefill_end = (slot, e.length)
        return e.length


@pytest.fixture
def adapter(monkeypatch):
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm as adapter_module

    monkeypatch.setattr(adapter_module, "PREFIX", True)
    monkeypatch.setattr(adapter_module, "CHUNKED", True)
    monkeypatch.setattr(adapter_module, "SPREAD_ROWS", False)
    glm53_vllm_rows.reset()
    a = adapter_module.Glm53ForCausalLM(FakeGenerator(), 4 * GRID)
    assert a.prefix_on and glm53_vllm_rows.prefix_state() == ({}, [USERS] * ROWS)
    yield a
    glm53_vllm_rows.reset()


def call(a, slot: int, prompt: list, begin: int, end: int, first_block: int):
    tokens = torch.tensor(prompt[:end]).view(1, end)
    table = torch.tensor([[first_block, first_block + 1]])
    a.prefill_forward(tokens, torch.tensor([begin]), torch.tensor([end]), page_table=table, empty_slots=[slot])


def test_snapshot_then_hit(adapter):
    a, gen = adapter, adapter.gen
    prompt = list(range(2 * GRID + 300))
    # a long prompt in chunks on vLLM slot 5 (row 5, user 0): its continuation calls snapshot the state at their start
    for begin in (0, GRID, 2 * GRID):
        call(a, 5, prompt, begin, min(begin + GRID, len(prompt)), first_block=40)
    p5 = a.phys[5]
    assert p5 // USERS == 5
    assert gen.snaps == [(p5, GRID, 0), (p5, 2 * GRID, 0)]  # (one entry, extended)
    keys = prefix_keys(prompt)
    index, free = glm53_vllm_rows.prefix_state()
    assert index == {keys[1]: (2 * GRID, 5)} and free[5] == USERS - 1
    # the next turn: the scheduler's hit at 2 GRID, admitted on vLLM slot 3 (row 3): placed on a free seat of row 5
    turn = prompt + [1, 2, 3] * 100
    got = glm53_vllm_rows.prefix_hit(prefix_keys(turn), len(turn), index, free)
    assert got == (2 * GRID, keys[1], 5)
    glm53_vllm_rows.publish_hit(70, got[0], got[1])
    call(a, 3, turn, 2 * GRID, len(turn), first_block=70)
    p3 = a.phys[3]
    assert p3 // USERS == 5 and p3 != p5 and gen.restores == [(0, p3)]
    assert gen.calls[-1] == [(p3, len(turn), 2 * GRID)]  # (from the prefix's end)
    assert a.prefix_counts == {"hits": 1, "misses": 0, "snapshots": 2}
    assert glm53_vllm_rows.take_hit(70) is None  # (taken)


def test_hit_without_a_seat_or_entry(adapter):
    a, gen = adapter, adapter.gen
    prompt = list(range(GRID + 10))
    call(a, 2, prompt, 0, GRID, first_block=10)
    call(a, 2, prompt, GRID, len(prompt), first_block=10)
    key = prefix_keys(prompt)[0]
    row = a.phys[2] // USERS
    # every seat of the entry's row in use: the hit's prompt goes from 0
    a.active |= {j for j in range(gen.slots) if a.phys[j] // USERS == row}
    glm53_vllm_rows.publish_hit(20, GRID, key)
    call(a, 1, prompt + [3], GRID, len(prompt) + 1, first_block=20)
    assert gen.calls[-1][0][1:] == (len(prompt) + 1, 0) and not gen.restores
    # an entry no longer held (another key): from 0 as well
    glm53_vllm_rows.publish_hit(30, GRID, "gone")
    call(a, 9, prompt + [4], GRID, len(prompt) + 1, first_block=30)
    assert gen.calls[-1][0][2] == 0 and a.prefix_counts["misses"] == 2


def test_chunk_on_another_seats_state(adapter):
    """A call from `begin` with no hit on a seat whose last prefill was not that prompt's: from 0 (the same values)."""
    a, gen = adapter, adapter.gen
    prompt = list(range(GRID + 50))
    call(a, 4, prompt, 0, GRID, first_block=5)
    call(a, 6, prompt, 0, 100, first_block=6)  # (another prefill since)
    call(a, 4, prompt, GRID, len(prompt), first_block=5)
    assert gen.calls[-1] == [(a.phys[4], len(prompt), 0)] and not gen.snaps


def test_scheduler_head(monkeypatch):
    """`Glm53Scheduler._prefix_head` on a stand-in scheduler: the waiting head's hit set on the request (its computed
    tokens), kept while it waits, none when no seat is free or the index has none."""
    pytest.importorskip("vllm_tt_plugin.scheduler")
    from vllm.v1.request import RequestStatus

    from models.demos.blackhole.glm53_flash.tools import glm53_vllm_scheduler as sched

    class Queue(list):
        def peek_request(self):
            return self[0]

    prompt = list(range(GRID + 40))
    req = SimpleNamespace(
        request_id="r1", num_computed_tokens=0, status=RequestStatus.WAITING, num_preemptions=0,
        skip_reading_prefix_cache=False, prompt_token_ids=prompt, num_prompt_tokens=len(prompt),
    )  # fmt: skip
    fake = SimpleNamespace(
        _prefix_pending=None, _prefix_keys=None, requests={"r1": req}, kv_holding_waiting=Queue(), waiting=Queue([req])
    )
    fake._waiting_head = lambda: sched.Glm53Scheduler._waiting_head(fake)
    head = lambda free: sched.Glm53Scheduler._prefix_head(fake, free)  # noqa: E731
    glm53_vllm_rows.reset()
    assert head(3) is None  # (no index published)
    key = prefix_keys(prompt)[0]
    glm53_vllm_rows.publish_prefix({key: (GRID, 2)}, [0] * ROWS)
    assert head(3) is None  # (no free seat on the entry's row)
    glm53_vllm_rows.publish_prefix({key: (GRID, 2)}, [1] * ROWS)
    assert head(0) is None  # (no free seat at all)
    assert head(3) == ("r1", GRID, key) and req.num_computed_tokens == GRID
    assert head(3) == ("r1", GRID, key)  # (waiting still: the same hit)
    req.status = RequestStatus.RUNNING
    fake.waiting.clear()
    assert head(3) is None and fake._prefix_pending is None
    glm53_vllm_rows.reset()


def test_scheduler_caps(monkeypatch):
    """`Glm53Scheduler._capped` on a stand-in step (vLLM 0.31's admission against max_num_active_reqs, then the plugin's
    check of max_num_running_reqs): both caps come down for the step (a serving run stopped at that check when only the
    second did), the paused streaming sessions counted, and both go back after it."""
    pytest.importorskip("vllm_tt_plugin.scheduler")
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm_scheduler as sched

    def step(self, throttle_prefills=False):
        while self.waiting and len(self.running) + self.num_waiting_for_streaming_input < self.max_num_active_reqs:
            self.running.append(self.waiting.pop(0))
        assert len(self.running) <= self.max_num_running_reqs
        return len(self.running)

    monkeypatch.setattr(sched.TTScheduler, "schedule", step)
    s = object.__new__(sched.Glm53Scheduler)
    s.running, s.waiting, s.num_waiting_for_streaming_input = ["a", "b"], ["c", "d", "e", "f"], 0
    s.max_num_running_reqs = s.max_num_active_reqs = 8
    assert s._capped(2, False) == 2  # (CHUNKED partway: no seat for a waiting request)
    assert s._capped(3, False) == 3  # (GLM53_PREFILL_BATCH: one free row)
    assert (s.max_num_running_reqs, s.max_num_active_reqs) == (8, 8)
    s.num_waiting_for_streaming_input = 1
    assert s._capped(4, False) == 4  # (a paused session holds a slot: one more admitted, running 4)
    assert s._capped(None, False) == 6  # (as configured: 8 slots, the streaming one among them, the waiting drained)


def test_serving_switch_defaults(monkeypatch):
    """DFlash2's serving defaults: the scheduler's gathering 4 / 1,500 ms (`glm53_vllm_rows.prefill_batch`) and the
    run-ahead's AHEAD_SEATS 1 / AHEAD_EDGE vllm (`serving_defaults`); a set variable wins, MTP / plain serving keeps the
    earlier form, GLM53_VLLM_RUN_AHEAD=0 leaves the run-ahead's off."""
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows as rows

    names = ("GLM53_DFLASH2", "GLM53_VLLM_RUN_AHEAD", "GLM53_PREFILL_BATCH", "GLM53_PREFILL_BATCH_WAIT_MS")
    for var in names + rows.RUN_AHEAD_SWITCHES + tuple(rows.RUN_AHEAD_VALUES) + tuple(rows.SERVING_PREFILL_VALUES):
        monkeypatch.setenv(var, "")  # (recorded, so that what serving_defaults sets goes back after the test)
        monkeypatch.delenv(var)
    assert rows.prefill_batch() == (0, 200.0) and not rows.serving_defaults()
    assert "GLM53_VLLM_AHEAD_SEATS" not in os.environ and "GLM53_PREFILL_CTX_EARLY" not in os.environ
    monkeypatch.setenv("GLM53_DFLASH2", "1")
    assert rows.prefill_batch() == (4, 1500.0)
    assert rows.serving_defaults()
    assert os.environ["GLM53_VLLM_AHEAD_SEATS"] == "1" and os.environ["GLM53_VLLM_AHEAD_EDGE"] == "vllm"
    assert os.environ["GLM53_VLLM_AHEAD"] == "1"
    assert all(os.environ[var] == value for var, value in rows.SERVING_PREFILL_VALUES.items())
    monkeypatch.setenv("GLM53_PREFILL_BATCH", "0")
    monkeypatch.setenv("GLM53_VLLM_AHEAD_EDGE", "cache")
    assert rows.prefill_batch() == (0, 1500.0) and rows.serving_defaults()
    assert os.environ["GLM53_VLLM_AHEAD_EDGE"] == "cache"  # (a set variable wins)
    for var in rows.RUN_AHEAD_SWITCHES + tuple(rows.RUN_AHEAD_VALUES):
        monkeypatch.delenv(var, raising=False)
    for var in rows.SERVING_PREFILL_VALUES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GLM53_VLLM_RUN_AHEAD", "0")
    assert not rows.serving_defaults() and "GLM53_VLLM_AHEAD_SEATS" not in os.environ
    assert all(os.environ[var] == value for var, value in rows.SERVING_PREFILL_VALUES.items())  # (not the run-ahead's)


def test_partway_prompt_on_its_seat(adapter):
    """The plugin hands a partway prompt's next chunk the lowest free state slot (another one when a request with a
    lower slot ended between its chunks): the chunk runs on its earlier chunks' seat (the two vLLM slots' seats
    swapped), from `begin`, and the snapshot is that seat's."""
    a, gen = adapter, adapter.gen
    prompt = list(range(2 * GRID + 70))
    call(a, 6, prompt, 0, GRID, first_block=50)
    seat, other = a.phys[6], a.phys[5]
    call(a, 5, prompt, GRID, 2 * GRID, first_block=50)  # (slot 6 then 5: the same request)
    assert a.phys[5] == seat and a.phys[6] == other and a.seat_moves == 1
    assert gen.calls[-1] == [(seat, 2 * GRID, GRID)] and gen.snaps == [(seat, GRID, 0)]
    call(a, 5, prompt, 2 * GRID, len(prompt), first_block=50)
    assert gen.calls[-1] == [(seat, len(prompt), 2 * GRID)] and a.seat_moves == 1
    assert a.prefix_counts == {"hits": 0, "misses": 0, "snapshots": 2}


def test_partway_prompt_on_its_seat_store_off(monkeypatch):
    """The same with no prefix store (GLM53_PREFIX_ENTRIES=0, or none fits): the chunk on its own seat from `begin`;
    a chunk whose seat's last prefill was another's goes from 0."""
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm as adapter_module

    monkeypatch.setattr(adapter_module, "PREFIX", False)
    monkeypatch.setattr(adapter_module, "CHUNKED", True)
    monkeypatch.setattr(adapter_module, "SPREAD_ROWS", False)
    glm53_vllm_rows.reset()
    gen = FakeGenerator()
    gen.prefix = None
    a = adapter_module.Glm53ForCausalLM(gen, 4 * GRID)
    assert not a.prefix_on and glm53_vllm_rows.prefix_state() is None
    prompt = list(range(GRID + 40))
    call(a, 3, prompt, 0, GRID, first_block=8)
    seat = a.phys[3]
    call(a, 0, prompt, GRID, len(prompt), first_block=8)
    assert a.phys[0] == seat and gen.calls[-1] == [(seat, len(prompt), GRID)] and a.seat_moves == 1
    call(a, 1, prompt, 0, GRID, first_block=9)
    call(a, 2, prompt, 0, 100, first_block=10)  # (another prefill since)
    call(a, 1, prompt, GRID, len(prompt), first_block=9)
    assert gen.calls[-1] == [(a.phys[1], len(prompt), 0)]
    glm53_vllm_rows.reset()
