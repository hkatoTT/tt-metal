"""The serving generator's prefill batching (ttnn/generator.py, `prefill_many`) on a stand-in model: no device.

Requests go in waves of up to one per mesh row (each row's longest first), each row's request into its own user, chunk
by chunk from position 0; each row sees exactly its prompt (the first `valid` tokens of each of its chunks), a row
whose prompt has ended computes nothing more, logits are read only after a chunk some prompt ends in, each request
gets its own row's, and the wave's last chunk names every row's user (where its KDA state goes).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn.common import ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402

V = 11


@pytest.fixture(autouse=True)
def _no_device_buffers(monkeypatch):
    """The stand-in model has no mesh: no prefill buffers made at the generator's start (moe.COL_ROW_RS's, since
    91cc1d9014; test_schedule and test_prefill_sp_waves failed on them)."""
    from models.demos.blackhole.glm53_flash.ttnn import moe

    monkeypatch.setattr(moe, "COL_ROW_RS", False)


class FakeModel:
    """Collects the tokens each (row, user) slot receives; the 'logits' are a one-hot at the count so far (mod V)."""

    mesh = None
    dsa = None
    sp_ready = False

    def __init__(self):
        self.received = {}
        self.calls = []  # (start, slots, rows, valid, logits, last)
        self.committed = {}  # (row, user) -> the tokens it had received when its wave ended

    def empty_state(self, users, spec=0):
        return []

    def alloc_io(self, users):
        return None

    def prefill(
        self,
        ids,
        start,
        states,
        slots,
        rows=None,
        valid=None,
        logits=True,
        last=True,
        next_tokens=None,
        greedy_ids=False,
    ):
        self.calls.append((start, list(slots), list(rows), list(valid), logits, last))
        for r in range(ROWS):
            if rows[r]:
                assert slots[r] is not None, f"row {r} computes without a request"
                got = self.received.setdefault((r, slots[r]), [])
                assert len(got) == start, f"row {r} user {slots[r]}: chunk at {start} after {len(got)} tokens"
                got.extend(ids[r, : valid[r]].tolist())
        if last:
            for r, s in enumerate(slots):
                if s is not None:
                    self.committed[(r, s)] = list(self.received[(r, s)])
        if not logits:
            return None
        counts = [len(self.received.get((r, slots[r]), [])) % V for r in range(ROWS)]
        if greedy_ids:  # (the device's argmax: the one-hot's place)
            return torch.tensor(counts, dtype=torch.int64)
        return torch.nn.functional.one_hot(torch.tensor(counts), V).float()


def test_prefill_many_greedy_ids():
    """`greedy_ids`: each request's greedy first token (an int), the argmax of the logits it gets otherwise."""
    g = torch.Generator().manual_seed(1)
    requests = [(s, torch.randint(0, 1000, (n,), generator=g).tolist()) for s, n in ((0, 7), (3, 130), (9, 40))]
    logits = Glm53Generator(FakeModel(), users_per_row=2, chunk=128).prefill_many(requests)
    ids = Glm53Generator(FakeModel(), users_per_row=2, chunk=128).prefill_many(requests, greedy_ids=True)
    assert all(isinstance(i, int) for i in ids)
    assert ids == [int(x.argmax()) for x in logits]


def test_prefill_many_waves():
    model = FakeModel()
    gen = Glm53Generator(model, users_per_row=2, chunk=128)
    g = torch.Generator().manual_seed(0)
    # slot s is user s % 2 of row s // 2: slots 1, 2, 14 on rows 0, 1, 7 (users 1, 0, 0); slot 3 is row 1's second
    lengths = {1: 5, 3: 129, 2: 300, 14: 1}
    requests = [(slot, torch.randint(0, 1000, (n,), generator=g).tolist()) for slot, n in lengths.items()]
    out = gen.prefill_many(requests)
    for (slot, prompt), logits in zip(requests, out):
        row, user = divmod(slot, 2)
        assert model.received[(row, user)] == prompt and model.committed[(row, user)] == prompt
        assert int(logits.argmax()) == len(prompt) % V, f"slot {slot}: logits of another row or chunk"
    assert len(model.received) == len(requests), "no other slot is written"
    # wave 1: row 1's longest (slot 2, 300 tokens) with rows 0 and 7, each its own user; wave 2: slot 3
    first, second = model.calls[:3], model.calls[3:]
    assert [c[0] for c in first] == [0, 128, 256] and [c[0] for c in second] == [0, 128]
    assert all(c[1] == [1, 0] + [None] * 5 + [0] for c in first), "each row names its own user"
    _, _, rows, valid, logits, last = first[0]
    assert rows == [True, True] + [False] * 5 + [True] and logits and not last, "two prompts end in the first chunk"
    assert (valid[0], valid[1], valid[7]) == (5, 128, 1)
    assert first[1][2] == [False, True] + [False] * 6 and first[1][4] is False, "only row 1 left; nothing ends"
    assert first[2][3][1] == 300 - 256 and first[2][4] and first[2][5], "the wave's last chunk"
    assert [(c[1][1], c[3][1], c[4], c[5]) for c in second] == [(1, 128, False, False), (1, 1, True, True)]


def test_prefill_one_is_many():
    model = FakeModel()
    gen = Glm53Generator(model, users_per_row=4, chunk=32)
    logits = gen.prefill(13, list(range(40)))  # row 3, user 1
    assert model.received == {(3, 1): list(range(40))} and int(logits.argmax()) == 40 % V
    assert [(c[0], c[1][3], c[3][3], c[4], c[5]) for c in model.calls] == [
        (0, 1, 32, False, False),
        (32, 1, 8, True, True),
    ]


def test_schedule():
    gen = Glm53Generator(FakeModel(), users_per_row=1, chunk=(128, 512))
    assert gen.schedule(100) == [(0, 128)]
    assert gen.schedule(256) == [(0, 128), (128, 128)], "two short chunks cost less than one long one"
    assert gen.schedule(257) == [(0, 512)]
    assert gen.schedule(700) == [(0, 512), (512, 128), (640, 128)]
    assert gen.schedule(2048) == [(0, 512), (512, 512), (1024, 512), (1536, 512)]
    assert gen.schedule(700, chunk=128) == [(i * 128, 128) for i in range(6)]
    # one wave: the longest prompt's schedule, the shorter prompt ending inside the first (long) chunk
    model = FakeModel()
    gen = Glm53Generator(model, users_per_row=1, chunk=(128, 512))
    out = gen.prefill_many([(0, list(range(700))), (5, list(range(100)))])
    assert [c[0] for c in model.calls] == [0, 512, 640]
    assert model.calls[0][3][0] == 512 and model.calls[0][3][5] == 100 and model.calls[0][4]
    assert model.received[(0, 0)] == list(range(700)) and model.received[(5, 0)] == list(range(100))
    assert int(out[0].argmax()) == 700 % V and int(out[1].argmax()) == 100 % V


def test_waves_longest_first():
    model = FakeModel()
    gen = Glm53Generator(model, users_per_row=4, chunk=32)
    # row 0 gets three requests (users 0, 1, 2), row 1 two: the longest of each row go together
    requests = [(0, [1] * 10), (1, [2] * 70), (2, [3] * 40), (4, [4] * 5), (5, [5] * 60)]
    gen.prefill_many(requests)
    waves = [c[1] for c in model.calls if c[0] == 0]
    assert waves == [[1, 1] + [None] * 6, [2, 0] + [None] * 6, [0] + [None] * 7]
    assert all(model.received[(s // 4, s % 4)] == p for s, p in requests)


def test_prefill_refusals(expect_error):
    gen = Glm53Generator(FakeModel(), users_per_row=2, chunk=32)
    with expect_error(ValueError, "given twice"):
        gen.prefill_many([(4, [1]), (4, [2])])
    with expect_error(ValueError, "out of"):
        gen.prefill_many([(16, [1])])
    with expect_error(ValueError, "empty prompt"):
        gen.prefill_many([(0, [])])
    with expect_error(ValueError, "each dividing the next"):
        Glm53Generator(FakeModel(), users_per_row=1, chunk=(96, 128))
    assert gen.prefill_many([]) == []


class FakeSpModel(FakeModel):
    """FakeModel with the sequence-parallel prefill: each group's segments must continue its prompt on rows consecutive
    in the ring order, ending at its own row, the groups in the ring order from the first's first row."""

    sp_ready = True

    def __init__(self):
        super().__init__()
        self.waves = []  # per wave: [(rows, valid, user, start, last)]

    def prefill_sp(self, ids, states, groups, logits=True):
        self.waves.append([tuple(g) for g in groups])
        first = groups[0].rows[0]
        order = [(first + c) % ROWS for c in range(ROWS)]
        at, out = 0, {}
        for g in groups:
            pos = [order.index(r) for r in g.rows]
            assert pos[0] >= at and pos == list(range(pos[0], pos[0] + len(pos))), f"rows {g.rows} out of order"
            at = pos[-1] + 1
            got = self.received.setdefault((g.owner, g.user), [])
            assert len(got) == g.start, f"a group at {g.start} after {len(got)} tokens"
            for r, v in zip(g.rows, g.valid):
                got.extend(ids[r, :v].tolist())
            if g.last:
                self.committed[(g.owner, g.user)] = list(got)
                out[g.owner] = torch.nn.functional.one_hot(torch.tensor(len(got) % V), V).float()
        return out if logits else {}


def test_prefill_sp_waves():
    """A round of requests goes over the rows when that takes fewer waves (`sp_plan`): each request on the block of
    rows that ends at its own row, its next segments on the block's last rows, the prompt in order; a single prompt
    takes all 8 rows; one row each as before with the form off, for a prompt within a chunk, or when every request
    has one row."""
    model = FakeSpModel()
    gen = Glm53Generator(model, users_per_row=2, chunk=(128,))
    prompt = list(range(1, 128 * 8 + 128 * 2 + 41))  # 10 full segments and 40 tokens: two waves (8, 3)
    (logits,) = gen.prefill_many([(11, prompt)])  # row 5, user 1
    assert model.committed == {(5, 1): prompt}
    assert [[(rows, start, last) for rows, _, _, start, last in w] for w in model.waves] == [
        [((6, 7, 0, 1, 2, 3, 4, 5), 0, False)],
        [((3, 4, 5), 1024, True)],
    ]
    assert int(logits.argmax()) == len(prompt) % V
    assert not model.calls, "the one-row form ran"
    assert gen.sp_blocks([0, 1]) == {0: (2, 3, 4, 5, 6, 7, 0), 1: (1,)}
    assert gen.sp_blocks([3]) == {3: (4, 5, 6, 7, 0, 1, 2, 3)}
    gen.prefill_many([(3, list(range(100)))])  # a prompt within one chunk: one row
    gen.sp = False
    gen.prefill_many([(4, prompt)])
    gen.sp = True
    assert len(model.waves) == 2 and len(model.calls) == 1 + 11
    # two requests (rows 0 and 1): row 0's on rows 2..7, 0 (its 3 segments on 6, 7, 0), row 1's alone on row 1
    outs = gen.prefill_many([(0, prompt[:300]), (2, prompt[:200])])
    assert [[(rows, start, last) for rows, _, _, start, last in w] for w in model.waves[2:]] == [
        [((6, 7, 0), 0, True), ((1,), 0, False)],
        [((1,), 128, True)],
    ]
    assert model.committed[(0, 0)] == prompt[:300] and model.committed[(1, 0)] == prompt[:200]
    assert [int(lg.argmax()) for lg in outs] == [300 % V, 200 % V]
    # two long requests on rows 0 and 1 (blocks of 7 rows and 1: 8 waves, as one row each): one after another, each
    # on all 8 rows (a wave each, `sp_rounds`)
    alone_model = FakeSpModel()
    alone = Glm53Generator(alone_model, users_per_row=2, chunk=(128,))
    outs = alone.prefill_many([(0, prompt[:1024]), (2, prompt[1:1025])])
    assert [[(rows, start, last) for rows, _, _, start, last in w] for w in alone_model.waves] == [
        [((1, 2, 3, 4, 5, 6, 7, 0), 0, True)],
        [((2, 3, 4, 5, 6, 7, 0, 1), 0, True)],
    ]
    assert alone_model.committed == {(0, 0): prompt[:1024], (1, 0): prompt[1:1025]} and not alone_model.calls
    assert [int(lg.argmax()) for lg in outs] == [1024 % V, 1024 % V]
    assert gen.sp_rounds({0: 1024, 1: 1024}) == [((0,), 128, gen.sp_blocks([0])), ((1,), 128, gen.sp_blocks([1]))]
    assert gen.sp_rounds({0: 1024, 4: 1024})[0][0] == (0, 4)  # blocks of 4 rows each: together (as fast, one round)
    # a request on every row: one row each
    fresh = FakeSpModel()
    Glm53Generator(fresh, users_per_row=2, chunk=(128,)).prefill_many([(2 * r, prompt[: 200 + r]) for r in range(ROWS)])
    assert not fresh.waves and len(fresh.calls) == 2
    # wide chunks: a 1,024-token prompt in 8 segments of 128, a 2,048-token one in 4 of 512
    wide = Glm53Generator(FakeSpModel(), users_per_row=1, chunk=(128, 512))
    assert wide.sp_plan({2: 1024})[0] == 128 and wide.sp_plan({2: 2048})[0] == 512
    assert wide.sp_plan({2: 128}) is None


def test_prefill_begin(expect_error):
    """A chunked prefill (`begin`, tools/glm53_vllm.py CHUNKED): each call with the prompt so far runs only its waves
    from `begin`, the same as the whole prompt's from there (over the rows and on one row), each call ending as a whole
    prompt does; two requests or a `begin` off the grid are refused."""
    model = FakeSpModel()
    gen = Glm53Generator(model, users_per_row=2, chunk=(128,))
    prompt = list(range(1, 128 * 8 * 2 + 128 * 2 + 41))  # two full waves, then 2 full segments and 40 tokens
    gen.prefill_many([(11, prompt)])  # row 5, user 1
    whole = [[(rows, start) for rows, _, _, start, _ in w] for w in model.waves]
    model.waves = []
    cut = [0, 1024, 2048, len(prompt)]
    outs = [gen.prefill_many([(10, prompt[:e])], begin=b)[0] for b, e in zip(cut, cut[1:])]  # row 5, user 0
    assert model.committed[(5, 0)] == prompt and int(outs[-1].argmax()) == len(prompt) % V
    assert [[(rows, start) for rows, _, _, start, _ in w] for w in model.waves] == whole
    assert [w[0][4] for w in model.waves] == [True, True, True]
    gen.sp = False  # one row: the chunks from `begin`
    gen.prefill_many([(8, prompt)])  # row 4, user 0
    starts_whole = [c[0] for c in model.calls]
    model.calls = []
    for b, e in zip(cut, cut[1:]):
        gen.prefill_many([(9, prompt[:e])], begin=b)  # row 4, user 1
    assert [c[0] for c in model.calls] == starts_whole and model.committed[(4, 1)] == prompt
    with expect_error(ValueError, "one request"):
        gen.prefill_many([(0, prompt[:300]), (2, prompt[:300])], begin=128)
    with expect_error(ValueError, "not a chunk start"):
        Glm53Generator(FakeModel(), users_per_row=2, chunk=(128,)).prefill_many([(0, prompt[:300])], begin=100)
    gen.sp = True
    with expect_error(ValueError, "not a multiple"):
        gen.prefill_many([(6, prompt)], begin=100)
