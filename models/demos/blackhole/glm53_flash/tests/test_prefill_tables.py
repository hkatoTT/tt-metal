"""`ttnn/prefill_tables.py` (the prefill chunk's host tables in whole-tensor ops) against the Python loops they replaced
(copied below from `TtDsa._sp_indexer_tables` / `_indexer_tables` and `TtKda.sp_picks` / `prefill_picks` as of
65e2a4c07d): the same values and dtypes, bit for bit, over random sequence-parallel waves (1-3 requests, rows between
groups, any start, a short last segment) and one-row chunks; then each form's host time.  No device (torch only).

    pytest models/demos/blackhole/glm53_flash/tests/test_prefill_tables.py -s
"""

from __future__ import annotations

import random
import time
from typing import NamedTuple

import torch

from models.demos.blackhole.glm53_flash.ttnn import prefill_tables

ROWS, PAGE, SENT, PARAMS, RING = 8, 32, 0xFFFFFFFF, 16, 16  # (common.ROWS, dsa.PAGE, dsa_indexer's)
MAX_SEQ, KP = 8192, 4
POOL_PAGES = MAX_SEQ // KP // PAGE
SIZES = dict(kp=KP, page=PAGE, max_seq=MAX_SEQ, pool_pages=POOL_PAGES, sent=SENT, n_params=PARAMS, ring=RING)


class SpGroup(NamedTuple):  # (common.SpGroup)
    rows: tuple
    valid: tuple
    user: int
    start: int
    last: bool = True

    @property
    def owner(self) -> int:
        return self.rows[-1]


# -- the loops (65e2a4c07d) ----------------------------------------------------------------------------------------


def _old_dsa_sp(users, groups, member, t, n_read):
    kp, span = KP, t // KP // PAGE
    rows = torch.full((ROWS, 1, 1, ROWS * t), SENT, dtype=torch.int64)
    params = torch.zeros(ROWS, 1, t, PARAMS, dtype=torch.int64)
    pool_written = torch.full((ROWS, ROWS * span), users * POOL_PAGES, dtype=torch.int32)
    pool_read = torch.zeros(ROWS, n_read, dtype=torch.int32)
    pick = torch.zeros(ROWS, 1, PAGE, t)
    recent = torch.full((ROWS, 1), users, dtype=torch.int32)
    for r in range(ROWS):
        grp, i = member.get(r, (None, 0))
        user = grp.user if grp is not None and r == grp.owner else users - 1
        base, pools = user * MAX_SEQ, user * POOL_PAGES
        if grp is not None:
            for c, src in enumerate(grp.rows):
                rows[r, 0, 0, src * t : (src + 1) * t] = base + grp.start + c * t + torch.arange(t)
                first_page = pools + (grp.start + c * t) // kp // PAGE
                pool_written[r, src * span : (src + 1) * span] = first_page + torch.arange(span)
        positions = (grp.start + i * t if grp is not None else 0) + torch.arange(t)
        params[r, 0, :, 0] = positions
        params[r, 0, :, 1] = base
        params[r, 0, :, 2] = pools
        pool_read[r] = pools + torch.arange(n_read)
    for grp in groups:
        lead = grp.start + (len(grp.rows) - 1) * t
        first = (lead + grp.valid[-1]) // kp * kp - lead
        for j in range(kp):
            if 0 <= first + j < t:
                pick[grp.owner, 0, (lead + first + j) % RING, first + j] = 1.0
        recent[grp.owner] = grp.user
    return dict(rows=rows, params=params, pool_written=pool_written, pool_read=pool_read, recent_pick=pick,
                recent_table=recent)  # fmt: skip


def _old_dsa_row(users, slots, start, t, valid, n_read):
    kp = KP
    positions = start + torch.arange(t)
    rows = torch.full((ROWS, 1, 1, t), SENT, dtype=torch.int64)
    params = torch.zeros(ROWS, 1, t, PARAMS, dtype=torch.int64)
    params[:, 0, :, 0] = positions
    pool_written = torch.full((ROWS, t // kp // PAGE), users * POOL_PAGES, dtype=torch.int32)
    pool_read = torch.zeros(ROWS, n_read, dtype=torch.int32)
    pick = torch.zeros(ROWS, 1, PAGE, t)
    recent = torch.full((ROWS, 1), users, dtype=torch.int32)
    for r, s in enumerate(slots):
        if s is None:
            continue
        base = s * MAX_SEQ
        rows[r, 0, 0] = base + positions
        params[r, 0, :, 1] = base
        params[r, 0, :, 2] = s * POOL_PAGES
        pool_written[r] = s * POOL_PAGES + start // kp // PAGE + torch.arange(t // kp // PAGE)
        pool_read[r] = s * POOL_PAGES + torch.arange(n_read)
        first = (start + valid[r]) // kp * kp - start
        for j in range(kp):
            if 0 <= first + j < t:
                pick[r, 0, (start + first + j) % RING, first + j] = 1.0
        recent[r] = s
    return dict(rows=rows, params=params, pool_written=pool_written, pool_read=pool_read, recent_pick=pick,
                recent_table=recent)  # fmt: skip


def _old_history(valid, k, t):
    pick = torch.zeros(ROWS, k, 32 + t)
    for r, v in enumerate(valid):
        for i in range(k):
            n = v - k + i
            pick[r, i, 32 + n if n >= 0 else n + k] = 1.0
    return pick


def _old_kda_sp(member, k):
    base = lambda row, j: 32 + k * row + j  # noqa: E731
    inc, end = torch.zeros(ROWS, k, 64), torch.zeros(ROWS, k, 64)
    mask = torch.zeros(ROWS, ROWS)
    keep, owner = torch.ones(ROWS, 1, 1), torch.zeros(ROWS, 1, 1)
    for r in range(ROWS):
        if r not in member:
            for j in range(k):
                inc[r, j, j] = end[r, j, j] = 1.0
            continue
        grp, i = member[r]
        n, v = len(grp.rows), grp.valid[-1]
        for j in range(k):
            inc[r, j, base(grp.rows[i - 1], j) if i else j] = 1.0
            end[r, j, base(grp.owner, j) if n == 1 or v >= k or j >= k - v else base(grp.rows[-2], v + j)] = 1.0
        mask[r, list(grp.rows)] = 1.0
        keep[r] = 0.0 if grp.start == 0 else 1.0
        owner[r] = 1.0 if r == grp.owner else 0.0
    return dict(inc=inc, end=end, mask=mask, keep=keep, owner=owner)


# -- the cases -----------------------------------------------------------------------------------------------------


def _wave(rnd: random.Random, t: int) -> list:
    """A sequence-parallel wave's groups: 1-3 requests on consecutive rows of the ring from a random first row,
    sometimes a row between two groups, the last segment of each 1..T tokens, a start at a multiple of 128."""
    first = rnd.randrange(ROWS)
    n_req = rnd.randint(1, 3)
    cuts = sorted(rnd.sample(range(1, ROWS), n_req - 1))
    groups, at = [], 0
    for n in (b - a for a, b in zip([0] + cuts, cuts + [ROWS])):
        m = n - (1 if n > 1 and rnd.random() < 0.3 else 0)
        rows = tuple((first + at + i) % ROWS for i in range(m))
        valid = (t,) * (m - 1) + (rnd.choice([1, 2, 3, 4, 5, rnd.randint(1, t), t]),)
        start = rnd.randrange(0, (MAX_SEQ - m * t) // 128 + 1) * 128
        groups.append(SpGroup(rows, valid, rnd.randrange(32), start, rnd.random() < 0.7))
        at += n
    return groups


def _same(name: str, new: dict, old: dict) -> None:
    for key, want in old.items():
        got = new[key]
        assert got.dtype == want.dtype and got.shape == want.shape, f"{name} {key}: {got.dtype} {got.shape}"
        assert torch.equal(got, want), f"{name} {key}: values differ"


def test_prefill_tables():
    rnd = random.Random(53)
    users = 33
    timings: dict = {}

    def timed(label, fn):
        t0 = time.perf_counter()
        out = fn()
        timings.setdefault(label, []).append(time.perf_counter() - t0)
        return out

    for case in range(400):
        t = rnd.choice([128, 512, 1024])
        groups = _wave(rnd, t)
        member = {r: (g, i) for g in groups for i, r in enumerate(g.rows)}
        valid = [member[r][0].valid[member[r][1]] if r in member else 0 for r in range(ROWS)]
        n_read = rnd.choice([16, 32, 64])
        old = timed("dsa sp, loops", lambda: _old_dsa_sp(users, groups, member, t, n_read))
        new = timed("dsa sp, whole tensors", lambda: prefill_tables.dsa_sp_arrays(
            ROWS, member, groups, users, t, n_read=n_read, **SIZES))  # fmt: skip
        _same(f"case {case} dsa sp", new, old)
        old = timed("kda sp, loops", lambda: dict(_old_kda_sp(member, 3), hist=_old_history(valid, 3, t)))
        new = timed("kda sp, whole tensors", lambda: dict(prefill_tables.kda_sp_arrays(ROWS, member, 3),
                                                          hist=prefill_tables.history_pick(ROWS, valid, 3, t)))  # fmt: skip
        _same(f"case {case} kda sp", new, old)
        slots = [None if rnd.random() < 0.25 else rnd.randrange(32) for _ in range(ROWS)]
        start = rnd.randrange(0, (MAX_SEQ - t) // 128 + 1) * 128
        valid_row = [rnd.choice([1, 2, 3, 4, rnd.randint(1, t), t]) for _ in range(ROWS)]
        old = timed("dsa row, loops", lambda: _old_dsa_row(users, slots, start, t, valid_row, n_read))
        new = timed("dsa row, whole tensors", lambda: prefill_tables.dsa_row_arrays(
            ROWS, slots, start, t, valid_row, users, n_read=n_read, **SIZES))  # fmt: skip
        _same(f"case {case} dsa row", new, old)
        old = timed("kda row, loops", lambda: _old_history(valid_row, 3, t))
        new = timed("kda row, whole tensors", lambda: prefill_tables.history_pick(ROWS, valid_row, 3, t))
        _same(f"case {case} kda row", {"h": new}, {"h": old})
    for label, ts in timings.items():
        ts = sorted(ts)
        print(f"KTABLES {label}: median {ts[len(ts) // 2] * 1e3:.3f} ms a call ({len(ts)} calls)", flush=True)
