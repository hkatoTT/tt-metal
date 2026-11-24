# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A prefill chunk's host tables (torch only, no ttnn: `tests/test_prefill_tables.py` checks them on the CPU).

The DSA indexer's and the KDA layers' per-row tables of a chunk (`TtDsa._indexer_tables` / `_sp_indexer_tables`,
`TtKda.prefill_picks` / `sp_picks`) were filled entry by entry in Python loops over the rows and their segments: ~300
small torch calls a sequence-parallel wave, ~15 ms of host time for a pair of waves while the device waited.
Here the loops only gather Python integers, and each table is made by a few whole-tensor ops: the same values and
dtypes (the test compares them with the loops' tensors, bit for bit).
"""

from __future__ import annotations

import torch


def _put(t: torch.Tensor, index: list, value) -> None:
    """t[index] = value for lists of coordinates (one list a dim), nothing for none."""
    if index[0]:
        t.index_put_(tuple(torch.tensor(i, dtype=torch.int64) for i in index), torch.tensor(value, dtype=t.dtype))


def history_pick(rows: int, valid: list, k: int, t: int) -> torch.Tensor:
    """[R, k, 32 + T]: row r's next conv history (its last k inputs) picked from the 32 history columns and the
    chunk's T tokens by its count of real tokens valid[r] (`TtKda.prefill_picks`, `sp_picks`)."""
    pick = torch.zeros(rows, k, 32 + t)
    rs, js, cols = [], [], []
    for r, v in enumerate(valid):
        for i in range(k):
            n = v - k + i
            rs.append(r)
            js.append(i)
            cols.append(32 + n if n >= 0 else n + k)
    _put(pick, [rs, js, cols], 1.0)
    return pick


def kda_sp_arrays(rows: int, member: dict, k: int) -> dict:
    """`TtKda.sp_picks`' tables from its rows' segments (`member`: row -> (group, index)): the chain's next-input
    picks `in` / `end` [R, k, 64], the rank mask [R, R] (int32), keep / owner [R, 1, 1]."""
    base = lambda row, j: 32 + k * row + j  # noqa: E731  (row's last input j in the gathered block)
    inc_i, end_i = ([], [], []), ([], [], [])
    mask_i = ([], [])
    keep, owner = [1.0] * rows, [0.0] * rows
    for r in range(rows):
        if r not in member:  # no segment: its history and state stay as they are
            for j in range(k):
                for idx in (inc_i, end_i):
                    idx[0].append(r)
                    idx[1].append(j)
                    idx[2].append(j)
            continue
        grp, i = member[r]
        n, v = len(grp.rows), grp.valid[-1]
        for j in range(k):
            inc_i[0].append(r)
            inc_i[1].append(j)
            inc_i[2].append(base(grp.rows[i - 1], j) if i else j)
            # a last segment shorter than the history continues the previous one's last inputs
            end_i[0].append(r)
            end_i[1].append(j)
            end_i[2].append(base(grp.owner, j) if n == 1 or v >= k or j >= k - v else base(grp.rows[-2], v + j))
        for src in grp.rows:
            mask_i[0].append(r)
            mask_i[1].append(src)
        keep[r] = 0.0 if grp.start == 0 else 1.0
        owner[r] = 1.0 if r == grp.owner else 0.0
    inc, end, mask = torch.zeros(rows, k, 64), torch.zeros(rows, k, 64), torch.zeros(rows, rows)
    _put(inc, list(inc_i), 1.0)
    _put(end, list(end_i), 1.0)
    _put(mask, list(mask_i), 1.0)
    return {
        "inc": inc,
        "end": end,
        "mask": mask,
        "keep": torch.tensor(keep).view(rows, 1, 1),
        "owner": torch.tensor(owner).view(rows, 1, 1),
    }


def kda_sp_history_tables(rows: int, valid: list, member: dict, k: int, width: int = 16) -> tuple:
    """The index form of `history_pick` and `kda_sp_arrays`' `in` / `end` picks (and keep) for the sequence-parallel
    conv history's copy ops (`kda_prefill_ops.sp_own` / `sp_pick`, generator.SP_HISTORY): own [R, 1, width] int32
    ([src_0 .. src_{k-1}, keep]: src >= 0 the chunk's row, src < 0 the history's row -src - 1) and pick [R, 1, width]
    int32 ([inc_0 .. inc_{k-1}, end_0 .. end_{k-1}, keep]: the 64-row block's rows: 0 .. k - 1 the history, 32 + k r + j
    the gathered).  Each the row of the 1 in the matrix's line (`tests/test_sp_history.py` checks them against it)."""
    base = lambda row, j: 32 + k * row + j  # noqa: E731
    own = torch.zeros(rows, 1, width, dtype=torch.int32)
    pick = torch.zeros(rows, 1, width, dtype=torch.int32)
    for r in range(rows):
        v = valid[r]
        keep = 1
        for i in range(k):
            n = v - k + i
            own[r, 0, i] = n if n >= 0 else -(n + k) - 1
        if r not in member:
            for j in range(k):
                pick[r, 0, j] = j
                pick[r, 0, k + j] = j
        else:
            grp, i = member[r]
            n, last = len(grp.rows), grp.valid[-1]
            for j in range(k):
                pick[r, 0, j] = base(grp.rows[i - 1], j) if i else j
                pick[r, 0, k + j] = (
                    base(grp.owner, j) if n == 1 or last >= k or j >= k - last else base(grp.rows[-2], last + j)
                )
            keep = 0 if grp.start == 0 else 1
        own[r, 0, k] = keep
        pick[r, 0, 2 * k] = keep
    return own, pick


def kda_sp_maps_x_arcs(rows: int, member: dict) -> list:
    """Each row's arc for `kda_maps_x` (GLM53_KDA_MAPS_X): (a, b), the hops back and forth along the
    column's ring that reach every other row of its request (`member`: row -> (group, index)) with the fewest hops the
    longer way (then the fewest in all); (0, 0) for a row alone or of no request."""
    arcs = []
    for r in range(rows):
        f = [(m - r) % rows for m in member[r][0].rows if m != r] if r in member else []
        best = (0, 0, 0, 0)  # (longer way, both ways, a, b)
        if f:
            best = None
            for b in range(rows):
                a = max([rows - x for x in f if x > b], default=0)
                key = (max(a, b), a + b, a, b)
                if best is None or key < best:
                    best = key
        arcs.append((best[2], best[3]))
    return arcs


def kda_sp_maps_x_table(rows: int, member: dict) -> torch.Tensor:
    """`kda_maps_x.exchange`'s table [R, 8] int32 (maps_x_send.cpp): a row's a, b (`kda_sp_maps_x_arcs`), the rows whose
    arcs reach it (bit s: it waits for their packets), the ready increments it waits for and, as bits, the rows to
    tell it is in the call, zeros after.
    Every row tells every other row and waits for all of them (rows - 1), not only the rows of its arc: with only those,
    two rows that no arc joins in a call ran on unpaced (the layouts change between calls), one got two calls ahead,
    and its ready increment for call k + 2 landed in the other's counter of the same parity while that one was still in
    call k (counted there, then set back to 0 at k's end): the sender then waited at k + 2 for a ready that was gone
    and its receivers for its packets (a traced stress of 16 layouts in turn: a device timeout).  Readies from
    all rows keep every row within one call of every other, which the two parities cover."""
    arcs = kda_sp_maps_x_arcs(rows, member)
    reach = [
        {(r + x) % rows for x in range(1, b + 1)} | {(r - x) % rows for x in range(1, a + 1)}
        for r, (a, b) in enumerate(arcs)
    ]
    table = torch.zeros(rows, 8, dtype=torch.int32)
    everyone = (1 << rows) - 1
    for d in range(rows):
        table[d, 0], table[d, 1] = arcs[d]
        table[d, 2] = sum(1 << s for s in range(rows) if d in reach[s])
        table[d, 3] = rows - 1
        table[d, 4] = everyone & ~(1 << d)
    return table


def dsa_sp_arrays(
    rows: int,
    member: dict,
    groups,
    users: int,
    t: int,
    *,
    kp: int,
    page: int,
    max_seq: int,
    pool_pages: int,
    n_read: int,
    sent: int,
    n_params: int,
    ring: int,
) -> dict:
    """`TtDsa._sp_indexer_tables`' tensors: rows [R, 1, 1, R T] int64, params [R, 1, T, P] int64, pool_written
    [R, R span] int32, pool_read [R, n_read] int32, recent_pick [R, 1, page, T], recent_table [R, 1] int32."""
    span = t // kp // page
    user, pos0 = [], []
    seg_r, seg_src, seg_first, seg_pool = [], [], [], []
    for r in range(rows):
        grp, i = member.get(r, (None, 0))
        u = grp.user if grp is not None and r == grp.owner else users - 1
        user.append(u)
        pos0.append(grp.start + i * t if grp is not None else 0)  # (no segment: unused)
        if grp is not None:
            for c, src in enumerate(grp.rows):
                seg_r.append(r)
                seg_src.append(src)
                seg_first.append(u * max_seq + grp.start + c * t)
                seg_pool.append(u * pool_pages + (grp.start + c * t) // kp // page)
    user_t = torch.tensor(user, dtype=torch.int64)
    base, pools = user_t * max_seq, user_t * pool_pages
    out_rows = torch.full((rows, rows, t), sent, dtype=torch.int64)
    written = torch.full((rows, rows, span), users * pool_pages, dtype=torch.int32)  # (scratch)
    if seg_r:
        at = (torch.tensor(seg_r, dtype=torch.int64), torch.tensor(seg_src, dtype=torch.int64))
        out_rows[at] = torch.tensor(seg_first, dtype=torch.int64)[:, None] + torch.arange(t)
        written[at] = (torch.tensor(seg_pool, dtype=torch.int64)[:, None] + torch.arange(span)).to(torch.int32)
    params = torch.zeros(rows, 1, t, n_params, dtype=torch.int64)
    params[:, 0, :, 0] = torch.tensor(pos0, dtype=torch.int64)[:, None] + torch.arange(t)
    params[:, 0, :, 1] = base[:, None]
    params[:, 0, :, 2] = pools[:, None]
    pick = torch.zeros(rows, 1, page, t)
    recent = [users] * rows  # (the scratch page)
    pick_i = ([], [], [], [])
    for grp in groups:
        lead = grp.start + (len(grp.rows) - 1) * t  # the last segment's first position
        first = (lead + grp.valid[-1]) // kp * kp - lead  # the pool after the last real token, from its first
        for j in range(kp):
            if 0 <= first + j < t:
                for idx, value in zip(pick_i, (grp.owner, 0, (lead + first + j) % ring, first + j)):
                    idx.append(value)
        recent[grp.owner] = grp.user
    _put(pick, list(pick_i), 1.0)
    return {
        "rows": out_rows.view(rows, 1, 1, rows * t),
        "params": params,
        "pool_written": written.view(rows, rows * span),
        "pool_read": (pools[:, None] + torch.arange(n_read)).to(torch.int32),
        "recent_pick": pick,
        "recent_table": torch.tensor(recent, dtype=torch.int32).view(rows, 1),
    }


def dsa_row_arrays(
    rows: int,
    slots: list,
    start: int,
    t: int,
    valid: list,
    users: int,
    *,
    kp: int,
    page: int,
    max_seq: int,
    pool_pages: int,
    n_read: int,
    sent: int,
    n_params: int,
    ring: int,
) -> dict:
    """`TtDsa._indexer_tables`' tensors (a one-row chunk at `start`, row r's request in user slots[r] or None)."""
    span = t // kp // page
    have = [r for r, s in enumerate(slots) if s is not None]  # (slots: at most `rows`)
    s_t = torch.tensor([slots[r] for r in have], dtype=torch.int64)
    out_rows = torch.full((rows, 1, 1, t), sent, dtype=torch.int64)
    params = torch.zeros(rows, 1, t, n_params, dtype=torch.int64)
    params[:, 0, :, 0] = start + torch.arange(t)
    written = torch.full((rows, span), users * pool_pages, dtype=torch.int32)
    read = torch.zeros(rows, n_read, dtype=torch.int32)
    pick = torch.zeros(rows, 1, page, t)
    recent = [users] * rows
    if have:
        h = torch.tensor(have, dtype=torch.int64)
        out_rows[h, 0, 0] = (s_t * max_seq)[:, None] + (start + torch.arange(t))
        params[h, 0, :, 1] = (s_t * max_seq)[:, None]
        params[h, 0, :, 2] = (s_t * pool_pages)[:, None]
        written[h] = ((s_t * pool_pages + start // kp // page)[:, None] + torch.arange(span)).to(torch.int32)
        read[h] = ((s_t * pool_pages)[:, None] + torch.arange(n_read)).to(torch.int32)
    pick_i = ([], [], [], [])
    for r in have:
        first = (start + valid[r]) // kp * kp - start  # the pool after the last real token, from its first token
        for j in range(kp):
            if 0 <= first + j < t:
                for idx, value in zip(pick_i, (r, 0, (start + first + j) % ring, first + j)):
                    idx.append(value)
        recent[r] = slots[r]
    _put(pick, list(pick_i), 1.0)
    return {
        "rows": out_rows,
        "params": params,
        "pool_written": written,
        "pool_read": read,
        "recent_pick": pick,
        "recent_table": torch.tensor(recent, dtype=torch.int32).view(rows, 1),
    }


__all__ = [
    "dsa_row_arrays",
    "dsa_sp_arrays",
    "history_pick",
    "kda_sp_arrays",
    "kda_sp_history_tables",
    "kda_sp_maps_x_arcs",
    "kda_sp_maps_x_table",
]
