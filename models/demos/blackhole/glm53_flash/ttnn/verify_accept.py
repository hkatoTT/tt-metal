"""Accepting a verify step's drafts (`TtGlm53Model.forward_verify`): per user, the drafts accepted n
(the leading drafts each equal to the model's greedy id one position before: draft j at p + j against the greedy id at
p + j - 1) and the bonus token (the greedy id at p + n), from which the next step goes on: KDA's keep code n, the next
token the bonus, the position p + n + 1.

`accept_host` on the host's tokens and greedy ids; `accept_device` on the device from forward_verify's ids and the
step's own token buffer, the result of every row on every chip, read back once ([ROWS, 2 U]: n, then the bonus)."""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.glm53_flash.ttnn.common import ROW_AXIS, ROWS, put


def accept_host(tokens: torch.Tensor, greedy: torch.Tensor, drafts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """tokens, greedy [ROWS, (K + 1) U] (virtual user j U + u: user u's token j, j = 0 the given token, 1 .. K its
    drafts; greedy the model's ids there) -> (n, bonus), each [ROWS, U] int64."""
    rows, b = tokens.shape
    t, g = tokens.reshape(rows, drafts + 1, -1), greedy.reshape(rows, drafts + 1, -1)
    n = (t[:, 1:] == g[:, :-1]).to(torch.int64).cumprod(dim=1).sum(dim=1)
    return n, g.gather(1, n[:, None, :]).squeeze(1)


_STEPS: dict = {}  # (mesh id, K + 1) -> [1, 1, K + 1, 1] fp32: 0 .. K (the bonus's pick)


def _own(ids, io, b: int):
    """Every row's ids [1, 1, 8 b, 1] fp32 -> this row's [1, 1, 1, b]."""
    grid = ttnn.reshape(ids, [1, 1, ROWS, b])
    own = ttnn.min(ttnn.add(grid, io.others), dim=2, keepdim=True)
    ttnn.deallocate(grid)
    return own


def _frac(x):
    """x - floor(x) (fp32); x deallocated."""
    whole = ttnn.floor(x)
    out = ttnn.subtract(x, whole)
    for t in (x, whole):
        ttnn.deallocate(t)
    return out


def forced_accept(base1, seats, drafts: int):
    """A test's acceptance (GLM53_TEST_FORCE_ACCEPT, `DFlash2Step`; the synthetic weights' drafts are never accepted):
    each user's n from its position and seat alone, its drafts taken whatever the greedy ids, so that the runs a test
    compares keep 0 .. K drafts alike: a = frac(0.0137 (p + 1) + 0.755 seat), b = frac(251.17 a^2 + 311.9 a),
    n = floor((K + 1) b) (a quadratic mix: frac(c p) alone locks into a few n as p steps by n + 1; this one spreads
    0 .. K evenly over 32 seats' runs in a host fp32 model).  base1 [1, 1, 1, U] int32 row-major (p + 1), seats
    [1, 1, 1, U] fp32 -> n [1, 1, 1, U] fp32."""
    b1 = ttnn.to_layout(base1, ttnn.TILE_LAYOUT)
    x = ttnn.typecast(b1, ttnn.float32)
    ttnn.deallocate(b1)
    xs, ss = ttnn.multiply(x, 0.0137), ttnn.multiply(seats, 0.7548776662)
    ttnn.deallocate(x)
    a = _frac(ttnn.add(xs, ss))
    for t in (xs, ss):
        ttnn.deallocate(t)
    sq = ttnn.multiply(a, a)
    hi, lo = ttnn.multiply(sq, 251.17), ttnn.multiply(a, 311.9)
    for t in (a, sq):
        ttnn.deallocate(t)
    b = _frac(ttnn.add(hi, lo))
    for t in (hi, lo):
        ttnn.deallocate(t)
    scaled = ttnn.multiply(b, float(drafts + 1))
    ttnn.deallocate(b)
    n = ttnn.floor(scaled)
    ttnn.deallocate(scaled)
    out = ttnn.clamp(n, min=0.0, max=float(drafts))
    ttnn.deallocate(n)
    return out


def accept_device(mesh, ids, io, raw=None, forced=None) -> object:
    """forward_verify's greedy ids [1, 1, 8 (K + 1) U, 1] fp32 (every row's on every chip) and `io` (SpecIO: its own
    row's tokens `ids` [1, (K + 1) U] uint32 row-major, `others`) -> [1, 1, 8, 2 U] fp32 row-major on every chip: row
    r's n (U), then its bonus tokens (U).  `raw` (the looser acceptance, GLM53_ACCEPT_RELAX: the plain argmax ids, as
    `ids`): [1, 1, 8, 3 U], each user's accepted drafts that were not the argmax after them.  `forced` (a test's,
    (base1, seats): `forced_accept`): n from the positions and seats in place of the drafts' match."""
    n_tok, u = io.drafts + 1, io.users
    b = n_tok * u
    own = _own(ids, io, b)  # [1, 1, 1, b]: this row's greedy ids
    tok = ttnn.typecast(ttnn.to_layout(ttnn.reshape(io.ids, [1, 1, 1, b]), ttnn.TILE_LAYOUT), ttnn.float32)
    g4 = ttnn.reshape(own, [1, 1, n_tok, u])
    t4 = ttnn.reshape(tok, [1, 1, n_tok, u])
    for x in (own, tok):
        ttnn.deallocate(x)
    drafts = ttnn.slice(t4, [0, 0, 1, 0], [1, 1, n_tok, u])
    match = ttnn.eq(drafts, ttnn.slice(g4, [0, 0, 0, 0], [1, 1, n_tok - 1, u]))
    ttnn.deallocate(t4)
    run = ttnn.cumprod(match, dim=2)
    ttnn.deallocate(match)
    n = ttnn.sum(run, dim=2, keepdim=True)  # [1, 1, 1, U]
    if forced is not None:  # (GLM53_TEST_FORCE_ACCEPT: no looser acceptance with it)
        assert raw is None, "GLM53_TEST_FORCE_ACCEPT with the looser acceptance"
        ttnn.deallocate(n)
        n = forced_accept(*forced, io.drafts)
    changed = None
    if raw is not None:  # the accepted drafts that differ from the plain argmax: n - sum(run x exact)
        r4 = ttnn.reshape(_own(raw, io, b), [1, 1, n_tok, u])
        exact = ttnn.eq(drafts, ttnn.slice(r4, [0, 0, 0, 0], [1, 1, n_tok - 1, u]))
        ttnn.deallocate(r4)
        kept = ttnn.sum(ttnn.mul(run, exact), dim=2, keepdim=True)
        ttnn.deallocate(exact)
        changed = ttnn.sub(n, kept)
        ttnn.deallocate(kept)
    ttnn.deallocate(drafts)
    ttnn.deallocate(run)
    key = (id(mesh), n_tok)
    if key not in _STEPS:
        steps = torch.arange(n_tok, dtype=torch.float32).reshape(1, 1, n_tok, 1)
        _STEPS[key] = put(mesh, steps, dims=(None, None), dtype=ttnn.float32)
    pick = ttnn.eq(_STEPS[key], n)  # [1, 1, K + 1, U]: 1 at row n
    chosen = ttnn.mul(g4, pick)
    for x in (g4, pick):
        ttnn.deallocate(x)
    bonus = ttnn.sum(chosen, dim=2, keepdim=True)  # [1, 1, 1, U]
    ttnn.deallocate(chosen)
    parts = [n, bonus] + ([changed] if changed is not None else [])
    both = ttnn.to_layout(ttnn.concat(parts, dim=3), ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, 1, 2 U] (3 U)
    for x in parts:
        ttnn.deallocate(x)
    out = ttnn.all_gather(both, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8, 2 U]
    ttnn.deallocate(both)
    return out


def accepted_to_host(out, users: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`accept_device`'s tensor -> (n, bonus), each [ROWS, U] int64 (the first chip's)."""
    flat = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(ROWS, -1).round().to(torch.int64)
    return flat[:, :users], flat[:, users : 2 * users]
