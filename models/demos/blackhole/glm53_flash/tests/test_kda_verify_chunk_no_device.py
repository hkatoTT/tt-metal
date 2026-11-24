"""The chunked KDA verify step in torch, no device: its algebra against the token-by-token delta
rule, and its rounding against the verify op's of today.

The chunked form (per (user, head) pair): the chunk's positions p = 0 .. L - 1 are the last step's accepted tokens to
redo (r of them, from the saved rows) and this step's T tokens.  With G_p = prod_{i <= p} decay_i (per key channel),
K+ = K G, K- = K / G, Q+ = Q G:
    A = strictly_lower(diag(beta) K+ K-^T),  Tinv = (I + A)^-1 = (I + N)(I + N^2)(I + N^4)(I + N^8), N = -A
    U = Tinv diag(beta) (V - K+ S0)
    o_p = Q+_p S0 + sum_{i <= p} (Q+ K-^T)_{p i} U_i
    S after p = G_p S0 + sum_{i <= p} (K-_i G_p)^T U_i
The op's per-block rounds: X1 = [-Tinv diag(beta) K+ ; Q+] S0 + [Tinv diag(beta) ; 0] V (rows 0-15 U, rows 16-31 Q+ S0),
O' = [0 ; Aqk | I] X1, each state output G_t S0 + Kd_t^T X1.

The verify step's semantics (a user's keep code c, the last step's drafts accepted): c < K (slots) starts from the
saved start state of the last step and redoes its first c + 1 tokens; c = K starts from the state.  The step writes
the start state it used (slot 0), the state after all T tokens (the state) and its T tokens' rows (k, decay, beta, v)
for the next step's redo.  The results equal today's checkpoint form's (start from checkpoint c or the state).

Run: pytest models/demos/blackhole/glm53_flash/tests/test_kda_verify_chunk_no_device.py -s
"""

from __future__ import annotations

import os

import torch

DK = DV = 128


def tf32(x: torch.Tensor, mode: str | None) -> torch.Tensor:
    """x rounded to tf32 (10 mantissa bits) as a matmul operand ('trunc' or 'rne'), or as is (None)."""
    if mode is None:
        return x
    f = x.to(torch.float32).contiguous()
    bits = f.view(torch.int32)
    if mode == "rne":
        bits = bits + 0xFFF + ((bits >> 13) & 1)
    return (bits & ~0x1FFF).view(torch.float32).to(x.dtype)


def bf16(x: torch.Tensor, on: bool) -> torch.Tensor:
    return x.to(torch.bfloat16).to(x.dtype) if on else x


def mm(a, b, mode):
    return tf32(a, mode) @ tf32(b, mode)


def token_step(s, q, k, v, e, beta, mode=None, round_state=False):
    """One token of the delta rule as today's op computes it (mode: tf32 operands of k S and q S; round_state: S'
    rounded to bf16 for the next token) -> (o [DV], S')."""
    ds = s * e[:, None]
    r = mm(k[None], ds, mode)[0]
    delta = (v - r) * beta
    s2 = ds + k[:, None] * delta[None]
    o = mm(q[None], s2, mode)[0]
    return o, bf16(s2, round_state)


def chunk(s0, k, e, beta, v, q, outs, mode=None, round_state=False):
    """The chunked form over positions 0 .. L - 1 (k, e [L, DK]; beta [L]; v [L, DV]; q [L, DK], zero rows where no
    output is wanted), from s0 -> (o [L, DV], {t: state after position t for t in outs}), in the op's rounds."""
    L = k.shape[0]
    g = torch.cumprod(e, dim=0)
    kp, km, qp = k * g, k / g, q * g
    m_kk = mm(kp, km.T, mode)
    m_qk = mm(qp, km.T, mode)
    strict = torch.tril(torch.ones(L, L, dtype=s0.dtype), -1)
    lower = torch.tril(torch.ones(L, L, dtype=s0.dtype))
    a = m_kk * strict * beta[:, None]
    n = -a
    eye = torch.eye(L, dtype=s0.dtype)
    p2 = mm(n, n, mode)
    p4 = mm(p2, p2, mode)
    p8 = mm(p4, p4, mode)
    y = eye + mm(n, eye, mode)
    for p in (p2, p4, p8):
        y = y + mm(p, y, mode)
    tb = y * beta[None, :]  # Tinv diag(beta)
    c_top = -mm(tb, kp, mode)  # rows of X1's U part
    u = mm(c_top, s0, mode) + mm(tb, v, mode)
    qs = mm(qp, s0, mode)
    o = qs + mm(m_qk * lower, u, mode)
    states = {}
    for t in outs:
        kd = km[: t + 1] * g[t][None]
        states[t] = bf16(g[t][:, None] * s0 + mm(kd.T, u[: t + 1], mode), round_state)
    return o, states


def simulate(steps, T, K, seed, form, mode=None, round_state=False, exact=False):
    """A user-head pair's verify steps (random keep codes) -> (the outputs [steps, T, DV], the confirmed state after
    each step [steps, DK, DV]).  form: 'truth' (float64 token by token), 'today' (the checkpoint form's numerics),
    'chunk' (the chunked form); exact: float64 throughout.  Every form starts from the same bf16 state (as the device
    holds it)."""
    g = torch.Generator().manual_seed(seed)
    dt = torch.float64 if form == "truth" or exact else torch.float32
    s0 = torch.randn(DK, DV, generator=g, dtype=torch.float64) * 0.1
    state = bf16(s0, True).to(dt)
    slots = [state.clone() for _ in range(K)]  # today's checkpoints
    base = state.clone()  # the chunked form's start state of the last step (slot 0)
    saved = None  # its tokens' rows
    code = K
    outs, confirmed = [], []
    for _ in range(steps):
        q = torch.nn.functional.normalize(torch.randn(T, DK, generator=g, dtype=torch.float64), dim=-1) / DK**0.5
        k = torch.nn.functional.normalize(torch.randn(T, DK, generator=g, dtype=torch.float64), dim=-1)
        v = torch.randn(T, DV, generator=g, dtype=torch.float64)
        e = torch.exp(-5 * torch.sigmoid(torch.randn(T, DK, generator=g, dtype=torch.float64) - 3))
        beta = torch.sigmoid(torch.randn(T, generator=g, dtype=torch.float64))
        q, k, v, e, beta = (x.to(dt) for x in (q, k, v, e, beta))
        if form in ("truth", "today"):
            s = slots[code] if code < K else state
            o = []
            new = []
            for t in range(T):
                ot, s = token_step(s, q[t], k[t], v[t], e[t], beta[t], mode, round_state and form == "today")
                o.append(ot)
                new.append(s)
            slots, state = new[:K], new[-1]
            outs.append(torch.stack(o))
        else:
            r = code + 1 if code < K else 0
            s_start = base if r else state
            if r:
                ko, eo, bo, vo = saved
                kk = torch.cat([ko[:r], k])
                ee = torch.cat([eo[:r], e])
                bb = torch.cat([bo[:r], beta])
                vv = torch.cat([vo[:r], v])
                qq = torch.cat([torch.zeros(r, DK, dtype=dt), q])
            else:
                kk, ee, bb, vv, qq = k, e, beta, v, q
            L = kk.shape[0]
            want = [L - 1] + ([r - 1] if r else [])
            o, st = chunk(s_start, kk, ee, bb, vv, qq, want, mode, round_state)
            base = st[r - 1] if r else s_start
            state = st[L - 1]
            saved = (k, e, beta, v)
            outs.append(o[r:])
        code = int(torch.randint(0, K + 1, (1,), generator=g))
        # the confirmed state: after the accepted tokens (code + 1 of this step's T)
        if form in ("truth", "today"):
            confirmed.append(slots[code] if code < K else state)
        else:
            if code < K:
                _, st = chunk(base, saved[0][: code + 1], saved[1][: code + 1], saved[2][: code + 1],
                              saved[3][: code + 1], torch.zeros(code + 1, DK, dtype=dt), [code], None, False)  # fmt: skip
                confirmed.append(st[code])
            else:
                confirmed.append(state)
    return torch.stack(outs).double(), torch.stack(confirmed).double()


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_chunk_algebra_equals_token_by_token():
    """float64: the chunked form (and its redo of accepted tokens) gives the token-by-token outputs and states."""
    for T, K in ((4, 3), (8, 7), (2, 1)):
        o_t, s_t = simulate(12, T, K, seed=T, form="truth")
        o_c, s_c = simulate(12, T, K, seed=T, form="chunk", exact=True)
        eo, es = rel(o_c, o_t), rel(s_c, s_t)
        print(f"KCHUNK algebra T={T} K={K}: outputs rel {eo:.2e}, confirmed states rel {es:.2e}")
        assert eo < 1e-10 and es < 1e-10, (T, eo, es)


def test_chunk_rounding_against_today():
    """The rounding of the chunked form (tf32 matmul operands, the state in bf16 once a stored state) against today's
    (tf32 operands of k S and q S, S' in bf16 every token), both against float64, over many steps."""
    steps = int(os.environ.get("GLM53_KCHUNK_STEPS", "40"))
    for T, K in ((4, 3), (8, 7)):
        for mode in ("trunc", "rne"):
            o_t, s_t = simulate(steps, T, K, seed=10 + T, form="truth")
            o_a, s_a = simulate(steps, T, K, seed=10 + T, form="today", mode=mode, round_state=True)
            o_c, s_c = simulate(steps, T, K, seed=10 + T, form="chunk", mode=mode, round_state=True)
            print(f"KCHUNK rounding T={T} tf32 {mode}: today outputs {rel(o_a, o_t):.2e} states {rel(s_a, s_t):.2e}; "
                  f"chunked outputs {rel(o_c, o_t):.2e} states {rel(s_c, s_t):.2e}")  # fmt: skip
