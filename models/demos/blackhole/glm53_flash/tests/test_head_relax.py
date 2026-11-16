"""The looser acceptance's head (`TtLmHead.greedy_mesh` with `relax`, GLM53_ACCEPT_RELAX) and teacher forcing's scores
(`TtLmHead.score_mesh` / `score_join`) against torch on the device's own mesh logits (the head's product read back from
the 32 chips): random weights of a small vocabulary (the last chip's columns past it masked), rows M = 8 T of 64 / 256
/ 512 (DFlash2 K = 7 at 8 and 32 users, K = 3 at 128), each token's "next" one the top, a near one (ranks 2-6), a far
one or none (-1).  For gaps < 0 / 0 / 0.3 / 1 / 3 nats: the relaxed ids are the next token exactly where its logit is
within the gap of the top, else the greedy ids (< 0: the plain head's ids, bit for bit); the same replayed from one
capture with the gap changed between replays; the extra time in a trace.  The scores: the next token's
log-probability, its gap and the top's log-probability within 2e-4 of float64 on the same logits.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_head_relax.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    ROWS,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.head import TtLmHead  # noqa: E402

VOCAB, D = 20_096, 1024  # (the TP head: a quarter a chip, whole tiles; the last chip 256 of its 640 ids real)
GAPS = (-1.0, 0.0, 0.3, 1.0, 3.0)


def mesh_logits(head, x) -> torch.Tensor:
    """The head's product as greedy_mesh makes it, read back -> [M, 32 w] fp32 (chip k's columns at k w; past the
    vocabulary -inf)."""
    x_all = head._gather_rows(x)
    part = ttnn.linear(x_all, head.mesh_weight, dtype=ttnn.float32, compute_kernel_config=HIFI4_FP32)
    chips = [ttnn.to_torch(c)[0, 0] for c in ttnn.get_device_tensors(part)]
    for t in (x_all, part):
        ttnn.deallocate(t)
    full = torch.cat(chips, dim=-1)  # chip k = 4 r + c in mesh order
    full[:, VOCAB:] = float("-inf")
    return full


def ids_of(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_head_relax(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(5)
    weight = (torch.randn(VOCAB, D, generator=g) * 0.05).to(torch.bfloat16)
    head = TtLmHead(mesh, weight)
    head.prepare_relax(-1.0)
    failures = []
    for t in (8, 32, 64):
        m = ROWS * t
        x = put(mesh, (torch.randn(ROWS, 1, t, D, generator=g)).to(torch.bfloat16), dims=(0, None))
        logits = mesh_logits(head, x).double()
        top_ids = logits.argmax(-1)  # (ties: the smallest id, as the head)
        ranked = logits.topk(6, dim=-1).indices
        kind = torch.randint(0, 4, (m,), generator=g)
        nxt = torch.where(kind == 0, top_ids, ranked[torch.arange(m), torch.randint(1, 6, (m,), generator=g)])
        nxt = torch.where(kind == 2, torch.randint(0, VOCAB, (m,), generator=g), nxt)
        nxt = torch.where(kind == 3, torch.full_like(nxt, -1), nxt)
        gap = logits.max(-1).values - logits.gather(1, nxt.clamp(min=0)[:, None])[:, 0]
        gap = torch.where(nxt < 0, torch.full_like(gap, float("inf")), gap)
        d_nxt = put(mesh, nxt.float().view(1, 1, m, 1), dims=(None, None), dtype=ttnn.float32)

        plain = head.greedy_mesh(x)
        plain_ids = ids_of(plain)
        if not torch.equal(plain_ids.long(), top_ids):
            failures.append((m, "the plain head against torch"))
        eager = {}
        for delta in GAPS:
            head.set_relax(delta)
            out = head.greedy_mesh(x, relax=(d_nxt, head.relax))
            got = ids_of(out)
            ttnn.deallocate(out)
            want = torch.where(gap <= delta, nxt, top_ids)
            same = torch.equal(got.long(), want)
            bits = delta >= 0 or torch.equal(got.view(torch.int32), plain_ids.view(torch.int32))
            eager[delta] = got
            print(
                f"M {m:3d} gap {delta:4.1f}: relaxed ids = torch {same} (taken {(gap <= delta).sum().item()} of {m}"
                f"{'' if delta >= 0 else f'; the plain ids bit for bit {bits}'})",
                flush=True,
            )
            if not (same and bits):
                bad = (got.long() != want).nonzero()[:5, 0].tolist()
                print(f"  differ at {bad}: got {got[bad].tolist()} want {want[bad].tolist()} gap {gap[bad].tolist()}")
                failures.append((m, delta))

        # one capture, the gap changed between replays
        head.set_relax(GAPS[0])
        out = head.greedy_mesh(x, relax=(d_nxt, head.relax))  # (eager first)
        ttnn.deallocate(out)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            captured = head.greedy_mesh(x, relax=(d_nxt, head.relax))
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        replay_ok = True
        for delta in GAPS[::-1]:
            head.set_relax(delta)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            replay_ok = replay_ok and torch.equal(ids_of(captured).view(torch.int32), eager[delta].view(torch.int32))
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t_relax = (time.perf_counter() - t0) / reps
        ttnn.release_trace(mesh, trace)
        ttnn.deallocate(captured)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            captured = head.greedy_mesh(x)
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t_plain = (time.perf_counter() - t0) / reps
        ttnn.release_trace(mesh, trace)
        ttnn.deallocate(captured)
        print(
            f"M {m:3d}: replays with the gap changed = eager {replay_ok}; traced head {t_plain * 1e6:.1f} -> relaxed "
            f"{t_relax * 1e6:.1f} us (+{(t_relax - t_plain) * 1e6:.1f})",
            flush=True,
        )
        if not replay_ok:
            failures.append((m, "replay"))

        # teacher forcing's scores on the same logits
        out = head.score_mesh(x, d_nxt)
        got = head.score_join(out, m)
        ttnn.deallocate(out)
        lse = torch.logsumexp(logits, -1)
        top = logits.max(-1).values
        nl = logits.gather(1, nxt.clamp(min=0)[:, None])[:, 0]
        real = nxt >= 0
        want = torch.stack([nl - lse, top - nl, top - lse], -1)
        err = (got[real] - want[real]).abs().max().item()
        top_exact = torch.equal(got[real & (kind == 0), 1], torch.zeros_like(got[real & (kind == 0), 1]))
        none_ok = bool((got[~real, 1] > 1e29).all())
        print(
            f"M {m:3d}: scores max |error| {err:.2e} (logp of the next, its gap, the top's logp); the top's gap 0 "
            f"exactly {top_exact}; no next -> gap ~BIG {none_ok}",
            flush=True,
        )
        if not (err < 2e-4 and top_exact and none_ok):
            failures.append((m, "scores"))
        for x_ in (x, d_nxt, plain):
            ttnn.deallocate(x_)
    assert not failures, failures


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_head_relax_fused(mesh_device):
    """The fused relaxed head (`head_relax.head_pick_relax`, the step's tokens in: head.RELAX_FUSED) against
    the ttnn form (`TtLmHead._relaxed`, the next tokens' column in), bit for bit, both outputs (the relaxed ids and the
    argmax ids): M = 64 (K = 7, 1 user a row) / 128 (K = 3, 4 users) / 256 (K = 7, 4 users), each row's next token the
    top, a near one, a far one or none (the last level), gaps < 0 / 0 / 0.3 / 1 / 3 and two gaps equal to some row's
    gap (the compare's boundary); the same replayed from one capture with the gap changed; the traced time of the plain
    head, the ttnn form and the fused one (the head alone: the inputs' making, `TtGlm53Model._relax`, is in the model's
    step times)."""
    from models.demos.blackhole.glm53_flash.ttnn import head_relax

    mesh = mesh_device
    g = torch.Generator().manual_seed(7)
    weight = (torch.randn(VOCAB, D, generator=g) * 0.05).to(torch.bfloat16)
    head = TtLmHead(mesh, weight)
    head.prepare_relax(-1.0)
    head_relax.prealloc(mesh)
    failures = []
    for users, k in ((1, 7), (4, 3), (4, 7)):
        b = (k + 1) * users
        t, m = b, ROWS * b
        x = put(mesh, torch.randn(ROWS, 1, t, D, generator=g).to(torch.bfloat16), dims=(0, None))
        logits = mesh_logits(head, x).double()
        top_ids = logits.argmax(-1)
        ranked = logits.topk(6, dim=-1).indices
        # each row's next token (the rows of the last level have none); the tokens they come from
        tokens = torch.randint(0, VOCAB, (ROWS, b), generator=g)
        nxt = torch.full((m,), -1, dtype=torch.int64)
        for row in range(m):
            r, v = divmod(row, b)
            if v + users >= b:
                continue
            kind = int(torch.randint(0, 3, (1,), generator=g))
            pick = top_ids[row] if kind == 0 else ranked[row, int(torch.randint(1, 6, (1,), generator=g))]
            pick = int(torch.randint(0, VOCAB, (1,), generator=g)) if kind == 2 else int(pick)
            tokens[r, v + users] = pick
            nxt[row] = pick
        f32 = logits.float()
        gap32 = f32.max(-1).values - f32.gather(1, nxt.clamp(min=0)[:, None])[:, 0]  # (fp32, as the device)
        real = gap32[nxt >= 0]
        mid = real[(real > 0.2) & (real < 2.5)]
        boundary = [float(real[real > 0].min())] + ([float(mid[0])] if mid.numel() else [])
        d_nxt = put(mesh, nxt.float().view(1, 1, m, 1), dims=(None, None), dtype=ttnn.float32)
        d_tok = put(
            mesh, tokens.int().view(1, 1, ROWS, b), dims=(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        for delta in list(GAPS) + boundary:
            head.set_relax(delta)
            a_out, a_raw = head.greedy_mesh(x, relax=(d_nxt, head.relax), keep_greedy=True)
            f_out, f_raw = head.greedy_mesh(x, relax=(d_tok, head.relax, users), keep_greedy=True)
            same = all(
                torch.equal(ids_of(p).view(torch.int32), ids_of(q).view(torch.int32))
                for p, q in ((a_out, f_out), (a_raw, f_raw))
            )
            taken = int((ids_of(f_out).long() != ids_of(f_raw).long()).sum())
            print(
                f"FUSED M {m:3d} gap {delta:.7g}: the fused head = the ttnn form bit for bit {same} (taken {taken})",
                flush=True,
            )
            if not same:
                failures.append((m, delta))
            for p in (a_out, a_raw, f_out, f_raw):
                ttnn.deallocate(p)

        timed = {}
        for name, fn in (
            ("plain", lambda: (head.greedy_mesh(x),)),
            ("ttnn form", lambda: head.greedy_mesh(x, relax=(d_nxt, head.relax), keep_greedy=True)),
            ("fused", lambda: head.greedy_mesh(x, relax=(d_tok, head.relax, users), keep_greedy=True)),
        ):
            for o in fn():
                ttnn.deallocate(o)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                kept = fn()
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if name == "fused":  # the gap changed between replays: the replay = eager
                for delta in (3.0, -1.0):
                    head.set_relax(delta)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    e_out, e_raw = head.greedy_mesh(x, relax=(d_tok, head.relax, users), keep_greedy=True)
                    if not torch.equal(ids_of(e_out).view(torch.int32), ids_of(kept[0]).view(torch.int32)):
                        failures.append((m, "replay", delta))
                    for p in (e_out, e_raw):
                        ttnn.deallocate(p)
            reps = 20
            t0 = time.perf_counter()
            for _ in range(reps):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            timed[name] = (time.perf_counter() - t0) / reps
            ttnn.release_trace(mesh, trace)
            for o in kept:
                ttnn.deallocate(o)
        print(
            f"FUSED M {m:3d}: traced plain {timed['plain'] * 1e6:.1f} us, ttnn form {timed['ttnn form'] * 1e6:.1f} "
            f"(+{(timed['ttnn form'] - timed['plain']) * 1e6:.1f}), fused {timed['fused'] * 1e6:.1f} "
            f"(+{(timed['fused'] - timed['plain']) * 1e6:.1f})",
            flush=True,
        )
        for p in (x, d_nxt, d_tok):
            ttnn.deallocate(p)
    assert not failures, failures
