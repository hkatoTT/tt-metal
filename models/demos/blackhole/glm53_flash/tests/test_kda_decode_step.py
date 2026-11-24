"""The KDA decode delta rule (`ttnn.experimental.kda.recurrent_decode_step`) against a float64 torch reference: the
same random inputs for GLM53_KDA_STEPS steps (default 64), the outputs and the state after the steps against the
reference (the state is carried from step to step, so rounding that reaches it accumulates; the state kept in bf16),
and the time per step (trace-timed).  No checkpoint.

GLM53_KDA_PAIRS: the (user, head) pairs per chip (default 16,64,256: 8 / 32 / 128 users, 16 heads per chip).
Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kda_decode_step.py -s
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

from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402

HEADS, DH = 16, 128
# the relative error (norm) against float64 after the steps (the state rounded to bf16 once a step; with fp32 states
# it was 3e-3)
LIMIT = 2e-2
# the model's plain step form (kda.PLAIN_FORM)
PLAIN_FORM = 384 | 1024


def reference_step(s, q, k, v, e, beta):
    """float64: s [P, dh, dh]; q, k, v, e [P, dh]; beta [P] -> (o [P, dh], s')."""
    s = s * e[:, :, None]
    delta = (v - torch.einsum("pk,pkv->pv", k, s)) * beta[:, None]
    s = s + k[:, :, None] * delta[:, None, :]
    return torch.einsum("pk,pkv->pv", q, s), s


def inputs(g, pairs):
    q = torch.nn.functional.normalize(torch.randn(pairs, DH, generator=g), dim=-1) / DH**0.5
    k = torch.nn.functional.normalize(torch.randn(pairs, DH, generator=g), dim=-1)
    v = torch.randn(pairs, DH, generator=g)
    e = torch.exp(-5 * torch.sigmoid(torch.randn(pairs, DH, generator=g) - 4))  # decays near 1, as most channels
    beta = torch.sigmoid(torch.randn(pairs, generator=g))
    return q, k, v, e, beta


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_decode_step(mesh_device):
    steps = int(os.environ.get("GLM53_KDA_STEPS", "64"))
    f32 = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()  # noqa: E731
    for pairs in (int(n) for n in os.environ.get("GLM53_KDA_PAIRS", "16,64,256").split(",")):
        b = pairs // HEADS
        g = torch.Generator().manual_seed(pairs)
        s0 = torch.randn(pairs, DH, DH, generator=g) * 0.1
        seq = [inputs(g, pairs) for _ in range(steps)]

        ref_s = s0.double()
        ref_o = None
        for q, k, v, e, beta in seq:
            ref_o, ref_s = reference_step(ref_s, *(x.double() for x in (q, k, v, e, beta)))

        def state():
            src = s0.reshape(b, HEADS, DH, DH)
            return ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        def run():
            s = state()
            o = None
            for q, k, v, e, beta in seq:
                dq, dk, dv, de = (ttnn.from_torch(x[None, None], **f32) for x in (q, k, v, e))
                db = ttnn.from_torch(beta.reshape(1, 1, pairs, 1), **f32)
                if o is not None:
                    ttnn.deallocate(o)
                o = ttnn.experimental.kda.recurrent_decode_step(s, dq, dk, dv, de, db, form=PLAIN_FORM)
                for t in (dq, dk, dv, de, db):
                    ttnn.deallocate(t)
            out = first(o).reshape(pairs, DH), first(s).reshape(pairs, DH, DH)
            ttnn.deallocate(o)
            ttnn.deallocate(s)
            return out

        def timed():
            s = state()
            q, k, v, e, beta = seq[0]
            dq, dk, dv, de = (ttnn.from_torch(x[None, None], **f32) for x in (q, k, v, e))
            db = ttnn.from_torch(beta.reshape(1, 1, pairs, 1), **f32)

            def step():
                return ttnn.experimental.kda.recurrent_decode_step(s, dq, dk, dv, de, db, form=PLAIN_FORM)

            ttnn.deallocate(step())
            ttnn.synchronize_device(mesh_device)
            reps = 20
            with trace_capture(mesh_device) as trace:
                outs = [step() for _ in range(reps)]
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            seconds = (time.perf_counter() - t0) / reps
            ttnn.release_trace(mesh_device, trace)
            for t in outs + [s, dq, dk, dv, de, db]:
                ttnn.deallocate(t)
            return seconds

        o, s = run()
        rel = lambda a, r: ((a - r).norm() / r.norm()).item()  # noqa: E731
        err = (rel(o, ref_o), rel(s, ref_s))
        print(
            f"{pairs} pairs, bf16 state: after {steps} steps, relative error vs float64: output {err[0]:.2e}, "
            f"state {err[1]:.2e}; {timed() * 1e6:.1f} us per step",
            flush=True,
        )
        assert max(err) <= LIMIT, err


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_step_forms(mesh_device):
    """recurrent_decode_step's plain step forms (`form`, bits: 128 the value blocks rotated by core, 256 the top rows
    read on NoC 1, 1024 the units by grid row speed) against the first form bit for bit (o, the state) on the same
    inputs, and each form's time (trace).  GLM53_KDA_FORM_CASES: pairs (default 256,1024: 128 / 512 users' plain step),
    GLM53_KDA_FORM_LIST the forms (default 0,128,384,1408; 1408 the model's).  With GLM53_KDA_ZONES=1 each form runs
    once, untimed (a kernel zone profile)."""
    f32 = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    st = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    zones = os.environ.get("GLM53_KDA_ZONES") == "1"
    bad = []
    for pairs in (int(x) for x in os.environ.get("GLM53_KDA_FORM_CASES", "256,1024").split(",")):
        b = pairs // HEADS
        g = torch.Generator().manual_seed(900 + pairs + 1)
        s0 = torch.randn(pairs, DH, DH, generator=g) * 0.1
        q, k, v, e, beta = inputs(g, pairs)
        forms = [int(f) for f in os.environ.get("GLM53_KDA_FORM_LIST", "0,128,384,1408").split(",")]
        results = {}
        for form in forms:
            s = ttnn.from_torch(s0.reshape(b, HEADS, DH, DH), **st)
            dq, dk, dv, de = (ttnn.from_torch(x[None, None], **f32) for x in (q, k, v, e))
            db = ttnn.from_torch(beta.reshape(1, 1, pairs, 1), **f32)

            def step():
                return ttnn.experimental.kda.recurrent_decode_step(s, dq, dk, dv, de, db, form=form)

            o = step()
            ttnn.synchronize_device(mesh_device)
            results[form] = [first(o), first(s)]
            ttnn.deallocate(o)
            seconds = None
            if not zones:
                reps = 20
                with trace_capture(mesh_device) as trace:
                    outs = [step() for _ in range(reps)]
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                seconds = (time.perf_counter() - t0) / reps
                ttnn.release_trace(mesh_device, trace)
                for t in outs:
                    ttnn.deallocate(t)
            for t in [s, dq, dk, dv, de, db]:
                ttnn.deallocate(t)
            same = all(torch.equal(x, y) for x, y in zip(results[form], results[forms[0]]))
            if not same:
                bad.append((pairs, form))
            t_us = f"{seconds * 1e6:.1f} us" if seconds is not None else "(untimed)"
            print(f"{pairs} pairs, form {form}: {t_us}, the same as form {forms[0]}: {same}", flush=True)
    assert not bad, f"forms that differ: {bad}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_step_chunk(mesh_device):
    """Runs `_test_kda_step_chunk` (its docstring), an exception printed and the test ending normally (a
    study that reports, run among others)."""
    try:
        _test_kda_step_chunk(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KDA_STEP_CHUNK FAILED: {err!r}"[:3000], flush=True)


def _test_kda_step_chunk(mesh_device):
    """The chunked verify form (`saved`) over GLM53_KDA_CHUNK_STEPS steps (default 6) of random
    keep codes (all K at the first: the fresh rows do nothing), against float64 token by token with the verify step's
    keep semantics (start from the state after the last step's token c, c < K, or after all of them): o, and the state
    the chunked form leaves (the start of the step, after the redone tokens) against the start.  Then its time (trace,
    the median of 3 runs of 20).
    KCHUNK lines, no assert.  GLM53_KDA_PAIRS (default 16,64,256), GLM53_KDA_VERIFY_T (default 2,4,8),
    GLM53_KDA_CHUNK_PM (default 0,1: q .. beta by token or by pair)."""
    from models.demos.blackhole.glm53_flash.ttnn.kda import chunk_saved_host

    f32 = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    st = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()  # noqa: E731
    rel = lambda a, r: ((a - r).norm() / r.norm()).item()  # noqa: E731
    steps = int(os.environ.get("GLM53_KDA_CHUNK_STEPS", "6"))
    configs = [
        (pairs, n, pm)
        for pm in (int(x) for x in os.environ.get("GLM53_KDA_CHUNK_PM", "0,1").split(","))
        for pairs in (int(x) for x in os.environ.get("GLM53_KDA_PAIRS", "16,64,256").split(","))
        for n in (int(x) for x in os.environ.get("GLM53_KDA_VERIFY_T", "2,4,8").split(","))
    ]
    for pairs, n, pm in configs:
        if True:
            b = pairs // HEADS
            K = n - 1
            g = torch.Generator().manual_seed(900 + pairs + n)
            # the chunked form's rows: by token (row p + k P) or by pair (row p T + k, pm: the model's)
            order = (
                lambda x: x.reshape(n, pairs, *x.shape[1:]).transpose(0, 1).reshape(x.shape) if pm else x
            )  # noqa: E731
            s0 = (torch.randn(pairs, DH, DH, generator=g) * 0.1).bfloat16().double()
            dev = lambda x: ttnn.from_torch(x.float().reshape(-1, HEADS, DH, DH), **st)  # noqa: E731
            s_c = dev(s0)
            saved = ttnn.from_torch(chunk_saved_host(b, HEADS, n), **f32)
            ref_full, ref_ck = s0.clone(), [s0.clone() for _ in range(K)]
            worst = {"chunk o": 0.0, "chunk start": 0.0}
            try:
                for step in range(steps):
                    seq = [inputs(g, pairs) for _ in range(n)]
                    if step == 0:
                        codes = torch.full((b,), K, dtype=torch.int32)
                    else:
                        codes = torch.randint(0, K + 1, (b,), generator=g, dtype=torch.int32)
                    pick = codes.repeat_interleave(HEADS)
                    stack = torch.stack(ref_ck) if K else ref_full[None]
                    start = torch.where(
                        (pick < K)[:, None, None], stack[pick.clamp(max=max(K - 1, 0)), torch.arange(pairs)], ref_full
                    )
                    s, ref_o, new_ck = start, [], []
                    for q, k, v, e, beta in seq:
                        o, s = reference_step(s, *(x.double() for x in (q, k, v, e, beta)))
                        ref_o.append(o)
                        new_ck.append(s)
                    rows = lambda i: torch.cat([seq[j][i] for j in range(n)], dim=0)  # noqa: E731
                    pq, pk, pv, pe = (ttnn.from_torch(order(rows(i))[None, None], **f32) for i in range(4))
                    pb = ttnn.from_torch(order(rows(4)).reshape(1, 1, n * pairs, 1), **f32)
                    flags = ttnn.from_torch(
                        codes[None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
                    )
                    o_c = ttnn.experimental.kda.recurrent_decode_step(
                        s_c, pq, pk, pv, pe, pb, keep=flags, heads=HEADS, tokens=n, saved=saved, pair_major=bool(pm),
                    )  # fmt: skip
                    for t in (pq, pk, pv, pe, pb):
                        ttnn.deallocate(t)
                    want_o = torch.stack(ref_o).reshape(n * pairs, DH)
                    got = {
                        "chunk o": rel(first(o_c).reshape(n * pairs, DH), want_o),
                        "chunk start": rel(first(s_c).reshape(pairs, DH, DH), start),
                    }
                    for key, val in got.items():
                        worst[key] = max(worst[key], val)
                    print(f"KCHUNK pm={pm} pairs={pairs} T={n} step {step} codes {codes[:8].tolist()}: "
                          + ", ".join(f"{k_} {v_:.2e}" for k_, v_ in got.items()), flush=True)  # fmt: skip
                    ref_full, ref_ck = new_ck[-1], new_ck[:K]
                    for t in (flags, o_c):
                        ttnn.deallocate(t)
            except Exception as err:  # (a failed op is printed, the run goes on)
                print(f"KCHUNK pm={pm} pairs={pairs} T={n} FAILED: {err!r}"[:2000], flush=True)
                return
            print(f"KCHUNK worst pm={pm} pairs={pairs} T={n}: " + ", ".join(f"{k_} {v_:.2e}" for k_, v_ in worst.items()),
                  flush=True)  # fmt: skip
            if os.environ.get("GLM53_KDA_CHUNK_TIME", "1") != "1":  # (a profile of the steps alone)
                continue
            # the times (the values do not matter here)
            seq = [inputs(g, pairs) for _ in range(n)]
            rows = lambda i: torch.cat([seq[j][i] for j in range(n)], dim=0)  # noqa: E731
            pq, pk, pv, pe = (ttnn.from_torch(order(rows(i))[None, None], **f32) for i in range(4))
            pb = ttnn.from_torch(order(rows(4)).reshape(1, 1, n * pairs, 1), **f32)
            codes = torch.randint(0, K + 1, (b,), generator=g, dtype=torch.int32)
            flags = ttnn.from_torch(codes[None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
            times = {}
            for name, kw in (("chunk", dict(saved=saved, pair_major=bool(pm))),):
                state = s_c
                ins = (pq, pk, pv, pe, pb)
                call = lambda: ttnn.experimental.kda.recurrent_decode_step(  # noqa: E731
                    state, *ins, keep=flags, heads=HEADS, tokens=n, **kw
                )
                ttnn.deallocate(call())
                ttnn.synchronize_device(mesh_device)
                reps = 20
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                try:
                    outs = [call() for _ in range(reps)]
                finally:
                    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) / reps * 1e6)
                ttnn.release_trace(mesh_device, trace)
                for t in outs:
                    ttnn.deallocate(t)
                times[name] = sorted(ts)[1]
            print(f"KCHUNK time pm={pm} pairs={pairs} T={n}: chunk {times['chunk']:.1f} us", flush=True)
            for t in (pq, pk, pv, pe, pb, flags, s_c, saved):
                ttnn.deallocate(t)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_chunk_commit_rows(mesh_device):
    """Runs `_test_kda_chunk_commit_rows` (its docstring), an exception printed and the test ending normally (a
    study that reports, run among others)."""
    try:
        _test_kda_chunk_commit_rows(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KDA_CHUNK_COMMIT_ROWS FAILED: {err!r}"[:3000], flush=True)


def _test_kda_chunk_commit_rows(mesh_device):
    """The prefill commit's reset of a user's saved rows in the chunked verify form (`TtKda._commit_rows`' ops,
    the chunked verify form): random rows [B, 1, R, 32] fp32 TILE, then the paged copy of the rows that do nothing
    (`chunk_saved_host`) into user u (`paged_fill_cache`, the table's u; -1: nothing), and the one-hot form (x (1 - h) +
    h x the rows), each user's rows against the expected, bit for bit; and `ttnn.repeat` of one user's rows over B (the
    fresh state's).  KROWS lines, no assert."""
    from models.demos.blackhole.glm53_flash.ttnn.kda import chunk_saved_host

    f32 = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    b = 4
    for n in (4, 8):
        inert = chunk_saved_host(1, HEADS, n)
        g = torch.Generator().manual_seed(n)
        host = torch.randn(b, *inert.shape[1:], generator=g)
        d_inert = ttnn.from_torch(inert, **f32)
        rep = ttnn.repeat(d_inert, ttnn.Shape([b, 1, 1, 1]))
        same = bool((first(rep) == inert.expand(b, -1, -1, -1)).all())
        print(f"KROWS T={n} rows {list(inert.shape)}: repeat over {b} users the same: {same}", flush=True)
        ttnn.deallocate(rep)
        for u in (2, -1):
            rows = ttnn.from_torch(host, **f32)
            table = ttnn.from_torch(
                torch.tensor([[u]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            )  # fmt: skip
            ttnn.experimental.paged_fill_cache(rows, d_inert, table)
            want = host.clone()
            if u >= 0:
                want[u] = inert[0]
            got = first(rows)
            print(f"KROWS T={n} paged into user {u}: per user the same {[(got[i] == want[i]).all().item() for i in range(b)]}",
                  flush=True)  # fmt: skip
            for t in (rows, table):
                ttnn.deallocate(t)
        rows = ttnn.from_torch(host, **f32)
        oh = torch.zeros(b, 1, 1, 1)
        oh[1] = 1.0
        d_oh = ttnn.from_torch(oh, **f32)
        others = ttnn.rsub(d_oh, 1.0)
        ttnn.multiply(rows, others, output_tensor=rows)
        ttnn.addcmul(rows, d_oh, d_inert, output_tensor=rows)
        want = host.clone()
        want[1] = inert[0]
        got = first(rows)
        print(f"KROWS T={n} one-hot user 1: per user the same {[(got[i] == want[i]).all().item() for i in range(b)]}",
              flush=True)  # fmt: skip
        for t in (rows, d_oh, others, d_inert):
            ttnn.deallocate(t)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_verify_chain(mesh_device):
    """Runs `_test_kda_verify_chain` (its docstring), an exception printed and the test ending normally (a
    study that reports, run among others)."""
    try:
        _test_kda_verify_chain(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KDA_VERIFY_CHAIN FAILED: {err!r}"[:3000], flush=True)


def _test_kda_verify_chain(mesh_device):
    """The KDA verify step's three device ops on random inputs, as `TtKda._step_verify` calls them (the chunked form,
    decode_prepare's pair-major rows): decode_prepare, recurrent_decode_step, decode_finish, each alone and the three in
    a row, timed from a trace (the median of 3 replays of 20 calls; the states advance call to call, the values do not
    matter).  GLM53_KDA_CHAIN_CASES users-a-row:T (default 1:8,4:8,16:4: 8 / 32 / 128 users).  GLM53_KDA_CHAIN_WHERE
    (default dram,l1): the hand-offs (prepare's outputs, the recurrence's o) in DRAM or L1 interleaved; after the first,
    whether the first call's o, the finish's output and the state are the same bits as the first where's.
    GLM53_KDA_ZONES=1: instead 2 eager runs of the three (the kernels' zones).  KCHAIN lines, no assert."""
    from models.demos.blackhole.glm53_flash.ttnn.kda import chunk_saved_host

    H, CH = HEADS, 3 * HEADS * DH
    lower, scale, eps = -5.0, 1.0 / DH**0.5, 1e-6
    up = lambda t, dtype: ttnn.from_torch(
        t[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )  # noqa: E731
    cases = [
        tuple(int(x) for x in c.split(":")) for c in os.environ.get("GLM53_KDA_CHAIN_CASES", "1:8,4:8,16:4").split(",")
    ]
    wheres = os.environ.get("GLM53_KDA_CHAIN_WHERE", "dram,l1").split(",")
    firsts = {}  # (users, T) -> {where: (o, the finish's output)} of the first call
    for (b, n), where in [(c, w) for c in cases for w in wheres]:
        g = torch.Generator().manual_seed(500 + 10 * b + n)
        k = n - 1
        p = up(torch.randn(n * b, 6432, generator=g).to(torch.bfloat16), ttnn.bfloat16)
        f = up(torch.randn(n * b, 2 * H * DH, generator=g).to(torch.bfloat16), ttnn.bfloat16)
        hn = [up(torch.randn(b, CH, generator=g), ttnn.float32) for _ in range(3)]
        sv = ttnn.from_torch(torch.randn(1, k, b, CH, generator=g), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                             device=mesh_device)  # fmt: skip
        tp = up(torch.randn(4, CH, generator=g) * 0.5, ttnn.float32)
        gc = up(torch.stack([torch.randn(H * DH, generator=g), torch.rand(H * DH, generator=g) + 0.5]), ttnn.float32)
        w = up(torch.rand(1, DH, generator=g) + 0.5, ttnn.float32)
        state = ttnn.from_torch((torch.randn(b, H, DH, DH, generator=g) * 0.1), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh_device)  # fmt: skip
        rows = ttnn.from_torch(
            chunk_saved_host(b, H, n), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        codes = torch.randint(0, n, (b,), generator=g, dtype=torch.int32)
        flags = ttnn.from_torch(codes[None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        l1 = where.startswith("l1")
        prep = lambda: ttnn.experimental.kda.decode_prepare(  # noqa: E731
            p, f, *hn, tp, gc, H, lower, scale, users=b, keep=flags, saved=sv, tokens=n, pair_major=True, l1_outputs=l1
        )
        o_shape = ttnn.Shape([1, 1, n * b * H, DH])
        new_o = lambda: (ttnn.allocate_tensor_on_device(o_shape, ttnn.float32, ttnn.TILE_LAYOUT, mesh_device,  # noqa: E731
                                                       ttnn.L1_MEMORY_CONFIG) if l1 else None)  # fmt: skip

        def rec_into(m, out=None):
            kw = {} if out is None else {"output": out}
            return ttnn.experimental.kda.recurrent_decode_step(state, *m, keep=flags, heads=H, tokens=n, saved=rows,
                                                               pair_major=True, **kw)  # fmt: skip

        mid = prep()
        o = rec_into(mid, new_o())
        fin = lambda: ttnn.experimental.kda.decode_finish(o, f, w, H, eps)  # noqa: E731
        flat = fin()
        firsts.setdefault((b, n), {})[where] = tuple(
            ttnn.to_torch(ttnn.get_device_tensors(t)[0]) for t in (o, flat, state)
        )
        ttnn.deallocate(flat)
        if len(firsts[(b, n)]) >= 2:
            w0, (o_a, f_a, s_a) = next(iter(firsts[(b, n)].items()))
            o_b, f_b, s_b = firsts[(b, n)][where]
            print(f"KCHAIN users={8 * b} T={n}: the first call's o / finish output / state the same bits in {w0} and "
                  f"{where}: {torch.equal(o_a, o_b)} / {torch.equal(f_a, f_b)} / {torch.equal(s_a, s_b)}", flush=True)  # fmt: skip
        o_keep = new_o()
        # (into o_keep: the op returns a new handle on it, not to be deallocated)
        rec = lambda: (rec_into(mid, o_keep), [])[1] if l1 else [rec_into(mid)]  # noqa: E731

        def chain():
            m = prep()
            oo = rec_into(m, new_o())
            for t in m:
                ttnn.deallocate(t)
            out = ttnn.experimental.kda.decode_finish(oo, f, w, H, eps)
            ttnn.deallocate(oo)
            return [out]

        if os.environ.get("GLM53_KDA_ZONES") == "1":
            for _ in range(2):
                for t in chain():
                    ttnn.deallocate(t)
                ttnn.synchronize_device(mesh_device)
            print(f"KCHAIN users={8 * b} T={n}: zone runs done", flush=True)
        else:
            times = {}
            for name, call in (("prepare", lambda: list(prep())), ("recurrence", rec),
                               ("finish", lambda: [fin()]), ("chain", chain)):  # fmt: skip
                call_out = call()
                for t in call_out:
                    ttnn.deallocate(t)
                ttnn.synchronize_device(mesh_device)
                reps = 20
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                try:  # (each call's outputs freed in turn: 20 calls' L1 outputs at once do not fit)
                    for _ in range(reps):
                        for t in call():
                            ttnn.deallocate(t)
                finally:
                    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) / reps * 1e6)
                ttnn.release_trace(mesh_device, trace)
                times[name] = sorted(ts)[1]
            print(f"KCHAIN {where} users={8 * b} T={n}: " + ", ".join(f"{k_} {v_:.1f} us" for k_, v_ in times.items())
                  + f" (the three alone {times['prepare'] + times['recurrence'] + times['finish']:.1f})", flush=True)  # fmt: skip
        for t in [*mid, o, p, f, *hn, sv, tp, gc, w, state, rows, flags] + ([o_keep] if o_keep is not None else []):
            ttnn.deallocate(t)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_prepare_split(mesh_device):
    """Runs `_test_kda_prepare_split` (its docstring), an exception printed and the test ending normally."""
    try:
        _test_kda_prepare_split(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KSPLIT FAILED: {err!r}"[:3000], flush=True)


def _test_kda_prepare_split(mesh_device):
    """decode_prepare's `split_channels` (over twice the cores: q / k / v's units by channel halves) against the unsplit op on the same inputs and
    fresh copies of the history and its checkpoint rows: q, k, v, the decay, beta, the history and the checkpoint rows
    bit for bit, each user's keep code at random; then both ops' times (trace).  GLM53_KDA_SPLIT_CASES users-a-row:T
    (default 1:8,4:8,16:4,1:2,16:2,4:4).  KSPLIT lines, no assert."""
    H, CH = HEADS, 3 * HEADS * DH
    lower, scale = -5.0, 1.0 / DH**0.5
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    up = lambda t, dtype: ttnn.from_torch(
        t[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )  # noqa: E731
    cases = [tuple(int(x) for x in c.split(":"))
             for c in os.environ.get("GLM53_KDA_SPLIT_CASES", "1:8,4:8,16:4,1:2,16:2,4:4").split(",")]  # fmt: skip
    for b, n in cases:
        g = torch.Generator().manual_seed(700 + 10 * b + n)
        k = n - 1
        proj = torch.randn(n * b, 6432, generator=g).to(torch.bfloat16)
        fg = torch.randn(n * b, 2 * H * DH, generator=g).to(torch.bfloat16)
        hist = [torch.randn(b, CH, generator=g) for _ in range(3)]
        saved = torch.randn(1, max(k, 1), b, CH, generator=g)
        taps = torch.randn(4, CH, generator=g) * 0.5
        gate = torch.stack([torch.randn(H * DH, generator=g), torch.rand(H * DH, generator=g) + 0.5])
        codes = torch.randint(0, n, (b,), generator=g, dtype=torch.int32)
        p, f = up(proj, ttnn.bfloat16), up(fg, ttnn.bfloat16)
        tp, gc = up(taps, ttnn.float32), up(gate, ttnn.float32)
        flags = ttnn.from_torch(codes[None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        got, times = {}, {}
        for split in (False, True):
            hn = [up(x, ttnn.float32) for x in hist]
            sv = ttnn.from_torch(saved, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
            call = lambda: ttnn.experimental.kda.decode_prepare(  # noqa: E731
                p, f, *hn, tp, gc, H, lower, scale, users=b, keep=flags, saved=sv, tokens=n, pair_major=True,
                split_channels=split,
            )  # fmt: skip
            outs = call()
            got[split] = [first(t) for t in outs] + [first(t) for t in hn] + [first(sv)]
            for t in outs:
                ttnn.deallocate(t)
            reps = 20
            ttnn.synchronize_device(mesh_device)
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:
                many = [call() for _ in range(reps)]
            finally:
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                ts.append((time.perf_counter() - t0) / reps * 1e6)
            ttnn.release_trace(mesh_device, trace)
            for out in many:
                for t in out:
                    ttnn.deallocate(t)
            times[split] = sorted(ts)[1]
            for t in [*hn, sv]:
                ttnn.deallocate(t)
        names = ["q", "k", "v", "decay", "beta", "history0", "history1", "history2", "saved"]
        same = {nm: bool(torch.equal(a, c)) for nm, a, c in zip(names, got[False], got[True])}
        print(f"KSPLIT users={8 * b} T={n}: the same bits {all(same.values())} {[nm for nm, v in same.items() if not v]}, "
              f"time {times[False]:.1f} -> {times[True]:.1f} us", flush=True)  # fmt: skip
        for t in (p, f, tp, gc, flags):
            ttnn.deallocate(t)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_finish_forms(mesh_device):
    """Runs `_test_kda_finish_forms` (its docstring), an exception printed and the test ending normally."""
    try:
        _test_kda_finish_forms(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KDA_FINISH_FORMS FAILED: {err!r}"[:3000], flush=True)


def _test_kda_finish_forms(mesh_device):
    """decode_finish (pieces of whole 4-row groups over the cores, two compute stages) on random
    inputs against torch (o rounded to bf16, the RMS norm, o_norm, sigmoid(gate), rounded to bf16), and its time from a
    trace (the median of 3 replays of 20 calls).  GLM53_KDA_FINISH_ROWS (default 1,2,4,8,16,32,48,64,128: o_proj's input
    rows, as the decode steps' and the verify steps' rows a chip: 8 / 32 / 128 users' plain decode 1 / 4 / 16, MTP 2 /
    8 / 32, the verify T = 8 at 8 / 32 users 8 / 32, T = 4 at 128 users 64, 1,024 users 128), o in L1 for at most 64
    rows (as the chunked verify form's hand-off) else DRAM.  KFINISH lines, no assert."""
    H, eps = HEADS, 1e-6
    rows_list = [int(x) for x in os.environ.get("GLM53_KDA_FINISH_ROWS", "1,2,4,8,16,32,48,64,128").split(",")]
    for rows in rows_list:
        g = torch.Generator().manual_seed(900 + rows)
        mem = ttnn.L1_MEMORY_CONFIG if rows <= 64 else ttnn.DRAM_MEMORY_CONFIG
        o_h = torch.randn(1, 1, rows * H, DH, generator=g)
        f_h = (torch.randn(1, 1, rows, 2 * H * DH, generator=g) * 2).to(torch.bfloat16)
        w_h = torch.rand(1, 1, 1, DH, generator=g) + 0.5
        o = ttnn.from_torch(o_h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mem)
        f = ttnn.from_torch(f_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        w = ttnn.from_torch(w_h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        ob = o_h.reshape(rows * H, DH).to(torch.bfloat16).float()
        gate = torch.sigmoid(f_h.reshape(rows, 2 * H * DH)[:, H * DH :].float().reshape(rows * H, DH))
        want = (ob * torch.rsqrt((ob * ob).mean(-1, keepdim=True) + eps) * w_h.reshape(DH) * gate).to(torch.bfloat16)
        want = want.reshape(rows, H * DH).float()
        call = lambda: ttnn.experimental.kda.decode_finish(o, f, w, H, eps)  # noqa: E731
        out = call()
        got = [ttnn.to_torch(t).reshape(rows, H * DH).float() for t in ttnn.get_device_tensors(out)]
        ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)
        reps = 20
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            for _ in range(reps):
                ttnn.deallocate(call())
        finally:
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) / reps * 1e6)
        ttnn.release_trace(mesh_device, trace)
        alike = sum(bool(torch.equal(x, got[0])) for x in got)
        diff = (got[0] - want).abs()
        print(f"KFINISH rows={rows}: chips alike {alike} / {len(got)}, against torch max diff {diff.max().item():.3g} "
              f"(equal {torch.equal(got[0], want)}), finite {bool(torch.isfinite(got[0]).all())}, "
              f"time {sorted(ts)[1]:.1f} us", flush=True)  # fmt: skip
        for t in (o, f, w):
            ttnn.deallocate(t)
