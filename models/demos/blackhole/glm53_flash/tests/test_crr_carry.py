"""The prefill MoE's combine carried in the way back's op (`col_row_rs.CARRY`, GLM53_CRR_CARRY:
`moe_prefill.combine(carry=)` on the cores ttnn.experimental.col_row_reduce_scatter leaves, the column's
reduce-scatter starting each batch once every combine core has written it) against the combine and the op one after the
other (`col_row_rs.run` on the combine's output): every chip's outputs bit for bit, and the times (traced).  A real plan
(random routing of the column's 8 T tokens), random Y and a random shared expert's partial.  Print only (KCC lines).

    GLM53_CC_T       tokens a row (default 512; the generator's col_row_rs sizes: 512, 1024, 128 with GLM53_CRR_128)
    GLM53_CC_AG      1: the op's row all-gather (`run(.., ag=True)`); 0 (default): the all-to-all after the row
                     reduce-scatter (the model's form at 512 and 1,024: mHC over the row by tokens)
    GLM53_CC_ROUNDS  timing rounds (default 3, 0: the bits alone); GLM53_CC_REPS calls a trace (default 10)
    GLM53_CC_STRESS  calls (default 0: none): a trace of GLM53_CC_REPS carried calls alternating two Y (the last one the
                     second), replayed to this many calls, each replay's output checked on the device against the
                     plain form's of the second Y (KCC stress lines)
    GLM53_CC_STRESS_FORM  carried (default) / plain: the form the stress replays (plain: the combine and the op in
                     turn, each call; to tell which form goes wrong)
    GLM53_CC_DIAG    1: before the timing, carried calls on alternating Y, eager (no trace): each call's combine
                     output and the op's column sum + shared (`total`, per batch) against the plain form's of the same
                     Y, per chip (which batches go wrong; KCC diag lines); GLM53_CC_DIAG_CALLS such calls (default 6;
                     a line for each call with a difference, and a summary).  2: the same calls checked on the device
                     instead (the op's output and `total` against the plain form's, chips differing; fast)
    (GLM53_CRR_COL_WORKERS: as in the model, both forms)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_crr_carry.py -s
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

from models.demos.blackhole.glm53_flash.ttnn import col_row_rs, moe_prefill  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS, put  # noqa: E402

D = 4096
T = int(os.environ.get("GLM53_CC_T", "512"))
AG = os.environ.get("GLM53_CC_AG", "0") == "1"
ROUNDS = int(os.environ.get("GLM53_CC_ROUNDS", "3"))
REPS = int(os.environ.get("GLM53_CC_REPS", "10"))
STRESS = int(os.environ.get("GLM53_CC_STRESS", "0"))
STRESS_FORM = os.environ.get("GLM53_CC_STRESS_FORM", "carried")
K, E_ALL, E_COL, BASE = 8, 288, 72, 144


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_crr_carry(mesh_device):
    try:
        _study(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KCC FAILED: {err!r}"[:3000], flush=True)


def _chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(c).to(torch.bfloat16) for c in ttnn.get_device_tensors(t)]


def _timed(mesh, fn) -> float:
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = []
    try:
        for _ in range(REPS):
            outs.append(fn())
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    s = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return s


def _study(mesh):
    n = ROWS * T
    col_row_rs.prealloc(mesh, [T])
    b = col_row_rs.batches_for(mesh, T)
    assert b > 1, f"no col_row_rs buffers for {T} tokens a row"
    carry = (col_row_rs.busy(mesh, T, AG), col_row_rs.CARRY_RELAY, col_row_rs.CARRY_SEM)
    print(f"KCC T={T} B={b} ag={int(AG)}: workers {col_row_rs._workers(T, AG)}, op cores {len(carry[0])}", flush=True)
    g = torch.Generator().manual_seed(120 + T)
    rep = dict(dims=(None, None), dtype=ttnn.float32)
    d_bias = put(mesh, torch.zeros(1, 1, 1, E_ALL), **rep)
    bases = torch.zeros(1, 1, 1, 8, dtype=torch.int32)
    bases[..., 0] = BASE
    d_base = put(mesh, bases, dims=(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    d_logits = put(mesh, (torch.randn(n, E_ALL, generator=g) * 2.0)[None, None], **rep)
    w = ttnn.experimental.deepseek_prefill.moe_decode_route(d_logits, d_bias, d_base, n, K, E_COL, 2.5)
    seg, ptok, pw, tpair = moe_prefill.plan(mesh, w)
    for z in (seg, ptok, pw, w, d_logits):
        ttnn.deallocate(z)
    m = moe_prefill.table_rows(n)
    ys = [
        put(
            mesh,
            (torch.randn(1, 1, m, D, generator=g) * 0.5).to(torch.bfloat16),
            dims=(None, None),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        for _ in range(2)
    ]
    # the shared expert's partial: a different one on each chip
    part = put(
        mesh, (torch.randn(ROWS, 4, T, D, generator=g) * 0.5).to(torch.bfloat16), dims=(0, 1), dtype=ttnn.bfloat16
    )
    part = ttnn.reshape(part, [1, 1, T, D])

    def plain(y=ys[0]):
        out = moe_prefill.combine(mesh, y, tpair, b)
        r = col_row_rs.run(mesh, out, part, T, ag=AG)
        ttnn.deallocate(out)
        return r

    def carried(y=ys[0]):
        c = moe_prefill.combine(mesh, y, tpair, b, carry=carry)
        r = col_row_rs.run(mesh, c, part, T, ag=AG)
        ttnn.deallocate(c.out)
        return r

    ref = plain()
    want = _chips(ref)
    t0 = time.perf_counter()
    got = carried()
    ttnn.synchronize_device(mesh)
    print(
        f"KCC first carried call {time.perf_counter() - t0:.2f} s (combine cores {len(moe_prefill._grid(mesh)) - len(carry[0]) - 1})",
        flush=True,
    )
    for k in range(2):  # (twice: the second call on the programs from the cache, its address slots bound anew)
        if k:
            ttnn.deallocate(got)
            got = carried()
            ttnn.synchronize_device(mesh)
        d = [int((a.view(torch.int16) != c.view(torch.int16)).sum()) for a, c in zip(_chips(got), want)]
        print(f"KCC bits call {k}: chips differing {sum(1 for v in d if v)} / {len(d)} ({sum(d)} elements)", flush=True)
    ttnn.deallocate(got)

    if os.environ.get("GLM53_CC_DIAG", "0") == "2":  # (the device's compare: the output and total, chips differing)
        st = col_row_rs._state(mesh, T)

        def ndiff(a, c) -> int:
            ne = ttnn.ne(a, c)
            sm = ttnn.sum(ne)
            v = [float(ttnn.to_torch(t).flatten()[0]) for t in ttnn.get_device_tensors(sm)]
            ttnn.deallocate(ne)
            ttnn.deallocate(sm)
            return sum(1 for z in v if round(z))

        refs = []
        for y in ys:  # (the plain form's output and total of each Y, kept on the device)
            o = moe_prefill.combine(mesh, y, tpair, b)
            r = col_row_rs.run(mesh, o, part, T, ag=AG)
            refs.append((r, ttnn.clone(st.total)))
            ttnn.deallocate(o)
        n_diag, bad_calls = int(os.environ.get("GLM53_CC_DIAG_CALLS", "6")), 0
        for k in range(n_diag):
            c = moe_prefill.combine(mesh, ys[k % 2], tpair, b, carry=carry)
            r = col_row_rs.run(mesh, c, part, T, ag=AG)
            dr, dt = ndiff(r, refs[k % 2][0]), ndiff(st.total, refs[k % 2][1])
            if dr or dt:
                bad_calls += 1
                if bad_calls <= 8:
                    print(
                        f"KCC diag call {k} (Y {k % 2}): output chips differing {dr}, total chips differing {dt}",
                        flush=True,
                    )
            ttnn.deallocate(c.out)
            ttnn.deallocate(r)
        print(f"KCC diag summary: {bad_calls} of {n_diag} eager calls with a difference (on the device)", flush=True)
        for z in refs:
            ttnn.deallocate(z[0])
            ttnn.deallocate(z[1])
    if os.environ.get("GLM53_CC_DIAG", "0") == "1":
        st = col_row_rs._state(mesh, T)

        def total_batches():
            return [c.reshape(b, T // b, D) for c in _chips(st.total)]

        refs = []
        for y in ys:  # (the plain form's combine output and total of each Y)
            o = moe_prefill.combine(mesh, y, tpair, b)
            ttnn.deallocate(col_row_rs.run(mesh, o, part, T, ag=AG))
            refs.append((_chips(o), total_batches()))
            ttnn.deallocate(o)
        n_diag, bad_calls = int(os.environ.get("GLM53_CC_DIAG_CALLS", "6")), 0
        for k in range(n_diag):
            y = ys[k % 2]
            c = moe_prefill.combine(mesh, y, tpair, b, carry=carry)
            r = col_row_rs.run(mesh, c, part, T, ag=AG)
            ttnn.synchronize_device(mesh)
            ro, rt = refs[k % 2]
            do = [int((a.view(torch.int16) != w.view(torch.int16)).sum()) for a, w in zip(_chips(c.out), ro)]
            tb = total_batches()
            dt = [
                [int((a[q].view(torch.int16) != w[q].view(torch.int16)).sum()) for q in range(b)]
                for a, w in zip(tb, rt)
            ]
            bad = [(i, v) for i, v in enumerate(dt) if any(v)]
            bad_calls += bool(bad) or any(do)
            if k < 6 or bad or any(do):
                print(f"KCC diag call {k} (Y {k % 2}): combine out chips differing {sum(1 for v in do if v)}; total "
                      f"differing (chip, per batch) {bad[:8]}", flush=True)  # fmt: skip
            ttnn.deallocate(c.out)
            ttnn.deallocate(r)
        print(f"KCC diag summary: {bad_calls} of {n_diag} eager calls with a difference", flush=True)

    def comb():
        return moe_prefill.combine(mesh, ys[0], tpair, b)

    def way_back():
        return col_row_rs.run(mesh, ref_in, part, T, ag=AG)

    ref_in = comb()
    res = {}
    for rnd in range(ROUNDS):
        for name, fn in (("combine", comb), ("op", way_back), ("combine + op", plain), ("carried", carried)):
            s = _timed(mesh, fn)
            res.setdefault(name, []).append(s)
            print(f"KCC time round {rnd} {name}: {s * 1e6:.1f} us", flush=True)
    for name, v in res.items():
        print(
            f"KCC summary T={T} {name}: median {sorted(v)[len(v) // 2] * 1e6:.1f} us ({[round(z * 1e6, 1) for z in v]})"
        )
    if STRESS > 0:
        ref2 = plain(ys[1])
        reps = max(2, REPS - REPS % 2)

        def diffs(a, c) -> int:
            ne = ttnn.ne(a, c)
            sm = ttnn.sum(ne)
            v = [float(ttnn.to_torch(t).flatten()[0]) for t in ttnn.get_device_tensors(sm)]
            ttnn.deallocate(ne)
            ttnn.deallocate(sm)
            return sum(1 for z in v if round(z))

        form = carried if STRESS_FORM == "carried" else plain
        ttnn.deallocate(form(ys[1]))
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = []
        try:
            for k in range(reps):
                outs.append(form(ys[k % 2]))
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        last = outs[-1]
        bad, calls, t0 = 0, 0, time.perf_counter()
        while calls < STRESS:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            calls += reps
            dc = diffs(last, ref2)
            if dc:
                bad += 1
                print(f"KCC stress after {calls} calls: chips differing {dc}")
            if calls % (reps * 100) == 0:
                print(f"KCC stress: {calls} calls, {bad} bad checks, {time.perf_counter() - t0:.1f} s", flush=True)
        ttnn.release_trace(mesh, trace)
        print(
            f"KCC stress T={T} ({STRESS_FORM}): {calls} calls done, {bad} bad checks, {time.perf_counter() - t0:.1f} s"
        )
    print("KCC done", flush=True)
