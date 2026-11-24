"""A KDA prefill's commit (`TtKda._commit`) by the paged copy alone and by `kda_prefill_ops.history_commit` against the
selections they replace, bit for bit on every chip, and both timed in a trace:
  - the delta-rule state: paged_fill_cache of the final state with the table's -1 skipped (a row without a request)
    against the old form (its user 0's own state x 0 + the final state x has, written to user 0 without a request);
  - the conv history: the request's 3 rows (bf16) into its user's row of the 3 history tensors [1, 1, B, 6144] fp32
    against the one-hot form (the row sliced and made fp32, h x (1 - one-hot) + one-hot x row).
Each mesh row has its own user (one row -1: no request).  Cases: B 1 / 4 / 16 / 32 / 64 users a row, bf16 states.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kda_commit.py -s
    GLM53_TEST_STRICT=1: fail on a difference (else printed only: a study's run ends cleanly)
    GLM53_KDA_COMMIT_REPLAYS=N: each 20-call trace replayed N times more (default 0; a stress: 100 and more, then
        checked against the first results)
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
    COLS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    put,
    slice_copy,
)
from models.demos.blackhole.glm53_flash.ttnn.kda_prefill_ops import history_commit  # noqa: E402

HEADS, DH, CHANS, K = 16, 128, 6144, 3
CASES = [1, 4, 16, 32, 64]
REPS, REPLAYS = 20, int(os.environ.get("GLM53_KDA_COMMIT_REPLAYS", "0"))
PER_ROW = dict(dims=(0, None))


def traced(mesh, fn, reps: int = REPS) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for _ in range(reps):
            fn()
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / reps
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ttnn.release_trace(mesh, trace)
    return seconds


def chips_of(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_commit(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(7)
    failures = []
    for users in CASES:
        slots = [int(torch.randint(0, users, (1,), generator=g)) for _ in range(ROWS)]
        slots[0] = users - 1  # the last user (a second tile row from 33 users)
        slots[3] = None  # a row without a request
        new_table = torch.tensor([-1 if s is None else s for s in slots], dtype=torch.int32).view(ROWS, 1)
        old_table = torch.tensor([s or 0 for s in slots], dtype=torch.int32).view(ROWS, 1)
        has = torch.tensor([0.0 if s is None else 1.0 for s in slots]).view(ROWS, 1, 1, 1)
        one_hot = torch.stack(
            [(torch.arange(users) == s).float() if s is not None else torch.zeros(users) for s in slots]
        ).view(ROWS, 1, users, 1)
        rm = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        d_new = put(mesh, new_table, **PER_ROW, **rm)
        d_old = put(mesh, old_table, **PER_ROW, **rm)
        d_has = put(mesh, has, **PER_ROW, dtype=ttnn.float32)
        d_hot = put(mesh, one_hot, **PER_ROW, dtype=ttnn.float32)

        state0 = torch.randn(users, HEADS, DH, DH, generator=g).to(torch.bfloat16)
        final = torch.randn(ROWS, HEADS, DH, DH, generator=g)  # each row's own (fp32)
        hist0 = [torch.randn(1, 1, users, CHANS, generator=g) for _ in range(K)]
        rows = torch.randn(ROWS, K, CHANS, generator=g).to(torch.bfloat16)  # each mesh row's request's
        d_final = put(mesh, final, dims=(0, None), dtype=ttnn.float32)
        d_rows = put(mesh, rows, **PER_ROW, dtype=ttnn.bfloat16)
        states = {a: put(mesh, state0, dims=(None, None), dtype=ttnn.bfloat16) for a in ("op", "ref")}
        hists = {a: [put(mesh, h, dims=(None, None), dtype=ttnn.float32) for h in hist0] for a in ("op", "ref")}

        def op():
            fin = ttnn.reshape(d_final, [1, HEADS, DH, DH])
            src = ttnn.typecast(fin, ttnn.bfloat16)
            ttnn.experimental.paged_fill_cache(states["op"], src, d_new)
            ttnn.deallocate(src)
            history_commit(mesh, d_rows, hists["op"], d_new)

        def ref():
            fin = ttnn.reshape(d_final, [1, HEADS, DH, DH])
            own = slice_copy(states["ref"], [0, 0, 0, 0], [1, HEADS, DH, DH])
            own_b, own = own, ttnn.typecast(own, ttnn.float32)
            ttnn.deallocate(own_b)
            drop = ttnn.rsub(d_has, 1.0)
            kept = ttnn.multiply(fin, d_has)
            new = ttnn.addcmul(kept, own, drop)
            new_b = ttnn.typecast(new, ttnn.bfloat16)
            ttnn.experimental.paged_fill_cache(states["ref"], new_b, d_old)
            for t in (own, drop, kept, new, new_b):
                ttnn.deallocate(t)
            stay = ttnn.rsub(d_hot, 1.0)
            for i, h in enumerate(hists["ref"]):
                row = ttnn.slice(d_rows, [0, i, 0], [1, i + 1, CHANS])
                new = ttnn.reshape(ttnn.typecast(row, ttnn.float32), [1, 1, 1, CHANS])
                ttnn.multiply(h, stay, output_tensor=h)
                ttnn.addcmul(h, d_hot, new, output_tensor=h)
                ttnn.deallocate(row)
                ttnn.deallocate(new)
            ttnn.deallocate(stay)

        op()
        ref()
        same_state = all(torch.equal(a, b) for a, b in zip(chips_of(states["op"]), chips_of(states["ref"])))
        same_hist = all(
            torch.equal(a, b) for i in range(K) for a, b in zip(chips_of(hists["op"][i]), chips_of(hists["ref"][i]))
        )
        # the host's expectation (chip index // COLS: its mesh row)
        exact = True
        got_state, got_hist = chips_of(states["op"]), [chips_of(h) for h in hists["op"]]
        for c in range(ROWS * COLS):
            r, s = c // COLS, slots[c // COLS]
            want = state0.clone()
            if s is not None:
                want[s] = final[r].to(torch.bfloat16)
            exact &= torch.equal(got_state[c].view_as(want), want)
            for i in range(K):
                w = hist0[i].clone()
                if s is not None:
                    w[0, 0, s] = rows[r, i].float()
                exact &= torch.equal(got_hist[i][c].view_as(w), w)
        t_op, t_ref = traced(mesh, op), traced(mesh, ref)
        after = all(torch.equal(a, b) for a, b in zip(chips_of(states["op"]), got_state)) and all(
            torch.equal(a, b) for i in range(K) for a, b in zip(chips_of(hists["op"][i]), got_hist[i])
        )
        print(
            f"B {users:3d}: state op = old form bit for bit {same_state}, history {same_hist}, op = host {exact}; "
            f"traced (state + history) {t_ref * 1e6:6.1f} -> {t_op * 1e6:6.1f} us; "
            f"after {(REPLAYS + 2) * REPS + 1} calls the same {after}",
            flush=True,
        )
        if not (same_state and same_hist and exact and after):
            failures.append(users)
        for t in [d_new, d_old, d_has, d_hot, d_final, d_rows, *states.values(), *hists["op"], *hists["ref"]]:
            ttnn.deallocate(t)
    print(f"failures: {failures}", flush=True)
    if os.environ.get("GLM53_TEST_STRICT") == "1":
        assert not failures, failures
