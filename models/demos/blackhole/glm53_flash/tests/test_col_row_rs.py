"""ttnn.experimental.col_row_reduce_scatter (the column's ring reduce-scatter, one
traversal a batch, fused with the row's ring reduce-scatter of its output in one program) against ttnn's two
reduce-scatters one after the other (the MoE's way back: the column over its 8 chips, dim 2; then the row's
all-reduce's reduce-scatter over its 4 chips, dim 3): every chip's outputs bit for bit, and the times (traced).  Random
bf16 partials, a different one on each chip.  Print only (KCRR lines).

    GLM53_CRR_T       tokens a row (default 1024)
    GLM53_CRR_B       batches (default 4)
    GLM53_CRR_ROUNDS  timing rounds (default 3, 0: the bits alone); GLM53_CRR_REPS calls a trace (default 10)
    GLM53_CRR_ROW_OFF / GLM53_CRR_RELAY  "x,y": the row reduce-scatter's core grid offset (default 0,4) and the relay
                      core (default 11,9)
    GLM53_CRR_ADD     1 (default): with the shared expert's partial added between the two (ttnn.add in the reference,
                      the op's add stage in the fused form); 0: without
    GLM53_CRR_AG      1 (default): the row's all-gather after the row reduce-scatter too (ttnn.all_gather in the
                      reference, the op's all-gather stage in the fused form)
    GLM53_CRR_STRESS  calls (default 0: none): a trace of GLM53_CRR_REPS fused calls alternating two inputs (the last
                      one the second), replayed to this many calls, each replay's outputs checked on the device
                      against ttnn's reference of the second input (KCRR stress lines)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_col_row_rs.py -s
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
    ROW_AXIS,
    ROWS,
    TP_AXIS,
    put,
)

D = 4096
T = int(os.environ.get("GLM53_CRR_T", "1024"))
B = int(os.environ.get("GLM53_CRR_B", "4"))
ROUNDS = int(os.environ.get("GLM53_CRR_ROUNDS", "3"))
ADD = os.environ.get("GLM53_CRR_ADD", "1") == "1"
AG = os.environ.get("GLM53_CRR_AG", "1") == "1"  # the row's all-gather in the op too (link 1; the row RS on link 0)
REPS = int(os.environ.get("GLM53_CRR_REPS", "10"))
STRESS = int(os.environ.get("GLM53_CRR_STRESS", "0"))


def _xy(name: str, default: str):
    x, y = (int(v) for v in os.environ.get(name, default).split(","))
    return ttnn.CoreCoord(x, y)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_col_row_rs(mesh_device):
    try:
        _study(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KCRR FAILED: {err!r}"[:3000], flush=True)


def _alloc(mesh, shape):
    spec = ttnn.TensorSpec(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
    return ttnn.allocate_tensor_on_device(spec, mesh)


def _chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(c).to(torch.bfloat16) for c in ttnn.get_device_tensors(t)]


def _timed(mesh, fn) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for _ in range(REPS):
            fn()
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    s = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    return s


def _study(mesh):
    assert ROWS == 8 and COLS == 4 and T % (32 * B) == 0
    row_off, relay = _xy("GLM53_CRR_ROW_OFF", "0,4"), _xy("GLM53_CRR_RELAY", "11,9")
    print(f"KCRR T={T} B={B}, row offset {row_off}, relay {relay}", flush=True)
    g = torch.Generator().manual_seed(110 + T)
    n = ROWS * T
    host = torch.randn(ROWS, COLS, n, D, generator=g) * torch.rand(ROWS, COLS, 1, D, generator=g).mul(4).exp2()
    host = host.to(torch.bfloat16)
    # the shared expert's partial: [ROWS, COLS, T, D], a different one on each chip
    sh = (torch.randn(ROWS, COLS, T, D, generator=g) * 4).to(torch.bfloat16)
    # the reference: ttnn's two reduce-scatters on the one-batch layout (and ttnn.add between them)
    x1 = put(mesh, host, dims=(0, 1), dtype=ttnn.bfloat16)  # chip (r, c): [1, 1, 8 T, D]
    s1 = put(mesh, sh, dims=(0, 1), dtype=ttnn.bfloat16)  # chip (r, c): [1, 1, T, D]

    def separate(x1=x1, s1=s1):
        col = ttnn.reduce_scatter(x1, dim=2, cluster_axis=ROW_AXIS)
        if ADD:
            tot = ttnn.add(col, s1)
            ttnn.deallocate(col)
            col = tot
        row = ttnn.reduce_scatter(col, dim=3, cluster_axis=TP_AXIS)
        if AG:
            full = ttnn.all_gather(row, dim=3, cluster_axis=TP_AXIS)
            ttnn.deallocate(row)
            row = full
        return col, row

    rcol, rrow = separate()
    ref_col = [c.reshape(T, D) for c in _chips(rcol)]  # (with ADD: col_output + shared)
    ref_row = [c.reshape(T, D if AG else D // COLS) for c in _chips(rrow)]
    ttnn.deallocate(rcol)
    ttnn.deallocate(rrow)

    # the fused op on the batch layout: batch b holds each mesh row's tokens b T / B ..
    hb = host.view(ROWS, COLS, ROWS, B, T // B, D).permute(0, 3, 1, 2, 4, 5).reshape(ROWS * B, COLS, n // B, D)
    xb = put(mesh, hb.contiguous(), dims=(0, 1), dtype=ttnn.bfloat16)  # chip (r, c): [B, 1, 8 T / B, D]
    del hb
    col_interm = _alloc(mesh, [B, 1, n // B, D])
    col_out = _alloc(mesh, [B, 1, T // B, D])
    row_interm = _alloc(mesh, [B, 1, T // B, D])
    row_out = _alloc(mesh, [B, 1, T // B, D // COLS])
    shared = ttnn.reshape(s1, [B, 1, T // B, D]) if ADD else None
    total = _alloc(mesh, [B, 1, T // B, D]) if ADD else None
    ag_out = _alloc(mesh, [1, 1, T, D]) if AG else None
    ag_counter = (
        ttnn.create_global_semaphore(
            mesh,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(
                            mesh.compute_with_storage_grid_size().x - 1, mesh.compute_with_storage_grid_size().y - 1
                        ),
                    )
                ]
            ),
            0,
        )
        if AG
        else None
    )
    grid = mesh.compute_with_storage_grid_size()
    full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    col_sems = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(3)]
    row_sems = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(3)]

    def fused(xb=xb, shared=shared):
        return ttnn.experimental.col_row_reduce_scatter(
            xb,
            col_interm,
            col_out,
            row_interm,
            row_out,
            col_sems,
            row_sems,
            ttnn.CoreCoord(0, 0),
            row_off,
            relay,
            shared=shared,
            total=total,
            ag_output=ag_out,
            ag_counter=ag_counter,
        )

    t0 = time.perf_counter()
    fused()
    ttnn.synchronize_device(mesh)
    print(f"KCRR first fused call {time.perf_counter() - t0:.2f} s", flush=True)
    for k in range(2):  # (twice: the second call on programs from the cache)
        if k:
            fused()
            ttnn.synchronize_device(mesh)
        got_col = [c.reshape(T, D) for c in _chips(total if ADD else col_out)]
        got_row = [c.reshape(T, D if AG else D // COLS) for c in _chips(ag_out if AG else row_out)]
        dc = [int((a.view(torch.int16) != b.view(torch.int16)).sum()) for a, b in zip(got_col, ref_col)]
        dr = [int((a.view(torch.int16) != b.view(torch.int16)).sum()) for a, b in zip(got_row, ref_row)]
        print(f"KCRR bits call {k} (add {int(ADD)}, ag {int(AG)}): column out{' + shared' if ADD else ''} chips differing {sum(1 for v in dc if v)} / 32 ({sum(dc)} elements), "
              f"row out chips differing {sum(1 for v in dr if v)} / 32 ({sum(dr)} elements)", flush=True)  # fmt: skip

    def run_separate():
        col, row = separate()
        ttnn.deallocate(col)
        ttnn.deallocate(row)

    res = {"separate": [], "fused": []}
    for rnd in range(ROUNDS):
        for name, fn in (("separate", run_separate), ("fused", fused)):
            s = _timed(mesh, fn)
            res[name].append(s)
            print(f"KCRR time round {rnd} {name}: {s * 1e6:.1f} us", flush=True)
    for name, v in res.items():
        if v:
            m = sorted(v)[len(v) // 2]
            print(f"KCRR summary T={T} B={B} {name}: median {m * 1e6:.1f} us ({[round(z * 1e6, 1) for z in v]})")
    if STRESS > 0:  # (a second input and its reference; the trace alternates the two, the last call the second)
        host2 = torch.randn(ROWS, COLS, n, D, generator=g) * torch.rand(ROWS, COLS, 1, D, generator=g).mul(4).exp2()
        host2 = host2.to(torch.bfloat16)
        sh2 = (torch.randn(ROWS, COLS, T, D, generator=g) * 4).to(torch.bfloat16)
        x2 = put(mesh, host2, dims=(0, 1), dtype=ttnn.bfloat16)
        s2 = put(mesh, sh2, dims=(0, 1), dtype=ttnn.bfloat16)
        rcol2, rrow2 = separate(x2, s2)  # [1, 1, T, D] each (the fused outputs' batches in the same token order)
        hb2 = host2.view(ROWS, COLS, ROWS, B, T // B, D).permute(0, 3, 1, 2, 4, 5).reshape(ROWS * B, COLS, n // B, D)
        xb2 = put(mesh, hb2.contiguous(), dims=(0, 1), dtype=ttnn.bfloat16)
        del hb2, host2, sh2
        shared2 = ttnn.reshape(s2, [B, 1, T // B, D]) if ADD else None
        got_c = ttnn.reshape(total if ADD else col_out, [1, 1, T, D])
        got_r = ag_out if AG else ttnn.reshape(row_out, [1, 1, T, D // COLS])

        def diffs(a, b) -> int:
            ne = ttnn.ne(a, b)
            sm = ttnn.sum(ne)
            v = [float(ttnn.to_torch(d).flatten()[0]) for d in ttnn.get_device_tensors(sm)]
            ttnn.deallocate(ne)
            ttnn.deallocate(sm)
            return sum(1 for z in v if round(z))

        reps = max(2, REPS - REPS % 2)
        fused(xb2, shared2)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            for k in range(reps):
                fused(xb if k % 2 == 0 else xb2, shared if k % 2 == 0 else shared2)
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        bad, calls, t0 = 0, 0, time.perf_counter()
        while calls < STRESS:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            calls += reps
            dc, dr = diffs(got_c, rcol2), diffs(got_r, rrow2)
            if dc or dr:
                bad += 1
                print(f"KCRR stress after {calls} calls: column chips differing {dc}, row chips differing {dr}")
            if calls % (reps * 100) == 0:
                print(f"KCRR stress: {calls} calls, {bad} bad checks, {time.perf_counter() - t0:.1f} s", flush=True)
        ttnn.release_trace(mesh, trace)
        print(f"KCRR stress T={T} B={B}: {calls} calls done, {bad} bad checks, {time.perf_counter() - t0:.1f} s")
    print("KCRR done", flush=True)
