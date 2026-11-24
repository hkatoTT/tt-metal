"""The row all-reduce's narrow landing areas (common._TpArAsync with GLM53_AR_LAND_NARROW=1: landing areas for one
tile row a mesh row, less L1 on every core), alone, against the wide ones ("today" in the prints) on the same inputs.

Per tokens a row in GLM53_ARN_T (default 8,16,32; width 4,096; TT_AR_ASYNC_COMPACT and GLM53_AR_TWO_STAGE as the
model sets them):
  - the L1 the state's landing areas take on every core (a probe's address before and after the state is made: the
    allocator is lock-step, so this is what every program's circular buffers lose);
  - values: `common.all_reduce_async` from DRAM (the move in, the op, the move out) in each state on two inputs, in an
    order where a call that wrote nothing would show the other input's sums; every row of every chip bit for bit
    against the other state and against the sums made on the host (fp32, the row's chips in order, rounded once);
  - GLM53_ARN_MODE "time" too (default values,time): the op alone (input already in its shard, output kept in L1), REPS
    calls a trace, GLM53_ARN_ROUNDS rounds (default 4) of today / narrow, the state made again for each (its median).
Prints only (a failure prints "STUDY FAILED"; a device timeout is raised).  With TT_METAL_WATCHER set outside, run
"values" only (eager).

Run (a device test): GLM53_DEVICE_TEST=1 pytest .../test_ar_land_narrow.py -s
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tests._trace import study, traced  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import common  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    TP_AXIS,
    put,
)

REPS = 20  # (an even count: the landing areas alternate)
HIDDEN = 4096


def probe(mesh) -> int:
    """The address a new one-tile L1 buffer gets (just below the lowest L1 buffer)."""
    p = ttnn.from_torch(
        torch.zeros(32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    a = p.buffer_address()
    ttnn.deallocate(p)
    return a


def state(mesh, narrow: bool):
    """A new _TpArAsync state (the old one's landing areas freed first) -> (state, its L1 bytes a core)."""
    for st in list(common._TP_AR_ASYNC.values()):
        st.release()
    common._TP_AR_ASYNC.clear()
    os.environ["GLM53_AR_LAND_NARROW"] = "1" if narrow else "0"
    before = probe(mesh)
    st = common._tp_ar_async(mesh, HIDDEN, TP_AXIS)
    after = probe(mesh)
    return st, before - after


def chips(t) -> torch.Tensor:
    """[ROWS, COLS, T, D] bf16 from a per-chip [1, 1, T, D] tensor (DRAM or L1)."""
    if t.is_sharded():
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    parts = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]
    return torch.stack(parts).reshape(ROWS, COLS, *parts[0].shape[-2:])


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def timed(mesh, fn) -> float:
    """Seconds a call: REPS calls in one trace, replayed (after two eager calls)."""
    ttnn.deallocate(fn())
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    with traced(mesh) as trace:
        outs = [fn() for _ in range(REPS)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return seconds


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_ar_land_narrow(mesh_device):
    saved = os.environ.get("GLM53_AR_LAND_NARROW")
    try:
        study(lambda: _run(mesh_device))
    finally:
        for st in list(common._TP_AR_ASYNC.values()):
            st.release()
        common._TP_AR_ASYNC.clear()
        if saved is None:
            os.environ.pop("GLM53_AR_LAND_NARROW", None)
        else:
            os.environ["GLM53_AR_LAND_NARROW"] = saved


def _run(mesh):
    ts = [int(v) for v in os.environ.get("GLM53_ARN_T", "8,16,32").split(",") if v]
    modes = os.environ.get("GLM53_ARN_MODE", "values,time").split(",")
    rounds = int(os.environ.get("GLM53_ARN_ROUNDS", "4"))
    g = torch.Generator().manual_seed(2260)
    host = {(t, i): (torch.randn(ROWS, COLS, t, HIDDEN, generator=g) * 2).bfloat16() for t in ts for i in (0, 1)}
    xs = {k: put(mesh, h, dims=(0, 1)) for k, h in host.items()}

    def want(t, i):  # the row's chips added in order in fp32, rounded once (all_reduce_async's reduction)
        h = host[(t, i)].float()
        s = ((h[:, 0] + h[:, 1]) + h[:, 2]) + h[:, 3]
        return s[:, None].expand(ROWS, COLS, t, HIDDEN).bfloat16()

    for form in ("today", "narrow"):
        st, nbytes = state(mesh, form == "narrow")
        print(
            f"ARN L1 {form}: the landing areas take {nbytes} B a core (bufs {[b.buffer_address() for b in st.bufs]}, "
            f"shard {[(b.memory_config().shard_spec.shape, b.memory_config().shard_spec.grid.num_cores()) for b in st.bufs]})",
            flush=True,
        )

    if "values" in modes:
        got = {}
        # an order where a call that wrote nothing would read the other input's sums: today 0, narrow 1, today 1,
        # narrow 0
        for form, i in (("today", 0), ("narrow", 1), ("today", 1), ("narrow", 0)):
            st, _ = state(mesh, form == "narrow")
            for t in ts:
                shard = st.shard_for(t)
                got[(form, t, i)] = chips(common.all_reduce_async(mesh, xs[(t, i)], TP_AXIS))
                print(
                    f"ARN values {form} t={t} input {i}: path {common._ar_path(t, HIDDEN)}, two-stage "
                    f"{common._ar_two_stage(t)}, packed {common._ar_compact(t)}, shard {shard.shard_spec.shape} on "
                    f"{shard.shard_spec.grid.num_cores()} cores",
                    flush=True,
                )
        for t in ts:
            for i in (0, 1):
                a, b, w = got[("today", t, i)], got[("narrow", t, i)], want(t, i)
                print(
                    f"ARN BITS t={t} input {i}: narrow against today {'SAME' if same_bits(a, b) else 'DIFFER'} "
                    f"({int((a != b).sum())} of {a.numel()} differ); today against the host sums "
                    f"{'SAME' if same_bits(a, w) else 'DIFFER'} ({int((a != w).sum())}); narrow against the host sums "
                    f"{'SAME' if same_bits(b, w) else 'DIFFER'} ({int((b != w).sum())}); input 0 against 1 "
                    f"{'SAME' if same_bits(got[('today', t, 0)], got[('today', t, 1)]) else 'DIFFER'}",
                    flush=True,
                )

    if "time" in modes:
        res = {}
        for r in range(rounds):
            for form in ("today", "narrow") if r % 2 == 0 else ("narrow", "today"):
                st, _ = state(mesh, form == "narrow")
                for t in ts:
                    a = ttnn.to_memory_config(xs[(t, 0)], st.shard_for(t))
                    with common.ar_keep_l1(True):
                        res.setdefault((form, t), []).append(timed(mesh, lambda: st(a)))
                    ttnn.deallocate(a)
        for t in ts:
            a, b = res[("today", t)], res[("narrow", t)]
            ma, mb = statistics.median(a), statistics.median(b)
            print(
                f"ARN TIME t={t}: today {ma * 1e6:.2f} us ({' / '.join(f'{v * 1e6:.1f}' for v in a)}), narrow "
                f"{mb * 1e6:.2f} us ({' / '.join(f'{v * 1e6:.1f}' for v in b)}), narrow - today {(mb - ma) * 1e6:+.2f} us",
                flush=True,
            )
    print("ARN DONE", flush=True)
