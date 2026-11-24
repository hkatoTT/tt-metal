"""The 2D x all-gather (ttnn/x2d_gather.py: the column's trunks and relays along the mesh rows) alone, against
ttnn.all_gather over the mesh column as `TtMoe._routed_once_local` calls it (x row-major [1, 1, T, 4096], a mesh row's own tokens):

  bits:   one eager call of each for each T; each chip's output against the column's tokens (host), bit for bit, and
          ttnn's the same (counts of differing values printed);
  time:   traced, GLM53_X2D_REPS calls back to back, GLM53_X2D_ROUNDS rounds of the two in turn (wall time a call);
  stress: GLM53_X2D_STRESS calls (a trace of GLM53_X2D_REPS calls alternating two inputs, replayed), each replay's
          output checked on the device against ttnn's gather of the last input (the count of differing values).

No asserts: it prints and ends with exit code 0.

    GLM53_X2D_T      comma list of tokens a row (default 128,512)
    GLM53_X2D_MODE   comma list of bits, time, stress (default bits,time)
Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
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

from models.demos.blackhole.glm53_flash.ttnn import x2d_gather as x2d  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROW_AXIS,
    ROWS,
    put,
)

HIDDEN = 4096
TOKENS = [int(v) for v in os.environ.get("GLM53_X2D_T", "128,512").split(",")]
MODES = os.environ.get("GLM53_X2D_MODE", "bits,time").split(",")
REPS = int(os.environ.get("GLM53_X2D_REPS", "10"))
ROUNDS = int(os.environ.get("GLM53_X2D_ROUNDS", "3"))
STRESS = int(os.environ.get("GLM53_X2D_STRESS", "0"))
# "time" also with these forms, ";" list of name:key=value[+key=value] (keys of X2dGather: nb, nbc, nbr, g, dbg, sdbg,
# rswap, ssplit, copiers, sender "x,y", sender2, relay "x,y", relay2), each its own instance (semaphores, landings)
VARIANTS = [v for v in os.environ.get("GLM53_X2D_VARIANTS", "").split(";") if v]


def variant(mesh, spec: str):
    name, _, kv = spec.partition(":")
    kw = {}
    for one in [v for v in kv.split("+") if v]:
        k, v = one.split("=")
        kw[k] = v if k in ("sender", "relay", "relay2", "sender2") else int(v)
    return name, x2d.X2dGather(mesh, HIDDEN, **kw)


def rows_input(mesh, t: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, 1, t, HIDDEN, generator=g).to(torch.bfloat16)
    return x, put(mesh, x, dims=(0, None), layout=ttnn.ROW_MAJOR_LAYOUT)


def ttnn_gather(x):
    return ttnn.all_gather(x, dim=2, cluster_axis=ROW_AXIS)


def host_diffs(out, want: torch.Tensor) -> list[int]:
    """Each chip's count of values that differ (bits) from want [8 T, D]."""
    w = want.view(torch.int16)
    return [
        int((ttnn.to_torch(d).reshape(-1, HIDDEN).view(torch.int16) != w).sum()) for d in ttnn.get_device_tensors(out)
    ]


def dev_diffs(mesh, out, ref) -> list[int]:
    """Each chip's count of values of out that differ from ref (both [1, 1, 8 T, D] row-major), on the device."""
    a = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(ref, ttnn.TILE_LAYOUT)
    ne = ttnn.ne(a, b)
    s = ttnn.sum(ne)
    v = [float(ttnn.to_torch(d).flatten()[0]) for d in ttnn.get_device_tensors(s)]
    for u in (a, b, ne, s):
        ttnn.deallocate(u)
    return [int(round(z)) for z in v]


def timed(mesh, fn, reps: int) -> float:
    fn()  # (compiled, eager, before the capture)
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
    s = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    return s


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_x2d_gather(mesh_device):
    mesh = mesh_device
    x2d.prealloc(mesh, HIDDEN, TOKENS)
    st = x2d._state(mesh, HIDDEN)
    print(
        f"\nX2D sender {st.sender}, relays {[(c.x, c.y) for c in st.relays]}, "
        f"copy cores {[(c.x, c.y) for c in st.copy_cores]}, NB {st.nb}, NBC {st.nbc}, NBR {st.nbr}, G {st.g}",
        flush=True,
    )
    for t in TOKENS:
        xh, xd = rows_input(mesh, t, 1000 + t)
        want = xh.reshape(ROWS * t, HIDDEN)
        if "bits" in MODES:
            try:
                t0 = time.perf_counter()
                out = x2d.x2d_gather(mesh, xd)
                ttnn.synchronize_device(mesh)
                print(f"X2D T={t} first call {time.perf_counter() - t0:.2f} s (compile)", flush=True)
                d = host_diffs(out, want)
                print(f"X2D bits T={t} x2d: chips differing {sum(1 for v in d if v)} / 32, values {d}", flush=True)
                out = x2d.x2d_gather(mesh, xd)
                ttnn.synchronize_device(mesh)
                d = host_diffs(out, want)
                print(f"X2D bits T={t} x2d again: chips differing {sum(1 for v in d if v)} / 32", flush=True)
                ref = ttnn_gather(xd)
                d = host_diffs(ref, want)
                print(f"X2D bits T={t} ttnn: chips differing {sum(1 for v in d if v)} / 32", flush=True)
                ttnn.deallocate(ref)
            except Exception as e:
                print(f"X2D bits T={t}: failed: {str(e).splitlines()[0][:300]}", flush=True)
                continue
        if "time" in MODES:
            forms = [("x2d", lambda: x2d.x2d_gather(mesh, xd))]
            for spec in VARIANTS:
                name, inst = variant(mesh, spec)
                inst.landing(t)
                forms.append((name, lambda inst=inst: inst(xd)))
            forms.append(("ttnn", None))
            res = {name: [] for name, _ in forms}

            def run_ttnn():
                o = ttnn_gather(xd)
                ttnn.deallocate(o)

            for rnd in range(ROUNDS):
                for name, fn in forms:
                    fn = fn or run_ttnn
                    try:
                        s = timed(mesh, fn, REPS)
                        res[name].append(s)
                        gbps = (ROWS - 1) * t * HIDDEN * 2 / s / 1e9
                        print(f"X2D time T={t} round {rnd} {name}: {s * 1e6:.1f} us ({gbps:.1f} GB/s into a chip)")
                    except Exception as e:
                        print(f"X2D time T={t} {name}: failed: {str(e).splitlines()[0][:300]}", flush=True)
            for name, v in res.items():
                if v:
                    m = sorted(v)[len(v) // 2]
                    print(f"X2D summary T={t} {name}: median {m * 1e6:.1f} us ({[round(z * 1e6, 1) for z in v]})")
        if "stress" in MODES and STRESS > 0:
            xh2, xd2 = rows_input(mesh, t, 2000 + t)
            reps = max(2, REPS - REPS % 2)
            ref = ttnn_gather(xd2)  # (the trace's last call takes xd2)
            try:
                x2d.x2d_gather(mesh, xd)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    for k in range(reps):
                        x2d.x2d_gather(mesh, xd if k % 2 == 0 else xd2)
                finally:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                bad = 0
                t0 = time.perf_counter()
                n = 0
                while n < STRESS:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    n += reps
                    d = dev_diffs(mesh, st.landing(t), ref)
                    if any(d):
                        bad += 1
                        print(f"X2D stress T={t} after {n} calls: differing {d}", flush=True)
                    if n % (reps * 50) == 0:
                        print(f"X2D stress T={t}: {n} calls, {bad} bad checks, {time.perf_counter() - t0:.1f} s")
                ttnn.release_trace(mesh, trace)
                print(f"X2D stress T={t}: {n} calls done, {bad} bad checks, {time.perf_counter() - t0:.1f} s")
            except Exception as e:
                print(f"X2D stress T={t}: failed: {str(e).splitlines()[0][:300]}", flush=True)
            ttnn.deallocate(ref)
    print("X2D done", flush=True)
