"""The KDA verify step's state read ahead (GLM53_KDA_D2=2) and its fused form (=1) against the two ops (decode_prepare +
recurrent_decode_step's chunked form) on the mesh, bit for bit (ttnn/kda_d2.py): the model's verify call (users,
keep, saved, tokens, pair_major, L1 hand-offs, split channels; o into L1) at GLM53_KD2_PAIRS "users a row:T" (default
16:4,4:8,1:8), random inputs (seeded), GLM53_KD2_STEPS steps in a row from the same start (default 3: the later steps
redo the earlier steps' saved rows), every chip's o, q / k / v / decay / beta, state, saved rows (their beta rows'
never-written floats left out), conv history and checkpoint compared. GLM53_KD2_FORMS (default "2": the read-ahead
form; "1": the fused form; a comma list for both). With GLM53_KD2_TIMES=N: the time of N traced replays of one step
of each form (host wall clock, synchronized, the op pair alone). No checkpoint.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kda_d2.py -s

Prints "KD2 CHECK <pair> <form>: bit for bit SAME | DIFF ..." and "KD2 TIME <pair> <form>: <us> us a step".
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

from models.demos.blackhole.glm53_flash.tests._trace import traced  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import kda_d2  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.kda import chunk_slot_tiles  # noqa: E402

H = kda_d2.H


def host(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def one_step(mesh, d, users, tokens, form):
    if form == "ops":
        return kda_d2.step_ops(mesh, d, users, tokens)
    o, mid, _ = kda_d2.step_py(mesh, d, users, tokens, "d2p" if form == "2" else "fused")
    return o, mid


def run_form(mesh, hostin, users, tokens, form, steps):
    d = kda_d2.upload(mesh, hostin)
    outs = []
    for _ in range(steps):
        o, mid = one_step(mesh, d, users, tokens, form)
        outs.append([host(mesh, t) for t in [o] + mid])
        for t in [o] + mid:
            ttnn.deallocate(t)
    fin = [host(mesh, t) for t in [d["st"], d["rows"], *d["hist"], d["sv"]]]
    for t in (d["p"], d["f"], *d["hist"], d["sv"], d["tp"], d["gc"], d["flags"], d["st"], d["rows"]):
        ttnn.deallocate(t)
    return outs, fin


def time_form(mesh, hostin, users, tokens, form, n):
    d = kda_d2.upload(mesh, hostin)
    for t in one_step(mesh, d, users, tokens, form):  # (eager first: programs built before the capture)
        for x in t if isinstance(t, list) else [t]:
            ttnn.deallocate(x)
    with traced(mesh) as trace:
        o, mid = one_step(mesh, d, users, tokens, form)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / n * 1e6
    ttnn.release_trace(mesh, trace)
    for t in [o] + mid:
        ttnn.deallocate(t)
    for t in (d["p"], d["f"], *d["hist"], d["sv"], d["tp"], d["gc"], d["flags"], d["st"], d["rows"]):
        ttnn.deallocate(t)
    return us


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_d2(mesh_device):
    mesh = mesh_device
    steps = int(os.environ.get("GLM53_KD2_STEPS", "3"))
    forms = os.environ.get("GLM53_KD2_FORMS", "2").split(",")
    times = int(os.environ.get("GLM53_KD2_TIMES", "0"))
    bad = 0
    for pair in os.environ.get("GLM53_KD2_PAIRS", "16:4,4:8,1:8").split(","):
        users, tokens = (int(v) for v in pair.split(":"))
        hostin = kda_d2.make_inputs(mesh, users, tokens, 2600 + 10 * users + tokens)
        ref = run_form(mesh, hostin, users, tokens, "ops", steps)
        names = ["o", "q", "k", "v", "decay", "beta"]
        fnames = ["state", "rows", "h0", "h1", "h2", "saved"]
        for form in forms:
            got = run_form(mesh, hostin, users, tokens, form, steps)
            diffs = []
            for s in range(steps):
                for nm, a, b in zip(names, ref[0][s], got[0][s]):
                    if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                        diffs.append(f"step{s}.{nm} max|d| {(a - b).abs().max().item():.3e}")
            for nm, a, b in zip(fnames, ref[1], got[1]):
                a, b = a.contiguous(), b.contiguous()
                if nm == "rows":  # (a slot's beta row: T floats written, its other 16 - T never: left out)
                    nt = chunk_slot_tiles(tokens)
                    a = a.reshape(-1, nt * 1024).clone()
                    b = b.reshape(-1, nt * 1024).clone()
                    a[:, 256 * tokens + tokens : 256 * tokens + 16] = 0
                    b[:, 256 * tokens + tokens : 256 * tokens + 16] = 0
                if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                    diffs.append(f"final.{nm} max|d| {(a - b).abs().max().item():.3e}")
            bad += len(diffs)
            o_last = ref[0][-1][0]
            print(
                f"KD2 CHECK {pair} {form}: {'bit for bit SAME' if not diffs else 'DIFF: ' + '; '.join(diffs[:10])} "
                f"({steps} steps, {o_last.shape[0]} rows of o over the chips, o finite {torch.isfinite(o_last).all().item()})",
                flush=True,
            )
        if times:
            for form in ["ops"] + forms:
                print(f"KD2 TIME {pair} {form}: {time_form(mesh, hostin, users, tokens, form, times):.1f} us a step "
                      f"({times} replays, host clock)", flush=True)  # fmt: skip
    print(f"KD2 DONE bad={bad}", flush=True)
    assert bad == 0
