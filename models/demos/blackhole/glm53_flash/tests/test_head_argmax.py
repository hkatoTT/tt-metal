"""The greedy head's per-chip top by one op (`head_argmax.head_argmax`) against the ops it replaced
in `TtLmHead.greedy_mesh` (mask add, max, compare, multiply, id add, min), bit for bit on every chip, and against
torch; both timed in a trace.  Rows M = 8 t: 8 / 16 / 64 / 128 / 256 (plain and MTP's verify step at 8 / 32 / 128
users); the logits with ties of the row's top (several columns, across runs), rows all negative, a -0.0 / +0.0 top,
and the last chip's ids past the vocabulary (masked).

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_head_argmax.py -s
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
)
from models.demos.blackhole.glm53_flash.ttnn.head_argmax import head_argmax  # noqa: E402

VOCAB, CHIPS = 154_880, ROWS * COLS
W = -(-VOCAB // (CHIPS * 32)) * 32  # 4,864 columns a chip, as TtLmHead's
VALID = [max(0, min(W, VOCAB - k * W)) for k in range(CHIPS)]


def traced(mesh, fn, reps: int = 20) -> float:
    outs = fn()
    for o in outs:
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    kept = [fn() for _ in range(reps)]
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    for outs in kept:
        for o in outs:
            ttnn.deallocate(o)
    return seconds


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_head_argmax(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(11)
    ids_all = torch.arange(CHIPS * W, dtype=torch.float32).view(ROWS, 1, 1, COLS * W)
    mask = put(mesh, torch.where(ids_all < VOCAB, 0.0, -1.0e30), dims=(0, 3), dtype=ttnn.float32)
    mesh_ids = put(mesh, ids_all, dims=(0, 3), dtype=ttnn.float32)
    failures = []
    for m in (8, 16, 64, 128, 256):
        logits = torch.randn(ROWS, 1, m, COLS * W, generator=g) * 4
        logits[:, :, 1] -= 50.0  # a row all negative
        top = logits.amax(dim=-1, keepdim=True)
        for r in range(2, m, 3):  # ties of the row's top over the mesh and within a chip (several runs)
            cols = torch.randint(0, COLS * W, (4,), generator=g)
            logits[:, :, r, cols] = top[:, :, r] + 1.0
        logits[:, :, 0] = -1.0
        logits[:, :, 0, 5] = -0.0  # the top 0.0 as -0.0 on one chip, +0.0 on another
        logits[:, :, 0, W + 7] = 0.0
        logits[:, :, -1, -3:] = 1.0e4  # past the vocabulary on the last chip: masked
        d = put(mesh, logits, dims=(0, 3), dtype=ttnn.float32)

        def op():
            return head_argmax(mesh, d, VALID, W)

        def ref():
            x = ttnn.add(d, mask)
            value = ttnn.max(x, dim=-1, keepdim=True)
            missed = ttnn.multiply(ttnn.ne(x, value), 1.0e9)
            ttnn.deallocate(x)
            ids = ttnn.min(ttnn.add(missed, mesh_ids), dim=-1, keepdim=True)
            ttnn.deallocate(missed)
            return value, ids

        got, want = op(), ref()
        same = True
        for t_got, t_want in zip(got, want):
            a = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(t_got)]
            b = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(t_want)]
            same = same and all(torch.equal(x.view(torch.int32), y.view(torch.int32)) for x, y in zip(a, b))
        # torch: chip k = 4 r + c's columns k W .. (k + 1) W - 1 of the global row
        chip_ids = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(got[1])]
        exact = True
        for k in range(CHIPS):
            r, c = divmod(k, COLS)
            part = logits[r, 0, :, c * W : (c + 1) * W].clone()
            part[:, VALID[k] :] = float("-inf")
            want_id = part.argmax(dim=-1).float() + k * W
            exact = exact and torch.equal(chip_ids[k][0, 0, :m, 0], want_id)
        t_op, t_ref = traced(mesh, op), traced(mesh, ref)
        print(
            f"M {m:3d}: op = the ops bit for bit {same}, ids = torch {exact}; traced {t_ref * 1e6:6.1f} -> "
            f"{t_op * 1e6:6.1f} us",
            flush=True,
        )
        if not (same and exact):
            failures.append(m)
        for t in (*got, *want, d):
            ttnn.deallocate(t)
    assert not failures, failures
