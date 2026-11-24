"""The sequence-parallel prefill's conv history by copies (`kda_prefill_ops.sp_own` / `sp_pick`, GLM53_KDA_SP_HISTORY)
against the ops `TtKda._prefill_sp` runs without it (the history multiply, untilize,
concat, tilize, the 0/1 matmuls, slice, add, the gather, concat, matmuls, untilize), bit for bit on every chip: the
row's last inputs (before the gather), the conv's carry and the next wave history, over random waves (requests on
runs of rows or interleaved, rows without a segment, a request's first segment: keep 0, last segments shorter than
the history) and random inputs with zeros and -0 in them.  Prints "SPH" lines; exit 0 whatever the bits (a probe).

    GLM53_SPH_T     segment lengths (default "128,512"); GLM53_SPH_CASES waves a length (default 6)
    GLM53_SPH_TIME  calls a trace (0: none): then each form's time (the ops before the gather, the gather, after it)
                    for the last wave of each length, traced
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_sp_history.py -s
"""

from __future__ import annotations

import os
import random
from collections import namedtuple

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn import prefill_tables  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.col_collectives import col_gather  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    ROWS,
    const_zeros,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.kda import SP_GATHER_WORKERS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.kda_prefill_ops import sp_own, sp_pick  # noqa: E402

K, CHANS, F = 3, 6144, 6432
G = namedtuple("G", "rows valid owner start")


def chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def bits(a, b) -> int:
    """How many chips differ bit for bit."""
    return sum(
        not torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16)) for x, y in zip(a, b)
    )


def wave(rnd: random.Random, t: int) -> dict:
    """A random wave's member (row -> (group, index)): 1-4 requests on runs of rows up the ring from a random first row,
    maybe with gaps, or interleaved (two requests' rows alternating)."""
    first = rnd.randrange(ROWS)
    order = [(first + c) % ROWS for c in range(ROWS)]
    member = {}
    if rnd.random() < 0.3:  # interleaved: request A on even ring positions, B on odd ones (each ending at its own)
        n = rnd.choice([2, 3, 4])
        for q, rows in enumerate((tuple(order[0 : 2 * n : 2]), tuple(order[1 : 2 * n : 2]))):
            valid = [t] * (n - 1) + [rnd.choice([1, 2, 3, 5, t])]
            g = G(rows, valid, rows[-1], rnd.choice([0, 512]))
            for i, r in enumerate(rows):
                member[r] = (g, i)
        return member
    pos = 0
    for _ in range(rnd.randint(1, 4)):
        m = rnd.randint(1, 4)
        rows = tuple(order[pos : pos + m])
        if not rows:
            break
        pos += m + (1 if rnd.random() < 0.3 else 0)
        valid = [t] * (len(rows) - 1) + [rnd.choice([1, 2, 3, 4, rnd.randint(1, t), t])]
        g = G(rows, valid, rows[-1], rnd.choice([0, 128, 512]))
        for i, r in enumerate(rows):
            member[r] = (g, i)
    return member


def with_zeros(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """x bf16 with some entries +0 and some -0."""
    m = torch.rand(x.shape, generator=g)
    x = x.clone()
    x[m < 0.02] = 0.0
    x[m > 0.98] = -0.0
    return x


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_sp_history(mesh_device):
    ts = [int(x) for x in os.environ.get("GLM53_SPH_T", "128,512").split(",")]
    cases = int(os.environ.get("GLM53_SPH_CASES", "6"))
    rnd, g = random.Random(5), torch.Generator().manual_seed(5)
    row = dict(dims=(0, None))
    i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    bad = []
    for t in ts:
        for case in range(cases):
            member = wave(rnd, t)
            valid = [member[r][0].valid[member[r][1]] if r in member else 0 for r in range(ROWS)]
            hist = prefill_tables.history_pick(ROWS, valid, K, t)
            arr = prefill_tables.kda_sp_arrays(ROWS, member, K)
            own_t, pick_t = prefill_tables.kda_sp_history_tables(ROWS, valid, member, K)
            proj_h = with_zeros((torch.randn(ROWS, t, F, generator=g) * 0.5).to(torch.bfloat16), g)
            wh_h = with_zeros((torch.randn(ROWS, K, CHANS, generator=g) * 0.5).to(torch.bfloat16), g)
            proj = put(mesh_device, proj_h, **row)
            wh_a = put(mesh_device, wh_h, **row)  # the old path's wave history
            wh_b = put(mesh_device, wh_h, **row)  # the copies' (its rows overwritten by sp_pick)
            keep16 = put(mesh_device, arr["keep"], **row)
            pick_head = put(mesh_device, hist[:, :, :32].contiguous(), **row)
            pick_inputs = put(mesh_device, hist[:, :, 32:].contiguous(), **row)
            in_pick = put(mesh_device, arr["inc"].view(ROWS, 1, K, 64), **row)
            end_pick = put(mesh_device, arr["end"].view(ROWS, 1, K, 64), **row)
            own_tbl = put(mesh_device, own_t, **row, **i32)
            pick_tbl = put(mesh_device, pick_t, **row, **i32)

            # the ops `TtKda._prefill_sp` runs without SP_HISTORY
            kept = ttnn.multiply(wh_a, keep16)
            hrm = ttnn.to_layout(kept, ttnn.ROW_MAJOR_LAYOUT)
            pad = const_zeros(mesh_device, [1, 32 - K, CHANS], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            head = ttnn.to_layout(ttnn.concat([hrm, pad], dim=1), ttnn.TILE_LAYOUT)
            from_inputs = ttnn.matmul(pick_inputs, proj, compute_kernel_config=HIFI4_FP32)
            from_head = ttnn.matmul(pick_head, head, compute_kernel_config=HIFI4_FP32)
            conv = ttnn.slice(from_inputs, [0, 0, 0], [1, K, CHANS])
            own_a = ttnn.add(conv, from_head)
            gathered_a = col_gather(mesh_device, ttnn.reshape(own_a, [1, 1, K, CHANS]), SP_GATHER_WORKERS)
            block = ttnn.concat([ttnn.reshape(head, [1, 1, 32, CHANS]), gathered_a], dim=2)
            incoming = ttnn.matmul(in_pick, block, compute_kernel_config=HIFI4_FP32)
            end_a = ttnn.matmul(end_pick, block, compute_kernel_config=HIFI4_FP32)
            inc_a = ttnn.to_layout(ttnn.reshape(incoming, [1, K, CHANS]), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy(ttnn.reshape(end_a, [1, K, CHANS]), wh_a)

            # the copies
            own_b = sp_own(mesh_device, proj, wh_b, own_tbl)
            gathered_b = col_gather(mesh_device, own_b, SP_GATHER_WORKERS)
            inc_b = sp_pick(mesh_device, gathered_b, wh_b, pick_tbl)

            n_own = bits(chips(ttnn.reshape(own_a, [1, 1, K, CHANS])), chips(own_b))
            n_g = bits(chips(gathered_a), chips(gathered_b))
            n_inc = bits(chips(inc_a), chips(inc_b))
            n_end = bits(chips(wh_a), chips(wh_b))
            forms = {r: (member[r][0].rows, member[r][1], member[r][0].valid[-1], member[r][0].start) for r in member}
            print(f"SPH t={t} case {case}: chips differing: own {n_own}, gathered {n_g}, carry {n_inc}, wave history "
                  f"{n_end} (of {len(chips(own_b))}); keep {[int(v) for v in arr['keep'].view(-1)]}; rows {forms}",
                  flush=True)  # fmt: skip
            if n_own or n_g or n_inc or n_end:
                bad.append((t, case))
            reps = int(os.environ.get("GLM53_SPH_TIME", "0"))
            if reps and case == cases - 1:
                import time

                from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture

                def old_pre():
                    k2 = ttnn.multiply(wh_a, keep16)
                    h2 = ttnn.to_layout(k2, ttnn.ROW_MAJOR_LAYOUT)
                    hd = ttnn.to_layout(ttnn.concat([h2, pad], dim=1), ttnn.TILE_LAYOUT)
                    fi = ttnn.matmul(pick_inputs, proj, compute_kernel_config=HIFI4_FP32)
                    fh = ttnn.matmul(pick_head, hd, compute_kernel_config=HIFI4_FP32)
                    return ttnn.add(ttnn.slice(fi, [0, 0, 0], [1, K, CHANS]), fh), hd

                def old_post(hd):
                    blk = ttnn.concat([ttnn.reshape(hd, [1, 1, 32, CHANS]), gathered_a], dim=2)
                    inc_ = ttnn.matmul(in_pick, blk, compute_kernel_config=HIFI4_FP32)
                    end_ = ttnn.matmul(end_pick, blk, compute_kernel_config=HIFI4_FP32)
                    inc_ = ttnn.to_layout(ttnn.reshape(inc_, [1, K, CHANS]), ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.copy(ttnn.reshape(end_, [1, K, CHANS]), wh_a)
                    return inc_

                forms = {
                    "old before the gather": lambda: old_pre()[0],
                    "old after the gather": lambda: old_post(old_pre()[1]),
                    "sp_own": lambda: sp_own(mesh_device, proj, wh_b, own_tbl),
                    "sp_pick": lambda: sp_pick(mesh_device, gathered_b, wh_b, pick_tbl),
                    "gather": lambda: col_gather(mesh_device, own_b, SP_GATHER_WORKERS),
                }
                for name, run in forms.items():
                    run()  # (compiled before the capture)
                    with trace_capture(mesh_device) as trace:
                        for _ in range(reps):
                            run()
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    us = (time.perf_counter() - t0) / reps * 1e6
                    ttnn.release_trace(mesh_device, trace)
                    print(f"SPH TIME t={t} {name}: {us:.1f} us a call", flush=True)
            for z in (proj, wh_a, wh_b, keep16, pick_head, pick_inputs, in_pick, end_pick, own_tbl, pick_tbl, kept, hrm,
                      head, from_inputs, from_head, conv, own_a, gathered_a, block, incoming, end_a, inc_a, own_b,
                      gathered_b, inc_b):  # fmt: skip
                ttnn.deallocate(z)
    print(f"SPH RESULT: {'all bit for bit' if not bad else f'DIFFERENT in {bad}'}", flush=True)
