"""DFlash2's acceptance and the drafter's inputs in one op (`verify_accept.accept_inputs`, GLM53_DFLASH2_ACCEPT_FUSED;
and the step's results row, `verify_accept.results_row`, against `dflash2_step.gathered_results`) against the separate
ops ("today" in the prints: `accept_device` + `dflash2_step.drafter_inputs`) on the same inputs: the accepted
[1, 1, 8, 2 U] fp32, the bonus [1, U] uint32 and the starts [1, U] int32, bit for bit on all 32 chips, at 8 / 32 / 128
users' shapes (U, K) = (1, 7), (4, 7), (16, 3) and a few more, over GLM53_ACI_CASES random cases each (every row its
own tokens; per user a random count of leading drafts equal to the greedy ids, 0 .. K, the rest random; ids up to the
vocabulary), then both forms traced and timed.

    GLM53_ACI_SHAPES  U:K,.. (default 1:7,4:7,16:3,2:7,8:3,16:7)
    GLM53_ACI_CASES   random cases a shape (default 6)
    GLM53_ACI_TIME=0  no timing (e.g. under the watcher)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_accept_inputs.py -s

Prints "ACI U .. K ..: case ..: accepted / bonus / starts the same" lines, "ACI TIME ..", and "ACI ALL SAME".
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS, put  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dflash2_step import drafter_inputs, gathered_results  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.verify_accept import (  # noqa: E402
    accept_device,
    accept_inputs,
    results_row,
)

VOCAB = 154880
REPS = 10


def same_bits(a, b) -> bool:
    xs, ys = ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)
    if len(xs) != len(ys):
        return False
    for p, q in zip(xs, ys):
        tp, tq = ttnn.to_torch(p), ttnn.to_torch(q)
        if tp.shape != tq.shape or tp.dtype != tq.dtype:
            return False
        if not torch.equal(tp.contiguous().view(torch.uint8), tq.contiguous().view(torch.uint8)):
            return False
    return True


def spec_of(t) -> tuple:
    return (tuple(t.shape), t.dtype, t.layout)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_accept_inputs(mesh_device):
    mesh = mesh_device
    shapes = [
        tuple(int(v) for v in s.split(":"))
        for s in os.environ.get("GLM53_ACI_SHAPES", "1:7,4:7,16:3,2:7,8:3,16:7").split(",")
    ]
    cases = int(os.environ.get("GLM53_ACI_CASES", "6"))
    timing = os.environ.get("GLM53_ACI_TIME", "1") != "0"
    rm = dict(dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    others_t = torch.full((ROWS, 1, ROWS, 1), 1.0e9)
    others_t[torch.arange(ROWS), 0, torch.arange(ROWS), 0] = 0.0
    others = put(mesh, others_t, dims=(0, None), dtype=ttnn.float32)
    g = torch.Generator().manual_seed(2510)
    ok = True
    for u, k in shapes:
        b = (k + 1) * u
        seats = put(mesh, torch.arange(ROWS * u, dtype=torch.float32).reshape(ROWS, 1, 1, u), dims=(0, None),
                    dtype=ttnn.float32)  # fmt: skip
        for case in range(cases):
            greedy = torch.randint(0, VOCAB, (ROWS, k + 1, u), generator=g)
            tokens = torch.randint(0, VOCAB, (ROWS, k + 1, u), generator=g)
            # per user n leading drafts equal to the greedy id before them (case 0: none, case 1: all K)
            n = torch.randint(0, k + 1, (ROWS, u), generator=g)
            if case == 0:
                n[:] = 0
            elif case == 1:
                n[:] = k
            for r in range(ROWS):
                for uu in range(u):
                    for j in range(1, int(n[r, uu]) + 1):
                        tokens[r, j, uu] = greedy[r, j - 1, uu]
            if case == 2:  # (ids 0 and the largest: the conversions' ends)
                greedy[0, :, 0] = 0
                greedy[1, :, 0] = VOCAB - 1
            ids = put(mesh, greedy.reshape(1, 1, ROWS * b, 1).float(), dims=(None, None), dtype=ttnn.float32)
            io_ids = put(mesh, tokens.reshape(ROWS, b).to(torch.int32), dims=(0, None), dtype=ttnn.uint32,
                         layout=ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
            base1 = put(mesh, torch.randint(1, 4000, (ROWS, 1, 1, u), generator=g, dtype=torch.int32), dims=(0, None),
                        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
            io = SimpleNamespace(users=u, drafts=k, ids=io_ids, others=others)
            # the last case: a test's forced acceptance (n from the positions and seats, `forced_accept`)
            forced = (base1, seats) if case == cases - 1 else None
            # the drafter's block [U, 8] uint32 row-major a row (walk's: t[s], the drafts, zeros)
            dr = torch.randint(0, VOCAB, (ROWS * u, 8), generator=g, dtype=torch.int32)
            dr[:, k + 1 :] = 0
            drafts = put(mesh, dr, dims=(0, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

            def today():
                acc = accept_device(mesh, ids, io, forced=forced)
                bo, st = drafter_inputs(acc, others, base1, u)
                return acc, bo, st

            def fused():
                return accept_inputs(mesh, ids, io, base1, forced=forced)

            def today_step():  # (the step's: the acceptance, the drafter's inputs, the results)
                acc, bo, st = today()
                return bo, st, gathered_results(acc, drafts, u), acc

            def fused_step():
                own, bo, st = accept_inputs(mesh, ids, io, base1, forced=forced, gather=False)
                return bo, st, results_row(mesh, own, drafts, u), own

            a0, b0, s0 = today()
            a1, b1, s1 = fused()
            sa, sb, ss = same_bits(a0, a1), same_bits(b0, b1), same_bits(s0, s1)
            specs = spec_of(a0) == spec_of(a1) and spec_of(b0) == spec_of(b1) and spec_of(s0) == spec_of(s1)
            # the host's reference for the fused one (chip 0 of each row)
            want_n = (tokens[:, 1:] == greedy[:, :-1]).long().cumprod(dim=1).sum(dim=1)
            got_acc = ttnn.to_torch(ttnn.get_device_tensors(a1)[0]).reshape(ROWS, 2 * u)
            if forced is not None:  # (n from the forced rule: the bonus at it)
                want_n = got_acc[:, :u].long()
            host_ok = torch.equal(got_acc[:, :u].long(), want_n) and torch.equal(
                got_acc[:, u:].long(), greedy.gather(1, want_n[:, None, :]).squeeze(1)
            )
            line = (f"ACI U {u} K {k}: case {case}{' (forced)' if forced else ''}: accepted {'same' if sa else 'DIFFER'} / bonus "
                    f"{'same' if sb else 'DIFFER'} / starts {'same' if ss else 'DIFFER'}, specs "
                    f"{'same' if specs else 'DIFFER'}, host {'ok' if host_ok else 'WRONG'}")  # fmt: skip
            print(line, flush=True)
            if not (sa and sb and ss and specs and host_ok):
                print(
                    f"  today {spec_of(a0)} {spec_of(b0)} {spec_of(s0)}; fused {spec_of(a1)} {spec_of(b1)} {spec_of(s1)}"
                )
            ok = ok and sa and sb and ss and specs and host_ok
            for t in (a0, b0, s0, a1, b1, s1):
                ttnn.deallocate(t)
            # the step's results: gathered_results against results_row
            r0, r1 = today_step(), fused_step()
            sr = same_bits(r0[2], r1[2]) and spec_of(r0[2]) == spec_of(r1[2])
            print(f"ACI U {u} K {k}: case {case}: results {'same' if sr else 'DIFFER'} {spec_of(r1[2])}", flush=True)
            if not sr:
                print(f"  today {spec_of(r0[2])}; fused {spec_of(r1[2])}")
            ok = ok and sr
            for t in r0 + r1:
                ttnn.deallocate(t)
            if timing and case == cases - 2:  # (the last random case before the forced one)
                times = {}
                for name, fn in (("today", today_step), ("fused", fused_step)):
                    for t in fn():
                        ttnn.deallocate(t)
                    ttnn.synchronize_device(mesh)
                    with trace_capture(mesh) as trace:
                        outs = [fn() for _ in range(REPS)]
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    times[name] = (time.perf_counter() - t0) / REPS
                    ttnn.release_trace(mesh, trace)
                    for group in outs:
                        for t in group:
                            ttnn.deallocate(t)
                print(f"ACI TIME U {u} K {k} (acceptance + inputs + results): today {times['today'] * 1e6:.1f} -> fused {times['fused'] * 1e6:.1f} us "
                      f"({(times['fused'] - times['today']) * 1e6:+.1f})", flush=True)  # fmt: skip
            for t in (ids, io_ids, base1, drafts):
                ttnn.deallocate(t)
        ttnn.deallocate(seats)
    print(f"ACI {'ALL SAME' if ok else 'NOT ALL SAME'}", flush=True)
    assert ok
