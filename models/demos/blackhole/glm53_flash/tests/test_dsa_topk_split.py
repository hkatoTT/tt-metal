"""`dsa_indexer.topk_split` (GLM53_DSA_TOPK_SPLIT, 4 by default: a long row's chunk sorts on helper cores, the fold as
topk_large_indices' on the row's core) against `ttnn.experimental.topk_large_indices` ("today" in the prints), bit for
bit (the 512 indices of every row, their order and the 0xFFFFFFFF of the -inf picks) on every chip, eager and replayed
from a trace, and their traced times.

Shapes (GLM53_TS_SHAPES: rows a chip x width): 8 / 32 rows x 65,536 (a 1M cache's decode, 8 / 32 users' K = 7
verify step), 2 / 16 rows, 64 rows (no room for 2 helpers a row: topk_large_indices itself), 1 row, 33 and 64 chunks.
Helpers a row GLM53_TS_HELPERS (default 4,3,2; fewer when the grid has no room).  Every chip its own rows (made on the
host), the kind row by row: normal bf16 scores, scores rounded to a few values (ties at every rank), two values, a -inf
tail (the unseen pools), fewer than 512 finite values, all -inf, and a fresh cache's rows (the visible pools' scores
0.0 / -0.0 but a few, the rest -inf: the model's verify steps from fresh states).  Then one trace with every shape's
split call, replayed, against topk_large_indices' picks.  Then the next program: a batched matmul with an item on every
core of the grid (the DSA absorb's form), right after topk_split, against the same matmul right after itself (helpers
that ended right after their last sort left a Dest setting behind that changed the next matmul's first output tile;
they end in transpose_dest).  Rows with NaN (GLM53_TS_NAN=1, default) are reported apart (not a model's input).
GLM53_TS_TIME=0: no times (e.g. under the watcher).
Prints "TS RESULT: all bit for bit" or the count of differing cases; no assert (a probe: its printed lines are the
result).
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_topk_split.py -s
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
    WEIGHT_MM,
    batched_matmul_config,
    put,
)

RM = ttnn.ROW_MAJOR_LAYOUT
TOPK = 512
KINDS = 7


def traced(mesh, fn, reps: int = 10) -> float:
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = []
    try:
        for _ in range(reps):
            outs.append(fn())
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        seconds = (time.perf_counter() - t0) / reps
    finally:
        ttnn.release_trace(mesh, trace)
        for o in outs:
            ttnn.deallocate(o)
    return seconds


def make_row(kind: int, width: int, i: int, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(width, generator=g)
    if kind == 1:  # a few values: ties at every rank
        x = torch.round(x * 4) / 4
    elif kind == 2:  # two values
        x = (torch.rand(width, generator=g) < 0.01).float()
    elif kind == 3:  # a -inf tail (pools past the position)
        x[int(width * 0.6) + i :] = float("-inf")
    elif kind == 4:  # fewer than 512 finite values
        x[300 + i :] = float("-inf")
    elif kind == 5:  # all -inf
        x[:] = float("-inf")
    elif kind == 6:  # a fresh cache: the visible pools 0.0 / -0.0 but a few, the rest -inf
        v = 90 + (i * 37) % 300
        z = torch.zeros(v)
        z[torch.rand(v, generator=g) < 0.4] = -0.0
        few = torch.randint(0, v, (3,), generator=g)
        z[few] = torch.randn(3, generator=g) * 1e-3
        x[:] = float("-inf")
        x[:v] = z
    elif kind == 7:  # NaN here and there (reported apart)
        x[torch.rand(width, generator=g) < 0.001] = float("nan")
        x[torch.rand(width, generator=g) < 0.001] = float("-nan")
    return x


def make_rows(n: int, width: int, g: torch.Generator, kinds) -> torch.Tensor:
    """Every chip's n rows ([ROWS, COLS, n, width] bf16): chip (r, c)'s row i of kind kinds[(chip + i) % len]."""
    out = torch.empty(ROWS, COLS, n, width)
    for r in range(ROWS):
        for c in range(COLS):
            chip = r * COLS + c
            for i in range(n):
                out[r, c, i] = make_row(kinds[(chip + i) % len(kinds)], width, i, g)
    return out.to(torch.bfloat16)


def bits(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(x).to(torch.int64) & 0xFFFFFFFF for x in ttnn.get_device_tensors(t)]


def first_diff(got, want, n) -> str:
    for c, (a, b) in enumerate(zip(want, got)):
        if torch.equal(a, b):
            continue
        a, b = a.reshape(n, TOPK), b.reshape(n, TOPK)
        d = a != b
        r = int(d.any(1).nonzero()[0])
        j = int(d[r].nonzero()[0])
        bad = sum(not torch.equal(x, y) for x, y in zip(want, got))
        return (
            f" ({bad} chips; chip {c}: {int(d.any(1).sum())} rows, first row {r} from {j}: "
            f"today {a[r, j : j + 4].tolist()} got {b[r, j : j + 4].tolist()})"
        )
    return ""


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_topk_split(mesh_device):
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import topk_split

    mesh = mesh_device
    helpers = [int(h) for h in os.environ.get("GLM53_TS_HELPERS", "4,3,2").split(",")]
    timing = os.environ.get("GLM53_TS_TIME", "1") == "1"
    g = torch.Generator().manual_seed(31)
    bad = cases = 0

    def split_with(h, rows):
        os.environ["GLM53_DSA_TOPK_SPLIT"] = str(h)
        try:
            return topk_split(mesh, rows, TOPK)
        finally:
            os.environ.pop("GLM53_DSA_TOPK_SPLIT", None)

    def check(label, n, rows, count=True):
        nonlocal bad, cases
        ref = ttnn.experimental.topk_large_indices(rows, k=TOPK)
        want = bits(ref)
        ttnn.deallocate(ref)
        line = []
        for h in helpers:
            out = split_with(h, rows)
            got = bits(out)
            ttnn.deallocate(out)
            same = all(torch.equal(a, w) for a, w in zip(got, want))
            if count:
                cases += 1
                bad += not same
            line.append(f"helpers {h}: " + ("bit for bit" if same else "DIFFERENT" + first_diff(got, want, n)))
        # replayed from a trace (the program cache hit: the runtime args from the call)
        h = helpers[0]
        ttnn.deallocate(split_with(h, rows))
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            out = split_with(h, rows)
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        got = bits(out)
        ttnn.release_trace(mesh, trace)
        ttnn.deallocate(out)
        same = all(torch.equal(a, w) for a, w in zip(got, want))
        if count:
            cases += 1
            bad += not same
        line.append(f"traced (helpers {h}): " + ("bit for bit" if same else "DIFFERENT" + first_diff(got, want, n)))
        if timing:
            t_off = traced(mesh, lambda: ttnn.experimental.topk_large_indices(rows, k=TOPK)) * 1e6
            ts = [f"{h}: {traced(mesh, lambda h=h: split_with(h, rows)) * 1e6:.1f}" for h in helpers]
            line.append(f"times us: today {t_off:.1f}, " + ", ".join(ts))
        print(f"TS {label}: " + "; ".join(line), flush=True)
        return want

    kept = []
    shapes = os.environ.get("GLM53_TS_SHAPES", "8x65536,32x65536,2x65536,16x65536,64x65536,1x65536,8x16896,32x32768")
    for spec in [s for s in shapes.split(",") if s]:
        n, width = (int(v) for v in spec.split("x"))
        rows = put(mesh, make_rows(n, width, g, list(range(KINDS))), dims=(0, 1), layout=RM)
        want = check(f"{n} rows x {width} ({width // TOPK} chunks)", n, rows)
        kept.append((n, rows, want))

    # one trace with every shape's split call (programs of different row counts back to back), replayed twice
    h = helpers[0]
    for _, rows, _ in kept:
        ttnn.deallocate(split_with(h, rows))
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        outs = [split_with(h, rows) for _, rows, _ in kept]
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    line = []
    for (n, _, want), out in zip(kept, outs):
        got = bits(out)
        same = all(torch.equal(a, w) for a, w in zip(got, want))
        cases += 1
        bad += not same
        line.append(f"{n} rows " + ("bit for bit" if same else "DIFFERENT" + first_diff(got, want, n)))
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    print(f"TS one trace, every shape (helpers {h}): " + "; ".join(line), flush=True)

    # one trace replayed 5 times: the 1-row split again and again with an 8-row split, topk_large_indices on 8 and on 2
    # rows and a 2-row split between (the model's decode: a 1-row call a layer, other programs between); every form run
    # once eagerly first (else a program is made while capturing: TT_FATAL !is_capturing_trace)
    by_n = {n: (rows, want) for n, rows, want in kept}
    if all(m in by_n for m in (1, 2, 8)):
        r1, w1 = by_n[1]
        r2, w2 = by_n[2]
        r8, w8 = by_n[8]
        forms = [  # (rows, the call, topk_large_indices' picks; None: another program between, not compared)
            (1, lambda: split_with(h, r1), w1),
            (8, lambda: split_with(h, r8), w8),
            (8, lambda: ttnn.experimental.topk_large_indices(r8, k=TOPK), w8),
            (1, lambda: split_with(h, r1), w1),
            (2, lambda: ttnn.experimental.topk_large_indices(r2, k=TOPK), None),
            (2, lambda: split_with(h, r2), w2),
            (1, lambda: split_with(h, r1), w1),
        ]
        for _, call, _ in forms:  # every form once eagerly before the capture (no program made while capturing)
            ttnn.deallocate(call())
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            outs = [(n, call(), want) for n, call, want in forms]
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        other = [out for _, out, want in outs if want is None]
        seq = [(n, out, want) for n, out, want in outs if want is not None]
        line = []
        for rep in range(5):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if rep in (0, 4):
                for idx, (n, out, want) in enumerate(seq):
                    got = bits(out)
                    same = all(torch.equal(a, w) for a, w in zip(got, want))
                    cases += 1
                    bad += not same
                    if not same:
                        line.append(f"replay {rep} call {idx} ({n} rows) DIFFERENT" + first_diff(got, want, n))
        ttnn.release_trace(mesh, trace)
        for _, out, _ in seq:
            ttnn.deallocate(out)
        for out in other:
            ttnn.deallocate(out)
        print(f"TS interleaved trace x5 (helpers {h}): " + ("; ".join(line) or "every call bit for bit"), flush=True)
    for _, rows, _ in kept:
        ttnn.deallocate(rows)

    # the next program: a batched matmul, one item on every core (topk_split's 8 rows' cores and their helpers among
    # them), right after topk_split against the same matmul right after itself, bit for bit
    grid = mesh.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    q = put(mesh, (torch.randn(1, ncores, 8, 256, generator=g) * 0.5).to(torch.bfloat16), dims=(None, None))
    w = put(mesh, torch.randn(1, ncores, 256, 512, generator=g) * 0.05, dims=(None, None), dtype=ttnn.bfloat8_b)
    mm_cfg = batched_matmul_config(mesh, k_tiles=8, n_tiles=16)

    def matmul_bits():
        out = ttnn.matmul(q, w, program_config=mm_cfg, compute_kernel_config=WEIGHT_MM)
        got = [ttnn.to_torch(x).contiguous().view(torch.int16) for x in ttnn.get_device_tensors(out)]
        ttnn.deallocate(out)
        return got

    rows = put(mesh, make_rows(8, 65536, g, [0]), dims=(0, 1), layout=RM)
    matmul_bits()
    after_itself = matmul_bits()
    line = []
    for h in helpers:
        matmul_bits()
        ttnn.deallocate(split_with(h, rows))
        same = all(torch.equal(a, b) for a, b in zip(matmul_bits(), after_itself))
        cases += 1
        bad += not same
        line.append(f"helpers {h}: " + ("the same" if same else "DIFFERENT"))
    print("TS the matmul right after topk_split: " + "; ".join(line), flush=True)
    for t in (rows, q, w):
        ttnn.deallocate(t)

    if os.environ.get("GLM53_TS_NAN", "1") == "1":  # (not counted: no NaN reaches the model's topk)
        rows = put(mesh, make_rows(8, 65536, g, [0, 7]), dims=(0, 1), layout=RM)
        check("NaN rows (not counted), 8 rows x 65536", 8, rows, count=False)
        ttnn.deallocate(rows)
    print(f"TS RESULT: {'all bit for bit' if bad == 0 else f'{bad} of {cases} DIFFERENT'} ({cases} cases)", flush=True)
