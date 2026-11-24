"""The sequence-parallel KDA's state maps to the rows that read them (`kda_maps_x.exchange`, GLM53_KDA_MAPS_X)
against ttnn.all_gather over the mesh column: every chip's slots of its own request's rows bit for bit (the other
slots are never read), over request layouts (pairs of neighbours, interleaved pairs, pairs 4 apart, fours, all 8, a
row alone and rows of none, random ones), two random inputs alternating so that a stale slot shows; eager calls, a
traced stress, and the times (traced).  Prints "KMX" lines; exit 0 whatever the bits (a probe).

    GLM53_KMX_EAGER   eager calls (default 40), layouts and inputs in turn
    GLM53_KMX_STRESS  traced calls (default 0: none): a trace of the layouts in turn replayed to this many calls, the
                      last call's slots checked after each replay
    GLM53_KMX_ROUNDS  timing rounds (default 3, 0: none): 10 traced calls of each, the exchange (pairs of neighbours,
                      interleaved pairs, all 8) and the all-gather

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kda_maps_x.py -s
"""

from __future__ import annotations

import os
import random
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn import kda_maps_x, prefill_tables  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROW_AXIS,
    ROWS,
    put,
)

H, DH = 16, 128
EAGER = int(os.environ.get("GLM53_KMX_EAGER", "40"))
STRESS = int(os.environ.get("GLM53_KMX_STRESS", "0"))
ROUNDS = int(os.environ.get("GLM53_KMX_ROUNDS", "3"))
REPS = 10


class _G:
    def __init__(self, rows):
        self.rows = rows


def _member(groups) -> dict:
    member = {}
    for g in groups:
        grp = _G(list(g))
        for i, r in enumerate(g):
            member[r] = (grp, i)
    return member


FIXED = {
    "pairs": [[0, 1], [2, 3], [4, 5], [6, 7]],
    "pairs1": [[1, 2], [3, 4], [5, 6], [7, 0]],
    "interleaved": [[2, 5], [3, 4], [6, 1], [7, 0]],
    "apart4": [[0, 4], [1, 5], [2, 6], [3, 7]],
    "fours": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "all8": [[3, 4, 5, 6, 7, 0, 1, 2]],
    "sparse": [[1], [3, 4], [6, 0]],
    "three": [[6, 1, 2], [3, 5]],
}


def _random_groups(rnd: random.Random) -> list:
    rows = list(range(ROWS))
    rnd.shuffle(rows)
    used = rows[: rnd.randint(1, ROWS)]
    groups, i = [], 0
    while i < len(used):
        k = rnd.randint(1, min(4, len(used) - i))
        groups.append(used[i : i + k])
        i += k
    return groups


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_maps_x(mesh_device):
    try:
        _study(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KMX FAILED: {err!r}"[:3000], flush=True)


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
    kda_maps_x.prealloc(mesh, H, DH)
    g = torch.Generator().manual_seed(77)
    ins = [
        put(mesh, torch.randn(ROWS, COLS * H, DH, 2 * DH, generator=g).to(torch.bfloat16), dims=(0, 1))
        for _ in range(2)
    ]  # chip (r, c): [1, H, DH, 2 DH]
    refs = [ttnn.all_gather(x, dim=0, cluster_axis=ROW_AXIS) for x in ins]  # [8, H, DH, 2 DH]
    rnd = random.Random(5)
    layouts = list(FIXED.items()) + [(f"random{i}", _random_groups(rnd)) for i in range(8)]
    tables, reads = [], []
    for _, groups in layouts:
        member = _member(groups)
        table = prefill_tables.kda_sp_maps_x_table(ROWS, member)
        tables.append(put(mesh, table, (0, None), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))
        reads.append({r: set(grp.rows) for r, (grp, _) in member.items()})  # (the slots each row reads)
    print(f"KMX layouts {[(n, gr) for n, gr in layouts[:8]]} + 8 random", flush=True)

    def differing(land, k_layout, k_in) -> int:
        """Chips (row r, any column) any of whose read slots (its request's rows) differ from the all-gather's."""
        ne = ttnn.ne(land, refs[k_in])
        bad = set()
        for slot in range(ROWS):
            sl = ttnn.slice(ne, [slot, 0, 0, 0], [slot + 1, H, DH, 2 * DH])
            sm = ttnn.sum(sl)
            for i, t in enumerate(ttnn.get_device_tensors(sm)):
                r = i // COLS  # (device tensors in row-major mesh order)
                if slot in reads[k_layout].get(r, ()) and round(float(ttnn.to_torch(t).flatten()[0])):
                    bad.add(i)
            ttnn.deallocate(sl)
            ttnn.deallocate(sm)
        ttnn.deallocate(ne)
        return len(bad)

    t0 = time.perf_counter()
    land = kda_maps_x.exchange(mesh, ins[0], tables[0])
    ttnn.synchronize_device(mesh)
    print(f"KMX first call {time.perf_counter() - t0:.2f} s: chips differing {differing(land, 0, 0)}", flush=True)
    bad_calls = 0
    for k in range(EAGER):
        kl, ki = k % len(layouts), (k // len(layouts) + k) % 2
        land = kda_maps_x.exchange(mesh, ins[ki], tables[kl])
        d = differing(land, kl, ki)
        if d:
            bad_calls += 1
            if bad_calls <= 8:
                print(f"KMX eager call {k} ({layouts[kl][0]}, input {ki}): chips differing {d}", flush=True)
    print(f"KMX eager: {EAGER} calls, {bad_calls} with a difference", flush=True)

    if STRESS > 0:
        seq = [(kl, (kl + 1) % 2) for kl in range(len(layouts))]  # (an even count of layouts: the inputs in turn)
        kl_last, ki_last = seq[-1]
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            for kl, ki in seq:
                kda_maps_x.exchange(mesh, ins[ki], tables[kl])
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        bad, calls, t0 = 0, 0, time.perf_counter()
        while calls < STRESS:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            calls += len(seq)
            d = differing(kda_maps_x._STATE[(id(mesh), H, DH)].land, kl_last, ki_last)
            if d:
                bad += 1
                if bad <= 8:
                    print(f"KMX stress after {calls} calls: chips differing {d}", flush=True)
        ttnn.release_trace(mesh, trace)
        print(f"KMX stress: {calls} calls, {bad} bad checks, {time.perf_counter() - t0:.1f} s", flush=True)

    if ROUNDS > 0:
        names = {n: i for i, (n, _) in enumerate(layouts)}
        res: dict = {}
        for rnd_i in range(ROUNDS):
            for name, fn in (
                ("all_gather", lambda: ttnn.deallocate(ttnn.all_gather(ins[0], dim=0, cluster_axis=ROW_AXIS))),
                ("pairs", lambda: kda_maps_x.exchange(mesh, ins[0], tables[names["pairs"]])),
                ("interleaved", lambda: kda_maps_x.exchange(mesh, ins[0], tables[names["interleaved"]])),
                ("fours", lambda: kda_maps_x.exchange(mesh, ins[0], tables[names["fours"]])),
                ("all8", lambda: kda_maps_x.exchange(mesh, ins[0], tables[names["all8"]])),
            ):
                s = _timed(mesh, fn)
                res.setdefault(name, []).append(s)
        for name, v in res.items():
            print(f"KMX time {name}: median {sorted(v)[len(v) // 2] * 1e6:.1f} us ({[round(z * 1e6, 1) for z in v]})")
    print("KMX done", flush=True)
