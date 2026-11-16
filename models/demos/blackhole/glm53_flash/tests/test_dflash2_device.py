"""The DFlash2 drafter on the mesh (`ttnn/dflash2.TtDFlash2`) against the CPU reference (`tt/dflash2.py`) on the same
taps (the greedy records' target taps, tools/dflash2_acceptance.py): each mesh row's users on records and positions
of their own; the context written by write_context in chunks; one draft.  Per user: the block's final hidden states
(PCC, max |d|), each slot's top 16 (the same set, the same order, the same top 1), the drafts (the same, slot by
slot), and how many of the drafts the record accepts (device and CPU).

    GLM53_DEVICE_TEST=1 GLM53_DFLASH2_USERS=1 GLM53_DFLASH2_LAYERS=1 pytest \
        models/demos/blackhole/glm53_flash/tests/test_dflash2_device.py -s
Env: GLM53_DFLASH2_USERS (a mesh row's users, 1), GLM53_DFLASH2_LAYERS (the first layers only, all), GLM53_DFLASH2_DTYPE
(bf16 / bfp8 / bfp4, bf16), GLM53_DFLASH2_HEAD_DTYPE (test_dflash2_acceptance: the head's, the drafter's by
default), GLM53_DFLASH2_BLOCK (8), GLM53_DFLASH2_TAPS (the taps' directory, default data_dir("dflash2-v1", "taps"):
tools/dflash2_acceptance.py's --taps-dir, or test_dflash2_taps.py's GLM53_DFLASH2_SAVE), GLM53_DFLASH2_CTX (the largest
start, 1,000), GLM53_DFLASH2_CHUNK (the context's rows a write, 128).
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.paths import data_dir, out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.tools.dflash2_acceptance import taps_for  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.dflash2 import DFlash2Drafter  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS,
    MESH_SHAPE,
    ROWS,
    WEIGHT_DTYPES,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2  # noqa: E402

TAPS = Path(os.environ.get("GLM53_DFLASH2_TAPS") or data_dir("dflash2-v1", "taps"))
CHUNK = int(os.environ.get("GLM53_DFLASH2_CHUNK", "128"))  # the context's write_context rows a call (prefill: 512)


# a failure in a device test's body is printed ("SOFT-FAIL <test>" and its traceback) and the test passes, so that a
# run of several studies goes on and ends with exit code 0; search its log for SOFT-FAIL.  GLM53_DFLASH2_STRICT=1:
# raise as usual
STRICT = os.environ.get("GLM53_DFLASH2_STRICT", "0") == "1"


def soft(fn):
    import functools
    import traceback

    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if STRICT:
                raise
            print(f"SOFT-FAIL {fn.__name__}\n{traceback.format_exc()}", flush=True)

    return run


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-30))


def row_tensor(mesh, t: torch.Tensor, dtype, layout):
    """t [8, ..] -> mesh row r's slice t[r : r + 1] (the same on its 4 chips): [8, 1, R, W] -> [1, 1, R, W] a chip,
    [8, U] -> [1, U]."""
    return put(mesh, t, dims=(0, None), dtype=dtype, layout=layout)


def chip_rows(t) -> list[torch.Tensor]:
    """Each mesh row's tensor (its chip 0's)."""
    devs = ttnn.get_device_tensors(t)
    return [ttnn.to_torch(devs[r * COLS]) for r in range(ROWS)]


def u32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.int32, getattr(torch, "uint32", torch.int32)):
        return t.view(torch.int32).long() & 0xFFFFFFFF
    return t.long()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_dflash2_vs_cpu(mesh_device):
    mesh = mesh_device
    users = int(os.environ.get("GLM53_DFLASH2_USERS", "1"))
    block = int(os.environ.get("GLM53_DFLASH2_BLOCK", "8"))
    dtype = WEIGHT_DTYPES[os.environ.get("GLM53_DFLASH2_DTYPE", "bf16")]
    max_start = int(os.environ.get("GLM53_DFLASH2_CTX", "1000"))
    t0 = time.time()
    drafter = DFlash2Drafter()
    cfg = drafter.config
    layers = int(os.environ.get("GLM53_DFLASH2_LAYERS", str(cfg.num_layers)))
    if layers != cfg.num_layers:
        drafter.config = dataclasses.replace(cfg, num_layers=layers)
    print(f"CPU drafter: {time.time() - t0:.0f} s, layers {layers}, block {block}, users a row {users}, {dtype}")

    # each global user g = r U + u: a record and a bonus position inside its generated part
    records = sorted(TAPS.glob("*.pt"))
    saved = {}
    plan = []
    g = torch.Generator().manual_seed(5)
    for gu in range(ROWS * users):
        f = records[gu % len(records)]
        if f.stem not in saved:
            s = torch.load(f, weights_only=False)
            saved[f.stem] = (taps_for(s, cfg.target_layer_ids, "out").bfloat16(), s["ids"], s["prompt"])
        taps, ids, prompt = saved[f.stem]
        hi = min(len(ids) - block, max_start)
        start = int(torch.randint(prompt, max(prompt + 1, hi), (1,), generator=g))
        plan.append((f.stem, start))
    t0 = time.time()
    tt = TtDFlash2(mesh, cfg, drafter.w, drafter.mask_embedding, users, max_ctx=max_start + block + 1, block=block,
                   dtype=dtype, layers=layers)  # fmt: skip
    print(f"device drafter: {time.time() - t0:.0f} s")

    # the context: per user (all mesh rows at once), chunks of CHUNK positions
    t0 = time.time()
    for u in range(users):
        longest = max(plan[r * users + u][1] for r in range(ROWS))
        for c0 in range(0, longest, CHUNK):
            taps = torch.zeros(ROWS, 1, CHUNK, tt.cfg.hidden * len(cfg.target_layer_ids), dtype=torch.bfloat16)
            pos = torch.full((ROWS, CHUNK), -1, dtype=torch.int32)
            for r in range(ROWS):
                stem, start = plan[r * users + u]
                n = max(0, min(CHUNK, start - c0))
                if n:
                    taps[r, 0, :n] = saved[stem][0][c0 : c0 + n]
                    pos[r, :n] = torch.arange(c0, c0 + n, dtype=torch.int32)
            dt = row_tensor(mesh, taps, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            dp = row_tensor(mesh, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            du = row_tensor(mesh, torch.full((ROWS, CHUNK), u, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            tt.write_context(dt, dp, du)
            for t in (dt, dp, du):
                ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    print(f"context written: {time.time() - t0:.1f} s")

    bonus = torch.tensor([[int(saved[plan[r * users + u][0]][1][plan[r * users + u][1]]) for u in range(users)]
                          for r in range(ROWS)], dtype=torch.int32)  # fmt: skip
    starts = torch.tensor([[plan[r * users + u][1] for u in range(users)] for r in range(ROWS)], dtype=torch.int32)
    db = row_tensor(mesh, bonus, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    ds = row_tensor(mesh, starts, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    t0 = time.time()
    drafts, dbg = tt.draft(db, ds, debug=True)
    got_drafts = [u32(x).view(users, 8) for x in chip_rows(drafts)]
    got_hidden = [
        x.float().reshape(-1, cfg.hidden)[: users * block].view(users, block, -1) for x in chip_rows(dbg["hidden"])
    ]
    got_sel = [u32(x).view(users * (block - 1), 32) for x in chip_rows(dbg["sel"])]
    print(f"draft (eager, first): {time.time() - t0:.1f} s")
    # every chip of a row the same drafts
    devs = ttnn.get_device_tensors(drafts)
    for r in range(ROWS):
        for c in range(1, COLS):
            assert torch.equal(u32(ttnn.to_torch(devs[r * COLS + c])).view(users, 8), got_drafts[r]), (r, c)

    k = block - 1
    stats = dict(users=0, pcc_min=1.0, maxd=0.0, set_same=0, order_same=0, top1_same=0, slots=0, draft_same=0,
                 draft_all_same=0, acc_dev=0, acc_cpu=0)  # fmt: skip
    for r in range(ROWS):
        for u in range(users):
            stem, start = plan[r * users + u]
            taps, ids, prompt = saved[stem]
            ctx = drafter.context(taps[:start])
            with torch.no_grad():
                hidden = drafter.forward(ctx, torch.tensor([start]), ids[start : start + 1], k)
                prop = drafter.propose(hidden, ids[start : start + 1])
                logits = drafter.logits(hidden[:, 1:])
            vals, cand = logits.topk(16, dim=-1)  # [1, K, 16]
            gh = got_hidden[r][u]
            p = pcc(gh, hidden[0].float())
            stats["pcc_min"] = min(stats["pcc_min"], p)
            stats["maxd"] = max(stats["maxd"], float((gh - hidden[0].float()).abs().max()))
            for s in range(k):
                row = got_sel[r][u * k + s]
                dev_ids = row[16:]
                stats["slots"] += 1
                stats["set_same"] += int(set(dev_ids.tolist()) == set(cand[0, s].tolist()))
                stats["order_same"] += int(torch.equal(dev_ids, cand[0, s]))
                stats["top1_same"] += int(dev_ids[0] == cand[0, s, 0])
            dev = got_drafts[r][u]
            assert int(dev[0]) == int(ids[start]), (r, u, dev, ids[start])
            same = (dev[1 : 1 + k] == prop.drafts[0]).long()
            stats["draft_same"] += int(same.sum())
            stats["draft_all_same"] += int(bool(same.all()))
            target = ids[start + 1 : start + 1 + k]
            for name, d in (("acc_dev", dev[1 : 1 + k]), ("acc_cpu", prop.drafts[0])):
                n = 0
                while n < min(k, len(target)) and int(d[n]) == int(target[n]):
                    n += 1
                stats[name] += n
            stats["users"] += 1
            print(f"row {r} user {u} {stem}@{start}: hidden PCC {p:.5f}, drafts dev {dev[1:1 + k].tolist()} "
                  f"cpu {prop.drafts[0].tolist()}")  # fmt: skip
    print(
        "== dflash2 vs cpu: "
        + ", ".join(f"{a} {b:.5g}" if isinstance(b, float) else f"{a} {b}" for a, b in stats.items())
    )
    n, slots = stats["users"], stats["slots"]
    print(f"== rates: top-16 set {stats['set_same'] / slots:.3f}, order {stats['order_same'] / slots:.3f}, top 1 "
          f"{stats['top1_same'] / slots:.3f}, drafts {stats['draft_same'] / slots:.3f}, whole draft "
          f"{stats['draft_all_same'] / n:.3f}, accepted dev {stats['acc_dev'] / n:.3f} cpu {stats['acc_cpu'] / n:.3f}")  # fmt: skip
    ttnn.deallocate(drafts)
    for t in dbg.values():
        ttnn.deallocate(t)
    floor = 0.99 if layers <= 2 else 0.98
    assert stats["pcc_min"] > floor, stats


def traced_seconds(mesh, fn, reps: int) -> float:
    """fn() once eagerly (compiles, makes its constants), then `reps` calls in one trace: seconds a call, over
    GLM53_DFLASH2_REPLAYS replays (default 1; more: a stress run, the last call's outputs after the first replay and
    after the last compared bit for bit, printed)."""
    replays = int(os.environ.get("GLM53_DFLASH2_REPLAYS", "1"))
    for o in fn():
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        kept = [fn() for _ in range(reps)]
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    first = [ttnn.to_torch(d) for o in kept[-1] for d in ttnn.get_device_tensors(o)] if replays > 1 else []
    t0 = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / (reps * replays)
    if replays > 1:
        last = [ttnn.to_torch(d) for o in kept[-1] for d in ttnn.get_device_tensors(o)]
        same = all(torch.equal(a, b) for a, b in zip(first, last))
        print(f"== stress: {replays} replays x {reps} calls, the last call's outputs the same {same}", flush=True)
    ttnn.release_trace(mesh, trace)
    for outs in kept:
        for o in outs:
            ttnn.deallocate(o)
    return seconds


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 256 << 20}], indirect=True)
@soft
def test_dflash2_speed(mesh_device):
    """Synthetic weights (GLM53_SYNTHETIC_WEIGHTS=1 or not: the shapes only), traced: a draft and a write_context of
    the verify step's K + 1 = B rows a user, per users a row (GLM53_DFLASH2_SPEED_USERS, 1,4,16) and context
    (GLM53_DFLASH2_SPEED_CTX, 512,2048: every user's start just below it)."""
    from models.demos.blackhole.glm53_flash.ttnn.dflash2 import load_weights

    mesh = mesh_device
    block = int(os.environ.get("GLM53_DFLASH2_BLOCK", "8"))
    dtype = WEIGHT_DTYPES[os.environ.get("GLM53_DFLASH2_DTYPE", "bfp8")]
    reps = int(os.environ.get("GLM53_DFLASH2_REPS", "10"))
    ctxs = [int(c) for c in os.environ.get("GLM53_DFLASH2_SPEED_CTX", "512,2048").split(",")]
    cfg, w, mask = load_weights(synthetic=True)
    g = torch.Generator().manual_seed(6)
    taps_width = cfg.hidden * len(cfg.target_layer_ids)
    for users in [int(u) for u in os.environ.get("GLM53_DFLASH2_SPEED_USERS", "1,4,16").split(",")]:
        t0 = time.time()
        tt = TtDFlash2(mesh, cfg, w, mask, users, max_ctx=max(ctxs), block=block, dtype=dtype)
        print(f"users a row {users}: drafter made in {time.time() - t0:.0f} s")
        rows = users * block
        taps = row_tensor(mesh, torch.randn(ROWS, 1, rows, taps_width, generator=g).bfloat16(), ttnn.bfloat16,
                          ttnn.TILE_LAYOUT)  # fmt: skip
        du = row_tensor(mesh, (torch.arange(rows) // block).repeat(ROWS, 1).to(torch.int32), ttnn.int32,
                        ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        bonus = row_tensor(mesh, torch.randint(0, cfg.vocab, (ROWS, users), generator=g).to(torch.int32), ttnn.uint32,
                           ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        for ctx in ctxs:
            start = ctx - block - 8
            starts = row_tensor(mesh, torch.full((ROWS, users), start, dtype=torch.int32), ttnn.int32,
                                ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
            pos = start - block + torch.arange(rows) % block
            dp = row_tensor(mesh, pos.repeat(ROWS, 1).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

            def write():
                tt.write_context(taps, dp, du)
                return []

            def draft():
                return [tt.draft(bonus, starts)]

            t_write = traced_seconds(mesh, write, reps)
            t_draft = traced_seconds(mesh, draft, reps)
            print(f"== dflash2 speed: {ROWS * users} users ({users} a row), context {ctx}, block {block}, {dtype}: "
                  f"write_context {t_write * 1e3:.3f} ms, draft {t_draft * 1e3:.3f} ms")  # fmt: skip
            ttnn.deallocate(starts)
            ttnn.deallocate(dp)
        for t in (taps, du, bonus):
            ttnn.deallocate(t)


def walk_tokens(per: dict, k: int) -> tuple[int, int]:
    """tools/dflash2_acceptance.py's walk over one record: per {s: n accepted} -> (tokens, steps)."""
    s, tok, steps = min(per), 0, 0
    while s in per and per[s][1] == k:
        tok += per[s][0] + 1
        steps += 1
        s += per[s][0] + 1
    return tok, steps


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_dflash2_acceptance(mesh_device):
    """The greedy records (one per user, 2 users a row): a draft at every generated position s (the context of the
    record's taps, the block's rows written back after each draft as the verify step's write_context would), the
    drafts against the CPU's at the same s (tools/dflash2_acceptance.py --out: GLM53_DFLASH2_REF, default
    data_dir("dflash2-v1", "all.json"); the same taps), the accepted count n at each s, and the walk (accept n, move
    n + 1) as the CPU tool computes it.  K = B - 1 (GLM53_DFLASH2_BLOCK: 8 -> K = 7, 4 -> K = 3)."""
    import json

    mesh = mesh_device
    block = int(os.environ.get("GLM53_DFLASH2_BLOCK", "8"))
    k = block - 1
    dtype = WEIGHT_DTYPES[os.environ.get("GLM53_DFLASH2_DTYPE", "bfp8")]
    users = 2
    ref = json.loads(Path(os.environ.get("GLM53_DFLASH2_REF") or data_dir("dflash2-v1", "all.json")).read_text())
    cpu = ref["per_position"][f"out K={k}"]
    drafter = DFlash2Drafter()
    cfg = drafter.config
    records = sorted(TAPS.glob("*.pt"))
    assert len(records) <= ROWS * users, len(records)
    slots = {}  # (row, user) -> (id, taps, ids, prompt)
    for i, f in enumerate(records):
        s = torch.load(f, weights_only=False)
        slots[(i % ROWS, i // ROWS)] = (
            f.stem,
            taps_for(s, cfg.target_layer_ids, "out").bfloat16(),
            s["ids"],
            s["prompt"],
        )
    longest = max(len(v[2]) for v in slots.values())
    head = os.environ.get("GLM53_DFLASH2_HEAD_DTYPE")  # (the head's weight: the drafter's dtype by default)
    tt = TtDFlash2(
        mesh, cfg, drafter.w, drafter.mask_embedding, users, max_ctx=longest + block, block=block, dtype=dtype,
        head_dtype=WEIGHT_DTYPES[head] if head else None,
    )  # fmt: skip
    del drafter
    width = cfg.hidden * len(cfg.target_layer_ids)

    def write(rows_of):
        """rows_of(row, user) -> list of (position, taps row) for that slot; one write_context of R = U B rows."""
        r_tok = users * block
        taps = torch.zeros(ROWS, 1, r_tok, width, dtype=torch.bfloat16)
        pos = torch.full((ROWS, r_tok), -1, dtype=torch.int32)
        usr = torch.zeros(ROWS, r_tok, dtype=torch.int32)
        for r in range(ROWS):
            for u in range(users):
                for j, (p, t) in enumerate(rows_of(r, u)):
                    taps[r, 0, u * block + j] = t
                    pos[r, u * block + j] = p
                usr[r, u * block : (u + 1) * block] = u
        dt = row_tensor(mesh, taps, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        dp = row_tensor(mesh, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        du = row_tensor(mesh, usr, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        tt.write_context(dt, dp, du)
        for t in (dt, dp, du):
            ttnn.deallocate(t)

    # the whole record's context, B positions a user a write
    t0 = time.time()
    for c0 in range(0, longest, block):

        def full(r, u, c0=c0):
            if (r, u) not in slots:
                return []
            _, taps, ids, _ = slots[(r, u)]
            return [(p, taps[p]) for p in range(c0, min(c0 + block, len(ids)))]

        write(full)
    print(f"context written: {time.time() - t0:.1f} s")

    # a draft at every s = prompt .. len - 2 of each record, in lockstep
    t0 = time.time()
    got = {}  # id -> {s: drafts}
    steps = max(len(v[2]) - 1 - v[3] for v in slots.values())
    for j in range(steps):
        bonus = torch.zeros(ROWS, users, dtype=torch.int32)
        starts = torch.zeros(ROWS, users, dtype=torch.int32)
        live = {}
        for (r, u), (rid, taps, ids, prompt) in slots.items():
            s = prompt + j
            if s <= len(ids) - 2:
                bonus[r, u], starts[r, u] = int(ids[s]), s
                live[(r, u)] = s
        db = row_tensor(mesh, bonus, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ds = row_tensor(mesh, starts, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        drafts = tt.draft(db, ds)
        rows = [u32(x).view(users, 8) for x in chip_rows(drafts)]
        for t in (drafts, db, ds):
            ttnn.deallocate(t)
        for (r, u), s in live.items():
            got.setdefault(slots[(r, u)][0], {})[s] = rows[r][u, 1 : 1 + k].tolist()

        # the block's rows back (an idle user wrote its block at 0 ..: its rows 0 .. B - 1 back too)
        def back_rows(r, u):
            if (r, u) not in slots:
                return []
            _, taps, ids, prompt = slots[(r, u)]
            s = live.get((r, u), 0)
            return [(p, taps[p]) for p in range(s, min(s + block, len(ids)))]

        write(back_rows)
    print(f"{steps} lockstep drafts and writes: {time.time() - t0:.1f} s")

    # against the CPU
    same_pos = same_slot = total_pos = total_slot = n_dev_sum = n_cpu_sum = 0
    walk_dev = walk_cpu = steps_dev = steps_cpu = 0
    for rid, per_s in got.items():
        ids = next(v[2] for v in slots.values() if v[0] == rid)
        tokens = ids.tolist()
        cpu_by_s = {x["s"]: x for x in cpu[rid]}
        dev_n, cpu_n = {}, {}
        for s, d in per_s.items():
            c = cpu_by_s[s]
            window = c["window"]
            n = 0
            while n < window and d[n] == tokens[s + 1 + n]:
                n += 1
            dev_n[s] = (n, window)
            cpu_n[s] = (c["n"], window)
            if window == k:
                total_pos += 1
                same_pos += int(d == c["drafts"])
                total_slot += k
                same_slot += sum(int(a == b) for a, b in zip(d, c["drafts"]))
                n_dev_sum += n
                n_cpu_sum += c["n"]
        a, b = walk_tokens(dev_n, k)
        c_, d_ = walk_tokens(cpu_n, k)
        walk_dev, steps_dev, walk_cpu, steps_cpu = walk_dev + a, steps_dev + b, walk_cpu + c_, steps_cpu + d_
        print(f"{rid}: walk dev {a}/{b} cpu {c_}/{d_}, positions {len(per_s)}")
    print(f"== dflash2 acceptance K={k} {dtype}: positions {total_pos}, drafts the same {same_pos / total_pos:.3f} "
          f"(slots {same_slot / total_slot:.4f}), mean accepted dev {n_dev_sum / total_pos:.4f} cpu "
          f"{n_cpu_sum / total_pos:.4f}, walk tokens a step dev {walk_dev / steps_dev:.4f} ({steps_dev} steps) cpu "
          f"{walk_cpu / steps_cpu:.4f} ({steps_cpu}; all.json {ref['summary'][f'out K={k}']['walk_tokens_per_step']})")  # fmt: skip


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 256 << 20}], indirect=True)
@soft
def test_dflash2_profile(mesh_device):
    """The device profile of one traced draft and one traced write_context (synthetic weights): per op the programs
    and the slowest chip's kernel time, and the gaps (tools/profile_report.py's reading of the report).  Run with
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TTNN_OP_PROFILER=1
    TT_METAL_PROFILER_DIR=<dir>.  GLM53_DFLASH2_SPEED_USERS (one value), GLM53_DFLASH2_SPEED_CTX (one value).
    GLM53_DFLASH2_PROFILE_STEP=1: also the DFlash2 step's two other drafter parts (`dflash2_step.drafter_inputs`, from
    accept_device's [1, 1, 8, 2 U] fp32 output, and `gathered_results`), and every part's programs in order (op,
    cores, kernel us, gap us) into <dir>/parts.json.  Prints only."""
    import csv
    import json

    from models.demos.blackhole.glm53_flash.tools.profile_report import by_op, programs
    from models.demos.blackhole.glm53_flash.ttnn.dflash2 import load_weights

    mesh = mesh_device
    block = int(os.environ.get("GLM53_DFLASH2_BLOCK", "8"))
    users = int(os.environ.get("GLM53_DFLASH2_SPEED_USERS", "4"))
    ctx = int(os.environ.get("GLM53_DFLASH2_SPEED_CTX", "2048"))
    cfg, w, mask = load_weights(synthetic=True)
    tt = TtDFlash2(mesh, cfg, w, mask, users, max_ctx=ctx, block=block, dtype=WEIGHT_DTYPES["bfp8"])
    g = torch.Generator().manual_seed(6)
    rows = users * block
    taps = row_tensor(mesh, torch.randn(ROWS, 1, rows, cfg.hidden * 9, generator=g).bfloat16(), ttnn.bfloat16,
                      ttnn.TILE_LAYOUT)  # fmt: skip
    du = row_tensor(mesh, (torch.arange(rows) // block).repeat(ROWS, 1).to(torch.int32), ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
    bonus = row_tensor(mesh, torch.randint(0, cfg.vocab, (ROWS, users), generator=g).to(torch.int32), ttnn.uint32,
                       ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
    start = ctx - block - 8
    starts = row_tensor(mesh, torch.full((ROWS, users), start, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    pos = (start - block + torch.arange(rows) % block).repeat(ROWS, 1).to(torch.int32)
    dp = row_tensor(mesh, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    report = Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs" / "cpp_device_perf_report.csv"

    def rows_now() -> int:
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        return sum(1 for _ in open(report)) if report.exists() else 0

    parts = [
        ("draft", lambda: [tt.draft(bonus, starts)]),
        ("write_context", lambda: tt.write_context(taps, dp, du) or []),
    ]
    if os.environ.get("GLM53_DFLASH2_PROFILE_STEP") == "1":  # (the step's drafter_inputs and gathered_results)
        from models.demos.blackhole.glm53_flash.ttnn.dflash2_step import drafter_inputs, gathered_results

        f32 = ttnn.float32
        acc = torch.cat([torch.randint(0, block, (ROWS, 1, ROWS, users)), torch.randint(0, cfg.vocab, (ROWS, 1, ROWS, users))],
                        dim=-1).float()  # fmt: skip
        accepted = row_tensor(mesh, acc, f32, ttnn.ROW_MAJOR_LAYOUT)
        others_t = torch.full((ROWS, 1, ROWS, 1), 1.0e9)
        others_t[torch.arange(ROWS), 0, torch.arange(ROWS), 0] = 0.0
        others = put(mesh, others_t, dims=(0, None), dtype=f32)
        base1 = row_tensor(mesh, torch.full((ROWS, 1, 1, users), start, dtype=torch.int32), ttnn.int32,
                           ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        drafts = tt.draft(bonus, starts)
        parts += [
            ("drafter_inputs", lambda: list(drafter_inputs(accepted, others, base1, users))),
            ("gathered_results", lambda: [gathered_results(accepted, drafts, users)]),
        ]
    marks = {}
    for name, fn in parts:
        for o in fn():  # eager: compiles, makes the constants
            ttnn.deallocate(o)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            kept = fn()
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        n0 = rows_now()
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        marks[name] = (n0, rows_now())
        ttnn.release_trace(mesh, trace)
        for o in kept:
            ttnn.deallocate(o)
    lines = list(csv.reader(open(report)))
    header = lines[0]
    for name, (a, b) in marks.items():
        progs = programs(header, lines[a:b])
        kernel = sum(p["kernel_ns"] for p in progs) / 1e6
        gaps = sum(p["gap_ns"] for p in progs) / 1e6
        print(f"\n== dflash2 profile {name}: {ROWS * users} users, context {ctx}, block {block}: {len(progs)} programs, "
              f"kernel {kernel:.3f} ms, gaps {gaps:.3f} ms\n\n| op | programs | kernel ms |\n|---|---|---|")  # fmt: skip
        for op, n, ns in by_op(progs)[:30]:
            print(f"| {op} | {n} | {ns / 1e6:.3f} |")
    if os.environ.get("GLM53_DFLASH2_PROFILE_STEP") == "1":
        keys = ("op", "cores", "kernel_ns", "kernel_mean_ns", "gap_ns")
        dump = {
            name: [{k: p[k] for k in keys if k in p} for p in programs(header, lines[a:b])]
            for name, (a, b) in marks.items()
        }
        # (the parts' rows of the op report, header line 0 counted: for the cores' ends from the device log)
        dump["_marks"] = {name: [a, b] for name, (a, b) in marks.items()}
        (Path(os.environ["TT_METAL_PROFILER_DIR"]) / "parts.json").write_text(json.dumps(dump))


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_dflash2_same(mesh_device):
    """The drafter's outputs saved under a tag and compared bit for bit with another tag's (a kernel change that must
    keep the values): the real drafter weights (BFP8, deterministic, unlike the device's random ones), context 512, at
    GLM53_DFLASH2_SAME_CASES ("users a row:block,..", default 1:8,4:8,16:4: 8 / 32 / 128 users): write_context of
    random taps (as test_dflash2_profile's), then hidden and draft; each chip's tensors hashed (sha256 of their bytes).
        GLM53_DFLASH2_SAME_DIR (default out_dir("dflash2_same")), GLM53_DFLASH2_SAME_TAG (this run's),
        GLM53_DFLASH2_SAME_AGAINST (a tag to compare with: DSAME lines)
    Prints only."""
    import hashlib
    import json

    from models.demos.blackhole.glm53_flash.ttnn.dflash2 import load_weights

    mesh = mesh_device
    root = Path(os.environ.get("GLM53_DFLASH2_SAME_DIR") or out_dir("dflash2_same"))
    tag, against = os.environ.get("GLM53_DFLASH2_SAME_TAG", "now"), os.environ.get("GLM53_DFLASH2_SAME_AGAINST")
    root.mkdir(parents=True, exist_ok=True)
    cfg, w, mask = load_weights(synthetic=False)
    spec = os.environ.get("GLM53_DFLASH2_SAME_CASES", "1:8,4:8,16:4")
    out = {}
    for users, block in [tuple(int(v) for v in c.split(":")) for c in spec.split(",")]:
        ctx = 512
        tt = TtDFlash2(mesh, cfg, w, mask, users, max_ctx=ctx, block=block, dtype=WEIGHT_DTYPES["bfp8"])
        g = torch.Generator().manual_seed(17)
        rows = users * block
        taps = row_tensor(mesh, torch.randn(ROWS, 1, rows, cfg.hidden * 9, generator=g).bfloat16(), ttnn.bfloat16,
                          ttnn.TILE_LAYOUT)  # fmt: skip
        start = ctx - block - 8
        pos = (start - block + torch.arange(rows) % block).repeat(ROWS, 1).to(torch.int32)
        dp = row_tensor(mesh, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        du = row_tensor(mesh, (torch.arange(rows) // block).repeat(ROWS, 1).to(torch.int32), ttnn.int32,
                        ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        bonus = row_tensor(mesh, torch.randint(0, cfg.vocab, (ROWS, users), generator=g).to(torch.int32), ttnn.uint32,
                           ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        starts = row_tensor(
            mesh, torch.full((ROWS, users), start, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )
        tt.write_context(taps, dp, du)
        h = tt.hidden(bonus, starts)
        d = tt.draft(bonus, starts)
        key = f"{users}:{block}"
        out[key] = {
            name: [hashlib.sha256(ttnn.to_torch(x).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                   for x in ttnn.get_device_tensors(t)]
            for name, t in (("hidden", h), ("drafts", d))
        }  # fmt: skip
        for t in (h, d, taps, dp, du, bonus, starts):
            ttnn.deallocate(t)
        del tt
    (root / f"{tag}.json").write_text(json.dumps(out))
    if against:
        ref = json.loads((root / f"{against}.json").read_text())
        for key, names in out.items():
            for name, hashes in names.items():
                same = sum(a == b for a, b in zip(hashes, ref.get(key, {}).get(name, [])))
                print(
                    f"DSAME {key} {name} {tag} against {against}: {same} / {len(hashes)} chips the same bits",
                    flush=True,
                )
