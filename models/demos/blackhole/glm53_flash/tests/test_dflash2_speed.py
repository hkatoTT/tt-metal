"""The DFlash2 step's traced time: `ttnn/dflash2_step.py`'s step (the target's verify step,
`forward_verify`, T = K + 1 tokens a user; its acceptance on the device; the drafter's context write of the step's own
taps, K + 1 rows a user; the drafter's inputs from the acceptance and its next block, `draft`) in one trace, at
GLM53_VERIFY_USERS users a row (1,4,16: 8 / 32 / 128 users); against the verify step alone, the drafter's two alone and
the MTP verify step of one draft (`forward_spec`) on the same load.  The step's inputs written from the host before
each replay (random tokens, the drafts all kept, as test_verify_device.py); the drafter's inputs fixed (its users'
blocks just below the cache's end).  The tokens/s at GLM53_DFLASH2_TOKENS_A_STEP tokens a verify step (the walk's on
the greedy records, tools/dflash2_acceptance.py) against the MTP's at its acceptance 0.884.  Synthetic weights
(GLM53_SYNTHETIC_WEIGHTS=1).

    GLM53_DFLASH2_K (3), GLM53_DFLASH2_PAIRS (only these "users a row:K" pairs), GLM53_DFLASH2_TOKENS_A_STEP
    (3.36: the measured one at K = 3), GLM53_DFLASH2_MTP_MS_<users> (the MTP step's ms at that many users, given: not
    run), GLM53_DFLASH2_ARMS ("": the defaults; ";"-separated
    "name@NAME=value+NAME=value", environment variables a program factory reads, e.g. GLM53_MOE_PROG_F: each arm
    timed in turn on the one load, the program cache cleared before and after it)
    GLM53_DEVICE_TEST=1 GLM53_SYNTHETIC_WEIGHTS=1 pytest models/demos/blackhole/glm53_flash/tests/test_dflash2_speed.py -s
"""

from __future__ import annotations

import gc
import json
import os
import time
import traceback

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.tests.test_verify_device import MAX_SEQ, release, timed_steps  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS, put  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dflash2_step import DFlash2Step  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

MTP_TOKENS_A_STEP = (
    1.884  # the MTP verify step's at its acceptance 0.884 (real weights, measured on the greedy records)
)


def guarded(run) -> None:
    """A study's run: an exception is printed, not raised (the run ends cleanly, with exit code 0), raised with
    GLM53_DFLASH2_STRICT=1."""
    try:
        run()
    except Exception:
        if os.environ.get("GLM53_DFLASH2_STRICT") == "1":
            raise
        traceback.print_exc()
        print("STUDY FAILED (printed, not raised)", flush=True)


def dram_used(mesh) -> int:
    """One chip's DRAM allocated, bytes (every chip the same)."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(view.total_bytes_allocated_per_bank) * int(view.num_banks)


def arms() -> list:
    """GLM53_DFLASH2_ARMS -> [(name, [(NAME, value) ..])]; unset: one arm, the defaults."""
    out = []
    for a in os.environ.get("GLM53_DFLASH2_ARMS", "").split(";"):
        name, _, env = a.partition("@")
        out.append((name, [tuple(p.split("=", 1)) for p in env.split("+") if p]))
    return out


def with_env(mesh, env: list, run):
    """`run()` with these environment variables set (the program cache cleared so that the programs are made again)."""
    if not env:
        return run()
    undo = [(k, os.environ.get(k)) for k, _ in env]
    for k, v in env:
        os.environ[k] = v
    mesh.clear_program_cache()
    try:
        return run()
    finally:
        for k, v in reversed(undo):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        mesh.clear_program_cache()


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dflash2_speed(mesh_device):
    guarded(lambda: _dflash2_speed(mesh_device))


def _dflash2_speed(mesh_device):
    mesh = mesh_device
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    ks = [int(v) for v in os.environ.get("GLM53_DFLASH2_K", "3").split(",")]
    # GLM53_DFLASH2_PAIRS ("1:7,4:7,16:3"): the fixed points' (users a row, K) pairs only, on one load
    pairs = {tuple(int(x) for x in p.split(":")) for p in os.environ.get("GLM53_DFLASH2_PAIRS", "").split(",") if p}
    # tokens a verify step per K ("3:3.16,7:4.2"; one value: every K)
    given = os.environ.get("GLM53_DFLASH2_TOKENS_A_STEP", "3.36")
    per_step = (
        {int(a): float(b) for a, b in (kv.split(":") for kv in given.split(","))}
        if ":" in given
        else {k: float(given) for k in ks}
    )
    parts_wanted = os.environ.get("GLM53_DFLASH2_PARTS", "verify,drafter,step").split(",")
    # the drafter's weights random (its time does not depend on them: no routing) with the target's real
    drafter_synthetic = synthetic or os.environ.get("GLM53_DFLASH2_DRAFTER_SYNTHETIC") == "1"
    steps = int(os.environ.get("GLM53_TRACE_SPEED_STEPS", "12"))
    vstart = int(
        os.environ.get("GLM53_TRACE_START", "0")
    )  # the verify step's first position (a long cache: near its end)
    t0 = time.time()
    checkpoint = ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint()
    model = TtGlm53Model(mesh, checkpoint, max_seq=MAX_SEQ, mtp=True)
    cfg, w, mask = load_weights(synthetic=drafter_synthetic)
    model.tap_layers = frozenset(layer + 1 for layer in cfg.target_layer_ids)
    result = {"synthetic_weights": synthetic, "drafter_synthetic": drafter_synthetic, "ks": ks, "speed": {}}
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    rm = dict(dims=(0, None), layout=ttnn.ROW_MAJOR_LAYOUT)
    for u in [int(x) for x in os.environ.get("GLM53_VERIFY_USERS", "1,4,16").split(",")]:
        users = ROWS * u
        line = result["speed"][f"{users}_users"] = {}
        g = torch.Generator().manual_seed(3000 + u)
        mtp = float(os.environ.get(f"GLM53_DFLASH2_MTP_MS_{users}", "0")) / 1e3  # (given: the MTP step not run)
        if not mtp:
            states = model.empty_state(u, spec=1)
            io = model.alloc_spec_io(u, drafts=1)
            mtp = timed_steps(mesh, model, io, 1, u, lambda: model.forward_spec(io, states), g, vstart, steps)
            release(states)
        line["mtp_k1_step_s"] = round(mtp, 5)
        mtp_tps = users * MTP_TOKENS_A_STEP / mtp
        print(f"DFLASH2 users={users} MTP K=1 (forward_spec): {mtp * 1e3:.2f} ms, {mtp_tps:.0f} tok/s", flush=True)
        for k in ks:
            if pairs and (u, k) not in pairs:  # (GLM53_DFLASH2_PAIRS: only these users-a-row:K pairs)
                continue
            n = (k + 1) * u  # the step's rows a mesh row: virtual user j U + u is user u's token j
            if n % 2:  # (write_context takes an even number of rows: the step's would need a pad row)
                print(f"DFLASH2 users={users} K={k}: skipped ({n} rows a mesh row, odd)", flush=True)
                continue
            block = k + 1 if 32 % (k + 1) == 0 else 8  # (the drafter's blocks: 2, 4, 8; K drafts of a block of 8)
            used0 = dram_used(mesh)
            dev = TtDFlash2(
                mesh, cfg, w, mask, users=u, max_ctx=MAX_SEQ, block=block, head_weight=model.lm_head.mesh_weight,
                embedding=model.embed,
            )  # fmt: skip
            line[f"K{k}_drafter_dram_mib"] = round((dram_used(mesh) - used0) / 2**20, 2)
            print(f"DFLASH2 users={users} K={k}: the drafter's DRAM a chip {line[f'K{k}_drafter_dram_mib']} MiB "
                  f"(K / V {os.environ.get('GLM53_DFLASH2_KV', 'bfp8')}, cache {MAX_SEQ})", flush=True)  # fmt: skip
            states = model.empty_state(u, spec=k)
            st = DFlash2Step(model, dev, u, k)
            io = st.io
            start = MAX_SEQ - 2 * (k + 1) - 8  # the drafter's block start; the step's tokens at start - K - 1 ..
            # the step's context positions and the drafter's base (fixed: test_verify_device's writes rewrite the rest)
            st.write(
                torch.randint(0, cfg.vocab, (ROWS, n), generator=g),
                (start - (k + 1) + torch.arange(n) // u).repeat(ROWS, 1),
                torch.full((ROWS, u), k),
            )
            dp = put(mesh, (start - (k + 1) + torch.arange(n) // u).repeat(ROWS, 1).int(), dtype=ttnn.int32, **rm)
            du = put(mesh, (torch.arange(n) % u).repeat(ROWS, 1).int(), dtype=ttnn.int32, **rm)
            db = put(mesh, torch.randint(0, cfg.vocab, (ROWS, u), generator=g).int(), dtype=ttnn.uint32, **rm)
            ds = put(mesh, torch.full((ROWS, u), start, dtype=torch.int32), dtype=ttnn.int32, **rm)
            fixed_taps = [
                put(mesh, torch.randn(ROWS, 1, n, cfg.hidden, generator=g).bfloat16(), dims=(0, None))
                for _ in cfg.target_layer_ids
            ]

            def drafter():
                dev.write_context(fixed_taps, dp, du)
                return dev.draft(db, ds)

            parts = {
                "verify": lambda: model.forward_verify(io, states),
                "drafter": drafter,
                "step": lambda: st._run(states),
            }
            for arm, env in arms():
                times = {}
                tag = f"_{arm}" if arm else ""  # (the defaults: the keys and the line as before)

                def time_parts():
                    for name in parts_wanted:
                        times[name] = timed_steps(mesh, model, io, k, u, parts[name], g, vstart, steps)

                with_env(mesh, env, time_parts)
                for name in parts_wanted:
                    line[f"K{k}_{name}{tag}_s"] = round(times[name], 5)
                alone = "".join(f", {name} {times[name] * 1e3:.2f}" for name in parts_wanted if name != "step")
                note = ""
                if k in per_step and "step" in times:
                    tps = users * per_step[k] / times["step"]
                    line[f"K{k}{tag}_tokens_s"] = round(tps)
                    note = f"; at {per_step[k]} tokens a step {tps:.0f} tok/s against the MTP's {mtp_tps:.0f} ({tps / mtp_tps - 1:+.1%})"
                print(
                    f"DFLASH2 users={users} K={k} (block {block})"
                    + (f" [{arm}]" if arm else "")
                    + ": "
                    + (f"step {times['step'] * 1e3:.2f} ms" if "step" in times else "(no step)")
                    + f"{alone}{note}",
                    flush=True,
                )
            release(states)
            st.release()
            for t in fixed_taps + [dp, du, db, ds]:
                ttnn.deallocate(t)
            del dev
            gc.collect()
    path = out_dir() / f"dflash2-speed-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}", flush=True)
