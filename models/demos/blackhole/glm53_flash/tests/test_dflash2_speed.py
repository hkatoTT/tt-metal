"""The DFlash2 step's traced time: `ttnn/dflash2_step.py`'s step (the target's verify step,
`forward_verify`, T = K + 1 tokens a user; its acceptance on the device; the drafter's context write of the step's own
taps, K + 1 rows a user; the drafter's inputs from the acceptance and its next block, `draft`) in one trace, at
GLM53_VERIFY_USERS users a row (1,4,16: 8 / 32 / 128 users); against the verify step alone, the drafter's two alone and
the MTP verify step of one draft (`forward_spec`) on the same load.  The step's inputs written from the host before
each replay (random tokens, the drafts all kept, as test_verify_device.py); the drafter's inputs fixed (its users'
blocks just below the cache's end).  The tokens/s at GLM53_DFLASH2_TOKENS_A_STEP tokens a verify step (the walk's on
the greedy records, tools/dflash2_acceptance.py) against the MTP's at its acceptance 0.884.  Synthetic weights
(GLM53_SYNTHETIC_WEIGHTS=1).

    GLM53_DFLASH2_PARTS (default verify,drafter,step; with GLM53_DFLASH2_SAMPLING=1 also step_greedy, the step's
    greedy form, and step_drawn, its drawing form with every seat sampled (top-k 32, top-p 0.95, temperature 1): the
    two forms GLM53_DFLASH2_SAMPLING_SPLIT switches between; step then is the drawing form with greedy seats, the form
    before the split)
    GLM53_DFLASH2_K (3), GLM53_DFLASH2_PAIRS (only these "users a row:K" pairs), GLM53_DFLASH2_TOKENS_A_STEP
    (3.36: the measured one at K = 3), GLM53_DFLASH2_MTP_MS_<users> (the MTP step's ms at that many users, given: not
    run), GLM53_DFLASH2_ARMS ("": the defaults; ";"-separated
    "name@NAME=value+NAME=value", environment variables a program factory reads, e.g. GLM53_MOE_PROG_F: each arm
    timed in turn on the one load, the program cache cleared before and after it)
    GLM53_L1_LANDING_KB (a probe: an L1 tensor of that many KB a core on the 12 x 10 grid, held from the load on;
    the circular buffers of every program run must fit below it)
    GLM53_W1_C (in an arm; ttnn/w1_landing.py): the L1 landing of the next MoE layer's experts' weights (GLM53_W1) of
    that many slots a core, its env armed for the arm's programs and its DRISCs run around each timed replay of the
    verify step and the step (their windows, skipped windows and the blocks taken from the landing / read again printed)
    The model is made as served: with `dflash_shape` (users a row, K + 1) of the run's pairs (`serving_shape`: one
    shape a load, or the same defaults for all), so the expert-weight landing (its slots, MEASURED_ENV and the shape's
    switches) and the narrow landing are the served ones; the model's landing armed and run around the verify step and
    the step as Glm53Generator does.  An arm's env goes over them as before; an arm (or the process) that sets
    GLM53_W1_C / GLM53_W1 decides the landing itself (the model then without a landing of its own).
    GLM53_TEST_NO_SHAPE=1: the earlier form (no dflash_shape: the served defaults only where the arms write them)
    GLM53_W1_BITS (N; 0, the default: none): after each arm's parts, N traced verify steps (`forward_verify`, ids and
    logits) from fresh states and seeded inputs, the landing's DRISCs around them in a landing arm; the last step's ids
    and logits against the first arm's, bit for bit ("W1BITS same" / "W1BITS DIFFER"); with GLM53_DFLASH2_PARTS empty:
    the bits phase only (a stress: N = 2,000)
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
from models.demos.blackhole.glm53_flash.tests._trace import is_device_timeout  # noqa: E402
from models.demos.blackhole.glm53_flash.tests.test_verify_device import MAX_SEQ, release, timed_steps  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS, put  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dflash2_step import DFlash2Step  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.w1_landing import W1Landing  # noqa: E402

MTP_TOKENS_A_STEP = (
    1.884  # the MTP verify step's at its acceptance 0.884 (real weights, measured on the greedy records)
)


def guarded(run) -> None:
    """A study's run: an exception is printed, not raised (the run ends cleanly, with exit code 0), raised with
    GLM53_DFLASH2_STRICT=1."""
    try:
        run()
    except Exception as e:
        # (a device timeout is raised too: the run ends non-zero, the device to be checked before the next)
        if os.environ.get("GLM53_DFLASH2_STRICT") == "1" or is_device_timeout(e):
            raise
        traceback.print_exc()
        print("STUDY FAILED (printed, not raised)", flush=True)


def dram_used(mesh) -> int:
    """One chip's DRAM allocated, bytes (every chip the same)."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(view.total_bytes_allocated_per_bank) * int(view.num_banks)


def l1_landing(mesh):
    """GLM53_L1_LANDING_KB (a probe: does an L1 tensor of that many KB a core, held from the load on, leave room for
    every program of the run?): a uint32 L1 tensor of that size on each of the
    12 x 10 worker cores (height-sharded, one row a core), allocated now and held to the end; None when unset or 0."""
    kb = float(os.environ.get("GLM53_L1_LANDING_KB", "0"))
    if kb <= 0:
        return None
    words = int(kb * 1024) // 4
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9))])
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
    )
    t = ttnn.allocate_tensor_on_device(ttnn.Shape([120, words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mem)
    print(f"L1 landing: {words * 4} B a core on the 12 x 10 grid at {t.buffer_address():#x} (to "
          f"{t.buffer_address() + words * 4:#x})", flush=True)  # fmt: skip
    return t


class W1Arm:
    """GLM53_W1_C=C in an arm: the expert-weight landing of C slots made and the MoE layers armed with their specs
    (`W1Landing.arm_model`: the experts per mesh column, GLM53_W1_POPULAR's table or "0" the lowest ids; the down cores'
    depths from the arm's GLM53_MOE_DN_DEPTH / GLM53_MOE_H_SLOTS, 0: the op's) inside the arm (the program cache
    cleared, so the arm's programs take the landing), the DRISCs started and stopped around a part's replays
    (timed_steps' `around`), and what they did printed (blocks from the landing: every chip's)."""

    def __init__(self, mesh, model, c: int, mesh_tokens: int, label: str, landing=None):
        from models.demos.blackhole.glm53_flash.ttnn.moe import TtMoe

        moes = [layer.mlp for layer in model.layers if isinstance(layer.mlp, TtMoe)]
        # `landing`: the model's own (made at its load from `dflash_shape`, as served): armed and run here, kept after
        self.own = landing is None
        self.land = landing if landing is not None else W1Landing(mesh, c, dtype=moes[0].e_w1.dtype)  # (BFP4: prog)
        c = self.land.c
        self.model = model
        dn_env, hs_env = os.environ.get("GLM53_MOE_DN_DEPTH"), os.environ.get("GLM53_MOE_H_SLOTS")
        if self.own:  # (an arm's GLM53_W1_C: the down cores' depths from the arm's env, 0 the op's, as before)
            dn, hs = int(dn_env or 0), int(hs_env or 0)
        else:  # (the model's landing: arm_model's defaults as Glm53Generator, unless the arm sets them)
            dn, hs = (int(dn_env) if dn_env else None), (int(hs_env) if hs_env else None)
        specs = self.land.arm_model(model, mesh_tokens, dn_depth=dn, h_slots=hs)
        self.label = label
        self.part = ""
        first = specs[0][23:]
        print(f"W1 {label}: {len(specs)} MoE layers, C {c}, the first layer's experts by mesh column "
              f"{[first[i * c:(i + 1) * c] for i in range(len(first) // c)]} (mesh tokens {mesh_tokens}, popular table "
              f"{os.environ.get('GLM53_W1_POPULAR', 'w1_popular.json')}, dn {dn}, h {hs})", flush=True)  # fmt: skip

    def start(self):
        self.before = self.land.counts(range(4))
        self.land.start()

    def stop(self):
        stats = self.land.stop()
        d = [a - b for a, b in zip(self.land.counts(range(4)), self.before)]
        print(f"W1 {self.label} {self.part}: DRISC windows {stats.get('windows')}, skipped {stats.get('skipped')}, cut "
              f"{stats.get('cut')}, a window W1 {stats.get('w1_cycles')} + W2 {stats.get('w2_cycles')} cycles (chip 0, median); the first mesh "
              f"row's 4 chips' gate | up blocks from the landing {d[0]}, read again {d[1]}; down halves from the landing "
              f"{d[2]}, read again {d[3]}", flush=True)  # fmt: skip
        return stats

    def release(self):
        W1Landing.disarm_model(self.model)
        if self.own:
            self.land.release()


def shape_defaults(shape) -> tuple:
    """What TtGlm53Model sets by default from `dflash_shape` (the served form, `w1_landing.landing_c` and
    `common.ar_land_narrow_default`) at this process's env: (the landing's slots, the env it sets, the narrow
    landing)."""
    from models.demos.blackhole.glm53_flash.ttnn import w1_landing as wl

    prog = os.environ.get("GLM53_MOE_PROG_F") is not None
    given = os.environ.get("GLM53_W1", "")
    if given != "":
        c = int(given)
    else:
        c = (wl.PROG_MEASURED_C.get(shape, 0) if prog else wl.default_c(*shape)) if shape else 0
    measured = wl.PROG_MEASURED_C if prog else wl.MEASURED_C
    shapes = wl.PROG_MEASURED_ENV_SHAPE if prog else wl.MEASURED_ENV_SHAPE
    env = {}
    if given == "" and shape and c and shape in measured:
        env = {**wl.MEASURED_ENV, **shapes.get(shape, {})}
    narrow = bool(shape) and shape[0] * shape[1] <= 32
    return c, tuple(sorted(env.items())), narrow


def serving_shape(us: list, ks: list, pairs: set, arm_list: list):
    """The model's `dflash_shape` for this run, as served (glm53_vllm.w1_shape: (users a mesh row, K + 1)): the first
    (users a row, K) the run times when every one it times gets the same defaults from it (`shape_defaults`), else None
    (a warning: one shape a load); None with GLM53_TEST_NO_SHAPE=1 (the earlier form: the defaults only from the arms'
    env).  -> (shape, the landing by the arms): an arm (or the process) that sets GLM53_W1_C / GLM53_W1 decides the
    landing as before, so the model then makes no landing of its own (GLM53_W1=0 at its load); the rest of the shape's
    defaults (the narrow landing) stay, an arm's env over them."""
    if os.environ.get("GLM53_TEST_NO_SHAPE") == "1":
        print("SHAPE: none (GLM53_TEST_NO_SHAPE=1): the model made without dflash_shape", flush=True)
        return None, False
    run = [(u, k) for u in us for k in ks if (not pairs or (u, k) in pairs) and ((k + 1) * u) % 2 == 0]
    if not run:
        return None, False
    by_arms = any(k in ("GLM53_W1_C", "GLM53_W1") for _, env in arm_list for k, _ in env) or bool(
        os.environ.get("GLM53_W1_C")
    )
    shape = (run[0][0], run[0][1] + 1)
    keys = {(u, k): shape_defaults((u, k + 1)) for u, k in run}
    if len(set(keys.values())) > 1:
        print(f"SHAPE WARNING: the pairs {run} take different served defaults {keys}: the model made without "
              f"dflash_shape (the earlier form); time one shape a load for the served form", flush=True)  # fmt: skip
        return None, False
    c, env, narrow = keys[run[0]]
    print(f"SHAPE: dflash_shape {shape} (as served): W1 {'by the arms (GLM53_W1_C / GLM53_W1)' if by_arms else f'C {c}'}"
          f", env {dict(env) if not by_arms else {}}, narrow landing {int(narrow)}", flush=True)  # fmt: skip
    return shape, by_arms


def arms() -> list:
    """GLM53_DFLASH2_ARMS -> [(name, [(NAME, value) ..])]; unset: one arm, the defaults."""
    out = []
    for a in os.environ.get("GLM53_DFLASH2_ARMS", "").split(";"):
        name, _, env = a.partition("@")
        out.append((name, [tuple(p.split("=", 1)) for p in env.split("+") if p]))
    return out


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal bit for bit (NaN payloads and -0.0 too)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(width), b.contiguous().view(width))
    return torch.equal(a, b)


def bits_phase(model, mesh, u: int, k: int, steps: int, around) -> dict:
    """GLM53_W1_BITS: `steps` traced verify steps (`forward_verify`, K + 1 tokens a user, its ids and logits) from
    fresh states, the inputs from a seeded generator (the same in every arm), `around` (the landing's DRISCs) around
    the replays: {ids, logits: the last step's, s: the median step}."""
    g = torch.Generator().manual_seed(7100 + 10 * u + k)
    vstates = model.empty_state(u, spec=k)
    vio = model.alloc_spec_io(u, drafts=k, mtp=False)
    got = {}

    def keep(outs):
        ids, lg = outs
        got["ids"] = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).clone()
        got["logits"] = model.logits_to_host(lg, (k + 1) * u).clone()

    got["s"] = timed_steps(
        mesh, model, vio, k, u, lambda: model.forward_verify(vio, vstates, logits=True), g, 0, steps,
        around=around, on_outputs=keep,
    )  # fmt: skip
    release(vstates)
    return got


def with_env(mesh, env: list, run):
    """`run()` with these environment variables set (the program cache cleared so that the programs are made again;
    GLM53_DFLASH2_ARMS_KEEP_CACHE=1: kept, for switches the model reads at each call, e.g. GLM53_DSA_HEADMAJOR, whose
    arms run other ops rather than other programs of the same ops)."""
    if not env:
        return run()
    clear = os.environ.get("GLM53_DFLASH2_ARMS_KEEP_CACHE") != "1"
    undo = [(k, os.environ.get(k)) for k, _ in env]
    for k, v in env:
        os.environ[k] = v
    if clear:
        mesh.clear_program_cache()
    try:
        return run()
    finally:
        for k, v in reversed(undo):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if clear:
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
    parts_wanted = [p for p in os.environ.get("GLM53_DFLASH2_PARTS", "verify,drafter,step").split(",") if p]
    bits_steps = int(os.environ.get("GLM53_W1_BITS", "0"))
    # the drafter's weights random (its time does not depend on them: no routing) with the target's real
    drafter_synthetic = synthetic or os.environ.get("GLM53_DFLASH2_DRAFTER_SYNTHETIC") == "1"
    steps = int(os.environ.get("GLM53_TRACE_SPEED_STEPS", "12"))
    vstart = int(
        os.environ.get("GLM53_TRACE_START", "0")
    )  # the verify step's first position (a long cache: near its end)
    t0 = time.time()
    checkpoint = ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint()
    # the served form's defaults from the DFlash2 step's shape (the expert-weight landing, the narrow landing:
    # `serving_shape`)
    us = [int(x) for x in os.environ.get("GLM53_VERIFY_USERS", "1,4,16").split(",")]
    shape, w1_by_arms = serving_shape(us, ks, pairs, arms())
    undo_w1 = os.environ.get("GLM53_W1")
    if shape and w1_by_arms:
        os.environ["GLM53_W1"] = "0"  # (no landing of the model's own: the arms' GLM53_W1_C makes theirs)
    try:
        model = TtGlm53Model(mesh, checkpoint, max_seq=MAX_SEQ, mtp=True, dflash_shape=shape)
    finally:
        if shape and w1_by_arms:
            if undo_w1 is None:
                os.environ.pop("GLM53_W1", None)
            else:
                os.environ["GLM53_W1"] = undo_w1
    cfg, w, mask = load_weights(synthetic=drafter_synthetic)
    model.tap_layers = frozenset(layer + 1 for layer in cfg.target_layer_ids)
    result = {"synthetic_weights": synthetic, "drafter_synthetic": drafter_synthetic, "ks": ks, "speed": {}}
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    landing = l1_landing(mesh)  # (GLM53_L1_LANDING_KB: a probe's L1 tensor held through the run; none by default)
    rm = dict(dims=(0, None), layout=ttnn.ROW_MAJOR_LAYOUT)
    for u in us:
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
            # (the timed parts' states; none with GLM53_DFLASH2_PARTS empty: the bits phase makes its own, and two sets
            # of a 1M cache's do not fit at 32 users)
            states = model.empty_state(u, spec=k) if parts_wanted else None
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
                # (GLM53_DFLASH2_SAMPLING=1: the split's two forms, `DFlash2Step.SAMPLING_SPLIT`)
                "step_greedy": lambda: st._run(states, sampled=False),
                "step_drawn": lambda: st._run(states, sampled=True),
            }
            seats = torch.arange(ROWS * u, dtype=torch.int64).view(ROWS, u)

            def seat_sampling(drawn: bool) -> None:
                """Every seat greedy (top-k 1), or drawing (top-k 32, top-p 0.95, temperature 1), written before a
                part's capture (GLM53_DFLASH2_SAMPLING=1)."""
                if st.sampling is None:
                    return
                k = torch.full((ROWS, u), 32 if drawn else 1, dtype=torch.int64)
                p = torch.full((ROWS, u), 0.95 if drawn else 1.0)
                t = torch.full((ROWS, u), 1.0 if drawn else 0.0)
                st.set_sampling(k, p, t, 9000 + seats)

            first_bits = None
            for arm, env in arms():
                times = {}
                tag = f"_{arm}" if arm else ""  # (the defaults: the keys and the line as before)
                bits = {}

                def time_parts():
                    w1c = int(os.environ.get("GLM53_W1_C", "0"))
                    w1 = W1Arm(mesh, model, w1c, ROWS * n, f"{users}u K{k}{tag}") if w1c else None
                    if w1 is None and model.w1 is not None:  # (the model's landing, as served: Glm53Generator's)
                        w1 = W1Arm(mesh, model, model.w1.c, ROWS * n, f"{users}u K{k}{tag} (the model's)", model.w1)
                    try:
                        for name in parts_wanted:
                            if name in ("step_greedy", "step_drawn") and st.sampling is None:
                                print(f"DFLASH2 {name}: skipped (needs GLM53_DFLASH2_SAMPLING=1)", flush=True)
                                times[name] = float("nan")
                                continue
                            seat_sampling(name == "step_drawn")
                            # (TT_METAL_CB_MARGIN_LOG builds: the phase's tag in the circular buffers' margin lines)
                            os.environ["TT_METAL_CB_MARGIN_TAG"] = f"{name} {users}u K{k}{tag}"
                            around = w1 if name in ("verify", "step") else None
                            if w1 is not None:
                                w1.part = name
                            times[name] = timed_steps(
                                mesh, model, io, k, u, parts[name], g, vstart, steps, around=around
                            )
                        os.environ.pop("TT_METAL_CB_MARGIN_TAG", None)
                        if bits_steps:
                            if w1 is not None:
                                w1.part = f"bits ({bits_steps} verify steps)"
                            bits.update(bits_phase(model, mesh, u, k, bits_steps, w1))
                    finally:
                        if w1 is not None:
                            w1.release()

                with_env(mesh, env, time_parts)
                if bits:
                    if first_bits is None:
                        first_bits = (arm or "defaults", bits)
                        print(f"W1BITS users={users} K={k} [{arm}]: the reference ({bits_steps} steps, "
                              f"{bits['s'] * 1e3:.2f} ms a step)", flush=True)  # fmt: skip
                    else:
                        same = all(same_bits(bits[key], first_bits[1][key]) for key in ("ids", "logits"))
                        line[f"K{k}{tag}_w1bits_same"] = same
                        print(f"W1BITS users={users} K={k} [{arm}] against [{first_bits[0]}]: "
                              f"{'same' if same else 'DIFFER'} (ids and logits of the last of {bits_steps} steps; "
                              f"{bits['s'] * 1e3:.2f} ms a step)", flush=True)  # fmt: skip
                if not parts_wanted:
                    continue
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
            if states is not None:
                release(states)
            st.release()
            for t in fixed_taps + [dp, du, db, ds]:
                ttnn.deallocate(t)
            del dev
            gc.collect()
    if landing is not None:
        ttnn.deallocate(landing)
    path = out_dir() / f"dflash2-speed-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}", flush=True)
