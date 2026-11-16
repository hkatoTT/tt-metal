"""The decode step as a captured trace: the same logits as the eager step, the device-side argmax, and how fast the
greedy step (inputs written, trace replayed, ids read back) runs.

    GLM53_TRACE_USERS = comma list of users per row for the speed part (default 1,2,4,8,16,32)
    GLM53_TRACE_START = the context position the speed part starts at (default 0: a few tokens of context)
    GLM53_TRACE_MAX_SEQ = the cache length (default 512; must exceed the start + 13)
    GLM53_TRACE_SPEED_ONLY = 1 skips the correctness parts
    GLM53_SYNTHETIC_WEIGHTS = 1 builds the model from the checkpoint's shapes only, with random weights made on the
        device (checkpoint.ShapeOnlyCheckpoint): the same timing, seconds instead of minutes to load; the
        correctness parts still compare trace with eager
    GLM53_TRACE_TOKENS = the speed part's input tokens: random (default), records (each user one of the acceptance
        records' real text, prompt then generation), same (every user the same record: all route alike)
Results: out_dir()/trace-<stamp>.json.  Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402


def tokens_at(t: int, users: int, vocab: int) -> torch.Tensor:
    return torch.randint(0, vocab, (ROWS, users), generator=torch.Generator().manual_seed(1000 + t))


RECORDS = Path(__file__).resolve().parents[1] / "tools" / "acceptance" / "greedy-v1"


def speed_tokens(mode: str, t: int, users: int, vocab: int, start: int) -> torch.Tensor:
    """[ROWS, users] input tokens for step t of the speed part (GLM53_TRACE_TOKENS)."""
    if mode == "random":
        return tokens_at(t, users, vocab)
    texts = [
        d["prompt_token_ids"] + d["generated_token_ids"]
        for d in map(json.loads, map(Path.read_text, sorted(RECORDS.glob("*.json"))))
    ]
    out = torch.empty(ROWS, users, dtype=torch.int64)
    for r in range(ROWS):
        for u in range(users):
            text = texts[0] if mode == "same" else texts[(r * users + u) % len(texts)]
            out[r, u] = text[(start + t) % len(text)]
    return out


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_device_trace(mesh_device):
    result = {"tree": os.popen(f"git -C {Path(__file__).parent} rev-parse --short HEAD").read().strip()}
    t0 = time.time()
    start = int(os.environ.get("GLM53_TRACE_START", "0"))
    result["start_position"] = start
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    result["synthetic_weights"] = synthetic
    checkpoint = ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint()
    model = TtGlm53Model(mesh_device, checkpoint, max_seq=int(os.environ.get("GLM53_TRACE_MAX_SEQ", "512")))
    vocab = model.config.vocab_size
    result["load_seconds"] = round(time.time() - t0)
    print(f"loaded in {result['load_seconds']}s", flush=True)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)  # (one chip's banks; every chip holds the same)
    banks = int(view.num_banks)
    result["dram_gib_per_chip"] = {
        "allocated": round(int(view.total_bytes_allocated_per_bank) * banks / 2**30, 2),
        "free": round(int(view.total_bytes_free_per_bank) * banks / 2**30, 2),
    }
    print(f"DRAM per chip after the load: {result['dram_gib_per_chip']} GiB", flush=True)

    t_correct = time.time()
    if os.environ.get("GLM53_TRACE_SPEED_ONLY") == "1":
        speed(mesh_device, model, vocab, start, result)
        return
    # correctness: trace replay against eager steps from the same state (the capture's compile run is step 0)
    users = 2
    pos = lambda t: torch.full((ROWS, users), t)  # noqa: E731
    states_e, states_t = model.empty_state(users), model.empty_state(users)
    io = model.alloc_io(users)
    model.write_inputs(io, tokens_at(0, users, vocab), pos(0))
    trace, logits_t = model.capture(io, states_t)
    model.step(tokens_at(0, users, vocab), pos(0), states_e)
    worst = 1.0
    for t in range(1, 8):
        model.write_inputs(io, tokens_at(t, users, vocab), pos(t))
        got = model.replay(trace, logits_t, users)
        want = model.step(tokens_at(t, users, vocab), pos(t), states_e)
        same = torch.equal(got, want)
        p = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
        worst = min(worst, p)
        top1 = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
        print(f"step {t}: trace vs eager bitwise={same} pcc={p:.7f} top1={top1:.3f}", flush=True)
    result["trace_vs_eager_worst_pcc"] = worst
    ttnn.release_trace(mesh_device, trace)

    # the device-side argmax (greedy trace) against the argmax of the eager logits, continuing from the same states
    model.write_inputs(io, tokens_at(8, users, vocab), pos(8))
    trace_g, ids_t = model.capture(io, states_t, greedy=True)  # its compile run is step 8 for states_t
    model.step(tokens_at(8, users, vocab), pos(8), states_e)
    agree = []
    for t in range(9, 14):
        model.write_inputs(io, tokens_at(t, users, vocab), pos(t))
        ttnn.execute_trace(mesh_device, trace_g, cq_id=0, blocking=False)
        got_ids = model.ids_to_host(ids_t, users)
        want_ids = model.step(tokens_at(t, users, vocab), pos(t), states_e).argmax(-1)
        agree.append((got_ids == want_ids).float().mean().item())
        print(f"step {t}: device argmax equals the eager argmax for {agree[-1]:.3f} of the users", flush=True)
    result["device_argmax_agreement"] = sum(agree) / len(agree)
    assert min(agree) == 1.0, "the device argmax differs from the eager argmax"
    ttnn.release_trace(mesh_device, trace_g)
    for st in states_e + states_t:
        st.deallocate()
    assert worst > 0.9999, "the traced step does not reproduce the eager step"
    result["correctness_seconds"] = round(time.time() - t_correct)
    print(f"correctness parts took {result['correctness_seconds']}s", flush=True)

    speed(mesh_device, model, vocab, start, result)


def speed(mesh_device, model, vocab: int, start: int, result: dict) -> None:
    """Replay time per greedy step: inputs written, trace replayed, device synchronized (and the ids read back),
    the median over steps 3-12, at the context position `start` + 1..12."""
    result["speed"] = {}
    mode = os.environ.get("GLM53_TRACE_TOKENS", "random")
    result["speed_tokens"] = mode
    for u in [int(x) for x in os.environ.get("GLM53_TRACE_USERS", "1,2,4,8,16,32").split(",")]:
        t0 = time.time()
        states = model.empty_state(u)
        io = model.alloc_io(u)
        ttnn.synchronize_device(mesh_device)
        t1 = time.time()
        model.write_inputs(io, speed_tokens(mode, 0, u, vocab, start), torch.full((ROWS, u), start))
        trace, ids = model.capture(io, states, greedy=True)
        ttnn.synchronize_device(mesh_device)
        t2 = time.time()
        device_only, with_readback = [], []
        # GLM53_TRACE_SPEED_STEPS: more steps (a stress run for rare hangs; the times the same median from step 3),
        # the positions wrapping back past the cache's end (GLM53_TRACE_MAX_SEQ) as the verify step's stresses do
        limit = int(os.environ.get("GLM53_TRACE_MAX_SEQ", "512"))
        for t in range(1, int(os.environ.get("GLM53_TRACE_SPEED_STEPS", "12")) + 1):
            a = time.perf_counter()
            pos = start + t if start + t < limit else start + 1 + t % (limit - start - 1)
            model.write_inputs(io, speed_tokens(mode, t, u, vocab, start), torch.full((ROWS, u), pos))
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            b = time.perf_counter()
            model.ids_to_host(ids, u)
            c = time.perf_counter()
            device_only.append(b - a)
            with_readback.append(c - a)
        med = lambda v: sorted(v[2:])[len(v[2:]) // 2]  # noqa: E731
        users_total = ROWS * u
        result["speed"][f"{users_total}_users"] = {
            "step_s": round(med(device_only), 4),
            "step_with_ids_readback_s": round(med(with_readback), 4),
            "tok_s_total": round(users_total / med(device_only), 2),
            "tok_s_total_with_ids_readback": round(users_total / med(with_readback), 2),
        }
        setup = {
            "states_s": round(t1 - t0, 1),
            "compile_and_capture_s": round(t2 - t1, 1),
            "steps_s": round(time.time() - t2, 1),
        }
        result["speed"][f"{users_total}_users"]["setup"] = setup
        print(f"traced {users_total} users: {result['speed'][f'{users_total}_users']}", flush=True)
        ttnn.release_trace(mesh_device, trace)
        for st in states:
            st.deallocate()

    path = out_dir() / f"trace-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}", flush=True)
