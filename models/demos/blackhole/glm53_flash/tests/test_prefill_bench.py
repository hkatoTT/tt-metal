"""Prefill speed through the serving generator: the time to the first token by prompt length (eager chunks; the
logits of each prompt's last token read back), after the warm-up, for one request and for several given at once (one
per mesh row, `prefill_many`).

    GLM53_PREFILL_LENGTHS = comma list of prompt lengths (default 128,512,1024,2048)
    GLM53_PREFILL_CHUNKS = comma list of chunk lengths, each measured on every length (default 128,512); with
        several, also the generator's choice per prompt (`Glm53Generator.schedule`, reported as chunk "auto")
    GLM53_PREFILL_REPEAT = timed prompts per length (default 2)
    GLM53_PREFILL_REQUESTS = comma list of requests given at once, each on its own mesh row (default 1,8)
    GLM53_PREFILL_TRACE = comma list of chunk lengths whose one-chunk prefill is captured as a trace (default none):
        the lengths that fit one such chunk are measured through it too (reported as chunk "<length>-trace"); with
        GLM53_PREFILL_TRACE_ALL=1 every chunk of those lengths replays a trace, so every length is measured through
        them, and with several chunk lengths all captured, the generator's choice too ("auto-trace")
    GLM53_PREFILL_REPLAYS = n (default 0): also each captured chunk form replayed n times back to back (seconds per
        replay: the device's time, the host out of the way) and, for each chunk length, n eager chunks at start T
        (the host's issue time per chunk, and with the device's: per chunk to the end)
    GLM53_PREFILL_BENCH_USERS = users per mesh row (default 1; the requests take user 0 of their rows): a wave's last
        chunk writes the KDA states into the users, which a captured chunk does over all of them
    GLM53_SYNTHETIC_WEIGHTS = 1 (default): random weights made on the device (checkpoint.ShapeOnlyCheckpoint; the
        decode timing matched the real weights within 0.3 %); 0: the real checkpoint
    GLM53_PREFILL_BENCH_EAGER = 0: with GLM53_PREFILL_TRACE, the traced runs only
    GLM53_PREFILL_BENCH_SP = 0: `dsa.PREFILL_SP` off (a long prompt on its own row, no extra DSA user)
    GLM53_FIXED_INPUTS_SKIP = 0: a traced prefill's inputs written every time (`common.FIXED_SKIP_SAME` off), so that
        a repeated prompt is timed as a new one (the prefill fixed point's setting); 1 (default): the model's own
Results: out_dir()/prefill-bench-<stamp>.json.  Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
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
from models.demos.blackhole.glm53_flash.ttnn import common as common_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import model as model_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefill_bench(mesh_device):
    torch.set_num_threads(16)
    lengths = [int(x) for x in os.environ.get("GLM53_PREFILL_LENGTHS", "128,512,1024,2048").split(",")]
    chunks = [int(x) for x in os.environ.get("GLM53_PREFILL_CHUNKS", "128,512").split(",")]
    repeat = int(os.environ.get("GLM53_PREFILL_REPEAT", "2"))
    counts = [int(x) for x in os.environ.get("GLM53_PREFILL_REQUESTS", "1,8").split(",")]
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS", "1") == "1"
    common_module.FIXED_SKIP_SAME = os.environ.get("GLM53_FIXED_INPUTS_SKIP", "1") == "1"
    if os.environ.get("GLM53_PREFILL_BENCH_SP") == "0":  # (each prompt on its own row: no sequence-parallel rows, so
        dsa_module.PREFILL_SP = False  # no extra DSA user: the long cache's memory a user less, the 1M question)
    t0 = time.time()
    checkpoint = ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint()
    model = TtGlm53Model(mesh_device, checkpoint, max_seq=-(-max(lengths) // max(chunks)) * max(chunks))
    users = int(os.environ.get("GLM53_PREFILL_BENCH_USERS", "1"))
    gen = Glm53Generator(model, users_per_row=users, chunk=tuple(chunks))
    result = {
        "synthetic_weights": synthetic,
        "chunks": chunks,
        "users_per_row": users,
        "tree": os.popen(f"git -C {Path(__file__).parent} rev-parse --short HEAD").read().strip(),
        "load_seconds": round(time.time() - t0, 1),
    }
    traced = tuple(int(x) for x in os.environ.get("GLM53_PREFILL_TRACE", "").split(",") if x)
    result["warmup_seconds"] = gen.warmup(greedy=True, logits=False, chunks=chunks, traced=traced)
    result["trace_all"] = every = model_module.PREFILL_TRACE_ALL
    if traced:
        result["trace_region_mb"] = round(model_module.trace_bytes(mesh_device) / 2**20, 1)
    print(f"loaded in {result['load_seconds']} s; warm-up {result['warmup_seconds']}", flush=True)

    replays = int(os.environ.get("GLM53_PREFILL_REPLAYS", "0"))
    if replays:
        result["replay"] = {}
        for key, captured in gen.prefill_traces.items():
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(mesh_device, captured.trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            result["replay"][str(tuple(key))] = round((time.perf_counter() - t1) / replays, 4)
        for t in chunks:
            ids = torch.zeros(8, t, dtype=torch.int64)
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            for _ in range(replays):
                model.prefill(ids, t, gen.states, [0] * 8, logits=False, last=False)
            issued = time.perf_counter() - t1
            ttnn.synchronize_device(mesh_device)
            done = time.perf_counter() - t1
            result["replay"][f"eager {t} at {t}"] = {
                "issue_s": round(issued / replays, 4),
                "s": round(done / replays, 4),
            }
        print(f"seconds per chunk, replayed / eager: {result['replay']}", flush=True)

    g = torch.Generator().manual_seed(0)
    vocab = model.config.vocab_size
    result["lengths"] = {}
    for count in counts:
        runs = [(chunk, False) for chunk in chunks + ([None] if len(chunks) > 1 else [])] + [(t, True) for t in traced]
        if traced and os.environ.get("GLM53_PREFILL_BENCH_EAGER") == "0":  # (the traced runs only: a long prompt's
            runs = [r for r in runs if r[1]]  # eager chunks compile their start forms one by one)
        if every and len(chunks) > 1 and set(chunks) <= set(traced):
            runs.append((None, True))
        for chunk, use_trace in runs:
            for length in lengths:
                if use_trace and not every and length > chunk:
                    continue
                seconds, device = [], []
                before = dict(gen.counts)  # the generator's waves and chunks, traced or not, over the repeats
                for r in range(repeat):
                    rows = [(r * 3 + k) % 8 for k in range(count)]  # other mesh rows each time
                    requests = [(row * users, torch.randint(0, vocab, (length,), generator=g).tolist()) for row in rows]
                    ttnn.synchronize_device(mesh_device)
                    t1 = time.time()
                    out = gen.prefill_many(requests, chunk=chunk, traced=use_trace)
                    seconds.append(time.time() - t1)
                    # until the device is done: the work issued after the last logits' read (the KDA commit's replay)
                    ttnn.synchronize_device(mesh_device)
                    device.append(time.time() - t1)
                    finite = all(bool(torch.isfinite(x).all()) for x in out)  # random weights may overflow
                best = min(seconds)
                n = len(gen.schedule(length, chunk))
                row = {
                    "requests": count,
                    "seconds": [round(s, 3) for s in seconds],
                    "ttft_s": round(best, 3),
                    "tok_s": round(count * length / best, 1),
                    "device_seconds": [round(s, 4) for s in device],
                    "device_tok_s": round(count * length / min(device), 1),
                    "chunks": n,
                    "s_per_chunk": round(best / n, 3),
                    "finite_logits": finite,
                    "counts": {k: (v - before[k]) // repeat for k, v in gen.counts.items()},
                }
                label = f"{chunk or 'auto'}" + ("-trace" if use_trace else "")
                key = f"{length}@{label}" + ("" if count == 1 else f"x{count}")
                result["lengths"][key] = row
                print(f"prefill {count} x {length} tokens in chunks of {label}: {row}", flush=True)
    gen.release()
    path = out_dir() / f"prefill-bench-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}", flush=True)
