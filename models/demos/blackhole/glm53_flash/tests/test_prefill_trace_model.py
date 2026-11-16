"""Every prefill chunk replayed from a captured trace (GLM53_PREFILL_TRACE_ALL, `TtGlm53Model.capture_prefills`)
against the eager chunks, bit for bit, through the serving generator on the same prompts and slots: the logits after
each prompt, then the decode that follows (one step's full logits, then greedy ids), which reads the states the
prefill left (KDA's committed states, DSA's cache).  Synthetic weights by default (the flow and the bits, not the text).

    GLM53_PTM_CACHE = the cache length (default 2048: the dense DSA; above 2048 the indexer)
    GLM53_PTM_CHUNKS = the generator's chunk lengths, all captured (default 128,512); one pass each with that chunk
        throughout, and with several one more in the generator's choice per prompt ("auto")
    GLM53_PTM_USERS = users per mesh row (default 2: two waves); every slot gets a prompt but the last user of the
        last GLM53_PTM_IDLE rows (default 4 with 2 users or more: the last wave has rows without a request, whose
        users the commit must leave as they are; the decode compares the slots with a prompt)
    GLM53_PTM_LENGTHS = the prompts' lengths, slot by slot (row-major); default spread over the cache so that the rows of
        a wave end in different chunks
    GLM53_PTM_DECODE = decode steps after each pass (default 4: one with the full logits, then greedy)
    GLM53_PTM_WARMUP_TOKENS = the eager warm-up's prompt (default the whole cache)
    GLM53_SYNTHETIC_WEIGHTS = 1 (default) random weights made on the device; 0: the real checkpoint
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_prefill_trace_model.py -s
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import model as model_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model, trace_bytes  # noqa: E402

FRACTIONS = [0.999, 0.93, 0.74, 0.54, 0.34, 0.255, 0.15, 0.05, 0.64, 0.5, 0.44, 0.31, 0.2505, 0.125, 0.063, 0.03]


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t


def default_lengths(cache: int, slots: int, steps: int) -> list[int]:
    top = cache - steps - 1
    return [
        max(1, min(top, int(FRACTIONS[i % len(FRACTIONS)] * cache) - 3 * (i // len(FRACTIONS)))) for i in range(slots)
    ]


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefill_trace_model(mesh_device):
    torch.set_num_threads(16)
    model_module.PREFILL_TRACE_ALL = True
    cache = int(os.environ.get("GLM53_PTM_CACHE", "2048"))
    chunks = tuple(int(c) for c in os.environ.get("GLM53_PTM_CHUNKS", "128,512").split(","))
    users = int(os.environ.get("GLM53_PTM_USERS", "2"))
    steps = int(os.environ.get("GLM53_PTM_DECODE", "4"))
    slots = ROWS * users
    lengths = [int(x) for x in os.environ.get("GLM53_PTM_LENGTHS", "").split(",") if x] or default_lengths(
        cache, slots, steps
    )
    assert len(lengths) == slots and max(lengths) + steps <= cache, (lengths, slots, cache)
    idle = int(os.environ.get("GLM53_PTM_IDLE", "4" if users > 1 else "0"))
    active = [s for s in range(slots) if not (s // users >= ROWS - idle and s % users == users - 1)]
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS", "1") == "1"
    t0 = time.time()
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=cache)
    result = {
        "cache": cache,
        "chunks": chunks,
        "users": users,
        "lengths": lengths,
        "load_seconds": round(time.time() - t0),
    }
    gen = Glm53Generator(model, users, chunk=chunks)
    warm = int(os.environ.get("GLM53_PTM_WARMUP_TOKENS", "0")) or None
    result["warmup_seconds"] = gen.warmup(greedy=True, logits=True, tokens=warm, traced=chunks)
    result["trace_region_mb"] = round(trace_bytes(mesh_device) / 2**20, 1)
    result["traces"] = {str(tuple(k)): round(c.bytes / 2**20, 2) for k, c in gen.prefill_traces.items()}
    print(f"warm-up {result['warmup_seconds']}; trace region in use {result['trace_region_mb']} MB", flush=True)

    g = torch.Generator().manual_seed(7)
    vocab = model.config.vocab_size
    requests = [(s, torch.randint(0, vocab, (n,), generator=g).tolist()) for s, n in enumerate(lengths)]
    requests = [requests[s] for s in active]
    positions0 = torch.full((slots,), -1, dtype=torch.int64)  # -1: no request
    positions0[active] = torch.tensor([lengths[s] for s in active])
    result["active_slots"] = active

    def run(chunk, traced: bool):
        counts = dict(gen.counts)
        t1 = time.time()
        logits = gen.prefill_many(requests, chunk=chunk, traced=traced)
        seconds = time.time() - t1
        delta = {k: gen.counts[k] - counts[k] for k in counts}
        tokens = torch.zeros(slots, dtype=torch.int64)
        tokens[active] = torch.stack([lg.argmax() for lg in logits]).to(torch.int64)
        positions = positions0.clone()
        first = gen.decode(tokens, positions, greedy=False)  # [slots, V]
        on = positions >= 0
        tokens, positions = first.argmax(-1).to(torch.int64), torch.where(on, positions + 1, -1)
        ids = []
        for _ in range(steps - 1):
            tokens = gen.decode(tokens, positions, greedy=True)
            ids.append(tokens[active].clone())
            positions = torch.where(on, positions + 1, -1)
        return dict(
            logits=torch.stack(logits),
            decode=first[active],
            ids=torch.stack(ids) if ids else None,
            seconds=seconds,
            counts=delta,
        )

    passes = list(chunks) + ([None] if len(chunks) > 1 else [])
    bad = {}
    result["passes"] = {}
    for chunk in passes:
        label = str(chunk or "auto")
        eager = run(chunk, False)
        traced = run(chunk, True)
        same = {
            "prefill_logits": torch.equal(bits(eager["logits"]), bits(traced["logits"])),
            "decode_logits": torch.equal(bits(eager["decode"]), bits(traced["decode"])),
            "decode_ids": eager["ids"] is None or torch.equal(eager["ids"], traced["ids"]),
        }
        rows_off = [
            active[i]
            for i in range(len(active))
            if not torch.equal(bits(eager["logits"][i]), bits(traced["logits"][i]))
        ]
        finite = bool(torch.isfinite(eager["logits"]).all())
        tokens = sum(lengths[s] for s in active)
        entry = {
            "equal": same,
            "slots_differing": rows_off,
            "finite_logits": finite,
            "eager_s": round(eager["seconds"], 3),
            "traced_s": round(traced["seconds"], 3),
            "eager_tok_s": round(tokens / eager["seconds"], 1),
            "traced_tok_s": round(tokens / traced["seconds"], 1),
            "eager_counts": eager["counts"],
            "traced_counts": traced["counts"],
        }
        if not all(same.values()):
            bad[label] = entry
        result["passes"][label] = entry
        print(f"pass {label}: {entry}", flush=True)
        assert traced["counts"]["traced_chunks"] == traced["counts"]["chunks"], f"pass {label}: eager chunks remain"
    gen.release()
    path = out_dir() / f"prefill-trace-{cache}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}", flush=True)
    assert not bad, f"traced prefill differs from the eager one: {bad}"
