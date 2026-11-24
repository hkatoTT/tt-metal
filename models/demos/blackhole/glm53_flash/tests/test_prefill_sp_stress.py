"""The sequence-parallel prefill over and over: random rounds of
1 to 3 requests on random rows with random prompt lengths through the generator (`sp_plan`: one request on all 8
rows, or several sharing them), the waves replaying their captured forms, synthetic weights; every round's logits
finite.  A hang shows as the run's timeout (run it with TT_METAL_OPERATION_TIMEOUT_SECONDS and tt-triage).

    GLM53_SP_STRESS = rounds (default 300), GLM53_SP_STRESS_CACHE (default 2,048 tokens), GLM53_SP_STRESS_SEED (1),
    GLM53_SP_STRESS_CHUNKS (default 128,512; 128,512,1024 with the long waves, 1,024 tokens a row),
    GLM53_SP_STRESS_REQUESTS (default 1,3: the requests a round, at least and at most; 6,8 mostly one a row)
    GLM53_DEVICE_TEST=1 GLM53_SYNTHETIC_WEIGHTS=1 pytest ... -s
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

from models.demos.blackhole.glm53_flash.checkpoint import ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefill_sp_stress(mesh_device):
    rounds = int(os.environ.get("GLM53_SP_STRESS", "300"))
    cache = int(os.environ.get("GLM53_SP_STRESS_CACHE", "2048"))
    rng = random.Random(int(os.environ.get("GLM53_SP_STRESS_SEED", "1")))
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint(), max_seq=cache)
    assert model.sp_ready, "dsa.PREFILL_SP is needed"
    users = 2
    chunks = tuple(int(c) for c in os.environ.get("GLM53_SP_STRESS_CHUNKS", "128,512").split(","))
    lo, hi = (int(c) for c in os.environ.get("GLM53_SP_STRESS_REQUESTS", "1,3").split(","))
    gen = Glm53Generator(model, users, chunk=chunks)
    print(f"warm-up {gen.warmup(greedy=True, logits=False, traced=chunks)}", flush=True)
    vocab = model.config.vocab_size
    sizes = {}  # the replayed chunks by length (one-row and sequence-parallel forms)
    for name in ("replay_prefill", "replay_prefill_sp", "replay_prefill_sp_pair"):
        real = getattr(model, name)

        def counted(captured, *a, _real=real, _name=name, **k):
            key = f"{_name[15:] or 'row'} {captured.t}"
            sizes[key] = sizes.get(key, 0) + 1
            return _real(captured, *a, **k)

        setattr(model, name, counted)
    t0, tokens, kinds = time.time(), 0, {}
    for n in range(rounds):
        rows = rng.sample(range(ROWS), rng.randint(lo, hi))
        requests = [
            (r * users + rng.randrange(users), [rng.randrange(vocab) for _ in range(rng.randint(129, cache))])
            for r in rows
        ]
        before = gen.counts["traced_chunks"]
        logits = gen.prefill_many(requests)
        assert all(bool(torch.isfinite(lg).all()) for lg in logits), f"round {n}: logits not finite"
        kind = f"{len(requests)} req, {'traced' if gen.counts['traced_chunks'] > before else 'eager'}"
        kinds[kind] = kinds.get(kind, 0) + 1
        tokens += sum(len(p) for _, p in requests)
        if n % 50 == 49:
            print(f"{n + 1} rounds, {tokens} tokens in {time.time() - t0:.1f} s, {kinds}, replays {sizes}", flush=True)
    ttnn.synchronize_device(mesh_device)
    print(
        f"{rounds} rounds of the sequence-parallel prefill without a hang: {tokens} tokens in "
        f"{time.time() - t0:.1f} s; {kinds}; counts {gen.counts}",
        flush=True,
    )
    gen.release()
