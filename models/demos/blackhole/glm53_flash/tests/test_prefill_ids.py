"""A refill's first tokens: the generator's `prefill_many(greedy_ids=True)` (the argmax
on the device, 8 ids read back; GLM53_PREFILL_IDS=0: the host's) against the argmax of the logits it returns
otherwise, the same ids, and both timed by the requests in the wave (one a row, short prompts in one chunk of 128
replayed from its trace): whether a wave's time grows with its requests on the device or on the host.

    GLM53_PREFILL_IDS_ROWS = comma list of requests a wave (default 1,2,4,6,8)
    GLM53_PREFILL_IDS_REPEAT = waves of each (default 6; the first of each not timed)
    GLM53_SYNTHETIC_WEIGHTS = 1 (default): random weights (the ids' equality and the time; the same code path)

Run (a device test): GLM53_DEVICE_TEST=1 pytest .../tests/test_prefill_ids.py -s
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import model as model_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

USERS = 4


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefill_ids(mesh_device):
    torch.set_num_threads(16)
    counts = [int(x) for x in os.environ.get("GLM53_PREFILL_IDS_ROWS", "1,2,4,6,8").split(",")]
    repeat = int(os.environ.get("GLM53_PREFILL_IDS_REPEAT", "6"))
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS", "1") == "1"
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=1024)
    gen = Glm53Generator(model, users_per_row=USERS, chunk=(128,))
    warm = gen.warmup(greedy=True, logits=False, chunks=(128,), traced=(128,))
    print(f"KPIDS warm-up {warm}; PREFILL_IDS {model_module.PREFILL_IDS}", flush=True)
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(7)
    differ = 0
    for n in counts:
        times = {"logits": [], "ids": []}
        for k in range(repeat):
            user = k % USERS
            reqs = []
            for r in range(n):  # one a row, user k % USERS, 60 .. 128 tokens
                length = int(torch.randint(60, 129, (1,), generator=g))
                reqs.append((r * USERS + user, torch.randint(0, vocab, (length,), generator=g).tolist()))
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            logits = gen.prefill_many(reqs)
            want = [int(x.reshape(-1)[:vocab].argmax()) for x in logits]
            t1 = time.perf_counter()
            got = gen.prefill_many(reqs, greedy_ids=True)
            t2 = time.perf_counter()
            if k > 0:
                times["logits"].append(t1 - t0)
                times["ids"].append(t2 - t1)
            if got != want:
                differ += 1
                print(f"KPIDS n={n} wave {k}: ids {got} != the logits' argmax {want}", flush=True)
        med = {key: statistics.median(v) * 1e3 for key, v in times.items()}
        print(
            f"KPIDS n={n}: a wave {med['logits']:.1f} ms with the logits read (+ the host's argmax), "
            f"{med['ids']:.1f} ms with the ids",
            flush=True,
        )
    print(f"KPIDS waves whose ids differ: {differ} (rows {ROWS}, users {USERS})", flush=True)
    assert differ == 0
