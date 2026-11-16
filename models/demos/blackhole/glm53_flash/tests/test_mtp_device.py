"""The MTP layer on the device (`ttnn/mtp.py`) against `tt/mtp.py`, decode steps from an empty cache.

8 rows x 4 users, 5 steps: each step every user's next-token embedding (rows of the real embedding table) and a
hidden state (random, of a normed hidden's scale) go through the layer; the output and the head's input against the
CPU layer run over the whole sequence at once (causal).  Real weights (layer 45).  Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_mtp_device.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.config import Glm53Config  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.common import bf16  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.mtp import MtpLayer  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS,
    MESH_SHAPE,
    ROWS,
    gather_rows,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.mtp import TtMtp  # noqa: E402

USERS, STEPS = 4, 5


def pcc(a, b) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_mtp_device(mesh_device, pcc_thresholds):
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    reference = MtpLayer(checkpoint, config)
    module = TtMtp(mesh_device, checkpoint, config, max_seq=128)
    embed = bf16(checkpoint.tensor("model.language_model.embed_tokens.weight"))
    g = torch.Generator().manual_seed(45)
    n = ROWS * USERS
    ids = torch.randint(0, config.vocab_size, (n, STEPS), generator=g)
    embeds = F.embedding(ids, embed)  # [n, S, H]
    hidden = torch.randn(n, STEPS, config.hidden_size, generator=g).to(torch.bfloat16)  # a normed hidden's scale
    with torch.no_grad():
        want, _ = reference(embeds, hidden)
        want_head = reference.head_input(want)
    state = module.empty_state(USERS)
    worst = 1.0
    for s in range(STEPS):
        e = put(mesh_device, embeds[:, s].reshape(ROWS, 1, USERS, -1), dims=(0, None))
        h = put(mesh_device, hidden[:, s].reshape(ROWS, 1, USERS, -1), dims=(0, None))
        positions = module.attention.positions_on_device(torch.full((ROWS, USERS), s, dtype=torch.int64))
        y, state = module.step(e, h, state, positions)
        head = module.head_input(y)
        got = gather_rows(mesh_device, y).reshape(n, -1)
        got_head = gather_rows(mesh_device, head).reshape(n, -1)
        scores = (pcc(got, want[:, s]), pcc(got_head, want_head[:, s]))
        worst = min(worst, *scores)
        print(f"step {s}: MTP output pcc {scores[0]:.6f}, head input pcc {scores[1]:.6f}", flush=True)
        for z in (e, h, y, head):
            ttnn.deallocate(z)
    assert worst >= pcc_thresholds.get("test_mtp_device", 0.99), f"worst pcc {worst:.6f}"
