# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DFlash2's next inputs made on the device (`ttnn/next_inputs.py`, GLM53_DFLASH2_NEXT_INPUTS, a part of serving's
run-ahead) against the host's prediction, for any acceptance: the serving flows of the adapter flow test run with
synthetic weights, whose drafts are never accepted (n = 0 always), so this test feeds the op made-up step results
instead.

Each trial: random seats (some empty), each with a position (some near the cache's end: the positions clamped), its
tokens and keep, written as `Glm53Generator.dflash_verify` writes them (`DFlash2Step.write`) and unpacked
(`InputPack.unpack`, the step's first op); then a made-up gathered result (each user's n from 0 to K, its bonus, its
drafter row: t[s], K drafts, zeros), `NextInputs.run` over the packed tensor, and the packed row on every chip against
the one the host fills for `NextInputs.predict`'s inputs (`Glm53Generator._check_next`).  No step is run.

    GLM53_DEVICE_TEST=1 GLM53_SYNTHETIC_WEIGHTS=1 GLM53_DFLASH2_NEXT_INPUTS=1 pytest .../tests/test_next_inputs_device.py -s
GLM53_NI_MAX_LEN (default 9408: the long cache; 2048: the dense one), GLM53_NI_TRIALS (default 40).
"""

from __future__ import annotations

import os

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

USERS = 4
SEATS = ROWS * USERS


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_next_inputs_device(mesh_device):
    assert os.environ.get("GLM53_DFLASH2_NEXT_INPUTS") == "1", "set GLM53_DFLASH2_NEXT_INPUTS=1"
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm
    from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights

    max_len = int(os.environ.get("GLM53_NI_MAX_LEN", "9408"))
    trials = int(os.environ.get("GLM53_NI_TRIALS", "40"))
    chunks = glm53_vllm.prefill_chunks()
    chunk = max(chunks)
    max_seq = -(-max_len // chunk) * chunk
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint(), max_seq=max_seq, mtp=False)
    dk = glm53_vllm.dflash2_k(SEATS)
    cfg, w, dmask = load_weights(None, synthetic=True)
    drafter = TtDFlash2(
        mesh_device, cfg, w, dmask, users=USERS, max_ctx=max_seq + 8, block=4 if dk <= 3 else 8,
        head_weight=model.lm_head.mesh_weight, embedding=model.embed,
    )  # fmt: skip
    gen = Glm53Generator(model, USERS, chunk=chunks, spec=0, dflash=(drafter, dk))
    step, nxt = gen.dstep, gen.dstep.next
    assert nxt is not None, "GLM53_DFLASH2_NEXT_INPUTS: the step has no NextInputs"
    vocab, k, u = model.config.vocab_size, dk, USERS
    long = model.dsa is not None and model.dsa.long
    print(f"NI: K {k}, cache {gen.max_seq} (DSA {'long' if long else 'dense'}), max_pos {nxt.max_pos}", flush=True)
    g = torch.Generator().manual_seed(7)
    clamped = 0
    for trial in range(trials):
        active = torch.rand(SEATS, generator=g) < (0.9 if trial % 2 else 0.5)
        near = torch.rand(SEATS, generator=g) < 0.2  # (near the cache's end: the next positions clamped)
        p = torch.randint(0, gen.max_seq - 1, (SEATS,), generator=g)
        p = torch.where(near, gen.max_seq - 1 - torch.randint(0, 2 * k + 2, (SEATS,), generator=g), p)
        positions = torch.where(active[:, None], p[:, None] + torch.arange(k + 1)[None, :], -1)
        tokens = torch.randint(0, vocab, (SEATS, k + 1), generator=g)
        keep = torch.randint(0, k + 1, (SEATS,), generator=g)
        inputs = gen._dflash_inputs(tokens, positions, keep)
        step.write(*inputs)
        step.io.pack.unpack()
        # a made-up gathered result [1, 1, 8, 10 U] (fp32 whole numbers): n, bonus, then each user's drafter row
        n = torch.randint(0, k + 1, (ROWS, u), generator=g)
        bonus = torch.randint(0, vocab, (ROWS, u), generator=g)
        drafts = torch.randint(0, vocab, (ROWS, u, k), generator=g)
        dr = torch.zeros(ROWS, u, 8, dtype=torch.int64)
        dr[:, :, 0] = torch.randint(0, 64, (ROWS, u), generator=g)
        dr[:, :, 1 : 1 + k] = drafts
        res = torch.cat([n, bonus, dr.reshape(ROWS, 8 * u)], dim=1).float().reshape(1, 1, ROWS, 10 * u)
        results = ttnn.from_torch(
            res,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        nxt.run(results)
        predicted = nxt.predict(n, bonus, drafts, inputs[1], inputs[3])
        clamped += int((predicted[1] == nxt.max_pos).sum())
        gen._check_next(predicted)
        ttnn.deallocate(results)
    checks = gen.dflash_writes
    print(f"NI: {trials} trials, {checks.get('checked', 0)} the same on every chip, {checks.get('bad', 0)} differing; "
          f"{clamped} positions clamped at max_pos", flush=True)  # fmt: skip
    assert checks.get("bad", 0) == 0 and checks.get("checked", 0) == trials
    assert clamped > 0, "no trial reached the cache's end"
