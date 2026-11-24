# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM53_KDA_SP_FOLD: the sequence-parallel KDA prefill's state housekeeping folded
(the chain reading the wave state with `initial_keep`, the wave state by one `ttnn.where`) against the ops it replaces
(state x keep, after x others + s_end x owner): the same requests and waves (test_kda_device's `sp_schedule` / `run_sp`,
layer 0's real weights, random prompts) through the layer with the switch off and on: every output, user state and
conv history compared bit for bit (the counts of values that differ).  Print only (KSF lines); a probe.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kda_sp_fold.py -s
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.config import Glm53Config  # noqa: E402
from models.demos.blackhole.glm53_flash.tests.test_kda_device import read_states, run_sp, sp_schedule  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.kda import KdaWeights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.kda import SP_FOLD, TtKda  # noqa: E402

CASES = [
    (512, {5: (1500, 2)}),  # segments of 512: one request over 3 rows, the serving's prefill chunk
    (512, {5: (1100, 2), 1: (900, 3)}),  # two requests, two waves (keep 1 on the second)
    (128, {7: (1000, 1)}),  # the 1 x 1K form: 8 rows x 128
    (64, {5: (232, 2), 1: (322, 3)}),  # two requests on 4 rows each, row 1's in two waves
]


def _bits(a: torch.Tensor, b: torch.Tensor) -> int:
    a, b = a.contiguous(), b.contiguous()
    if a.dtype == torch.float32:
        return int((a.view(torch.int32) != b.view(torch.int32)).sum())
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_kda_sp_fold(mesh_device):
    try:
        _study(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KSF FAILED: {err!r}"[:3000], flush=True)


def _run(mesh, module, users, t, prompts, waves, fold: bool, state=None):
    os.environ[SP_FOLD] = "1" if fold else "0"
    try:
        got, st = run_sp(mesh, module, users, t, prompts, waves, state or module.empty_state(users))
        recurrent, history = read_states(mesh, module, st, users)
        return got, recurrent, history, st
    finally:
        os.environ.pop(SP_FOLD, None)


def _study(mesh):
    layer, users = 0, 4
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    module = TtKda(mesh, KdaWeights.from_checkpoint(checkpoint, layer), config)
    g = torch.Generator().manual_seed(77)
    clean = 0
    for t, requests in CASES:
        prompts = {
            o: (torch.randn(n, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
            for o, (n, _) in requests.items()
        }
        waves = sp_schedule(requests, t)
        got0, rec0, hist0, _ = _run(mesh, module, users, t, prompts, waves, False)
        got1, rec1, hist1, _ = _run(mesh, module, users, t, prompts, waves, True)
        dout = sum(_bits(got1[o], got0[o]) for o in got0)
        drec = _bits(rec1, rec0)
        dhist = sum(_bits(h1, h0) for h1, h0 in zip(hist1, hist0))
        ok = dout == 0 and drec == 0 and dhist == 0
        clean += ok
        print(f"KSF case t {t} {requests} ({len(waves)} waves): outputs {dout}, user states {drec}, conv histories "
              f"{dhist} values differ{' (bit for bit)' if ok else ''}", flush=True)  # fmt: skip
    print(f"KSF summary: {clean} / {len(CASES)} cases bit for bit", flush=True)
    print("KSF done", flush=True)
