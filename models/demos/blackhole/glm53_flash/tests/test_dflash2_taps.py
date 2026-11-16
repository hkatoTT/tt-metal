"""The DFlash2 drafter's taps from the device model: the per-token mean of the streams after
the drafter's target layers, collected by the prefill (`TtGlm53Model.taps`), against the CPU reference's
(tools/dflash2_acceptance.py's saved taps, `after[layer]`: the mean of the 4 mHC streams after that layer) on greedy
records, every position of the prompt and the record's tokens: the pcc and the largest difference over the largest
value, per layer.  The real weights (GLM53_CHECKPOINT), one request on row 0, eager.  GLM53_DFLASH2_SAVE (a directory):
the device's taps saved there in the CPU taps' form, for tools/dflash2_acceptance.py --taps-dir (the drafter's
acceptance on the device's taps).

    GLM53_DFLASH2_TAPS (default data_dir("dflash2-v1", "taps")), GLM53_DFLASH2_IDS (fact,code; all: every saved record)
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dflash2_taps.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import data_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_mod  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

TARGET_LAYERS = (5, 9, 14, 19, 24, 28, 33, 38, 42)  # the drafter's config.json target_layer_ids


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_dflash2_taps(mesh_device):
    taps_dir = Path(os.environ.get("GLM53_DFLASH2_TAPS") or data_dir("dflash2-v1", "taps"))
    ids = os.environ.get("GLM53_DFLASH2_IDS", "fact,code").split(",")
    if ids == ["all"]:
        ids = sorted(f.stem for f in taps_dir.glob("*.pt"))
    save = os.environ.get("GLM53_DFLASH2_SAVE")
    if save:
        Path(save).mkdir(parents=True, exist_ok=True)
    model = TtGlm53Model(mesh_device, Glm53Checkpoint(), max_seq=1024)
    dsa_mod.PREFILL_SP = False  # (one request on its own row)
    gen = Glm53Generator(model, users_per_row=1, chunk=(128, 512))
    model.tap_layers = frozenset(layer + 1 for layer in TARGET_LAYERS)
    worst = 1.0
    for rid in ids:
        saved = torch.load(taps_dir / f"{rid}.pt", weights_only=False)
        tokens = saved["ids"].tolist()
        n = len(tokens)
        model.taps = []
        gen.prefill_many([(0, tokens)], chunk=512, traced=False)
        got, model.taps = model.taps, None
        assert len(got) == len(TARGET_LAYERS), f"{len(got)} taps for {len(TARGET_LAYERS)} layers"
        mine = {}
        for layer, t in zip(TARGET_LAYERS, got):
            dev = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0, 0, :n]  # row 0's chip 0
            ttnn.deallocate(t)
            mine[layer] = dev.to(torch.bfloat16)
            ref = saved["after"][layer].float()[:n]
            p = pcc(dev, ref)
            rel = float((dev - ref).abs().max() / ref.abs().max())
            worst = min(worst, p)
            print(f"{rid}: layer {layer} tap ({n} positions): pcc {p:.6f}, max |diff| / max {rel:.3e}", flush=True)
        if save:
            keep = {k: saved[k] for k in ("ids", "prompt", "main_top") if k in saved}
            torch.save({"after": mine, **keep}, Path(save) / f"{rid}.pt")
    gen.release()
    print(f"worst pcc {worst:.6f}", flush=True)
    assert worst > 0.99
