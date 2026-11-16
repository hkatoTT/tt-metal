"""The model's values against a saved run, bit for bit: for a change that must not move them (a code cleanup, a kernel
rearranged).  Synthetic weights (seeded per tensor: the same in every run), the generator with MTP (one draft), its
warm-up's captures; then:
  - a long request alone (the sequence-parallel prefill over the 8 rows) and 8 short ones (a row each, traced chunks),
    the logits after each prompt;
  - GLM53_SAME_STEPS plain decode steps (the captured logits step) on those slots, the logits;
  - as many verify steps (eager, with the logits: model and MTP), the drafts taken from the step before.
Everything is saved to GLM53_SAME_DIR/<GLM53_SAME_TAG>.pt (default out_dir("model_same")/now.pt); with
GLM53_SAME_AGAINST another tag's saved values are compared bit for bit (the test fails on any difference).
Run (a device test):
    GLM53_DEVICE_TEST=1 GLM53_SYNTHETIC_WEIGHTS=1 pytest models/demos/blackhole/glm53_flash/tests/test_model_same.py -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

USERS = 2


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_same(mesh_device):
    assert os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1", "synthetic weights (the same in every run)"
    root = Path(os.environ.get("GLM53_SAME_DIR") or out_dir("model_same"))
    tag, against = os.environ.get("GLM53_SAME_TAG", "now"), os.environ.get("GLM53_SAME_AGAINST")
    steps = int(os.environ.get("GLM53_SAME_STEPS", "6"))
    cache = 4096
    t0 = time.time()
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint(), max_seq=cache, mtp=True)
    gen = Glm53Generator(model, USERS, chunk=(128, 512), spec=1)
    warm = gen.warmup(greedy=True, logits=True, traced=(128, 512))
    print(f"loaded in {time.time() - t0:.0f} s; warm-up {warm}", flush=True)
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(11)
    prompt = lambda n: torch.randint(0, vocab, (n,), generator=g).tolist()  # noqa: E731
    got = {}

    # the long request alone (over the rows), then a short one a row (slots 2 r + 1)
    long_req = [(0, prompt(3000))]
    short = [(2 * r + 1, prompt(60 + 45 * r)) for r in range(ROWS)]
    got["prefill long"] = torch.stack(gen.prefill_many(long_req)).float()
    got["prefill short"] = torch.stack(gen.prefill_many(short)).float()
    reqs = long_req + short
    slots = [s for s, _ in reqs]
    tok = {s: int(lg.argmax()) for s, lg in zip(slots, [*got["prefill long"], *got["prefill short"]])}
    pos = {s: len(p) for s, p in reqs}

    # plain decode steps (the captured logits step)
    n = gen.slots
    for step in range(steps):
        tokens = torch.zeros(n, dtype=torch.int64)
        positions = torch.full((n,), -1, dtype=torch.int64)
        for s in slots:
            tokens[s], positions[s] = tok[s], pos[s]
        lg = gen.decode(tokens, positions, greedy=False).float()
        got[f"plain {step}"] = lg[slots]
        for s in slots:
            tok[s], pos[s] = int(lg[s].argmax()), pos[s] + 1

    # verify steps, eager with the logits (as test_model_device_long's), each slot's draft from the step before
    rows = lambda t: t.view(ROWS, USERS, 2).transpose(1, 2).reshape(ROWS, 2 * USERS)  # noqa: E731
    draft = {s: (tok[s] * 7 + 1) % vocab for s in slots}
    keep = torch.ones(n, dtype=torch.int64)
    for step in range(steps):
        tokens = torch.zeros(n, 2, dtype=torch.int64)
        positions = torch.zeros(n, 2, dtype=torch.int64)
        for s in slots:
            tokens[s] = torch.tensor([tok[s], draft[s]])
            positions[s] = torch.tensor([pos[s], pos[s] + 1])
        model.write_spec_inputs(gen.sio, rows(tokens), rows(positions), keep.view(ROWS, USERS).float())
        ids_d, drafts_d, lg_m, lg_d = model.forward_spec(gen.sio, gen.states, logits=True)
        # [ROWS, 2 U] (the users' tokens, then their drafts) -> [slots, 2]; the logits [slots, 2, V]
        per_slot = lambda t: t.view(ROWS, 2, USERS, *t.shape[2:]).transpose(1, 2).reshape(n, 2, *t.shape[2:])  # noqa
        ids = per_slot(model.ids_to_host(ids_d, 2 * USERS))
        drafts = per_slot(model.ids_to_host(drafts_d, 2 * USERS))
        got[f"verify {step} model"] = per_slot(model.logits_to_host(lg_m, 2 * USERS).float())[slots]
        got[f"verify {step} mtp"] = per_slot(model.logits_to_host(lg_d, 2 * USERS).float())[slots]
        got[f"verify {step} ids"] = torch.stack([ids, drafts])[:, slots]
        for t in (ids_d, drafts_d, lg_m, lg_d):
            ttnn.deallocate(t)
        for s in slots:
            a = int(draft[s] == int(ids[s, 0]))
            keep[s] = a
            tok[s] = int(ids[s, a])
            pos[s] += a + 1
            draft[s] = int(drafts[s, a])
    gen.release()

    root.mkdir(parents=True, exist_ok=True)
    torch.save(got, root / f"{tag}.pt")
    finite = all(bool(torch.isfinite(v.float()).all()) for v in got.values())
    line = f"{tag}: {len(got)} tensors saved, finite {finite}"
    if against:
        old = torch.load(root / f"{against}.pt")
        diff = [k for k in got if not torch.equal(got[k], old[k])]
        line += f"; against {against}: {'bit for bit' if not diff else 'DIFFER: ' + ', '.join(diff)}"
        for k in diff[:5]:
            d = (got[k].float() - old[k].float()).abs()
            line += f"\n  {k}: max |d| {d.max().item():.3g}"
    print(line, flush=True)
    assert finite
    if against:
        assert not diff
