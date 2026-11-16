"""The generator's warm-up called in parts, as the vLLM adapter does (`tools/glm53_vllm.py`: `warmup_model_prefill`, then
`warmup_model_decode`), against one call (the tests' form).  With the first call's captures (MTP's verify step and the
first drafts) kept, the second call's buffers would sit in their scratch, and the plain greedy decode gave zeros after
the first token once the first drafts' trace had run; `Glm53Generator.warmup` releases the earlier calls' captures and
captures them all again after its own allocations.

Orders (GLM53_WO_ORDERS, default "adapter,single"; each after the last one's captures are released):
  adapter      the adapter's two calls
  single       one call: warmup(greedy=True, logits=True, traced=(128, 512))
For each: the warm-up's times and trace region, then GLM53_WO_PROMPTS random prompts of GLM53_WO_LEN tokens in slots
over the rows, prefilled together, STEPS tokens of the plain greedy decode, the same prefill again and the verify
step (MTP, one draft): its acceptance and its tokens against the plain decode's, both against the last order's.
    GLM53_WO_CACHE (default 2048: the dense cache), GLM53_SYNTHETIC_WEIGHTS=1: random weights (the bug's zeros showed
    on them too, 8 / 8 prompts; a short regression test, about 2 minutes)
Run (a device test): GLM53_DEVICE_TEST=1 GLM53_MTP=1 pytest ... -s
With TT_METAL_TRACE_ALLOC_TRACKING=1 GLM53_TRACE_ALLOC_GUARD=1 (or log) it also checks the guard itself: a buffer allocated
after the warm-up and alive at a replay is caught (`common.arm_trace_alloc_guard`), and none of the orders' own steps trips
it.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import common as common_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import model as model_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

USERS = 4  # a row: serving's 32 seats
STEPS = int(os.environ.get("GLM53_WO_STEPS", "32"))


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_warmup_order(mesh_device):
    cache = int(os.environ.get("GLM53_WO_CACHE", "2048"))
    n = int(os.environ.get("GLM53_WO_LEN", "1024"))
    count = int(os.environ.get("GLM53_WO_PROMPTS", "8"))
    orders = os.environ.get("GLM53_WO_ORDERS", "adapter,single").split(",")
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    t0 = time.time()
    model = TtGlm53Model(
        mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=cache, mtp=True
    )
    gen = Glm53Generator(model, USERS, chunk=(128, 512), spec=1)
    print(f"loaded in {time.time() - t0:.0f} s; cache {cache}, {count} prompts of {n}", flush=True)
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, vocab, (n,), generator=g).tolist() for _ in range(count)]
    slots = [(k % 8) * USERS + (k * 5 // 8) % USERS for k in range(count)]  # a row each, users 0-3
    nslots = gen.slots

    def warm(order: str) -> dict:
        used = model_module.trace_bytes(mesh_device)
        t1 = time.time()
        if order.startswith("adapter"):
            parts = [gen.warmup(greedy=False, logits=False)]
            parts.append(gen.warmup(greedy=True, logits=True, tokens=gen.chunk, traced=(128, 512)))
        else:
            parts = [gen.warmup(greedy=True, logits=True, traced=(128, 512))]
        mb = (model_module.trace_bytes(mesh_device) - used) / 2**20
        print(f"{order}: warm-up {time.time() - t1:.1f} s, trace region {mb:.1f} MB: {parts}", flush=True)
        return {"seconds": round(time.time() - t1, 1), "trace_mb": round(mb, 1)}

    def plain() -> list:
        gen.first.clear()
        out = [[int(lg.argmax())] for lg in gen.prefill_many(list(zip(slots, prompts)))]
        pos = [len(p) for p in prompts]
        for _ in range(STEPS):
            tokens = torch.zeros(nslots, dtype=torch.int64)
            positions = torch.full((nslots,), -1, dtype=torch.int64)
            for k, s in enumerate(slots):
                tokens[s], positions[s] = out[k][-1], pos[k]
            ids = gen.decode(tokens, positions, greedy=True)
            for k, s in enumerate(slots):
                out[k].append(int(ids[s]))
                pos[k] += 1
        return out

    def spec() -> tuple:
        gen.first.clear()
        first = [int(lg.argmax()) for lg in gen.prefill_many(list(zip(slots, prompts)))]
        out, tok = [[f] for f in first], list(first)
        draft = [gen.first[s] for s in slots]
        pos, keep = [len(p) for p in prompts], [1] * count
        steps, accepted = [0] * count, [0] * count
        while any(len(o) < STEPS + 1 for o in out):
            tokens = torch.zeros(nslots, 2, dtype=torch.int64)
            positions = torch.full((nslots, 2), -1, dtype=torch.int64)
            flags = torch.ones(nslots, dtype=torch.int64)
            for k, s in enumerate(slots):
                tokens[s] = torch.tensor([tok[k], draft[k]])
                positions[s] = torch.tensor([pos[k], pos[k] + 1])
                flags[s] = keep[k]
            ids, drafts = gen.verify(tokens, positions, flags)
            for k, s in enumerate(slots):
                gt, gd = int(ids[s, 0]), int(ids[s, 1])
                a = int(draft[k] == gt)
                if len(out[k]) < STEPS + 1:
                    steps[k] += 1
                    accepted[k] += a
                out[k] += [draft[k], gd] if a else [gt]
                keep[k], pos[k], tok[k] = a, pos[k] + a + 1, gd if a else gt
                draft[k] = int(drafts[s, a])
        return [o[: STEPS + 1] for o in out], [a / max(1, st) for a, st in zip(accepted, steps)]

    def same(a: list, b: list) -> int:
        m = 0
        while m < len(a) and m < len(b) and a[m] == b[m]:
            m += 1
        return m

    results, last = {}, None
    for i, order in enumerate(orders):
        if i:
            gen.release()
        info = warm(order)
        p_out = plain()
        s_out, acc = spec()
        zeros = sum(all(t == 0 for t in o[1:]) for o in p_out)
        info.update(
            plain_zero_prompts=zeros,
            acceptance=round(sum(acc) / len(acc), 3),
            spec_vs_plain=[same(a, b) for a, b in zip(s_out, p_out)],
        )
        if last is not None:
            info["plain_vs_last"] = [same(a, b) for a, b in zip(p_out, last[0])]
            info["spec_vs_last"] = [same(a, b) for a, b in zip(s_out, last[1])]
        print(
            f"{order}: plain decode all zeros after the first token in {zeros} / {count} prompts; acceptance "
            f"{[round(a, 3) for a in acc]} (mean {info['acceptance']}); verify = plain for {info['spec_vs_plain']}"
            + (
                f"; against {orders[i - 1]}: plain {info['plain_vs_last']}, verify {info['spec_vs_last']}"
                if last is not None
                else ""
            ),
            flush=True,
        )
        print(f"  first tokens: plain {[o[:6] for o in p_out[:3]]}, verify {[o[:6] for o in s_out[:3]]}", flush=True)
        results[order] = info
        last = (p_out, s_out)
    print(f"summary {results}", flush=True)
    if common_module.TRACE_ALLOC_GUARD != "0":  # the guard catches a buffer made after the captures, alive at a replay
        assert common_module._GUARD["flagged"] == 0, "a buffer allocated after the warm-up was alive at a replay"
        tokens = torch.zeros(nslots, dtype=torch.int64)
        positions = torch.full((nslots,), -1, dtype=torch.int64)
        stray = ttnn.zeros([1, 1, 32, 32], device=mesh_device, layout=ttnn.TILE_LAYOUT)
        try:
            gen.decode(tokens, positions, greedy=True)
            caught = common_module._GUARD["flagged"] > 0  # "log": printed
        except RuntimeError as e:  # "1": stopped before the replay
            caught = "allocated after the captures" in str(e)
        ttnn.deallocate(stray)
        gen.decode(tokens, positions, greedy=True)  # clean again
        print(
            f"trace allocation guard ({common_module.TRACE_ALLOC_GUARD}): the stray buffer caught {caught}", flush=True
        )
        assert caught, "the trace allocation guard missed a buffer allocated after the captures"
    gen.release()
    if "adapter" in results:
        assert results["adapter"]["plain_zero_prompts"] == 0, "the adapter's warm-up broke the plain decode"
    if orders[-2:] == ["adapter", "single"]:  # the two orders give the same tokens
        full = [STEPS + 1] * count
        assert (
            results["single"]["plain_vs_last"] == full and results["single"]["spec_vs_last"] == full
        ), "the adapter's warm-up and one call give other tokens"
