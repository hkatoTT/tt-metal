"""The prefix store sized from the free DRAM at a long max_model_len (GLM53_PREFIX_ENTRIES unset: "auto", the vLLM
serving's default): the adapter made as the plugin makes it
(`Glm53ForCausalLM.initialize_vllm_model`, 32 seats, DFlash2, chunked prefill), its start-up line on the store's size,
the generator's warm-up, then one long prompt through the adapter in calls of 8,192 (each continuation snapshots its
state) and a hit on it (the prompt's last snapshot + new tokens, restored into a free seat of the entry's row): no
allocation fails, and the chip's DRAM after each.

    GLM53_PA_MAX_LEN   max_model_len (default 1,048,576)
    GLM53_PA_PROMPT    the long prompt's tokens (default max_model_len - 2,048), cut short by GLM53_PA_BUDGET seconds
                       of prefill calls (default 300: the prompt as far as it got)
    GLM53_PA_WARMUP    "serving" (the adapter's two warm-up calls, the prefill traces too) or "eager" (default: the
                       generator's warm-up of the whole cache without prefill traces, as the long-context tests; the
                       serving's at 1M takes longer than this test's time limit)

Synthetic weights by default (GLM53_SYNTHETIC_WEIGHTS=0: real; the DRAM is the same).  Prints "PREFIX AUTO: PASS" (the
store sized, the long prompt's calls and the hit with no error, the hit restored) and exits 0 (a study):

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_prefix_auto_size.py -s
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.prefix_keys import AUTO, GRID, prefix_keys  # noqa: E402

SEATS = 32


def dram(mesh) -> str:
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    banks, gib = int(view.num_banks), 2**30
    return (
        f"DRAM a chip {int(view.total_bytes_allocated_per_bank) * banks / gib:.2f} GiB allocated, "
        f"{int(view.total_bytes_free_per_bank) * banks / gib:.2f} GiB free (largest block "
        f"{int(view.largest_contiguous_bytes_free_per_bank) * banks / gib:.2f})"
    )


@pytest.mark.timeout(1400)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefix_auto_size(mesh_device):
    os.environ.setdefault("GLM53_SYNTHETIC_WEIGHTS", "1")
    os.environ["GLM53_MTP"] = "0"
    os.environ["GLM53_DFLASH2"] = "1"
    os.environ.pop("GLM53_PREFIX_ENTRIES", None)
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm

    glm53_vllm.MTP, glm53_vllm.DFLASH2, glm53_vllm.SPEC, glm53_vllm.CHUNKED = False, True, True, True
    glm53_vllm.PREFIX_SETTING, glm53_vllm.PREFIX = AUTO, True
    max_len = int(os.environ.get("GLM53_PA_MAX_LEN", str(1 << 20)))
    want = int(os.environ.get("GLM53_PA_PROMPT", str(max_len - 2048)))
    budget = float(os.environ.get("GLM53_PA_BUDGET", "300"))
    warm = os.environ.get("GLM53_PA_WARMUP", "eager")
    glm53_vllm_rows.reset()
    t0 = time.time()
    print(f"before the model: {dram(mesh_device)}", flush=True)
    adapter = glm53_vllm.Glm53ForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=None), mesh_device, max_batch_size=SEATS, max_seq_len=max_len
    )
    gen = adapter.gen
    store = gen.prefix
    print(f"made in {time.time() - t0:.0f} s; cache {gen.max_seq}; {gen.prefix_note}; {dram(mesh_device)}", flush=True)
    t0 = time.time()
    if warm == "serving":
        adapter.warmup_model_prefill()
        adapter.warmup_model_decode()
    else:
        gen.warmup(greedy=True, logits=False, chunks=(gen.chunk,), traced=())
    print(f"warmed up ({warm}) in {time.time() - t0:.0f} s; {dram(mesh_device)}", flush=True)
    verdict = [store is not None]
    if store is None:
        print("PREFIX AUTO: OFF (no store: the start-up's reason above)", flush=True)
        return

    vocab = gen.model.config.vocab_size
    g = torch.Generator().manual_seed(41)
    prompt = torch.randint(0, vocab, (want,), generator=g).tolist()
    block = [100]

    def call(slot: int, toks: list, begin: int, end: int, first: int) -> float:
        t1 = time.time()
        adapter.prefill_forward(
            torch.tensor(toks[:end]).view(1, end),
            torch.tensor([begin]),
            torch.tensor([end]),
            page_table=torch.tensor([[first, first + 1]]),
            empty_slots=torch.tensor([slot]),
            sampling_params=None,
        )
        return time.time() - t1

    # the long prompt in calls of GRID on slot 5 (as the plugin's chunked prefill), until the budget runs out
    t0, at, times = time.time(), 0, []
    while at < len(prompt) and time.time() - t0 < budget:
        end = min(len(prompt), (at // GRID + 1) * GRID)
        times.append(call(5, prompt, at, end, block[0]))
        at = end
    reached = at
    print(
        f"the long prompt: {reached} of {len(prompt)} tokens in {len(times)} calls, {time.time() - t0:.0f} s (the last "
        f"{times[-1]:.2f} s); snapshots {adapter.prefix_counts['snapshots']}; {dram(mesh_device)}",
        flush=True,
    )
    # a hit: its longest snapshot + 300 new tokens on slot 11 (row 3: moved to a free seat of the entry's row)
    length = (reached - 1) // GRID * GRID  # (the last call's start: its snapshot)
    turn = prompt[:length] + torch.randint(0, vocab, (300,), generator=g).tolist()
    state = glm53_vllm_rows.prefix_state()
    got = glm53_vllm_rows.prefix_hit(prefix_keys(turn), len(turn), *state) if state else None
    hits0 = adapter.prefix_counts["hits"]
    if got is not None:
        block[0] += 10
        glm53_vllm_rows.publish_hit(block[0], got[0], got[1])
        dt = call(11, turn, got[0], len(turn), block[0])
        print(f"the hit at {got[0]}: {len(turn)} tokens in {dt:.2f} s; {dram(mesh_device)}", flush=True)
    restored = adapter.prefix_counts["hits"] - hits0
    verdict += [got is not None and got[0] == length, restored == 1]
    print(f"prefix counts {adapter.prefix_counts}; seat moves {adapter.seat_moves}", flush=True)
    print(f"PREFIX AUTO: {'PASS' if all(verdict) else 'FAIL'} {verdict}", flush=True)
