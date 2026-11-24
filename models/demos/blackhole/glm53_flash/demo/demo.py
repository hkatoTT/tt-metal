# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash text generation on a Blackhole Galaxy (the 8x4 mesh).

Loads the BF16 checkpoint ($GLM53_CHECKPOINT; the large matrices are converted to BFP8 as they load, a few
minutes), renders each prompt with the checkpoint's chat template, prefills the prompts and decodes greedily through
the serving generator (`ttnn/generator.py`).  With $GLM53_DFLASH2_DRAFTER set the decode is DFlash2 speculative
decoding: the drafter on the device proposes K tokens a step and the model keeps those equal to its own greedy choice
(the greedy output, faster); without it, the plain greedy decode, one token a step.  With the drafter the model is
built for its DFlash2 step as the vLLM server builds it (`dflash_shape`: the users a mesh row and K + 1 tokens each),
so it takes that shape's defaults, such as the L1 landing of the next MoE layer's expert weights (GLM53_W1).  Prints
each prompt's generation and the decode speed.

    pytest models/demos/blackhole/glm53_flash/demo/demo.py

Environment:
    GLM53_DEMO_PROMPT       one prompt (default: a short question)
    GLM53_DEMO_PROMPTS      a JSON file of prompts instead: a list of strings or of {"prompt": ...} (up to 64)
    GLM53_DEMO_MAX_NEW      new tokens a prompt at most (default 256; a prompt stops earlier at an end token)
    GLM53_DEMO_REASONING    the chat template's reasoning effort: low (default: a short reasoning part), high or max
    GLM53_DEMO_K            DFlash2's drafts a step, 1 .. 7 (default 7)
    GLM53_SYNTHETIC_WEIGHTS=1  random weights made on the device (loads in seconds; the text is noise)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint
from models.demos.blackhole.glm53_flash.paths import checkpoint_dir
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model

DEFAULT_PROMPT = "Explain in a few sentences why the sky is blue."
CHUNKS = (128, 512)  # the prefill's chunk lengths (the generator picks them per prompt)
MAX_PROMPTS = 8 * ROWS  # up to 8 users a mesh row


def load_prompts() -> list[str]:
    path = os.environ.get("GLM53_DEMO_PROMPTS")
    if path:
        items = json.loads(Path(path).read_text())
        prompts = [item["prompt"] if isinstance(item, dict) else str(item) for item in items]
    else:
        prompts = [os.environ.get("GLM53_DEMO_PROMPT", DEFAULT_PROMPT)]
    if not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(f"{len(prompts)} prompts: 1 to {MAX_PROMPTS}")
    return prompts


def render(tokenizer, prompts: list[str], effort: str) -> list[list[int]]:
    """Each prompt as a user message through the chat template, ending with the assistant's turn."""
    out = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            reasoning_effort=effort,
        )
        out.append([int(i) for i in rendered["input_ids"]])
    return out


def plain_decode(gen, firsts, starts, slots, done) -> tuple[list, int]:
    """One token a step for every slot (the captured greedy step) until each prompt is done -> (tokens, steps)."""
    outs = [[f] for f in firsts]
    tok = torch.zeros(gen.slots, dtype=torch.int64)
    pos = torch.full((gen.slots,), -1, dtype=torch.int64)  # -1: no request in the slot
    live = {i for i in range(len(firsts)) if not done(outs[i])}
    for i in live:
        tok[slots[i]], pos[slots[i]] = firsts[i], starts[i]
    steps = 0
    while live:
        ids = gen.decode(tok, pos, greedy=True)
        steps += 1
        for i in list(live):
            s = slots[i]
            outs[i].append(int(ids[s]))
            tok[s], pos[s] = ids[s], pos[s] + 1
            if done(outs[i]):
                live.discard(i)
                pos[s] = -1
    return outs, steps


def dflash_decode(gen, k, firsts, starts, slots, done) -> tuple[list, int, float]:
    """DFlash2 steps (`Glm53Generator.dflash_verify`: each slot's token and its K drafts verified, the drafts equal to
    the model's greedy tokens accepted, the next K drafts made) -> (tokens, steps, the mean tokens a step a prompt)."""
    n_slots = gen.slots
    bonus = torch.zeros(n_slots, dtype=torch.int64)  # each slot's next token (not yet run through the model)
    drafts = torch.zeros(n_slots, k, dtype=torch.int64)
    pos = torch.full((n_slots,), -1, dtype=torch.int64)  # the bonus token's position, -1: no request
    keep = torch.full((n_slots,), k, dtype=torch.int64)  # the drafts accepted in the slot's last step (K: new)
    outs = [[f] for f in firsts]
    live = set()
    for i, s in enumerate(slots):
        first_drafts = gen.first.pop(s)  # (the drafter's first drafts after the prompt)
        if not done(outs[i]):
            bonus[s], pos[s], drafts[s] = firsts[i], starts[i], torch.tensor(first_drafts)
            live.add(i)
    steps = advanced = turns = 0
    while live:
        tokens = torch.cat([bonus[:, None], drafts], dim=1)
        positions = torch.where(pos[:, None] >= 0, pos[:, None] + torch.arange(k + 1)[None, :], -1)
        accepted, nxt, new_drafts = gen.dflash_verify(tokens, positions, keep)
        steps += 1
        for i in list(live):
            s = slots[i]
            n = int(accepted[s])
            outs[i] += drafts[s, :n].tolist() + [int(nxt[s])]
            advanced, turns = advanced + n + 1, turns + 1
            bonus[s], drafts[s], keep[s], pos[s] = nxt[s], new_drafts[s], n, pos[s] + n + 1
            if done(outs[i]):
                live.discard(i)
                pos[s] = -1
    return outs, steps, advanced / max(turns, 1)


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_demo(mesh_device):
    from transformers import AutoTokenizer

    root = checkpoint_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(root))
    eos = set(json.loads((root / "generation_config.json").read_text())["eos_token_id"])
    max_new = int(os.environ.get("GLM53_DEMO_MAX_NEW", "256"))
    prompts = load_prompts()
    ids = render(tokenizer, prompts, os.environ.get("GLM53_DEMO_REASONING", "low"))

    def trim(seq: list) -> list:
        seq = seq[:max_new]
        return next((seq[: j + 1] for j, t in enumerate(seq) if t in eos), seq)

    def done(seq: list) -> bool:
        return len(seq) >= max_new or any(t in eos for t in seq)

    users = -(-len(prompts) // ROWS)  # users a mesh row
    slots = [(i % ROWS) * users + i // ROWS for i in range(len(prompts))]  # prompt i on mesh row i % 8
    longest = max(len(p) for p in ids) + max_new + 16
    max_seq = max(1024, -(-longest // CHUNKS[-1]) * CHUNKS[-1])  # the cache, in tokens a user
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"

    k, dflash = int(os.environ.get("GLM53_DEMO_K", "7")), None
    drafted = bool(os.environ.get("GLM53_DFLASH2_DRAFTER"))
    if drafted and not 1 <= k <= 7:
        raise ValueError(f"GLM53_DEMO_K={k}: 1 to 7")

    t0 = time.time()
    model = TtGlm53Model(
        mesh_device,
        ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(),
        max_seq=max_seq,
        mtp=False,
        dflash_shape=(users, k + 1) if drafted else None,  # (the served form's defaults for this DFlash2 step)
    )
    if drafted:
        from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights

        cfg, w, mask = load_weights()
        drafter = TtDFlash2(
            mesh_device, cfg, w, mask, users=users, max_ctx=max_seq + 8, block=4 if k <= 3 else 8,
            head_weight=model.lm_head.mesh_weight, embedding=model.embed,
        )  # fmt: skip
        dflash = (drafter, k)
    gen = Glm53Generator(model, users, chunk=CHUNKS, dflash=dflash)
    mode = f"DFlash2 with K = {k}" if dflash else "plain greedy decode (GLM53_DFLASH2_DRAFTER not set)"
    logger.info(f"loaded in {time.time() - t0:.0f} s: {len(prompts)} prompts, {users} a mesh row, cache {max_seq}")
    t0 = time.time()
    gen.warmup(greedy=dflash is None, logits=False)  # (compiles the prefill, then captures the decode's trace)
    logger.info(f"warmed up in {time.time() - t0:.0f} s; {mode}")

    t0 = time.time()
    firsts = gen.prefill_many(list(zip(slots, ids)), traced=False, greedy_ids=True)
    firsts = [int(f) for f in firsts]
    prefill_s = time.time() - t0
    starts = [len(p) for p in ids]  # the first generated token's position
    t0 = time.time()
    if dflash:
        outs, steps, per_step = dflash_decode(gen, k, firsts, starts, slots, done)
    else:
        outs, steps = plain_decode(gen, firsts, starts, slots, done)
    decode_s = max(time.time() - t0, 1e-6)
    gen.release()

    outs = [trim(o) for o in outs]
    for prompt, out in zip(prompts, outs):
        text = tokenizer.decode([t for t in out if t not in eos], skip_special_tokens=False)
        logger.info(f"\nPROMPT: {prompt}\nOUTPUT ({len(out)} tokens): {text}\n")
    tokens = sum(len(o) - 1 for o in outs)  # (the first token of each came from the prefill)
    logger.info(f"prefill: {sum(starts)} prompt tokens in {prefill_s:.2f} s (eager)")
    logger.info(
        f"decode: {tokens} tokens in {decode_s:.2f} s, {steps} steps of {decode_s / max(steps, 1) * 1e3:.1f} ms; "
        f"{tokens / decode_s:.1f} tok/s in all, {tokens / decode_s / len(prompts):.1f} tok/s a prompt"
        + (f"; {per_step:.2f} tokens a step a prompt" if dflash else "")
    )
