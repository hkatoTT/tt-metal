# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash text generation on a Blackhole Galaxy (the 8x4 mesh).

Loads the BF16 checkpoint ($GLM53_CHECKPOINT, else $HF_MODEL; the large matrices are converted to BFP8 as they load,
a few minutes), renders each prompt with the checkpoint's chat template, prefills the prompts and decodes greedily
through the serving generator (`ttnn/generator.py`).  With $GLM53_DFLASH2_DRAFTER set the decode is DFlash2 speculative
decoding: the drafter on the device proposes K tokens a step and the model keeps those equal to its own greedy choice
(the greedy output, faster); without it, the plain greedy decode, one token a step.  With the drafter the model is
built for its DFlash2 step as the vLLM server builds it (`dflash_shape`: the users a mesh row and K + 1 tokens each),
so it takes that shape's defaults, such as the L1 landing of the next MoE layer's expert weights (GLM53_W1).  Prints
each prompt's generation and the decode speed.

    pytest models/demos/blackhole/glm53_flash/demo/demo.py
    pytest models/demos/blackhole/glm53_flash/demo/demo.py \
        --input_prompts models/demos/blackhole/glm53_flash/demo/sample_prompts/questions.json --max_generated_tokens 128

Options (pytest command-line options, named as in the other tt-metal demos; `demo/conftest.py`), each over its
environment variable, over the default:
    --input_prompts FILE         GLM53_DEMO_PROMPTS: a JSON file of prompts, a list of strings or of {"prompt": ...}
                                 (up to 64; prompt i runs on mesh row i % 8); GLM53_DEMO_PROMPT: one prompt (default:
                                 a short question)
    --max_generated_tokens N     GLM53_DEMO_MAX_NEW: new tokens a prompt at most (default 256; a prompt stops earlier
                                 at an end token)
    --batch_size N               the prompts to run: the prompts repeated (or cut) to N, 1 .. 64 (default: as many
                                 as given)
    --max_seq_len N              the cache a user, in tokens (default: room for the longest prompt and its new
                                 tokens), rounded up to a multiple of 512, at least 1,024.  Past 2,051 tokens the DSA
                                 cache takes its long-context form, and the warm-up compiles more prefill forms
Environment only:
    GLM53_DEMO_REASONING    the chat template's reasoning effort: low (default: a short reasoning part), high or max
    GLM53_DEMO_K            DFlash2's drafts a step, 1 .. 7 (default 7)
    GLM53_SYNTHETIC_WEIGHTS=1  random weights made on the device (loads in seconds; the text is noise)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

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
MIN_CACHE = 1024


@dataclass(frozen=True)
class DemoSettings:
    prompts: list
    max_new: int
    reasoning: str
    k: int
    max_seq: int | None  # the cache a user asked for (None: from the prompts, `cache_length`)
    source: str  # where the prompts came from (for the log)


def load_prompts(path: str | Path) -> list[str]:
    """A prompts file: a JSON list of strings or of {"prompt": ...} objects (other keys are ignored)."""
    items = json.loads(Path(path).read_text())
    if not isinstance(items, list) or not items:
        raise ValueError(f'{path}: a JSON list of prompts (strings or {{"prompt": ...}} objects)')
    prompts = []
    for i, item in enumerate(items):
        if isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompts.append(item["prompt"])
        elif isinstance(item, str):
            prompts.append(item)
        else:
            raise ValueError(f'{path}: item {i} is neither a string nor an object with a "prompt" string')
    return prompts


def demo_settings(
    option: Callable[[str], object] = lambda name: None, environ: Mapping[str, str] = os.environ
) -> DemoSettings:
    """The demo's settings: each pytest option (`option("--input_prompts")`, ..., None where not given) over its
    environment variable (GLM53_DEMO_*), over the default."""

    def number(name: str | None, env: str | None, default: int | None) -> int | None:
        value = option(name) if name else None
        if value is None and env and environ.get(env):
            value = environ[env]
        return default if value is None else int(value)

    path = option("--input_prompts") or environ.get("GLM53_DEMO_PROMPTS")
    if path:
        prompts, source = load_prompts(path), str(path)
    else:
        prompts = [environ.get("GLM53_DEMO_PROMPT") or DEFAULT_PROMPT]
        source = "GLM53_DEMO_PROMPT" if environ.get("GLM53_DEMO_PROMPT") else "the default prompt"
    batch = number("--batch_size", None, None)
    if batch is not None:
        if not 1 <= batch <= MAX_PROMPTS:
            raise ValueError(f"--batch_size {batch}: 1 to {MAX_PROMPTS}")
        prompts = [prompts[i % len(prompts)] for i in range(batch)]  # (repeated in order, or the first `batch`)
    if not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(f"{len(prompts)} prompts ({source}): 1 to {MAX_PROMPTS} (--batch_size picks the first ones)")
    max_new = number("--max_generated_tokens", "GLM53_DEMO_MAX_NEW", 256)
    if max_new < 1:
        raise ValueError(f"--max_generated_tokens {max_new}: at least 1")
    max_seq = number("--max_seq_len", None, None)
    if max_seq is not None and max_seq < 1:
        raise ValueError(f"--max_seq_len {max_seq}: at least 1")
    k = number(None, "GLM53_DEMO_K", 7)
    reasoning = environ.get("GLM53_DEMO_REASONING") or "low"
    return DemoSettings(prompts, max_new, reasoning, k, max_seq, source)


def cache_length(longest_prompt: int, max_new: int, asked: int | None = None) -> int:
    """The cache a user (the model's max_seq): room for the longest prompt, its new tokens and a few more, in multiples
    of the longest prefill chunk, at least MIN_CACHE; `asked` (--max_seq_len): that, rounded up the same way, if it
    has the room."""
    step = CHUNKS[-1]
    needed = longest_prompt + max_new + 16
    if asked is not None and asked < needed:
        raise ValueError(
            f"--max_seq_len {asked}: the longest prompt ({longest_prompt} tokens) and --max_generated_tokens "
            f"({max_new}) need {needed}"
        )
    length = needed if asked is None else asked
    return max(MIN_CACHE, -(-length // step) * step)


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
def test_demo(mesh_device, request):
    from transformers import AutoTokenizer

    settings = demo_settings(lambda name: request.config.getoption(name, default=None))
    root = checkpoint_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(root))
    eos = set(json.loads((root / "generation_config.json").read_text())["eos_token_id"])
    max_new, prompts = settings.max_new, settings.prompts
    ids = render(tokenizer, prompts, settings.reasoning)

    def trim(seq: list) -> list:
        seq = seq[:max_new]
        return next((seq[: j + 1] for j, t in enumerate(seq) if t in eos), seq)

    def done(seq: list) -> bool:
        return len(seq) >= max_new or any(t in eos for t in seq)

    users = -(-len(prompts) // ROWS)  # users a mesh row
    slots = [(i % ROWS) * users + i // ROWS for i in range(len(prompts))]  # prompt i on mesh row i % 8
    max_seq = cache_length(max(len(p) for p in ids), max_new, settings.max_seq)  # the cache, in tokens a user
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"

    k, dflash = settings.k, None
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
    logger.info(
        f"loaded in {time.time() - t0:.0f} s: {len(prompts)} prompts ({settings.source}), {users} a mesh row, "
        f"cache {max_seq}"
    )
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
