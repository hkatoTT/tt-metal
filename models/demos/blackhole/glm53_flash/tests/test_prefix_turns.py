"""A conversation's turns through the vLLM adapter with the prefix store on (tools/glm53_vllm.py
PREFIX, as the plugin calls it, the scheduler's part by its decision function `glm53_vllm_rows.prefix_hit`): each turn's
prompt is the last prompt, its output and new tokens; a turn with a prefix in the store is prefilled from there
(restored into a free seat of the entry's row), against the same prompt prefilled from 0 on the same seat (no hit):
the logits after the prompt and the turn's greedy output (DFlash2 verify steps of that seat alone) must be the same bit
for bit.

Turns (vLLM's state slots on the adapter's physical seats: slot k on mesh row k mod 8, user k // 8; U = 4 a row):
  1. GLM53_PT_LEN1 tokens (default 11,000) on slot 5 (row 5) in chunks of 8,192: its second call snapshots the state at
     8,192; GLM53_PT_STEPS verify steps.
  2. turn 1's prompt + its output + GLM53_PT_NEW2 new tokens (default 5,500: past 16,384): the hit at 8,192 on slot 11
     (row 3: moved to a free seat of row 5), its calls from 8,192 (the second snapshots at 16,384); then the reference,
     the same prompt from 0 on that seat.
  3. turn 2's prompt + its output + GLM53_PT_NEW3 tokens (default 300): the hit at 16,384 on slot 20 (row 4: moved),
     then its reference.
The cache GLM53_PT_CACHE (default 17,408).  Real weights by default
(GLM53_SYNTHETIC_WEIGHTS=1: random).  Prints the verdict ("PREFIX TURNS: PASS") and the
prefill calls' times; exits 0 (a study):

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_prefix_turns.py -s
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
spec_decode = pytest.importorskip("vllm_tt_plugin.spec_decode")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.prefix_keys import GRID, prefix_keys  # noqa: E402

USERS = 4  # a row: 32 seats
SEATS = ROWS * USERS


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefix_turns(mesh_device):
    os.environ.setdefault("GLM53_SYNTHETIC_WEIGHTS", "0")
    os.environ["GLM53_MTP"] = "0"
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm

    glm53_vllm.MTP, glm53_vllm.DFLASH2, glm53_vllm.SPEC, glm53_vllm.CHUNKED = False, True, True, True
    # (greedy requests only: the generator is made here without GLM53_DFLASH2_SAMPLING, which the adapter's start-up
    # sets when SAMPLING is on, its default where the plugin certifies sampled ids)
    glm53_vllm.SAMPLING = False
    glm53_vllm.PREFIX = True
    synthetic = os.environ["GLM53_SYNTHETIC_WEIGHTS"] == "1"
    len1 = int(os.environ.get("GLM53_PT_LEN1", "11000"))
    new2 = int(os.environ.get("GLM53_PT_NEW2", "5500"))
    new3 = int(os.environ.get("GLM53_PT_NEW3", "300"))
    steps = int(os.environ.get("GLM53_PT_STEPS", "8"))
    max_len = int(os.environ.get("GLM53_PT_CACHE", "17408"))
    t0 = time.time()
    chunks = glm53_vllm.prefill_chunks()
    model = TtGlm53Model(
        mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=max_len, mtp=False
    )
    from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights

    k = glm53_vllm.dflash2_k(SEATS)
    cfg, w, dmask = load_weights(os.environ.get("GLM53_DFLASH2_DRAFTER"), synthetic=synthetic)
    drafter = TtDFlash2(
        mesh_device, cfg, w, dmask, users=USERS, max_ctx=max_len + 8, block=4 if k <= 3 else 8,
        head_weight=model.lm_head.mesh_weight, embedding=model.embed,
    )  # fmt: skip
    gen = Glm53Generator(model, USERS, chunk=chunks, spec=0, dflash=(drafter, k), prefix_entries=4)
    glm53_vllm_rows.reset()
    adapter = glm53_vllm.Glm53ForCausalLM(gen, max_len)
    assert adapter.prefix_on
    print(f"loaded in {time.time() - t0:.0f} s; cache {gen.max_seq}, K {k}", flush=True)
    t0 = time.time()
    adapter.warmup_model_prefill()
    adapter.warmup_model_decode()
    print(f"warmed up in {time.time() - t0:.0f} s", flush=True)

    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(31)
    mode = spec_decode.ACCEPT_MODE_ARGMAX_IDS
    counts = [1] * SEATS
    block = [100]  # (each request's first KV block: the page table's first entry, as the plugin's)

    def prefill_call(slot: int, prompt: list, begin: int, end: int, first: int) -> tuple:
        t1 = time.time()
        out = adapter.prefill_forward(
            torch.tensor(prompt[:end]).view(1, end),
            torch.tensor([begin]),
            torch.tensor([end]),
            page_table=torch.tensor([[first, first + 1]]),
            empty_slots=torch.tensor([slot]),
            sampling_params=None,
        )
        return out[0, 0, :], time.time() - t1

    def prefill(slot: int, prompt: list, begin: int) -> tuple:
        """A request's prefill calls as the plugin makes them with chunked prefill (from `begin`, ending on the
        8,192 grid) -> (the logits after the prompt, the calls' (begin, end, seconds))."""
        block[0] += 10
        calls, at = [], begin
        while at < len(prompt):
            end = min(len(prompt), (at // GRID + 1) * GRID)
            logits, dt = prefill_call(slot, prompt, at, end, block[0])
            calls.append((at, end, round(dt, 3)))
            at = end
        return logits, calls

    def propose() -> list:
        out = adapter.propose_draft_tokens(
            k, torch.zeros(SEATS, 1, dtype=torch.int32), torch.zeros(SEATS, dtype=torch.int64), torch.tensor(counts)
        )
        return out.draft_token_ids.tolist()

    def generate(slot: int, logits: torch.Tensor, end: int) -> list:
        """The slot alone after its prefill: its first token and `steps` verify steps -> its tokens."""
        for s in range(SEATS):
            counts[s] = 1
        r = dict(tok=int(logits.argmax()), pos=end)
        toks = [r["tok"]]
        r["draft"] = propose()[slot]
        for _ in range(steps):
            tokens = torch.zeros(SEATS, k + 1, dtype=torch.int64)
            pos = torch.full((SEATS, k + 1), -1, dtype=torch.int64)
            valid = torch.zeros(SEATS, dtype=torch.int32)
            tokens[slot] = torch.tensor([r["tok"]] + r["draft"])
            pos[slot] = r["pos"] + torch.arange(k + 1)
            valid[slot] = k
            out = adapter.decode_forward(
                tokens,
                pos,
                spec_mode=mode,
                num_valid_drafts=valid,
                accepted_counts=torch.tensor(counts, dtype=torch.int32),
                sampling_params=None,
            )
            ids = out.argmax_ids.tolist()[slot]
            a = 0
            while a < k and r["draft"][a] == ids[a]:
                a += 1
            r["pos"], r["tok"] = r["pos"] + a + 1, ids[a]
            toks.extend(ids[: a + 1])
            counts[slot] = a + 1
            r["draft"] = propose()[slot]
        return toks

    def hit_for(prompt: list):
        """The scheduler's choice for a waiting request of this prompt, published as it does on admission."""
        state = glm53_vllm_rows.prefix_state()
        got = glm53_vllm_rows.prefix_hit(prefix_keys(prompt), len(prompt), *state)
        if got is not None:
            glm53_vllm_rows.publish_hit(block[0] + 10, got[0], got[1])  # (the next request's first block)
        return got

    verdict = []
    # turn 1
    prompt = torch.randint(0, vocab, (len1,), generator=g).tolist()
    logits, calls = prefill(5, prompt, 0)
    out = generate(5, logits, len(prompt))
    print(f"turn 1: {len(prompt)} tokens, calls {calls}; snapshots {adapter.prefix_counts['snapshots']}", flush=True)
    for turn, (slot, new) in enumerate(((11, new2), (20, new3)), start=2):
        prompt = prompt + out + torch.randint(0, vocab, (new,), generator=g).tolist()
        got = hit_for(prompt)
        hits0 = adapter.prefix_counts["hits"]
        logits_h, calls_h = prefill(slot, prompt, got[0] if got else 0)
        seat = adapter.phys[slot]
        out_h = generate(slot, logits_h, len(prompt))
        # the reference: the same prompt from 0 on the same seat
        logits_r, calls_r = prefill(slot, prompt, 0)
        assert adapter.phys[slot] == seat
        out_r = generate(slot, logits_r, len(prompt))
        same_l, same_t = torch.equal(logits_h, logits_r), out_h == out_r
        hit = adapter.prefix_counts["hits"] - hits0
        first = next((i for i, (x, y) in enumerate(zip(out_h, out_r)) if x != y), None)
        print(
            f"turn {turn}: {len(prompt)} tokens on slot {slot} (seat {seat}, row {seat // USERS}); hit {got and got[0]} "
            f"({'restored' if hit else 'NOT restored'}); calls {calls_h} against from 0 {calls_r}: the prefill "
            f"{sum(c[2] for c in calls_h):.2f} s against {sum(c[2] for c in calls_r):.2f} s; logits "
            f"{'the same' if same_l else 'DIFFERENT'} (max |diff| {(logits_h - logits_r).abs().max().item():.3g}); "
            f"output {'the same' if same_t else f'DIFFERENT from token {first}'} ({len(out_h)} tokens)",
            flush=True,
        )
        verdict.append(bool(hit) and same_l and same_t)
        out = out_h
    print(f"prefix counts {adapter.prefix_counts}", flush=True)
    print(f"PREFIX TURNS: {'PASS' if all(verdict) else 'FAIL'} {verdict}", flush=True)
