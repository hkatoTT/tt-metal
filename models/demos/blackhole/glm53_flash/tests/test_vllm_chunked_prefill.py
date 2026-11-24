"""vLLM's chunked prefill through the adapter (tools/glm53_vllm.py CHUNKED, GLM53_VLLM_CHUNKED): a long prompt
prefilled in chunks (`prefill_forward` from start_pos > 0, the plugin's contract: the token history up to the chunk's
end) with DFlash2 verify steps of other seats between the chunks, against the same prompt prefilled in one call on the
same seat.  The logits after the prompt and the prompt's seat's greedy continuation (DFlash2 verify steps with that seat
alone) must be the same bit for bit.

Arms, in one process (the adapter warmed up as vLLM does): (A) the whole prompt in one call, then GLM53_CP_STEPS verify
steps of its seat alone; (B) two other seats prefilled first (GLM53_CP_OTHER tokens each), then the prompt in chunks of
GLM53_CP_CHUNK (default 8,192: one SP wave pair, as --max-num-batched-tokens and the plugin's prefill_chunk_alignment
in serving), GLM53_CP_BETWEEN verify steps of the other two after each chunk but the last, then the others end and the
prompt's seat decodes alone as in (A).  GLM53_CP_LEN (default 16,684: two full chunks and 300 tokens).  Synthetic
weights by default (GLM53_SYNTHETIC_WEIGHTS=0: real; the values only need to be the same).  Also (A2) the whole prompt again (its
run-to-run sameness) and (C) the chunks with nothing between them (no other seat).  Prints the verdict and exits 0 (a
study's run):

    GLM53_DEVICE_TEST=1 pytest .../tests/test_vllm_chunked_prefill.py -s
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
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

USERS = 4  # a row: 32 seats
SEATS = ROWS * USERS


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_vllm_chunked_prefill(mesh_device):
    os.environ.setdefault("GLM53_SYNTHETIC_WEIGHTS", "1")
    os.environ["GLM53_MTP"] = "0"
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm

    glm53_vllm.MTP, glm53_vllm.DFLASH2, glm53_vllm.SPEC, glm53_vllm.CHUNKED = False, True, True, True
    # (greedy requests only: the generator is made here without GLM53_DFLASH2_SAMPLING, which the adapter's start-up
    # sets when SAMPLING is on, its default where the plugin certifies sampled ids)
    glm53_vllm.SAMPLING = False
    synthetic = os.environ["GLM53_SYNTHETIC_WEIGHTS"] == "1"
    n = int(os.environ.get("GLM53_CP_LEN", "16684"))
    size = int(os.environ.get("GLM53_CP_CHUNK", "8192"))
    between = int(os.environ.get("GLM53_CP_BETWEEN", "3"))
    steps = int(os.environ.get("GLM53_CP_STEPS", "16"))
    other_len = int(os.environ.get("GLM53_CP_OTHER", "1000"))
    seat, others = 5, (0, 9)  # vLLM's state slots (the adapter's physical seats: other mesh rows)
    t0 = time.time()
    chunks = glm53_vllm.prefill_chunks()
    max_len = -(-(n + 8 * (steps + 2)) // max(chunks)) * max(chunks)
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
    gen = Glm53Generator(model, USERS, chunk=chunks, spec=0, dflash=(drafter, k))
    adapter = glm53_vllm.Glm53ForCausalLM(gen, max_len)
    print(f"loaded in {time.time() - t0:.0f} s; cache {gen.max_seq}, K {k}, prompt {n} in chunks of {size}", flush=True)
    t0 = time.time()
    adapter.warmup_model_prefill()
    adapter.warmup_model_decode()
    print(f"warmed up in {time.time() - t0:.0f} s", flush=True)

    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(29)
    prompt = torch.randint(0, vocab, (n,), generator=g)
    other_prompts = [torch.randint(0, vocab, (other_len,), generator=g) for _ in others]
    mode = spec_decode.ACCEPT_MODE_ARGMAX_IDS
    counts = [1] * SEATS

    def prefill(slots: list, prompts: list, begin: int, end: int):
        """One prefill call as the plugin makes it -> the host logits [rows, V] fp32 (host sampling)."""
        tokens = torch.zeros(len(slots), end, dtype=torch.int64)
        for i, p in enumerate(prompts):
            tokens[i] = p[:end]
        out = adapter.prefill_forward(
            tokens,
            torch.tensor([begin] * len(slots)),
            torch.tensor([end] * len(slots)),
            empty_slots=torch.tensor(slots),
            sampling_params=None,
        )
        return out[:, 0, :]

    def propose() -> list:
        out = adapter.propose_draft_tokens(
            k, torch.zeros(SEATS, 1, dtype=torch.int32), torch.zeros(SEATS, dtype=torch.int64), torch.tensor(counts)
        )
        return out.draft_token_ids.tolist()

    def start(active: dict, slots: list, logits: torch.Tensor, ends: list) -> None:
        drafts = None
        for s, row, end in zip(slots, logits, ends):
            counts[s] = 1
            active[s] = dict(tok=int(row.argmax()), pos=end, toks=[int(row.argmax())])
        drafts = propose()
        for s in slots:
            active[s]["draft"] = drafts[s]

    def verify(active: dict, count: int) -> None:
        for _ in range(count):
            tokens = torch.zeros(SEATS, k + 1, dtype=torch.int64)
            pos = torch.full((SEATS, k + 1), -1, dtype=torch.int64)
            valid = torch.zeros(SEATS, dtype=torch.int32)
            for s, r in active.items():
                tokens[s] = torch.tensor([r["tok"]] + r["draft"])
                pos[s] = r["pos"] + torch.arange(k + 1)
                valid[s] = k
            out = adapter.decode_forward(
                tokens,
                pos,
                spec_mode=mode,
                num_valid_drafts=valid,
                accepted_counts=torch.tensor(counts, dtype=torch.int32),
                sampling_params=None,
            )
            ids = out.argmax_ids.tolist()
            for s, r in active.items():
                a = 0
                while a < k and r["draft"][a] == ids[s][a]:
                    a += 1
                r["pos"], r["tok"] = r["pos"] + a + 1, ids[s][a]
                r["toks"].extend(ids[s][: a + 1])
                counts[s] = a + 1
            drafts = propose()
            for s, r in active.items():
                r["draft"] = drafts[s]

    def decode_alone(logits) -> list:
        """The prompt's seat alone after its prefill: its first token and `steps` verify steps -> its tokens."""
        for s in range(SEATS):
            counts[s] = 1
        run: dict = {}
        start(run, [seat], logits, [n])
        verify(run, steps)
        return run[seat]["toks"]

    def whole() -> tuple:
        t1, waves0 = time.time(), gen.counts["waves"]
        logits = prefill([seat], [prompt], 0, n)
        return logits, time.time() - t1, gen.counts["waves"] - waves0

    def chunked(count: int) -> tuple:
        """The prompt in chunks; with `count` > 0 the two others prefilled first and `count` verify steps of theirs
        after each chunk but the last."""
        for s in range(SEATS):
            counts[s] = 1
        run: dict = {}
        if count:
            start(run, list(others), prefill(list(others), other_prompts, 0, other_len), [other_len] * len(others))
        times, waves0 = [], gen.counts["waves"]
        for begin, end in zip(bounds, bounds[1:]):
            t2 = time.time()
            logits = prefill([seat], [prompt], begin, end)
            times.append(round(time.time() - t2, 3))
            if end < n and count:
                verify(run, count)
        return logits, times, gen.counts["waves"] - waves0

    bounds = list(range(0, n, size)) + [n]
    # (A) the whole prompt in one call; (A2) again (the whole prefill's run-to-run sameness)
    logits_a, t_a, waves_a = whole()
    toks_a = decode_alone(logits_a)
    logits_a2, _, _ = whole()
    toks_a2 = decode_alone(logits_a2)
    # (B) the others first, the prompt in chunks with their verify steps between; (C) the chunks with nothing between
    logits_b, t_b, waves_b = chunked(between)
    toks_b = decode_alone(logits_b)
    logits_c, t_c, _ = chunked(0)
    toks_c = decode_alone(logits_c)
    print(
        f"whole: {t_a:.3f} s, {waves_a} waves; chunks {bounds}: {t_b} s, {waves_b} waves ({between} verify steps of "
        f"the others between), {t_c} s (nothing between)",
        flush=True,
    )
    verdict = []
    for name, lg, tk in (
        ("whole again", logits_a2, toks_a2),
        (f"chunks, {between} steps between", logits_b, toks_b),
        ("chunks, nothing between", logits_c, toks_c),
    ):
        same_l, same_t = torch.equal(logits_a, lg), toks_a == tk
        diff = (logits_a - lg).abs().max().item()
        first = next((i for i, (x, y) in enumerate(zip(toks_a, tk)) if x != y), None)
        print(
            f"{name} against the whole: logits {'the same' if same_l else 'DIFFERENT'} (max |diff| {diff:.3g}); "
            f"the seat's {steps} verify steps {'the same' if same_t else f'DIFFERENT from token {first}'}",
            flush=True,
        )
        verdict.append(same_l and same_t)
    print(f"chunked prefill bits: {'PASS' if all(verdict) else 'FAIL'} {verdict}", flush=True)
