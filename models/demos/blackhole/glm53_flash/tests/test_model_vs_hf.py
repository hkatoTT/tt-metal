"""The first layers of the `tt/` text model against HF's own `from_pretrained` model cut to the same layers (CPU only).

Covers what the per-layer tests do not: the embedding, the stream expansion and collapse, the final norm, and decode
with HF's cache (the KDA conv and recurrent states, the MLA latent cache, the indexer cache) against the `tt/` states.
Four layers hold every kind: KDA + dense MLP (0-2) and DSA + MoE (3).
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("transformers.models.glm5_next")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, checkpoint_root  # noqa: E402
from models.demos.blackhole.glm53_flash.tools import hf_reference  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.model import Glm53TextModel  # noqa: E402

if not os.environ.get("GLM53_CHECKPOINT") or not (checkpoint_root() / "model.safetensors.index.json").exists():
    pytest.skip("no checkpoint (GLM53_CHECKPOINT)", allow_module_level=True)

NUM_LAYERS = 4


@pytest.fixture(scope="module")
def checkpoint():
    return Glm53Checkpoint()


@pytest.fixture(scope="module")
def hf_model(checkpoint):
    return hf_reference.hf_truncated_model(checkpoint, NUM_LAYERS)


@pytest.fixture(scope="module")
def tt_model(checkpoint):
    return Glm53TextModel(checkpoint, num_layers=NUM_LAYERS)


def report(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> bool:
    a, b = ours.float(), theirs.float()
    diff = (a - b).abs()
    exact = torch.equal(ours, theirs)
    print(f"{name}: bitwise={exact} max|diff|={diff.max().item():.3e} max|ref|={b.abs().max().item():.3e}")
    return exact


@pytest.mark.parametrize("prefill, steps", [(35, 6), (2600, 3)])
def test_prefill_then_decode_matches_hf(checkpoint, hf_model, tt_model, prefill, steps):
    """Final hidden states after a prefill, then after each one-token step with the cache, equal HF's."""
    ids = torch.randint(
        0, tt_model.config.vocab_size, (1, prefill + steps), generator=torch.Generator().manual_seed(prefill)
    )
    text = hf_model.model.language_model
    with torch.no_grad():
        out = text(input_ids=ids[:, :prefill], use_cache=True)
        cache = out.past_key_values
        theirs = [out.last_hidden_state]
        for i in range(prefill, prefill + steps):
            out = text(input_ids=ids[:, i : i + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            theirs.append(out.last_hidden_state)

        hidden, state, _ = tt_model.hidden(ids[:, :prefill])
        ours = [hidden]
        for i in range(prefill, prefill + steps):
            hidden, state, _ = tt_model.hidden(ids[:, i : i + 1], state)
            ours.append(hidden)

    all_exact = True
    for step, (a, b) in enumerate(zip(ours, theirs)):
        all_exact &= report(f"prefill {prefill} step {step}", a, b)
        assert a.shape == b.shape
        assert torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1] > 0.99999
    print("all bitwise" if all_exact else "not all bitwise")


def test_hf_grouped_mm_experts_differ_only_by_rounding(checkpoint, hf_model):
    """HF's default `grouped_mm` experts against its `eager` loop on the same 4 layers: the spread inside HF itself."""
    grouped = hf_reference.hf_truncated_model(checkpoint, NUM_LAYERS, experts_implementation="grouped_mm")
    ids = torch.randint(0, 154880, (1, 130), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        a = grouped.model.language_model(input_ids=ids).last_hidden_state
        b = hf_model.model.language_model(input_ids=ids).last_hidden_state
    report("grouped_mm vs eager (130 tokens)", a, b)
    assert torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1] > 0.9999


@pytest.mark.parametrize("chunks", [(20, 15), (3, 1, 30)])
def test_chunked_prefill_then_decode_matches_hf(checkpoint, hf_model, tt_model, chunks):
    """A prompt fed in several pieces (a continued prefill, as a chunked prefill or a follow-up turn does), then 3 steps."""
    total = sum(chunks) + 3
    ids = torch.randint(0, tt_model.config.vocab_size, (1, total), generator=torch.Generator().manual_seed(total))
    pieces, start = [], 0
    for n in list(chunks) + [1, 1, 1]:
        pieces.append((start, start + n))
        start += n
    text = hf_model.model.language_model
    theirs, ours = [], []
    with torch.no_grad():
        cache = None
        for a, b in pieces:
            out = text(input_ids=ids[:, a:b], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            theirs.append(out.last_hidden_state)
        state = None
        for a, b in pieces:
            hidden, state, _ = tt_model.hidden(ids[:, a:b], state)
            ours.append(hidden)
    all_exact = True
    for (a, b), x, y in zip(pieces, ours, theirs):
        all_exact &= report(f"tokens {a}..{b - 1}", x, y)
        assert torch.corrcoef(torch.stack([x.float().flatten(), y.float().flatten()]))[0, 1] > 0.99999
    print("all bitwise" if all_exact else "not all bitwise")
