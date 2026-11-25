"""The `tt/` CPU reference against HF Transformers' `glm5_next` modules on the real checkpoint (CPU only, no device).

Needs the reference venv (transformers with `glm5_next`) and the checkpoint (`GLM53_CHECKPOINT`).
Each test builds an HF decoder layer and the `tt/` layer from the same checkpoint tensors and runs both on one input.
The `tt/` modules follow HF operation by operation, so the expected result is bitwise equality; a test that is not
bitwise prints the size of the difference.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.glm5_next")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.config import Glm53Config  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import checkpoint_skip_reason  # noqa: E402
from models.demos.blackhole.glm53_flash.tools import hf_reference  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.dsa import DsaState  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.layer import DecoderLayer  # noqa: E402

NO_CHECKPOINT = checkpoint_skip_reason()  # (GLM53_CHECKPOINT, else HF_MODEL)
if NO_CHECKPOINT:
    pytest.skip(NO_CHECKPOINT, allow_module_level=True)


@pytest.fixture(scope="module")
def checkpoint():
    return Glm53Checkpoint()


@pytest.fixture(scope="module")
def config(checkpoint):
    return Glm53Config.from_dict(checkpoint.config)


@pytest.fixture(scope="module")
def hf_config(checkpoint):
    return hf_reference.hf_text_config(checkpoint)


def streams_input(checkpoint, config, seq: int, seed: int) -> torch.Tensor:
    """[1, seq, 4, hidden] bf16: embeddings of random tokens, the four streams perturbed apart."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, config.vocab_size, (1, seq), generator=g)
    emb = checkpoint.tensor("model.language_model.embed_tokens.weight")[ids].float()
    noise = torch.randn(1, seq, config.hc_mult, config.hidden_size, generator=g) * emb.std() * 0.5
    return (emb.unsqueeze(2) + noise).to(torch.bfloat16)


def report(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> bool:
    a, b = ours.float(), theirs.float()
    diff = (a - b).abs()
    pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    exact = torch.equal(ours, theirs)
    print(
        f"{name}: bitwise={exact} max|diff|={diff.max().item():.3e} mean|diff|={diff.mean().item():.3e} "
        f"max|ref|={b.abs().max().item():.3e} pcc={pcc:.7f}"
    )
    return exact


def run_hf_layer(module, streams):
    mask = torch.ones(streams.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        out, _ = module(streams, attention_mask=mask)
    return out


# 0: KDA + dense MLP, 3: DSA + MoE, 4: KDA + MoE, 43: the last DSA, 44: the last KDA.
@pytest.mark.parametrize("layer", [0, 3, 4, 43, 44])
@pytest.mark.parametrize("seq", [1, 3, 24, 37, 130])
def test_layer_prefill_matches_hf(checkpoint, config, hf_config, layer, seq):
    streams = streams_input(checkpoint, config, seq, seed=layer * 1000 + seq)
    theirs = run_hf_layer(hf_reference.hf_layer(checkpoint, layer, hf_config), streams)
    with torch.no_grad():
        ours, _, _ = DecoderLayer(checkpoint, config, layer)(streams)
    exact = report(f"layer {layer} seq {seq}", ours, theirs)
    a, b = ours.float().flatten(), theirs.float().flatten()
    assert exact or torch.corrcoef(torch.stack([a, b]))[0, 1] > 0.99999


@pytest.mark.parametrize("layer", [4, 3])
def test_layer_decode_continues_prefill(checkpoint, config, layer):
    """Prefill of S tokens equals prefill of S - 3 followed by 3 one-token steps (KDA: chunked vs recurrent form)."""
    streams = streams_input(checkpoint, config, 70, seed=7)
    tt_layer = DecoderLayer(checkpoint, config, layer)
    with torch.no_grad():
        whole, _, _ = tt_layer(streams)
        part, state, _ = tt_layer(streams[:, :67])
        steps = [part]
        for i in range(67, 70):
            out, state, _ = tt_layer(streams[:, i : i + 1], state)
            steps.append(out)
    stepped = torch.cat(steps, dim=1)
    report(f"layer {layer} prefill 70 vs 67 + 3 steps", stepped, whole)
    a, b = stepped.float().flatten(), whole.float().flatten()
    assert torch.corrcoef(torch.stack([a, b]))[0, 1] > 0.9999


def test_indexer_keeps_every_token_within_budget(checkpoint, config):
    """With at most index_topk / kpool complete pools visible, DSA selects every causal token (plain attention)."""
    from models.demos.blackhole.glm53_flash.tt.dsa import Dsa, DsaWeights

    dsa = Dsa(DsaWeights.from_checkpoint(checkpoint, 3), config)
    seq = config.index_topk + config.index_kpool - 1  # 2051
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(1, seq, config.hidden_size, generator=g) * 0.05).to(torch.bfloat16)
    index_key = torch.randn(1, seq, config.index_head_dim, generator=g).to(torch.bfloat16)
    index_gate = torch.randn(1, seq, config.index_head_dim, generator=g).to(torch.bfloat16)
    q_resid = torch.randn(1, seq, config.q_lora_rank, generator=g).to(torch.bfloat16)
    tokens = dsa.indexer(x, q_resid, index_key, index_gate, start=0).tokens[0]
    for t in [0, 1, 3, 4, 1000, seq - 1]:
        chosen = sorted(int(i) for i in tokens[t] if i >= 0)
        assert chosen == list(range(t + 1)), t


@pytest.mark.parametrize("layer", [3])
def test_dsa_sparse_selection_matches_hf(checkpoint, config, hf_config, layer):
    """Past the budget (2,600 tokens: 650 pools for 512 slots) the selection is really sparse; still equal to HF."""
    streams = streams_input(checkpoint, config, 2600, seed=11)
    theirs = run_hf_layer(hf_reference.hf_layer(checkpoint, layer, hf_config), streams)
    with torch.no_grad():
        ours, state, aux = DecoderLayer(checkpoint, config, layer)(streams)
    assert isinstance(state, DsaState)
    selected = (aux.selection.tokens[0, -1] >= 0).sum().item()
    print(f"last query keeps {selected} of 2600 tokens")
    assert selected < 2600
    exact = report(f"layer {layer} seq 2600 (sparse)", ours, theirs)
    a, b = ours.float().flatten(), theirs.float().flatten()
    assert exact or torch.corrcoef(torch.stack([a, b]))[0, 1] > 0.99999
