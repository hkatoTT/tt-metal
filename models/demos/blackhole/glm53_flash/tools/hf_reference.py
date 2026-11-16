"""Hugging Face Transformers (`glm5_next`) as the CPU reference, in pieces that fit in host memory.

The full bf16 model (643 GB) does not fit in a host of ~0.5 TB, so the HF modules are built one decoder layer at a time:
`hf_layer` makes an HF `Glm5NextTextDecoderLayer` and loads that layer's checkpoint tensors into it, renamed and merged
as HF's own loader does (`state_dict_for_layer`), in the dtypes HF keeps (bf16, except the modules HF pins to fp32).
`hf_truncated_model` loads the first N layers with HF's own `from_pretrained`; `compare-loader` checks that the two
loading paths give identical parameters, so the per-layer path is HF's model and not my reading of it.

Needs a reference venv whose transformers has `glm5_next` (tt/ was checked against 5.20.2), not the tt-metal venv.

    python -m models.demos.blackhole.glm53_flash.tools.hf_reference compare-loader --layers 4
    python -m models.demos.blackhole.glm53_flash.tools.hf_reference compare-full
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint
from models.demos.blackhole.glm53_flash.tt.common import LAYER_PREFIX

# HF keeps these in fp32 whatever the model dtype (`_keep_in_fp32_modules_strict`).
FP32_MODULES = ("e_score_correction_bias", "conv1d", "dt_bias", "A_log")
_RENAMES = {
    "self_attn.f_a_proj.weight": "self_attn.forget_gate.f_a_proj.weight",
    "self_attn.f_b_proj.weight": "self_attn.forget_gate.f_b_proj.weight",
    "self_attn.dt_bias": "self_attn.forget_gate.dt_bias",
    "self_attn.A_log": "self_attn.forget_gate.A_log",
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_scale": "ffn_hc.scale",
}


def text_config_dict(checkpoint: Glm53Checkpoint, num_layers: int | None = None) -> dict:
    """The full config.json, cut to the first `num_layers` layers when given."""
    config = copy.deepcopy(checkpoint.config)
    if num_layers is not None:
        text = config["text_config"]
        text["num_hidden_layers"] = num_layers
        for key in ("layer_types", "mlp_layer_types", "indexer_types"):
            text[key] = text[key][:num_layers]
        linear = text["linear_attn_config"]
        linear["kda_layers"] = [i for i in linear["kda_layers"] if i < num_layers]
        linear["full_attn_layers"] = [i for i in linear["full_attn_layers"] if i < num_layers]
    return config


def hf_text_config(
    checkpoint: Glm53Checkpoint,
    num_layers: int | None = None,
    attn_implementation: str = "sdpa",
    experts_implementation: str = "eager",
):
    from transformers import Glm5NextConfig

    config = Glm5NextConfig.from_dict(text_config_dict(checkpoint, num_layers))
    text = config.text_config
    text._attn_implementation = attn_implementation
    text._experts_implementation = experts_implementation
    return text


def _runtime_dtype(name: str) -> torch.dtype:
    return torch.float32 if any(m in name for m in FP32_MODULES) else torch.bfloat16


def state_dict_for_layer(checkpoint: Glm53Checkpoint, layer: int) -> dict[str, torch.Tensor]:
    """Layer `layer`'s tensors under HF's module names, merged and cast the way HF's loader does."""
    prefix = LAYER_PREFIX.format(layer)
    raw = {name[len(prefix) :]: checkpoint.tensor(name) for name in checkpoint.names(prefix)}
    out = {}
    experts: dict[int, dict[str, torch.Tensor]] = {}
    convs = {}
    for name, tensor in raw.items():
        if name.startswith("mlp.experts."):
            _, _, e, proj, _ = name.split(".")
            experts.setdefault(int(e), {})[proj] = tensor
        elif name.endswith("_conv1d.weight"):
            convs[name.split(".")[1][0]] = tensor
        else:
            out[_RENAMES.get(name, name)] = tensor
    if convs:
        out["self_attn.conv1d.weight"] = torch.cat([convs["q"], convs["k"], convs["v"]], dim=0)
    if experts:
        order = sorted(experts)
        if order != list(range(len(order))):
            raise ValueError(f"layer {layer}: experts are not 0..{len(order) - 1}")
        out["mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([experts[e]["gate_proj"], experts[e]["up_proj"]], dim=0) for e in order]
        )
        out["mlp.experts.down_proj"] = torch.stack([experts[e]["down_proj"] for e in order])
    return {name: t.to(_runtime_dtype(name)) for name, t in out.items()}


def hf_layer(checkpoint: Glm53Checkpoint, layer: int, text_config=None):
    """An HF decoder layer for `layer` with its checkpoint weights (eval mode, no grad needed)."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    text_config = text_config or hf_text_config(checkpoint)
    with torch.device("meta"):
        module = Glm5NextTextDecoderLayer(text_config, layer)
    state = state_dict_for_layer(checkpoint, layer)
    expected = {name for name, _ in module.named_parameters()} | {name for name, _ in module.named_buffers()}
    if set(state) != expected:
        raise ValueError(
            f"layer {layer}: checkpoint gives {sorted(set(state) - expected)} extra and {sorted(expected - set(state))} missing"
        )
    module.load_state_dict(state, assign=True, strict=True)
    return module.eval()


def hf_truncated_model(
    checkpoint: Glm53Checkpoint,
    num_layers: int,
    attn_implementation: str = "sdpa",
    experts_implementation: str = "eager",
):
    """HF's `from_pretrained` on a config cut to `num_layers` layers (the rest of the checkpoint is skipped).

    The experts run HF's `eager` loop (each expert in turn, the weighted outputs added per token in bf16 by ascending
    expert id), the reference definition `tt/` follows.  `from_pretrained` would otherwise pick `grouped_mm`, which
    adds the same terms in another order (rounding-level differences).
    """
    from transformers import Glm5NextForConditionalGeneration

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "config.json").write_text(json.dumps(text_config_dict(checkpoint, num_layers)))
        for name in ["model.safetensors.index.json", "generation_config.json"]:
            (tmp / name).symlink_to(checkpoint.root / name)
        for shard in sorted(set(checkpoint.weight_map.values())):
            (tmp / shard).symlink_to(checkpoint.root / shard)
        model = Glm5NextForConditionalGeneration.from_pretrained(
            tmp,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            experts_implementation=experts_implementation,
        )
    return model.eval()


def compare_loader(checkpoint: Glm53Checkpoint, num_layers: int) -> int:
    """Check `hf_layer` against HF's own loader on the first `num_layers` layers; print every difference."""
    model = hf_truncated_model(checkpoint, num_layers)
    problems = 0
    for layer in range(num_layers):
        ours = dict(hf_layer(checkpoint, layer).state_dict())
        theirs = dict(model.model.language_model.layers[layer].state_dict())
        if set(ours) != set(theirs):
            print(f"layer {layer}: names differ: {sorted(set(ours) ^ set(theirs))}")
            problems += 1
            continue
        for name in sorted(theirs):
            a, b = ours[name], theirs[name]
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                print(
                    f"layer {layer} {name}: ours {a.dtype} {tuple(a.shape)} vs HF {b.dtype} {tuple(b.shape)}"
                    f" equal={a.shape == b.shape and torch.equal(a.float(), b.float())}"
                )
                problems += 1
        print(f"layer {layer}: {len(theirs)} tensors compared", flush=True)
    print("loader check:", "OK" if problems == 0 else f"{problems} problems")
    return 1 if problems else 0


def hf_streamed_hidden(checkpoint: Glm53Checkpoint, input_ids: torch.Tensor, num_layers: int | None = None):
    """HF's text-model forward without a cache, holding one decoder layer at a time: the final normed hidden states.

    The loop is HF's `Glm5NextTextModel.forward` (embedding, 4 identical streams, the layers with the top-k handoff,
    the stream mean, the final norm), checked against HF's own model in tests/test_model_vs_hf.py.
    """
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead, Glm5NextTextRMSNorm

    config = hf_text_config(checkpoint)
    n = config.num_hidden_layers if num_layers is None else num_layers
    embed = torch.nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id, device="meta")
    embed.load_state_dict({"weight": checkpoint.tensor("model.language_model.embed_tokens.weight")}, assign=True)
    norm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(torch.bfloat16)
    norm.load_state_dict({"weight": checkpoint.tensor("model.language_model.norm.weight")})
    mask = torch.ones(input_ids.shape, dtype=torch.bool)
    with torch.no_grad():
        streams = embed(input_ids).unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
        topk = None
        for i in range(n):
            layer = hf_layer(checkpoint, i, config)
            streams, topk = layer(streams, attention_mask=mask, prev_topk_indices=topk)
            del layer
            print(f"hf layer {i} done", flush=True)
        return norm(Glm5NextTextHyperHead()(streams))


def compare_full(checkpoint: Glm53Checkpoint, prompt: str, num_layers: int | None) -> int:
    """The whole text model: tt/ against the streamed HF forward on one chat prompt; prints the top next tokens."""
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from models.demos.blackhole.glm53_flash.tt.model import Glm53TextModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint.root)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"]
    print(f"{ids.shape[1]} prompt tokens: {tokenizer.decode(ids[0])!r}", flush=True)
    model = Glm53TextModel(checkpoint, num_layers=num_layers)
    with torch.no_grad():
        ours, _, _ = model.hidden(ids)
    print("tt/ done", flush=True)
    theirs = hf_streamed_hidden(checkpoint, ids, num_layers)
    exact = torch.equal(ours, theirs)
    diff = (ours.float() - theirs.float()).abs().max().item()
    print(f"final hidden: bitwise={exact} max|diff|={diff:.3e}")
    logits = F.linear(ours[0, -1].float(), checkpoint.tensor("lm_head.weight").float())
    top = logits.topk(5)
    print(
        "next token top 5:", [(tokenizer.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)]
    )
    return 0 if exact else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    loader = sub.add_parser("compare-loader")
    loader.add_argument("--layers", type=int, default=4)
    full = sub.add_parser("compare-full")
    full.add_argument("--prompt", default="What is the capital of France? Answer in one word.")
    full.add_argument("--layers", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    checkpoint = Glm53Checkpoint(args.checkpoint)
    if args.command == "compare-loader":
        return compare_loader(checkpoint, args.layers)
    if args.command == "compare-full":
        return compare_full(checkpoint, args.prompt, args.layers)
    return 2


if __name__ == "__main__":
    sys.exit(main())
