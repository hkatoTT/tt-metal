"""The GLM-5.3-Flash text model's shape, read from the pinned ``config.json`` and checked against what this port implements.

Anything the port does not implement (RoPE in the sparse attention, grouped KV heads, shared indexer layers, grouped
routing, ...) is rejected here, so a different checkpoint fails at load time instead of computing something else.
"""

from __future__ import annotations

from dataclasses import dataclass

KDA = "kda"  # Kimi delta attention (linear attention with a per-channel forget gate)
DSA = "dsa"  # DeepSeek sparse attention: MLA with a k-pool indexer
_LAYER_TYPES = {
    "linear_attention": KDA,
    "deepseek_sparse_attention": DSA,
    "indexed_attention": DSA,
    "full_attention": DSA,
}


@dataclass(frozen=True)
class Glm53Config:
    vocab_size: int
    hidden_size: int
    num_layers: int  # the backbone, without the MTP layer
    num_mtp_layers: int
    layer_types: tuple[str, ...]  # KDA or DSA per layer
    dense_layers: tuple[int, ...]  # layers whose MLP is dense (the rest are MoE)
    rms_norm_eps: float
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    # mHC residual
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    # MLPs
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_token: int
    num_shared_experts: int
    routed_scaling_factor: float
    swiglu_limit: float
    # KDA
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel: int
    kda_gate_lower_bound: float
    # DSA (MLA without RoPE)
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    v_head_dim: int
    # DSA indexer
    index_num_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int

    @classmethod
    def from_dict(cls, config: dict) -> "Glm53Config":
        text = config.get("text_config", config)
        problems = []

        def expect(key, value):
            if text.get(key) != value:
                problems.append(f"{key}={text.get(key)!r} (expected {value!r})")

        expect("qk_rope_head_dim", 0)
        expect("mla_use_nope", True)
        expect("scoring_func", "sigmoid")
        expect("topk_method", "noaux_tc")
        expect("n_group", 1)
        expect("topk_group", 1)
        expect("norm_topk_prob", True)
        expect("hidden_act", "silu")
        expect("attention_bias", False)
        expect("mhc", True)
        expect("index_kpool_compress", True)
        expect("index_kpool_always_select_tail", True)
        expect("tie_word_embeddings", False)
        if text.get("num_key_value_heads") != text.get("num_attention_heads"):
            problems.append("grouped KV heads")
        if text.get("qk_nope_head_dim") != text.get("qk_head_dim"):
            problems.append("qk_head_dim != qk_nope_head_dim")
        n = text["num_hidden_layers"]
        layer_types = tuple(_LAYER_TYPES.get(t, t) for t in text["layer_types"])
        if len(layer_types) != n or set(layer_types) - {KDA, DSA}:
            problems.append(f"layer_types {text['layer_types']}")
        if any(t != "full" for t in text.get("indexer_types", ["full"] * n)):
            problems.append("shared indexer layers")
        linear = text["linear_attn_config"]
        if sorted(linear["kda_layers"]) != [i for i, t in enumerate(layer_types) if t == KDA]:
            problems.append("linear_attn_config.kda_layers disagrees with layer_types")
        mlp_types = text["mlp_layer_types"]
        if len(mlp_types) != n or set(mlp_types) - {"dense", "sparse"}:
            problems.append(f"mlp_layer_types {mlp_types}")
        if text["index_topk"] % text["index_kpool"]:
            problems.append("index_topk is not a multiple of index_kpool")
        if problems:
            raise ValueError("unsupported GLM-5.3-Flash config: " + "; ".join(problems))

        eos = text["eos_token_id"]
        return cls(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            num_layers=n,
            num_mtp_layers=text.get("num_nextn_predict_layers", 0),
            layer_types=layer_types,
            dense_layers=tuple(i for i, t in enumerate(mlp_types) if t == "dense"),
            rms_norm_eps=text["rms_norm_eps"],
            pad_token_id=text["pad_token_id"],
            eos_token_ids=tuple(eos) if isinstance(eos, list) else (eos,),
            hc_mult=text["hc_mult"],
            hc_sinkhorn_iters=text["hc_sinkhorn_iters"],
            hc_eps=text["hc_eps"],
            intermediate_size=text["intermediate_size"],
            moe_intermediate_size=text["moe_intermediate_size"],
            num_experts=text["n_routed_experts"],
            num_experts_per_token=text["num_experts_per_tok"],
            num_shared_experts=text["n_shared_experts"],
            routed_scaling_factor=text["routed_scaling_factor"],
            swiglu_limit=text["swiglu_limit"],
            kda_num_heads=linear["num_heads"],
            kda_head_dim=linear["head_dim"],
            kda_conv_kernel=linear["short_conv_kernel_size"],
            kda_gate_lower_bound=linear["gate_lower_bound"],
            num_heads=text["num_attention_heads"],
            q_lora_rank=text["q_lora_rank"],
            kv_lora_rank=text["kv_lora_rank"],
            qk_head_dim=text["qk_nope_head_dim"],
            v_head_dim=text["v_head_dim"],
            index_num_heads=text["index_n_heads"],
            index_head_dim=text["index_head_dim"],
            index_topk=text["index_topk"],
            index_kpool=text["index_kpool"],
        )

    def is_moe(self, layer: int) -> bool:
        return layer not in self.dense_layers

    @property
    def kda_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == KDA)

    @property
    def dsa_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == DSA)
