// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct SdpaDecodeParams {
    bool is_causal = false;
    bool paged_attention = false;
    std::vector<uint32_t> cur_pos;
    std::optional<float> scale = std::nullopt;
    std::optional<uint32_t> sliding_window_size = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config = std::nullopt;
    DeviceComputeKernelConfig compute_kernel_config;
    uint32_t k_chunk_size = 0;
    // Share cache is only meaningful for some unpaged configurations; default is false.
    std::optional<bool> share_cache = std::nullopt;
    // When true, enables multi-latent attention (MLA) path where V is derived from K.
    std::optional<bool> use_mla = std::nullopt;
    std::optional<uint32_t> head_dim_v = std::nullopt;
    // Paired paged-cache geometry overrides (shared with chunked prefill SDPA). See
    // ttnn::operations::transformer::PagedCacheGeometryOverride.
    ttnn::operations::transformer::PagedCacheGeometryOverride paged_cache_geometry{};
    // Optional circular-buffer capacity (in tokens) for the K/V cache view. When set,
    // the kernel computes ``virtual_pos %= cache_position_modulo`` before resolving the
    // page_table entry for every tile read, so a bounded sliding-window cache of
    // capacity N can be addressed by absolute positions ≥ N. Required for layers using
    // vLLM's SlidingWindowSpec, which sizes the per-layer page_table to
    // sliding_window/block_size entries and zero-pads the rest. Must be a multiple of
    // the effective block_size and ≥ sliding_window_size when both are set. Paged-mode
    // only (validated in validate_on_program_cache_miss).
    std::optional<uint32_t> cache_position_modulo = std::nullopt;
    // Speculative multi-position mode. 0 = off (legacy behaviour, byte-for-byte).
    // When set to Tg (candidates per batch GROUP), the op folds Tg candidates that share one
    // KV cache onto a single batch row, so each row scans its KV once instead of Tg times:
    // Q is [1, B, Tg*32, DH] (batch b, row-tile j = candidate b*Tg + j), the page_table has
    // B rows, and cur_pos_tensor holds B*Tg positions, ascending within each group of Tg —
    // candidate b*Tg+j attends to [0, cur_pos[b*Tg+j]] inclusive, exactly like batch row
    // b*Tg+j with that cur_pos in the legacy B*Tg-row form. B groups let the op fill a grid
    // that one group's row-tile count cannot (L1 caps cores/head by Tg).
    // Requires num_kv_heads==1, causal, paged, no sliding window, no MLA, unsharded bf16
    // TILE Q with Tg*32 padded rows per batch.
    uint32_t spec_multi_pos_tiles = 0;
    // Heads-first Q and output (opt-in; false = the legacy layouts, byte-for-byte). MLA paged decode only: Q is
    // [1, NH, B, D] (each head's B tokens as rows: the layout of a per-head matmul's output) in L1, the output
    // [1, NH, B, head_dim_v] interleaved, so the caller needs no transposes around the op. The reader puts each
    // token's Q tile (heads as rows, what the compute takes) together from 32-byte face rows and the writer puts the
    // output rows back the same way: the compute and its values are unchanged. See dataflow/heads_first.hpp.
    bool heads_first = false;
    // Of the NH heads, how many the reader (NoC 0) moves; the writer (NoC 1) takes the rest: Q's rows at the start,
    // the output's at the end (heads_first only; TT_SDPA_HF_Q_READER / TT_SDPA_HF_OUT_READER, UINT32_MAX: the
    // program factory's defaults, 0 and NH / 2).
    uint32_t heads_first_q_reader = 0;
    uint32_t heads_first_out_reader = 0;
    // The face rows issued with the NoC command buffer's state and counted once (else one plain call a row; heads_first
    // only; TT_SDPA_HF_LEAN=0 for the plain form).
    bool heads_first_lean = true;
};

struct SdpaDecodeInputs {
    // Mandatory tensors
    Tensor q;
    Tensor k;

    // Optional V tensor; when MLA is enabled, V is derived from K and this may be nullopt.
    std::optional<Tensor> v;

    // Optional auxiliary tensors
    std::optional<Tensor> cur_pos_tensor;
    std::optional<Tensor> page_table_tensor;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> attention_sink;
};

}  // namespace ttnn::prim
