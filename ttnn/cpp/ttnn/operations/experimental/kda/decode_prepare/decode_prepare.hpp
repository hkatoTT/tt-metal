// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::kda {

// KDA decode before the delta rule (the width-4 conv over the history and this step's q | k | v inputs, SiLU rounded
// to bf16, the L2 norms of q and k, q scaled; the decay exp(lower sigmoid((f + dt_bias) rate)) and beta), the history
// shifted in place.  Inputs [.., B, ..] with one row per user (TILE, DRAM): proj BFLOAT16 (q | k | v, then the H beta
// logits), fg BFLOAT16 (f first), history0-2 FLOAT32 [B, 3 H 128] (oldest first), taps FLOAT32 [4, 3 H 128],
// gate_consts FLOAT32 [2, H 128] (dt_bias, rate).
// Returns {q, k, v, decay [1, 1, B H, 128], beta [1, 1, B H, 1]} FLOAT32 TILE, one row per (user, head) pair: the
// inputs of `recurrent_decode_step`.
std::array<ttnn::Tensor, 5> decode_prepare(
    const ttnn::Tensor& proj,
    const ttnn::Tensor& fg,
    const ttnn::Tensor& history0,
    const ttnn::Tensor& history1,
    const ttnn::Tensor& history2,
    const ttnn::Tensor& taps,
    const ttnn::Tensor& gate_consts,
    uint32_t heads,
    float lower,
    float q_scale,
    uint32_t users = 0,
    const std::optional<ttnn::Tensor>& keep = std::nullopt,
    const std::optional<ttnn::Tensor>& saved = std::nullopt,
    uint32_t tokens = 1,
    bool pair_major = false,
    bool l1_outputs = false,
    bool split_channels = false,
    bool row_split = false,
    const std::optional<ttnn::Tensor>& history0_out = std::nullopt,
    const std::optional<ttnn::Tensor>& history1_out = std::nullopt,
    const std::optional<ttnn::Tensor>& history2_out = std::nullopt,
    const std::optional<ttnn::Tensor>& saved_out = std::nullopt);

}  // namespace ttnn::experimental::kda
