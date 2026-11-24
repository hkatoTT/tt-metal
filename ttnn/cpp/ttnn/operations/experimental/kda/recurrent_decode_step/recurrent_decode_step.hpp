// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::kda {

// One KDA (Kimi delta attention) decode step for P (user, head) pairs, the delta rule on each pair's
// [128 x 128] state (kept in bf16, its update in fp32):
//   S <- S * decay (per key channel);  S <- S + k^T ((v - k S) beta);  o = q S
//   state: [.., 128, 128] BFLOAT16 TILE in DRAM, P states, updated in place.
//   q, k, v, decay: [.., P, 128] FLOAT32 TILE in DRAM, one row per pair (q and k L2-normalized, q scaled by
//   1/sqrt(128), decay = exp(g)).  beta: [.., P, 1] FLOAT32 TILE in DRAM.
// Returns o [.., P, 128] FLOAT32 TILE.  The state's decay and update are exact fp32 in Dest, the state rounded to bf16
// once a step; k S and q S go through the matrix engine.
// With `saved` (and `keep`, `heads`, `tokens`): the chunked verify form of a verify step of `tokens` (2 - 8) tokens a
// pair (RecurrentDecodeStepTensorArgs).
ttnn::Tensor recurrent_decode_step(
    const ttnn::Tensor& state,
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& decay,
    const ttnn::Tensor& beta,
    const std::optional<ttnn::Tensor>& source = std::nullopt,
    const std::optional<ttnn::Tensor>& keep = std::nullopt,
    uint32_t heads = 0,
    const std::optional<ttnn::Tensor>& output = std::nullopt,
    uint32_t form = 0,
    uint32_t tokens = 0,
    const std::optional<ttnn::Tensor>& saved = std::nullopt,
    bool pair_major = false,
    const std::optional<ttnn::Tensor>& saved_source = std::nullopt);

}  // namespace ttnn::experimental::kda
