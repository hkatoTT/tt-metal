// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_mix {

// The mHC read after mhc_stream_project, for at most 32 tokens: the slices' partials summed, each token's mixes and
// RMS scale, then mhc_split_sinkhorn's pre / post / comb, and the block input's weights P [T, R] (each token's pre on
// its own n stream rows; the block input is P @ streams).
//   partial: [S, R, 128] FLOAT32 TILE.  mix_consts: [13, 32, 32] FLOAT32 TILE.  consts: [8, 32, 32] FLOAT32 TILE.
// Returns {post [T, n], comb [T, n n], P [T, R]} FLOAT32 TILE.
std::array<ttnn::Tensor, 3> mhc_stream_mix(
    const ttnn::Tensor& partial,
    const ttnn::Tensor& mix_consts,
    const ttnn::Tensor& consts,
    uint32_t tokens,
    uint32_t n,
    uint32_t sinkhorn_iters,
    float eps,
    float norm_eps,
    uint32_t hidden);

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_mix

namespace ttnn {
using operations::experimental::deepseek_prefill::mhc_stream_mix::mhc_stream_mix;
}  // namespace ttnn
