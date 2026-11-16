// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_read {

// The whole mHC read and the block's RMS norm in one op, for at most 16 tokens: mhc_stream_project on `slices` cores
// (each a K slice of every row tile, kept in L1), mhc_stream_mix on one core (the partials sent to it over the NoC),
// then each slice's columns of the block input P @ streams (rounded to bf16), its rows' sums of squares summed on the
// mix core, and the normed input x * rsqrt(mean + rms_eps) * gamma.
//   streams: [1, 1, R, D] BFLOAT16 TILE (R <= 64).  fn: [D, n 24] BFLOAT16 TILE.  mix_consts: [13, 32, 32], consts:
//   [8, 32, 32] FLOAT32 TILE.  gamma: [1, 1, 1, D] BFLOAT16 TILE.
// Returns {post [T, n], comb [T, n n]} FLOAT32 TILE and the normed block input [1, 1, T, D] BFLOAT16 TILE.
// The prefill's groups form (R = n T, more than 64 rows): `offset` (a multiple of 32) the call's first token,
// `tokens` its tokens, of the streams' R / n, the outputs' rows (a long wave in calls of up to 512 tokens);
// `outputs`: an earlier call's to write into (required from offset 1 on), else new ones of R / n tokens.
std::array<ttnn::Tensor, 3> mhc_stream_read(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& fn,
    const ttnn::Tensor& mix_consts,
    const ttnn::Tensor& consts,
    const ttnn::Tensor& gamma,
    uint32_t tokens,
    uint32_t n,
    uint32_t sinkhorn_iters,
    uint32_t slices,
    float eps,
    float norm_eps,
    float rms_eps,
    uint32_t offset = 0,
    const std::optional<std::array<ttnn::Tensor, 3>>& outputs = std::nullopt);

// mhc_stream_write (streams, block_out, post, comb: the sub-layer's output and the read before it's coefficients) and
// the next sub-layer's mhc_stream_read of the new streams in one program (decode: at most 16 tokens, or 32 in the groups
// of 8): the write's kernels on the cores the read leaves free, each new streams tile handed to the read's slice core
// in L1 as well as written to DRAM.  The same values as the two ops.  Returns {new streams, post, comb, normed input}.
std::array<ttnn::Tensor, 4> mhc_stream_write_read(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& block_out,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    const ttnn::Tensor& fn,
    const ttnn::Tensor& mix_consts,
    const ttnn::Tensor& consts,
    const ttnn::Tensor& gamma,
    uint32_t tokens,
    uint32_t n,
    uint32_t sinkhorn_iters,
    uint32_t slices,
    float eps,
    float norm_eps,
    float rms_eps,
    uint32_t offset = 0,
    const std::optional<std::array<ttnn::Tensor, 4>>& outputs = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_read

namespace ttnn {
using operations::experimental::deepseek_prefill::mhc_stream_read::mhc_stream_read;
using operations::experimental::deepseek_prefill::mhc_stream_read::mhc_stream_write_read;
}  // namespace ttnn
