// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcStreamReadParams {
    uint32_t tokens;   // T (at most 16: at most 2 row tiles of streams; or a multiple of 32, the groups form)
    uint32_t n;        // streams per token (4)
    uint32_t iters;    // Sinkhorn iterations
    uint32_t slices;   // K slices of the hidden dimension, one core each
    float eps;         // Sinkhorn / pre epsilon
    float norm_eps;    // mHC RMS norm epsilon
    float inv_count;   // 1 / (n D)
    float rms_eps;     // the block's RMS norm epsilon
    float inv_hidden;  // 1 / D
    // the prefill's groups of 32: this call's first token in the tensors (a multiple of 32), whose tokens are the
    // streams' rows / n (the outputs' too): a long wave's tokens in calls of up to 512, each writing its rows of the
    // same outputs (passed to the later calls); 0 otherwise
    uint32_t offset = 0;
};

// streams: [1, 1, R, D] BFLOAT16 TILE (R = 4 Tp stream rows).  fn: [D, n 24] BFLOAT16 TILE (mhc_stream_project's).
// mix_consts: [13, 32, 32] FLOAT32 TILE, consts: [8, 32, 32] FLOAT32 TILE (mhc_stream_mix's).  gamma: [1, 1, 1, D]
// BFLOAT16 TILE (the block's RMS norm weight in row 0).
struct MhcStreamReadTensorArgs {
    const Tensor& streams;
    const Tensor& fn;
    const Tensor& mix_consts;
    const Tensor& consts;
    const Tensor& gamma;
    // the outputs (post, comb, the block input) to write into, an earlier call's (`offset`), or none: new ones
    std::optional<Tensor> post_out = std::nullopt;
    std::optional<Tensor> comb_out = std::nullopt;
    std::optional<Tensor> x_out = std::nullopt;
};

// mhc_stream_write_read: mhc_stream_write (streams, block_out, post, comb: the sub-layer's output and the coefficients of
// the read before it) and the next sub-layer's mhc_stream_read of the new streams (fn, mix_consts, consts, gamma) in one
// program, the new streams handed to the read's slice cores in L1 (and written to DRAM, the next write's input).
struct MhcStreamWriteReadTensorArgs {
    const Tensor& streams;
    const Tensor& block_out;
    const Tensor& post;
    const Tensor& comb;
    const Tensor& fn;
    const Tensor& mix_consts;
    const Tensor& consts;
    const Tensor& gamma;
    // the outputs (the new streams, post, comb, the block input) to write into, or none (as the read's)
    std::optional<Tensor> streams_out = std::nullopt;
    std::optional<Tensor> post_out = std::nullopt;
    std::optional<Tensor> comb_out = std::nullopt;
    std::optional<Tensor> x_out = std::nullopt;
};

}  // namespace ttnn::experimental::prim
