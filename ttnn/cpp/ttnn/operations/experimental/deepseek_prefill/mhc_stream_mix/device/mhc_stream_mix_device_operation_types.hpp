// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcStreamMixParams {
    uint32_t tokens;   // T (at most 32)
    uint32_t n;        // streams per token (4)
    uint32_t iters;    // Sinkhorn iterations
    float eps;         // Sinkhorn / pre epsilon
    float norm_eps;    // RMS norm epsilon
    float inv_count;   // 1 / (n D)
};

// partial: [S, R, 128] FLOAT32 TILE (mhc_stream_project).  mix_consts: [13, 32, 32] FLOAT32 TILE (own-block masks,
// block sums, identity, ones, the pre repeat, the token sums per row tile).  consts: [8, 32, 32] FLOAT32 TILE
// (mhc_split_sinkhorn's).
struct MhcStreamMixTensorArgs {
    const Tensor& partial;
    const Tensor& mix_consts;
    const Tensor& consts;
};

}  // namespace ttnn::experimental::prim
