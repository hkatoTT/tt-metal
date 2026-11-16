// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcStreamWriteParams {
    uint32_t n;       // streams per token
    uint32_t tokens;  // T
};

// streams: [.., n Tp, D] BFLOAT16 TILE (row n t + i: token t's stream i; padded tokens' rows zero).
// block_out: [.., T, D] BFLOAT16 TILE.  post: [T, n], comb: [T, n n] FLOAT32 TILE (mhc_split_sinkhorn's outputs).
struct MhcStreamWriteTensorArgs {
    const Tensor& streams;
    const Tensor& block_out;
    const Tensor& post;
    const Tensor& comb;
};

}  // namespace ttnn::experimental::prim
