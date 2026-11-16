// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_write {

// The mHC write with the streams packed as rows (row n t + i: token t's stream i, n = 4, tokens padded to Tp, a
// multiple of 8, with zero rows):  new stream (t, j) = bf16(sum_i comb_t[i][j] stream (t, i)) + bf16(post_t[j]
// block_out_t), comb and post rounded to bf16 first, each product summed in fp32 and rounded once.
//   streams: [.., n Tp, D] BFLOAT16 TILE.  block_out: [.., T, D] BFLOAT16 TILE.
//   post: [T, n], comb: [T, n n] FLOAT32 TILE (mhc_split_sinkhorn's outputs).  All DRAM interleaved.
// Returns the new streams [.., n Tp, D] BFLOAT16 TILE (the padded tokens' rows zero).
ttnn::Tensor mhc_stream_write(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& block_out,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    uint32_t n);

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_write

namespace ttnn {
using operations::experimental::deepseek_prefill::mhc_stream_write::mhc_stream_write;
}  // namespace ttnn
