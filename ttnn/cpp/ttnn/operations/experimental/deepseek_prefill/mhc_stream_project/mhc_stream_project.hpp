// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_project {

// The mHC read's projection with the streams packed as rows, the hidden dim in `slices` K slices on their own cores:
// -> partial [slices, R, 128] FLOAT32 TILE: columns 0-95 the slice's streams @ fn, columns 96-127 the slice's Gram
// tile S @ S^T per row tile (its diagonal: each row's sum of squares).  bf16 products are exact, the sums fp32.
//   streams: [.., R, D] BFLOAT16 TILE.  fn: [.., D, 96] BFLOAT16 TILE.  DRAM interleaved.
ttnn::Tensor mhc_stream_project(const ttnn::Tensor& streams, const ttnn::Tensor& fn, uint32_t slices);

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_project

namespace ttnn {
using operations::experimental::deepseek_prefill::mhc_stream_project::mhc_stream_project;
}  // namespace ttnn
