// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_experts {

// The decode MoE's routed experts on one chip for 1-16 tile rows of tokens (every token of the mesh column): y = sum
// over the chip's experts j of weights[:, j] * (h_j @ down_j), h_j = SwiGLU(x @ gate_j, x @ up_j) clamped at `limit`,
// each chip I = 256 columns of every expert of its mesh column.
//   x: [.., N, D] BFLOAT16 TILE.  weights: [.., N, 32] FLOAT32 TILE (moe_decode_route's, the column's range).
//   w1, w2: the experts in the cores' layout (see MoeDecodeExpertsTensorArgs).
// Returns [1, 1, N, D] BFLOAT16 TILE (row_tokens > 0, 8 or 16:
// y by mesh row, each row_tokens rows at the top of their own tile row, the rest zero: [1, 1, N / row_tokens 32, D]).
// The in-op gather (h1_sem_x, h1_sem_r != 0, x_row): x and weights are the column's landing buffers, which the op fills
// over the fabric from each mesh row's x_row and routing rows (MoeDecodeExpertsTensorArgs).  With it, rs_sem and
// rs_mail != 0 (32 / 64 tokens a mesh row): the column's way back in the op, [1, 1, t, D] (MoeDecodeExpertsParams),
// rs_dtype's (BFLOAT16 by default; BFLOAT8_B: GLM53_AR_BFP8, the row all-reduce's input share in BFP8_b).
ttnn::Tensor moe_decode_experts(
    const ttnn::Tensor& x,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& w1,
    const ttnn::Tensor& w2,
    uint32_t local,
    uint32_t inter,
    float limit,
    uint32_t row_tokens = 0,
    const std::optional<ttnn::Tensor>& x_row = std::nullopt,
    uint32_t h1_sem_x = 0,
    uint32_t h1_sem_r = 0,
    uint32_t rs_sem = 0,
    uint32_t rs_mail = 0,
    const std::optional<tt::tt_metal::MemoryConfig>& rs_memory_config = std::nullopt,
    const std::vector<uint32_t>& w1_recv = {},
    const std::optional<tt::tt_metal::DataType>& rs_dtype = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_experts

namespace ttnn {
using operations::experimental::deepseek_prefill::moe_decode_experts::moe_decode_experts;
}  // namespace ttnn
