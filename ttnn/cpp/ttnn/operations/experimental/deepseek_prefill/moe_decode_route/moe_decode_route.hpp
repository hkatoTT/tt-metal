// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_route {

// Sigmoid routing with a selection bias for the chip's experts: per token (row), the top k of sigmoid(logits) + bias
// (ties to the lower expert), and for each chosen expert e among base .. base + local - 1 the weight
// bf16(sigmoid(logit_e) / (sum of the k chosen sigmoids + 1e-20) * scale) at column e - base; other columns zero.
// With local 0, every chosen expert's id (columns 0 .. k - 1) and weight (16 .. 16 + k - 1); with local above 16, the
// same for the chosen experts among base .. base + local - 1 only, their ids as e - base, and with `shared` on the chips
// whose local_base element 1 is 1, then the shared expert as id local with weight 1.0.  Rows tokens .. N - 1 choose
// nothing.
//   logits: [.., N, E] FLOAT32 TILE.  bias: [.., 1, E] FLOAT32 TILE.  local_base: [.., 1, >= 1] UINT32 ROW_MAJOR.
// Returns [1, 1, N, 32] FLOAT32 TILE.
ttnn::Tensor moe_decode_route(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& bias,
    const ttnn::Tensor& local_base,
    uint32_t tokens,
    uint32_t k,
    uint32_t local,
    float scale,
    bool shared = false);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_route

namespace ttnn {
using operations::experimental::deepseek_prefill::moe_decode_route::moe_decode_route;
}  // namespace ttnn
