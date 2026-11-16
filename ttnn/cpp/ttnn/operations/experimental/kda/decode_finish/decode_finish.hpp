// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::kda {

// KDA decode after the delta rule: o rounded to bf16, then the RMS norm per (user, head) times o_norm and
// sigmoid(gate), rounded to bf16, in o_proj's layout.
//   o: [.., B H, 128] FLOAT32 TILE (recurrent_decode_step's).  fg: [.., B, 2 H 128] BFLOAT16 TILE (the gate after f).
//   o_norm: [.., 1, 128] FLOAT32 TILE.  All DRAM interleaved.
// Returns [1, 1, B, H 128] BFLOAT16 TILE.
ttnn::Tensor decode_finish(
    const ttnn::Tensor& o,
    const ttnn::Tensor& fg,
    const ttnn::Tensor& o_norm,
    uint32_t heads,
    float eps);

}  // namespace ttnn::experimental::kda
