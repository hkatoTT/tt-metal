// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DecodeFinishParams {
    uint32_t heads;  // H per chip
    float eps;       // the RMS norm's epsilon
};

// o: [.., B H, 128] FLOAT32 TILE (recurrent_decode_step's output, one row per (user, head) pair).
// fg: [.., B, 2 H 128] BFLOAT16 TILE (the gate in columns H 128 ..).  o_norm: [.., 1, 128] FLOAT32 TILE.
struct DecodeFinishTensorArgs {
    const Tensor& o;
    const Tensor& fg;
    const Tensor& o_norm;
};

}  // namespace ttnn::experimental::prim
