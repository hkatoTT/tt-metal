// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MoeDecodeRouteParams {
    uint32_t tokens;  // the real tokens (rows 0..tokens-1); the rows after them choose no expert
    uint32_t k;       // experts per token (at most 16)
    uint32_t local;   // the chip's experts, base .. base + local - 1 (at most 16); 0: every chosen expert's id
    float scale;      // the routed scaling factor
    bool shared;      // local above 16: the chips with 1 in local_base element 1 add the shared expert (id local, 1.0)
};

// logits: [.., N, E] FLOAT32 TILE (the router's).  bias: [.., 1, E] FLOAT32 TILE (e_score_correction_bias, row 0).
// local_base: [.., 1, >= 2] UINT32 ROW_MAJOR, element 0 the chip's first expert, element 1 whether it adds the shared
// expert (with `shared`).
struct MoeDecodeRouteTensorArgs {
    const Tensor& logits;
    const Tensor& bias;
    const Tensor& local_base;
};

}  // namespace ttnn::experimental::prim
