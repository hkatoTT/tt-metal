// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcStreamProjectParams {
    uint32_t slices;  // S: K slices of the hidden dim
};

// streams: [.., R, D] BFLOAT16 TILE; fn: [.., D, 96] BFLOAT16 TILE (each stream's 24 projection columns side by side).
struct MhcStreamProjectTensorArgs {
    const Tensor& streams;
    const Tensor& fn;
};

}  // namespace ttnn::experimental::prim
