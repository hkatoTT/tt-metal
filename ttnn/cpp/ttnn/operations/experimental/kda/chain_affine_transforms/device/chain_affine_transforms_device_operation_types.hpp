// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct ChainAffineTransformsParams {
    uint32_t steps;
    uint32_t batch_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChainAffineTransformsInputs {
    Tensor transforms;
    Tensor initial_state;
    Tensor actual_start;
    // Optional, together: per device, rank_mask[r] (UINT32 row-major, one word per rank) 0 makes rank r's step read
    // `identity` ([1, 1, K, K + V] BF16 tile: [I | 0]) instead of its transition.
    std::optional<Tensor> rank_mask;
    std::optional<Tensor> identity;
    // Optional, per device: initial_keep's first element (a FLOAT32 TILE tensor, 0 or 1) 0 makes the initial state zero
    // (not read), as initial_state x 0 (but +0 for -0), 1 reads it as it is (as initial_state x 1): the caller's
    // multiply by a 0 / 1 keep folded into the op's read.
    std::optional<Tensor> initial_keep;
};

}  // namespace ttnn::experimental::prim
