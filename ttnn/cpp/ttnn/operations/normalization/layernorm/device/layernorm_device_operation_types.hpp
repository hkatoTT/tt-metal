// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "layernorm_types.hpp"

namespace ttnn::prim {

struct LayerNormParams {
    LayerNormType norm_type = LayerNormType::LAYERNORM;
    DistributedLayerNormStage distributed_norm_stage = DistributedLayerNormStage::NOT_DISTRIBUTED;
    float eps = 0.0f;
    MemoryConfig output_mem_config;
    LayerNormProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;
    std::optional<operations::unary::UnaryWithParam> fused_activation;
    // Row-major output written into the interior of a larger, already allocated [.., T + t, H + 2h, W + 2w, C]
    // tensor (output_tensor): (t, h, w) = front T frames, H and W border widths. All zero = plain output.
    std::array<uint32_t, 3> output_pad = {0, 0, 0};
};

struct LayerNormInputs {
    Tensor input;
    std::optional<Tensor> residual_input_tensor;  // b
    std::optional<Tensor> weight;                 // gamma
    std::optional<Tensor> bias;                   // beta
    std::optional<Tensor> stats;                  // for POST_ALL_GATHER
    std::optional<Tensor> recip_tensor;           // reciprocal LUT for welford algorithm
    std::optional<Tensor> output_tensor;          // preallocated output (required with output_pad)
};

}  // namespace ttnn::prim
