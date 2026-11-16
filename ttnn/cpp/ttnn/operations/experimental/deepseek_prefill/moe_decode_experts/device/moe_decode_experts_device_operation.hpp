// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "moe_decode_experts_program_factory.hpp"
#include "moe_decode_experts_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MoeDecodeExpertsDeviceOperation {
    using operation_attributes_t = MoeDecodeExpertsParams;
    using tensor_args_t = MoeDecodeExpertsTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MoeDecodeExpertsProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor moe_decode_experts(
    const Tensor& x,
    const Tensor& weights,
    const Tensor& w1,
    const Tensor& w2,
    uint32_t local,
    uint32_t inter,
    float limit,
    uint32_t row_tokens = 0,
    const std::optional<Tensor>& x_row = std::nullopt,
    uint32_t h1_sem_x = 0,
    uint32_t h1_sem_r = 0,
    uint32_t rs_sem = 0,
    uint32_t rs_mail = 0,
    const std::optional<tt::tt_metal::MemoryConfig>& rs_memory_config = std::nullopt);
}  // namespace ttnn::prim
