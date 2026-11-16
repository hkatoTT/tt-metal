// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "decode_matmul_program_factory.hpp"
#include "decode_matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct DecodeMatmulDeviceOperation {
    using operation_attributes_t = DecodeMatmulParams;
    using tensor_args_t = DecodeMatmulTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<DecodeMatmulProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor decode_matmul(
    const Tensor& x,
    const Tensor& w,
    std::optional<tt::tt_metal::DataType> output_dtype,
    uint32_t k_split,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn::prim
