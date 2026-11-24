// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "moe_decode_route_program_factory.hpp"
#include "moe_decode_route_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MoeDecodeRouteDeviceOperation {
    using operation_attributes_t = MoeDecodeRouteParams;
    using tensor_args_t = MoeDecodeRouteTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MoeDecodeRouteProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor moe_decode_route(
    const Tensor& logits,
    const Tensor& bias,
    const Tensor& local_base,
    uint32_t tokens,
    uint32_t k,
    uint32_t local,
    float scale,
    bool shared = false);

}  // namespace ttnn::prim
