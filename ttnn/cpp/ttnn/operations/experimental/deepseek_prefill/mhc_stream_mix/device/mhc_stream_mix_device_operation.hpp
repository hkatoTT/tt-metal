// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "mhc_stream_mix_program_factory.hpp"
#include "mhc_stream_mix_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MhcStreamMixDeviceOperation {
    using operation_attributes_t = MhcStreamMixParams;
    using tensor_args_t = MhcStreamMixTensorArgs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 3>;
    using tensor_return_value_t = std::array<Tensor, 3>;
    using program_factory_t = std::variant<MhcStreamMixProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::array<Tensor, 3> mhc_stream_mix(
    const Tensor& partial,
    const Tensor& mix_consts,
    const Tensor& consts,
    uint32_t tokens,
    uint32_t n,
    uint32_t iters,
    float eps,
    float norm_eps,
    uint32_t hidden);
}  // namespace ttnn::prim
