// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "recurrent_decode_step_program_factory.hpp"
#include "recurrent_decode_step_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct RecurrentDecodeStepDeviceOperation {
    using operation_attributes_t = RecurrentDecodeStepParams;
    using tensor_args_t = RecurrentDecodeStepTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RecurrentDecodeStepProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor kda_recurrent_decode_step(
    const Tensor& state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& decay,
    const Tensor& beta,
    const std::optional<Tensor>& source = std::nullopt,
    const std::optional<Tensor>& keep = std::nullopt,
    uint32_t heads = 0,
    const std::optional<Tensor>& output = std::nullopt,
    uint32_t form = 0,
    uint32_t tokens = 0,
    const std::optional<Tensor>& saved = std::nullopt,
    bool pair_major = false,
    const std::optional<Tensor>& saved_source = std::nullopt);
}  // namespace ttnn::prim
