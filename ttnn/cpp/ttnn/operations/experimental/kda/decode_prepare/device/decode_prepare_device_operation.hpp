// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "decode_prepare_program_factory.hpp"
#include "decode_prepare_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// The users a decode_prepare call takes (its `users`, or proj's rows)
uint32_t prepare_users(const DecodePrepareParams& args, const DecodePrepareTensorArgs& t);

struct DecodePrepareDeviceOperation {
    using operation_attributes_t = DecodePrepareParams;
    using tensor_args_t = DecodePrepareTensorArgs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 5>;
    using tensor_return_value_t = std::array<Tensor, 5>;
    using program_factory_t = std::variant<DecodePrepareProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::array<Tensor, 5> kda_decode_prepare(
    const Tensor& proj,
    const Tensor& fg,
    const Tensor& history0,
    const Tensor& history1,
    const Tensor& history2,
    const Tensor& taps,
    const Tensor& gate_consts,
    uint32_t heads,
    float lower,
    float q_scale,
    uint32_t users = 0,
    const std::optional<Tensor>& keep = std::nullopt,
    const std::optional<Tensor>& saved = std::nullopt,
    uint32_t tokens = 1,
    bool pair_major = false,
    bool l1_outputs = false,
    bool split_channels = false,
    bool row_split = false,
    const std::optional<Tensor>& history0_out = std::nullopt,
    const std::optional<Tensor>& history1_out = std::nullopt,
    const std::optional<Tensor>& history2_out = std::nullopt,
    const std::optional<Tensor>& saved_out = std::nullopt);
}  // namespace ttnn::prim
