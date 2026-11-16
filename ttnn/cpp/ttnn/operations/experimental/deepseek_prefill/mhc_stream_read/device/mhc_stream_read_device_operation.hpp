// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "mhc_stream_read_program_factory.hpp"
#include "mhc_stream_read_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MhcStreamReadDeviceOperation {
    using operation_attributes_t = MhcStreamReadParams;
    using tensor_args_t = MhcStreamReadTensorArgs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 3>;
    using tensor_return_value_t = std::array<Tensor, 3>;
    using program_factory_t = std::variant<MhcStreamReadProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

struct MhcStreamWriteReadDeviceOperation {
    using operation_attributes_t = MhcStreamReadParams;
    using tensor_args_t = MhcStreamWriteReadTensorArgs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 4>;
    using tensor_return_value_t = std::array<Tensor, 4>;
    using program_factory_t = std::variant<MhcStreamWriteReadProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::array<Tensor, 3> mhc_stream_read(
    const Tensor& streams,
    const Tensor& fn,
    const Tensor& mix_consts,
    const Tensor& consts,
    const Tensor& gamma,
    uint32_t tokens,
    uint32_t n,
    uint32_t iters,
    uint32_t slices,
    float eps,
    float norm_eps,
    float rms_eps,
    uint32_t offset = 0,
    const std::optional<std::array<Tensor, 3>>& outputs = std::nullopt);
std::array<Tensor, 4> mhc_stream_write_read(
    const Tensor& streams,
    const Tensor& block_out,
    const Tensor& post,
    const Tensor& comb,
    const Tensor& fn,
    const Tensor& mix_consts,
    const Tensor& consts,
    const Tensor& gamma,
    uint32_t tokens,
    uint32_t n,
    uint32_t iters,
    uint32_t slices,
    float eps,
    float norm_eps,
    float rms_eps,
    uint32_t offset = 0,
    const std::optional<std::array<Tensor, 4>>& outputs = std::nullopt);
}  // namespace ttnn::prim
