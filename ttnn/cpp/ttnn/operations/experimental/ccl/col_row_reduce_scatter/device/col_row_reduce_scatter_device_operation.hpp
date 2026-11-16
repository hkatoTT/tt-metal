// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/col_row_reduce_scatter/device/col_row_reduce_scatter_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/col_row_reduce_scatter/device/col_row_reduce_scatter_program_factory.hpp"

namespace ttnn::experimental::prim {

struct ColRowReduceScatterDeviceOperation {
    using operation_attributes_t = ColRowReduceScatterParams;
    using tensor_args_t = ColRowReduceScatterInputs;
    using spec_return_value_t = ColRowReduceScatterResultSpec;
    using tensor_return_value_t = ColRowReduceScatterResult;
    using program_factory_t = std::variant<ColRowReduceScatterMeshWorkloadFactory>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::ColRowReduceScatterResult col_row_reduce_scatter(
    const Tensor& input,
    const Tensor& col_intermediate,
    const Tensor& col_output,
    const Tensor& row_intermediate,
    const Tensor& row_output,
    const std::optional<Tensor>& shared,
    const std::optional<Tensor>& total,
    const std::optional<Tensor>& ag_output,
    const ttnn::experimental::prim::ColRowReduceScatterParams& params);

}  // namespace ttnn::prim
