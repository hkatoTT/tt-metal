// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/col_row_reduce_scatter/device/col_row_reduce_scatter_device_operation.hpp"

#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

void ColRowReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    const auto& in = t.input.logical_shape();
    const auto& co = t.col_output.logical_shape();
    const auto& ro = t.row_output.logical_shape();
    TT_FATAL(in.rank() == 4 && co.rank() == 4 && ro.rank() == 4, "col_row_reduce_scatter: rank-4 tensors");
    auto* mesh_device = t.input.device();
    TT_FATAL(mesh_device != nullptr, "col_row_reduce_scatter: a device tensor");
    const auto& shape = mesh_device->shape();
    TT_FATAL(shape.dims() == 2, "col_row_reduce_scatter: a 2D mesh");
    TT_FATAL(
        co[0] == in[0] && co[2] * shape[0] == in[2] && co[3] == in[3],
        "col_row_reduce_scatter: col_output [B, 1, rows / {}, D]",
        shape[0]);
    TT_FATAL(
        ro[0] == co[0] && ro[2] == co[2] && ro[3] * shape[1] == co[3],
        "col_row_reduce_scatter: row_output [B, 1, rows, D / {}]",
        shape[1]);
    TT_FATAL(
        t.col_intermediate.logical_shape() == in && t.row_intermediate.logical_shape() == co,
        "col_row_reduce_scatter: the intermediates input-shaped");
    TT_FATAL(args.col_semaphores.size() >= 3 && args.row_semaphores.size() >= 3, "col_row_reduce_scatter: 3 semaphores a ring");
    TT_FATAL(t.shared.has_value() == t.total.has_value(), "col_row_reduce_scatter: shared and total together");
    if (t.shared.has_value()) {
        TT_FATAL(
            t.shared->logical_shape() == co && t.total->logical_shape() == co,
            "col_row_reduce_scatter: shared and total shaped as col_output");
        TT_FATAL(
            t.shared->dtype() == DataType::BFLOAT16 && t.total->dtype() == DataType::BFLOAT16 &&
                t.col_output.dtype() == DataType::BFLOAT16,
            "col_row_reduce_scatter: the add stage takes bf16");
        TT_FATAL(args.add_cores >= 1, "col_row_reduce_scatter: add_cores");
    }
    if (t.ag_output.has_value()) {
        TT_FATAL(args.ag_counter.has_value(), "col_row_reduce_scatter: the all-gather's counter");
        TT_FATAL(
            t.ag_output->padded_shape()[2] == co[0] * co[2] && t.ag_output->padded_shape()[3] == co[3],
            "col_row_reduce_scatter: ag_output [1, 1, T, D]");
        TT_FATAL(co[3] == 4096 && shape[1] == 4, "col_row_reduce_scatter: the all-gather takes D 4,096 over 4 chips");
    }
}

ColRowReduceScatterDeviceOperation::spec_return_value_t ColRowReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return {.col_output = t.col_output.tensor_spec(), .row_output = t.row_output.tensor_spec()};
}

ColRowReduceScatterDeviceOperation::tensor_return_value_t ColRowReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return {.col_output = t.col_output, .row_output = t.row_output};
}


ttsl::hash::hash_t ColRowReduceScatterDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& t) {
    std::vector<uint32_t> sems;
    for (const auto& s : args.col_semaphores) {
        sems.push_back(static_cast<uint32_t>(s.address()));
    }
    for (const auto& s : args.row_semaphores) {
        sems.push_back(static_cast<uint32_t>(s.address()));
    }
    sems.push_back(args.col_barrier.has_value() ? static_cast<uint32_t>(args.col_barrier->address()) : 0u);
    sems.push_back(args.row_barrier.has_value() ? static_cast<uint32_t>(args.row_barrier->address()) : 0u);
    sems.push_back(args.ag_counter.has_value() ? static_cast<uint32_t>(args.ag_counter->address()) : 0u);
    return tt::tt_metal::operation::hash_operation<ColRowReduceScatterDeviceOperation>(
        args.col_num_links,
        args.row_num_links,
        args.topology,
        args.col_core_grid_offset,
        args.row_core_grid_offset,
        args.relay_core,
        args.add_core_grid_offset,
        args.add_cores,
        args.ag_core,
        args.ag_link,
        args.col_chunks_per_sync,
        args.col_workers_per_direction,
        args.row_chunks_per_sync,
        args.row_workers_per_direction,
        sems,
        t.input,
        t.col_intermediate,
        t.col_output,
        t.row_intermediate,
        t.row_output,
        t.shared,
        t.total,
        t.ag_output);
}

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
    const ttnn::experimental::prim::ColRowReduceScatterParams& params) {
    using Op = ttnn::experimental::prim::ColRowReduceScatterDeviceOperation;
    auto tensor_args = Op::tensor_args_t{
        .input = input,
        .col_intermediate = col_intermediate,
        .col_output = col_output,
        .row_intermediate = row_intermediate,
        .row_output = row_output,
        .shared = shared,
        .total = total,
        .ag_output = ag_output};
    return ttnn::device_operation::launch<Op>(params, tensor_args);
}

}  // namespace ttnn::prim
