// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/col_row_reduce_scatter/col_row_reduce_scatter.hpp"

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/col_row_reduce_scatter/device/col_row_reduce_scatter_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> col_row_reduce_scatter(
    const ttnn::Tensor& input,
    const ttnn::Tensor& col_intermediate,
    const ttnn::Tensor& col_output,
    const ttnn::Tensor& row_intermediate,
    const ttnn::Tensor& row_output,
    const std::vector<GlobalSemaphore>& col_semaphores,
    const std::vector<GlobalSemaphore>& row_semaphores,
    tt::tt_metal::CoreCoord col_core_grid_offset,
    tt::tt_metal::CoreCoord row_core_grid_offset,
    tt::tt_metal::CoreCoord relay_core,
    const std::optional<GlobalSemaphore>& col_barrier,
    const std::optional<GlobalSemaphore>& row_barrier,
    std::optional<uint32_t> col_num_links,
    std::optional<uint32_t> row_num_links,
    std::optional<uint32_t> col_chunks_per_sync,
    std::optional<uint32_t> col_workers_per_direction,
    std::optional<uint32_t> row_chunks_per_sync,
    std::optional<uint32_t> row_workers_per_direction,
    const std::optional<ttnn::Tensor>& shared,
    const std::optional<ttnn::Tensor>& total,
    tt::tt_metal::CoreCoord add_core_grid_offset,
    uint32_t add_cores,
    const std::optional<ttnn::Tensor>& ag_output,
    const std::optional<GlobalSemaphore>& ag_counter,
    tt::tt_metal::CoreCoord ag_core,
    uint32_t ag_link,
    const std::optional<tt::tt_metal::ProgramDescriptor>& extra_program,
    const std::vector<ttnn::Tensor>& extra_inputs,
    const std::vector<uint32_t>& extra_slots,
    tt::tt_metal::CoreCoord extra_relay_core,
    uint32_t extra_relay_sem,
    uint32_t extra_producers) {
    auto* mesh_device = input.device();
    TT_FATAL(mesh_device != nullptr, "col_row_reduce_scatter: a device tensor");
    const ttnn::experimental::prim::ColRowReduceScatterParams params{
        .col_num_links = col_num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, 0)),
        .row_num_links = row_num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, 1)),
        .topology = ttnn::ccl::Topology::Ring,
        .col_semaphores = col_semaphores,
        .row_semaphores = row_semaphores,
        .col_barrier = col_barrier,
        .row_barrier = row_barrier,
        .col_core_grid_offset = col_core_grid_offset,
        .row_core_grid_offset = row_core_grid_offset,
        .relay_core = relay_core,
        .add_core_grid_offset = add_core_grid_offset,
        .add_cores = add_cores,
        .ag_core = ag_core,
        .ag_counter = ag_counter,
        .ag_link = ag_link,
        .col_chunks_per_sync = col_chunks_per_sync,
        .col_workers_per_direction = col_workers_per_direction,
        .row_chunks_per_sync = row_chunks_per_sync,
        .row_workers_per_direction = row_workers_per_direction,
        .extra_program = extra_program,
        .extra_slots = extra_slots,
        .extra_relay_core = extra_relay_core,
        .extra_relay_sem = extra_relay_sem,
        .extra_producers = extra_producers,
    };
    auto out = ttnn::prim::col_row_reduce_scatter(
        input,
        col_intermediate,
        col_output,
        row_intermediate,
        row_output,
        shared,
        total,
        ag_output,
        extra_inputs,
        params);
    return {out.col_output, out.row_output};
}

}  // namespace ttnn::experimental
