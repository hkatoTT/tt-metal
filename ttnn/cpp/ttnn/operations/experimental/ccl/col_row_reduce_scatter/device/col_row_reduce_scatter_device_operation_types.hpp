// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::experimental::prim {

// The GLM-5.3 prefill MoE's way back: the column's ring reduce-scatter (cluster
// axis 0, dim 2) of a [B, 1, 8 T / B, D] input, one ring traversal per batch, and the row's ring reduce-scatter (cluster
// axis 1, dim 3) of its [B, 1, T / B, D] output in the same program, fused: the row's batch b starts once every column
// writer has written its batch b (a relay core counts them).  Each is ttnn's reduce_scatter_minimal_async ring, so the
// sums are the same bits as the two reduce-scatters one after the other.
struct ColRowReduceScatterParams {
    uint32_t col_num_links;
    uint32_t row_num_links;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> col_semaphores;
    std::vector<GlobalSemaphore> row_semaphores;
    std::optional<GlobalSemaphore> col_barrier;
    std::optional<GlobalSemaphore> row_barrier;
    CoreCoord col_core_grid_offset;
    CoreCoord row_core_grid_offset;
    CoreCoord relay_core;
    CoreCoord add_core_grid_offset;  // (with `shared`) the add stage's cores: add_cores in a row from here
    uint32_t add_cores = 8;
    // (with ag_output) the row all-reduce's all-gather in the op: its core and the receivers' counter (a global
    // semaphore); it takes link ag_link of the row, the row reduce-scatter then link 0 alone
    CoreCoord ag_core;
    std::optional<GlobalSemaphore> ag_counter;
    uint32_t ag_link = 1;
    std::optional<uint32_t> col_chunks_per_sync;
    std::optional<uint32_t> col_workers_per_direction;
    std::optional<uint32_t> row_chunks_per_sync;
    std::optional<uint32_t> row_workers_per_direction;
    // Another program carried in this one, on cores the op does not take (the prefill MoE's combine, whose output is
    // `input`): its semaphores, CBs and kernels as they are, except that each address slot (kernel k of it, runtime arg
    // i, every core: extra_slots' triples (k, i, j)) holds extra_inputs[j]'s address (j == extra_inputs.size():
    // input's), bound at every dispatch.  With extra_producers > 0 the column's reduce-scatter waits per batch (a
    // fused-op receiver) for the carried program: a relay kernel on extra_relay_core counts increments of semaphore
    // extra_relay_sem + b for batch b (extra_program's, on that core: one a batch) and once extra_producers have come
    // raises the column workers' semaphore.
    std::optional<tt::tt_metal::ProgramDescriptor> extra_program;
    std::vector<uint32_t> extra_slots;
    CoreCoord extra_relay_core;
    uint32_t extra_relay_sem = 0;
    uint32_t extra_producers = 0;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "col_num_links",
        "row_num_links",
        "col_core_grid_offset",
        "row_core_grid_offset",
        "relay_core",
        "add_core_grid_offset",
        "add_cores",
        "ag_core",
        "ag_link");
    auto attribute_values() const {
        return std::forward_as_tuple(
            col_num_links,
            row_num_links,
            col_core_grid_offset,
            row_core_grid_offset,
            relay_core,
            add_core_grid_offset,
            add_cores,
            ag_core,
            ag_link);
    }
};

struct ColRowReduceScatterInputs {
    Tensor input;             // [B, 1, 8 T / B, D]
    Tensor col_intermediate;  // as input
    Tensor col_output;        // [B, 1, T / B, D]
    Tensor row_intermediate;  // as col_output
    Tensor row_output;        // [B, 1, T / B, D / 4]
    // the shared expert's partial and the buffer for col_output + shared (the row's input): both or neither
    std::optional<Tensor> shared;  // [B, 1, T / B, D]
    std::optional<Tensor> total;   // [B, 1, T / B, D]
    std::optional<Tensor> ag_output;  // [1, 1, T, D]: the row's sum on every chip of the row (the all-gather in the op)
    std::vector<Tensor> extra_inputs;  // the carried program's tensors (extra_slots)
};

struct ColRowReduceScatterResult {
    Tensor col_output;
    Tensor row_output;
};

struct ColRowReduceScatterResultSpec {
    tt::tt_metal::TensorSpec col_output;
    tt::tt_metal::TensorSpec row_output;
};

}  // namespace ttnn::experimental::prim
