// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// The column's ring reduce-scatter of input [B, 1, 8 T / B, D] (dim 2, one traversal a batch) fused with the row's
// ring reduce-scatter of its output (dim 3): returns {col_output [B, 1, T / B, D], row_output [B, 1, T / B, D / 4]}
// (the GLM-5.3 prefill MoE's way back).  The same bits as the two reduce-scatters one after the other.
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
    const std::optional<GlobalSemaphore>& col_barrier = std::nullopt,
    const std::optional<GlobalSemaphore>& row_barrier = std::nullopt,
    std::optional<uint32_t> col_num_links = std::nullopt,
    std::optional<uint32_t> row_num_links = std::nullopt,
    std::optional<uint32_t> col_chunks_per_sync = std::nullopt,
    std::optional<uint32_t> col_workers_per_direction = std::nullopt,
    std::optional<uint32_t> row_chunks_per_sync = std::nullopt,
    std::optional<uint32_t> row_workers_per_direction = std::nullopt,
    const std::optional<ttnn::Tensor>& shared = std::nullopt,
    const std::optional<ttnn::Tensor>& total = std::nullopt,
    tt::tt_metal::CoreCoord add_core_grid_offset = tt::tt_metal::CoreCoord(0, 8),
    uint32_t add_cores = 8,
    const std::optional<ttnn::Tensor>& ag_output = std::nullopt,
    const std::optional<GlobalSemaphore>& ag_counter = std::nullopt,
    tt::tt_metal::CoreCoord ag_core = tt::tt_metal::CoreCoord(10, 9),
    uint32_t ag_link = 1);

}  // namespace ttnn::experimental
