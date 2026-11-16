// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "col_row_reduce_scatter_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/core_coord.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/col_row_reduce_scatter/col_row_reduce_scatter.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_col_row_reduce_scatter(nb::module_& mod) {
    ttnn::bind_function<"col_row_reduce_scatter", "ttnn.experimental.">(
        mod,
        R"doc(
        The column's ring reduce-scatter of input [B, 1, 8 T / B, D] (mesh axis 0, dim 2, one ring traversal per
        batch) fused with the row's ring reduce-scatter of its output (mesh axis 1, dim 3) in one program: the row's
        batch b starts once the column's batch b is written.  Returns [col_output, row_output] (the persistent buffers
        passed in).  The same bits as ttnn's two reduce-scatters one after the other (the GLM-5.3 prefill MoE).
        )doc",
        &ttnn::experimental::col_row_reduce_scatter,
        nb::arg("input"),
        nb::arg("col_intermediate"),
        nb::arg("col_output"),
        nb::arg("row_intermediate"),
        nb::arg("row_output"),
        nb::arg("col_semaphores"),
        nb::arg("row_semaphores"),
        nb::arg("col_core_grid_offset"),
        nb::arg("row_core_grid_offset"),
        nb::arg("relay_core"),
        nb::kw_only(),
        nb::arg("col_barrier") = nb::none(),
        nb::arg("row_barrier") = nb::none(),
        nb::arg("col_num_links") = nb::none(),
        nb::arg("row_num_links") = nb::none(),
        nb::arg("col_chunks_per_sync") = nb::none(),
        nb::arg("col_workers_per_direction") = nb::none(),
        nb::arg("row_chunks_per_sync") = nb::none(),
        nb::arg("row_workers_per_direction") = nb::none(),
        nb::arg("shared") = nb::none(),
        nb::arg("total") = nb::none(),
        nb::arg("add_core_grid_offset") = tt::tt_metal::CoreCoord(0, 8),
        nb::arg("add_cores") = 8,
        nb::arg("ag_output") = nb::none(),
        nb::arg("ag_counter") = nb::none(),
        nb::arg("ag_core") = tt::tt_metal::CoreCoord(10, 9),
        nb::arg("ag_link") = 1);
}

}  // namespace ttnn::operations::experimental::ccl
