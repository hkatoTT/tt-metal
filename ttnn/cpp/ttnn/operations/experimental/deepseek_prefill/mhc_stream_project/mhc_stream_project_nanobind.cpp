// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_project_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_stream_project.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_project::detail {

void bind_experimental_mhc_stream_project_operation(nb::module_& mod) {
    ttnn::bind_function<"mhc_stream_project", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            The mHC read's projection with the streams packed as rows, in K slices: -> [slices, R, 128] FLOAT32,
            columns 0-95 each slice's streams @ fn, columns 96-127 its Gram tile S @ S^T per row tile.

            Args:
                * :attr:`streams`: [.., R, D] BFLOAT16 TILE.
                * :attr:`fn`: [.., D, 96] BFLOAT16 TILE.
                * :attr:`slices`: K slices (dividing D's tiles).
        )doc",
        &mhc_stream_project,
        nb::arg("streams").noconvert(),
        nb::arg("fn").noconvert(),
        nb::arg("slices"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_project::detail
