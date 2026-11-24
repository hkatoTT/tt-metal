// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {
namespace nb = nanobind;
void bind_col_row_reduce_scatter(nb::module_& mod);
}  // namespace ttnn::operations::experimental::ccl
