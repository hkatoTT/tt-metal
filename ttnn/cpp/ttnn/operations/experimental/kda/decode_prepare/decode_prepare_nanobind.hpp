// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::kda::decode_prepare::detail {

void bind_decode_prepare(nb::module_& mod);

}  // namespace ttnn::operations::experimental::kda::decode_prepare::detail
