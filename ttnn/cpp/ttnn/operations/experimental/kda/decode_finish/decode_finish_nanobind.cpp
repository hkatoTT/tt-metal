// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_finish_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "decode_finish.hpp"

namespace ttnn::operations::experimental::kda::decode_finish::detail {

void bind_decode_finish(nb::module_& mod) {
    ttnn::bind_function<"decode_finish", "ttnn.experimental.kda.">(
        mod,
        R"doc(
            KDA decode after the delta rule: o rounded to bf16, the RMS norm per (user, head) times o_norm and
            sigmoid(gate), rounded to bf16, as [1, 1, B, H 128] BFLOAT16 (o_proj's input).  The work in pieces of
            whole 4-row groups over the cores where the units fit them, the compute in two stages.
        )doc",
        &ttnn::experimental::kda::decode_finish,
        nb::arg("o").noconvert(),
        nb::arg("fg").noconvert(),
        nb::arg("o_norm").noconvert(),
        nb::arg("heads"),
        nb::arg("eps"));
}

}  // namespace ttnn::operations::experimental::kda::decode_finish::detail
