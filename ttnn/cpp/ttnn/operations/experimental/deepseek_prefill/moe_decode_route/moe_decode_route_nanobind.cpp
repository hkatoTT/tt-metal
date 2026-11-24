// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode_route_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "moe_decode_route.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_route::detail {

void bind_experimental_moe_decode_route_operation(nb::module_& mod) {
    ttnn::bind_function<"moe_decode_route", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Sigmoid routing with a selection bias for the chip's experts: per token, the top k of sigmoid(logits) +
            bias (ties to the lower expert); for each chosen expert e among base .. base + local - 1 the weight
            bf16(sigmoid(logit_e) / (sum of the k chosen sigmoids + 1e-20) * scale) at column e - base.  With local
            0, every chosen expert: its id (UINT32 bits) in columns 0 .. k - 1 and its weight in columns 16 .. 16 + k
            - 1, in increasing id order (the experts op's sliced layout); with local above 16, the same for the
            chosen experts among base .. base + local - 1 only, their ids as e - base (the column-sliced layout).

            Args:
                * :attr:`logits`: [.., N, E] FLOAT32 TILE.
                * :attr:`bias`: [.., 1, E] FLOAT32 TILE.
                * :attr:`local_base`: [.., 1, >= 1] UINT32 ROW_MAJOR, the chip's first expert.
                * :attr:`tokens`: the real tokens (rows after them choose nothing).
                * :attr:`k`: experts per token.
                * :attr:`local`: experts per chip (0: the chosen experts' ids and weights; above 16: those of
                  base .. base + local - 1).
                * :attr:`scale`: the routed scaling factor.

            Returns:
                [1, 1, N, 32] FLOAT32 TILE.
        )doc",
        &moe_decode_route,
        nb::arg("logits").noconvert(),
        nb::arg("bias").noconvert(),
        nb::arg("local_base").noconvert(),
        nb::arg("tokens"),
        nb::arg("k"),
        nb::arg("local"),
        nb::arg("scale"),
        nb::arg("shared") = false);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_route::detail
