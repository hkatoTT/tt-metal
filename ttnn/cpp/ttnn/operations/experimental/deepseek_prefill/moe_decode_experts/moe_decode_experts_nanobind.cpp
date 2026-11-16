// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode_experts_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "moe_decode_experts.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_experts::detail {

void bind_experimental_moe_decode_experts_operation(nb::module_& mod) {
    ttnn::bind_function<"moe_decode_experts", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            The decode MoE's routed experts on one chip for 1-16 tile rows of tokens: y = sum over the chip's experts
            j of weights[:, j] * (h_j @ down_j), h_j the clamped SwiGLU of x @ gate_j and x @ up_j.  inter 256: the
            chip's 256 columns of every one of `local` experts (a mesh column's, and the shared expert's slot), in the
            cores' layout (64 gate | up cores, 32 down cores); weights moe_decode_route's ids and weights (the
            column's range).  row_tokens 8 or 16: y by mesh row, each row_tokens rows at the top of their own tile row
            and the rest zero (the padded layout of the row collectives).  The in-op gather (h1_sem_x,
            h1_sem_r: two global semaphores' L1 address; x_row [t, D] this mesh row's tokens): x and weights are the
            column's landing buffers, filled by the op over the fabric from every mesh row's x_row and routing rows.
            With it and 32 / 64 tokens a mesh row, rs_sem and rs_mail (two more global semaphores' L1 address): the
            column's way back in the op, this mesh row's tokens summed over the column, [1, 1, t, D] (the bits of
            ttnn.reduce_scatter over the column of what it returns otherwise); rs_memory_config: where it goes (none:
            DRAM).

            Returns:
                [1, 1, N, D] BFLOAT16 TILE ([1, 1, N / row_tokens 32, D] with row_tokens).
        )doc",
        &moe_decode_experts,
        nb::arg("x").noconvert(),
        nb::arg("weights").noconvert(),
        nb::arg("w1").noconvert(),
        nb::arg("w2").noconvert(),
        nb::arg("local"),
        nb::arg("inter"),
        nb::arg("limit"),
        nb::arg("row_tokens") = 0,
        nb::arg("x_row") = nb::none(),
        nb::arg("h1_sem_x") = 0,
        nb::arg("h1_sem_r") = 0,
        nb::arg("rs_sem") = 0,
        nb::arg("rs_mail") = 0,
        nb::arg("rs_memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_experts::detail
