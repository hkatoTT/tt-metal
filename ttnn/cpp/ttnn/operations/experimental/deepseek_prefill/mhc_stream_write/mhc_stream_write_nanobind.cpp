// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_write_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_stream_write.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_write::detail {

void bind_experimental_mhc_stream_write_operation(nb::module_& mod) {
    ttnn::bind_function<"mhc_stream_write", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            The mHC write with the streams packed as rows (row n t + i: token t's stream i, n = 4): new stream
            (t, j) = bf16(sum_i comb_t[i][j] stream (t, i)) + bf16(post_t[j] block_out_t), comb and post rounded to
            bf16 first, each product summed in fp32 and rounded once.

            Args:
                * :attr:`streams`: [.., n Tp, D] BFLOAT16 TILE (Tp a multiple of 8, padded tokens' rows zero).
                * :attr:`block_out`: [.., T, D] BFLOAT16 TILE.
                * :attr:`post`: [T, n] FLOAT32 TILE.
                * :attr:`comb`: [T, n n] FLOAT32 TILE.
                * :attr:`n`: streams per token (4).

            Returns:
                the new streams [.., n Tp, D] BFLOAT16 TILE.
        )doc",
        &mhc_stream_write,
        nb::arg("streams").noconvert(),
        nb::arg("block_out").noconvert(),
        nb::arg("post").noconvert(),
        nb::arg("comb").noconvert(),
        nb::arg("n"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_write::detail
