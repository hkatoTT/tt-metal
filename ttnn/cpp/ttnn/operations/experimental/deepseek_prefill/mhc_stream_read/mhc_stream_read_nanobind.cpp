// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_read_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_stream_read.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_read::detail {

void bind_experimental_mhc_stream_read_operation(nb::module_& mod) {
    ttnn::bind_function<"mhc_stream_read", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            The whole mHC read (at most 16 tokens) and the block's RMS norm in one op: mhc_stream_project's partials
            sent to one core over the NoC, mhc_stream_mix's pre / post / comb, the block input P @ streams by K slice,
            and its RMS norm with gamma.

            The prefill's groups form (more than 64 rows of streams): `offset` (a multiple of 32) the call's first
            token, `tokens` its tokens, of the streams' rows / n, the outputs' rows; `outputs` an earlier call's
            (post, comb, block input) to write into (required from offset 1 on).

            Returns:
                (post [T, n], comb [T, n n]) FLOAT32 TILE, the normed block input [1, 1, T, D] BFLOAT16 TILE.
        )doc",
        &mhc_stream_read,
        nb::arg("streams").noconvert(),
        nb::arg("fn").noconvert(),
        nb::arg("mix_consts").noconvert(),
        nb::arg("consts").noconvert(),
        nb::arg("gamma").noconvert(),
        nb::arg("tokens"),
        nb::arg("n"),
        nb::arg("sinkhorn_iters"),
        nb::arg("slices"),
        nb::arg("eps"),
        nb::arg("norm_eps"),
        nb::arg("rms_eps"),
        nb::arg("offset") = 0,
        nb::arg("outputs") = nb::none());
    ttnn::bind_function<"mhc_stream_write_read", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            mhc_stream_write and the next mhc_stream_read in one program (decode: at most 16 tokens a row, or 32 in the
            groups of 8): the new streams handed to the read's slice cores in L1 as well as written to DRAM; the same
            values as the two ops.

            `offset`, `outputs` (new streams, post, comb, block input) as mhc_stream_read's; block_out, post and comb
            are then the whole streams' tokens'.

            Returns:
                the new streams [1, 1, R, D] BFLOAT16 TILE, (post [T, n], comb [T, n n]) FLOAT32 TILE, the normed block
                input [1, 1, T, D] BFLOAT16 TILE.
        )doc",
        &mhc_stream_write_read,
        nb::arg("streams").noconvert(),
        nb::arg("block_out").noconvert(),
        nb::arg("post").noconvert(),
        nb::arg("comb").noconvert(),
        nb::arg("fn").noconvert(),
        nb::arg("mix_consts").noconvert(),
        nb::arg("consts").noconvert(),
        nb::arg("gamma").noconvert(),
        nb::arg("tokens"),
        nb::arg("n"),
        nb::arg("sinkhorn_iters"),
        nb::arg("slices"),
        nb::arg("eps"),
        nb::arg("norm_eps"),
        nb::arg("rms_eps"),
        nb::arg("offset") = 0,
        nb::arg("outputs") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_read::detail
