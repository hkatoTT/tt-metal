// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_mix_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_stream_mix.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_mix::detail {

void bind_experimental_mhc_stream_mix_operation(nb::module_& mod) {
    ttnn::bind_function<"mhc_stream_mix", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            The mHC read after mhc_stream_project (at most 32 tokens): the slices' partials summed, each token's mixes
            and RMS scale, mhc_split_sinkhorn's pre / post / comb, and the block input's weights P [T, R].

            Returns:
                (post [T, n], comb [T, n n], P [T, R]) FLOAT32 TILE.
        )doc",
        &mhc_stream_mix,
        nb::arg("partial").noconvert(),
        nb::arg("mix_consts").noconvert(),
        nb::arg("consts").noconvert(),
        nb::arg("tokens"),
        nb::arg("n"),
        nb::arg("sinkhorn_iters"),
        nb::arg("eps"),
        nb::arg("norm_eps"),
        nb::arg("hidden"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_mix::detail
