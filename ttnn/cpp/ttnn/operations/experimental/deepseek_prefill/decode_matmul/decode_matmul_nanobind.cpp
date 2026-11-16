// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "decode_matmul.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::decode_matmul::detail {

void bind_experimental_decode_matmul_operation(nb::module_& mod) {
    ttnn::bind_function<"decode_matmul", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            x @ w for decode (one tile row of tokens, weights read once from DRAM).  K is split into `k_split`
            slices, each on its own rectangle of cores (x's slice read once and multicast to it), and N into blocks
            of a few tiles, block b on core b of every rectangle.  Each core's partial over its K slice stays in fp32
            (HiFi4, fp32 Dest); the slices' partials of an output tile are added in fp32 in the order of K and
            rounded once to the output dtype.

            Args:
                * :attr:`x`: [1, 1, M, K] BFLOAT16 TILE, M <= 32, DRAM interleaved.
                * :attr:`w`: [1, 1, K, N] BFLOAT8_B or BFLOAT16 TILE, DRAM interleaved.

            Keyword Args:
                * :attr:`output_dtype`: BFLOAT16 (default) or FLOAT32.
                * :attr:`k_split`: K slices (1 .. K / 32); 0 chooses the split from the shapes (K whole on each
                  core, but 4 slices when N leaves fewer than 24 blocks of columns).
                * :attr:`memory_config`: the output's (default DRAM interleaved; e.g. L1 width-sharded for an
                  all-reduce that takes it so).

            Returns:
                [1, 1, M, N] TILE, as memory_config says.
        )doc",
        &decode_matmul,
        nb::arg("x").noconvert(),
        nb::arg("w").noconvert(),
        nb::kw_only(),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("k_split") = 0,
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"decode_matmul_plan", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            The split decode_matmul would use, without a device: (k_split, N tiles per core, N blocks, rectangle
            width, rectangle height) for K / 32 = k_tiles, N / 32 = n_tiles on a grid_x x grid_y compute grid with
            `banks` DRAM banks and weight tiles of weight_tile_bytes (1088 BFLOAT8_B, 2048 BFLOAT16).
        )doc",
        &decode_matmul_plan,
        nb::arg("k_tiles"),
        nb::arg("n_tiles"),
        nb::arg("grid_x"),
        nb::arg("grid_y"),
        nb::arg("weight_tile_bytes"),
        nb::arg("k_split") = 0,
        nb::arg("banks") = 8);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::decode_matmul::detail
