// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

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
                * :attr:`output_dtype`: BFLOAT16 (default), FLOAT32 or BFLOAT8_B.
                * :attr:`k_split`: K slices (1 .. K / 32); 0 chooses the split from the shapes (K whole on each
                  core, but 4 slices when N leaves fewer than 24 blocks of columns).
                * :attr:`memory_config`: the output's (default DRAM interleaved; e.g. L1 width-sharded for an
                  all-reduce that takes it so).
                * :attr:`x_heads`: H: x is [1, H, M, K / H] (each head's tiles together, as a per-head matmul's
                  output), read as the [1, 1, M, K] it stands for; 0 (default): x is [1, 1, M, K].  x may also be in
                  L1 (e.g. a width-sharded norm's output).
                * :attr:`out_heads`: H: the output as [1, H, M, N / H] (heads first); 0 (default): [1, 1, M, N].
                * :attr:`split_columns`: S (a multiple of 32): the output's columns S .. N - 1 written into
                  `split_out` ([1, 1, M, N - S], allocated by the caller, any memory config), the returned output
                  [1, 1, M, S]; 0 (default): none.  The same tiles as without these (only their places differ).
                * :attr:`fg_weight`, :attr:`fg_out`: W2 [1, 1, K2, N2] (BFLOAT8_B / BFLOAT16 TILE, DRAM interleaved)
                  and fg_out [1, 1, M, N2] (BFLOAT16 TILE, DRAM interleaved, allocated by the caller): also fg_out =
                  the output's last K2 columns @ W2, the same bits as ttnn.linear of those columns with its default
                  1D config (HiFi4, fp32 Dest, packer L1 accumulation), made on the cores left without a block of
                  the output (KDA's f | gate, GLM53_KDA_FG_FUSED).  K whole on each core only.
                * :attr:`conv_side`, :attr:`conv_base`, :attr:`conv_e`, :attr:`conv_s`, :attr:`conv_out`: also
                  conv_out = DFlash2's dynamic convolution of that side (dflash2_ops.conv) of x with the output as its
                  delta (the conv op's operands: base FLOAT32 [1, 2, 32, K], E [1, 1, 512, 32], S [1, 1, 32, 32];
                  conv_out [1, 1, M, K] BFLOAT16, allocated by the caller; all DRAM interleaved), the conv op's bits,
                  made on the cores left without a block of the output (GLM53_DFLASH2_CONV_FUSED).
                * :attr:`landing`: [A, B, tag, then (a, b) a core in decode_matmul_streams' order]: each core's
                  weight chunks 0 .. a - 1 from its L1 region at A and a .. a + b - 1 from the one at B where their
                  seals say the riders filled them (ttnn/w1_dense.py, GLM53_W1D: part of a dense weight put in L1
                  ahead of the op), else DRAM: the same values.  None (default): DRAM.

            Returns:
                [1, 1, M, N] TILE ([1, H, M, N / H] with out_heads, [1, 1, M, S] with split_columns), as
                memory_config says.
        )doc",
        &decode_matmul,
        nb::arg("x").noconvert(),
        nb::arg("w").noconvert(),
        nb::kw_only(),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("k_split") = 0,
        nb::arg("memory_config") = nb::none(),
        nb::arg("x_heads") = 0,
        nb::arg("out_heads") = 0,
        nb::arg("split_columns") = 0,
        nb::arg("split_out") = nb::none(),
        nb::arg("fg_weight") = nb::none(),
        nb::arg("fg_out") = nb::none(),
        nb::arg("conv_side") = 0,
        nb::arg("conv_base") = nb::none(),
        nb::arg("conv_e") = nb::none(),
        nb::arg("conv_s") = nb::none(),
        nb::arg("conv_out") = nb::none(),
        nb::arg("landing") = nb::none());

    ttnn::bind_function<"decode_matmul_streams", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            The weight streams decode_matmul runs for w at m_rows rows of x (k_split 0: chosen from the shapes;
            fg_k: fg's K tiles, as with fg_weight): one list a core, [x, y, k0, kc, n0, ncb, chunks, a chunk's slot
            bytes, cr, sb, stride, N / 32], in the order a landing's (a, b) pairs take (ttnn/w1_dense.py).
        )doc",
        &decode_matmul_streams,
        nb::arg("w").noconvert(),
        nb::arg("m_rows"),
        nb::arg("k_split") = 0,
        nb::arg("fg_k") = 0);

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
