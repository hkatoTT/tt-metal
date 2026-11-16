// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::decode_matmul {

// x @ w for decode: one tile row of tokens against weights read once from DRAM.  K is split into `k_split` slices
// (0: chosen from the shapes), each on its own rectangle of cores (x's slice read once and multicast to it); N is
// split into blocks of a few columns of one residue class mod the DRAM banks (a K row's tiles of a block adjacent in
// one bank, read with one request), block i on a core of every rectangle.  Each core's partial over its K
// slice stays in fp32 (HiFi4, fp32 Dest); the slices' partials of an output tile are added in fp32 in the order of K
// on one of the block's cores and rounded once to the output dtype.
//   x: [1, 1, M, K] BFLOAT16 TILE (M <= 32).  w: [1, 1, K, N] BFLOAT8_B or BFLOAT16 TILE.  Both DRAM interleaved.
// Returns [1, 1, M, N] `output_dtype` (BFLOAT16 by default, or FLOAT32) TILE, DRAM interleaved.
ttnn::Tensor decode_matmul(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt,
    uint32_t k_split = 0,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

// The split decode_matmul would use (no device): {k_split, n tiles per core, n blocks, rectangle width, height}.
std::array<uint32_t, 5> decode_matmul_plan(
    uint32_t k_tiles,
    uint32_t n_tiles,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t weight_tile_bytes,
    uint32_t k_split,
    uint32_t banks);

}  // namespace ttnn::operations::experimental::deepseek_prefill::decode_matmul

namespace ttnn {
using operations::experimental::deepseek_prefill::decode_matmul::decode_matmul;
}  // namespace ttnn
