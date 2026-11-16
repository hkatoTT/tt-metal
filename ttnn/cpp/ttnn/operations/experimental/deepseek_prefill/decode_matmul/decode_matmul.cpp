// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_matmul.hpp"
#include "device/decode_matmul_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::decode_matmul {

ttnn::Tensor decode_matmul(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    std::optional<tt::tt_metal::DataType> output_dtype,
    uint32_t k_split,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    return ttnn::prim::decode_matmul(x, w, output_dtype, k_split, memory_config);
}

std::array<uint32_t, 5> decode_matmul_plan(
    uint32_t k_tiles,
    uint32_t n_tiles,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t weight_tile_bytes,
    uint32_t k_split,
    uint32_t banks) {
    const auto p = ttnn::experimental::prim::decode_matmul_plan(
        k_tiles, n_tiles, grid_x, grid_y, weight_tile_bytes, k_split, banks);
    return {p.k_split, p.n_per_core, p.n_blocks, p.rect_w, p.rect_h};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::decode_matmul
