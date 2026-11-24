// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_matmul.hpp"
#include "device/decode_matmul_device_operation.hpp"
#include "device/decode_matmul_blocks.hpp"

#include <cstdlib>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::decode_matmul {

ttnn::Tensor decode_matmul(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    std::optional<tt::tt_metal::DataType> output_dtype,
    uint32_t k_split,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    uint32_t x_heads,
    uint32_t out_heads,
    uint32_t split_columns,
    const std::optional<ttnn::Tensor>& split_out,
    const std::optional<ttnn::Tensor>& fg_weight,
    const std::optional<ttnn::Tensor>& fg_out,
    uint32_t conv_side,
    const std::optional<ttnn::Tensor>& conv_base,
    const std::optional<ttnn::Tensor>& conv_e,
    const std::optional<ttnn::Tensor>& conv_s,
    const std::optional<ttnn::Tensor>& conv_out,
    const std::optional<std::vector<uint32_t>>& landing) {
    return ttnn::prim::decode_matmul(
        x,
        w,
        output_dtype,
        k_split,
        memory_config,
        x_heads,
        out_heads,
        split_columns,
        split_out,
        fg_weight,
        fg_out,
        conv_side,
        conv_base,
        conv_e,
        conv_s,
        conv_out,
        landing);
}

std::vector<std::vector<uint32_t>> decode_matmul_streams(
    const ttnn::Tensor& w, uint32_t m_rows, uint32_t k_split, uint32_t fg_k) {
    namespace prim = ttnn::experimental::prim;
    auto* device = w.device();
    TT_FATAL(device != nullptr, "decode_matmul_streams: w must be a device tensor");
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t kt = w.logical_shape()[-2] / tt::constants::TILE_WIDTH;
    const uint32_t nt = w.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t wbytes = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(w.dtype()));
    const uint32_t banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const auto plan = prim::decode_matmul_plan(kt, nt, grid.x, grid.y, wbytes, k_split, banks);
    const char* chunk = std::getenv("DECODE_MATMUL_CHUNK");  // (as decode_matmul's attributes)
    const uint32_t mt = (m_rows + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    const auto s = prim::decode_matmul_streams(
        kt,
        nt,
        mt,
        plan.k_split,
        plan.n_per_core,
        plan.rect_w,
        plan.rect_h,
        grid.x,
        banks,
        prim::decode_matmul_strided(),
        fg_k,
        chunk ? static_cast<uint32_t>(std::stoul(chunk)) : 0u);
    std::vector<std::vector<uint32_t>> out;
    for (const auto& c : s.cores) {
        out.push_back({c.x, c.y, c.k0, c.kc, c.n0, c.ncb, c.chunks, s.cr * s.sb * wbytes, s.cr, s.sb, s.stride, nt});
    }
    return out;
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
