// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_matmul_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/allocator.hpp>
#include "decode_matmul_blocks.hpp"

namespace ttnn::experimental::prim {

namespace {
void check_dram_tile(const Tensor& t, const char* name) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.is_allocated(), "{} must be an allocated device tensor", name);
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
    TT_FATAL(
        !t.memory_config().is_sharded() && t.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "{} must be DRAM interleaved",
        name);
}

uint32_t plan_div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

constexpr uint32_t MAX_K_SPLIT = 31;  // a bit of the owners' SEM_RED a slice
}  // namespace

DecodeMatmulPlan decode_matmul_plan(
    uint32_t kt, uint32_t nt, uint32_t gx, uint32_t gy, uint32_t wbytes, uint32_t k_split, uint32_t banks) {
    TT_FATAL(kt >= 1 && nt >= 1 && gx >= 1 && gy >= 1 && banks >= 1, "decode_matmul_plan: empty shapes or grid");
    TT_FATAL(wbytes >= 1, "decode_matmul_plan: the weight tile bytes");
    if (k_split != 0) {
        TT_FATAL(
            k_split <= std::min({kt, gx * gy, MAX_K_SPLIT}),
            "k_split {} must be at most K / 32 = {} and {}",
            k_split,
            kt,
            std::min(gx * gy, MAX_K_SPLIT));
    }
    // test switches: N tiles per core; DECODE_MATMUL_STRIDED=0: blocks of consecutive columns
    const char* nc_env = std::getenv("DECODE_MATMUL_NC");
    const uint32_t forced_nc = nc_env ? static_cast<uint32_t>(std::stoul(nc_env)) : 0;
    const bool strided = decode_matmul_strided();
    auto blocks = [&](uint32_t nc) {
        return static_cast<uint32_t>(decode_matmul_blocks(nt, banks, nc, strided).size());
    };
    // the rectangle of each of ks slices: the most cores (the slices then cover the grid), then the fewest rectangles
    // beyond ks, then the squarest
    auto rect = [&](uint32_t ks) {
        uint32_t cw = 0, rh = 0;
        for (uint32_t w = 1; w <= gx; ++w) {
            for (uint32_t h = 1; h <= gy; ++h) {
                const uint32_t count = (gx / w) * (gy / h);
                if (count < ks) {
                    continue;
                }
                const uint32_t cores = w * h, best_cores = cw * rh;
                const uint32_t best_count = cw ? (gx / cw) * (gy / rh) : 0;
                const auto skew = [](uint32_t a, uint32_t b) { return a > b ? a - b : b - a; };
                if (cw == 0 || cores > best_cores ||
                    (cores == best_cores &&
                     (count < best_count || (count == best_count && skew(w, h) < skew(cw, rh))))) {
                    cw = w;
                    rh = h;
                }
            }
        }
        return std::pair<uint32_t, uint32_t>{cw, rh};
    };
    // The split, from the device times of the 9 decode shapes (BH 12 x 10, 8 banks): K whole on
    // each core (no partials to add) won everywhere but the router; blocks of 2 columns (a request of 2 adjacent
    // pages a K row) beat 1 from N / 32 = 64 on (DSA a_proj 27.5 -> 26.0 us), 1 column was better at 32 (the shared
    // expert's gate | up 16.4 -> 15.1); with fewer than 24 blocks (the router's 9) K in 4 slices on 6 x 5 rectangles
    // (12.0 us, against 13.7 whole and 15-21 in 2, 3, 6, 8 or 12 slices).
    uint32_t ks = k_split;
    uint32_t nc = forced_nc ? forced_nc : (nt >= 64 ? 2 : 1);
    if (ks == 0) {
        ks = blocks(nc) >= 24 ? 1 : std::min({4u, kt, gx * gy});
    }
    const auto [cw, rh] = rect(ks);
    TT_FATAL(cw != 0, "decode_matmul_plan: {} K slices do not fit the {} x {} grid", ks, gx, gy);
    if (!forced_nc) {  // (strided, at least min(banks, nt) blocks: one per residue class)
        if (k_split != 0) {
            nc = plan_div_up(nt, std::min(cw * rh, nt));
        }
        while (nc < nt && blocks(nc) > cw * rh) {
            ++nc;
        }
    }
    const uint32_t nb = blocks(nc);
    TT_FATAL(
        nb <= cw * rh, "decode_matmul_plan: {} blocks of {} N tiles do not fit {} x {} cores a slice", nb, nc, cw, rh);
    return {ks, nc, nb, cw, rh};
}

bool decode_matmul_strided() {
    const char* v = std::getenv("DECODE_MATMUL_STRIDED");
    return v == nullptr || v[0] != '0';
}

void DecodeMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    using tt::tt_metal::DataType;
    check_dram_tile(t.x, "x");
    check_dram_tile(t.w, "w");
    TT_FATAL(t.x.dtype() == DataType::BFLOAT16, "x must be BFLOAT16");
    TT_FATAL(
        t.w.dtype() == DataType::BFLOAT8_B || t.w.dtype() == DataType::BFLOAT4_B || t.w.dtype() == DataType::BFLOAT16,
        "w must be BFLOAT8_B, BFLOAT4_B or BFLOAT16");
    TT_FATAL(
        args.output_dtype == DataType::BFLOAT16 || args.output_dtype == DataType::FLOAT32,
        "output_dtype must be BFLOAT16 or FLOAT32");
    const auto& xs = t.x.logical_shape();
    const auto& ws = t.w.logical_shape();
    const uint32_t m = xs[-2], k = xs[-1], n = ws[-1];
    TT_FATAL(m >= 1 && m <= 2 * tt::constants::TILE_HEIGHT, "x must have at most 64 rows (two tile rows), has {}", m);
    TT_FATAL(ws[-2] == k, "w must be [K, N] with x's K = {}, is [{}, {}]", k, ws[-2], n);
    TT_FATAL(
        k % tt::constants::TILE_WIDTH == 0 && n % tt::constants::TILE_WIDTH == 0, "K and N must be multiples of 32");
    TT_FATAL(t.x.logical_volume() == static_cast<uint64_t>(m) * k, "x must be [1, 1, M, K] (leading dims 1)");
    TT_FATAL(t.w.logical_volume() == static_cast<uint64_t>(k) * n, "w must be [1, 1, K, N] (leading dims 1)");
    const uint32_t kt = k / tt::constants::TILE_WIDTH;
    TT_FATAL(args.k_split >= 1 && args.k_split <= kt, "k_split {} must be 1 .. K / 32 = {}", args.k_split, kt);
}

DecodeMatmulDeviceOperation::spec_return_value_t DecodeMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& t) {
    auto shape = t.x.logical_shape();
    shape[-1] = t.w.logical_shape()[-1];
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            args.output_dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            args.output_memory_config));
}

DecodeMatmulDeviceOperation::tensor_return_value_t DecodeMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(args, t), t.x.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor decode_matmul(
    const Tensor& x,
    const Tensor& w,
    std::optional<tt::tt_metal::DataType> output_dtype,
    uint32_t k_split,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using OperationType = ttnn::experimental::prim::DecodeMatmulDeviceOperation;
    const auto grid = x.device()->compute_with_storage_grid_size();
    const uint32_t kt = x.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t nt = w.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t wbytes = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(w.dtype()));
    const uint32_t banks = x.device()->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const auto plan = ttnn::experimental::prim::decode_matmul_plan(kt, nt, grid.x, grid.y, wbytes, k_split, banks);
    auto env = [](const char* name) {
        const char* v = std::getenv(name);
        return v ? static_cast<uint32_t>(std::stoul(v)) : 0u;
    };
    auto operation_attributes = OperationType::operation_attributes_t{
        output_dtype.value_or(tt::tt_metal::DataType::BFLOAT16),
        plan.k_split,
        plan.n_per_core,
        plan.rect_w,
        plan.rect_h,
        ttnn::experimental::prim::decode_matmul_strided() ? 1u : 0u,
        env("DECODE_MATMUL_DEBUG"),
        env("DECODE_MATMUL_AHEAD"),
        env("DECODE_MATMUL_CHUNK"),
        env("DECODE_MATMUL_NOCS") == 1 ? 1u : 2u,
        memory_config.value_or(tt::tt_metal::MemoryConfig{
            tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM})};
    auto tensor_args = OperationType::tensor_args_t{x, w};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
