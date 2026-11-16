// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A prefill chunk's pools' keys in one op (dsa_indexer.pool_keys), the reader: the constants (the 4 bias tiles and the
// reduce scaler), then per item its gate and key tiles (tile idx of ig / ik [1, 1, T, 128]), and the previous item's
// p tile (compute's CB_PROD) into 8 padded tiles: pool k's 4 rows (128 contiguous bytes of face 2 (k / 4) and of the
// face after it) to rows 0-3 of tile k (faces 0 and 1), by local NoC reads.  The padded tiles' rows 4-31 are zeroed
// once (CB_PAD's pages, all of them) and never written.
//   CT: CB_IG, CB_IK, CB_BIAS, CB_SCALER, CB_PROD, CB_PAD, PAD_PAGES, then the accessors of ig, ik (bf16 tiles), bias
//       (fp32 tiles), then the shape key.  RT: the 3 addresses, the items [i0, i1).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t TILE = 2048;  // a bf16 tile
constexpr uint32_t FACE = 512;
constexpr uint32_t COLS = 4;
constexpr uint32_t POOLS = 8;
}  // namespace

void kernel_main() {
    constexpr uint32_t CB_IG = get_compile_time_arg_val(0);
    constexpr uint32_t CB_IK = get_compile_time_arg_val(1);
    constexpr uint32_t CB_BIAS = get_compile_time_arg_val(2);
    constexpr uint32_t CB_SCALER = get_compile_time_arg_val(3);
    constexpr uint32_t CB_PROD = get_compile_time_arg_val(4);
    constexpr uint32_t CB_PAD = get_compile_time_arg_val(5);
    constexpr uint32_t PAD_PAGES = get_compile_time_arg_val(6);
    constexpr auto ig_args = TensorAccessorArgs<7>();
    constexpr auto ik_args = TensorAccessorArgs<ig_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<ik_args.next_compile_time_args_offset()>();
    static_assert(ig_args.AlignedPageSize == TILE && ik_args.AlignedPageSize == TILE);
    static_assert(bias_args.AlignedPageSize == 2 * TILE);
    static_assert(PAD_PAGES % POOLS == 0);

    const auto ig = TensorAccessor(ig_args, get_arg_val<uint32_t>(0));
    const auto ik = TensorAccessor(ik_args, get_arg_val<uint32_t>(1));
    const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(2));
    const uint32_t i0 = get_arg_val<uint32_t>(3);
    const uint32_t i1 = get_arg_val<uint32_t>(4);
    if (i1 <= i0) {
        return;
    }

    cb_reserve_back(CB_BIAS, COLS);
    for (uint32_t t = 0; t < COLS; ++t) {
        noc_async_read_page(t, bias, get_write_ptr(CB_BIAS) + t * 2 * TILE);
    }
    // the padded tiles' zeros (no page pushed yet: the write pointer is the buffer's start)
    cb_reserve_back(CB_PAD, PAD_PAGES);
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t pad0 = get_write_ptr(CB_PAD);
        for (uint32_t o = 0; o < PAD_PAGES * TILE; o += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, pad0 + o, MEM_ZEROS_SIZE);
        }
    }
    noc_async_read_barrier();
    cb_push_back(CB_BIAS, COLS);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        CB_SCALER,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL,
        /*reduce_factor=*/1>();

    auto pad = [&]() {
        cb_wait_front(CB_PROD, 1);
        cb_reserve_back(CB_PAD, POOLS);
        const uint32_t src = get_read_ptr(CB_PROD), dst = get_write_ptr(CB_PAD);
        for (uint32_t k = 0; k < POOLS; ++k) {
            const uint32_t s = src + (k / 4) * 2 * FACE + (k % 4) * 128;
            noc_async_read(get_noc_addr(s), dst + k * TILE, 128);
            noc_async_read(get_noc_addr(s + FACE), dst + k * TILE + FACE, 128);
        }
        noc_async_read_barrier();
        cb_push_back(CB_PAD, POOLS);
        cb_pop_front(CB_PROD, 1);
    };
    for (uint32_t idx = i0; idx < i1; ++idx) {
        cb_reserve_back(CB_IG, 1);
        cb_reserve_back(CB_IK, 1);
        noc_async_read_page(idx, ig, get_write_ptr(CB_IG));
        noc_async_read_page(idx, ik, get_write_ptr(CB_IK));
        noc_async_read_barrier();
        cb_push_back(CB_IG, 1);
        cb_push_back(CB_IK, 1);
        if (idx > i0) {
            pad();
        }
    }
    pad();
}
