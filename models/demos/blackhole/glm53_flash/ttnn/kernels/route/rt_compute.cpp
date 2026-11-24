// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routing's compute (moe_decode_route's packed compute: the same SFPU sigmoid and add, fp32 unpacked to Dest, four
// tiles a pass): the scores sigmoid(logits) and the choice keys scores + bias.  With `trisc` and 3 rows or more, then
// the last count / 3 rows' routing on the unpack thread's RISC (`stage_row`, rt_common.hpp), staged in the scratch CB
// for the writer to write: the third router of a core (at 256 tokens 16 cores have 3 rows, and the writer's 2 in a
// row were most of the op's time).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "rt_common.hpp"

void kernel_main() {
    constexpr uint32_t CB_LOG = get_compile_time_arg_val(0);
    constexpr uint32_t CB_BIAS = get_compile_time_arg_val(1);
    constexpr uint32_t CB_SC = get_compile_time_arg_val(2);
    constexpr uint32_t CB_CH = get_compile_time_arg_val(3);
    constexpr uint32_t E = get_compile_time_arg_val(4);  // values a row (the block layout, rt_common.hpp)
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(5);
    constexpr uint32_t k_arg = get_compile_time_arg_val(6);  // k
    constexpr uint32_t k = k_arg & 0xffu;
    constexpr bool globals = ((k_arg >> 16) & 1u) != 0;  // (rt_common.hpp stage_row: | 1 << 16)
    constexpr uint32_t local = get_compile_time_arg_val(7);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(8);
    constexpr bool trisc = get_compile_time_arg_val(9) != 0;
    constexpr bool share = get_compile_time_arg_val(10) != 0;
    const uint32_t count = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(CB_LOG, CB_SC);
    const uint32_t np = (count * E + 1023) / 1024;
    {
        DeviceZoneScopedN("C-WAIT");
        cb_wait_front(CB_BIAS, np);
        cb_wait_front(CB_LOG, np);
    }
    DeviceZoneScopedN("C-SIG");
    cb_reserve_back(CB_SC, np);
    cb_reserve_back(CB_CH, np);
    for (uint32_t t = 0; t < np; t += 4) {
        const uint32_t n = np - t < 4 ? np - t : 4;  // Dest 0-3 the scores, 4-7 the bias and then the keys
        tile_regs_acquire();
        copy_init(CB_LOG);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(CB_LOG, t + i, i);
        }
        sigmoid_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            sigmoid_tile(i);
        }
        copy_init(CB_BIAS);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(CB_BIAS, t + i, 4 + i);
        }
        add_binary_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            add_binary_tile(i, 4 + i, 4 + i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile<true>(i, CB_SC, t + i);
            pack_tile<true>(4 + i, CB_CH, t + i);
        }
        tile_regs_release();
    }
    cb_push_back(CB_SC, np);
    cb_push_back(CB_CH, np);
    cb_pop_front(CB_LOG, np);
    cb_pop_front(CB_BIAS, np);

    const moe_route::Rows rows = moe_route::rows_of(count, trisc, share);
    if (rows.reader < count) {
        // (addresses through the unpack thread: the same on every thread; the scores' and keys' read pointers still at
        // their start)
        const uint32_t scr = moe_route::align64(get_tile_address(CB_SCR, 0));
        const uint32_t sc0 = get_tile_address(CB_SC, 0), ch0 = get_tile_address(CB_CH, 0);
        cb_wait_front(CB_SCR, 1);  // the writer's base and shared rows, its count of these rows zeroed
        cb_wait_front(CB_SC, np);  // (the packer's writes in L1)
        cb_wait_front(CB_CH, np);
#ifdef TRISC_UNPACK
        DeviceZoneScopedN("C-ROUTE");
        asm volatile("fence" ::: "memory");  // (the RISC's small L1 cache: lines the NoC and the packer wrote)
        volatile uint32_t* sync = reinterpret_cast<volatile uint32_t*>(scr + moe_route::TSYNC);
        const uint32_t base = sync[0];
        const uint32_t lo = sync[1], hi = sync[3];  // the rows that add the shared expert (`shared_span`)
        for (uint32_t j = rows.reader; j < count; ++j) {
            moe_route::stage_row<k, E, globals>(
                reinterpret_cast<const uint32_t*>(sc0 + j * moe_route::R),
                reinterpret_cast<const uint32_t*>(ch0 + j * moe_route::R),
                16 * count,
                base,
                local,
                moe_route::as_float(scale_bits),
                j >= lo && j < hi,
                scr + moe_route::WOUT + j * 2 * moe_route::R);
        }
        asm volatile("fence" ::: "memory");
        sync[2] = count - rows.reader;
#endif
    }
}
