// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "mp_common.hpp"

// The prefill MoE's plan, step 1 (moe_prefill.py `plan`): core c counts the pairs per expert over the route op's tile
// rows tr0 .. tr1 - 1 and writes the counts to page c of cnt (uint32 [P, 128]).  With a mask (bf16 tiles [N, 32],
// column 0), a token whose mask is 0 (a padding token, a row without a request) has no pairs.  `signal` (the plan in the
// x all-gather's program, moe.PLAN_WITH_X2D 2: the push in the same program on NCRISC): once its page is written, one
// increment of semaphore `sem` on the gathering core (runtime arg 6: its NoC x | y << 16), which tells the push cores.
void kernel_main() {
    using namespace moe_prefill;
    const uint32_t route_addr = get_arg_val<uint32_t>(0);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);
    const uint32_t tr0 = get_arg_val<uint32_t>(3);
    const uint32_t tr1 = get_arg_val<uint32_t>(4);
    const uint32_t mask_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t local = get_compile_time_arg_val(0);
    constexpr uint32_t cb = get_compile_time_arg_val(1);
    constexpr auto r_args = TensorAccessorArgs<2>();
    constexpr auto c_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t has_mask = get_compile_time_arg_val(c_args.next_compile_time_args_offset());
    constexpr auto m_args = TensorAccessorArgs<c_args.next_compile_time_args_offset() + 1>();
    constexpr bool signal = get_compile_time_arg_val(m_args.next_compile_time_args_offset()) != 0;
    constexpr uint32_t sem_id = get_compile_time_arg_val(m_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t CNT_PAGE = 128 * 4;

    const auto s_route = TensorAccessor(r_args, route_addr, 4 * F);
    const auto s_cnt = TensorAccessor(c_args, cnt_addr, CNT_PAGE);
    const auto s_mask = TensorAccessor(m_args, mask_addr, 2048);
    const uint32_t tile_l1 = get_write_ptr(cb);
    const uint32_t cnt_l1 = tile_l1 + 2 * 4 * F;
    const uint32_t mask_l1 = cnt_l1 + 512;  // the tile row's mask tile (bf16)
    volatile tt_l1_ptr uint16_t* mask = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_l1);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_l1);
    for (uint32_t e = 0; e < 128; ++e) {
        cnt[e] = 0;
    }
    if (tr0 < tr1) {
        noc_async_read(s_route.get_noc_addr(tr0), tile_l1, 4 * F);
    }
    for (uint32_t tr = tr0; tr < tr1; ++tr) {
        noc_async_read_barrier();
        const uint32_t cur = tile_l1 + ((tr - tr0) % 2) * 4 * F;
        if constexpr (has_mask) {
            noc_async_read(s_mask.get_noc_addr(tr), mask_l1, 2048);
            noc_async_read_barrier();
        }
        if (tr + 1 < tr1) {  // the next tile while this one is counted
            noc_async_read(s_route.get_noc_addr(tr + 1), tile_l1 + ((tr + 1 - tr0) % 2) * 4 * F, 4 * F);
        }
        for (uint32_t r = 0; r < 32; ++r) {
            if constexpr (has_mask) {
                if (mask[(r / 16) * 512 + (r % 16) * 16] == 0) {
                    continue;
                }
            }
            const uint32_t row = cur + (r / 16) * 2 * F + (r % 16) * R;
            volatile tt_l1_ptr uint32_t* id_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row);
            volatile tt_l1_ptr uint32_t* w_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row + F);
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t e = id_row[j];
                if (w_row[j] != 0 && e < local) {
                    cnt[e] += 1;
                }
            }
        }
    }
    noc_async_read_barrier();
    noc_async_write(cnt_l1, s_cnt.get_noc_addr(c), CNT_PAGE);
    noc_async_write_barrier();
    if constexpr (signal) {  // (the page acked: once the gatherer has all P increments, every page is readable)
        const uint32_t xy = get_arg_val<uint32_t>(6);
        noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, get_semaphore(sem_id)), 1);
        noc_async_atomic_barrier();
    }
}
