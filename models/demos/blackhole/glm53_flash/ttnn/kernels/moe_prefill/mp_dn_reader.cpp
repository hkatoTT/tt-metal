// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "mp_common.hpp"

// The prefill MoE's down, the reader (mp_common.hpp): per item (e, qq) with pairs, W2(e, q) of its 4 column blocks
// into DN_W2, then per tile row of e's pairs its 8 h tiles.
void kernel_main() {
    using namespace moe_prefill;
    uint32_t a = 0;
    const uint32_t h_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w2_addr = get_arg_val<uint32_t>(a++);
    const uint32_t seg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_items = get_arg_val<uint32_t>(a++);

    constexpr uint32_t h_tile = get_compile_time_arg_val(0);
    constexpr uint32_t w2_tile = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);  // e_w2's experts per core slice (72, 73 with the shared)
    constexpr auto h_args = TensorAccessorArgs<3>();
    constexpr auto w_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    // DNCB_NOCS 2: of every 4 W2 and h tiles the first q1 (DNCB_NOC1_QUARTERS) read on NoC 1, once the writer (BRISC,
    // NoC 1, which reads nothing after its seg page) has handed NoC 1 over (mp_noc1.hpp, S_NOC1); those reads carry
    // transaction id 1 and are waited for by it
    constexpr uint32_t nocs = get_compile_time_arg_val(s_args.next_compile_time_args_offset());
    constexpr uint32_t q1 = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 1);
    // BALANCE (mp_common.hpp): this core's segments of the items (e, qq)' tile rows split evenly over the `n_items`
    // cores (the runtime arg then: the core count, and the next one this core's index), each item start weighing
    // bal_a rows; 0: the runtime args' items whole
    constexpr bool balance = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 2) != 0;
    constexpr uint32_t bal_a = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t bal_pct = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 4);
    // DN_W2_RUNS: a K row's 4 tiles of a column block (adjacent in its bank: a shard is a column block of every
    // expert) in one request (4,352 bytes; NoC 1 for q1 of every 4 rows) in place of four (q1 of every 4 tiles)
    constexpr bool runs = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 5) != 0;
    // the progressive form (moe_prefill.py PROG): the slots past an expert's base where its correction is (0: none);
    // the item's W2 of the base, then of the correction (`slots` counts both)
    constexpr uint32_t corr = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 6);
    constexpr uint32_t sets = corr ? 2 : 1;

    const auto s_h = TensorAccessor(h_args, h_addr, h_tile);
    const auto s_w2 = TensorAccessor(w_args, w2_addr, w2_tile);
    const auto s_seg = TensorAccessor(s_args, seg_addr, SEG_WORDS * 4);
    const uint32_t seg_l1 = get_write_ptr(DN_SCR);
    noc_async_read(s_seg.get_noc_addr(0), seg_l1, SEG_WORDS * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* seg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seg_l1);
    const uint8_t other = 1 - noc_index;
    uint8_t n2 = noc_index;
    if constexpr (nocs == 2) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(S_NOC1)), 1);
        noc_async_read_set_trid(1, other);
        n2 = other;
    }
    auto barrier = [&]() {
        noc_async_read_barrier();
        if constexpr (nocs == 2) {
            noc_async_read_barrier_with_trid(1, other);
        }
    };

    constexpr uint32_t PER = 32 / DN_Q;  // items an expert
    BalancedWalk walk{};
    if constexpr (balance) {
        walk = balanced_walk(seg, 72, PER, bal_a, get_arg_val<uint32_t>(a), n_items, bal_pct);
    }
    for (uint32_t i = 0; balance || i < n_items; ++i) {
        uint32_t item = 0, r0 = 0, rows = 0;
        if constexpr (balance) {
            if (!next_segment(seg, PER, bal_a, walk, item, r0, rows)) {
                break;
            }
        } else {
            item = get_arg_val<uint32_t>(a + i);
            rows = (seg[SEG_COUNT + item / PER] + 31) / 32;
        }
        const uint32_t e = item / PER, qq = item % PER;
        if (rows == 0) {
            continue;
        }
        cb_reserve_back(DN_W2, sets * DN_Q * DN_K * DN_N);
        const uint32_t w0 = get_write_ptr(DN_W2);
        for (uint32_t jj = 0; jj < sets * DN_Q; ++jj) {
            const uint32_t j = jj % DN_Q, half = jj / DN_Q;
            const uint32_t w = w0 + half * DN_Q * DN_K * DN_N * w2_tile;  // (PROG: the correction's after the base's)
            const uint32_t first = ((DN_Q * qq + j) * slots + e + half * corr) * DN_K;  // W2's tile row of (e, q), K 0
            for (uint32_t k = 0; k < DN_K; ++k) {
                if constexpr (runs) {
                    const uint8_t q = ((j * DN_K + k) & 3) < q1 ? n2 : noc_index;
                    noc_async_read(s_w2.get_noc_addr((first + k) * DN_N, 0, q), w + (j * DN_K + k) * DN_N * w2_tile,
                                   DN_N * w2_tile, q);
                    continue;
                }
                for (uint32_t n = 0; n < DN_N; ++n) {
                    const uint8_t q = (n & 3) < q1 ? n2 : noc_index;
                    noc_async_read(s_w2.get_noc_addr((first + k) * DN_N + n, 0, q),
                                   w + ((j * DN_K + k) * DN_N + n) * w2_tile, w2_tile, q);
                }
            }
        }
        barrier();
        cb_push_back(DN_W2, sets * DN_Q * DN_K * DN_N);
        const uint32_t row0 = seg[e] / 32 + r0;
        for (uint32_t r = 0; r < rows; ++r) {
            cb_reserve_back(DN_HT, DN_K);
            const uint32_t dst = get_write_ptr(DN_HT);
            for (uint32_t k = 0; k < DN_K; ++k) {
                const uint8_t q = (k & 3) < q1 ? n2 : noc_index;
                noc_async_read(s_h.get_noc_addr((row0 + r) * DN_K + k, 0, q), dst + k * h_tile, h_tile, q);
            }
            barrier();
            cb_push_back(DN_HT, DN_K);
        }
    }
    if constexpr (nocs == 2) {
        noc_async_read_set_trid(0, other);
    }
}
