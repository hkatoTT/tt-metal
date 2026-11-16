// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The column's way back inside the experts op: a reducer core's data movement (NCRISC,
// NoC 0, no DRAM reads, no multicast).  Reducer j holds the tile columns 8 j .. 8 j + 7 of every slice (a block: the
// slice's R tile rows by those 8 columns, B = 8 R tiles), chunk parity j % 2, and runs the ring of
// reduce_scatter_minimal_async for them (rs_compute.cpp does the adds, the same bits):
//   own blocks (cb_own, 8): this chip's partials, written here by the two down cores of these columns as they round
//     them, in the order slices c + 4, c + 3, c + 5, c + 2, c + 6, c + 1, c + 7, c (c: this chip's mesh row); each
//     tile row a write and an increment of this core's mailbox word (R a block from each: the even down core's in the
//     low 16 bits, the odd one's in the high 16; a single count for both let a block through with one core's half);
//   landings (cb_lo from mesh row c - 1, cb_hi from c + 1): the ring's partial sums, written over the fabric by the
//     neighbours' return senders (rs_send.cpp), each block's last packet with an increment of this core's word of the
//     global semaphore rs_sem (low 16 bits cb_lo, high 16 cb_hi);
//   sums (cb_sum, 5) from the compute (a tile pushed to cb_sig for each), and the output block (cb_out).
// This kernel pushes each landed block to its cb for the compute, tells this core's two return senders (link
// (j / 2) % 2, up to c + 1 and down to c - 1) how many of their blocks are ready here (a word in their cb_flags at
// 16 m, an inline write as the in-op gather's mailbox), and writes the output block to this mesh row's output [t, D]
// (tile rows 0 .. R - 1, columns 8 j ..).  It polls every source in turn (no wait on one source holds another back).
// At the start it puts cb_own's address in the down cores' mailbox words; at the end it sets its own two words back to
// 0.  Bounded (WAIT_LIMIT polls without a change, as rs_send.cpp): past it every block still missing is pushed as it is
// (the compute ends; wrong values).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_own = get_compile_time_arg_val(0);
    constexpr uint32_t cb_lo = get_compile_time_arg_val(1);
    constexpr uint32_t cb_hi = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sig = get_compile_time_arg_val(5);
    constexpr uint32_t cb_flags = get_compile_time_arg_val(6);
    constexpr uint32_t R = get_compile_time_arg_val(7);   // tile rows a slice
    constexpr uint32_t wt = get_compile_time_arg_val(8);  // the output's tile columns
    constexpr auto out_args = TensorAccessorArgs<9>();
    constexpr uint32_t W = 8, B = R * W;
    constexpr uint32_t WAIT_LIMIT = 1u << 25;

    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t j = get_arg_val<uint32_t>(a++);
    const uint32_t parity = get_arg_val<uint32_t>(a++);
    const uint32_t rs_sem = get_arg_val<uint32_t>(a++);
    const uint32_t rs_mail = get_arg_val<uint32_t>(a++);
    const uint32_t d0x = get_arg_val<uint32_t>(a++), d0y = get_arg_val<uint32_t>(a++);  // down cores 2 j, 2 j + 1
    const uint32_t d1x = get_arg_val<uint32_t>(a++), d1y = get_arg_val<uint32_t>(a++);
    const uint32_t upx = get_arg_val<uint32_t>(a++), upy = get_arg_val<uint32_t>(a++);  // the return senders
    const uint32_t dnx = get_arg_val<uint32_t>(a++), dny = get_arg_val<uint32_t>(a++);
    const uint32_t m = get_arg_val<uint32_t>(a++);  // this core's slot in their cb_flags
    const auto out = TensorAccessor(out_args, out_addr);

    // the blocks the senders take from here, by stream: even parity up cb_own 0, sums 0, 1, 3; down cb_own 2, sums
    // 2, 4.  Odd: up cb_own 1, sums 2, 4; down cb_own 0, sums 0, 1, 3 (rs_send.cpp's lists)
    const bool even = parity == 0;
    const uint32_t own_up = even ? 0 : 1, own_dn = even ? 2 : 0;
    // sum s goes up (1) or down (0)
    const uint32_t sum_up = even ? 0b01011 : 0b10100;
    const uint32_t n_lo = even ? 4 : 3, n_hi = even ? 3 : 4;

    {  // cb_own's address to the down cores (their mailbox word), before they round
        const uint32_t own_base = get_write_ptr(cb_own);
        noc_inline_dw_write(get_noc_addr(d0x, d0y, rs_mail), own_base);
        noc_inline_dw_write(get_noc_addr(d1x, d1y, rs_mail), own_base);
        noc_async_write_barrier();
    }
    const uint32_t flag = get_write_ptr(cb_flags) + 16 * m;
    const uint64_t up_flag = get_noc_addr(upx, upy, flag), dn_flag = get_noc_addr(dnx, dny, flag);
    uint32_t n_up = 0, n_dn = 0;
    volatile tt_l1_ptr uint32_t* mail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_mail);
    volatile tt_l1_ptr uint32_t* land = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_sem);

    DeviceZoneScopedN("RS-MOVE");
    uint32_t own = 0, lo = 0, hi = 0, sums = 0;
    bool done = false, to = false;
    uint32_t idle = 0, last = 0;
    while (!done) {
        invalidate_l1_cache();
        const uint32_t seen = own + lo + hi + sums;
        if (seen != last) {
            last = seen;
            idle = 0;
        } else if (++idle == WAIT_LIMIT) {
            to = true;  // (everything still missing pushed as it is, below)
        }
        const uint32_t own_rows = to ? (R * 8) << 16 | R * 8 : *mail;
        const uint32_t own_min = (own_rows & 0xffff) < (own_rows >> 16) ? (own_rows & 0xffff) : (own_rows >> 16);
        while (own < 8 && own_min >= R * (own + 1)) {
            cb_reserve_back(cb_own, B);
            cb_push_back(cb_own, B);
            if (own == own_up) {
                noc_inline_dw_write(up_flag, ++n_up);
            }
            if (own == own_dn) {
                noc_inline_dw_write(dn_flag, ++n_dn);
            }
            ++own;
        }
        const uint32_t landed = to ? (n_hi << 16 | n_lo) : *land;
        while (lo < n_lo && (landed & 0xffff) > lo) {
            cb_reserve_back(cb_lo, B);
            cb_push_back(cb_lo, B);
            ++lo;
        }
        while (hi < n_hi && (landed >> 16) > hi) {
            cb_reserve_back(cb_hi, B);
            cb_push_back(cb_hi, B);
            ++hi;
        }
        while (sums < 5 && cb_pages_available_at_front(cb_sig, 1)) {
            cb_wait_front(cb_sig, 1);
            cb_pop_front(cb_sig, 1);
            if ((sum_up >> sums) & 1) {
                noc_inline_dw_write(up_flag, ++n_up);
            } else {
                noc_inline_dw_write(dn_flag, ++n_dn);
            }
            ++sums;
        }
        if (sums == 5 && cb_pages_available_at_front(cb_out, B)) {
            cb_wait_front(cb_out, B);
            uint32_t l1 = get_read_ptr(cb_out);
            const uint32_t tile = get_tile_size(cb_out);
            for (uint32_t r = 0; r < R; ++r) {
                for (uint32_t i = 0; i < W; ++i) {
                    noc_async_write_tile(r * wt + W * j + i, out, l1);
                    l1 += tile;
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, B);
            done = true;
        }
    }
    *mail = 0;
    *land = 0;
    noc_async_write_barrier();
}
