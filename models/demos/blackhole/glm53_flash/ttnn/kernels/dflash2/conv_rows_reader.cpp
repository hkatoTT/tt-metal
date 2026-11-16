// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's dynamic convolution (dflash2_ops.conv, the rows form), the reader: the core's column tiles [j0, j1) tile row
// by tile row, each row pushed as soon as it is in so that the compute works on it while the next rows are read (two
// rows in flight, transaction ids 1 and 2).  First, once: per j its expansion tile E_{j % 16} and the side's two base
// tiles (fp32, the row repeated), and the shift tile S.  Then per tile row i: the two delta tiles (taps 0 and 1: tile
// columns 16 side + 8 t + g) of each of the range's delta groups g = j / 16, once a row (not once a column tile), and
// the x tiles (i, j).  The compute indexes them (row i: CB_D 2 (i ng + g - g0) + t, CB_X i nj + j - j0) and pops them
// at its end.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t d_addr = get_arg_val<uint32_t>(1);
    const uint32_t base_addr = get_arg_val<uint32_t>(2);
    const uint32_t e_addr = get_arg_val<uint32_t>(3);
    const uint32_t s_addr = get_arg_val<uint32_t>(4);
    const uint32_t j0 = get_arg_val<uint32_t>(5);
    const uint32_t j1 = get_arg_val<uint32_t>(6);

    constexpr uint32_t R = get_compile_time_arg_val(0);   // tile rows
    constexpr uint32_t XW = get_compile_time_arg_val(1);  // x's column tiles (128)
    constexpr uint32_t DW = get_compile_time_arg_val(2);  // delta's column tiles (32)
    constexpr uint32_t side = get_compile_time_arg_val(3);
    constexpr uint32_t CB_X = get_compile_time_arg_val(4);
    constexpr uint32_t CB_D = get_compile_time_arg_val(5);
    constexpr uint32_t CB_E = get_compile_time_arg_val(6);
    constexpr uint32_t CB_S = get_compile_time_arg_val(7);
    constexpr uint32_t CB_B = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto d_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048, F32 = 4096;

    const auto s_x = TensorAccessor(x_args, x_addr, BF);
    const auto s_d = TensorAccessor(d_args, d_addr, BF);
    const auto s_b = TensorAccessor(b_args, base_addr, F32);
    const auto s_e = TensorAccessor(e_args, e_addr, BF);
    const auto s_s = TensorAccessor(s_args, s_addr, BF);
    const uint32_t nj = j1 - j0;
    const uint32_t g0 = j0 / 16, ng = (j1 - 1) / 16 - g0 + 1;  // the range's delta groups

    cb_reserve_back(CB_S, 1);
    cb_reserve_back(CB_E, nj);
    cb_reserve_back(CB_B, 2 * nj);
    noc_async_read(s_s.get_noc_addr(0), get_write_ptr(CB_S), BF);
    for (uint32_t j = j0; j < j1; ++j) {
        const uint32_t a = j - j0;
        noc_async_read(s_e.get_noc_addr(j % 16), get_write_ptr(CB_E) + a * BF, BF);
        noc_async_read(s_b.get_noc_addr(j), get_write_ptr(CB_B) + (2 * a) * F32, F32);           // tap 0
        noc_async_read(s_b.get_noc_addr(XW + j), get_write_ptr(CB_B) + (2 * a + 1) * F32, F32);  // tap 1
    }
    noc_async_read_barrier();
    cb_push_back(CB_S, 1);
    cb_push_back(CB_E, nj);
    cb_push_back(CB_B, 2 * nj);

    // (transaction ids from a clean state: earlier programs' tagged reads can leave their counters non-zero)
    noc_async_read_set_trid(0, noc_index);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
    cb_reserve_back(CB_D, 2 * ng * R);
    cb_reserve_back(CB_X, nj * R);
    const uint32_t d0 = get_write_ptr(CB_D), x0 = get_write_ptr(CB_X);
    for (uint32_t i = 0; i <= R; ++i) {
        if (i < R) {  // row i's reads under id 1 + i % 2
            noc_async_read_set_trid(1 + (i & 1), noc_index);
            for (uint32_t g = 0; g < ng; ++g) {
                const uint32_t at = 2 * (i * ng + g);
                noc_async_read(s_d.get_noc_addr(i * DW + side * 16 + g0 + g), d0 + at * BF, BF);
                noc_async_read(s_d.get_noc_addr(i * DW + side * 16 + 8 + g0 + g), d0 + (at + 1) * BF, BF);
            }
            for (uint32_t j = j0; j < j1; ++j) {
                noc_async_read(s_x.get_noc_addr(i * XW + j), x0 + (i * nj + j - j0) * BF, BF);
            }
        }
        if (i > 0) {  // row i - 1 in: to the compute
            noc_async_read_barrier_with_trid(1 + ((i - 1) & 1), noc_index);
            cb_push_back(CB_D, 2 * ng);
            cb_push_back(CB_X, nj);
        }
    }
    noc_async_read_set_trid(0, noc_index);
}
