// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// The KDA prefill's decay gate (kda_prefill_ops.py `gate`), the reader: the core's tiles idx in [i0, i1) in column-major
// order (column c = idx / R, tile row r = idx % R), the f tile (r, c) of fg [1, T, F] (bf16), and when the column
// changes its dt_bias and rate tiles c of [1, 32, W] (fp32, row 0 repeated in every row).
void kernel_main() {
    const uint32_t fg_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t rate_addr = get_arg_val<uint32_t>(2);
    const uint32_t i0 = get_arg_val<uint32_t>(3);
    const uint32_t i1 = get_arg_val<uint32_t>(4);

    constexpr uint32_t R = get_compile_time_arg_val(0);         // tile rows (T / 32)
    constexpr uint32_t FG_COLS = get_compile_time_arg_val(1);   // fg's tile columns
    constexpr uint32_t CB_F = get_compile_time_arg_val(2);
    constexpr uint32_t CB_B = get_compile_time_arg_val(3);
    constexpr uint32_t CB_RT = get_compile_time_arg_val(4);
    constexpr auto f_args = TensorAccessorArgs<5>();
    constexpr auto b_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const uint32_t f_tile = get_tile_size(CB_F);
    const uint32_t w_tile = get_tile_size(CB_B);
    const auto s_f = TensorAccessor(f_args, fg_addr, f_tile);
    const auto s_b = TensorAccessor(b_args, bias_addr, w_tile);
    const auto s_r = TensorAccessor(r_args, rate_addr, w_tile);

    uint32_t col = 0xFFFFFFFFu;
    for (uint32_t idx = i0; idx < i1; ++idx) {
        const uint32_t c = idx / R, r = idx % R;
        if (c != col) {
            col = c;
            cb_reserve_back(CB_B, 1);
            cb_reserve_back(CB_RT, 1);
            noc_async_read(s_b.get_noc_addr(c), get_write_ptr(CB_B), w_tile);
            noc_async_read(s_r.get_noc_addr(c), get_write_ptr(CB_RT), w_tile);
            noc_async_read_barrier();
            cb_push_back(CB_B, 1);
            cb_push_back(CB_RT, 1);
        }
        cb_reserve_back(CB_F, 1);
        noc_async_read(s_f.get_noc_addr(r * FG_COLS + c), get_write_ptr(CB_F), f_tile);
        noc_async_read_barrier();
        cb_push_back(CB_F, 1);
    }
}
