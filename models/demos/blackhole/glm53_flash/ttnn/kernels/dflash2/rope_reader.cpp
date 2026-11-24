// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's head norm weight and RoPE (dflash2_ops.rope), the reader: per item (tile row r, column tile t) of
// [i0, i1), n's tiles t and o = (t + 2) % 4 (bf16), cos and sin tile t (fp32), and the weight's tiles t and o (bf16;
// all 4 once when it has one tile row for all).  halves (the compile-time arg after the key) > 0: n is q's `halves`
// tile rows then k's as many; the weight two tile rows (q's, then k's: all 8 once), cos and sin the halves' (row r %
// halves).
void kernel_main() {
    const uint32_t n_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_arg_val<uint32_t>(2);
    const uint32_t s_addr = get_arg_val<uint32_t>(3);
    const uint32_t i0 = get_arg_val<uint32_t>(4);
    const uint32_t i1 = get_arg_val<uint32_t>(5);
    constexpr uint32_t W_ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t CB_N = get_compile_time_arg_val(1);
    constexpr uint32_t CB_W = get_compile_time_arg_val(2);
    constexpr uint32_t CB_C = get_compile_time_arg_val(3);
    constexpr uint32_t CB_S = get_compile_time_arg_val(4);
    constexpr auto n_args = TensorAccessorArgs<5>();
    constexpr auto w_args = TensorAccessorArgs<n_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t halves = get_compile_time_arg_val(c_args.next_compile_time_args_offset() + 2);
    // GLM53_DFLASH2_CTX_HEADS (dflash2_ops.rope cos_period / w_div; 0: off): the rows' cos / sin tile row r mod
    // COS_PERIOD, their weight tile row r / W_DIV (write_context's K in head-major rows: the positions repeat each head)
    constexpr uint32_t COS_PERIOD = get_compile_time_arg_val(c_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t W_DIV = get_compile_time_arg_val(c_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t BF = 2048, F32 = 4096, WT = 4, W_ONCE = halves ? 2 * WT : WT;
    const auto s_n = TensorAccessor(n_args, n_addr, BF);
    const auto s_w = TensorAccessor(w_args, w_addr, BF);
    const auto s_c = TensorAccessor(c_args, c_addr, F32);
    const auto s_s = TensorAccessor(c_args, s_addr, F32);

    if constexpr (W_ROWS == 1) {
        cb_reserve_back(CB_W, W_ONCE);
        for (uint32_t t = 0; t < W_ONCE; ++t) {
            noc_async_read(s_w.get_noc_addr(t), get_write_ptr(CB_W) + t * BF, BF);
        }
        noc_async_read_barrier();
        cb_push_back(CB_W, W_ONCE);
    }
    for (uint32_t item = i0; item < i1; ++item) {
        const uint32_t r = item / WT, t = item % WT, o = (t + 2) % WT;
        if constexpr (W_ROWS != 1) {
            cb_reserve_back(CB_W, 2);
            uint32_t wr = r;
            if constexpr (W_DIV != 0) {
                wr = r / W_DIV;
            }
            noc_async_read(s_w.get_noc_addr(wr * WT + t), get_write_ptr(CB_W), BF);
            noc_async_read(s_w.get_noc_addr(wr * WT + o), get_write_ptr(CB_W) + BF, BF);
        }
        cb_reserve_back(CB_N, 2);
        noc_async_read(s_n.get_noc_addr(r * WT + t), get_write_ptr(CB_N), BF);
        noc_async_read(s_n.get_noc_addr(r * WT + o), get_write_ptr(CB_N) + BF, BF);
        cb_reserve_back(CB_C, 1);
        uint32_t cr = halves ? r % (halves ? halves : 1) : r;  // (q's and k's rows the same positions)
        if constexpr (COS_PERIOD != 0) {
            cr = r % COS_PERIOD;
        }
        noc_async_read(s_c.get_noc_addr(cr * WT + t), get_write_ptr(CB_C), F32);
        cb_reserve_back(CB_S, 1);
        noc_async_read(s_s.get_noc_addr(cr * WT + t), get_write_ptr(CB_S), F32);
        noc_async_read_barrier();
        if constexpr (W_ROWS != 1) {
            cb_push_back(CB_W, 2);
        }
        cb_push_back(CB_N, 2);
        cb_push_back(CB_C, 1);
        cb_push_back(CB_S, 1);
    }
}
