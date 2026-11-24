// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rope_kv_items.hpp"

// DFlash2's RoPE with the block's K / V into the caches (dflash2_ops.rope_kv), the
// reader: first the weight's 8 tiles once (q's tile row, then k's), then per item of [i0, i1):
//   rope items 0 .. 8 H - 1 (H = halves: q's H tile rows then k's, 4 column tiles each; item = row 4 + t): n's tiles t
//   and o = (t + 2) % 4 (bf16), cos and sin tile t of row r % H (fp32), as rope_reader.cpp;
//   v items 8 H .. 8 H + nv - 1 (vi = item - 8 H: tile row R = vi / (4 heads), head h = vi / 4 % heads, column tile t =
//   vi % 4): qkv's tile (R, v0t + 4 h + t) (bf16).
void kernel_main() {
    const uint32_t n_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_arg_val<uint32_t>(2);
    const uint32_t s_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t i0 = get_arg_val<uint32_t>(5);
    const uint32_t i1 = get_arg_val<uint32_t>(6);
    constexpr uint32_t CB_N = get_compile_time_arg_val(0);
    constexpr uint32_t CB_W = get_compile_time_arg_val(1);
    constexpr uint32_t CB_C = get_compile_time_arg_val(2);
    constexpr uint32_t CB_S = get_compile_time_arg_val(3);
    constexpr uint32_t CB_V = get_compile_time_arg_val(4);
    constexpr uint32_t H = get_compile_time_arg_val(5);       // halves: q's tile rows (= k's)
    constexpr uint32_t heads = get_compile_time_arg_val(6);   // K / V heads a chip
    constexpr uint32_t src_wt = get_compile_time_arg_val(7);  // qkv's column tiles
    constexpr uint32_t v0t = get_compile_time_arg_val(8);     // v's first column tile in qkv
    constexpr auto n_args = TensorAccessorArgs<9>();
    constexpr auto w_args = TensorAccessorArgs<n_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048, F32 = 4096, WT = 4, W_ONCE = 2 * WT, ROPE = 8 * H;
    const auto s_n = TensorAccessor(n_args, n_addr, BF);
    const auto s_w = TensorAccessor(w_args, w_addr, BF);
    const auto s_c = TensorAccessor(c_args, c_addr, F32);
    const auto s_s = TensorAccessor(c_args, s_addr, F32);
    const auto s_v = TensorAccessor(v_args, v_addr, BF);

    if (rkv_has_rope<ROPE>(i0, i1)) {  // (the weight only where there are rope items)
        cb_reserve_back(CB_W, W_ONCE);
        for (uint32_t t = 0; t < W_ONCE; ++t) {
            noc_async_read(s_w.get_noc_addr(t), get_write_ptr(CB_W) + t * BF, BF);
        }
        noc_async_read_barrier();
        cb_push_back(CB_W, W_ONCE);
    }
    for (uint32_t phys = i0; phys < i1; ++phys) {
        const uint32_t item = rkv_logical<ROPE>(phys);
        if (item < ROPE) {
            const uint32_t r = item / WT, t = item % WT, o = (t + 2) % WT;
            cb_reserve_back(CB_N, 2);
            noc_async_read(s_n.get_noc_addr(r * WT + t), get_write_ptr(CB_N), BF);
            noc_async_read(s_n.get_noc_addr(r * WT + o), get_write_ptr(CB_N) + BF, BF);
            cb_reserve_back(CB_C, 1);
            const uint32_t cr = r % H;  // (q's and k's rows the same positions)
            noc_async_read(s_c.get_noc_addr(cr * WT + t), get_write_ptr(CB_C), F32);
            cb_reserve_back(CB_S, 1);
            noc_async_read(s_s.get_noc_addr(cr * WT + t), get_write_ptr(CB_S), F32);
            noc_async_read_barrier();
            cb_push_back(CB_N, 2);
            cb_push_back(CB_C, 1);
            cb_push_back(CB_S, 1);
        } else {
            const uint32_t vi = item - ROPE;
            const uint32_t R = vi / (WT * heads), h = (vi / WT) % heads, t = vi % WT;
            cb_reserve_back(CB_V, 1);
            noc_async_read(s_v.get_noc_addr(R * src_wt + v0t + h * WT + t), get_write_ptr(CB_V), BF);
            noc_async_read_barrier();
            cb_push_back(CB_V, 1);
        }
    }
}
