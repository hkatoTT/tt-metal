// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// DFlash2's head norm weight and RoPE (dflash2_ops.rope), the compute: per item (tile row, column tile t), with o the
// tile of the other half (rotate_half: [-x2, x1]), y_t = bf16(w_t n_t), y_o = bf16(w_o n_o) (the reference's weight
// times the bf16 normalized value), out_t = bf16(y_t cos_t - y_o sin_t) (t < 2) or + (t >= 2), in fp32 on the SFPU
// (the reference's fp32 RoPE), RNE.
void kernel_main() {
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t i1 = get_arg_val<uint32_t>(1);
    constexpr uint32_t W_ROWS = get_compile_time_arg_val(0);  // 1: one weight tile row for all (4 tiles); else 2 an item
    constexpr uint32_t CB_N = get_compile_time_arg_val(1);
    constexpr uint32_t CB_W = get_compile_time_arg_val(2);
    constexpr uint32_t CB_C = get_compile_time_arg_val(3);
    constexpr uint32_t CB_S = get_compile_time_arg_val(4);
    constexpr uint32_t CB_O = get_compile_time_arg_val(5);
    // > 0: q's tile rows then k's as many, the weight's 8 tiles q's then k's (the key's 2 compile-time args before it)
    constexpr uint32_t halves = get_compile_time_arg_val(8);
    constexpr uint32_t WT = 4;

    compute_kernel_hw_startup(CB_N, CB_W, CB_O);
    if constexpr (W_ROWS == 1) {
        cb_wait_front(CB_W, halves ? 2 * WT : WT);
    }
    auto y = [&](uint32_t n_idx, uint32_t w_idx, uint32_t dst, uint32_t tmp) {
        reconfig_data_format_srca(CB_N);
        copy_init(CB_N);
        copy_tile(CB_N, n_idx, dst);
        reconfig_data_format_srca(CB_W);
        copy_init(CB_W);
        copy_tile(CB_W, w_idx, tmp);
        mul_binary_tile_init();
        mul_binary_tile(dst, tmp, dst);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(dst);
    };
    for (uint32_t item = i0; item < i1; ++item) {
        const uint32_t t = item % WT, o = (t + 2) % WT;
        const uint32_t wb = halves && item / WT >= halves ? WT : 0;  // k's weight tiles
        if constexpr (W_ROWS != 1) {
            cb_wait_front(CB_W, 2);
        }
        cb_wait_front(CB_N, 2);
        cb_wait_front(CB_C, 1);
        cb_wait_front(CB_S, 1);
        tile_regs_acquire();
        y(0, W_ROWS == 1 ? wb + t : 0, 0, 3);
        y(1, W_ROWS == 1 ? wb + o : 1, 1, 3);
        reconfig_data_format_srca(CB_C);
        copy_init(CB_C);
        copy_tile(CB_C, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);
        reconfig_data_format_srca(CB_S);
        copy_init(CB_S);
        copy_tile(CB_S, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        if (t < 2) {
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
        } else {
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        }
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        tile_regs_commit();
        cb_reserve_back(CB_O, 1);
        tile_regs_wait();
        pack_tile(0, CB_O);
        tile_regs_release();
        cb_push_back(CB_O, 1);
        cb_pop_front(CB_N, 2);
        cb_pop_front(CB_C, 1);
        cb_pop_front(CB_S, 1);
        if constexpr (W_ROWS != 1) {
            cb_pop_front(CB_W, 2);
        }
    }
    if constexpr (W_ROWS == 1) {
        cb_pop_front(CB_W, WT);
    }
}
