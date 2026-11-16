// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The split sparse attention's combine (dsa_indexer.sparse_combine), the compute: for a unit's K partials (the
// splits with keys) and the reader's row maxima M: w_k = exp((m_k - M) scale), L = sum_k w_k l_k, then each of its C
// out tiles sum_k o_k (w_k / L) (per row: column 0 of the m, l, w tiles; o's rows broadcast).  The sums are packed
// with the packer's L1 accumulation into fp32 tiles (no SFPU op between two FPU ops in Dest: with G = 8 that form went
// wrong), the result packed once to bf16.
//   CT: C, SCALE (fp32 bits), CB_M, CB_L, CB_O, CB_CTRL, CB_MAX, CB_W, CB_OUT, CB_W2, CB_SUM (fp32), CB_RECIP,
//       CB_ACC (fp32).  RT: 1 (a unit) or 0 (idle).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t C = get_compile_time_arg_val(0);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_m = get_compile_time_arg_val(2);
    constexpr uint32_t cb_l = get_compile_time_arg_val(3);
    constexpr uint32_t cb_o = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(5);
    constexpr uint32_t cb_max = get_compile_time_arg_val(6);
    constexpr uint32_t cb_w = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t cb_w2 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(10);
    constexpr uint32_t cb_recip = get_compile_time_arg_val(11);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(12);
    const uint32_t n_units = get_arg_val<uint32_t>(0);  // this core's units, each in turn (0: idle)
    if (n_units == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_m, cb_max, cb_w);
    for (uint32_t unit = 0; unit < n_units; ++unit) {

    cb_wait_front(cb_ctrl, 1);
    const uint32_t k_count = ckernel::read_tile_value(cb_ctrl, 0, 0);
    cb_pop_front(cb_ctrl, 1);
    cb_wait_front(cb_m, k_count);
    cb_wait_front(cb_l, k_count);
    cb_wait_front(cb_max, 1);

    // w_k = exp((m_k - M) scale)
    cb_reserve_back(cb_w, k_count);
    reconfig_data_format(cb_m, cb_max);
    pack_reconfig_data_format(cb_w);
    for (uint32_t k = 0; k < k_count; ++k) {
        tile_regs_acquire();
        sub_init(cb_m, cb_max);
        sub_tiles(cb_m, cb_max, k, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_fp32);
        exp_tile_init<false>();
        exp_tile<false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_w, k);
        tile_regs_release();
    }
    cb_push_back(cb_w, k_count);
    cb_wait_front(cb_w, k_count);
    cb_pop_front(cb_max, 1);
    cb_pop_front(cb_m, k_count);

    // L = sum_k w_k l_k (fp32, the packer's L1 accumulation), then 1 / L
    cb_reserve_back(cb_sum, 1);
    reconfig_data_format(cb_l, cb_w);
    pack_reconfig_data_format(cb_sum);
    for (uint32_t k = 0; k < k_count; ++k) {
        tile_regs_acquire();
        mul_init(cb_l, cb_w);
        mul_tiles(cb_l, cb_w, k, k, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_l1_acc(k > 0);
        pack_tile<true>(0, cb_sum, 0);
        tile_regs_release();
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_sum, 1);
    cb_pop_front(cb_l, k_count);
    cb_wait_front(cb_sum, 1);
    cb_reserve_back(cb_recip, 1);
    reconfig_data_format_srca(cb_sum);
    pack_reconfig_data_format(cb_recip);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_sum);
    copy_tile(cb_sum, 0, 0);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_recip);
    tile_regs_release();
    cb_push_back(cb_recip, 1);
    cb_pop_front(cb_sum, 1);
    cb_wait_front(cb_recip, 1);

    // w2_k = w_k / L
    cb_reserve_back(cb_w2, k_count);
    reconfig_data_format(cb_w, cb_recip);
    pack_reconfig_data_format(cb_w2);
    for (uint32_t k = 0; k < k_count; ++k) {
        tile_regs_acquire();
        mul_init(cb_w, cb_recip);
        mul_tiles(cb_w, cb_recip, k, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_w2, k);
        tile_regs_release();
    }
    cb_push_back(cb_w2, k_count);
    cb_wait_front(cb_w2, k_count);
    cb_pop_front(cb_w, k_count);
    cb_pop_front(cb_recip, 1);

    // the out tiles: sum_k o_k[c] w2_k into an fp32 tile, then to bf16
    cb_wait_front(cb_o, k_count * C);
    for (uint32_t c = 0; c < C; ++c) {
        cb_reserve_back(cb_acc, 1);
        reconfig_data_format(cb_o, cb_w2);
        pack_reconfig_data_format(cb_acc);
        for (uint32_t k = 0; k < k_count; ++k) {
            tile_regs_acquire();
            mul_bcast_cols_init(cb_o, cb_w2);
            mul_tiles_bcast_cols(cb_o, cb_w2, k * C + c, k, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(k > 0);
            pack_tile<true>(0, cb_acc, 0);
            tile_regs_release();
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(cb_acc, 1);
        cb_wait_front(cb_acc, 1);
        cb_reserve_back(cb_out, 1);
        reconfig_data_format_srca(cb_acc);
        pack_reconfig_data_format(cb_out);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_acc);
        copy_tile(cb_acc, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_acc, 1);
    }
    cb_pop_front(cb_o, k_count * C);
    cb_pop_front(cb_w2, k_count);
    }  // units
}
