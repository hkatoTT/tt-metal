// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "rope_kv_items.hpp"

// DFlash2's RoPE with the block's K / V into the caches (dflash2_ops.rope_kv), the
// compute.  Rope items (q's 4 H, then k's 4 H): rope_compute.cpp's arithmetic as it is (y = bf16(w n) per
// half, bf16(y_t cos -+ y_o sin) in fp32 on the SFPU, RNE), q's tile packed to CB_O (bf16) as there; k's tile, its
// values bf16 by that last typecast, packed to CB_KV in the cache's BFP8_b (the program's bfp8_pack_precise: the
// packer converts the values themselves, as dsa_indexer.bfp_cast does from their bf16 tile).  V items: qkv's v tile
// copied through Dest and packed to CB_KV in BFP8_b (bfp_cast_compute.cpp's steps).
void kernel_main() {
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t i1 = get_arg_val<uint32_t>(1);
    constexpr uint32_t CB_N = get_compile_time_arg_val(0);
    constexpr uint32_t CB_W = get_compile_time_arg_val(1);
    constexpr uint32_t CB_C = get_compile_time_arg_val(2);
    constexpr uint32_t CB_S = get_compile_time_arg_val(3);
    constexpr uint32_t CB_V = get_compile_time_arg_val(4);
    constexpr uint32_t CB_O = get_compile_time_arg_val(5);
    constexpr uint32_t CB_KV = get_compile_time_arg_val(6);
    constexpr uint32_t H = get_compile_time_arg_val(7);
    constexpr uint32_t WT = 4, ROPE = 8 * H, QITEMS = 4 * H;

    compute_kernel_hw_startup(CB_N, CB_W, CB_O);
    const bool rope_here = rkv_has_rope<ROPE>(i0, i1);
    if (rope_here) {
        cb_wait_front(CB_W, 2 * WT);
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
    for (uint32_t phys = i0; phys < i1; ++phys) {
        const uint32_t item = rkv_logical<ROPE>(phys);
        if (item < ROPE) {
            const uint32_t t = item % WT, o = (t + 2) % WT;
            const uint32_t wb = item >= QITEMS ? WT : 0;  // k's weight tiles
            cb_wait_front(CB_N, 2);
            cb_wait_front(CB_C, 1);
            cb_wait_front(CB_S, 1);
            tile_regs_acquire();
            y(0, wb + t, 0, 3);
            y(1, wb + o, 1, 3);
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
            const uint32_t to = item < QITEMS ? CB_O : CB_KV;
            if (to == CB_KV) {
                pack_reconfig_data_format(CB_O, CB_KV);
            }
            cb_reserve_back(to, 1);
            tile_regs_wait();
            pack_tile(0, to);
            tile_regs_release();
            cb_push_back(to, 1);
            if (to == CB_KV) {
                pack_reconfig_data_format(CB_KV, CB_O);
            }
            cb_pop_front(CB_N, 2);
            cb_pop_front(CB_C, 1);
            cb_pop_front(CB_S, 1);
        } else {
            cb_wait_front(CB_V, 1);
            reconfig_data_format_srca(CB_V);
            copy_init(CB_V);
            tile_regs_acquire();
            copy_tile(CB_V, 0, 0);
            tile_regs_commit();
            pack_reconfig_data_format(CB_O, CB_KV);
            cb_reserve_back(CB_KV, 1);
            tile_regs_wait();
            pack_tile(0, CB_KV);
            tile_regs_release();
            cb_push_back(CB_KV, 1);
            pack_reconfig_data_format(CB_KV, CB_O);
            cb_pop_front(CB_V, 1);
        }
    }
    if (rope_here) {
        cb_pop_front(CB_W, 2 * WT);
    }
}
