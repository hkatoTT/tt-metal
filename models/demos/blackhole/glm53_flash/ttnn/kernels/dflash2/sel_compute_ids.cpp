// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"

// DFlash2's selector terms with the codebook rows gathered in the op (dflash2_ops.sel_scores_ids), the compute: as
// sel_compute.cpp, the item's predecessor and successor rows first tilized here from the reader's row-major blocks
// (CB_PRR / CB_SUR: 32 rows of rank; the unpacker's tilize, a copy of the bf16 values, as ttnn.embedding's tiles).
// Per item (user, slot), its K tiles of the
// predecessors' codebook rows pr, the spread proj pj and the successors' rows su ([32 rows, 32] each): P_k = pr_k pj_k
// in fp32 (a product of two bf16: exact), hi_k = bf16(P_k), lo_k = bf16(P_k - hi_k) (exact: P_k = hi_k + lo_k), then
// S = sum_k hi_k su_k^T + lo_k su_k^T in fp32 Dest (HiFi4: the bf16 products exact) -> S [32, 32] fp32.
void kernel_main() {
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t i1 = get_arg_val<uint32_t>(1);
    constexpr uint32_t KT = get_compile_time_arg_val(0);  // K tiles (rank / 32)
    constexpr uint32_t CB_PR = get_compile_time_arg_val(1);
    constexpr uint32_t CB_PJ = get_compile_time_arg_val(2);
    constexpr uint32_t CB_SU = get_compile_time_arg_val(3);
    constexpr uint32_t CB_HI = get_compile_time_arg_val(4);
    constexpr uint32_t CB_LO = get_compile_time_arg_val(5);
    constexpr uint32_t CB_O = get_compile_time_arg_val(6);
    constexpr uint32_t parts = get_compile_time_arg_val(7);  // 3: hi and lo (1: hi only, 2: lo only: tests)
    constexpr uint32_t CB_PRR = get_compile_time_arg_val(8);  // the row-major blocks
    constexpr uint32_t CB_SUR = get_compile_time_arg_val(9);

    compute_kernel_hw_startup(CB_PRR, CB_PJ, CB_PR);
    for (uint32_t item = i0; item < i1; ++item) {
        for (uint32_t which = 0; which < 2; ++which) {
            const uint32_t icb = which == 0 ? CB_PRR : CB_SUR, ocb = which == 0 ? CB_PR : CB_SU;
            cb_wait_front(icb, KT);
            cb_reserve_back(ocb, KT);
            reconfig_data_format_srca(icb);
            pack_reconfig_data_format(ocb);
            tilize_init(icb, KT, ocb);
            tilize_block(icb, KT, ocb);
            tilize_uninit(icb, ocb);
            cb_push_back(ocb, KT);
            cb_pop_front(icb, KT);
        }
        cb_wait_front(CB_PR, KT);
        cb_wait_front(CB_PJ, KT);
        cb_reserve_back(CB_HI, KT);
        cb_reserve_back(CB_LO, KT);
        for (uint32_t k = 0; k < KT; ++k) {
            // hi
            tile_regs_acquire();
            reconfig_data_format_srca(CB_PR);
            copy_init(CB_PR);
            copy_tile(CB_PR, k, 0);
            reconfig_data_format_srca(CB_PJ);
            copy_init(CB_PJ);
            copy_tile(CB_PJ, k, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
            typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_HI);
            pack_tile<true>(0, CB_HI, k);
            tile_regs_release();
        }
        cb_push_back(CB_HI, KT);
        cb_wait_front(CB_HI, KT);
        for (uint32_t k = 0; k < KT; ++k) {
            // lo = bf16(P - hi)
            tile_regs_acquire();
            reconfig_data_format_srca(CB_PR);
            copy_init(CB_PR);
            copy_tile(CB_PR, k, 0);
            reconfig_data_format_srca(CB_PJ);
            copy_init(CB_PJ);
            copy_tile(CB_PJ, k, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            reconfig_data_format_srca(CB_HI);
            copy_init(CB_HI);
            copy_tile(CB_HI, k, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
            typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_LO);
            pack_tile<true>(0, CB_LO, k);
            tile_regs_release();
        }
        cb_push_back(CB_LO, KT);
        cb_pop_front(CB_PR, KT);
        cb_pop_front(CB_PJ, KT);
        cb_wait_front(CB_LO, KT);
        cb_wait_front(CB_SU, KT);
        // S = sum_k hi_k su_k^T + lo_k su_k^T
        tile_regs_acquire();
        if constexpr ((parts & 1) != 0) {
            reconfig_data_format(CB_SU, CB_HI);
            matmul_init(CB_HI, CB_SU, 1);
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_tiles(CB_HI, CB_SU, k, k, 0);
            }
        }
        if constexpr ((parts & 2) != 0) {
            reconfig_data_format(CB_SU, CB_LO);
            matmul_init(CB_LO, CB_SU, 1);
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_tiles(CB_LO, CB_SU, k, k, 0);
            }
        }
        tile_regs_commit();
        cb_reserve_back(CB_O, 1);
        tile_regs_wait();
        pack_reconfig_data_format(CB_O);
        pack_tile(0, CB_O);
        tile_regs_release();
        cb_push_back(CB_O, 1);
        cb_pop_front(CB_HI, KT);
        cb_pop_front(CB_LO, KT);
        cb_pop_front(CB_SU, KT);
    }
}
