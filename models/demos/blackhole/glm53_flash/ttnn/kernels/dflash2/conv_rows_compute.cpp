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
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// DFlash2's dynamic convolution (dflash2_ops.conv, the rows form), the compute: per output tile (i, j), in fp32 Dest,
//   c0 = D0 E_p + base0, c1 = D1 E_p + base1 (E_p: the delta's two group columns of j spread over their 16 channels
//   each; one product a value: exact), xs = S x (the row before in the block, zero at a block's start; exact),
//   out = bf16(c0 x + c1 xs) (SFPU fp32 adds and multiplies, as the reference's fp32 math; one rounding, RNE).
// The reader's tiles come a tile row at a time (conv_rows_reader.cpp: indexed, popped at the end); the tiles (i, j)
// tile row by tile row, as the arithmetic of conv_compute.cpp (the same values).
void kernel_main() {
    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t j1 = get_arg_val<uint32_t>(1);
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t CB_X = get_compile_time_arg_val(1);
    constexpr uint32_t CB_D = get_compile_time_arg_val(2);
    constexpr uint32_t CB_E = get_compile_time_arg_val(3);
    constexpr uint32_t CB_S = get_compile_time_arg_val(4);
    constexpr uint32_t CB_B = get_compile_time_arg_val(5);
    constexpr uint32_t CB_O = get_compile_time_arg_val(6);
    const uint32_t nj = j1 - j0;

    compute_kernel_hw_startup(CB_D, CB_E, CB_O);
    cb_wait_front(CB_S, 1);
    cb_wait_front(CB_E, nj);
    cb_wait_front(CB_B, 2 * nj);
    const uint32_t g0 = j0 / 16, ng = (j1 - 1) / 16 - g0 + 1;  // the range's delta groups (a pair a row each)
    for (uint32_t i = 0; i < R; ++i) {
        cb_wait_front(CB_D, 2 * ng * (i + 1));
        cb_wait_front(CB_X, nj * (i + 1));
        for (uint32_t a = 0; a < nj; ++a) {
            const uint32_t at = i * nj + a;
            const uint32_t dt = 2 * (i * ng + (j0 + a) / 16 - g0);
            tile_regs_acquire();
            // the products (srcA = in1, srcB = in0)
            reconfig_data_format(CB_E, CB_D);
            matmul_init(CB_D, CB_E, 0);
            matmul_tiles(CB_D, CB_E, dt, a, 0);
            matmul_tiles(CB_D, CB_E, dt + 1, a, 1);
            reconfig_data_format(CB_X, CB_S);
            matmul_init(CB_S, CB_X, 0);
            matmul_tiles(CB_S, CB_X, 0, at, 2);
            // the coefficients
            reconfig_data_format_srca(CB_B);
            copy_init(CB_B);
            copy_tile(CB_B, 2 * a, 3);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 3, 0);
            copy_init(CB_B);
            copy_tile(CB_B, 2 * a + 1, 3);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(1, 3, 1);
            // c0 x + c1 xs
            reconfig_data_format_srca(CB_X);
            copy_init(CB_X);
            copy_tile(CB_X, at, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 3, 0);
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
            typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
            tile_regs_commit();
            cb_reserve_back(CB_O, 1);
            tile_regs_wait();
            pack_tile(0, CB_O);
            tile_regs_release();
            cb_push_back(CB_O, 1);
        }
    }
    cb_pop_front(CB_S, 1);
    cb_pop_front(CB_E, nj);
    cb_pop_front(CB_B, 2 * nj);
    cb_pop_front(CB_D, 2 * ng * R);
    cb_pop_front(CB_X, nj * R);
}
