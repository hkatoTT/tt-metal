// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"

// DFlash2's MLP gate (dflash2_ops.silu_mul), the compute: per output tile SiLU(gate) packed to bf16 (CB_S), then
// CB_S times up on the SFPU (bf16 Dest: the product rounded to nearest even) -> bf16, as ttnn.multiply(gate, up,
// input_tensor_a_activations=[SILU]) does it for bf16 (binary_ng's SFPU multiply: its LHS pass, the activation in
// Dest packed in the input's format, then both operands copied to Dest and multiplied there).
void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t t1 = get_arg_val<uint32_t>(1);
    constexpr uint32_t CB_G = get_compile_time_arg_val(0);
    constexpr uint32_t CB_U = get_compile_time_arg_val(1);
    constexpr uint32_t CB_S = get_compile_time_arg_val(2);
    constexpr uint32_t CB_O = get_compile_time_arg_val(3);
    compute_kernel_hw_startup(CB_G, CB_U, CB_O);
    for (uint32_t t = t0; t < t1; ++t) {
        cb_wait_front(CB_G, 1);
        cb_reserve_back(CB_S, 1);
        tile_regs_acquire();
        copy_init(CB_G);
        copy_tile(CB_G, 0, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_S);
        tile_regs_release();
        cb_push_back(CB_S, 1);
        cb_pop_front(CB_G, 1);

        cb_wait_front(CB_S, 1);
        cb_wait_front(CB_U, 1);
        cb_reserve_back(CB_O, 1);
        tile_regs_acquire();
        copy_init(CB_S);
        copy_tile(CB_S, 0, 0);
        copy_init(CB_U);
        copy_tile(CB_U, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_O);
        tile_regs_release();
        cb_push_back(CB_O, 1);
        cb_pop_front(CB_S, 1);
        cb_pop_front(CB_U, 1);
    }
}
