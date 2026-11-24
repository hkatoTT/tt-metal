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

// The KDA prefill's decay gate, the compute: per tile, bf16(sigmoid((fp32(f) + dt_bias) * rate)) with the same SFPU
// steps and rounding as the ttnn ops it replaces (typecast to fp32, binary_ng add and multiply on fp32, sigmoid
// (accurate) on fp32, typecast to bf16): f unpacked to Dest (bf16 -> fp32, exact), dt_bias and rate unpacked to Dest
// as fp32 (UnpackToDestFp32), add_binary_tile, mul_binary_tile, sigmoid_tile, typecast_tile fp32 -> bf16 (RNE), packed
// bf16.  dt_bias and rate stay in their buffers while the column lasts (the reader's order).
void kernel_main() {
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t i1 = get_arg_val<uint32_t>(1);
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t CB_F = get_compile_time_arg_val(1);
    constexpr uint32_t CB_B = get_compile_time_arg_val(2);
    constexpr uint32_t CB_RT = get_compile_time_arg_val(3);
    constexpr uint32_t CB_O = get_compile_time_arg_val(4);

    compute_kernel_hw_startup(CB_F, CB_O);
    uint32_t col = 0xFFFFFFFFu;
    for (uint32_t idx = i0; idx < i1; ++idx) {
        const uint32_t c = idx / R;
        if (c != col) {
            if (col != 0xFFFFFFFFu) {
                cb_pop_front(CB_B, 1);
                cb_pop_front(CB_RT, 1);
            }
            col = c;
            cb_wait_front(CB_B, 1);
            cb_wait_front(CB_RT, 1);
        }
        cb_wait_front(CB_F, 1);
        cb_reserve_back(CB_O, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_F);
        copy_tile_to_dst_init_short(CB_F);
        copy_tile(CB_F, 0, 0);
        reconfig_data_format_srca(CB_F, CB_B);
        copy_tile_to_dst_init_short(CB_B);
        copy_tile(CB_B, 0, 1);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        copy_tile_to_dst_init_short(CB_RT);
        copy_tile(CB_RT, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(0);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_O);
        tile_regs_release();
        cb_push_back(CB_O, 1);
        cb_pop_front(CB_F, 1);
    }
    if (col != 0xFFFFFFFFu) {
        cb_pop_front(CB_B, 1);
        cb_pop_front(CB_RT, 1);
    }
}
