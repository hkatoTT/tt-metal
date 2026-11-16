// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "mp_common.hpp"

// The prefill MoE's combine, the compute (mp_common.hpp): per item (its slots' count from the writer's mailbox), all
// its slots' blocks in CB_S, then each flat piece: slot 0's into Dest (unpacked to Dest: exact fp32), each further
// slot's added (SFPU, fp32, in slot order), packed bf16 to CB_O (row-major), then tilized to CB_T; zeros when no
// token of the row has a pair.
void kernel_main() {
    using namespace moe_prefill;
    const uint32_t n_items = get_arg_val<uint32_t>(0);
    // (the program cache key N at 0), then the pieces a slot: 16, or 32 (Y_BFP8: a token row of 1,024 a piece)
    constexpr uint32_t PIECES = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(CB_S, CB_O);
    for (uint32_t i = 0; i < n_items; ++i) {
        uint32_t slots = 0;
        UNPACK(slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        cb_reserve_back(CB_O, PIECES);
        cb_wait_front(CB_S, K * PIECES);  // the whole ring (the reader's), its first `slots` blocks used
        reconfig_data_format_srca(CB_S);
        pack_reconfig_data_format(CB_O);
        for (uint32_t k = 0; k < PIECES; ++k) {
            tile_regs_acquire();
            if (slots == 0) {
                fill_tile_init();
                fill_tile(0, 0.0f);
            } else {
                copy_tile_to_dst_init_short(CB_S);
                copy_tile(CB_S, k, 0);
                for (uint32_t j = 1; j < slots; ++j) {
                    copy_tile_to_dst_init_short(CB_S);
                    copy_tile(CB_S, j * PIECES + k, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_O);
            tile_regs_release();
        }
        cb_push_back(CB_O, PIECES);
        cb_pop_front(CB_S, K * PIECES);
        compute_kernel_lib::tilize<PIECES, CB_O, CB_T>(1);
    }
}
