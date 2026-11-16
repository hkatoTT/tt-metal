// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dsa_indexer.bfp_cast, the compute: each tile copied through Dest and packed in CB_OUT's format (the program's
// bfp8_pack_precise: the packer converts from Float16_b, not from a BFP8 source, which sets the
// rounding).  RT: the tiles.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
    const uint32_t n = get_arg_val<uint32_t>(0);
    if (n == 0) {
        return;
    }
    compute_kernel_hw_startup(CB_IN, CB_OUT);
    copy_tile_init(CB_IN);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(CB_IN, 1);
        cb_reserve_back(CB_OUT, 1);
        tile_regs_acquire();
        copy_tile(CB_IN, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_IN, 1);
    }
}
