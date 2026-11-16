// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router's matmul (ttnn/router_mm.py) on one core: its output tile (a tile row of tokens, a column tile of 32
// experts) = x [1 x K tiles] @ w [K x 1], with ttnn.matmul's arithmetic for the router's program config (moe.py
// `_router_config`: per core 1 x 1 tile, in0_block_w 16, HiFi4, fp32 Dest, packer L1 accumulation; the factory's
// bmm_large_block_zm_fused_bias_activation.cpp): each block of 16 K tiles summed in Dest from zero by matmul_block,
// packed to the partials (block 0 plain, blocks 1 .. NB - 2 added in L1 by the packer, fp32), and the last block on
// the partials reloaded into Dest (unpacked to Dest as fp32), its 16 tiles added, packed to the output.  CB_X holds
// the whole tile row of x (pushed in chunks by the x kernel, never popped until the end), CB_W the weight blocks.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

namespace {
constexpr uint32_t CB_X = 0, CB_W = 1, CB_OUT = 16, CB_PART = 24;
}

void kernel_main() {
    constexpr uint32_t KB = get_compile_time_arg_val(0);  // in0_block_w (16)
    constexpr uint32_t NB = get_compile_time_arg_val(1);  // K blocks (K / 16)
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_W, CB_PART);
    matmul_block_init(CB_X, CB_W, 0, 1, 1, KB);
    bool reload = false;
    for (uint32_t b = 0; b < NB; ++b) {
        const bool last = b == NB - 1;
        cb_wait_front(CB_X, (b + 1) * KB);
        cb_wait_front(CB_W, KB);
        tile_regs_acquire();
        if (reload) {  // (ttnn's reload_from_cb_to_dst)
            reconfig_data_format_srca(CB_W, CB_PART);
            copy_init(CB_PART);
            cb_wait_front(CB_PART, 1);
            copy_block(CB_PART, 0, 0, 1);
            cb_pop_front(CB_PART, 1);
            reconfig_data_format_srca(CB_PART, CB_W);
            matmul_block_init(CB_X, CB_W, 0, 1, 1, KB);
        }
        for (uint32_t i = 0; i < KB; ++i) {
            matmul_block(CB_X, CB_W, b * KB + i, i, 0, 0, 1, 1, KB);
        }
        if (last) {
            tile_regs_commit();
            cb_reserve_back(CB_OUT, 1);
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUT);
            pack_reconfig_l1_acc(0);
            pack_tile(0, CB_OUT);
            tile_regs_release();
            cb_push_back(CB_OUT, 1);
        } else {
            tile_regs_commit();
            cb_reserve_back(CB_PART, 1);
            tile_regs_wait();
            if (b == 0) {
                pack_reconfig_l1_acc(0);
            } else if (b == 1) {
                pack_reconfig_l1_acc(1);
            }
            pack_tile(0, CB_PART);
            tile_regs_release();
            cb_push_back(CB_PART, 1);
            if (b + 2 < NB) {  // (the next block packs onto the same tile: L1 accumulation in place)
                cb_wait_front(CB_PART, 1);
                cb_pop_front(CB_PART, 1);
            }
            if (b + 2 == NB) {
                reload = true;
            }
        }
        cb_pop_front(CB_W, KB);
    }
    cb_pop_front(CB_X, NB * KB);
}
