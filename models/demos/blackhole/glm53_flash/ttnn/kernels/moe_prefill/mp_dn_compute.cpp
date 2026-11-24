// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "mp_common.hpp"

// The prefill MoE's down, the compute (mp_common.hpp): per item (its tile rows' count from the writer's mailbox), W2(e,
// q) kept; per row, y = h W2 over the 8 K tiles in fp32 Dest (HiFi4), untilized from Dest (fp32) to DN_Y, the item's
// 4 column blocks side by side in one 32-row x 512-column block (the writer then writes 2 KB a Y row in place of
// 4 x 512 bytes).  Y_BFP8 (y8): DN_Y fp32, then the row's 16 flat 1,024-value pieces of it unpacked to Dest as
// they are (fp32) and packed in bfp8 to DN_Y8 (each 16 consecutive values of a row an exponent; one rounding), which
// the writer takes in place of DN_Y.
void kernel_main() {
    using namespace moe_prefill;
    uint32_t n_items = get_arg_val<uint32_t>(0);
    constexpr bool balance = get_compile_time_arg_val(0) != 0;  // BALANCE: the segments' count from the writer
    constexpr bool y8 = get_compile_time_arg_val(1) != 0;       // Y_BFP8
    // the progressive form: 2 (the item's W2 of the base, then of the correction: h times both into the same Dest)
    constexpr uint32_t sets = get_compile_time_arg_val(2);
    constexpr uint32_t W2_TILES = DN_Q * DN_K * DN_N;  // a set's
    constexpr uint32_t OUT = DN_Q * DN_N;                        // DN_Y's pages a push

    compute_kernel_hw_startup<SrcOrder::Reverse>(DN_HT, DN_W2, DN_Y);  // matmul: in0 -> SrcB, in1 -> SrcA
    pack_untilize_dest_init<DN_N, OUT>(DN_Y);
    if constexpr (balance) {
        UNPACK(n_items = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(n_items = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(n_items = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    }
    for (uint32_t i = 0; i < n_items; ++i) {
        uint32_t rows = 0;
        UNPACK(rows = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(rows = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(rows = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        if (rows == 0) {
            continue;
        }
        cb_wait_front(DN_W2, sets * W2_TILES);
        for (uint32_t r = 0; r < rows; ++r) {
            cb_wait_front(DN_HT, DN_K);
            cb_reserve_back(DN_Y, OUT);
            for (uint32_t j = 0; j < DN_Q; ++j) {  // the item's column blocks, h read once
                tile_regs_acquire();
                fill_tile_init();  // Dest is not cleared between programs
                for (uint32_t n = 0; n < DN_N; ++n) {
                    fill_tile(n, 0.0f);
                }
                matmul_block_init(DN_HT, DN_W2, 0, DN_N, 1, DN_K);
                for (uint32_t k = 0; k < sets * DN_K; ++k) {  // h tile k times W2's K tile row k of block j
                    const uint32_t kk = k % DN_K, base = (k / DN_K) * W2_TILES;  // (PROG: the correction's after)
                    matmul_block(DN_HT, DN_W2, kk, base + (j * DN_K + kk) * DN_N, 0, 0, DN_N, 1, DN_K);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<DN_N, OUT>(DN_Y, 1, j);
                tile_regs_release();
            }
            cb_push_back(DN_Y, OUT);
            cb_pop_front(DN_HT, DN_K);
            if constexpr (y8) {  // the row's flat pieces fp32 -> bfp8 (normal packing, no DEST remap meanwhile)
                pack_untilize_uninit(DN_Y);
#ifdef ARCH_BLACKHOLE
                MATH((llk_math_reconfig_remap(false)));
#endif
                reconfig_data_format_srca(DN_W2, DN_Y);
                pack_reconfig_data_format(DN_Y8);
                copy_tile_to_dst_init_short(DN_Y);
                cb_wait_front(DN_Y, OUT);
                cb_reserve_back(DN_Y8, OUT);
                for (uint32_t k = 0; k < OUT; k += 4) {  // (4 pieces a Dest half: fp32, half sync)
                    tile_regs_acquire();
                    for (uint32_t d = 0; d < 4; ++d) {
                        copy_tile(DN_Y, k + d, d);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t d = 0; d < 4; ++d) {
                        pack_tile<true>(d, DN_Y8, k + d);
                    }
                    tile_regs_release();
                }
                cb_push_back(DN_Y8, OUT);
                cb_pop_front(DN_Y, OUT);
                reconfig_data_format_srca(DN_Y, DN_W2);
                pack_untilize_dest_init<DN_N, OUT>(DN_Y);  // (the packer's format and DEST remap back)
            }
        }
        cb_pop_front(DN_W2, sets * W2_TILES);
    }
    pack_untilize_uninit(DN_Y);
}
