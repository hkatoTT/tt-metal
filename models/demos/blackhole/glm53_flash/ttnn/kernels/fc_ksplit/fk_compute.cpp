// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The K-split decode matmul (ttnn/fc_ksplit.py): one output column tile n of [mt tile rows of x] @ w, with ttnn.matmul's
// arithmetic for decode_linear's 1D config bit for bit (per core mt x 1 output tiles, out_subblock 1 x 1, in0_block_w
// KB, HiFi4 / the given fidelity, fp32 Dest, packer L1 accumulation; bmm_large_block_zm_fused_bias_activation.cpp):
// block b's KB tiles summed in Dest from zero by matmul_block for each tile row, the partials P = B_0 (packed plain),
// P += B_1 .. B_{NB-2} (the packer's L1 accumulation, fp32), and the last block summed onto P reloaded into Dest
// (unpacked to Dest as fp32), packed to the output.
//
// The column's S cores split the blocks: block b < NB - 1 is summed by slice PAT[b % L] (FK_PAT: L in bits 0 .. 3, then 2 bits
// an entry), the last by slice 0 (the owner).
// A helper (slice > 0) packs each of its B_b plain (fp32) into CB_SEND, which its fk_w writes to the owner; the owner sums its own
// blocks as ttnn does and, for the others in order, unpacks helper h's B_b (written into CB_RECV0 + h, fp32) into Dest and packs it onto
// P with the same L1 accumulation: the same Dest values and the same packer adds in the same order as one core.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

namespace {
constexpr uint32_t slice_of(uint32_t b) { return (FK_PAT >> (4 + 2 * (b % (FK_PAT & 15)))) & 3; }
constexpr uint32_t CB_X = 0, CB_W = 1, CB_OUT = 16, CB_PART = 24, CB_SEND = 25, CB_RECV0 = 26;  // helper h's at CB_RECV0 + h
}

void kernel_main() {
    constexpr uint32_t KB = get_compile_time_arg_val(0);  // in0_block_w
    constexpr uint32_t NB = get_compile_time_arg_val(1);  // K blocks
    constexpr uint32_t MT = get_compile_time_arg_val(2);  // x's tile rows
    constexpr uint32_t S = get_compile_time_arg_val(3);   // slices
    const uint32_t slice = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_W, slice == 0 ? CB_PART : CB_SEND);
    matmul_block_init(CB_X, CB_W, 0, 1, 1, KB);
    // one block of this core's: each tile row's KB tiles into Dest (on top of what is there), as ttnn's subblock loop
    auto sum_block = [&](uint32_t r) {
        for (uint32_t i = 0; i < KB; ++i) {
            matmul_block(CB_X, CB_W, r * KB + i, i, 0, 0, 1, 1, KB);
        }
    };
    if (slice != 0) {  // a helper: B_b of its blocks, plain fp32, for the owner
        pack_reconfig_data_format(CB_SEND);
        pack_reconfig_l1_acc(0);
        for (uint32_t b = 0; b < NB - 1; ++b) {
            if (slice_of(b) != slice) {
                continue;
            }
            cb_wait_front(CB_X, MT * KB);
            cb_wait_front(CB_W, KB);
            cb_reserve_back(CB_SEND, MT);
            for (uint32_t r = 0; r < MT; ++r) {
                tile_regs_acquire();
                sum_block(r);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, CB_SEND, r);
                tile_regs_release();
            }
            cb_push_back(CB_SEND, MT);
            cb_pop_front(CB_W, KB);
            cb_pop_front(CB_X, MT * KB);
        }
        return;
    }
    // the owner
    for (uint32_t b = 0; b < NB - 1; ++b) {
        const bool own = slice_of(b) == 0;
        const uint32_t cb_r = CB_RECV0 + slice_of(b);
        if (own) {
            cb_wait_front(CB_X, MT * KB);
            cb_wait_front(CB_W, KB);
        } else {
            cb_wait_front(cb_r, MT);
            reconfig_data_format_srca(CB_W, cb_r);
            copy_init(cb_r);
        }
        for (uint32_t r = 0; r < MT; ++r) {
            tile_regs_acquire();
            if (own) {
                sum_block(r);
            } else {
                copy_tile(cb_r, r, 0);
            }
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
        }
        if (own) {
            cb_pop_front(CB_W, KB);
            cb_pop_front(CB_X, MT * KB);
        } else {
            cb_pop_front(cb_r, MT);
            reconfig_data_format_srca(cb_r, CB_W);
            matmul_block_init(CB_X, CB_W, 0, 1, 1, KB);
        }
        if (b + 2 < NB) {  // (the next block packs onto the same tiles: the partials a ring of MT pages)
            cb_wait_front(CB_PART, MT);
            cb_pop_front(CB_PART, MT);
        }
    }
    // the last block, on the partials reloaded (ttnn's reload_from_cb_to_dst), packed to the output
    cb_wait_front(CB_X, MT * KB);
    cb_wait_front(CB_W, KB);
    for (uint32_t r = 0; r < MT; ++r) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_W, CB_PART);
        copy_init(CB_PART);
        cb_wait_front(CB_PART, 1);
        copy_tile(CB_PART, 0, 0);
        cb_pop_front(CB_PART, 1);
        reconfig_data_format_srca(CB_PART, CB_W);
        matmul_block_init(CB_X, CB_W, 0, 1, 1, KB);
        sum_block(r);
        tile_regs_commit();
        cb_reserve_back(CB_OUT, 1);
        tile_regs_wait();
        pack_reconfig_data_format(CB_OUT);
        pack_reconfig_l1_acc(0);
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
    }
    cb_pop_front(CB_W, KB);
    cb_pop_front(CB_X, MT * KB);
}
