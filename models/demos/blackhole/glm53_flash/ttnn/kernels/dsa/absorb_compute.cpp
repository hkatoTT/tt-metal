// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The DSA decode's per-head absorb matmul (ttnn/dsa_absorb.py) on one core: its head's R tile rows of a [R x K] times
// its column slice of w [K x NC], with ttnn.matmul's arithmetic for `common.batched_matmul_config` (the factory's
// bmm_large_block_zm_fused_bias_activation_metal2.cpp with in0_block_w = K, one K block, out_subblock_h 1,
// out_subblock_w CW, fp32 Dest; one block: no partials, no packer L1 accumulation): each output subblock of CW tiles
// summed over k = 0 .. K - 1 in Dest from zero by matmul_block, the packer reconfigured to the output and the subblock
// packed.  The same calls in the same order for each output tile: only which core computes which tiles differs.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

namespace {
constexpr uint32_t CB_A = 0, CB_W = 1, CB_OUT = 16, CB_PART = 24;
}

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t CW = get_compile_time_arg_val(3);  // out_subblock_w
    constexpr uint32_t KC = get_compile_time_arg_val(4);  // w's k rows a pushed chunk (absorb_reader.cpp)
    static_assert(NC % CW == 0);
    // (the factory's start: the partials CB, fp32 with fp32 Dest, as the packer's first format)
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_A, CB_W, CB_PART);
    matmul_block_init(CB_A, CB_W, false, CW, 1, K);
    cb_wait_front(CB_A, R * K);
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t sb = 0; sb < NC / CW; ++sb) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < K; ++k) {
                if (r == 0 && sb == 0 && k % KC == 0) {
                    const uint32_t k1 = k + KC < K ? k + KC : K;
                    cb_wait_front(CB_W, k1 * NC);
                }
                matmul_block(CB_A, CB_W, r * K + k, k * NC + sb * CW, 0, false, CW, 1, K);
            }
            tile_regs_commit();
            cb_reserve_back(CB_OUT, CW);
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUT);
            pack_block(0, CB_OUT, CW);
            tile_regs_release();
            cb_push_back(CB_OUT, CW);
        }
    }
    cb_pop_front(CB_W, K * NC);
    cb_pop_front(CB_A, R * K);
}
