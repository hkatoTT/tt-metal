// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "scores_common.hpp"

// DSA indexer pool scores, the compute (scores_common.hpp): for each unit, S = relu(sum_c Q[c] K[c]^T) in fp32 Dest,
// packed fp32, then W S packed bf16.  Runtime args: the core's queries and computed blocks (VISIBLE: at most; the
// count from the reader through CB_CNT, once for shared queries, before each query's otherwise).
void kernel_main() {
    using namespace dsa_scores;
    const uint32_t queries = get_arg_val<uint32_t>(0);
    const uint32_t blocks = get_arg_val<uint32_t>(1);
    constexpr bool shared = get_compile_time_arg_val(0) != 0;
    constexpr bool q_rm = get_compile_time_arg_val(1) != 0;  // decode: Q row-major in CB_QRM, tilized here
    constexpr bool visible = get_compile_time_arg_val(2) != 0;
    if (queries == 0 || blocks == 0) {
        return;
    }
    compute_kernel_hw_startup(CB_Q, CB_K, CB_S);

    // matmul_init leaves the unpackers' formats as they are (srcA = in1, srcB = in0): set them for each matmul
    auto unit = [&](uint32_t q0, uint32_t w0) {
        tile_regs_acquire();
        reconfig_data_format(CB_K, CB_Q);
        matmul_init(CB_Q, CB_K, 1);
        for (uint32_t c = 0; c < 4; ++c) {
            matmul_tiles(CB_Q, CB_K, q0 + c, c, 0);
        }
        relu_tile_init();
        relu_tile(0);
        tile_regs_commit();
        cb_reserve_back(CB_S, 1);
        tile_regs_wait();
        pack_reconfig_data_format(CB_S);
        pack_tile(0, CB_S);
        tile_regs_release();
        cb_push_back(CB_S, 1);

        cb_wait_front(CB_S, 1);
        tile_regs_acquire();
        reconfig_data_format(CB_S, CB_W);
        matmul_init(CB_W, CB_S, 0);
        matmul_tiles(CB_W, CB_S, w0, 0, 0);
        tile_regs_commit();
        cb_reserve_back(CB_OUT, 1);
        tile_regs_wait();
        pack_reconfig_data_format(CB_OUT);
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_S, 1);
    };

    auto n_blocks = [&]() {
        if constexpr (!visible) {
            return blocks;
        }
        cb_wait_front(CB_CNT, 1);
        const uint32_t v = ckernel::read_tile_value(CB_CNT, 0, 0);
        cb_pop_front(CB_CNT, 1);
        return v;
    };

    if constexpr (shared) {
        const uint32_t nb = n_blocks();
        if constexpr (q_rm) {  // (decode's groups: a user's queries tilized first)
            for (uint32_t q = 0; q < queries; ++q) {
                compute_kernel_lib::tilize<4, CB_QRM, CB_Q>(1, 32);
            }
        }
        cb_wait_front(CB_Q, queries * 4);
        cb_wait_front(CB_W, queries);
        for (uint32_t j = 0; j < nb; ++j) {
            cb_wait_front(CB_K, 4);
            for (uint32_t q = 0; q < queries; ++q) {
                unit(q * 4, q);
            }
            cb_pop_front(CB_K, 4);
        }
        cb_pop_front(CB_Q, queries * 4);
        cb_pop_front(CB_W, queries);
    } else {
        for (uint32_t q = 0; q < queries; ++q) {
            const uint32_t nb = n_blocks();
            if constexpr (q_rm) {
                compute_kernel_lib::tilize<4, CB_QRM, CB_Q>(1, 32);
            }
            cb_wait_front(CB_Q, 4);
            cb_wait_front(CB_W, 1);
            for (uint32_t j = 0; j < nb; ++j) {
                cb_wait_front(CB_K, 4);
                unit(0, 0);
                cb_pop_front(CB_K, 4);
            }
            cb_pop_front(CB_Q, 4);
            cb_pop_front(CB_W, 1);
        }
    }
}
