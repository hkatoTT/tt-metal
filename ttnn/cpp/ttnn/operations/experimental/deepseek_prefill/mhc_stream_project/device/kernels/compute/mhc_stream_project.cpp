// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per unit: Dest 0-2 = sum_k S_k @ fn_k (the 96 projection columns), Dest 3 = sum_k S_k @ S_k^T (the Gram tile).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t CB_S = 0;
constexpr uint32_t CB_FN = 1;
constexpr uint32_t CB_OUT = 16;
constexpr uint32_t PROJ_TILES = 3;
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_S, CB_FN, CB_OUT);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    const uint32_t count = get_arg_val<uint32_t>(0);

    CircularBuffer s(CB_S), fn(CB_FN), out(CB_OUT);
    for (uint32_t u = 0; u < count; ++u) {
        s.wait_front(k_tiles);
        fn.wait_front(k_tiles * PROJ_TILES);
        out.reserve_back(PROJ_TILES + 1);
        tile_regs_acquire();
        matmul_init(CB_S, CB_FN);
        for (uint32_t k = 0; k < k_tiles; ++k) {
            for (uint32_t n = 0; n < PROJ_TILES; ++n) {
                matmul_tiles(CB_S, CB_FN, k, k * PROJ_TILES + n, n);
            }
        }
        matmul_init(CB_S, CB_S, 1);
        for (uint32_t k = 0; k < k_tiles; ++k) {
            matmul_tiles(CB_S, CB_S, k, k, PROJ_TILES);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t n = 0; n <= PROJ_TILES; ++n) {
            pack_tile<true>(n, CB_OUT, n);
        }
        tile_regs_release();
        out.push_back(PROJ_TILES + 1);
        s.pop_front(k_tiles);
        fn.pop_front(k_tiles * PROJ_TILES);
    }
}
