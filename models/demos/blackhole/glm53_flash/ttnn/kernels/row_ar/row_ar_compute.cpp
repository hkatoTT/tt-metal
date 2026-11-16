// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row all-reduce (row_all_reduce.py), a worker core's compute: the sum of the row's four chips' slices, in the
// order of the chips (the same on every chip of the row), into fp32 Dest, packed to bf16.  A slice is compacted rows,
// so the pages are not tiles; the adds are elementwise.
//   fpu 2: (s0 + s1) and (s2 + s3) on the FPU into two Dest tiles (exact: two bf16s), then their sum by the SFPU in
//          fp32 (the order differs from fpu 0's, but is the same on every chip);
//   fpu 0: the four slices unpacked to Dest as fp32 and added in order by the SFPU in fp32 (slower).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_recv = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t npt = get_compile_time_arg_val(2);    // pages a slot
    constexpr uint32_t pages = get_compile_time_arg_val(3);  // pages of the slice (<= npt)
    constexpr uint32_t fpu = get_compile_time_arg_val(4);
    constexpr uint32_t s0 = 0;  // slot c's page i: s0 + c npt + i

    compute_kernel_hw_startup(cb_recv, cb_recv, cb_out);
    cb_wait_front(cb_recv, 4 * npt);
    cb_reserve_back(cb_out, pages);
    DeviceZoneScopedN("AR_SUM");
    for (uint32_t i = 0; i < pages; ++i) {
        tile_regs_acquire();
        if constexpr (fpu == 2) {
            add_init(cb_recv, cb_recv);
            add_tiles(cb_recv, cb_recv, s0 + i, s0 + npt + i, 0);
            add_tiles(cb_recv, cb_recv, s0 + 2 * npt + i, s0 + 3 * npt + i, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        } else {
            copy_init(cb_recv);
            for (uint32_t c = 0; c < 4; ++c) {
                copy_tile(cb_recv, s0 + c * npt + i, c);
            }
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            add_binary_tile(0, 2, 0);
            add_binary_tile(0, 3, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, pages);
    cb_pop_front(cb_recv, 4 * npt);
}
