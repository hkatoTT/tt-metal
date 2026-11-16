// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE's column collectives (col_collectives.py), a worker core's compute: its input tiles untilized (row-major
// rows of its columns, for the reader and writer to send; not for the gather's `direct` reads), then
//   mode 0 (gather): the block (cb_buf, 32 trows rows: the column's 8 chips' t rows each, then zero rows) tilized;
//   mode 1 (scatter): the 8 chips' slots (cb_buf, 4 KB each) summed in the order of the chips into fp32 Dest,
//                     ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)) (FPU pairs, then SFPU adds), packed bf16
//                     (row-major rows, for the reader and writer to put into tiles);
//   mode 2 (padded scatter): the same sums (of the slots' face rows: the reader reads them without an untilize).
// `ar` (the scatters, then the all-reduce over the mesh row): then the row's 4 chips' slices of those sums (cb_rbuf,
// slot c a chip) added as row_ar_compute.cpp's fpu 2 does, (s0 + s1) and (s2 + s3) on the FPU into two Dest tiles,
// then their sum by the SFPU in fp32, packed bf16 (cb_out2).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_buf = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);  // gather: the tiles; scatter: the summed rows
    constexpr uint32_t tiles = get_compile_time_arg_val(5);
    constexpr uint32_t buf_pages = get_compile_time_arg_val(6);  // of cb_buf
    constexpr uint32_t spp = get_compile_time_arg_val(7);        // scatter: pages a slot
    constexpr uint32_t pages = get_compile_time_arg_val(8);       // scatter: pages of a slice (<= spp)
    constexpr bool direct = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t trows = get_compile_time_arg_val(10);  // gather: the block's tile rows (8 t rows, then zeros)
    constexpr bool ar = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t cb_rbuf = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out2 = get_compile_time_arg_val(13);
    constexpr uint32_t rnpt = get_compile_time_arg_val(14);  // pages a row slot

    compute_kernel_hw_startup(cb_in, cb_rm);
    if constexpr (!(mode == 0 && direct) && mode != 2) {
        DeviceZoneScopedN("CA_UNTILIZE");
        compute_kernel_lib::untilize<tiles, cb_in, cb_rm>(1);
    }
    cb_wait_front(cb_buf, buf_pages);
    DeviceZoneScopedN("CA_TILIZE");
    if constexpr (mode == 0) {
        compute_kernel_lib::tilize<tiles, cb_buf, cb_out>(trows);
    } else {
        reconfig_data_format(cb_buf, cb_buf);
        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, pages);
        for (uint32_t i = 0; i < pages; ++i) {  // slot c's page i: c spp + i
            tile_regs_acquire();
            add_init(cb_buf, cb_buf);
            for (uint32_t k = 0; k < 4; ++k) {
                add_tiles(cb_buf, cb_buf, 2 * k * spp + i, (2 * k + 1) * spp + i, k);
            }
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            add_binary_tile(2, 3, 2);
            add_binary_tile(0, 2, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, pages);
        if constexpr (ar) {
            cb_wait_front(cb_rbuf, 4 * rnpt);
            cb_reserve_back(cb_out2, pages);
            DeviceZoneScopedN("CA_RSUM");
            for (uint32_t i = 0; i < pages; ++i) {  // slot c's page i: c rnpt + i
                tile_regs_acquire();
                add_init(cb_rbuf, cb_rbuf);
                add_tiles(cb_rbuf, cb_rbuf, i, rnpt + i, 0);
                add_tiles(cb_rbuf, cb_rbuf, 2 * rnpt + i, 3 * rnpt + i, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out2);
                tile_regs_release();
            }
            cb_push_back(cb_out2, pages);
            cb_pop_front(cb_rbuf, 4 * rnpt);
        }
    }
}
