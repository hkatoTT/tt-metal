// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE's column collectives (col_collectives.py), a worker core's writer (BRISC), on NoC 0 beside the reader on
// NoC 1:
//   mode 0 (gather): first rows 8 t .. 32 trows - 1 of the block (cb_buf) zeroed (the reader waits for sem_zero
//                    before the compute tilizes it), then the output tiles before `split` of each tile row (the reader
//                    writes the rest);
//   mode 1 (scatter): first the untilized rows for chips split_stage .. 7 but this one (to the sender's stage, as the
//                    reader does for the others) and bit 2 worker + 1 of the stage's semaphore, then zeros into the
//                    staging tiles before `split` (a circular buffer), and the summed rows into them and those tiles
//                    to the output;
//   mode 2 (padded scatter): the reader stages all; zeros into the staging tiles before `split`, the summed faces' rows
//                    into them and those tiles to the output.
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "col_common.hpp"
#include "../oneshot_util.hpp"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t cb_buf = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);  // gather: the tiles; scatter: the summed rows
    constexpr uint32_t sem_gather = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);
    constexpr uint32_t split_stage = get_compile_time_arg_val(7);
    constexpr uint32_t out_pages = get_compile_time_arg_val(8);
    constexpr uint32_t sem_zero = get_compile_time_arg_val(9);
    constexpr uint32_t cb_staging = get_compile_time_arg_val(10);
    constexpr uint32_t trows = get_compile_time_arg_val(11);      // gather: the output's tile rows
    constexpr uint32_t row_tiles = get_compile_time_arg_val(12);  // the output's tiles a tile row
    // the sums put into the output: cb_out, or (ar: then the all-reduce over the mesh row) the row's sums
    constexpr uint32_t cb_fin = get_compile_time_arg_val(13);
    constexpr uint32_t chips = 8;
    constexpr auto out_args = TensorAccessorArgs<14>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile0 = get_arg_val<uint32_t>(1);
    const uint32_t tiles = get_arg_val<uint32_t>(2);
    const uint32_t split = get_arg_val<uint32_t>(3);
    const uint32_t my = get_arg_val<uint32_t>(4);
    const uint32_t sender_x = get_arg_val<uint32_t>(5);
    const uint32_t sender_y = get_arg_val<uint32_t>(6);
    const uint32_t worker = get_arg_val<uint32_t>(7);
    const uint32_t workers = get_arg_val<uint32_t>(8);

    const uint32_t row_bytes = tiles * 64;
    const auto s_out = TensorAccessor(out_args, out_addr, 2048);
    if constexpr (mode == 0 && chips * rows < 32 * trows) {
        DeviceZoneScopedN("CA_ZERO");
        const uint32_t block = get_write_ptr(cb_buf);
        for (uint32_t a = block + chips * rows * row_bytes; a < block + 32 * trows * row_bytes; a += MEM_ZEROS_SIZE) {
            noc_async_write(MEM_ZEROS_BASE, get_noc_addr(a), MEM_ZEROS_SIZE);
        }
        noc_async_write_barrier();
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_zero)), 1);
    }
    if constexpr (mode == 1) {
        DeviceZoneScopedN("CA_STAGE2");
        cb_wait_front(cb_rm, tiles);
        col_stage_rows(
            get_read_ptr(cb_rm),
            rows * row_bytes,
            split_stage,
            chips,
            my,
            0,  // this chip's own slot: by the reader
            get_write_ptr(cb_stage),
            workers,
            worker,
            sender_x,
            sender_y);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(sender_x, sender_y, get_semaphore(sem_gather)), 1u << (2 * worker + 1));
        oneshot_zero(get_write_ptr(cb_staging), split * 2048);
    }
    if constexpr (mode == 2) {
        oneshot_zero(get_write_ptr(cb_staging), split * 2048);
    }
    cb_wait_front(cb_fin, out_pages);
    DeviceZoneScopedN("CA_WRITE");
    if constexpr (mode == 0) {
        for (uint32_t tr = 0; tr < trows; ++tr) {
            col_put_tiles(s_out, get_read_ptr(cb_out) + tr * tiles * 2048, tr * row_tiles + tile0, 0, split);
        }
    } else if constexpr (mode == 2) {
        col_put_faces(s_out, get_read_ptr(cb_fin), rows, get_write_ptr(cb_staging), tile0, 0, split);
    } else {
        col_put_rows(s_out, get_read_ptr(cb_fin), rows, row_bytes, get_write_ptr(cb_staging), tile0, 0, split);
    }
    noc_async_atomic_barrier();
}
