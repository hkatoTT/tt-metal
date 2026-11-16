// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE's column collectives (col_collectives.py), a worker core's reader (NCRISC).  Its `tiles` tiles of the input
// (one tile row) to the compute, which untilizes them: row-major rows of `row_bytes` (this core's columns), row j at
// j row_bytes.  Then
//   mode 0 (gather): rows 0 .. t - 1 (this chip's tokens) into this chip's rows (my t ..) of its block (cb_buf), and to
//                    the sender core's stage (slot `worker`).  `direct` (t = 1): the row from the tiles (face 0's and
//                    1's first rows, by 64-byte reads and local copies), no untilize;
//   mode 1 (scatter): rows d t .. d t + t - 1 (the tokens of mesh row d) for d < split_stage to the sender core's
//                    stage (slot d workers + worker), and this chip's own into its slot of cb_buf; then zeros into its
//                    share of the staging tiles (from `split` on);
//   mode 2 (padded scatter): no untilize: of each of its tiles in tile row d (the tokens of mesh row d, rows 0 .. t -
//   1)
//                    face 0's and face 1's first t rows (t 32 bytes each) read from DRAM into chip d's slice of cb_rm
//                    (tile j's at j 64 t), then staged as mode 1 (all of them by the reader: split_stage 8);
// then this worker's bit of the stage's semaphore (the scatter: bit 2 worker; the writer sets 2 worker + 1).  The
// column's other chips send each group of workers' slices in one packet into the DRAM landing area (page (parity,
// chip, the group's leader), the members' slices in turn), bumping the leader's receive semaphore for that chip once
// it is written.  The leader waits for each chip in turn, tells its members (sem_ready: the count of chips so far),
// and every member reads its slice of that chip's page into cb_buf (the gather: rows c t of the block, whose rows from
// 8 t the writer zeroes; the scatter: slot c).  cb_buf to the compute, and, beside the writer, the output tiles from
// `split` on.  `ar` (the scatters, then the all-reduce over the mesh row's 4 chips, col_collectives.col_scatter_ar):
// the compute's sum (cb_out) into this chip's slot of cb_rbuf (4 slots, one a chip of the row) and to the row sender
// core's stage (slot `worker`, then bit `worker` of its semaphore: row_ar_sender.cpp sends it to the row's other
// chips, into their DRAM landing area, page (parity, this chip, worker), bumping this worker's receive semaphore for
// this chip there); for each of the row's other chips the wait for its slice and its read into its slot (as
// row_ar_reader.cpp); cb_rbuf to the compute, which adds the four (cb_out2), and the output tiles from those sums.
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "col_common.hpp"
#include "../oneshot_util.hpp"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);  // gather: the tiles; scatter: the summed rows
    constexpr uint32_t sem_gather = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);    // t
    constexpr uint32_t cb_buf = get_compile_time_arg_val(7);  // gather: the 32-row block; scatter: 8 slots of 4 KB
    constexpr uint32_t sem_ready = get_compile_time_arg_val(8);
    constexpr bool direct = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t split_stage = get_compile_time_arg_val(10);
    constexpr uint32_t out_pages = get_compile_time_arg_val(11);  // of cb_out
    constexpr uint32_t sem_zero = get_compile_time_arg_val(12);   // gather: the writer has zeroed the block's tail
    constexpr uint32_t buf_pages = get_compile_time_arg_val(13);  // of cb_buf
    constexpr uint32_t cb_staging = get_compile_time_arg_val(14);  // scatter: the output tiles' staging
    constexpr uint32_t trows = get_compile_time_arg_val(15);       // gather: the output's tile rows
    constexpr uint32_t row_tiles = get_compile_time_arg_val(16);   // the output's tiles a tile row
    constexpr bool ar = get_compile_time_arg_val(17) != 0;         // then the all-reduce over the row
    constexpr uint32_t cb_rbuf = get_compile_time_arg_val(18);     // ar: the row's 4 slots
    constexpr uint32_t cb_rstage = get_compile_time_arg_val(19);   // ar: the row sender's stage (the same address here)
    constexpr uint32_t cb_out2 = get_compile_time_arg_val(20);     // ar: the row's sums
    constexpr uint32_t sem_rgather = get_compile_time_arg_val(21);  // ar: the row sender's stage semaphore
    constexpr uint32_t rnpt = get_compile_time_arg_val(22);        // ar: pages (2 KB) a row slot
    constexpr uint32_t chips = 8;                                 // the column
    constexpr uint32_t slot_bytes = 4096;                         // cb_buf's slots (scatter), a landing page
    constexpr auto x_args = TensorAccessorArgs<23>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto area_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto rarea_args = TensorAccessorArgs<area_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile0 = get_arg_val<uint32_t>(1);
    const uint32_t tiles = get_arg_val<uint32_t>(2);
    const uint32_t parity = get_arg_val<uint32_t>(3);
    const uint32_t my = get_arg_val<uint32_t>(4);  // this chip's row of the mesh
    const uint32_t sender_x = get_arg_val<uint32_t>(5);
    const uint32_t sender_y = get_arg_val<uint32_t>(6);
    const uint32_t worker = get_arg_val<uint32_t>(7);
    const uint32_t workers = get_arg_val<uint32_t>(8);
    const uint32_t out_addr = get_arg_val<uint32_t>(9);
    const uint32_t split = get_arg_val<uint32_t>(10);
    const uint32_t area_addr = get_arg_val<uint32_t>(11);
    // 12 .. 19: this parity's receive semaphores, one for each chip of the column (the leader's)
    const uint32_t lead = get_arg_val<uint32_t>(20);     // the group's leader (a worker index)
    const uint32_t member = get_arg_val<uint32_t>(21);   // this worker's place in its group (0: the leader)
    const uint32_t members = get_arg_val<uint32_t>(22);  // the leader: the group's other workers, their (x, y) from 23
    // ar, from 23 + 2 members: this chip's place in the row, the row sender's NoC x, y, the row stage's parity, its
    // landing area, then its receive semaphores (this parity), one for each chip of the row
    const uint32_t ra = 23 + 2 * members;

    const uint32_t row_bytes = tiles * 64;
    const uint32_t slice = rows * row_bytes;
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    const uint32_t stage = get_write_ptr(cb_stage);
    cb_reserve_back(cb_buf, buf_pages);
    const uint32_t buf = get_write_ptr(cb_buf);
    auto place = [&](uint32_t c) { return mode == 0 ? buf + c * slice : buf + c * slot_bytes; };
    if constexpr (mode == 2) {
        DeviceZoneScopedN("CA_READ");
        // (each read 64-byte aligned at both ends, t even: a smaller DRAM read lands right only so)
        const uint32_t half = rows * 32;  // a face's first t rows
        cb_reserve_back(cb_rm, chips * slice / 2048);
        const uint32_t rm = get_write_ptr(cb_rm);
        for (uint32_t d = 0; d < chips; ++d) {
            for (uint32_t j = 0; j < tiles; ++j) {
                const uint32_t tile = d * row_tiles + tile0 + j;
                const uint32_t dst = rm + d * slice + j * 2 * half;
                if constexpr (rows == 16) {  // faces 0 and 1 whole: one read
                    noc_async_read(s_x.get_noc_addr(tile, 0), dst, 2 * half);
                } else {
                    noc_async_read(s_x.get_noc_addr(tile, 0), dst, half);
                    noc_async_read(s_x.get_noc_addr(tile, 512), dst + half, half);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_rm, chips * slice / 2048);
    } else if (!(mode == 0 && direct)) {
        DeviceZoneScopedN("CA_READ");
        cb_reserve_back(cb_in, tiles);
        const uint32_t in = get_write_ptr(cb_in);
        for (uint32_t j = 0; j < tiles; ++j) {
            noc_async_read(s_x.get_noc_addr(tile0 + j), in + j * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, tiles);
    }
    {
        DeviceZoneScopedN("CA_STAGE");
        const uint32_t own = place(my);
        if constexpr (mode == 0) {
            if constexpr (direct) {
                // t = 1: face 0's and face 1's first two rows (64 bytes, to 64-byte aligned L1 like their source: a
                // DRAM read to another alignment lands wrong) into cb_in, then their first rows into the block
                const uint32_t tmp = get_write_ptr(cb_in);
                for (uint32_t j = 0; j < tiles; ++j) {
                    noc_async_read(s_x.get_noc_addr(tile0 + j, 0), tmp + j * 128, 64);
                    noc_async_read(s_x.get_noc_addr(tile0 + j, 512), tmp + j * 128 + 64, 64);
                }
                noc_async_read_barrier();
                for (uint32_t j = 0; j < tiles; ++j) {
                    noc_async_write(tmp + j * 128, get_noc_addr(own + j * 64), 32);
                    noc_async_write(tmp + j * 128 + 64, get_noc_addr(own + j * 64 + 32), 32);
                }
                noc_async_write_barrier();
            } else {
                cb_wait_front(cb_rm, tiles);
                noc_async_write(get_read_ptr(cb_rm), get_noc_addr(own), slice);
                noc_async_write_barrier();
            }
            noc_async_write(own, get_noc_addr(sender_x, sender_y, stage + worker * slice), slice);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(sender_x, sender_y, get_semaphore(sem_gather)), 1u << worker);
        } else {
            const uint32_t rm = get_read_ptr(cb_rm);
            if constexpr (mode == 1) {
                cb_wait_front(cb_rm, tiles);
            }
            col_stage_rows(rm, slice, 0, split_stage, my, 0, stage, workers, worker, sender_x, sender_y);
            noc_async_write(rm + my * slice, get_noc_addr(own), slice);
            noc_async_write_barrier();
            // the padded scatter: one bit a worker (the reader stages them all)
            noc_semaphore_inc(
                get_noc_addr(sender_x, sender_y, get_semaphore(sem_gather)), 1u << (mode == 2 ? worker : 2 * worker));
            if (split < tiles) {
                oneshot_zero(get_write_ptr(cb_staging) + split * 2048, (tiles - split) * 2048);
            }
        }
    }
    {
        DeviceZoneScopedN("CA_WAIT");
        const auto s_area = TensorAccessor(area_args, area_addr, slot_bytes);
        const uint32_t page0 = parity * chips * workers + lead;  // + c workers: chip c's page for this group
        auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready));
        uint32_t k = 0;
        for (uint32_t c = 0; c < chips; ++c) {
            if (c == my) {
                continue;
            }
            ++k;
            if (member == 0) {
                auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(12 + c));
                noc_semaphore_wait(sem, 1);
                noc_semaphore_set(sem, 0);  // (this parity's next writes come two calls later)
                for (uint32_t m = 0; m < members; ++m) {
                    const uint32_t x = get_arg_val<uint32_t>(23 + 2 * m), y = get_arg_val<uint32_t>(24 + 2 * m);
                    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_ready)), 1);
                }
            } else {
                noc_semaphore_wait_min(ready, k);
            }
            noc_async_read(s_area.get_noc_addr(page0 + c * workers, member * slice), place(c), slice);
        }
        noc_async_read_barrier();
    }
    if constexpr (mode == 0 && chips * rows < 32 * trows) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_zero)), 1);
    }
    cb_push_back(cb_buf, buf_pages);
    if constexpr (ar) {
        const uint32_t c_me = get_arg_val<uint32_t>(ra);
        const uint32_t rs_x = get_arg_val<uint32_t>(ra + 1), rs_y = get_arg_val<uint32_t>(ra + 2);
        const uint32_t rparity = get_arg_val<uint32_t>(ra + 3);
        const uint32_t rarea_addr = get_arg_val<uint32_t>(ra + 4);
        constexpr uint32_t rslot = rnpt * 2048;
        cb_reserve_back(cb_rbuf, 4 * rnpt);
        const uint32_t rbuf = get_write_ptr(cb_rbuf);
        {
            DeviceZoneScopedN("CA_RSTAGE");
            cb_wait_front(cb_out, out_pages);  // the column's sum of this worker's slice
            const uint32_t sum = get_read_ptr(cb_out);
            noc_async_write(sum, get_noc_addr(rbuf + c_me * rslot), slice);
            noc_async_write(sum, get_noc_addr(rs_x, rs_y, get_write_ptr(cb_rstage) + worker * rslot), slice);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(rs_x, rs_y, get_semaphore(sem_rgather)), 1u << worker);
        }
        {
            DeviceZoneScopedN("CA_RWAIT");
            const auto s_rarea = TensorAccessor(rarea_args, rarea_addr, rslot);
            for (uint32_t c = 0; c < 4; ++c) {
                if (c == c_me) {
                    continue;
                }
                auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(ra + 5 + c));
                noc_semaphore_wait(sem, 1);
                noc_semaphore_set(sem, 0);  // (this parity's next writes come two calls later)
                noc_async_read(s_rarea.get_noc_addr((rparity * 4 + c) * workers + worker), rbuf + c * rslot, slice);
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_rbuf, 4 * rnpt);
    }
    constexpr uint32_t cb_fin = ar ? cb_out2 : cb_out;  // the sums put into the output
    if (split < tiles) {
        const auto s_out = TensorAccessor(out_args, out_addr, 2048);
        cb_wait_front(cb_fin, out_pages);
        DeviceZoneScopedN("CA_WRITE2");
        if constexpr (mode == 0) {
            for (uint32_t tr = 0; tr < trows; ++tr) {
                col_put_tiles(s_out, get_read_ptr(cb_out) + tr * tiles * 2048, tr * row_tiles + tile0, split, tiles);
            }
        } else if constexpr (mode == 2) {
            col_put_faces(s_out, get_read_ptr(cb_fin), rows, get_write_ptr(cb_staging), tile0, split, tiles);
        } else {
            col_put_rows(s_out, get_read_ptr(cb_fin), rows, row_bytes, get_write_ptr(cb_staging), tile0, split, tiles);
        }
    }
    noc_async_atomic_barrier();
}
