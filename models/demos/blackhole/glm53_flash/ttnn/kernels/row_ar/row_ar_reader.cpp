// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row all-reduce (row_all_reduce.py), a worker core's reader (NCRISC): its tiles' first t rows of x into this chip's
// slot of its receive buffer (a circular buffer, 4 slots), compacted (per tile: face 0's rows, then face 1's, `seg`
// bytes each), a copy of that slice to the sender core's stage (then this worker's bit of the stage's semaphore set),
// zeros into its share of the staging tiles, then, for each of the row's three other chips, the wait for its slice
// (its fabric write lands in the DRAM landing area, page (parity, chip, worker), and bumps this core's receive
// semaphore for that chip; this chip's own is not waited on, a 2D multicast may bump it) and its read from DRAM into
// that chip's slot; the four slots to the compute, and then, beside the writer, the last of its output tiles (from
// `split` on).
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "row_ar_common.hpp"
#include "../oneshot_util.hpp"

void kernel_main() {
    constexpr uint32_t cb_recv = get_compile_time_arg_val(0);   // the receive buffer: 4 slots, one a chip of the row
    constexpr uint32_t cb_stage = get_compile_time_arg_val(1);  // the sender's stage (the same address here)
    constexpr uint32_t seg = get_compile_time_arg_val(2);       // bytes of a face's rows
    constexpr uint32_t sem_gather = get_compile_time_arg_val(3);
    constexpr uint32_t npt = get_compile_time_arg_val(4);  // pages (2 KB) a slot
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);   // t
    constexpr uint32_t pages = get_compile_time_arg_val(7);  // the compute's output pages
    constexpr uint32_t cb_staging = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto area_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile0 = get_arg_val<uint32_t>(1);
    const uint32_t tiles = get_arg_val<uint32_t>(2);
    const uint32_t parity = get_arg_val<uint32_t>(3);
    const uint32_t my_idx = get_arg_val<uint32_t>(4);  // this chip's place in the row
    const uint32_t sender_x = get_arg_val<uint32_t>(5);
    const uint32_t sender_y = get_arg_val<uint32_t>(6);
    const uint32_t worker = get_arg_val<uint32_t>(7);
    const uint32_t workers = get_arg_val<uint32_t>(8);
    // 9 .. 12: this parity's receive semaphores (global semaphores), one for each chip of the row
    const uint32_t out_addr = get_arg_val<uint32_t>(13);
    const uint32_t split = get_arg_val<uint32_t>(14);
    const uint32_t area_addr = get_arg_val<uint32_t>(15);

    constexpr uint32_t slot = npt * 2048;
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    const auto s_area = TensorAccessor(area_args, area_addr, slot);
    const uint32_t recv_base = get_write_ptr(cb_recv);
    const uint32_t own = recv_base + my_idx * slot;
    const uint32_t bytes = tiles * 2 * seg;
    {
        DeviceZoneScopedN("AR_READ");
        for (uint32_t j = 0; j < tiles; ++j) {
            noc_async_read(s_x.get_noc_addr(tile0 + j, 0), own + j * 2 * seg, seg);
            noc_async_read(s_x.get_noc_addr(tile0 + j, 512), own + j * 2 * seg + seg, seg);
        }
        noc_async_read_barrier();
    }
    {
        DeviceZoneScopedN("AR_STAGE");
        const uint32_t stage = get_write_ptr(cb_stage) + worker * slot;
        noc_async_write(own, get_noc_addr(sender_x, sender_y, stage), bytes);
        noc_async_write_barrier();
        // bit `worker` of the stage's semaphore: the sender sends each slice as it lands
        noc_semaphore_inc(get_noc_addr(sender_x, sender_y, get_semaphore(sem_gather)), 1u << worker);
    }
    const uint32_t staging = get_write_ptr(cb_staging);
    if (split < tiles) {
        oneshot_zero(staging + split * 2048, (tiles - split) * 2048);
    }
    {
        DeviceZoneScopedN("AR_WAIT");
        for (uint32_t c = 0; c < 4; ++c) {
            if (c == my_idx) {
                continue;
            }
            auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(9 + c));
            noc_semaphore_wait(sem, 1);
            noc_semaphore_set(sem, 0);  // (this parity's next writes come two calls later)
            noc_async_read(s_area.get_noc_addr((parity * 4 + c) * workers + worker), recv_base + c * slot, bytes);
        }
        noc_async_read_barrier();
    }
    cb_reserve_back(cb_recv, 4 * npt);
    cb_push_back(cb_recv, 4 * npt);
    if (split < tiles) {
        const auto s_out = TensorAccessor(out_args, out_addr, 2048);
        cb_wait_front(cb_out, pages);
        DeviceZoneScopedN("AR_WRITE2");
        row_ar_put_tiles<seg, rows>(s_out, get_read_ptr(cb_out), staging, tile0, split, tiles);
    }
    noc_async_atomic_barrier();
}
