// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

// The router matmul's x (ttnn/router_mm.py; NoC 0, the sender's reads and multicast as decode_matmul's control kernel): this core's tile row of x (the
// row's whole K, kt tiles) into CB_X in chunks of `chunk` tiles.  The row's sender (column 0) reads each chunk from
// DRAM (several in flight), multicasts it to the row's other cores (the same address on each: their CB_X in step) and
// then sets their SEM_X to the chunks sent (a word a chunk in CB_SCR, never rewritten); the others wait for SEM_X to
// reach each chunk.  Every chunk is pushed to the compute as it is there.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t row = get_arg_val<uint32_t>(1);  // this core's tile row of x
    const uint32_t sender = get_arg_val<uint32_t>(2);
    const uint32_t rect = get_arg_val<uint32_t>(3);  // the row's cores (with the sender)
    const uint32_t mc0_x = get_arg_val<uint32_t>(4);
    const uint32_t mc0_y = get_arg_val<uint32_t>(5);
    const uint32_t mc1_x = get_arg_val<uint32_t>(6);
    const uint32_t mc1_y = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t sem_x = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);     // x's K tiles
    constexpr uint32_t chunk = get_compile_time_arg_val(4);  // tiles a multicast
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr uint32_t chunks = (kt + chunk - 1) / chunk;

    // this RISC's transaction ids from a clean state (earlier programs' tagged reads and multicasts can leave the NIU's
    // counters of ids 1 .. 15 non-zero; decode_matmul's control kernel likewise)
    noc_async_read_set_trid(0, noc_index);
    noc_async_write_set_trid(0, noc_index);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
    DeviceZoneScopedN("RM-X");
    const uint32_t xt = get_local_cb_interface(cb_x).fifo_page_size;
    cb_reserve_back(cb_x, kt);
    const uint32_t xbase = get_write_ptr(cb_x);
    if (sender) {
        const auto s_x = TensorAccessor(x_args, x_addr);
        const uint32_t words = get_write_ptr(cb_scr);
        const uint32_t sem_addr = get_semaphore(sem_x);
        constexpr uint32_t AHEAD = 4;
        constexpr uint32_t TRID0 = 9;  // ids 9 .. 13
        auto finish = [&](uint32_t c) {
            noc_async_read_barrier_with_trid(TRID0 + c % (AHEAD + 1));
            const uint32_t first = c * chunk;
            const uint32_t n = kt - first < chunk ? kt - first : chunk;
            if (rect > 1) {
                const uint32_t src = xbase + first * xt;
                noc_async_write_multicast(
                    src, get_noc_multicast_addr(mc0_x, mc0_y, mc1_x, mc1_y, src), n * xt, rect - 1, true);
                volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16 * c);
                *word = c + 1;
                noc_semaphore_set_multicast(
                    words + 16 * c, get_noc_multicast_addr(mc0_x, mc0_y, mc1_x, mc1_y, sem_addr), rect - 1);
            }
            cb_push_back(cb_x, n);
        };
        uint32_t done = 0;
        for (uint32_t c = 0; c < chunks; ++c) {
            if (c - done == AHEAD) {
                finish(done);
                ++done;
            }
            noc_async_read_set_trid(TRID0 + c % (AHEAD + 1));
            const uint32_t first = c * chunk;
            const uint32_t last = kt - first < chunk ? kt : first + chunk;
            for (uint32_t i = first; i < last; ++i) {
                noc_async_read(s_x.get_noc_addr(row * kt + i), xbase + i * xt, xt);
            }
        }
        for (; done < chunks; ++done) {
            finish(done);
        }
        noc_async_read_set_trid(0);
        noc_async_write_barrier();
    } else {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x));
        for (uint32_t c = 0; c < chunks; ++c) {
            noc_semaphore_wait_min(sem, c + 1);
            const uint32_t first = c * chunk;
            cb_push_back(cb_x, kt - first < chunk ? kt - first : chunk);
        }
        noc_semaphore_set(sem, 0);  // (the next call's chunks count from 0)
    }
}
