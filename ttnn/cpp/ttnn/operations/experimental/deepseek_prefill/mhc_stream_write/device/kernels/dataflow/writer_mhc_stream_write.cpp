// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    // the hand-off of mhc_stream_write_read (1): each new streams tile also written into the slice core of the read
    // that projects it (its CB_S, at the same address on every core of that program: this one's `cb_s`), followed by
    // an increment of that core's semaphore `sem_s` (its count of tiles landed).  Tile (rt, c): slice s = c / k_tiles,
    // k = c % k_tiles; the groups forms (`group_rows` R row tiles a group: 1 for the groups of 8, 4 for the prefill's
    // groups of 32): the slice core of group rt / R, at its CB_S's tile (rt % R) k_tiles + k; else (0) the single
    // read's slice core, at tile rt k_tiles + k.  The slice cores' NoC x, y in the common runtime args (group major).
    // A run of tiles to one slice core is acknowledged once (the barrier, then the semaphore increased by the run's
    // count); each tile's CB slot freed once its writes have left it (flushed).
    constexpr bool handoff = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t sem_s = get_compile_time_arg_val(3);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t slices = get_compile_time_arg_val(5);
    constexpr uint32_t group_rows = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    // (the plain write has no hand-off and zeros here: divisors at least 1, so its unused hand-off code compiles)
    constexpr uint32_t kt = k_tiles ? k_tiles : 1, gr = group_rows ? group_rows : 1;

    const uint32_t tile = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s_out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer out(cb_out);
    [[maybe_unused]] constexpr uint32_t d_tiles = kt * (slices ? slices : 1);
    [[maybe_unused]] const uint32_t s_base = get_local_cb_interface(cb_s).fifo_wr_ptr;  // (the slices' too)
    [[maybe_unused]] uint32_t run_core = ~0U, run = 0;
    [[maybe_unused]] auto signal = [&]() {  // the run's tiles landed: its slice core's count up by the run
        if (run != 0) {
            noc_async_write_barrier();
            const uint32_t x = get_common_arg_val<uint32_t>(2 * run_core);
            const uint32_t y = get_common_arg_val<uint32_t>(1 + 2 * run_core);
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_s)), run);
            run = 0;
        }
    };
    // mhc_stream_write_read's hand-off: a long prefill wave's later call writes the streams' units past the earlier
    // calls' (runtime arg 3), its slice cores still by its own units
    [[maybe_unused]] uint32_t u_off = 0;
    if constexpr (handoff) {
        u_off = get_arg_val<uint32_t>(3);
    }
    for (uint32_t u = start; u < start + count; ++u) {
        out.wait_front(1);
        noc.async_write(out, s_out, tile, {}, {.page_id = u + u_off});
        if constexpr (handoff) {
            const uint32_t rt = u / d_tiles, c = u % d_tiles;
            const uint32_t s = c / kt, k = c % kt;
            const uint32_t core = group_rows ? (rt / gr) * slices + s : s;
            const uint32_t slot = group_rows ? (rt % gr) * k_tiles + k : rt * k_tiles + k;
            if (core != run_core) {
                signal();
                run_core = core;
            }
            const uint32_t x = get_common_arg_val<uint32_t>(2 * core), y = get_common_arg_val<uint32_t>(1 + 2 * core);
            noc_async_write(out.get_read_ptr(), get_noc_addr(x, y, s_base + slot * tile), tile);
            ++run;
            noc_async_writes_flushed();
        } else {
            noc.async_write_barrier();
        }
        out.pop_front(1);
    }
    if constexpr (handoff) {
        signal();
    }
    noc_async_write_barrier();
    if constexpr (handoff) {
        noc_async_atomic_barrier();  // (the slices' semaphore increments)
    }
}
