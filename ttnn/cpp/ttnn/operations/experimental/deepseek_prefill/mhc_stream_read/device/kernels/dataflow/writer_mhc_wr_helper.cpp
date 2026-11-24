// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// mhc_stream_write_read's prefill groups with the write spread over the cores (MHC_WR_MERGED): a helper core's writer.
// Per unit (slice j = g S + s, column kk), its 4 new streams tiles (CB_OUT, row tile order) into the slice core's CB_S
// (at the same address on every core of the program: tile kk R + rt, the slice's column-major order; the slice core
// writes its whole slice to DRAM once complete, off the write's critical path: the writes here in DRAM cost ~38 us at
// 512 tokens).  MHC_WR_HELPER_DRAM: also to the new streams in DRAM here (page (4 g + rt) d_tiles + s K + kk).
// After its last unit of a slice: once its writes have
// landed (the barrier), the slice's flag of this sender (its slot: SEM_H0 + slot) set by a write (no semaphore
// increments, no atomic barrier); the slice core waits for its senders' flags.
//   CT: CB_OUT, CB_S, SEM_H0, d_tiles, k_tiles, k_own, slices, then the accessor of the new streams
//   RT: its address, the first unit, the units, the call's first group in the tensors, then the slots of the slices
//       it sends to, in order.  Common RT: the slice cores' NoC x, y (slice j = g S + s of the call)
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t off_g = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t sem_h0 = get_compile_time_arg_val(2);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t k_own = get_compile_time_arg_val(5);
    constexpr uint32_t slices = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr uint32_t helped = k_tiles - k_own, row_tiles = 4;

    const auto s_out = TensorAccessor(out_args, out_addr);
    CircularBuffer out(cb_out);
    const uint32_t tile = get_local_cb_interface(cb_out).fifo_page_size;
    const uint32_t s_base = get_local_cb_interface(cb_s).fifo_wr_ptr;  // (the slice cores' CB_S: the same address)

    uint32_t cur = 0xFFFFFFFFu, touched = 0;
    auto flag = [&]() {  // the current slice's tiles landed: its flag for this sender
        noc_async_write_barrier();
        const uint32_t slot = get_arg_val<uint32_t>(4 + touched);
        const uint32_t sem = get_semaphore(sem_h0 + slot);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) = 1;
        const uint32_t x = get_common_arg_val<uint32_t>(2 * cur), y = get_common_arg_val<uint32_t>(1 + 2 * cur);
        noc_semaphore_set_remote(sem, get_noc_addr(x, y, sem));
        ++touched;
    };
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t u = start + i;
        const uint32_t j = u / helped, kk = k_own + u % helped;
        if (j != cur) {
            if (cur != 0xFFFFFFFFu) {
                flag();
            }
            cur = j;
        }
        const uint32_t g = off_g + j / slices, s = j % slices;
        const uint32_t x = get_common_arg_val<uint32_t>(2 * j), y = get_common_arg_val<uint32_t>(1 + 2 * j);
        out.wait_front(row_tiles);
        const uint32_t src = out.get_read_ptr();
        for (uint32_t rt = 0; rt < row_tiles; ++rt) {
#ifdef MHC_WR_HELPER_DRAM
            noc_async_write(src + rt * tile, s_out.get_noc_addr((4 * g + rt) * d_tiles + s * k_tiles + kk), tile);
#endif
            noc_async_write(src + rt * tile, get_noc_addr(x, y, s_base + (kk * row_tiles + rt) * tile), tile);
        }
        noc_async_writes_flushed();
        out.pop_front(row_tiles);
    }
    if (cur != 0xFFFFFFFFu) {
        flag();
    }
    noc_async_write_barrier();  // (the flags)
}
