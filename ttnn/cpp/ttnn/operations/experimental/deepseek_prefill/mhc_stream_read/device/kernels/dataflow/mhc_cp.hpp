// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

// mhc_stream_write_read's decode forms (MHC_CP the CB index, MHC_CP_PAGES its pages, MHC_CP_SEM the write cores'
// semaphore): the mix core's writer multicasts the write side's comb and post pages (which the mix core's reader put
// into CB MHC_CP) into the same CB of the write cores (one CB over the mix cores and the write cores, just after
// CB_S: the same address on all of them), then sets their semaphore to 1.  Runtime args from `base`: the write cores'
// rectangles' count (0: another mix core sends them), then each one's NoC corners (low x, y, high x, y) and cores.
FORCE_INLINE void mhc_send_comb_post(uint32_t base) {
#ifdef MHC_CP
    const uint32_t rects = get_arg_val<uint32_t>(base);
    if (rects == 0) {
        return;
    }
    CircularBuffer cp(MHC_CP);
    cp.wait_front(2 * MHC_CP_PAGES);
    const uint32_t src = cp.get_read_ptr();
    const uint32_t bytes = 2 * MHC_CP_PAGES * get_local_cb_interface(MHC_CP).fifo_page_size;
    auto mcast = [&](uint32_t i, uint32_t addr) {
        const uint32_t lx = get_arg_val<uint32_t>(base + 1 + 5 * i), ly = get_arg_val<uint32_t>(base + 2 + 5 * i);
        const uint32_t hx = get_arg_val<uint32_t>(base + 3 + 5 * i), hy = get_arg_val<uint32_t>(base + 4 + 5 * i);
        return noc_index == 0 ? get_noc_multicast_addr(lx, ly, hx, hy, addr) : get_noc_multicast_addr(hx, hy, lx, ly, addr);
    };
    for (uint32_t i = 0; i < rects; ++i) {
        noc_async_write_multicast(src, mcast(i, src), bytes, get_arg_val<uint32_t>(base + 5 + 5 * i));
    }
    noc_async_write_barrier();
    const uint32_t sem = get_semaphore(MHC_CP_SEM);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) = 1;
    for (uint32_t i = 0; i < rects; ++i) {
        noc_semaphore_set_multicast(sem, mcast(i, sem), get_arg_val<uint32_t>(base + 5 + 5 * i));
    }
    cp.pop_front(2 * MHC_CP_PAGES);
#else
    (void)base;
#endif
}
