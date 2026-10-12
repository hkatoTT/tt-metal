// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Neighbor pad fabric readers: stage up to HK_NP_RD_BATCH sticks into the sender CB and wait for
// their NoC reads once, then hand them to the writer one page at a time (EXPERIMENT, hkato).
#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

#ifndef HK_NP_RD_BATCH
#define HK_NP_RD_BATCH 1
#endif

struct HkRdBatch {
    uint32_t cb_id;
    uint32_t n = 0;
    uint32_t base = 0;

    // L1 address of the next staged page (reserves a whole batch when a new one starts).
    uint32_t next() {
        if (n == 0) {
            cb_reserve_back(cb_id, HK_NP_RD_BATCH);
            base = get_write_ptr(cb_id);
        }
        const auto& iface = get_local_cb_interface(cb_id);
        uint32_t addr = base + n * iface.fifo_page_size;
        if (addr >= iface.fifo_limit) {
            addr -= iface.fifo_size;
        }
        n++;
        return addr;
    }
    void flush() {
        if (n == 0) {
            return;
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            cb_push_back(cb_id, 1);
        }
        n = 0;
    }
    void done_one() {
        if (n >= HK_NP_RD_BATCH) {
            flush();
        }
    }
};

template <uint32_t stick_size_bytes>
inline void hk_zero_stick(Noc& noc, uint32_t addr) {
    noc.async_write_zeros(CoreLocalMem<uint8_t>(addr), stick_size_bytes);
    noc.write_zeros_l1_barrier();
}
