// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// This core's weight block (K tiles k0 .. k0 + kc - 1, N tiles n0 + i stride, i < ncb, of the [K / 32, N / 32] tiles,
// page k N/32 + n) for the compute, in its order: per sub-block of sb columns, per chunk of cr K rows.  A chunk is a
// slot of cr x sb tiles (row r at r sb; a short last sub-block or chunk leaves the rest unread).  With the stride =
// the DRAM banks, a row's tiles of a sub-block are consecutive pages of one bank: one request; with stride 1, one
// request a tile.  Up to `ahead` chunks are in flight, each with its own transaction id, pushed in order as they land.
void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t k0 = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);
    const uint32_t ncb = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);
    const uint32_t kc = get_arg_val<uint32_t>(5);
    const uint32_t cr = get_arg_val<uint32_t>(6);
    const uint32_t sb = get_arg_val<uint32_t>(7);
    const uint32_t ahead = get_arg_val<uint32_t>(8);
    const uint32_t stride = get_arg_val<uint32_t>(9);
    // 2: this core's requests alternate between NoC 0 and NoC 1 once the control kernel hands NoC 1 over (one NoC's
    // reads of a bank top out below the bank's rate); 1: NoC 0 only.  Dedicated NoC mode: both RISCs read with
    // command buffer 1 of their own NoC and share its NIU's counters, so this RISC reads on NoC 1 only on cores where
    // the control kernel (BRISC, NoC 1) does not read (not on x's senders: 1 there), and only after that kernel has
    // set its command buffers' transaction ids to 0 and cleared NoC 1's outstanding counters of ids 1 .. 15 (earlier
    // programs can leave them non-zero) and then set sem_noc1: until then NoC 0 only, with no transaction id set on
    // NoC 1 (a read the control kernel issues would carry it).  The reads here wait on their transaction ids only
    // (1 .. depth; this RISC's firmware keeps no counts of NoC 1).
    const uint32_t nocs = get_arg_val<uint32_t>(10);
    // with nocs 2: at most this many chunks in flight until NoC 1 is handed over (0: `ahead` throughout), so that the
    // banks' queues stay short while x's sender reads x behind them (x then lands sooner, and NoC 1 comes sooner)
    const uint32_t ahead0 = get_arg_val<uint32_t>(11);
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr bool skip = get_compile_time_arg_val(1) != 0;  // a test switch: no reads (the slots pushed as they are)
    constexpr uint32_t sem_noc1 = get_compile_time_arg_val(2);
    constexpr auto w_args = TensorAccessorArgs<3>();
    if (ncb == 0) {
        return;
    }
    const auto s_w = TensorAccessor(w_args, w_addr);
    CircularBuffer w(cb_w);
    const uint32_t tile = get_local_cb_interface(cb_w).fifo_page_size;
    const uint32_t slot = cr * sb, depth = ahead + 1, per_sub = (kc + cr - 1) / cr;
    const uint32_t chunks = ((ncb + sb - 1) / sb) * per_sub;
    const uint32_t base = w.get_write_ptr();  // the ring's start: nothing pushed yet
    DeviceZoneScopedN("W");
    uint32_t pushed = 0;
    const uint8_t other = 1 - noc_index;  // NoC 1
    bool both = false;                     // NoC 1 handed over (sem_noc1), from chunk first2 on
    uint32_t first2 = chunks;
    auto landed = [&](uint32_t c) {  // chunk c's reads all here
        noc_async_read_barrier_with_trid(c % depth + 1, noc_index);
        if (c >= first2) {
            noc_async_read_barrier_with_trid(c % depth + 1, other);
        }
    };
    for (uint32_t issued = 0; issued < chunks; ++issued) {
        const uint32_t lim = both || ahead0 == 0 ? ahead : ahead0;
        while (issued - pushed >= lim) {  // the oldest in flight to the compute
            landed(pushed);
            w.push_back(slot);
            ++pushed;
        }
        w.reserve_back((issued - pushed + 1) * slot);  // room for the ones in flight and this one
        const uint32_t dst = base + (issued % depth) * slot * tile;
        if (nocs == 2 && !both) {
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_noc1)) != 0) {
                both = true;
                first2 = issued;
            }
        }
        noc_async_read_set_trid(issued % depth + 1, noc_index);  // ids 1 .. depth
        if (both) {
            noc_async_read_set_trid(issued % depth + 1, other);
        }
        if constexpr (skip) {
            continue;
        }
        const uint32_t c0 = (issued / per_sub) * sb;
        const uint32_t sbw = ncb - c0 < sb ? ncb - c0 : sb;
        const uint32_t g = issued % per_sub;
        const uint32_t rows = kc - g * cr < cr ? kc - g * cr : cr;  // the slice's last chunk may be short
        const uint32_t row0 = k0 + g * cr;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t page = (row0 + r) * nt + n0 + c0 * stride;
            if (stride != 1) {  // pages page, page + B, ..: adjacent in one bank
                const uint8_t k = both && ((issued * cr + r) & 1) ? other : noc_index;
                noc_async_read(s_w.get_noc_addr(page, 0, k), dst + r * sb * tile, sbw * tile, k);
                continue;
            }
            for (uint32_t c = 0; c < sbw; ++c) {
                noc_async_read(s_w.get_noc_addr(page + c), dst + (r * sb + c) * tile, tile);
            }
        }
    }
    for (; pushed < chunks; ++pushed) {
        landed(pushed);
        w.push_back(slot);
    }
    noc_async_read_set_trid(0, noc_index);
    if (both) {
        noc_async_read_set_trid(0, other);
    }
    noc_async_atomic_barrier();
}
