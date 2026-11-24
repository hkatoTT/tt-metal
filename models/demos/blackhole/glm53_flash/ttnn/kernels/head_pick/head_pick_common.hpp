// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The greedy head's pick over the mesh (head_pick.py), shared by its row (BRISC) and column (NCRISC) kernels.

#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

static_assert(std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "head pick: 2D fabric");

namespace head_pick {

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
using Conn = tt::tt_fabric::WorkerToFabricEdmSender;

// A float's bits -> an unsigned key in the floats' order (-0 as +0, so that equal values compare equal): the
// data-movement RISC-Vs have no FPU, and a float compare there is a library call (256 rows' folds took longer than the
// all-gathers they replace).  The ids (whole numbers as floats, not negative) compare as their bits.
FORCE_INLINE uint32_t order_key(uint32_t b) {
    b = b == 0x80000000u ? 0u : b;
    return b ^ (static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) | 0x80000000u);
}

// A payload: per row of tokens its best (value, id) as two floats.  `into` keeps the larger value, and of equal values
// the smaller id (torch's argmax keeps the first): the same pair whatever the order the payloads come in.
FORCE_INLINE void reduce_into(uint32_t into, uint32_t from, uint32_t rows) {
    volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(into);
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t va = order_key(a[2 * r]), vb = order_key(b[2 * r]);
        const uint32_t ib = b[2 * r + 1];
        if (vb > va || (vb == va && ib < a[2 * r + 1])) {
            a[2 * r] = b[2 * r];
            a[2 * r + 1] = ib;
        }
    }
}

FORCE_INLINE void copy_words(uint32_t to, uint32_t from, uint32_t words) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(to);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from);
    for (uint32_t i = 0; i < words; ++i) {
        d[i] = s[i];
    }
}

// one packet: the payload written to `dst` on the route's chips, then their semaphore `sem` bumped (once written)
FORCE_INLINE void send(Conn& conn, Header h, uint32_t src, uint64_t dst, uint64_t sem, uint32_t bytes) {
    h->to_noc_fused_unicast_write_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true}, bytes);
    conn.wait_for_empty_write_slot();
    conn.send_payload_without_header_non_blocking_from_address(src, bytes);
    conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(h), sizeof(PACKET_HEADER_TYPE));
}

// the ring's other chips' payloads (each in its DRAM landing page, its arrival a bump of its receive semaphore) folded
// into `acc` in the order they land (`in`: a scratch payload; the fold does not depend on the order)
template <typename Accessor>
FORCE_INLINE void gather_ring(
    const Accessor& s_area,
    uint32_t page_base,
    uint32_t ring,
    uint32_t me,
    uint32_t sem_args,
    uint32_t acc,
    uint32_t in,
    uint32_t rows) {
    uint32_t pending = ((1u << ring) - 1) & ~(1u << me);
    while (pending != 0) {
        invalidate_l1_cache();
        for (uint32_t left = pending; left != 0; left &= left - 1) {
            const uint32_t k = __builtin_ctz(left);
            auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(sem_args + k));
            if (*sem == 0) {
                continue;
            }
            noc_semaphore_set(sem, 0);  // (this parity's next writes come two calls later)
            pending &= ~(1u << k);
            noc_async_read(s_area.get_noc_addr(page_base + k), in, rows * 8);
            noc_async_read_barrier();
            reduce_into(acc, in, rows);
        }
    }
}

}  // namespace head_pick
