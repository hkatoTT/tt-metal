// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The relaxed greedy head's pick over the mesh (head_relax.py): head_pick's (head_pick_common.hpp) with a third word a
// row, the next token's logit where this chip holds it (-BIG elsewhere), folded by its maximum.  The same fabric
// plumbing (landing pages and receive semaphores of the "head_row3" / "head_col3" families).

#include "head_pick_common.hpp"

namespace head_relax {

using head_pick::order_key;

constexpr uint32_t WORDS = 3;  // a row's (value, id, next logit)

// the (value, id) as head_pick's reduce_into (the larger value, of equal values the smaller id); the next logit's
// maximum (as ordered keys: exact, any order)
FORCE_INLINE void reduce_into(uint32_t into, uint32_t from, uint32_t rows) {
    volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(into);
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t va = order_key(a[3 * r]), vb = order_key(b[3 * r]);
        const uint32_t ib = b[3 * r + 1];
        if (vb > va || (vb == va && ib < a[3 * r + 1])) {
            a[3 * r] = b[3 * r];
            a[3 * r + 1] = ib;
        }
        if (order_key(b[3 * r + 2]) > order_key(a[3 * r + 2])) {
            a[3 * r + 2] = b[3 * r + 2];
        }
    }
}

// head_pick's gather_ring for these payloads
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
            noc_async_read(s_area.get_noc_addr(page_base + k), in, rows * WORDS * 4);
            noc_async_read_barrier();
            reduce_into(acc, in, rows);
        }
    }
}

}  // namespace head_relax
