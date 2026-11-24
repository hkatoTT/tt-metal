// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The greedy head's pick over the mesh, v2 (head_pick.py, the default): the same
// fabric plumbing and landing pages as head_pick_common.hpp (v1), with the RISC-V work cut.
//   - A row's payload is one 64-bit key, two words: hi = order_key(the value's bits), lo = ~(the id's bits).  The
//     larger key is the larger value, and of equal values the smaller id (the ids are whole floats >= 0: their bits
//     are in their order), so v1's rule (reduce_into) is the key's maximum: the same pair whatever the order.
//   - The landed payloads are read into slots of their own, and a stage's fold takes all of them in one pass a row.
//   - The two RISC-Vs each fold half the rows of both stages (local semaphores between them).
//   - The output is the input's id tiles with column 0 replaced (head_argmax writes them: column 0 the ids, the rest
//     zero, as v1 wrote its output), so no tile is zeroed here.

#include "head_pick_common.hpp"

namespace head_pick2 {

using head_pick::order_key;

FORCE_INLINE void compiler_fence() { asm volatile("fence" ::: "memory"); }

// column 0 of row r in a tiled [M, 1] fp32 tensor's tiles laid out one after another (bytes)
FORCE_INLINE uint32_t col0(uint32_t r) {
    constexpr uint32_t tile = 4096, face = 1024, face_row = 64;
    return (r % 32 / 16) * 2 * face + (r % 16) * face_row + (r / 32) * tile;
}

// rows [r0, r1) of the stage's best: `own` and the `ring` slots (slot k at slots + k slot_bytes, the k = me one
// skipped) folded by their key's maximum into `out` (out may be own)
FORCE_INLINE void fold_into(
    uint32_t out, uint32_t own, uint32_t slots, uint32_t slot_bytes, uint32_t ring, uint32_t me, uint32_t r0,
    uint32_t r1) {
    uint32_t* o = reinterpret_cast<uint32_t*>(out);
    const uint32_t* a = reinterpret_cast<const uint32_t*>(own);
    for (uint32_t r = r0; r < r1; ++r) {
        uint32_t bh = a[2 * r], bl = a[2 * r + 1];
        for (uint32_t k = 0; k < ring; ++k) {
            if (k == me) {
                continue;
            }
            const uint32_t* s = reinterpret_cast<const uint32_t*>(slots + k * slot_bytes);
            const uint32_t h = s[2 * r], l = s[2 * r + 1];
            if (h > bh || (h == bh && l > bl)) {
                bh = h;
                bl = l;
            }
        }
        o[2 * r] = bh;
        o[2 * r + 1] = bl;
    }
}

// as fold_into, the best's id (~lo) into column 0 of row r of the id tiles at `tiles`
FORCE_INLINE void fold_ids(
    uint32_t tiles, uint32_t own, uint32_t slots, uint32_t slot_bytes, uint32_t ring, uint32_t me, uint32_t r0,
    uint32_t r1) {
    const uint32_t* a = reinterpret_cast<const uint32_t*>(own);
    for (uint32_t r = r0; r < r1; ++r) {
        uint32_t bh = a[2 * r], bl = a[2 * r + 1];
        for (uint32_t k = 0; k < ring; ++k) {
            if (k == me) {
                continue;
            }
            const uint32_t* s = reinterpret_cast<const uint32_t*>(slots + k * slot_bytes);
            const uint32_t h = s[2 * r], l = s[2 * r + 1];
            if (h > bh || (h == bh && l > bl)) {
                bh = h;
                bl = l;
            }
        }
        *reinterpret_cast<uint32_t*>(tiles + col0(r)) = ~bl;
    }
}

// the ring's other chips' payloads (each in its DRAM landing page, its arrival a bump of its receive semaphore) read
// into their slots as they land (slot k for chip k); returns once all are in L1
template <typename Accessor>
FORCE_INLINE void land_ring(
    const Accessor& s_area,
    uint32_t page_base,
    uint32_t ring,
    uint32_t me,
    uint32_t sem_args,
    uint32_t slots,
    uint32_t slot_bytes,
    uint32_t bytes) {
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
            noc_async_read(s_area.get_noc_addr(page_base + k), slots + k * slot_bytes, bytes);
        }
    }
    noc_async_read_barrier();
    compiler_fence();
}

FORCE_INLINE void local_set(uint32_t sem_id) {
    compiler_fence();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id)), 1);
}

FORCE_INLINE void local_wait(uint32_t sem_id) {
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id)), 1);
    compiler_fence();
}

}  // namespace head_pick2
