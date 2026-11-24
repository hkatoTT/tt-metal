// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The greedy head's pick over the mesh for the own mesh row's tokens only, v3 (head_pick.py `own`,
// GLM53_HEAD_PICK_OWN): v2's keys, landing pages and parities, with the stages swapped and
// each chip folding only its own mesh row's t = M / 8 rows.  The column stage first: each chip sends its payload P (all
// M rows: each column peer needs its own rows of it) down its mesh column, and folds the seven peers' rows of its own
// mesh row (the slices of their pages) with its own into Rc (t rows: the best over the column's 8 chips); then the row
// stage: Rc to the row's three others, and the four Rc folded into the mesh's best of the own rows.  The folds are
// 10 t a chip in place of v2's 10 M.  The output's rows outside the own mesh row's are left undefined (the input's
// ids, or nothing written): its one reader, the DFlash2 step's fused acceptance (verify_accept.accept_inputs), reads
// the own row's span only.

#include "head_pick2_common.hpp"

namespace head_pick3 {

// rows [j0, j1) of `out` (t rows): `own` (rows of this chip's payload) and the `ring` slots (slot k at
// slots + k slot_bytes, the k = me one skipped, row j at its 8 j) folded by the key's maximum
FORCE_INLINE void fold_slice(
    uint32_t out, uint32_t own, uint32_t slots, uint32_t slot_bytes, uint32_t ring, uint32_t me, uint32_t j0,
    uint32_t j1) {
    uint32_t* o = reinterpret_cast<uint32_t*>(out);
    const uint32_t* a = reinterpret_cast<const uint32_t*>(own);
    for (uint32_t j = j0; j < j1; ++j) {
        uint32_t bh = a[2 * j], bl = a[2 * j + 1];
        for (uint32_t k = 0; k < ring; ++k) {
            if (k == me) {
                continue;
            }
            const uint32_t* s = reinterpret_cast<const uint32_t*>(slots + k * slot_bytes);
            const uint32_t h = s[2 * j], l = s[2 * j + 1];
            if (h > bh || (h == bh && l > bl)) {
                bh = h;
                bl = l;
            }
        }
        o[2 * j] = bh;
        o[2 * j + 1] = bl;
    }
}

// as fold_slice, the best's id (~lo) into column 0 of row row0 + j of the id tiles at `tiles`
FORCE_INLINE void fold_slice_ids(
    uint32_t tiles, uint32_t row0, uint32_t own, uint32_t slots, uint32_t slot_bytes, uint32_t ring, uint32_t me,
    uint32_t j0, uint32_t j1) {
    const uint32_t* a = reinterpret_cast<const uint32_t*>(own);
    for (uint32_t j = j0; j < j1; ++j) {
        uint32_t bh = a[2 * j], bl = a[2 * j + 1];
        for (uint32_t k = 0; k < ring; ++k) {
            if (k == me) {
                continue;
            }
            const uint32_t* s = reinterpret_cast<const uint32_t*>(slots + k * slot_bytes);
            const uint32_t h = s[2 * j], l = s[2 * j + 1];
            if (h > bh || (h == bh && l > bl)) {
                bh = h;
                bl = l;
            }
        }
        *reinterpret_cast<uint32_t*>(tiles + head_pick2::col0(row0 + j)) = ~bl;
    }
}

// the ring's other chips' payloads: from each landed page (its arrival a bump of its receive semaphore) the `bytes`
// at `offset` read into its slot (slot k for chip k); returns once all are in L1
template <typename Accessor>
FORCE_INLINE void land_slices(
    const Accessor& s_area,
    uint32_t page_base,
    uint32_t ring,
    uint32_t me,
    uint32_t sem_args,
    uint32_t slots,
    uint32_t slot_bytes,
    uint32_t offset,
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
            noc_async_read(s_area.get_noc_addr(page_base + k) + offset, slots + k * slot_bytes, bytes);
        }
    }
    noc_async_read_barrier();
    head_pick2::compiler_fence();
}

}  // namespace head_pick3
