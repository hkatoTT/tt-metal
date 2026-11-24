// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// dflash2_ops.rope_kv's item order (GLM53_DFLASH2_ROPE_KV_RUNS=1: RKV_SPREAD 1).  The program's cores take the physical
// items [c PER, (c + 1) PER); with RKV_SPREAD the v items (logical ROPE .. ROPE + NV - 1, the heavy ones: a tile's 32
// rows into the V cache) go one a core, each the first physical item of cores 0 .. NV - 1, and the rope items (q's,
// then k's) fill the other places in order.  Without it the physical item is the logical one.
#ifndef RKV_SPREAD
#define RKV_SPREAD 0
#endif
#ifndef RKV_PER
#define RKV_PER 1
#endif
#ifndef RKV_NV
#define RKV_NV 0
#endif

template <uint32_t ROPE>
inline uint32_t rkv_logical(uint32_t i) {
    if constexpr (RKV_SPREAD == 0) {
        return i;
    } else {
        constexpr uint32_t per = RKV_PER, nv = RKV_NV;
        if (i % per == 0 && i / per < nv) {
            return ROPE + i / per;
        }
        const uint32_t before = (i + per - 1) / per;  // v places below i (at c PER, c < NV)
        return i - (before < nv ? before : nv);
    }
}

template <uint32_t ROPE>
inline bool rkv_has_rope(uint32_t i0, uint32_t i1) {
    for (uint32_t i = i0; i < i1; ++i) {
        if (rkv_logical<ROPE>(i) < ROPE) {
            return true;
        }
    }
    return false;
}
