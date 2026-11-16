// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// DFlash2's candidate lists (top16.cpp, top16_merge.cpp): fp32 values as ordered integer keys, and a sorted list of
// the 16 largest (descending; an equal key goes after the ones already in, so the first seen stays ahead).
constexpr uint32_t TOPN = 16;
constexpr uint32_t EMPTY = 0xFFFFFFFFu;

inline __attribute__((always_inline)) int32_t key_of(uint32_t bits) {
    // negative floats flipped below the positive ones; -0.0 (key -1) onto +0.0 (key 0); branch-free
    const int32_t s = static_cast<int32_t>(bits);
    return (s ^ ((s >> 31) & 0x7FFFFFFF)) + static_cast<int32_t>(bits == 0x80000000u);
}

inline __attribute__((always_inline)) uint32_t bits_of(int32_t key) {
    return key < 0 ? static_cast<uint32_t>(key ^ 0x7FFFFFFF) : static_cast<uint32_t>(key);
}

// insert (k, c) into the list of n (key, col), keeping its first TOPN; returns the new count
inline __attribute__((always_inline)) uint32_t
list_insert(int32_t* key, uint32_t* col, uint32_t n, int32_t k, uint32_t c) {
    if (n == TOPN) {
        if (k <= key[TOPN - 1]) {
            return n;
        }
        n = TOPN - 1;
    }
    uint32_t j = n;
    while (j > 0 && key[j - 1] < k) {
        key[j] = key[j - 1];
        col[j] = col[j - 1];
        --j;
    }
    key[j] = k;
    col[j] = c;
    return n + 1;
}
