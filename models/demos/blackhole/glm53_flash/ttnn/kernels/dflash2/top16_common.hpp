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

// One RISC's scan of rows r0 .. r1 - 1 of an item (top16.cpp; the TRISC split's unpack thread too, top16_trisc.cpp):
// the item's tiles at `data` (tile t - ct0 at (t - ct0) tile_b: its left face's g_max rows of 64 bytes, then its right
// face's), each row's 16 largest of the valid columns in descending order (equal keys: the first seen ahead) into
// `mine` + r 16 * 8 (16 keys, then 16 columns; EMPTY / 0 past the valid ones).  A face row with no value above a full
// list's last is skipped (a non-negative last: a plain signed compare of the bits).
inline __attribute__((always_inline)) void scan_rows(
    uint32_t data, uint32_t r0, uint32_t r1, uint32_t ct0, uint32_t ct1, uint32_t valid_cols, uint32_t mine,
    uint32_t g_max, uint32_t tile_b) {
    constexpr uint32_t FROW = 64, LIST = TOPN * 8;
    for (uint32_t r = r0; r < r1; ++r) {
        int32_t key[TOPN];
        uint32_t col[TOPN];
        uint32_t n = 0;
        for (uint32_t t = ct0; t < ct1; ++t) {
            const uint32_t col0 = t * 32;
            if (col0 >= valid_cols) {
                break;
            }
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t base = col0 + f * 16;
                if (base >= valid_cols) {
                    break;
                }
                const int32_t* v = reinterpret_cast<const int32_t*>(data + (t - ct0) * tile_b + f * g_max * FROW) + r * 16;
                const uint32_t cnt = valid_cols - base >= 16 ? 16 : valid_cols - base;
                if (n == TOPN && cnt == 16) {
                    const int32_t thr = key[TOPN - 1];
                    bool any = false;
                    if (thr >= 0) {  // (a key >= 0 is its float's bits: a larger value has larger signed bits)
#pragma GCC unroll 16
                        for (uint32_t j = 0; j < 16; ++j) {
                            any |= v[j] > thr;
                        }
                    } else {
#pragma GCC unroll 16
                        for (uint32_t j = 0; j < 16; ++j) {
                            any |= key_of(static_cast<uint32_t>(v[j])) > thr;
                        }
                    }
                    if (!any) {
                        continue;
                    }
                }
                for (uint32_t j = 0; j < cnt; ++j) {
                    n = list_insert(key, col, n, key_of(static_cast<uint32_t>(v[j])), base + j);
                }
            }
        }
        for (uint32_t j = n; j < TOPN; ++j) {
            col[j] = EMPTY;
            key[j] = 0;
        }
        uint32_t* dst = reinterpret_cast<uint32_t*>(mine + r * LIST);
        for (uint32_t j = 0; j < TOPN; ++j) {
            dst[j] = static_cast<uint32_t>(key[j]);
            dst[TOPN + j] = col[j];
        }
    }
}
