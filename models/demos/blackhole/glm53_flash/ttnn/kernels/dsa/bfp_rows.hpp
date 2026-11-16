// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// bf16 -> BFP8_b / BFP4_b on a data movement core (the DSA indexer's pools' keys stored in BFP, dsa.POOLS_DTYPE): a
// group is a tile face row, 16 bf16 values, which share an exponent byte (the bf16 exponent of their largest
// magnitude, E); each value's mantissa (with its hidden bit, 8 bits m at exponent e) becomes m / 2^(E - e + 8 - BITS)
// rounded to nearest, ties away from zero (tools/kv_round.bfp_rna: the device's own BFP8 packer rounds so too, but its
// BFP4 truncates the BFP8 value, gq), saturated at 2^BITS - 1 (BITS: 7 for BFP8, 3 for BFP4), with its sign (a zero
// magnitude without it).  bf16 subnormals are taken as 0 (a group of them: exponent 0, all zero).
// A BFP tile: 64 exponent bytes (face f row r at 16 f + r), then the 4 faces' mantissas, each face row 16 bytes (BFP8,
// byte j value j) or 8 bytes (BFP4, value 2 j in the low nibble of byte j: sign bit 3, then 3 magnitude bits).
namespace dsa_bfp {

constexpr uint32_t tile_bytes(uint32_t bits) { return bits == 8 ? 1088 : (bits == 4 ? 576 : 2048); }

// the 16 bf16 values at src (4-byte aligned, L1) -> face `face` row `row` of the BFP tile at tile (L1)
template <uint32_t BITS>
inline void pack_group(uint32_t src, uint32_t tile, uint32_t face, uint32_t row) {
    static_assert(BITS == 8 || BITS == 4, "BFP8_b or BFP4_b");
    constexpr uint32_t MBITS = BITS == 8 ? 7 : 3;   // magnitude bits
    constexpr uint32_t LIM = (1u << MBITS) - 1;
    constexpr uint32_t SH0 = 8 - MBITS;             // the largest exponent's shift (BFP8 1, BFP4 5)
    constexpr uint32_t FACE_B = BITS * 32;          // a face's mantissa bytes
    constexpr uint32_t ROW_B = BITS * 2;            // a face row's
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    uint32_t v[16];
    uint32_t emax = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t w = in[i];
        v[2 * i] = w & 0xFFFF;
        v[2 * i + 1] = w >> 16;
    }
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t e = (v[i] >> 7) & 0xFF;
        emax = e > emax ? e : emax;
    }
    uint32_t q[16];
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t e = (v[i] >> 7) & 0xFF;
        uint32_t mag = 0;
        if (e != 0) {
            const uint32_t m = 0x80 | (v[i] & 0x7F);
            const uint32_t sh = emax - e + SH0;
            mag = sh >= 9 ? 0 : (m + (1u << (sh - 1))) >> sh;
            mag = mag > LIM ? LIM : mag;
        }
        q[i] = mag == 0 ? 0 : (((v[i] >> 15) << MBITS) | mag);
    }
    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile)[face * 16 + row] = static_cast<uint8_t>(emax);
    volatile tt_l1_ptr uint32_t* out =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + 64 + face * FACE_B + row * ROW_B);
    if constexpr (BITS == 8) {
        for (uint32_t i = 0; i < 4; ++i) {
            out[i] = q[4 * i] | (q[4 * i + 1] << 8) | (q[4 * i + 2] << 16) | (q[4 * i + 3] << 24);
        }
    } else {
        for (uint32_t i = 0; i < 2; ++i) {
            uint32_t w = 0;
            for (uint32_t j = 0; j < 8; ++j) {
                w |= q[8 * i + j] << (4 * j);
            }
            out[i] = w;
        }
    }
}

// row `r` (0..31) of a bf16 tile at src_tile -> row r of the BFP tile at dst_tile (its 2 face rows)
template <uint32_t BITS>
inline void pack_tile_row(uint32_t src_tile, uint32_t dst_tile, uint32_t r) {
    for (uint32_t f = 0; f < 2; ++f) {
        const uint32_t face = (r / 16) * 2 + f;
        pack_group<BITS>(src_tile + face * 512 + (r % 16) * 32, dst_tile, face, r % 16);
    }
}

// a whole bf16 tile -> a BFP tile
template <uint32_t BITS>
inline void pack_tile(uint32_t src_tile, uint32_t dst_tile) {
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t row = 0; row < 16; ++row) {
            pack_group<BITS>(src_tile + face * 512 + row * 32, dst_tile, face, row);
        }
    }
}

}  // namespace dsa_bfp
