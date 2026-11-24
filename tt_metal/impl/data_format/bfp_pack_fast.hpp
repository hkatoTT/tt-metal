// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block-float packing of the standard tile (32 x 32, four 16 x 16 faces, exponent per face row, no exponent padding)
// without the per-row vectors of the general form: the same bytes as `pack_as_bfp_tiles` (round to nearest, ties to
// even, shared exponent the face row's largest, zero / denormal inputs 0, the sign dropped on a zero mantissa) for
// is_exp_a = false.  Header-only and free of the device context, so a CPU test can compile it against a copy of the
// general form.
//
// Two inputs: tiles already in the tilized (TILED_NFACES) order, and a row-major [rows, cols] matrix (rows and cols
// multiples of 32) packed in the tile order tilizing gives (tile rows, then tile columns), in one pass.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace tt::tt_metal::bfp_pack_fast {

constexpr size_t kFaceRows = 64;   // face rows of a tile: 4 faces x 16 rows
constexpr size_t kTileValues = 1024;

// MW: the stored mantissa bits (with the hidden bit), the value's bits less its sign: 7 (BFP8), 3 (BFP4), 1 (BFP2)
template <int MW>
constexpr size_t mantissa_row_bytes() {
    return 16 * (MW + 1) / 8;
}

template <int MW>
constexpr size_t tile_bytes() {
    return kFaceRows + kFaceRows * mantissa_row_bytes<MW>();
}

// One face row (16 values as fp32 bits) -> its shared exponent byte and its packed mantissas (value 0 in the low bits
// of the first byte), as convert_u32_to_bfp<.., false> with get_max_exp's exponent (is_exp_a false).
template <int MW>
inline void face_row(const uint32_t* u, uint8_t* exp_dst, uint8_t* mant_dst) {
    constexpr uint32_t kShift = 24 - MW;
    constexpr uint32_t kTie = 1u << (kShift - 1);
    constexpr uint32_t kRoundMask = (1u << kShift) - 1;
    constexpr uint32_t kMax = (1u << MW) - 1;
    uint32_t e[16];
    uint32_t mx = 0;
    for (int i = 0; i < 16; ++i) {
        e[i] = (u[i] >> 23) & 0xffu;
        mx = std::max(mx, e[i]);
    }
    uint8_t v[16];
    for (int i = 0; i < 16; ++i) {
        // 24 bits with the hidden one, shifted down by the exponent's distance from the shared one (a shift of 24 or
        // more leaves 0, as the general form's)
        uint32_t mant = ((u[i] & 0x7fffffu) | 0x800000u) >> std::min(mx - e[i], 31u);
        uint32_t rest = mant & kRoundMask;
        uint32_t q = mant >> kShift;
        q += static_cast<uint32_t>(rest > kTie) | (static_cast<uint32_t>(rest == kTie) & (q & 1u));
        q = std::min(q, kMax);
        uint32_t sign = (u[i] >> 31) & static_cast<uint32_t>(q != 0);
        uint32_t r = (sign << MW) | q;
        v[i] = e[i] == 0 ? 0 : static_cast<uint8_t>(r);
    }
    *exp_dst = static_cast<uint8_t>(mx);
    if constexpr (MW == 7) {
        std::memcpy(mant_dst, v, 16);
    } else if constexpr (MW == 3) {
        for (int k = 0; k < 8; ++k) {
            mant_dst[k] = static_cast<uint8_t>(v[2 * k] | (v[2 * k + 1] << 4));
        }
    } else {
        static_assert(MW == 1, "BFP8, BFP4 or BFP2");
        for (int k = 0; k < 4; ++k) {
            mant_dst[k] =
                static_cast<uint8_t>(v[4 * k] | (v[4 * k + 1] << 2) | (v[4 * k + 2] << 4) | (v[4 * k + 3] << 6));
        }
    }
}

// 16 source values -> fp32 bits.  Raw: uint16_t (bfloat16's bits) or uint32_t (float's bits)
template <typename Raw>
inline void load16(const Raw* src, uint32_t* u) {
    static_assert(std::is_same_v<Raw, uint16_t> || std::is_same_v<Raw, uint32_t>);
    if constexpr (std::is_same_v<Raw, uint16_t>) {
        uint16_t h[16];
        std::memcpy(h, src, sizeof(h));
        for (int i = 0; i < 16; ++i) {
            u[i] = static_cast<uint32_t>(h[i]) << 16;
        }
    } else {
        std::memcpy(u, src, 16 * sizeof(uint32_t));
    }
}

// Tiles [first, end) of tilized input (1,024 values a tile, faces in order, rows of 16 in a face) into `out`, the bytes
// of tile `first` on
template <int MW, typename Raw>
void pack_tiled_range(const Raw* tiled, size_t first, size_t end, uint8_t* out) {
    constexpr size_t kRow = mantissa_row_bytes<MW>();
    uint32_t u[16];
    for (size_t t = first; t < end; ++t) {
        const Raw* src = tiled + t * kTileValues;
        uint8_t* dst = out + (t - first) * tile_bytes<MW>();
        for (size_t fr = 0; fr < kFaceRows; ++fr) {
            load16(src + fr * 16, u);
            face_row<MW>(u, dst + fr, dst + kFaceRows + fr * kRow);
        }
    }
}

// The tiles [c0, c1) of tile row `tr` of a row-major matrix `cols` wide into `out` (the bytes of tile (tr, c0) on),
// reading the source rows in order: source row r of the tile row gives face rows (r / 16) * 32 + tc * 16 + r % 16,
// tc = 0, 1, of each tile.
template <int MW, typename Raw>
void pack_row_major_strip(const Raw* rm, size_t cols, size_t tr, size_t c0, size_t c1, uint8_t* out) {
    constexpr size_t kRow = mantissa_row_bytes<MW>();
    uint32_t u[16];
    for (size_t r = 0; r < 32; ++r) {
        const Raw* row = rm + (tr * 32 + r) * cols;
        const size_t fr0 = (r / 16) * 32 + r % 16;
        for (size_t c = c0; c < c1; ++c) {
            uint8_t* dst = out + (c - c0) * tile_bytes<MW>();
            for (size_t tc = 0; tc < 2; ++tc) {
                const size_t fr = fr0 + tc * 16;
                load16(row + c * 32 + tc * 16, u);
                face_row<MW>(u, dst + fr, dst + kFaceRows + fr * kRow);
            }
        }
    }
}

// A row-major matrix's work in strips of at most kStrip tiles of one tile row: strip s of tile row tr covers tiles
// [s kStrip, min((s + 1) kStrip, tile columns)), the strips in tile order
constexpr size_t kStrip = 32;

struct RowMajorGrid {
    size_t tile_rows, tile_cols, strips_per_row;
    RowMajorGrid(size_t rows, size_t cols) :
        tile_rows(rows / 32), tile_cols(cols / 32), strips_per_row((cols / 32 + kStrip - 1) / kStrip) {}
    size_t strips() const { return tile_rows * strips_per_row; }
};

// Strips [first, end) of the grid into `out` (the whole matrix's bytes)
template <int MW, typename Raw>
void pack_row_major_range(const Raw* rm, size_t cols, const RowMajorGrid& g, size_t first, size_t end, uint8_t* out) {
    for (size_t s = first; s < end; ++s) {
        const size_t tr = s / g.strips_per_row;
        const size_t c0 = (s % g.strips_per_row) * kStrip;
        const size_t c1 = std::min(c0 + kStrip, g.tile_cols);
        pack_row_major_strip<MW, Raw>(rm, cols, tr, c0, c1, out + (tr * g.tile_cols + c0) * tile_bytes<MW>());
    }
}

// One tile from a 32 x 32 block of bfloat16 bits transposed: tile element (i, j) = src[j row_stride + i] (the block's
// 32 rows of 32 values each read once, then the face rows taken from the transposed copy).
template <int MW>
void pack_transposed_block(const uint16_t* src, size_t row_stride, uint8_t* out) {
    constexpr size_t kRow = mantissa_row_bytes<MW>();
    uint16_t t[32][32];  // t[i][j] = src[j][i]
    for (size_t j = 0; j < 32; ++j) {
        uint16_t row[32];
        std::memcpy(row, src + j * row_stride, sizeof(row));
        for (size_t i = 0; i < 32; ++i) {
            t[i][j] = row[i];
        }
    }
    uint32_t u[16];
    for (size_t r = 0; r < 32; ++r) {
        const size_t fr0 = (r / 16) * 32 + r % 16;
        for (size_t tc = 0; tc < 2; ++tc) {
            const size_t fr = fr0 + tc * 16;
            load16(&t[r][tc * 16], u);
            face_row<MW>(u, out + fr, out + kFaceRows + fr * kRow);
        }
    }
}

// The concatenation of transposed slices, packed in the tile order tilizing gives: each slice a row-major
// [rows, cols] block of bfloat16 bits with its own row stride (all slices the same rows and cols, multiples of 32);
// dim 1: the result is [cols, slices x rows] (slice s's transpose in columns s rows ..); dim 0: [slices x cols, rows]
// (slice s's transpose in rows s cols ..).  Tiles [first, end) in the result's tile order into `out` (the result's
// bytes from tile 0).
struct TransposedSlices {
    const uint16_t* const* base;  // each slice's first value
    const size_t* row_stride;     // each slice's row stride (values)
    size_t slices, rows, cols;
    int dim;
    size_t out_rows() const { return dim == 1 ? cols : slices * cols; }
    size_t out_cols() const { return dim == 1 ? slices * rows : rows; }
    size_t tiles() const { return (out_rows() / 32) * (out_cols() / 32); }
};

template <int MW>
void pack_transposed_slices_range(const TransposedSlices& t, size_t first, size_t end, uint8_t* out) {
    const size_t tile_cols = t.out_cols() / 32;
    for (size_t k = first; k < end; ++k) {
        const size_t R = k / tile_cols, C = k % tile_cols;
        size_t s, src_row, src_col;  // the block's slice and its first row and column there
        if (t.dim == 1) {  // out[a, s rows + b] = slice_s[b, a]
            s = (C * 32) / t.rows;
            src_row = (C * 32) % t.rows;
            src_col = R * 32;
        } else {  // out[s cols + a, b] = slice_s[b, a]
            s = (R * 32) / t.cols;
            src_row = C * 32;
            src_col = (R * 32) % t.cols;
        }
        pack_transposed_block<MW>(
            t.base[s] + src_row * t.row_stride[s] + src_col, t.row_stride[s], out + k * tile_bytes<MW>());
    }
}

}  // namespace tt::tt_metal::bfp_pack_fast
