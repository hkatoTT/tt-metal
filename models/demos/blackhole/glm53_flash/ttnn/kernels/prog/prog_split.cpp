// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// The progressive form's split of a bf16 weight (ttnn/prog_split.py): for each of this core's tiles (runtime args 3, 4:
// its first tile and count), base = BFP4_b(w) and correction = BFP4_b(w - base), packed as the host packer
// (tt_metal/impl/data_format/blockfloat_common.cpp: a face row's 16 values share their largest exponent, each
// mantissa with its hidden bit shifted right by the difference, rounded to nearest-even to 3 bits and saturated; a
// zero or denormal input 0, a zero mantissa's sign dropped), the residual w - base exact (as the host's fp32).  A tile:
// 4 faces of 16 x 16 values, face-major; a BFP4_b tile the 64 face rows' exponents (bytes, 4 to a word, row 0 in the
// low byte) and then the 1,024 4-bit codes (sign << 3 | mantissa, 8 to a word, value 0 in the low nibble).
namespace prog_split {

constexpr uint32_t FACE_ROWS = 64;   // face rows a tile
constexpr uint32_t BFP4_BYTES = 576;  // a BFP4_b tile: 64 exponent bytes, 512 code bytes
constexpr uint32_t BF16_BYTES = 2048;  // a bf16 tile

// one value (an fp32 bit pattern) as a 4-bit code against the shared exponent `se` (the host's convert_u32_to_bfp)
inline uint32_t code_of(uint32_t bits, uint32_t se) {
    const uint32_t exp = (bits >> 23) & 0xFF;
    if (exp == 0) {
        return 0;
    }
    uint32_t mant = (1u << 23) | (bits & 0x7FFFFF);
    const uint32_t diff = se - exp;  // (se >= exp)
    mant = diff > 31 ? 0 : mant >> diff;
    const uint32_t round = mant & ((1u << 21) - 1);
    mant >>= 21;
    if (round > (1u << 20) || (round == (1u << 20) && (mant & 1))) {
        mant += 1;
    }
    if (mant > 7) {
        mant = 7;
    }
    if (mant == 0) {
        return 0;
    }
    return ((bits >> 31) << 3) | mant;
}

// a code's value as an fp32 bit pattern (the host's convert_bfp_to_u32 for BFP4_b): mantissa m (1 .. 7) of 3 bits
// below the shared exponent's hidden bit, normalized
inline uint32_t value_of(uint32_t code, uint32_t se) {
    uint32_t m = code & 7;
    if (m == 0) {
        return 0;
    }
    int32_t e = static_cast<int32_t>(se);
    // the value m 2^(se - 127 - 2): normalize m (1 .. 7) to 1.xx
    while ((m & 4) == 0) {
        m <<= 1;
        e -= 1;
    }
    if (e <= 0) {
        return 0;  // (below fp32's normal range: not met by weights)
    }
    return ((code >> 3) << 31) | (static_cast<uint32_t>(e) << 23) | ((m & 3) << 21);
}

// w - v exactly as an fp32 bit pattern, both fp32 patterns with v's mantissa of at most 3 bits and |w - v| small
inline uint32_t residual(uint32_t w, uint32_t v) {
    if (v == 0) {
        return w;
    }
    // the two as signed integers at the scale of w's last bit: w has at most 8 significant bits (bf16), v at most 3,
    // and v != 0 means |w| is at least half v's step, so their exponents differ by a few at most
    const int32_t ew = static_cast<int32_t>((w >> 23) & 0xFF), ev = static_cast<int32_t>((v >> 23) & 0xFF);
    const int32_t lo = (ew < ev ? ew : ev) - 7;  // the scale: the smaller exponent's 8th bit
    auto as_int = [&](uint32_t x, int32_t ex) {
        const int32_t m = static_cast<int32_t>(((1u << 23) | (x & 0x7FFFFF)) >> 16);  // 8 bits: 1.xxxxxxx
        const int32_t s = ex - 7 - lo;
        const int32_t mag = s >= 0 ? m << s : m >> (-s);
        return (x >> 31) ? -mag : mag;
    };
    int32_t r = as_int(w, ew) - as_int(v, ev);
    if (r == 0) {
        return 0;
    }
    const uint32_t sign = r < 0 ? 1u : 0u;
    uint32_t mag = static_cast<uint32_t>(r < 0 ? -r : r);
    // value mag 2^lo (lo in fp32's biased exponents, lo - 127 the power): normalize to 1.xxx 2^e
    int32_t top = 31 - __builtin_clz(mag);
    const int32_t e = lo + top;
    const uint32_t frac = top >= 23 ? (mag >> (top - 23)) & 0x7FFFFF : (mag << (23 - top)) & 0x7FFFFF;
    if (e <= 0) {
        return 0;
    }
    return (sign << 31) | (static_cast<uint32_t>(e) << 23) | frac;
}

// 16 fp32 patterns -> their codes into 2 words and the shared exponent
inline uint32_t pack_row(const uint32_t* v, volatile tt_l1_ptr uint32_t* words) {
    uint32_t se = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t e = (v[i] >> 23) & 0xFF;
        se = e > se ? e : se;
    }
    for (uint32_t w = 0; w < 2; ++w) {
        uint32_t word = 0;
        for (int32_t i = 7; i >= 0; --i) {
            word = (word << 4) | code_of(v[8 * w + i], se);
        }
        words[w] = word;
    }
    return se;
}

}  // namespace prog_split

void kernel_main() {
    using namespace prog_split;
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr = get_arg_val<uint32_t>(1);
    const uint32_t corr_addr = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    // where the tiles go (runtime args 5 ..): mode 0 the same tile ids (two plain outputs); mode 1 the experts op's w1
    // (moe.py `_init_experts_op`: rows (set E + e) T + c of D / 32 tiles, gate | up column tiles 2 wide, C the half);
    // mode 2 its w2 (down core q's rows ((q 2 + set) E + e) T + i, WQ tiles wide).  The source [.., W tiles wide] is
    // a dense copy: w1's [D, experts x 256] (or the shared expert's slice), w2's [experts x 256, D].  The correction
    // goes to set 1 (`corr_off` tiles past the base's place).  ttnn.concat along the width changed ~10 % of BFP4
    // values, so the layout is written here, tile by tile.
    const uint32_t mode = get_arg_val<uint32_t>(5);
    const uint32_t width = get_arg_val<uint32_t>(6);     // the source's width in tiles
    const uint32_t e_off = get_arg_val<uint32_t>(7);     // the source's first expert slot
    const uint32_t slots = get_arg_val<uint32_t>(8);     // E (the slots a set)
    const uint32_t d_tiles = get_arg_val<uint32_t>(9);   // w1: D / 32; w2: the tiles a row of w2 (WQ)
    const uint32_t half = get_arg_val<uint32_t>(10);     // w1: 0 gate, 1 up
    const uint32_t corr_off = get_arg_val<uint32_t>(11);  // the correction's tiles past the base's (modes 1, 2)
    constexpr uint32_t T_TILES = 8;                      // an expert's 256 columns of a chip
    auto place = [&](uint32_t s) -> uint32_t {
        if (mode == 1) {
            const uint32_t k = s / width, col = s % width;
            const uint32_t row = ((e_off + col / T_TILES) * T_TILES + col % T_TILES) * d_tiles + k;
            return row * 2 + half;
        }
        if (mode == 2) {
            const uint32_t kr = s / width, kc = s % width;
            const uint32_t core = kc / d_tiles;
            const uint32_t row = ((core * 2) * slots + e_off + kr / T_TILES) * T_TILES + kr % T_TILES;
            return row * d_tiles + kc % d_tiles;
        }
        return s;
    };
    constexpr uint32_t cb = get_compile_time_arg_val(0);  // L1: 2 input tiles and 2 x 2 output tiles
    constexpr auto i_args = TensorAccessorArgs<1>();
    constexpr auto b_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto s_in = TensorAccessor(i_args, in_addr, BF16_BYTES);
    const auto s_base = TensorAccessor(b_args, base_addr, BFP4_BYTES);
    const auto s_corr = TensorAccessor(c_args, corr_addr, BFP4_BYTES);
    const uint32_t l1 = get_write_ptr(cb);
    auto in_l1 = [&](uint32_t k) { return l1 + k * BF16_BYTES; };
    auto out_l1 = [&](uint32_t k, uint32_t which) { return l1 + 2 * BF16_BYTES + (2 * k + which) * BFP4_BYTES; };
    if (count == 0) {
        return;
    }
    noc_async_read(s_in.get_noc_addr(first), in_l1(0), BF16_BYTES);
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t k = t & 1;
        noc_async_read_barrier();  // tile t in
        if (t + 1 < count) {       // the next one in flight while this one is packed
            noc_async_read(s_in.get_noc_addr(first + t + 1), in_l1(k ^ 1), BF16_BYTES);
        }
        noc_async_writes_flushed();  // (tile t - 2's writes out of out_l1(k, ..))
        volatile tt_l1_ptr uint16_t* in = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_l1(k));
        volatile tt_l1_ptr uint8_t* be = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(out_l1(k, 0));
        volatile tt_l1_ptr uint8_t* ce = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(out_l1(k, 1));
        volatile tt_l1_ptr uint32_t* bm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1(k, 0) + 64);
        volatile tt_l1_ptr uint32_t* cm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1(k, 1) + 64);
        for (uint32_t row = 0; row < FACE_ROWS; ++row) {  // face-major rows: face row / 16, row in face row % 16
            uint32_t w[16], bv[16], r[16];
            for (uint32_t i = 0; i < 16; ++i) {
                w[i] = static_cast<uint32_t>(in[row * 16 + i]) << 16;
            }
            const uint32_t se = pack_row(w, bm + 2 * row);
            be[row] = static_cast<uint8_t>(se);
            for (uint32_t i = 0; i < 16; ++i) {
                const uint32_t code = (bm[2 * row + i / 8] >> (4 * (i % 8))) & 0xF;
                bv[i] = value_of(code, se);
                r[i] = residual(w[i], bv[i]);
            }
            ce[row] = static_cast<uint8_t>(pack_row(r, cm + 2 * row));
        }
        const uint32_t to = place(first + t);
        noc_async_write(out_l1(k, 0), s_base.get_noc_addr(to), BFP4_BYTES);
        noc_async_write(out_l1(k, 1), s_corr.get_noc_addr(to + corr_off), BFP4_BYTES);
    }
    noc_async_write_barrier();
}
