// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"

// The decode MoE's routing by generic_op (moe_route.py `route_only`): the rows moe_decode_route writes for a column's
// experts, bit for bit.  A core's rows (count of them) sit in L1 by blocks of 16 experts: expert e of the core's row
// j at word (e / 16 * count + j) * 16 + e % 16 (the logits, the bias, the scores and the choice keys alike), so that
// the rows of one 16-row group of a logits tile come in one read a block; a row's blocks are `stride` = 16 count words
// apart.
namespace moe_route {
constexpr uint32_t F = 1024, R = 64;  // fp32 face bytes, face row bytes
// The scratch CB (its start rounded up to 64 B): a zero row at WZERO, the chip's local base page for the writer at
// WBASE and for the reader at RBASE, the output rows' staging (128 B a row: the ids' 64 B, the weights'; 16 rows) at
// WOUT, and at TSYNC the words the writer and the compute's unpack thread pass (with `trisc`): [0] the local base,
// [1] and [3] the core's rows that add the shared expert, lo .. hi - 1 (`shared_span`; the writer's, before it pushes
// the scratch CB, which the compute waits for), [2] the compute's rows staged (set by the compute once they are, the
// writer having zeroed it before that push).  With `live` (moe_route.LiveTokens) the live tokens' bit mask (bit r % 16
// of word r / 16: token row r live; up to 1024 rows) at LIVE, read by the reader with the logits (so there before
// the compute's scores, which the writer waits for), and the reader's own zero row at RZERO.
constexpr uint32_t WZERO = 0, WBASE = 64, RBASE = 128, WOUT = 192, TSYNC = WOUT + 16 * 2 * R;
constexpr uint32_t LIVE = TSYNC + 64, RZERO = LIVE + 256;
constexpr uint32_t SCR_BYTES = RZERO + 64 + 64;

// whether token row r is live in the mask at `mask` (rows without a request choose no expert: their routing row zero)
inline bool live_row(uint32_t mask, uint32_t r) {
    return ((reinterpret_cast<volatile uint32_t*>(mask)[r / 16] >> (r % 16)) & 1u) != 0;
}

// A core's rows over its routers: with `trisc` (3 rows a core or more), the last count / 3 rows by the compute's
// unpack thread (its RISC idle after the scores; the writer writes them), then of the rest the first half (rounded up)
// by the writer and the others by the reader with `share`, else all by the writer.  -> (writer's end, reader's end)
struct Rows {
    uint32_t writer, reader;
};
inline Rows rows_of(uint32_t count, bool trisc, bool share) {
    const uint32_t t = trisc && count >= 3 ? count / 3 : 0;
    const uint32_t rest = count - t;
    return {share ? (rest + 1) / 2 : rest, rest};
}

inline uint32_t align64(uint32_t a) { return (a + 63u) & ~63u; }

// The core's rows j (its token rows start + j) that add the shared expert, [lo, hi): by the local base page's [1]
// `flag` 1 all of them; 2 (the shared expert spread over the mesh columns, moe_route.SHARED_SPLIT) those of the tokens
// part * tokens / parts .. (part + 1) * tokens / parts - 1 (the page's [2] part, [3] parts); else none
struct Span {
    uint32_t lo, hi;
};
inline Span shared_span(
    bool with_shared, uint32_t flag, uint32_t part, uint32_t parts, uint32_t start, uint32_t count, uint32_t tokens) {
    if (!with_shared || flag == 0) {
        return {0, 0};
    }
    if (flag == 1 || parts == 0) {
        return {0, count};
    }
    auto clip = [&](uint32_t v) { return v <= start ? 0u : (v - start < count ? v - start : count); };
    return {clip(part * tokens / parts), clip((part + 1) * tokens / parts)};
}
// fp32 bits -> an unsigned key in the floats' order (negative: all bits flipped; else the sign bit set)
inline uint32_t ordered(uint32_t b) { return b ^ (static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) | 0x80000000u); }

union Bits {
    uint32_t u;
    float f;
};
inline float as_float(uint32_t u) {
    Bits b;
    b.u = u;
    return b.f;
}
inline uint32_t as_bits(float f) {
    Bits b;
    b.f = f;
    return b.u;
}
inline uint32_t round_bf16(uint32_t u) { return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u; }

// the top k of E keys (fp32 bits: block b's 16 at p + b stride) -> ids (largest first, ties to the lower expert) and
// their ordered keys (the comparison moe_decode_route makes); four values loaded at a time (the loads in flight
// together), the experts in increasing order
template <uint32_t k, uint32_t E>
inline void top_k(const uint32_t* p, uint32_t stride, uint32_t* ids, uint32_t* keys) {
    for (uint32_t i = 0; i < k; ++i) {
        keys[i] = 0;
        ids[i] = 0;
    }
    uint32_t kmin = 0;
    // (the insertion out of line: most keys are at most kmin, and the branch past them is then not taken)
    auto offer = [&](uint32_t key, uint32_t e) {
        if (__builtin_expect(key > kmin, 0)) {
            uint32_t pos = k - 1;
            while (pos > 0 && keys[pos - 1] < key) {
                keys[pos] = keys[pos - 1];
                ids[pos] = ids[pos - 1];
                --pos;
            }
            keys[pos] = key;
            ids[pos] = e;
            kmin = keys[k - 1];
        }
    };
    static_assert(E % 16 == 0);
    for (uint32_t b = 0; b < E / 16; ++b) {
        const uint32_t* q = p + b * stride;
#pragma GCC unroll 4
        for (uint32_t i = 0; i < 16; i += 4) {
            const uint32_t b0 = q[i], b1 = q[i + 1], b2 = q[i + 2], b3 = q[i + 3];
            const uint32_t e = 16 * b + i;
            offer(ordered(b0), e);
            offer(ordered(b1), e + 1);
            offer(ordered(b2), e + 2);
            offer(ordered(b3), e + 3);
        }
    }
}

// One token row of the routing as moe_decode_route writes it for a column's experts (local above 16): the top k of the
// E choice keys `ch` (the row's block 0; blocks `stride` words apart, as `sc`), the chosen scores `sc` summed in
// increasing expert order, and for each chosen expert of base .. base + local - 1 its id e - base in columns 0 .. and
// bf16((score * (1 / (sum + 1e-20))) * scale) in columns 16 .., then the shared expert (id local, 1.0); staged at
// `stage` (128 B: the ids' 64 B, the weights').  (No NoC: the compute's unpack thread stages rows too.)
// globals (the progressive form's budget, moe.py GLM53_MOE_PROG_F): the row's k picks over all E experts (+ 1, 0 for
// none: k <= 8) also packed two to a word into id columns 12 .. 15, their weights 0 (every reader stops at the row's
// first zero weight, at most column 9), so that the experts op's builders can count every mesh column's chosen experts
template <uint32_t k, uint32_t E, bool globals = false>
inline void stage_row(
    const uint32_t* sc,
    const uint32_t* ch,
    uint32_t stride,
    uint32_t base,
    uint32_t local,
    float scale,
    bool shared,
    uint32_t stage) {
    uint32_t ids[k], keys[k];
    {
        DeviceZoneScopedN("T-TOPK");
        top_k<k, E>(ch, stride, ids, keys);
    }
    DeviceZoneScopedN("T-REST");
    for (uint32_t i = 1; i < k; ++i) {  // increasing order, so the sum runs as a reduction along the row would
        const uint32_t v = ids[i];
        uint32_t q = i;
        while (q > 0 && ids[q - 1] > v) {
            ids[q] = ids[q - 1];
            --q;
        }
        ids[q] = v;
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < k; ++i) {
        sum += as_float(sc[(ids[i] / 16) * stride + ids[i] % 16]);
    }
    const float inv = 1.0f / (sum + 1e-20f);
    volatile uint32_t* o = reinterpret_cast<volatile uint32_t*>(stage);
    volatile uint32_t* ow = reinterpret_cast<volatile uint32_t*>(stage + R);
    for (uint32_t i = 0; i < 16; ++i) {
        o[i] = 0;
        ow[i] = 0;
    }
    uint32_t q = 0;
    for (uint32_t i = 0; i < k; ++i) {
        const uint32_t e = ids[i];
        if (e >= base && e < base + local) {
            o[q] = e - base;
            ow[q++] = round_bf16(as_bits((as_float(sc[(e / 16) * stride + e % 16]) * inv) * scale));
        }
    }
    if (shared) {
        o[q] = local;
        ow[q] = 0x3f800000u;  // 1.0
    }
    if constexpr (globals) {
        static_assert(k <= 8, "the global picks: 8 at most in columns 12 .. 15");
        for (uint32_t i = 0; i < k; ++i) {
            o[12 + i / 2] = o[12 + i / 2] | ((ids[i] + 1) << (16 * (i % 2)));
        }
    }
}

#ifndef COMPILE_FOR_TRISC
// stage_row, then the row written to row r of w (row_addr(r, half): its columns 16 half ..)
template <uint32_t k, uint32_t E, bool globals, typename RowAddr>
inline void route_row(
    const uint32_t* sc,
    const uint32_t* ch,
    uint32_t stride,
    uint32_t base,
    uint32_t local,
    float scale,
    bool shared,
    uint32_t stage,
    RowAddr&& row_addr,
    uint32_t r) {
    stage_row<k, E, globals>(sc, ch, stride, base, local, scale, shared, stage);
    noc_async_write(stage, row_addr(r, 0), R);
    noc_async_write(stage + R, row_addr(r, 1), R);
}
#endif
}  // namespace moe_route
