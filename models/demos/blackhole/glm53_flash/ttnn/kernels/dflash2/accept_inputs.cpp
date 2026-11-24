// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's acceptance and the drafter's inputs in one op (verify_accept.accept_inputs, GLM53_DFLASH2_ACCEPT_FUSED): in
// place of accept_device's and drafter_inputs's ~40 small ttnn ops.  One core a chip, this mesh row's row r.  No
// multicast, no fabric (the all-gather of `both` over the rows stays ttnn's).
//
// Per user u of the row (virtual user v = j U + u its token j, j = 0 .. K): g(v) the model's greedy id there (the
// verify step's ids [1, 1, 8 B, 1] fp32 TILE, every row's, row r's at r B + v: column 0 of the tiles), t(v) the step's
// own token (io.ids [1, B] uint32 row-major):
//   n(u) = the leading j = 1 .. K with t(j U + u) == g((j - 1) U + u);  bonus(u) = g(n U + u)
// (accept_device's cumprod / sum / pick, exact on these whole numbers).  Outputs (each one DRAM page):
//   both [1, 1, 1, 2 U] fp32 row-major: n, then bonus (accept_device's before its all-gather);
//   bonus [1, U] uint32 row-major, starts [1, U] int32 row-major = base1(u) + n(u) (drafter_inputs's).
// With `forced` (a test's forced acceptance, verify_accept.forced_accept's n [1, 1, 1, U] fp32 TILE, made by its ttnn
// ops): n(u) read from it in place of the drafts' match (whole numbers 0 .. K), the rest as above.
//   CT: U, K, the CB, forced (0 / 1).  RT: ids' address, the first tile and the tile count of row r's span, tokens'
//   address and bytes, base1's address and bytes, the row r, both's / bonus's / starts's addresses and bytes, then
//   (forced) n's address.

namespace {
// a whole number in fp32 (0 <= x < 2^24) as an integer, from its bits (no FPU here)
inline uint32_t f2u(uint32_t bits) {
    const uint32_t e = (bits >> 23) & 0xFF;
    if (e < 127) {
        return 0;
    }
    const uint32_t m = (bits & 0x7FFFFF) | 0x800000;
    const uint32_t s = e - 127;
    return s >= 23 ? m << (s - 23) : m >> (23 - s);
}
// an integer below 2^24 as fp32 bits (exact)
inline uint32_t u2f(uint32_t x) {
    if (x == 0) {
        return 0;
    }
    uint32_t e = 31;
    while (((x >> e) & 1u) == 0) {
        --e;
    }
    return ((e + 127) << 23) | ((x << (23 - e)) & 0x7FFFFF);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t U = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t cb = get_compile_time_arg_val(2);
    constexpr bool forced = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t NT = (U + 31) / 32;  // (forced) n's tiles
    constexpr uint32_t B = (K + 1) * U;
    constexpr uint32_t SLOT = 64, TILE = 4096;

    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile0 = get_arg_val<uint32_t>(1);
    const uint32_t tiles = get_arg_val<uint32_t>(2);
    const uint32_t tok_addr = get_arg_val<uint32_t>(3);
    const uint32_t tok_bytes = get_arg_val<uint32_t>(4);
    const uint32_t b1_addr = get_arg_val<uint32_t>(5);
    const uint32_t b1_bytes = get_arg_val<uint32_t>(6);
    const uint32_t row = get_arg_val<uint32_t>(7);
    const uint32_t both_addr = get_arg_val<uint32_t>(8);
    const uint32_t both_bytes = get_arg_val<uint32_t>(9);
    const uint32_t bonus_addr = get_arg_val<uint32_t>(10);
    const uint32_t bonus_bytes = get_arg_val<uint32_t>(11);
    const uint32_t starts_addr = get_arg_val<uint32_t>(12);
    const uint32_t starts_bytes = get_arg_val<uint32_t>(13);

    // L1 (each piece at a 64-byte aligned address: every source and destination is a DRAM page, 64-byte aligned)
    auto up = [](uint32_t a) { return (a + SLOT - 1) & ~(SLOT - 1); };
    const uint32_t l1_ids = up(get_write_ptr(cb));
    const uint32_t l1_tok = up(l1_ids + tiles * TILE);
    const uint32_t l1_b1 = up(l1_tok + tok_bytes);
    const uint32_t l1_both = up(l1_b1 + b1_bytes);
    const uint32_t l1_bonus = up(l1_both + both_bytes);
    const uint32_t l1_starts = up(l1_bonus + bonus_bytes);
    const uint32_t l1_n = up(l1_starts + starts_bytes);
    const InterleavedAddrGen<true> s_ids{.bank_base_address = ids_addr, .page_size = TILE};
    const InterleavedAddrGen<true> s_tok{.bank_base_address = tok_addr, .page_size = tok_bytes};
    const InterleavedAddrGen<true> s_b1{.bank_base_address = b1_addr, .page_size = b1_bytes};
    for (uint32_t t = 0; t < tiles; ++t) {
        noc_async_read(s_ids.get_noc_addr(tile0 + t), l1_ids + t * TILE, TILE);
    }
    noc_async_read(s_tok.get_noc_addr(0), l1_tok, tok_bytes);
    noc_async_read(s_b1.get_noc_addr(0), l1_b1, b1_bytes);
    if constexpr (forced) {
        const InterleavedAddrGen<true> s_n{.bank_base_address = get_arg_val<uint32_t>(14), .page_size = TILE};
        for (uint32_t t = 0; t < NT; ++t) {
            noc_async_read(s_n.get_noc_addr(t), l1_n + t * TILE, TILE);
        }
    }
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_ids);
    const volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_tok);
    const volatile tt_l1_ptr int32_t* b1 = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1_b1);
    volatile tt_l1_ptr uint32_t* both = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_both);
    volatile tt_l1_ptr uint32_t* bonus = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_bonus);
    volatile tt_l1_ptr int32_t* starts = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1_starts);
    // g(v): row i = r B + v of the [8 B, 1] tiles, column 0: tile i / 32 (from tile0), face 0 (rows 0-15) or 2
    // (rows 16-31), element (i % 16) 16 of the 16 x 16 face
    auto g = [&](uint32_t v) -> uint32_t {
        const uint32_t i = row * B + v;
        const uint32_t ri = i % 32;
        const uint32_t word = (i / 32 - tile0) * 1024 + (ri / 16) * 512 + (ri % 16) * 16;
        return f2u(ids[word]);
    };
    const volatile tt_l1_ptr uint32_t* fn = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_n);
    for (uint32_t u = 0; u < U; ++u) {
        uint32_t n = 0;
        if constexpr (forced) {  // (row 0 of [1, U]: tile u / 32, face 0 or 1, element u % 16)
            n = f2u(fn[(u / 32) * 1024 + ((u % 32) / 16) * 256 + (u % 16)]);
        } else {
            while (n < K && tok[(n + 1) * U + u] == g(n * U + u)) {
                ++n;
            }
        }
        const uint32_t b = g(n * U + u);
        both[u] = u2f(n);
        both[U + u] = u2f(b);
        bonus[u] = b;
        starts[u] = b1[u] + static_cast<int32_t>(n);
    }
    const InterleavedAddrGen<true> s_both{.bank_base_address = both_addr, .page_size = both_bytes};
    const InterleavedAddrGen<true> s_bonus{.bank_base_address = bonus_addr, .page_size = bonus_bytes};
    const InterleavedAddrGen<true> s_starts{.bank_base_address = starts_addr, .page_size = starts_bytes};
    noc_async_write(l1_both, s_both.get_noc_addr(0), both_bytes);
    noc_async_write(l1_bonus, s_bonus.get_noc_addr(0), bonus_bytes);
    noc_async_write(l1_starts, s_starts.get_noc_addr(0), starts_bytes);
    noc_async_write_barrier();
}
