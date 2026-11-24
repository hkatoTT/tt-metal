// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's next step's inputs on the device (next_inputs.py, GLM53_DFLASH2_NEXT_INPUTS): at the end of a step, from
// its acceptance and the drafter's next block, the packed input row (input_pack.py) of the step that continues every
// seat as the serving adapter will ask for it, written over the packed tensor (the next step's `unpack` reads it; the
// host writes it again only where its inputs differ).  One core a chip, this mesh row's row.  No multicast.
//
// Per user u of the row (virtual user v = j U + u its token j, j = 0 .. K), live (bit r B + v of the mesh's live mask,
// the step's own): n and the bonus from the gathered results (row r: n, bonus, then each user's drafter row of 8: t[s],
// its drafts, zeros; fp32 whole numbers), p its position at this step (ctx_pos[u]):
//   token(v) = bonus (j = 0) or draft j;  position(v) = min(p + n + 1 + j, max_pos);  keep(u) = n;
// not live (a seat without a request): token 0, position 0, keep K (generator._dflash_inputs's).  Fields (each a run of
// elements in the packed row: element e at byte offset + (e stride / page words) slot + (e stride % page words) 4):
// kind 0 tokens (B), 1 keep (U), 2 positions (B: DSA's params, word 0 of each, or cur_pos), 3 ctx_pos (B), 4 base1 (U:
// position(u) + 1).
//   CT: U, K, max_pos, the fields' count.  RT: packed address and bytes, the results' address and page bytes, the mesh
//   row r, ctx_pos's address and bytes, the live mask's address and bytes, then per field: kind, byte offset, page
//   words, slot bytes, stride.

namespace {
// a whole number in fp32 (0 <= x < 2^24) as an integer, from its bits (no FPU on this RISC-V)
inline uint32_t f2u(uint32_t bits) {
    const uint32_t e = (bits >> 23) & 0xFF;
    if (e < 127) {
        return 0;
    }
    const uint32_t m = (bits & 0x7FFFFF) | 0x800000;
    const uint32_t s = e - 127;
    return s >= 23 ? m << (s - 23) : m >> (23 - s);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t U = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t max_pos = get_compile_time_arg_val(2);
    constexpr uint32_t n_fields = get_compile_time_arg_val(3);
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr uint32_t B = (K + 1) * U;
    constexpr uint32_t SLOT = 64;

    const uint32_t packed_addr = get_arg_val<uint32_t>(0);
    const uint32_t packed_bytes = get_arg_val<uint32_t>(1);
    const uint32_t res_addr = get_arg_val<uint32_t>(2);
    const uint32_t res_page = get_arg_val<uint32_t>(3);
    const uint32_t row = get_arg_val<uint32_t>(4);
    const uint32_t ctx_addr = get_arg_val<uint32_t>(5);
    const uint32_t ctx_bytes = get_arg_val<uint32_t>(6);
    const uint32_t live_addr = get_arg_val<uint32_t>(7);
    const uint32_t live_bytes = get_arg_val<uint32_t>(8);

    // L1: the packed row, the results' row, ctx_pos, the live mask (each at a 64-byte aligned address; every source is
    // a DRAM page, 64-byte aligned)
    auto up = [](uint32_t a) { return (a + SLOT - 1) & ~(SLOT - 1); };
    const uint32_t l1_packed = up(get_write_ptr(cb));
    const uint32_t l1_res = up(l1_packed + packed_bytes);
    const uint32_t l1_ctx = up(l1_res + res_page);
    const uint32_t l1_live = up(l1_ctx + ctx_bytes);
    const InterleavedAddrGen<true> s_packed{.bank_base_address = packed_addr, .page_size = packed_bytes};
    const InterleavedAddrGen<true> s_res{.bank_base_address = res_addr, .page_size = res_page};
    const InterleavedAddrGen<true> s_ctx{.bank_base_address = ctx_addr, .page_size = ctx_bytes};
    const InterleavedAddrGen<true> s_live{.bank_base_address = live_addr, .page_size = live_bytes};
    noc_async_read(s_packed.get_noc_addr(0), l1_packed, packed_bytes);
    noc_async_read(s_res.get_noc_addr(row), l1_res, res_page);
    noc_async_read(s_ctx.get_noc_addr(0), l1_ctx, ctx_bytes);
    noc_async_read(s_live.get_noc_addr(0), l1_live, live_bytes);
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_res);
    const volatile tt_l1_ptr int32_t* ctx = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1_ctx);
    const volatile tt_l1_ptr uint32_t* live = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_live);
    uint32_t tok[B], pos[B], keep[U];
    for (uint32_t u = 0; u < U; ++u) {
        const uint32_t v0 = row * B + u;  // (the mesh token of user u's token 0; its K others alike)
        const bool on = ((live[v0 / 16] >> (v0 % 16)) & 1u) != 0;
        if (!on) {
            for (uint32_t j = 0; j <= K; ++j) {
                tok[j * U + u] = 0;
                pos[j * U + u] = 0;
            }
            keep[u] = K;
            continue;
        }
        const uint32_t n = f2u(res[u]);
        const uint32_t base = static_cast<uint32_t>(ctx[u]) + n + 1;
        tok[u] = f2u(res[U + u]);
        for (uint32_t j = 1; j <= K; ++j) {
            tok[j * U + u] = f2u(res[2 * U + 8 * u + j]);
        }
        for (uint32_t j = 0; j <= K; ++j) {
            const uint32_t p = base + j;
            pos[j * U + u] = p < max_pos ? p : max_pos;
        }
        keep[u] = n;
    }

    for (uint32_t f = 0; f < n_fields; ++f) {
        const uint32_t a = 9 + 5 * f;
        const uint32_t kind = get_arg_val<uint32_t>(a);
        const uint32_t offset = get_arg_val<uint32_t>(a + 1);
        const uint32_t page_words = get_arg_val<uint32_t>(a + 2);
        const uint32_t slot = get_arg_val<uint32_t>(a + 3);
        const uint32_t stride = get_arg_val<uint32_t>(a + 4);
        const uint32_t count = kind == 1 || kind == 4 ? U : B;
        for (uint32_t e = 0; e < count; ++e) {
            const uint32_t i = e * stride;
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                l1_packed + offset + (i / page_words) * slot + (i % page_words) * 4);
            uint32_t value = 0;
            switch (kind) {
                case 0: value = tok[e]; break;
                case 1: value = keep[e]; break;
                case 4: value = pos[e] + 1; break;
                default: value = pos[e]; break;  // 2, 3
            }
            *w = value;
        }
    }
    noc_async_write(l1_packed, s_packed.get_noc_addr(0), packed_bytes);
    noc_async_write_barrier();
}
