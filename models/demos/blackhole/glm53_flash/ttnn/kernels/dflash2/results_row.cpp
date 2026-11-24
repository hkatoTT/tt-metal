// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's step results, this mesh row's row, in one op (dflash2_step.gathered_results with the fused acceptance,
// GLM53_DFLASH2_ACCEPT_FUSED): the acceptance's own row `both` [1, 1, 1, 2 U] fp32 row-major (n, then the bonus tokens:
// verify_accept.accept_inputs's before its all-gather) and the drafter's block [U, 8] uint32 row-major (one page a user)
// -> [1, 1, 1, 10 U] fp32 row-major: n (U), bonus (U), then each user's 8 draft words as fp32 (whole numbers below
// 2^24: exact); the all-gather of it over the rows (ttnn's) is gathered_results's tensor, byte for byte, in place of
// the acceptance's all-gather, the drafts' reshape / tilize / typecast / untilize, their all-gather and the concat.
// One core a chip.  No multicast.
//   CT: U, the CB.  RT: both's address and bytes, the drafts' address and page bytes (32), the output's address and bytes.

namespace {
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
    constexpr uint32_t cb = get_compile_time_arg_val(1);
    constexpr uint32_t SLOT = 64;

    const uint32_t both_addr = get_arg_val<uint32_t>(0);
    const uint32_t both_bytes = get_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_arg_val<uint32_t>(2);
    const uint32_t d_page = get_arg_val<uint32_t>(3);
    const uint32_t out_addr = get_arg_val<uint32_t>(4);
    const uint32_t out_bytes = get_arg_val<uint32_t>(5);

    // L1: both, the drafts' pages (each in a 64-byte slot: a DRAM page is 64-byte aligned, a small read lands at the same
    // offset mod 64), the output row
    auto up = [](uint32_t a) { return (a + SLOT - 1) & ~(SLOT - 1); };
    const uint32_t l1_both = up(get_write_ptr(cb));
    const uint32_t l1_d = up(l1_both + both_bytes);
    const uint32_t l1_out = up(l1_d + U * SLOT);
    const InterleavedAddrGen<true> s_both{.bank_base_address = both_addr, .page_size = both_bytes};
    const InterleavedAddrGen<true> s_d{.bank_base_address = d_addr, .page_size = d_page};
    noc_async_read(s_both.get_noc_addr(0), l1_both, both_bytes);
    for (uint32_t u = 0; u < U; ++u) {
        noc_async_read(s_d.get_noc_addr(u), l1_d + u * SLOT, d_page);
    }
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* both = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_both);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_out);
    for (uint32_t i = 0; i < 2 * U; ++i) {
        out[i] = both[i];
    }
    for (uint32_t u = 0; u < U; ++u) {
        const volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_d + u * SLOT);
        for (uint32_t j = 0; j < 8; ++j) {
            out[2 * U + 8 * u + j] = u2f(d[j]);
        }
    }
    const InterleavedAddrGen<true> s_out{.bank_base_address = out_addr, .page_size = out_bytes};
    noc_async_write(l1_out, s_out.get_noc_addr(0), out_bytes);
    noc_async_write_barrier();
}
