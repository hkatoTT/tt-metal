// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DSA decode (dsa_latent.q_heads): q [1, 1, B, H D] bf16 (tiled) into the heads [1, H R, B, D] (tiled), head h's
// copy s (R copies of each, s < R) = q's columns h D .. (h + 1) D: whole tiles moved (output tile ((h R + s) M + r) W + j
// = q's tile r H W + h W + j; M tile rows, W = D / 32 tiles a head).  A core takes a range of output tiles, read in
// groups of up to `group` into L1 and written back.  `r0` (the prefill's waves): q's tile
// rows from r0 (the output's tile row r = q's r0 + r; 0: from the first).
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t p0 = get_arg_val<uint32_t>(2);  // this core's output tiles [p0, p1)
    const uint32_t p1 = get_arg_val<uint32_t>(3);
    const uint32_t r0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t M = get_compile_time_arg_val(0);  // tile rows
    constexpr uint32_t H = get_compile_time_arg_val(1);  // heads
    constexpr uint32_t W = get_compile_time_arg_val(2);  // tiles a head
    constexpr uint32_t R = get_compile_time_arg_val(3);  // copies of each head
    constexpr uint32_t group = get_compile_time_arg_val(4);
    constexpr uint32_t cb = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048;

    const auto s_src = TensorAccessor(src_args, src_addr, TILE);
    const auto s_out = TensorAccessor(out_args, out_addr, TILE);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t g = p0; g < p1; g += group) {
        const uint32_t n = p1 - g < group ? p1 - g : group;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t p = g + i;
            const uint32_t j = p % W, r = (p / W) % M, hs = p / (W * M), h = hs / R;
            noc_async_read(s_src.get_noc_addr(((r0 + r) * H + h) * W + j), l1 + i * TILE, TILE);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(l1 + i * TILE, s_out.get_noc_addr(g + i), TILE);
        }
        noc_async_write_barrier();
    }
}
