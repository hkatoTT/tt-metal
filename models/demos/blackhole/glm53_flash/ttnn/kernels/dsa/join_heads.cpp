// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DSA prefill (dsa_latent.join_heads): the heads' outputs [1, H, T0, D] and (M1 > 0) [1, H, T1, D] bf16 (tiled) into
// [1, 1, T0 + T1, H D] (tiled), part 0's rows first: whole tiles moved (output tile r H W + h W + j = part 0's tile
// (h M0 + r) W + j for r < M0, else part 1's (h M1 + r - M0) W + j; M0 / M1 tile rows, W = D / 32 tiles a head), as
// ttnn.concat of the parts over the rows and nlp_concat_heads.  A core takes a range of output tiles, read in groups of
// up to `group` into L1 and written back.
void kernel_main() {
    const uint32_t a0_addr = get_arg_val<uint32_t>(0);
    const uint32_t a1_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t p0 = get_arg_val<uint32_t>(3);  // this core's output tiles [p0, p1)
    const uint32_t p1 = get_arg_val<uint32_t>(4);

    constexpr uint32_t M0 = get_compile_time_arg_val(0);  // part 0's tile rows
    constexpr uint32_t M1 = get_compile_time_arg_val(1);  // part 1's (0: none)
    constexpr uint32_t H = get_compile_time_arg_val(2);   // heads
    constexpr uint32_t W = get_compile_time_arg_val(3);   // tiles a head
    constexpr uint32_t group = get_compile_time_arg_val(4);
    constexpr uint32_t cb = get_compile_time_arg_val(5);
    constexpr auto a0_args = TensorAccessorArgs<6>();
    constexpr auto a1_args = TensorAccessorArgs<a0_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<a1_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048;

    const auto s_a0 = TensorAccessor(a0_args, a0_addr, TILE);
    const auto s_a1 = TensorAccessor(a1_args, a1_addr, TILE);
    const auto s_out = TensorAccessor(out_args, out_addr, TILE);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t g = p0; g < p1; g += group) {
        const uint32_t n = p1 - g < group ? p1 - g : group;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t o = g + i;
            const uint32_t c = o % (H * W), r = o / (H * W), h = c / W, j = c % W;
            if (M1 == 0 || r < M0) {
                noc_async_read(s_a0.get_noc_addr((h * M0 + r) * W + j), l1 + i * TILE, TILE);
            } else {
                noc_async_read(s_a1.get_noc_addr((h * M1 + (r - M0)) * W + j), l1 + i * TILE, TILE);
            }
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(l1 + i * TILE, s_out.get_noc_addr(g + i), TILE);
        }
        noc_async_write_barrier();
    }
}
