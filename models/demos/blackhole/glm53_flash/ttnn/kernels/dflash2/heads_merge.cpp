// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2 (dflash2_ops.heads_merge): the paged sdpa decode's output [1, 2 U, 32, 128] bf16 (tiled; virtual user
// (u, j), row h B + pos: q head G j + h at position pos) back to the block's rows [1, 1, U B, kvh G 128] (tiled; row
// u B + pos, q head G j + h at 128-column block G j + h), the rows past U B zero.  A core takes output tiles: each
// reads the source tile of every user in its tile row and copies the rows' face rows (32 bytes) on the RISC-V.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t i0 = get_arg_val<uint32_t>(2);  // this core's output tiles (tile row major)
    const uint32_t i1 = get_arg_val<uint32_t>(3);

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);
    constexpr uint32_t kvh = get_compile_time_arg_val(2);
    constexpr uint32_t U = get_compile_time_arg_val(3);
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048, FACE = 512, FROW = 32, WT = 4;
    constexpr uint32_t out_wt = kvh * G * WT;  // tiles across the output
    constexpr uint32_t per_tile = 32 / B;      // users a tile row

    const auto s_src = TensorAccessor(src_args, src_addr, TILE);
    const auto s_out = TensorAccessor(out_args, out_addr, TILE);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t srcs = l1;                     // per_tile source tiles
    const uint32_t out = srcs + per_tile * TILE;  // the output tile

    for (uint32_t i = i0; i < i1; ++i) {
        const uint32_t tr = i / out_wt, ct = i % out_wt;
        const uint32_t head = ct / WT, t = ct % WT;  // the chip's q head, its column tile
        const uint32_t j = head / G, h = head % G;
        uint32_t n = 0;
        for (uint32_t a = 0; a < per_tile; ++a) {
            const uint32_t u = tr * per_tile + a;
            if (u < U) {
                noc_async_read(s_src.get_noc_addr((u * kvh + j) * WT + t), srcs + a * TILE, TILE);
                ++n;
            }
        }
        for (uint32_t off = 0; off < TILE; off += MEM_ZEROS_SIZE) {
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), out + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        for (uint32_t a = 0; a < n; ++a) {
            for (uint32_t pos = 0; pos < B; ++pos) {
                const uint32_t rs = h * B + pos, rd = a * B + pos;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t* from = reinterpret_cast<const uint32_t*>(
                        srcs + a * TILE + ((rs / 16) * 2 + f) * FACE + (rs % 16) * FROW);
                    uint32_t* to = reinterpret_cast<uint32_t*>(out + ((rd / 16) * 2 + f) * FACE + (rd % 16) * FROW);
                    for (uint32_t w = 0; w < FROW / 4; ++w) {
                        to[w] = from[w];
                    }
                }
            }
        }
        noc_async_write(out, s_out.get_noc_addr(i), TILE);
        noc_async_write_barrier();
    }
}
