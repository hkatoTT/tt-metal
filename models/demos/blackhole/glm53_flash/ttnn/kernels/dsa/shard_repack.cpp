// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// The split latents' partials for the row's exchange (dsa_indexer.shard_repack): sparse_split's
// partials of 32-head units (unit u = (2 s + h) G + j: query s's heads 32 h .. 32 h + 31, split j; PART tiles each, a
// head a row) -> the exchange tensor [1, 1, 4 R, PART x 32] tiles, chip d's chunk (tile rows d Q G / 2 ..) its 16 heads
// 16 d .. of every (s, j): item m = s G + j in tile row m / 2, rows 16 (m mod 2) .. (a tile's top half is its faces 0
// and 1, its first 1 KB; the bottom half its last 1 KB).  Units u0 .. u1 - 1 a core: read a unit's tiles, write each
// half to its chip's chunk.
//   CT: G, Q, PART, cb, then the accessors of the partials and the exchange tensor.  RT: partials, out, u0, u1
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t u1 = get_arg_val<uint32_t>(3);

    constexpr uint32_t G = get_compile_time_arg_val(0);
    constexpr uint32_t Q = get_compile_time_arg_val(1);
    constexpr uint32_t PART = get_compile_time_arg_val(2);
    constexpr uint32_t cb = get_compile_time_arg_val(3);
    constexpr auto i_args = TensorAccessorArgs<4>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048, HALF = 1024;
    constexpr uint32_t CHUNK_ROWS = Q * G / 2;  // tile rows a chip's chunk
    static_assert((Q * G) % 2 == 0, "shard_repack: Q G even (two items a tile)");

    const auto s_in = TensorAccessor(i_args, in_addr, TILE);
    const auto s_out = TensorAccessor(o_args, out_addr, TILE);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t u = u0; u < u1; ++u) {
        for (uint32_t t = 0; t < PART; ++t) {
            noc_async_read(s_in.get_noc_addr(u * PART + t), l1 + t * TILE, TILE);
        }
        noc_async_read_barrier();
        const uint32_t sq = u / G, j = u % G, s = sq / 2, h = sq % 2;
        const uint32_t m = s * G + j;
        for (uint32_t hh = 0; hh < 2; ++hh) {
            const uint32_t d = 2 * h + hh;
            const uint32_t page = (d * CHUNK_ROWS + m / 2) * PART;
            for (uint32_t t = 0; t < PART; ++t) {
                noc_async_write(l1 + t * TILE + hh * HALF, s_out.get_noc_addr(page + t, (m % 2) * HALF), HALF);
            }
        }
        noc_async_write_barrier();
    }
}
