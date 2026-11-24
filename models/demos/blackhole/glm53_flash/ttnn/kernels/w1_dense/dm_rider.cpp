// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// A rider (GLM53_W1D, ttnn/w1_dense.py): on a core another program leaves idle, part of this core's weight
// stream of a decode_matmul to come, read from DRAM into an L1 region.  The stream: this core's decode_matmul_streams
// entry (K tiles k0 .. k0 + kc - 1, N tiles n0 + i stride, i < ncb, of the [K / 32, N / 32] tiles, in chunks of cr K
// rows x sb columns, per sub-block of sb columns, per chunk of cr rows); chunks c0 .. c1 - 1, chunk c at region + 64 +
// (c - cbase) slot, laid out as decode_matmul_weights.cpp lays a chunk in its ring slot (row r at r sb tiles; a short
// last sub-block or chunk leaves the rest of its slot unread).  The seal (the region's first word) is 0 while the reads
// are in flight, then `seal` ((tag << 16) | c1: the decode_matmul checks it and takes the chunks only on a match).
// The program this kernel rides in never touches the region (an L1 tensor held from its first rider to its
// decode_matmul: tt-metal keeps every program's circular buffers below it in that window).  Compile-time arguments: the
// tile's bytes, then the weight's accessor (a BFP8 interleaved DRAM tensor).
void kernel_main() {
    constexpr uint32_t tile = get_compile_time_arg_val(0);
    constexpr auto w_args = TensorAccessorArgs<1>();
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t region = get_arg_val<uint32_t>(1);
    const uint32_t seal_value = get_arg_val<uint32_t>(2);
    const uint32_t k0 = get_arg_val<uint32_t>(3);
    const uint32_t n0 = get_arg_val<uint32_t>(4);
    const uint32_t ncb = get_arg_val<uint32_t>(5);
    const uint32_t nt = get_arg_val<uint32_t>(6);
    const uint32_t kc = get_arg_val<uint32_t>(7);
    const uint32_t cr = get_arg_val<uint32_t>(8);
    const uint32_t sb = get_arg_val<uint32_t>(9);
    const uint32_t stride = get_arg_val<uint32_t>(10);
    const uint32_t c0 = get_arg_val<uint32_t>(11);
    const uint32_t c1 = get_arg_val<uint32_t>(12);
    const uint32_t cbase = get_arg_val<uint32_t>(13);
    const auto s_w = TensorAccessor(w_args, w_addr);
    const uint32_t per_sub = (kc + cr - 1) / cr, bytes = cr * sb * tile;
    volatile tt_l1_ptr uint32_t* seal = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(region);
    seal[0] = 0;
    for (uint32_t c = c0; c < c1; ++c) {
        const uint32_t dst = region + 64 + (c - cbase) * bytes;
        const uint32_t col0 = (c / per_sub) * sb;
        const uint32_t sbw = ncb - col0 < sb ? ncb - col0 : sb;
        const uint32_t g = c % per_sub;
        const uint32_t rows = kc - g * cr < cr ? kc - g * cr : cr;
        const uint32_t row0 = k0 + g * cr;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t page = (row0 + r) * nt + n0 + col0 * stride;
            if (stride != 1) {  // pages page, page + B, ..: adjacent in one bank, one request
                noc_async_read(s_w.get_noc_addr(page, 0, noc_index), dst + r * sb * tile, sbw * tile, noc_index);
                continue;
            }
            for (uint32_t i = 0; i < sbw; ++i) {
                noc_async_read(s_w.get_noc_addr(page + i), dst + (r * sb + i) * tile, tile);
            }
        }
    }
    noc_async_read_barrier();
    seal[0] = seal_value;
}
