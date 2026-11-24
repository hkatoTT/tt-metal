// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// KDA sequence-parallel prefill (kda_prefill_ops.sp_pick): a mesh row's segment's conv
// history (`inc`, row-major [1, K, C] bf16: the conv op's carry) and its request's next wave history (rows 0 .. K - 1
// of the wave history's tiles [1, K, C] bf16, written in place), each row a copy of one row of the block the 0/1
// picks of `TtKda._prefill_sp` read: rows 0 .. K - 1 the wave history (zero where the row's request begins: keep 0),
// rows 32 + K r + j the gathered last inputs (gathered [1, 1, 32, C] tiled: row K r + j mesh row r's j-th), in place of
// the history multiply, untilize, concat, tilize, concat, the two 0/1 matmuls, the untilize and the copy (a -0 comes
// out +0, as the 0/1 products' sum).  Table (row-major int32 per chip): [inc_0 .. inc_{K-1}, end_0 .. end_{K-1}, keep]
// (the block's rows).  Each 32-byte face row is read into a 64-byte slot of L1 at the source's offset mod 64 and
// written from there (16-byte aligned at both ends); the history's tiles of this core are read before any is written.
// This core: tiles t0 .. t1 - 1 across C.
void kernel_main() {
    const uint32_t gathered_addr = get_arg_val<uint32_t>(0);
    const uint32_t hist_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t inc_addr = get_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(4);
    const uint32_t t1 = get_arg_val<uint32_t>(5);

    constexpr uint32_t rows = get_compile_time_arg_val(0);        // K (< 16)
    constexpr uint32_t table_page = get_compile_time_arg_val(1);  // the table's page (bytes)
    constexpr uint32_t per = get_compile_time_arg_val(2);         // a core's tiles at most
    constexpr uint32_t cb = get_compile_time_arg_val(3);
    constexpr uint32_t inc_page = get_compile_time_arg_val(4);    // the carry's row (bytes: C bf16)
    constexpr auto g_args = TensorAccessorArgs<5>();
    constexpr auto hist_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<hist_args.next_compile_time_args_offset()>();
    constexpr auto inc_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048, FACE = 512, FROW = 32, SLOT = 64;
    constexpr uint32_t outs_n = 2 * rows;  // inc rows, then the history's
    constexpr uint32_t slots = outs_n * per * 2;

    const uint32_t l1 = get_write_ptr(cb);
    const auto s_table = TensorAccessor(table_args, table_addr, table_page);
    noc_async_read(s_table.get_noc_addr(0), l1, table_page);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1);
    uint32_t src[outs_n];
    for (uint32_t o = 0; o < outs_n; ++o) {
        src[o] = static_cast<uint32_t>(table[o]);
    }
    const bool keep = table[outs_n] != 0;
    const uint32_t src_slots = (l1 + table_page + SLOT - 1) & ~(SLOT - 1);  // (64-byte aligned whatever the CB's base)

    const auto s_g = TensorAccessor(g_args, gathered_addr, TILE);
    const auto s_hist = TensorAccessor(hist_args, hist_addr, TILE);
    uint32_t n = 0;
    for (uint32_t o = 0; o < outs_n; ++o) {
        const uint32_t s = src[o];
        const bool from_hist = s < 32;
        const uint32_t row = from_hist ? s : s - 32;
        for (uint32_t t = t0; t < t1; ++t) {
            for (uint32_t h = 0; h < 2; ++h, ++n) {
                if (from_hist && !keep) {  // (a request's first segment: its history is zeros)
                    volatile tt_l1_ptr uint32_t* z =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_slots + n * SLOT + h * FROW);
                    for (uint32_t w = 0; w < FROW / 4; ++w) {
                        z[w] = 0;
                    }
                    continue;
                }
                const uint32_t off = (((row % 32) / 16) * 2 + h) * FACE + (row % 16) * FROW;
                const uint64_t addr = from_hist ? s_hist.get_noc_addr(t, off) : s_g.get_noc_addr(t, off);
                noc_async_read(addr, src_slots + n * SLOT + off % SLOT, FROW);
            }
        }
    }
    noc_async_read_barrier();  // (every history row of this core's tiles read before any is written)

    // each face row (-0 -> +0, as the 0/1 products' sum) to its place: the carry's row o at column tile t (64 bytes,
    // the two face rows one after the other), the history's row o - K in faces 0 / 1 of tile t (its other rows not
    // written: zero since the history was made, nothing writes them)
    const auto s_inc = TensorAccessor(inc_args, inc_addr, inc_page);
    n = 0;
    for (uint32_t o = 0; o < outs_n; ++o) {
        const uint32_t s = src[o];
        const bool zeros = s < 32 && !keep;
        const uint32_t row = s < 32 ? s : s - 32;
        for (uint32_t t = t0; t < t1; ++t) {
            for (uint32_t h = 0; h < 2; ++h, ++n) {
                const uint32_t at = zeros ? h * FROW : ((((row % 32) / 16) * 2 + h) * FACE + (row % 16) * FROW) % SLOT;
                volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_slots + n * SLOT + at);
                for (uint32_t w = 0; w < 16; ++w) {
                    if (v[w] == 0x8000) {
                        v[w] = 0;
                    }
                }
                const uint64_t dst = o < rows ? s_inc.get_noc_addr(o, t * SLOT + h * FROW)
                                              : s_hist.get_noc_addr(t, h * FACE + (o - rows) * FROW);
                noc_async_write(src_slots + n * SLOT + at, dst, FROW);
            }
        }
    }
    noc_async_write_barrier();
}
