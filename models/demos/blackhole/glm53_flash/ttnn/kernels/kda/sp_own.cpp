// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// KDA sequence-parallel prefill (kda_prefill_ops.sp_own): a mesh row's last K conv inputs
// (the K rows its segment's next history would hold) as rows 0 .. K - 1 of tiles [1, 1, K, C] bf16, each row a copy of
// one source row picked by the table: a row of the chunk's projection (proj [1, T, F] tiled, F >= C: its first C
// columns) or of the wave history (hist [1, K, C] tiled; zero where the row's request begins: keep 0), in place of
// `TtKda._prefill_sp`'s history multiply, untilize, concat, tilize, the two 0/1 matmuls, slice and add (one 1 and zeros
// a product: exact, a -0 comes out +0, which the copies do too).  Table (row-major int32 per chip): [src_0 ..
// src_{K-1}, keep]: src >= 0 the projection's row, src < 0 the history's row -src - 1.  Per tile across C a row is two
// face rows: each 32-byte face row is read into a 64-byte slot of L1 at the source's offset mod 64 (a NoC transfer
// under 64 bytes needs the same offset mod 64 at both ends) and written from there to its output face row (16-byte
// aligned at both ends); the output's other rows are not written.
// This core: tiles t0 .. t1 - 1.
void kernel_main() {
    const uint32_t proj_addr = get_arg_val<uint32_t>(0);
    const uint32_t hist_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(4);
    const uint32_t t1 = get_arg_val<uint32_t>(5);

    constexpr uint32_t row_tiles = get_compile_time_arg_val(0);   // the projection's tiles across F
    constexpr uint32_t rows = get_compile_time_arg_val(1);        // K (< 16)
    constexpr uint32_t table_page = get_compile_time_arg_val(2);  // the table's page (bytes)
    constexpr uint32_t per = get_compile_time_arg_val(3);         // a core's tiles at most
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr auto proj_args = TensorAccessorArgs<5>();
    constexpr auto hist_args = TensorAccessorArgs<proj_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<hist_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048, FACE = 512, FROW = 32, SLOT = 64;
    constexpr uint32_t slots = rows * per * 2;
    static_assert(FROW % 16 == 0, "face rows: 16-byte NoC writes");

    const uint32_t l1 = get_write_ptr(cb);
    const auto s_table = TensorAccessor(table_args, table_addr, table_page);
    noc_async_read(s_table.get_noc_addr(0), l1, table_page);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1);
    int32_t src[rows];
    for (uint32_t r = 0; r < rows; ++r) {
        src[r] = table[r];
    }
    const bool keep = table[rows] != 0;
    const uint32_t src_slots = (l1 + table_page + SLOT - 1) & ~(SLOT - 1);  // (64-byte aligned whatever the CB's base)

    const auto s_proj = TensorAccessor(proj_args, proj_addr, TILE);
    const auto s_hist = TensorAccessor(hist_args, hist_addr, TILE);
    const auto s_out = TensorAccessor(out_args, out_addr, TILE);
    uint32_t n = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const int32_t s = src[r];
        const bool from_proj = s >= 0;
        const uint32_t row = from_proj ? static_cast<uint32_t>(s) : static_cast<uint32_t>(-s - 1);
        for (uint32_t t = t0; t < t1; ++t) {
            for (uint32_t h = 0; h < 2; ++h, ++n) {
                const uint32_t off = (((row % 32) / 16) * 2 + h) * FACE + (row % 16) * FROW;
                if (!from_proj && !keep) {  // (a request's first segment: its history is zeros)
                    volatile tt_l1_ptr uint32_t* z =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_slots + n * SLOT + h * FROW);
                    for (uint32_t w = 0; w < FROW / 4; ++w) {
                        z[w] = 0;
                    }
                    continue;
                }
                const uint64_t addr = from_proj ? s_proj.get_noc_addr((row / 32) * row_tiles + t, off)
                                                : s_hist.get_noc_addr(t, off);
                noc_async_read(addr, src_slots + n * SLOT + off % SLOT, FROW);
            }
        }
    }
    noc_async_read_barrier();

    // each face row (-0 -> +0, as the 0/1 products' sum) into row r of the output tile's faces 0 / 1 (the other rows
    // are not written: `col_gather` takes rows 0 .. K - 1 only)
    n = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const int32_t s = src[r];
        const bool zeros = s < 0 && !keep;
        const uint32_t row = s >= 0 ? static_cast<uint32_t>(s) : static_cast<uint32_t>(-s - 1);
        for (uint32_t t = t0; t < t1; ++t) {
            for (uint32_t h = 0; h < 2; ++h, ++n) {
                const uint32_t at = zeros ? h * FROW : ((((row % 32) / 16) * 2 + h) * FACE + (row % 16) * FROW) % SLOT;
                volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_slots + n * SLOT + at);
                for (uint32_t w = 0; w < 16; ++w) {
                    if (v[w] == 0x8000) {
                        v[w] = 0;
                    }
                }
                noc_async_write(src_slots + n * SLOT + at, s_out.get_noc_addr(t, h * FACE + r * FROW), FROW);
            }
        }
    }
    noc_async_write_barrier();
}
