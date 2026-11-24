// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DSA indexer (dsa_indexer.py): rows r0 .. r1 - 1 of `src` (row-major: a token's latent, or a pool's key) into the
// rows `rows` names in a row-major cache (`rows` [n] uint32: the row, 0xFFFFFFFF for none), through this core's L1
// (`cb`: `block` rows a pass, then the rows page).  FROM_PARAMS: `rows` is the indexer's params ([n] pages of PARAMS
// uint32: position, base, ..): row base + position (position 0xFFFFFFFF: none), each pass's pages read with its rows.
// MUL, ADD (after the accessors; 1, 0: every row as named): the cache holds only the rows r with r mod MUL = ADD, row r
// at r / MUL (the split latents: token t of a user on mesh column t mod 4, its user's base a multiple of
// 4), the other rows skipped.
//   CT: block, row bytes, the rows (or params) page, cb, FROM_PARAMS, then the accessors, MUL, ADD.
//   RT: src, dst, rows, r0, r1
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t rows_addr = get_arg_val<uint32_t>(2);
    const uint32_t r0 = get_arg_val<uint32_t>(3);
    const uint32_t r1 = get_arg_val<uint32_t>(4);

    constexpr uint32_t block = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t rows_page = get_compile_time_arg_val(2);  // the rows tensor's page (bytes)
    constexpr uint32_t cb = get_compile_time_arg_val(3);
    constexpr uint32_t from_params = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto rows_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr uint32_t MUL = get_compile_time_arg_val(rows_args.next_compile_time_args_offset());
    constexpr uint32_t ADD = get_compile_time_arg_val(rows_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t SENT = 0xFFFFFFFFu;

    const auto s_src = TensorAccessor(src_args, src_addr, row_bytes);
    const auto s_dst = TensorAccessor(dst_args, dst_addr, row_bytes);
    const auto s_rows = TensorAccessor(rows_args, rows_addr, rows_page);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t rows_l1 = l1 + block * row_bytes;
    if constexpr (!from_params) {
        noc_async_read(s_rows.get_noc_addr(0), rows_l1, rows_page);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1);
    for (uint32_t b = r0; b < r1; b += block) {
        const uint32_t n = r1 - b < block ? r1 - b : block;
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(s_src.get_noc_addr(b + i), l1 + i * row_bytes, row_bytes);
            if constexpr (from_params) {  // (a page a row: 64 B, DRAM-aligned)
                noc_async_read(s_rows.get_noc_addr(b + i), rows_l1 + i * rows_page, rows_page);
            }
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t r;
            if constexpr (from_params) {
                volatile tt_l1_ptr uint32_t* p =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1 + i * rows_page);
                r = p[0] == SENT ? SENT : p[1] + p[0];
            } else {
                r = rows[b + i];
            }
            if constexpr (MUL > 1) {
                r = r != SENT && r % MUL == ADD ? r / MUL : SENT;
            }
            if (r != SENT) {
                noc_async_write(l1 + i * row_bytes, s_dst.get_noc_addr(r), row_bytes);
            }
        }
        noc_async_write_barrier();
    }
}
