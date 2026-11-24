// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DSA indexer, the scores split over a row's chips (dsa_indexer.shard_candidates, the split scores):
// for queries q0 .. q1 - 1, this chip's top k (topk_large_indices' row of `top`: its columns of `rows`, best first,
// 0xFFFFFFFF after the visible ones) as candidates for the row's choice: out [1, 2, N, k] (row-major, 16-bit words),
// plane 0 their scores (bf16 bits; -inf for none), plane 1 their columns (uint16).  After the row's all_gather (dim 3)
// plane 0's row is the row's 4 k candidates' scores (topk_large_indices picks k of them), plane 1 their columns
// (select_tokens maps a pick to its pool).  The query's whole row of scores is read into L1 and looked up.
//   CT: width (bf16 a row of `rows`), k, cb, then the accessors of rows, top, out; N.  RT: rows, top, out, q0, q1
void kernel_main() {
    const uint32_t rows_addr = get_arg_val<uint32_t>(0);
    const uint32_t top_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t q0 = get_arg_val<uint32_t>(3);
    const uint32_t q1 = get_arg_val<uint32_t>(4);

    constexpr uint32_t width = get_compile_time_arg_val(0);
    constexpr uint32_t k = get_compile_time_arg_val(1);
    constexpr uint32_t cb = get_compile_time_arg_val(2);
    constexpr auto r_args = TensorAccessorArgs<3>();
    constexpr auto t_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr uint32_t n_queries = get_compile_time_arg_val(o_args.next_compile_time_args_offset());  // N
    constexpr uint32_t SENT = 0xFFFFFFFFu;
    constexpr uint16_t NEG_INF = 0xFF80u;
    constexpr uint32_t ROW = width * 2, TOP = k * 4, OUT = k * 2;

    const auto s_rows = TensorAccessor(r_args, rows_addr, ROW);
    const auto s_top = TensorAccessor(t_args, top_addr, TOP);
    const auto s_out = TensorAccessor(o_args, out_addr, OUT);
    const uint32_t row_l1 = get_write_ptr(cb);
    const uint32_t top_l1 = row_l1 + ROW;
    const uint32_t out_l1 = top_l1 + TOP;
    const uint16_t* row = reinterpret_cast<const uint16_t*>(row_l1);
    const uint32_t* top = reinterpret_cast<const uint32_t*>(top_l1);
    uint16_t* score = reinterpret_cast<uint16_t*>(out_l1);
    uint16_t* col = reinterpret_cast<uint16_t*>(out_l1 + OUT);
    for (uint32_t q = q0; q < q1; ++q) {
        noc_async_read(s_rows.get_noc_addr(q), row_l1, ROW);
        noc_async_read(s_top.get_noc_addr(q), top_l1, TOP);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < k; ++i) {
            const uint32_t c = top[i];
            const bool none = c == SENT || c >= width;
            score[i] = none ? NEG_INF : row[c];
            col[i] = none ? 0 : static_cast<uint16_t>(c);
        }
        noc_async_write(out_l1, s_out.get_noc_addr(q), OUT);
        noc_async_write(out_l1 + OUT, s_out.get_noc_addr(n_queries + q), OUT);
        noc_async_write_barrier();
    }
}
