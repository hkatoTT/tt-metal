// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// The gathered prefix (dsa_indexer.prefix_rows; a prefill chunk over the split latents): this chip's rows of a user's
// prefix, rows base / MUL .. base / MUL + N - 1 of the row-major cache (its user's first row `base` from the params'
// row 0, word 1: a mesh row's queries are one user's), into out [1, 1, N, D] (rows r0 .. r1 - 1 here); a base whose
// rows would pass the cache's end (a row without a request) reads from row 0 (its rows unused).
//   CT: row bytes, N, the cache's rows, MUL, cb, then the accessors of the cache, params, out.  RT: cache, params, out,
//   r0, r1
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t params_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t r0 = get_arg_val<uint32_t>(3);
    const uint32_t r1 = get_arg_val<uint32_t>(4);

    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t n_rows = get_compile_time_arg_val(1);
    constexpr uint32_t total_rows = get_compile_time_arg_val(2);
    constexpr uint32_t mul = get_compile_time_arg_val(3);
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr auto s_args = TensorAccessorArgs<5>();
    constexpr auto p_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t BLOCK = 16, P_BYTES = 64;

    if (r1 <= r0) {
        return;
    }
    const auto s_src = TensorAccessor(s_args, src_addr, row_bytes);
    const auto s_par = TensorAccessor(p_args, params_addr, P_BYTES);
    const auto s_out = TensorAccessor(o_args, out_addr, row_bytes);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t par_l1 = l1 + BLOCK * row_bytes;
    noc_async_read(s_par.get_noc_addr(0), par_l1, P_BYTES);
    noc_async_read_barrier();
    uint32_t base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(par_l1)[1] / mul;
    if (base + n_rows > total_rows) {
        base = 0;
    }
    for (uint32_t b = r0; b < r1; b += BLOCK) {
        const uint32_t n = r1 - b < BLOCK ? r1 - b : BLOCK;
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(s_src.get_noc_addr(base + b + i), l1 + i * row_bytes, row_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(l1 + i * row_bytes, s_out.get_noc_addr(b + i), row_bytes);
        }
        noc_async_write_barrier();
    }
}
