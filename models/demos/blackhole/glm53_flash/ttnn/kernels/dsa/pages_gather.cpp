// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DSA indexer (dsa_indexer.py): pages i0 .. i1 - 1 of the output (tiled, `tpp` tiles a page of 32 rows) from the
// cache's pages `table` names (`table` [n] int32: the cache's page), tile by tile through this core's L1 (`cb`:
// `block` tiles a pass, then the table page).
void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t i0 = get_arg_val<uint32_t>(3);
    const uint32_t i1 = get_arg_val<uint32_t>(4);

    constexpr uint32_t tpp = get_compile_time_arg_val(0);    // tiles a page
    constexpr uint32_t tile = get_compile_time_arg_val(1);   // bytes a tile
    constexpr uint32_t table_page = get_compile_time_arg_val(2);
    constexpr uint32_t block = get_compile_time_arg_val(3);  // tiles a pass (a multiple of tpp)
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr auto c_args = TensorAccessorArgs<5>();
    constexpr auto t_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();

    const auto s_cache = TensorAccessor(c_args, cache_addr, tile);
    const auto s_table = TensorAccessor(t_args, table_addr, table_page);
    const auto s_out = TensorAccessor(o_args, out_addr, tile);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t table_l1 = l1 + block * tile;
    noc_async_read(s_table.get_noc_addr(0), table_l1, table_page);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
    constexpr uint32_t pages_a_pass = block / tpp;
    for (uint32_t b = i0; b < i1; b += pages_a_pass) {
        const uint32_t n = i1 - b < pages_a_pass ? i1 - b : pages_a_pass;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t page = table[b + i];
            for (uint32_t c = 0; c < tpp; ++c) {
                noc_async_read(s_cache.get_noc_addr(page * tpp + c), l1 + (i * tpp + c) * tile, tile);
            }
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n * tpp; ++i) {
            noc_async_write(l1 + i * tile, s_out.get_noc_addr(b * tpp + i), tile);
        }
        noc_async_write_barrier();
    }
}
