// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// A copy of page blocks (row_split.py `take`): out page o = block b = o / block_pages, page p = o % block_pages, from src
// page src_off + b block_stride + p; this core's out pages o0 .. o1 - 1, `batch` pages a pass through its L1 (`cb`).
// src_off is the chip's own (each chip of a mesh row takes its own part).
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t o0 = get_arg_val<uint32_t>(2);
    const uint32_t o1 = get_arg_val<uint32_t>(3);
    const uint32_t src_off = get_arg_val<uint32_t>(4);

    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t block_pages = get_compile_time_arg_val(1);
    constexpr uint32_t block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto s_src = TensorAccessor(src_args, src_addr, page_bytes);
    const auto s_out = TensorAccessor(out_args, out_addr, page_bytes);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t o = o0; o < o1; o += batch) {
        const uint32_t n = o1 - o < batch ? o1 - o : batch;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t q = o + i;
            const uint32_t page = src_off + (q / block_pages) * block_stride + q % block_pages;
            noc_async_read(s_src.get_noc_addr(page), l1 + i * page_bytes, page_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(l1 + i * page_bytes, s_out.get_noc_addr(o + i), page_bytes);
        }
        noc_async_write_barrier();
    }
}
