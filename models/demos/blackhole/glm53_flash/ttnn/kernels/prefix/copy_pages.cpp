// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// prefix_store.copy_pages_many (the prefix cache): for each of `pairs` pairs of interleaved DRAM buffers of one
// page size (FP8 rows, BFP / bf16 / fp32 tiles: bytes only), segments of the source's pages copied to the destination,
// the same segments for every pair, per chip from one row of a plan (row-major uint32 [plans, PLAN_WIDTH]: S, then
// (source page, destination page, pages) for each of S segments).  This core (`core` of `cores`) takes the pages whose
// index over all the pairs' segments in order is `core` mod `cores` (consecutive pages sit in consecutive DRAM banks),
// up to `slots` at a time: read into L1, then written out; the slots are written out at the end of each pair.  Every
// transfer starts 64-byte aligned at both ends (the plan's and the slots' L1 rounded up to 64; DRAM pages are 64-byte
// aligned) and is at least 64 bytes (the pages are multiples of 64).
//   CT: cores, the plan's page (bytes), slots, a slot's bytes (>= the page), the CB, pairs, then the accessors of the
//       sources, the plan and the destinations (interleaved: every pair's alike).  RT: the plan's address, the page
//       (bytes), core, the plan's row, then each pair's source and destination addresses.
void kernel_main() {
    const uint32_t plan_addr = get_arg_val<uint32_t>(0);
    const uint32_t page = get_arg_val<uint32_t>(1);
    const uint32_t core = get_arg_val<uint32_t>(2);
    const uint32_t plan_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t cores = get_compile_time_arg_val(0);
    constexpr uint32_t plan_page = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr uint32_t pairs = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();
    constexpr auto plan_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<plan_args.next_compile_time_args_offset()>();
    constexpr uint32_t ALIGN = 64;

    const uint32_t plan_l1 = (get_write_ptr(cb) + ALIGN - 1) & ~(ALIGN - 1);
    const uint32_t data = plan_l1 + ((plan_page + ALIGN - 1) & ~(ALIGN - 1));
    const auto s_plan = TensorAccessor(plan_args, plan_addr, plan_page);
    noc_async_read(s_plan.get_noc_addr(plan_row), plan_l1, plan_page);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_l1);
    const uint32_t segments = plan[0];

    uint32_t to_page[slots];
    uint32_t base = 0;  // the pages of the segments before this one (over the pairs before this one too)
    for (uint32_t q = 0; q < pairs; ++q) {
        const auto s_src = TensorAccessor(src_args, get_arg_val<uint32_t>(4 + 2 * q), page);
        const auto s_dst = TensorAccessor(dst_args, get_arg_val<uint32_t>(5 + 2 * q), page);
        uint32_t n = 0;  // pages read into the slots, not yet written
        for (uint32_t s = 0; s < segments; ++s) {
            const uint32_t from = plan[1 + 3 * s];
            const uint32_t to = plan[2 + 3 * s];
            const uint32_t count = plan[3 + 3 * s];
            // the first page of this segment that is this core's: (base + j) % cores == core
            for (uint32_t j = (core + cores - base % cores) % cores; j < count; j += cores) {
                noc_async_read(s_src.get_noc_addr(from + j), data + n * slot_bytes, page);
                to_page[n++] = to + j;
                if (n == slots) {
                    noc_async_read_barrier();
                    for (uint32_t i = 0; i < n; ++i) {
                        noc_async_write(data + i * slot_bytes, s_dst.get_noc_addr(to_page[i]), page);
                    }
                    noc_async_writes_flushed();  // (the slots out of L1 before they are read into again)
                    n = 0;
                }
            }
            base += count;
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(data + i * slot_bytes, s_dst.get_noc_addr(to_page[i]), page);
        }
        noc_async_writes_flushed();  // (this pair's last slots out of L1 before the next pair reads into them)
    }
    noc_async_write_barrier();
}
