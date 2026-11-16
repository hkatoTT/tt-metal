// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_row_reduce_scatter's add stage, a core's reader: for each batch b, once the
// relay core has raised this core's semaphore to b + 1 (the column's reduce-scatter has written batch b), reads this
// core's tiles [start, end) of the batch from the column's output and from the shared expert's partial, chunks of up
// to CHUNK tiles into cb_a / cb_b.  Waits bounded.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t batches = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_batch = get_compile_time_arg_val(4);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(5);
    constexpr auto a_args = TensorAccessorArgs<6>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t SPIN_LIMIT = 1u << 27;

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t end = get_arg_val<uint32_t>(3);
    const uint32_t tile_bytes = get_tile_size(cb_a);
    const auto sa = TensorAccessor(a_args, a_addr);
    const auto sb = TensorAccessor(b_args, b_addr);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    for (uint32_t b = 0; b < batches; ++b) {
        for (uint32_t n = 0; n < SPIN_LIMIT; ++n) {
            invalidate_l1_cache();
            if (*sem >= b + 1) {
                break;
            }
        }
        for (uint32_t i = start; i < end; i += CHUNK) {
            const uint32_t k = end - i < CHUNK ? end - i : CHUNK;
            cb_reserve_back(cb_a, CHUNK);
            cb_reserve_back(cb_b, CHUNK);
            uint32_t la = get_write_ptr(cb_a), lb = get_write_ptr(cb_b);
            for (uint32_t j = 0; j < k; ++j) {
                const uint32_t page = b * tiles_per_batch + i + j;
                noc_async_read_page(page, sa, la);
                noc_async_read_page(page, sb, lb);
                la += tile_bytes;
                lb += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, CHUNK);
            cb_push_back(cb_b, CHUNK);
        }
    }
}
