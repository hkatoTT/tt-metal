// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_row_reduce_scatter's add stage, a core's writer: writes this core's sums of
// each batch to the total (the row reduce-scatter's input), then, once they are acked, increments the relay core's
// semaphore (the row reduce-scatter's batch b starts when every add core has done so).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t batches = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_batch = get_compile_time_arg_val(2);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(3);
    constexpr uint32_t relay_sem_id = get_compile_time_arg_val(4);
    constexpr auto o_args = TensorAccessorArgs<5>();

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t end = get_arg_val<uint32_t>(2);
    const uint32_t relay_x = get_arg_val<uint32_t>(3);
    const uint32_t relay_y = get_arg_val<uint32_t>(4);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto so = TensorAccessor(o_args, o_addr);
    const uint64_t relay_noc = get_noc_addr(relay_x, relay_y, get_semaphore(relay_sem_id));

    for (uint32_t b = 0; b < batches; ++b) {
        for (uint32_t i = start; i < end; i += CHUNK) {
            const uint32_t k = end - i < CHUNK ? end - i : CHUNK;
            cb_wait_front(cb_out, CHUNK);
            uint32_t l = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < k; ++j) {
                noc_async_write_page(b * tiles_per_batch + i + j, so, l);
                l += tile_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, CHUNK);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(relay_noc, 1);
    }
    noc_async_atomic_barrier();
}
