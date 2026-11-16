// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_row_reduce_scatter's relay core: every writer of the column's ring
// reduce-scatter increments this core's semaphore once it has written a batch's outputs; when all `producers` have
// done so for batch b, this raises the semaphore of every worker of the row's reduce-scatter by one (their
// ReduceScatterOpReceiver waits for b + 1 before the batch's first ring step).  Every wait is bounded (a lost signal
// ends the op with wrong values rather than a hang).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t local_sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t batches = get_compile_time_arg_val(1);
    constexpr uint32_t producers = get_compile_time_arg_val(2);
    constexpr uint32_t SPIN_LIMIT = 1u << 27;

    size_t a = 0;
    const uint32_t recv_sem_id = get_arg_val<uint32_t>(a++);
    const uint32_t n_recv = get_arg_val<uint32_t>(a++);
    const size_t cores = a;
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(local_sem_id));
    const uint32_t recv_addr = get_semaphore(recv_sem_id);
    for (uint32_t b = 0; b < batches; ++b) {
        for (uint32_t n = 0; n < SPIN_LIMIT; ++n) {
            invalidate_l1_cache();
            if (*sem >= producers * (b + 1)) {
                break;
            }
        }
        for (uint32_t k = 0; k < n_recv; ++k) {
            const uint32_t x = get_arg_val<uint32_t>(cores + 2 * k), y = get_arg_val<uint32_t>(cores + 2 * k + 1);
            noc_semaphore_inc(get_noc_addr(x, y, recv_addr), 1);
        }
    }
    noc_async_atomic_barrier();
}
