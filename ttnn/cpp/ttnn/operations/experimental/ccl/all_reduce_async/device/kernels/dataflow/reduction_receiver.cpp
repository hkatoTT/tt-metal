// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_num_reduction_tiles = get_compile_time_arg_val(1);
    // AllReduceAsyncParams::compact_rows (0: whole tiles): the shares came as each tile's first compact_rows rows of
    // faces 0 and 1, packed, and the compute summed them packed into staged_cb_id; here the output tiles' rows
    // compact_rows .. 31 go to zero while the shares come, then the sums go back into rows 0 .. compact_rows - 1
    constexpr uint32_t compact_rows = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t staged_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(5);

    // runtime args
    size_t arg_idx = 0;
    const uint32_t has_work = get_arg_val<uint32_t>(arg_idx++);
    if (has_work == 0) {
        return;
    }

    const uint32_t signal_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const size_t out_ready_sem_bank_addr =
        get_common_arg_val<uint32_t>(ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore);
    const uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    volatile tt_l1_ptr uint32_t* out_ready_sema =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);

    CircularBuffer cb(cb_id);

    [[maybe_unused]] uint32_t out_addr = 0, out_tile = 0;
    if constexpr (compact_rows != 0) {
        out_addr = get_write_ptr(out_cb_id);
        out_tile = get_local_cb_interface(out_cb_id).fifo_page_size;
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        auto zero = [&](uint32_t dst, uint32_t bytes) {
            for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
                noc_async_read(zeros, dst + o, bytes - o < MEM_ZEROS_SIZE ? bytes - o : MEM_ZEROS_SIZE);
            }
        };
        for (uint32_t k = 0; k < out_tiles; ++k) {
            const uint32_t tile = out_addr + k * out_tile;
            if constexpr (compact_rows < 16) {  // faces 0 and 1's rows compact_rows .. 15
                zero(tile + compact_rows * 32, (16 - compact_rows) * 32);
                zero(tile + 512 + compact_rows * 32, (16 - compact_rows) * 32);
            }
            zero(tile + 1024, 1024);  // faces 2 and 3 (rows 16 .. 31)
        }
    }

    // 1. Wait for signal from All-Gather worker
    {
        DeviceZoneScopedN("AR-RX-WAIT");
        noc_semaphore_wait(out_ready_sema, out_ready_sem_wait_value);
    }

    // 2. Signal compute kernel to start processing
    cb.push_back(total_num_reduction_tiles);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);

    if constexpr (compact_rows != 0) {
        constexpr uint32_t part = compact_rows * 32;  // a face's token rows, bytes
        constexpr uint32_t pages = out_tiles * compact_rows / 32;
        CircularBuffer staged(staged_cb_id);
        {
            DeviceZoneScopedN("AR-RX-SUM");  // (waiting for the compute's packed sums)
            staged.wait_front(pages);
        }
        DeviceZoneScopedN("AR-RX-PUT");
        const uint32_t src = staged.get_read_ptr();
        for (uint32_t k = 0; k < out_tiles; ++k) {
            noc_async_read(get_noc_addr(src + 2 * k * part), out_addr + k * out_tile, part);
            noc_async_read(get_noc_addr(src + (2 * k + 1) * part), out_addr + k * out_tile + 512, part);
        }
        noc_async_read_barrier();
        staged.pop_front(pages);
    }
}
