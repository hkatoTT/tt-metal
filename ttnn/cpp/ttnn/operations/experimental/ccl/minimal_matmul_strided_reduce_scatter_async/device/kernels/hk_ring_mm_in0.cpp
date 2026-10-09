// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// EXPERIMENT (hkato, optional, default off; TT_MMRS_RING=1): in0 (activation) data movement of the matmul + ring
// reduce-scatter on the matmul cores. Forwards in0 along each core row as minimal_matmul does, tells the column's
// relay core when a block of partials is ready in the send CB (and pops it once the relay has copied it out), and
// writes the final step's output rows. Every wait polls service(), so a full CB never blocks the hand-offs.

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t ring_size = get_compile_time_arg_val(6);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(8);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t valid_sem_id = get_compile_time_arg_val(11);
    constexpr bool is_injector = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t relay_done_sem_id = get_compile_time_arg_val(13);
    constexpr uint32_t out_Wt = get_compile_time_arg_val(14);
    constexpr auto in0_args = TensorAccessorArgs<15>();
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t block_tiles = M_block_tiles * N_block_tiles;
    constexpr uint32_t num_send_blocks = (ring_size - 1) * M_blocks_per_core;

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t send_cb = tt::CBIndex::c_2;
    constexpr uint32_t final_cb = tt::CBIndex::c_4;
    constexpr uint32_t scratch_cb = tt::CBIndex::c_8;

    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const bool is_sink = get_arg_val<uint32_t>(argidx++) != 0;
    const uint32_t dest_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t dest_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_valid_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t col_in_slice = get_arg_val<uint32_t>(argidx++);
    const uint32_t relay_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t relay_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t src_index = get_arg_val<uint32_t>(argidx++);

    const auto in0_reader = TensorAccessor(in0_args, in0_addr, in0_tile_size);
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_size);
    const TensorShape2D in0_shape(M_tiles, K_tiles, M_tiles, K_tiles);

    volatile tt_l1_ptr uint32_t* sender_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sender_sem_id));
    volatile tt_l1_ptr uint32_t* receiver_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(receiver_sem_id));
    const uint32_t valid_sem_addr = get_semaphore(valid_sem_id);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_sem_addr), 1);
    volatile tt_l1_ptr uint32_t* relay_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(relay_done_sem_id));
    // this source's ready counter in the relay core's scratch (the scratch CB sits at the same L1 address everywhere)
    const uint64_t relay_ready_noc_addr = get_noc_addr(relay_x, relay_y, get_write_ptr(scratch_cb) + src_index * 16);

    auto valid_rows = [&](uint32_t mb) -> uint32_t {
        const uint32_t m = M_start_tile + mb * M_block_tiles;
        return m >= M_valid_end ? 0 : std::min(M_block_tiles, M_valid_end - m);
    };
    uint32_t total_final_rows = 0;
    for (uint32_t mb = 0; mb < M_blocks_per_core; mb++) {
        total_final_rows += valid_rows(mb);
    }

    uint32_t signaled = 0;
    uint32_t popped = 0;
    uint32_t final_rows = 0;
    uint32_t f_mb = 0;
    uint32_t f_row = 0;
    auto service = [&]() {
        if (signaled < num_send_blocks && cb_pages_available_at_front(send_cb, (signaled - popped + 1) * block_tiles)) {
            noc_semaphore_inc(relay_ready_noc_addr, 1);
            signaled++;
        }
        if (popped < signaled && *relay_done >= popped + 1) {
            cb_pop_front(send_cb, block_tiles);
            popped++;
        }
        if (final_rows < total_final_rows && cb_pages_available_at_front(final_cb, N_block_tiles)) {
            while (f_row >= valid_rows(f_mb)) {
                f_mb++;
                f_row = 0;
            }
            const uint32_t r = M_start_tile + f_mb * M_block_tiles + f_row;
            const uint32_t l1 = get_read_ptr(final_cb);
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                noc_async_write(
                    l1 + n * out_tile_size, out_writer.get_noc_addr(r * out_Wt + col_in_slice + n), out_tile_size);
            }
            noc_async_writes_flushed();
            cb_pop_front(final_cb, N_block_tiles);
            f_row++;
            final_rows++;
        }
    };

    for (uint32_t k = 0; k < ring_size; k++) {
        for (uint32_t mb = 0; mb < M_blocks_per_core; mb++) {
            const uint32_t m_tile = M_start_tile + mb * M_block_tiles;
            const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
            const uint32_t block_bytes = (m_tile_end - m_tile) * K_block_tiles * in0_tile_size;
            for (uint32_t kb = 0; kb < K_num_blocks; kb++) {
                while (!cb_pages_reservable_at_back(in0_cb, in0_block_num_tiles)) {
                    service();
                }
                cb_reserve_back(in0_cb, in0_block_num_tiles);
                const uint32_t l1 = get_write_ptr(in0_cb);
                if constexpr (is_injector) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        in0_cb,
                        in0_tile_size,
                        m_tile,
                        m_tile_end,
                        kb * K_block_tiles,
                        (kb + 1) * K_block_tiles);
                } else {
                    noc_semaphore_set(receiver_sem, 0);
                    noc_semaphore_inc(get_noc_addr(sender_x, sender_y, (uint32_t)sender_sem), 1);
                    while (*receiver_sem != 1) {
                        service();
                    }
                }
                cb_push_back(in0_cb, in0_block_num_tiles);
                if (!is_sink) {
                    while (*sender_sem < 1) {
                        service();
                    }
                    noc_semaphore_set(sender_sem, 0);
                    noc_async_write(l1, get_noc_addr(dest_x, dest_y, l1), block_bytes);
                    noc_semaphore_set_remote(valid_sem_addr, get_noc_addr(dest_x, dest_y, (uint32_t)receiver_sem));
                }
                service();
            }
        }
    }

    while (popped < num_send_blocks || final_rows < total_final_rows) {
        service();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
