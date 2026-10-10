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
#ifdef HK_IN0_GATHER
    // The HK_IN0_GATHER cores of a row group share in0: each reads its own rows of every K block from DRAM and writes
    // them into the same CB slot of the others. Per-member flags (program semaphores, so zero at launch) carry the
    // block number: ready[g] = slot free on member g, parts[g] = member g's rows have landed here.
    constexpr uint32_t group_size = HK_IN0_GATHER;
    constexpr uint32_t rows_per_member = (M_block_tiles + group_size - 1) / group_size;
    const uint32_t my_pos = get_arg_val<uint32_t>(argidx++);
    uint32_t member_x[group_size];
    uint32_t member_y[group_size];
    for (uint32_t g = 0; g < group_size; g++) {
        member_x[g] = get_arg_val<uint32_t>(argidx++);
        member_y[g] = get_arg_val<uint32_t>(argidx++);
    }
    uint32_t ready_addr[group_size];
    uint32_t parts_addr[group_size];
    for (uint32_t g = 0; g < group_size; g++) {
        ready_addr[g] = get_semaphore(get_arg_val<uint32_t>(argidx++));
    }
    for (uint32_t g = 0; g < group_size; g++) {
        parts_addr[g] = get_semaphore(get_arg_val<uint32_t>(argidx++));
    }
#endif

    const auto in0_reader = TensorAccessor(in0_args, in0_addr, in0_tile_size);
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_size);
    const TensorShape2D in0_shape(M_tiles, K_tiles, M_tiles, K_tiles);

#ifndef HK_IN0_GATHER
    volatile tt_l1_ptr uint32_t* sender_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sender_sem_id));
    volatile tt_l1_ptr uint32_t* receiver_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(receiver_sem_id));
    const uint32_t valid_sem_addr = get_semaphore(valid_sem_id);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_sem_addr), 1);
#else
    // source word of the flag writes (the scratch CB is 256 B; the relay counters use the first 144 B)
    const uint32_t flag_src_addr = get_write_ptr(scratch_cb) + 192;
    volatile tt_l1_ptr uint32_t* flag_src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_src_addr);
    uint32_t blk = 0;
#endif
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
#ifdef HK_IN0_GATHER
                // every member's CB holds block blk at this same address
                blk++;
                noc_async_writes_flushed();  // the previous flag writes have left flag_src
                *flag_src = blk;
                for (uint32_t g = 0; g < group_size; g++) {
                    if (g != my_pos) {
                        noc_semaphore_set_remote(
                            flag_src_addr, get_noc_addr(member_x[g], member_y[g], ready_addr[my_pos]));
                    }
                }
                for (uint32_t g = 0; g < group_size; g++) {
                    if (g != my_pos) {
                        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr[g]) < blk) {
                            service();
                        }
                    }
                }
                {
                    const uint32_t cur_rows = m_tile_end - m_tile;
                    const uint32_t r0 = std::min(my_pos * rows_per_member, cur_rows);
                    const uint32_t r1 = std::min(r0 + rows_per_member, cur_rows);
#ifdef HK_IN0_READ_NOC
                    constexpr uint8_t read_noc = HK_IN0_READ_NOC;
#else
                    const uint8_t read_noc = noc_index;
#endif
                    // Own transaction id: in dynamic NOC mode the read barrier compares the NOC's response count with
                    // both RISCs' issue counters, and the other RISC reads on the same NOC.
                    constexpr uint32_t in0_trid = 5;
                    noc_async_read_set_trid(in0_trid, read_noc);
                    for (uint32_t r = r0; r < r1 && m_tile + r < M_tiles; r++) {
                        for (uint32_t kk = 0; kk < K_block_tiles; kk++) {
                            noc_async_read(
                                in0_reader.get_noc_addr((m_tile + r) * K_tiles + kb * K_block_tiles + kk, 0, read_noc),
                                l1 + (r * K_block_tiles + kk) * in0_tile_size,
                                in0_tile_size,
                                read_noc);
                        }
                    }
                    noc_async_read_barrier_with_trid(in0_trid, read_noc);
                    noc_async_read_set_trid(0, read_noc);
                    const uint32_t off = r0 * K_block_tiles * in0_tile_size;
                    const uint32_t bytes = (r1 - r0) * K_block_tiles * in0_tile_size;
                    for (uint32_t g = 0; g < group_size; g++) {
                        if (g != my_pos && bytes > 0) {
                            noc_async_write(l1 + off, get_noc_addr(member_x[g], member_y[g], l1 + off), bytes);
                        }
                    }
                    // The rows go out through the write command buffer and the flag through the register one; without
                    // this flush a flag could leave (and land) before the tail of a multi-packet row write, and the
                    // member would hand stale rows to its compute (rare, timing dependent).
                    noc_async_writes_flushed();
                    for (uint32_t g = 0; g < group_size; g++) {
                        if (g != my_pos) {
                            noc_semaphore_set_remote(
                                flag_src_addr, get_noc_addr(member_x[g], member_y[g], parts_addr[my_pos]));
                        }
                    }
                }
                for (uint32_t g = 0; g < group_size; g++) {
                    if (g != my_pos) {
                        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_addr[g]) < blk) {
                            service();
                        }
                    }
                }
                cb_push_back(in0_cb, in0_block_num_tiles);
#else
                if constexpr (is_injector) {
#ifdef HK_IN0_READ_NOC
                    // dynamic NOC mode: read DRAM on the other NOC (the forwarding writes stay on noc_index)
                    for (uint32_t r = 0; r < m_tile_end - m_tile && m_tile + r < M_tiles; r++) {
                        for (uint32_t kk = 0; kk < K_block_tiles; kk++) {
                            noc_async_read(
                                in0_reader.get_noc_addr(
                                    (m_tile + r) * K_tiles + kb * K_block_tiles + kk, 0, HK_IN0_READ_NOC),
                                l1 + (r * K_block_tiles + kk) * in0_tile_size,
                                in0_tile_size,
                                HK_IN0_READ_NOC);
                        }
                    }
                    noc_async_read_barrier(HK_IN0_READ_NOC);
#else
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        in0_cb,
                        in0_tile_size,
                        m_tile,
                        m_tile_end,
                        kb * K_block_tiles,
                        (kb + 1) * K_block_tiles);
#endif
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
#endif
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
