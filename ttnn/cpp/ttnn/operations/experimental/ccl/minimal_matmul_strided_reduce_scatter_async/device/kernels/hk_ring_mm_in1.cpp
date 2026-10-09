// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// EXPERIMENT (hkato, optional, default off; TT_MMRS_RING=1): in1 (weight) data movement of the matmul + ring
// reduce-scatter on the matmul cores. Forwards in1 down each core column as minimal_matmul does (the column of
// ring step k is this core's 5-tile share of slice s(k)), hands the compute the partials the neighbour device sent
// for this core's tiles (recv CB, filled remotely), the gate row and the final step's residual rows.
//
// HK_RELAY (the in1 sink row, one core per column): also relays the column's partials to the same cores of the
// ring neighbour over a fabric connection of its own (columns 0..3 forward on links 0..3, columns 4..7 backward).
// A source signals a ready block through this core's scratch counters; the relay copies it to staging, sends it to
// the mirror core's recv CB (the last packet of a block bumps the mirror core's recv semaphore) and releases the
// source's send CB slot. Before the first send it waits for the peer relay's "in the op" signal, so remote writes
// never land in another program's L1.

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"
#ifndef HK_SEND_SLOTS
#define HK_SEND_SLOTS 1  // blocks in the send CB (set by the factory)
#endif
#ifdef HK_RELAY
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
namespace fabric_api = tt::tt_fabric::linear::experimental;
#endif

void kernel_main() {
    constexpr uint32_t K_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t ring_size = get_compile_time_arg_val(6);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t a_tile_size = get_compile_time_arg_val(8);
    constexpr uint32_t b_tile_size = get_compile_time_arg_val(9);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t valid_sem_id = get_compile_time_arg_val(12);
    constexpr bool is_injector = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t recv_sem_id = get_compile_time_arg_val(14);
    constexpr uint32_t out_Wt = get_compile_time_arg_val(15);  // width of a / b / output, in tiles (N_tiles / ring)
    constexpr uint32_t send_tile_size = get_compile_time_arg_val(16);
    constexpr uint32_t relay_done_sem_id = get_compile_time_arg_val(17);
    constexpr uint32_t num_srcs = get_compile_time_arg_val(18);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(19);
    constexpr uint32_t group_tiles = get_compile_time_arg_val(20);
    constexpr auto in1_args = TensorAccessorArgs<21>();
    constexpr auto a_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
#ifdef HK_RING_BIAS
    constexpr auto bias_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr uint32_t bias_cb = tt::CBIndex::c_10;
    constexpr uint32_t bias_tile_size = get_tile_size(bias_cb);
#endif

    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t block_tiles = M_block_tiles * N_block_tiles;
    constexpr uint32_t num_recv_blocks = (ring_size - 1) * M_blocks_per_core;

    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t a_cb = tt::CBIndex::c_5;
    constexpr uint32_t b_cb = tt::CBIndex::c_6;
    constexpr uint32_t recv_cb = tt::CBIndex::c_7;

    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t a_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t b_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_dests = get_arg_val<uint32_t>(argidx++);  // 0 (sink), 1, or 2 (shared-in1 injector)
    const uint32_t dest_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t dest_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_valid_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t col_in_slice = get_arg_val<uint32_t>(argidx++);
    const uint32_t dest2_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t dest2_y = get_arg_val<uint32_t>(argidx++);
    uint32_t slices[ring_size];
    for (uint32_t k = 0; k < ring_size; k++) {
        slices[k] = get_arg_val<uint32_t>(argidx++);
    }

    const auto in1_reader = TensorAccessor(in1_args, in1_addr, in1_tile_size);
    const auto a_reader = TensorAccessor(a_args, a_addr, a_tile_size);
    const auto b_reader = TensorAccessor(b_args, b_addr, b_tile_size);
    const TensorShape2D in1_shape(K_tiles, N_tiles, K_tiles, N_tiles);

    volatile tt_l1_ptr uint32_t* sender_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sender_sem_id));
    volatile tt_l1_ptr uint32_t* receiver_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(receiver_sem_id));
    const uint32_t valid_sem_addr = get_semaphore(valid_sem_id);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_sem_addr), 1);
    volatile tt_l1_ptr uint32_t* recv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(recv_sem_id));

    auto valid_rows = [&](uint32_t m_start, uint32_t m_valid_end, uint32_t mb) -> uint32_t {
        const uint32_t m = m_start + mb * M_block_tiles;
        return m >= m_valid_end ? 0 : std::min(M_block_tiles, m_valid_end - m);
    };

#ifdef HK_RELAY
    constexpr uint32_t send_cb = tt::CBIndex::c_2;
    constexpr uint32_t scratch_cb = tt::CBIndex::c_8;
    constexpr uint32_t staging_cb = tt::CBIndex::c_9;
    constexpr uint32_t num_send_blocks = (ring_size - 1) * M_blocks_per_core;
    const uint32_t ready_sem_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t peer_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t peer_y = get_arg_val<uint32_t>(argidx++);
    uint32_t src_x[num_srcs];
    uint32_t src_y[num_srcs];
    uint32_t src_m_start[num_srcs];
    uint32_t src_m_valid_end[num_srcs];
    for (uint32_t s = 0; s < num_srcs; s++) {
        src_x[s] = get_arg_val<uint32_t>(argidx++);
        src_y[s] = get_arg_val<uint32_t>(argidx++);
        src_m_start[s] = get_arg_val<uint32_t>(argidx++);
        src_m_valid_end[s] = get_arg_val<uint32_t>(argidx++);
    }
    size_t fab_arg_idx = argidx;

    // Zero the sources' ready counters before this core first asks its in1 sender for data: no source can finish
    // its first block before then (the column's in1 chain ends here), so no increment is lost.
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(scratch_cb));
    for (uint32_t s = 0; s < num_srcs; s++) {
        scratch[s * 4] = 0;
    }
    // Same CB layout on every core and every device
    const uint32_t send_base = get_read_ptr(send_cb);
    const uint32_t recv_base = get_write_ptr(recv_cb);
    const uint32_t staging = get_write_ptr(staging_cb);
    const uint32_t recv_sem_addr = get_semaphore(recv_sem_id);
    const uint32_t relay_done_addr = get_semaphore(relay_done_sem_id);
    volatile tt_l1_ptr uint32_t* ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem_addr);

    auto fabric_connection = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(fab_arg_idx, 1);
    auto data_pkt = PacketHeaderPool::allocate_header(1);
    auto fused_pkt = PacketHeaderPool::allocate_header(1);
    auto ready_pkt = PacketHeaderPool::allocate_header(1);
    fabric_connection.open_finish();
    auto* sender = &fabric_connection.get(0).sender;

    fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(data_pkt, 1);
    fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
        UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush>(
        fused_pkt, 1, tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0u, 0u, 1u}, 0);
    // Tell the peer relay (opposite direction, same link) on the neighbour that this device runs the op
    fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        ready_pkt,
        1,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(peer_x, peer_y, ready_sem_addr, 0), 1u});
    fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::None>(sender, ready_pkt);
    noc_async_writes_flushed();

    uint32_t r_seq = 0;  // block (k * M_blocks_per_core + mb) being relayed
    uint32_t r_src = 0;  // source core within the column
    uint32_t r_tile = 0;
    bool r_barrier = false;
    auto relay_step = [&]() {
        if (r_seq == num_send_blocks) {
            return;
        }
        if (!r_barrier) {
            if (*ready_sem == 0) {
                return;
            }
            noc_semaphore_set(ready_sem, 0);
            r_barrier = true;
        }
        const uint32_t mb = r_seq % M_blocks_per_core;
        const uint32_t ntiles = valid_rows(src_m_start[r_src], src_m_valid_end[r_src], mb) * N_block_tiles;
        if (r_tile == 0 && scratch[r_src * 4] < r_seq + 1) {
            return;  // the source has not produced this block yet
        }
        if (ntiles == 0) {
            // nothing to send, but the mirror core still counts the block
            fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                sender,
                ready_pkt,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    safe_get_noc_addr(src_x[r_src], src_y[r_src], recv_sem_addr, 0), 1u});
            noc_async_writes_flushed();
            noc_semaphore_inc(get_noc_addr(src_x[r_src], src_y[r_src], relay_done_addr), 1);
            if (++r_src == num_srcs) {
                r_src = 0;
                r_seq++;
            }
            return;
        }
        const uint32_t g = std::min(group_tiles, ntiles - r_tile);
        const uint32_t off = r_tile * send_tile_size;
        noc_async_read(
            get_noc_addr(
                src_x[r_src], src_y[r_src], send_base + (r_seq % HK_SEND_SLOTS) * block_tiles * send_tile_size + off),
            staging,
            g * send_tile_size);
        noc_async_read_barrier();
        const uint32_t dst_base = recv_base + r_seq * block_tiles * send_tile_size + off;
        for (uint32_t p = 0; p < g; p += tiles_per_packet) {
            const uint32_t n = std::min(tiles_per_packet, g - p);
            const uint64_t dst = safe_get_noc_addr(src_x[r_src], src_y[r_src], dst_base + p * send_tile_size, 0);
            if (r_tile + p + n == ntiles) {
                fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                    UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
                    UnicastFusedAtomicIncUpdateMask::PayloadSize>(
                    sender,
                    fused_pkt,
                    staging + p * send_tile_size,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        dst, safe_get_noc_addr(src_x[r_src], src_y[r_src], recv_sem_addr, 0), 1u},
                    n * send_tile_size);
            } else {
                fabric_api::fabric_unicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    sender,
                    data_pkt,
                    staging + p * send_tile_size,
                    tt::tt_fabric::NocUnicastCommandHeader{dst},
                    n * send_tile_size);
            }
            noc_async_writes_flushed();
        }
        r_tile += g;
        if (r_tile == ntiles) {
            noc_semaphore_inc(get_noc_addr(src_x[r_src], src_y[r_src], relay_done_addr), 1);
            r_tile = 0;
            if (++r_src == num_srcs) {
                r_src = 0;
                r_seq++;
            }
        }
    };
#endif

#ifdef HK_RING_BIAS
    // The bias row of this device's own slice, added once at the last step (the bias is replicated, not split)
    {
        const auto bias_reader = TensorAccessor(bias_args, bias_addr, bias_tile_size);
        cb_reserve_back(bias_cb, N_block_tiles);
        const uint32_t l1 = get_write_ptr(bias_cb);
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            noc_async_read(
                bias_reader.get_noc_addr(slices[ring_size - 1] * out_Wt + col_in_slice + n),
                l1 + n * bias_tile_size,
                bias_tile_size);
        }
        noc_async_read_barrier();
        cb_push_back(bias_cb, N_block_tiles);
    }
#else
    (void)bias_addr;
#endif
    // The gate row is the same for every block
    {
        cb_reserve_back(b_cb, N_block_tiles);
        const uint32_t l1 = get_write_ptr(b_cb);
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            noc_async_read(b_reader.get_noc_addr(col_in_slice + n), l1 + n * b_tile_size, b_tile_size);
        }
        noc_async_read_barrier();
        cb_push_back(b_cb, N_block_tiles);
    }

    uint32_t total_a_rows = 0;
    for (uint32_t mb = 0; mb < M_blocks_per_core; mb++) {
        total_a_rows += valid_rows(M_start_tile, M_valid_end, mb);
    }
    uint32_t recv_pushed = 0;
    uint32_t a_rows = 0;
    uint32_t a_mb = 0;
    uint32_t a_row = 0;
    auto service = [&]() {
        // the recv CB holds every block of the op, so a block can be handed over as soon as it has landed
        while (recv_pushed < num_recv_blocks && *recv_sem >= recv_pushed + 1) {
            cb_reserve_back(recv_cb, block_tiles);
            cb_push_back(recv_cb, block_tiles);
            recv_pushed++;
        }
        if (a_rows < total_a_rows && cb_pages_reservable_at_back(a_cb, N_block_tiles)) {
            while (a_row >= valid_rows(M_start_tile, M_valid_end, a_mb)) {
                a_mb++;
                a_row = 0;
            }
            cb_reserve_back(a_cb, N_block_tiles);
            const uint32_t l1 = get_write_ptr(a_cb);
            const uint32_t r = M_start_tile + a_mb * M_block_tiles + a_row;
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                noc_async_read(a_reader.get_noc_addr(r * out_Wt + col_in_slice + n), l1 + n * a_tile_size, a_tile_size);
            }
            noc_async_read_barrier();
            cb_push_back(a_cb, N_block_tiles);
            a_row++;
            a_rows++;
        }
#ifdef HK_RELAY
        relay_step();
#endif
    };

    for (uint32_t k = 0; k < ring_size; k++) {
        const uint32_t n_tile = slices[k] * out_Wt + col_in_slice;
        for (uint32_t mb = 0; mb < M_blocks_per_core; mb++) {
            for (uint32_t kb = 0; kb < K_num_blocks; kb++) {
                while (!cb_pages_reservable_at_back(in1_cb, in1_block_num_tiles)) {
                    service();
                }
                cb_reserve_back(in1_cb, in1_block_num_tiles);
                const uint32_t l1 = get_write_ptr(in1_cb);
                if constexpr (is_injector) {
                    read_in1_block_sync<K_block_tiles, N_block_tiles>(
                        in1_reader,
                        in1_shape,
                        in1_cb,
                        in1_tile_size,
                        kb * K_block_tiles,
                        (kb + 1) * K_block_tiles,
                        n_tile,
                        n_tile + N_block_tiles);
                } else {
                    noc_semaphore_set(receiver_sem, 0);
                    noc_semaphore_inc(get_noc_addr(sender_x, sender_y, (uint32_t)sender_sem), 1);
                    while (*receiver_sem != 1) {
                        service();
                    }
                }
                cb_push_back(in1_cb, in1_block_num_tiles);
                if (num_dests > 0) {
                    while (*sender_sem < num_dests) {
                        service();
                    }
                    noc_semaphore_set(sender_sem, 0);
                    noc_async_write(l1, get_noc_addr(dest_x, dest_y, l1), in1_block_num_tiles * in1_tile_size);
                    if (num_dests > 1) {
                        noc_async_write(l1, get_noc_addr(dest2_x, dest2_y, l1), in1_block_num_tiles * in1_tile_size);
                    }
                    noc_semaphore_set_remote(valid_sem_addr, get_noc_addr(dest_x, dest_y, (uint32_t)receiver_sem));
                    if (num_dests > 1) {
                        noc_semaphore_set_remote(
                            valid_sem_addr, get_noc_addr(dest2_x, dest2_y, (uint32_t)receiver_sem));
                    }
                }
                service();
            }
        }
    }

#ifdef HK_RELAY
    while (recv_pushed < num_recv_blocks || a_rows < total_a_rows || r_seq < num_send_blocks) {
        service();
    }
    fabric_connection.close_start();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    fabric_connection.close_finish();
#else
    while (recv_pushed < num_recv_blocks || a_rows < total_a_rows) {
        service();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
