// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_row_reduce_scatter's all-gather stage: the row all-reduce's all-gather of the
// row reduce-scatter's output, batch by batch as each is written.  This chip (rank q of its mesh row's ring of 4) holds
// quarter q of D of every batch's sums ([B, 1, R 32, D / 4], tiles); out [1, 1, B R 32, D] (tiles) gets them at tile
// columns 32 q .. on every chip of the row.  A copy: the same bits as ttnn.all_gather.
//   role 0 (BRISC): a 2D multicast of `hops` hops one way along the row (to q + 1 .. q + hops), and the local copy;
//   role 1 (NCRISC): `hops` the other way.  The odd packets go 3 - hops instead (a ring of 4: the far chip gets the
//   even packets one way and the odd ones the other), so each direction's links carry 1.5 quarters a batch, not 2 and
//   1; the batch's increment then goes 2
//   hops both ways (the far chip's packets come on both paths), `expect` 4 a batch.  Packets of 4 tiles of a tile row (a scatter write of four 2 KB pages), read
//   NB ahead by transaction ids.  A batch starts once every writer of the row reduce-scatter has written it (`producers`
//   increments of this core's semaphore sem_id + b for batch b); after its last packet, a header-only packet on the same path
//   increments the counter on the receivers' all-gather core once the batch's writes are flushed (the fused scatter +
//   increment takes 2 chunks only: NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS).  At the end role 0 waits for role 1 and for `expect` increments (every peer's
//   batches), and sets the counter back to 0.  Every wait is bounded.
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

static_assert(std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "col_row_ag: 2D fabric");

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
using Conn = tt::tt_fabric::WorkerToFabricEdmSender;

namespace {
constexpr uint32_t SPIN_LIMIT = 1u << 27;

FORCE_INLINE bool wait_min(uint32_t addr, uint32_t v) {
    for (uint32_t n = 0; n < SPIN_LIMIT; ++n) {
        invalidate_l1_cache();
        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) >= v) {
            return true;
        }
    }
    return false;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t batches = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);  // tile rows a batch
    constexpr uint32_t sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t producers = get_compile_time_arg_val(4);
    constexpr uint32_t NB = get_compile_time_arg_val(5);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(6);
    constexpr uint32_t sem_done = get_compile_time_arg_val(7);
    constexpr uint32_t expect = get_compile_time_arg_val(8);
    constexpr auto src_args = TensorAccessorArgs<9>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048, PER = 4, SLOT = PER * TILE, QW = 32, OW = 128;  // tiles a packet; tile columns
    constexpr uint32_t PACKETS = R * (QW / PER);
    static_assert(NB >= 2 && NB <= 15, "col_row_ag: transaction ids 1 .. 15");

    size_t a = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t q = get_arg_val<uint32_t>(a++);  // this chip's rank in the row
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t dst_chip = get_arg_val<uint32_t>(a++);  // the neighbour this role's route starts toward
    const uint32_t hops = get_arg_val<uint32_t>(a++);
    const uint32_t counter = get_arg_val<uint32_t>(a++);  // the receivers' counter (global semaphore)
    const uint32_t me_x = get_arg_val<uint32_t>(a++);     // the all-gather core's NoC x, y (the same on every chip)
    const uint32_t me_y = get_arg_val<uint32_t>(a++);
    auto conn = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    const auto s_src = TensorAccessor(src_args, src_addr);
    const auto s_out = TensorAccessor(out_args, out_addr);
    const uint32_t stage = get_write_ptr(cb_stage) + role * NB * SLOT;
    const uint64_t counter_noc = safe_get_noc_addr(me_x, me_y, counter, 0);

    conn.open_start();
    PacketHeaderPool::reset();
    Header h = PacketHeaderPool::allocate_header();
    Header ho = PacketHeaderPool::allocate_header();  // (the odd packets', 3 - hops)
    Header hi = PacketHeaderPool::allocate_header();  // (the batch's increment)
    {
        const auto dir = get_next_hop_router_direction(mesh_id, dst_chip);
        auto route = [&](Header hd, uint32_t n) {
            uint16_t hp[4] = {0, 0, 0, 0};
            hp[dir] = static_cast<uint16_t>(n);
            fabric_set_mcast_route(hd, dst_chip, mesh_id, hp[0], hp[1], hp[2], hp[3]);
        };
        route(h, hops);
        route(ho, 3 - hops);
        route(hi, 2);
    }
    hi->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{counter_noc, 1, true});
    conn.open_finish();

    // packet j of batch b: tile row r = j / 8 of the batch, tile columns 4 (j % 8) ..
    auto src_page = [&](uint32_t b, uint32_t j, uint32_t k) { return (b * R + j / 8) * QW + (j % 8) * PER + k; };
    auto out_page = [&](uint32_t b, uint32_t j, uint32_t k) {
        return (b * R + j / 8) * OW + q * QW + (j % 8) * PER + k;
    };
    auto read_packet = [&](uint32_t b, uint32_t j) {
        const uint32_t slot = stage + (j % NB) * SLOT;
        noc_async_read_set_trid(j % NB + 1);
        for (uint32_t k = 0; k < PER; ++k) {
            noc_async_read(s_src.get_noc_addr(src_page(b, j, k)), slot + k * TILE, TILE);
        }
    };

    bool failed = false;
    for (uint32_t b = 0; b < batches && !failed; ++b) {
        // the row reduce-scatter has written batch b: every writer's increment of semaphore sem_id + b (one a batch)
        if (!wait_min(get_semaphore(sem_id + b), producers)) {
            failed = true;
            break;
        }
        for (uint32_t j = 0; j < NB && j < PACKETS; ++j) {
            read_packet(b, j);
        }
        for (uint32_t j = 0; j < PACKETS; ++j) {
            const uint32_t src = stage + (j % NB) * SLOT;
            noc_async_read_barrier_with_trid(j % NB + 1);
            if constexpr (role == 0) {  // the local copy
                for (uint32_t k = 0; k < PER; ++k) {
                    noc_async_write(src + k * TILE, s_out.get_noc_addr(out_page(b, j, k)), TILE);
                }
            }
            const uint64_t d0 = s_out.get_noc_addr(out_page(b, j, 0), 0, 0);
            const uint64_t d1 = s_out.get_noc_addr(out_page(b, j, 1), 0, 0);
            const uint64_t d2 = s_out.get_noc_addr(out_page(b, j, 2), 0, 0);
            const uint64_t d3 = s_out.get_noc_addr(out_page(b, j, 3), 0, 0);
            noc_async_writes_flushed();  // (the header is rewritten: its last send has left)
            const Header hd = (j & 1) ? ho : h;
            hd->to_noc_unicast_scatter_write(
                tt::tt_fabric::NocUnicastScatterCommandHeader(
                    {d0, d1, d2, d3},
                    {static_cast<uint16_t>(TILE), static_cast<uint16_t>(TILE), static_cast<uint16_t>(TILE)}),
                SLOT);
            uint32_t n = 0;
            while (!conn.edm_has_space_for_packet() && ++n < SPIN_LIMIT) {
            }
            if (n >= SPIN_LIMIT) {
                failed = true;
                break;
            }
            conn.send_payload_without_header_non_blocking_from_address(src, SLOT);
            conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(hd), sizeof(PACKET_HEADER_TYPE));
            noc_async_writes_flushed();  // (the slot is read again NB packets on)
            if (j + 1 == PACKETS) {  // the batch's increment, behind its packets on the same path
                n = 0;
                while (!conn.edm_has_space_for_packet() && ++n < SPIN_LIMIT) {
                }
                if (n >= SPIN_LIMIT) {
                    failed = true;
                    break;
                }
                conn.send_payload_flush_non_blocking_from_address(
                    reinterpret_cast<uint32_t>(hi), sizeof(PACKET_HEADER_TYPE));
                noc_async_writes_flushed();
            }
            if (j + NB < PACKETS) {
                read_packet(b, j + NB);
            }
        }
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
    conn.close();
    noc_async_write_barrier();
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done));
    if constexpr (role == 1) {
        *done = 1;
    } else {
        failed |= !wait_min(get_semaphore(sem_done), 1);
        *done = 0;
        failed |= !wait_min(counter, expect);  // every peer's batches have landed here
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter) = 0;
    }
    (void)failed;
}
