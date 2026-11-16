// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row all-reduce (row_all_reduce.py), a sender RISC on the sender core: one-shot, over the 2D fabric.  Opens a
// connection to each ring neighbour's router (on its link) and sends its share of the slices (workers w_first,
// w_first + w_step, ..), each as soon as the worker has put it in the stage, to the row's three other chips, into
// their DRAM landing area (persistent DRAM, the same address on every chip: page (parity, this chip, worker)), each
// packet a write fused with an increment of that worker core's receive semaphore for this chip (after the write is
// acknowledged: the flush).
//   mcast 0: a unicast to each neighbour, and one to the opposite chip over the connection its route leaves by;
//   mcast 1: a 2-hop multicast one way (the neighbour and the opposite chip) and a unicast the other way, the way
//            alternating from slice to slice.  (A 2D multicast may also deliver on this chip: into its own page of
//            the area, the same bytes, which no one reads, and its own semaphore, which the readers do not wait on.)
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

static_assert(
    std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "row all-reduce: 2D fabric expected");

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;

void kernel_main() {
    constexpr uint32_t cb_stage = get_compile_time_arg_val(0);
    constexpr uint32_t workers = get_compile_time_arg_val(1);
    constexpr uint32_t sem_gather = get_compile_time_arg_val(2);
    constexpr uint32_t slot = get_compile_time_arg_val(3);  // bytes a slot (stage, receive buffer)
    constexpr bool mcast = get_compile_time_arg_val(4) != 0;
    constexpr auto area_args = TensorAccessorArgs<5>();

    size_t a = 0;
    const uint32_t slice = get_arg_val<uint32_t>(a++);    // bytes a worker
    const uint32_t area_addr = get_arg_val<uint32_t>(a++);  // the DRAM landing area (pages of `slot` bytes)
    const uint32_t dst_sem = get_arg_val<uint32_t>(a++);  // the peers' receive semaphore for this chip (this parity)
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t opp_chip = get_arg_val<uint32_t>(a++);
    const uint32_t w_first = get_arg_val<uint32_t>(a++);
    const uint32_t w_step = get_arg_val<uint32_t>(a++);
    const uint32_t page0 = get_arg_val<uint32_t>(a++);  // this chip's first page (parity, this chip, worker 0)
    const size_t xy = a;  // the workers' NoC x, y
    a += 2 * workers;
    auto fwd = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    auto bwd = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);

    DeviceZoneScopedN("AR_SEND");
    fwd.open_start();
    bwd.open_start();
    PacketHeaderPool::reset();
    Header h0 = PacketHeaderPool::allocate_header();
    Header h1 = PacketHeaderPool::allocate_header();
    Header h2 = PacketHeaderPool::allocate_header();
    Header h3 = PacketHeaderPool::allocate_header();
    bool opp_via_fwd = false;
    if constexpr (mcast) {
        // h0: 2 hops the forward way, h1: the backward neighbour; h2: 2 hops the backward way, h3: the forward one
        uint16_t hops_f[4] = {0, 0, 0, 0}, hops_b[4] = {0, 0, 0, 0};
        hops_f[get_next_hop_router_direction(mesh_id, fwd_chip)] = 2;
        hops_b[get_next_hop_router_direction(mesh_id, bwd_chip)] = 2;
        fabric_set_mcast_route(h0, fwd_chip, mesh_id, hops_f[0], hops_f[1], hops_f[2], hops_f[3]);
        fabric_set_unicast_route(h1, bwd_chip, mesh_id);
        fabric_set_mcast_route(h2, bwd_chip, mesh_id, hops_b[0], hops_b[1], hops_b[2], hops_b[3]);
        fabric_set_unicast_route(h3, fwd_chip, mesh_id);
    } else {
        // h0: the forward neighbour, h1: the backward one, h2: the opposite chip
        fabric_set_unicast_route(h0, fwd_chip, mesh_id);
        fabric_set_unicast_route(h1, bwd_chip, mesh_id);
        fabric_set_unicast_route(h2, opp_chip, mesh_id);
        opp_via_fwd =
            get_next_hop_router_direction(mesh_id, opp_chip) == get_next_hop_router_direction(mesh_id, fwd_chip);
    }
    fwd.open_finish();
    bwd.open_finish();

    auto send = [&](tt::tt_fabric::WorkerToFabricEdmSender& conn, Header h, uint32_t src, uint64_t dst, uint64_t sem) {
        h->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true}, slice);
        conn.wait_for_empty_write_slot();
        conn.send_payload_without_header_non_blocking_from_address(src, slice);
        conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(h), sizeof(PACKET_HEADER_TYPE));
    };

    // the workers set their bits of the stage's semaphore as their slices land; each is sent as soon as it is there
    static_assert(workers <= 32, "row all-reduce: a bit a worker");
    auto* gathered = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather));
    uint32_t mine = 0;
    for (uint32_t w = w_first; w < workers; w += w_step) {
        mine |= 1u << w;
    }
    const uint32_t stage = get_read_ptr(cb_stage);
    const auto s_area = TensorAccessor(area_args, area_addr, slot);
    uint32_t k = 0;
    while (mine != 0) {
        invalidate_l1_cache();
        uint32_t ready = *gathered & mine;
        if (ready == 0) {
            continue;
        }
        mine &= ~ready;
        for (; ready != 0; ready &= ready - 1, ++k) {
            const uint32_t w = __builtin_ctz(ready);
            const uint32_t x = get_arg_val<uint32_t>(xy + 2 * w);
            const uint32_t y = get_arg_val<uint32_t>(xy + 2 * w + 1);
            const uint64_t dst = s_area.get_noc_addr(page0 + w, 0, 0);
            const uint64_t sem = safe_get_noc_addr(x, y, dst_sem, 0);
            const uint32_t src = stage + w * slot;
            if constexpr (mcast) {
                if ((k & 1) == 0) {
                    send(fwd, h0, src, dst, sem);
                    send(bwd, h1, src, dst, sem);
                } else {
                    send(bwd, h2, src, dst, sem);
                    send(fwd, h3, src, dst, sem);
                }
            } else {
                send(fwd, h0, src, dst, sem);
                send(bwd, h1, src, dst, sem);
                send(opp_via_fwd ? fwd : bwd, h2, src, dst, sem);
            }
            noc_async_writes_flushed();  // the headers are rewritten for the next worker
        }
    }
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
}
