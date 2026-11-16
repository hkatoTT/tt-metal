// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE's column collectives (col_collectives.py), a sender RISC on the sender core: one-shot over the 2D fabric,
// on the mesh column's ring of 8 chips.  Opens a connection to each ring neighbour's router (on its link) and sends
// its share of the groups of `group` workers (groups g_first, g_first + g_step, ..), each as soon as all its workers
// have put their slices in the stage (consecutive there), in one packet to the column's other chips, into their DRAM
// landing area (persistent DRAM, the same address on every chip: page (parity, this chip, the group's leader)), the
// packet a write fused with an increment (once the write is acknowledged) of the leader's receive semaphore for this
// chip:
//   mode 0 (gather): the same slices to all 7: a multicast of `hops_a` hops one way and one of `hops_b` the other (the
//                    two counts swapping from group to group);
//   mode 1 (scatter): the slices for each chip d (stage slots d workers + w), a unicast over the connection its route
//                    leaves by, the nearest chips first, the two ways in turn.
// (A 2D multicast may also deliver on this chip, into its own page, and bump its own semaphore, which no one waits on.)
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

static_assert(
    std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "column collectives: 2D fabric");

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
using Conn = tt::tt_fabric::WorkerToFabricEdmSender;

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(1);
    constexpr uint32_t workers = get_compile_time_arg_val(2);
    constexpr uint32_t sem_gather = get_compile_time_arg_val(3);
    constexpr uint32_t slice = get_compile_time_arg_val(4);  // bytes a worker
    constexpr uint32_t bits = get_compile_time_arg_val(5);   // of the stage's semaphore a worker sets (1 or 2)
    constexpr uint32_t group = get_compile_time_arg_val(6);  // workers a packet
    constexpr auto area_args = TensorAccessorArgs<7>();
    constexpr uint32_t groups = workers / group;
    constexpr uint32_t chips = 8;
    constexpr uint32_t page_bytes = 4096;  // a landing page

    size_t a = 0;
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t g_first = get_arg_val<uint32_t>(a++);
    const uint32_t g_step = get_arg_val<uint32_t>(a++);
    const uint32_t my = get_arg_val<uint32_t>(a++);
    const uint32_t area_addr = get_arg_val<uint32_t>(a++);  // the DRAM landing area
    const uint32_t page0 = get_arg_val<uint32_t>(a++);      // this chip's first page (parity, this chip, worker 0)
    const uint32_t dst_sem = get_arg_val<uint32_t>(a++);    // the leaders' receive semaphore for this chip (parity)
    const uint32_t hops_a = get_arg_val<uint32_t>(a++);
    const uint32_t hops_b = get_arg_val<uint32_t>(a++);
    const size_t chip_ids = a;  // the column's chips, by mesh row
    a += chips;
    const size_t xy = a;  // the group leaders' NoC x, y
    a += 2 * groups;
    auto fwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);
    auto bwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    DeviceZoneScopedN("CA_SEND");
    fwd.open_start();
    bwd.open_start();
    PacketHeaderPool::reset();
    Header h[chips];
    bool via_fwd[chips] = {};
    const auto dir_f = get_next_hop_router_direction(mesh_id, fwd_chip);
    if constexpr (mode == 0) {
        // h0: hops_a the forward way, h1: hops_b backward; h2: hops_b forward, h3: hops_a backward
        const auto dir_b = get_next_hop_router_direction(mesh_id, bwd_chip);
        for (uint32_t i = 0; i < 4; ++i) {
            h[i] = PacketHeaderPool::allocate_header();
            uint16_t hops[4] = {0, 0, 0, 0};
            const bool forward = (i & 1) == 0;
            hops[forward ? dir_f : dir_b] = (i == 0 || i == 3) ? hops_a : hops_b;
            fabric_set_mcast_route(h[i], forward ? fwd_chip : bwd_chip, mesh_id, hops[0], hops[1], hops[2], hops[3]);
        }
    } else {
        for (uint32_t d = 0; d < chips; ++d) {
            if (d == my) {
                continue;
            }
            const uint32_t chip = get_arg_val<uint32_t>(chip_ids + d);
            h[d] = PacketHeaderPool::allocate_header();
            fabric_set_unicast_route(h[d], chip, mesh_id);
            via_fwd[d] = get_next_hop_router_direction(mesh_id, chip) == dir_f;
        }
    }
    fwd.open_finish();
    bwd.open_finish();
    const auto s_area = TensorAccessor(area_args, area_addr, page_bytes);

    auto send = [&](Conn& conn, Header hd, uint32_t src, uint64_t dst, uint64_t sem, uint32_t n) {
        hd->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true}, n);
        conn.wait_for_empty_write_slot();
        conn.send_payload_without_header_non_blocking_from_address(src, n);
        conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(hd), sizeof(PACKET_HEADER_TYPE));
    };

    static_assert(workers * bits <= 32, "column collectives: the workers' bits in a word");
    auto* gathered = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather));
    constexpr uint32_t group_bits = (1u << (group * bits)) - 1;  // a group's workers' bits, from its first
    uint32_t mine = 0;  // bit g: group g still to send
    for (uint32_t g = g_first; g < groups; g += g_step) {
        mine |= 1u << g;
    }
    const uint32_t stage = get_read_ptr(cb_stage);
    constexpr uint32_t bytes = group * slice;
    uint32_t k = 0;
    while (mine != 0) {
        invalidate_l1_cache();
        const uint32_t word = *gathered;
        uint32_t ready = 0;
        for (uint32_t g = 0; g < groups; ++g) {
            ready |= ((word >> (g * group * bits)) & group_bits) == group_bits ? 1u << g : 0u;
        }
        ready &= mine;
        if (ready == 0) {
            continue;
        }
        mine &= ~ready;
        for (; ready != 0; ready &= ready - 1, ++k) {
            const uint32_t g = __builtin_ctz(ready);
            const uint32_t x = get_arg_val<uint32_t>(xy + 2 * g);
            const uint32_t y = get_arg_val<uint32_t>(xy + 2 * g + 1);
            const uint32_t w0 = g * group;
            const uint64_t dst = s_area.get_noc_addr(page0 + w0, 0, 0);
            const uint64_t sem = safe_get_noc_addr(x, y, dst_sem, 0);
            if constexpr (mode == 0) {
                const uint32_t src = stage + w0 * slice;
                if ((k & 1) == 0) {
                    send(fwd, h[0], src, dst, sem, bytes);
                    send(bwd, h[1], src, dst, sem, bytes);
                } else {
                    send(fwd, h[2], src, dst, sem, bytes);
                    send(bwd, h[3], src, dst, sem, bytes);
                }
            } else {
                // the nearest chips first, the two ways in turn: my + 1, my - 1, my + 2, .., my + 4
                for (uint32_t i = 1; i < chips; ++i) {
                    const uint32_t step = (i + 1) / 2;
                    const uint32_t d = (i & 1) ? (my + step) % chips : (my + chips - step) % chips;
                    send(via_fwd[d] ? fwd : bwd, h[d], stage + (d * workers + w0) * slice, dst, sem, bytes);
                }
            }
            noc_async_writes_flushed();  // the headers are rewritten for the next group
        }
    }
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
}
