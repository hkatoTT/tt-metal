// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sequence-parallel KDA's state maps to the rows that read them (kda_maps_x.py, GLM53_KDA_MAPS_X),
// a sender RISC: this chip (mesh row r) holds its segment's maps `packed` [1, 16, 128, 256] (512 bf16 tiles) and the
// column's landing `land` [8, 16, 128, 256] (persistent DRAM, the same address on every chip) gets them at slot r
// (pages 512 r ..) on every chip of its arc: `b` hops one way along the column's ring and `a` the other (a 2D multicast
// each way, its table row from the host: the fewest hops that reach every other row of its request; a row of no
// request or alone sends nothing).  The chips on the arc that are not of its request take the slot too (they never
// read it: chain_affine_transforms streams the identity for the rows outside a request's).
//   role 0 (BRISC, NoC 0, link 0) sends the even packets, role 1 (NCRISC, NoC 1, link 1) the odd ones: a packet two
//   tiles (a scatter write into their two pages), read NB ahead into this role's stage slots by transaction ids; the
//   last packet of each way fused with an increment (once its writes are flushed) of this row's `data` counter of this
//   call's parity on the receivers' sender core (this core's place on every chip).  A counter a source row, so that
//   a multicast that also delivers on its own chip bumps only a counter no one there waits on.
//   Before any packet, role 0 tells the chips of the table's `tell` (every other chip of the column) that this chip is
//   in the call (an increment of its `ready` counter: a unicast each), and waits for the table's count of them (from
//   every other chip: so that no packet of this call lands before a chip's previous call, whose chain read the slot,
//   is over, and no chip runs two calls ahead of another, past the two counter sets); role 1 waits for role 0.
//   At the end role 0 waits for the data it takes (two increments, one a role, from each row whose arc reaches it),
//   sets the counters of this parity back to 0 once role 1 is done, and flips the parity (a word of persistent L1, the same on
//   every chip and right through traces, as x2d_send.cpp's).  Every wait is bounded (a lost increment ends the call
//   with wrong values, not a hang).
// The table row (int32 x 8, row-major, a page a mesh row): a, b, the rows whose arcs reach this one (bit s), the ready
// increments to wait for (7), the rows to tell (bit d: every other row), (unused) x 3.
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

static_assert(std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "kda maps: 2D fabric");

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
using Conn = tt::tt_fabric::WorkerToFabricEdmSender;

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);  // stage slots a role (two tiles each)
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t cb_table = get_compile_time_arg_val(3);  // 256 B: the table row, at its source's offset mod 64
    constexpr uint32_t sem_go = get_compile_time_arg_val(4);    // role 0 -> 1: the ready counts are in
    constexpr uint32_t sem_done = get_compile_time_arg_val(5);  // role 1 -> 0: its packets are out
    constexpr uint32_t TILES = get_compile_time_arg_val(6);     // a slot's tiles (512)
    constexpr auto p_args = TensorAccessorArgs<7>();
    constexpr auto l_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048, SLOT = 2 * TILE, CHIPS = 8;
    constexpr uint32_t PACKETS = TILES / 2, MINE = PACKETS / 2;  // (role: packets 2 i + role)
    static_assert(TILES % 4 == 0 && NB >= 2 && NB <= 15, "kda maps: tiles in fours; transaction ids 1 .. 15");
    constexpr uint32_t SPIN_LIMIT = 1u << 27;  // (bounded waits)

    size_t a = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(a++);
    const uint32_t l_addr = get_arg_val<uint32_t>(a++);
    const uint32_t t_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r = get_arg_val<uint32_t>(a++);  // this chip's mesh row
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t col_fwd = get_arg_val<uint32_t>(a++);  // the ring neighbours' chip ids: (r + 1, c), (r - 1, c)
    const uint32_t col_bwd = get_arg_val<uint32_t>(a++);
    const uint32_t par_addr = get_arg_val<uint32_t>(a++);  // the parity word (global semaphore)
    const uint32_t ready0 = get_arg_val<uint32_t>(a++);    // the ready counters (global semaphores) of parity 0 / 1
    const uint32_t ready1 = get_arg_val<uint32_t>(a++);
    const size_t data_addrs = a;  // the data counters (global semaphores): [parity][source row]
    a += 2 * CHIPS;
    const uint32_t me_x = get_arg_val<uint32_t>(a++);  // this core's NoC x, y (the counters' core on every chip)
    const uint32_t me_y = get_arg_val<uint32_t>(a++);
    const size_t chip_ids = a;  // the column's chips, by mesh row
    a += CHIPS;
    auto fwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);  // toward col_fwd on this role's link
    auto bwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);  // toward col_bwd

    const auto s_p = TensorAccessor(p_args, p_addr, TILE);
    const auto s_l = TensorAccessor(l_args, l_addr, TILE);
    const auto s_t = TensorAccessor(t_args, t_addr, 32);
    const uint32_t stage = get_write_ptr(cb_stage) + role * NB * SLOT;
    volatile tt_l1_ptr uint32_t* par = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(par_addr);
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go));
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done));

    // the table row: 32 bytes read into the CB at the source's offset mod 64 (a read under 64 B: the same offset)
    const uint64_t t_noc = s_t.get_noc_addr(0);
    const uint32_t t_l1 = ((get_write_ptr(cb_table) + 63) & ~63u) + (static_cast<uint32_t>(t_noc) & 63u);
    noc_async_read(t_noc, t_l1, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t_l1);
    const uint32_t hops_b = row[0], hops_f = row[1], from = row[2], expect_ready = row[3], tell = row[4];

    invalidate_l1_cache();
    const uint32_t p = *par;
    const uint32_t ready = p == 0 ? ready0 : ready1;
    auto data_of = [&](uint32_t src) { return get_arg_val<uint32_t>(data_addrs + p * CHIPS + src); };
    volatile tt_l1_ptr uint32_t* ready_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
    bool failed = false;
    auto wait_min = [&](volatile tt_l1_ptr uint32_t* w, uint32_t v) {
        uint32_t n = 0;
        for (; n < SPIN_LIMIT; ++n) {
            invalidate_l1_cache();
            if (*w >= v) {
                break;
            }
        }
        failed |= n >= SPIN_LIMIT;
    };

    auto read_packet = [&](uint32_t i) {  // this role's packet i (tiles 2 q, 2 q + 1, q = 2 i + role) into slot i mod NB
        const uint32_t q = 2 * i + role;
        const uint32_t slot = stage + (i % NB) * SLOT;
        noc_async_read_set_trid(i % NB + 1);
        noc_async_read(s_p.get_noc_addr(2 * q), slot, TILE);
        noc_async_read(s_p.get_noc_addr(2 * q + 1), slot + TILE, TILE);
    };
    const bool sends = hops_f + hops_b > 0;
    if (sends) {
        for (uint32_t i = 0; i < NB && i < MINE; ++i) {  // (the packed maps do not wait for anything)
            read_packet(i);
        }
    }

    fwd.open_start();
    bwd.open_start();
    PacketHeaderPool::reset();
    const auto dir_f = get_next_hop_router_direction(mesh_id, col_fwd);
    const auto dir_b = get_next_hop_router_direction(mesh_id, col_bwd);
    Header h[2];  // way 0: hops_f toward col_fwd, 1: hops_b toward col_bwd
    for (uint32_t way = 0; way < 2; ++way) {
        h[way] = PacketHeaderPool::allocate_header();
        uint16_t hops[4] = {0, 0, 0, 0};
        const uint32_t n = way == 0 ? hops_f : hops_b;
        hops[way == 0 ? dir_f : dir_b] = static_cast<uint16_t>(n);
        if (n > 0) {
            fabric_set_mcast_route(h[way], way == 0 ? col_fwd : col_bwd, mesh_id, hops[0], hops[1], hops[2], hops[3]);
        }
    }
    Header hr = PacketHeaderPool::allocate_header();  // (role 0: the ready increments, a unicast each)
    fwd.open_finish();
    bwd.open_finish();

    auto send_conn = [&](Conn& conn, uint32_t src, uint32_t n_bytes, Header hd) {
        uint32_t n = 0;
        while (!conn.edm_has_space_for_packet() && ++n < SPIN_LIMIT) {
        }
        if (n >= SPIN_LIMIT) {
            failed = true;
            return;
        }
        if (n_bytes > 0) {
            conn.send_payload_without_header_non_blocking_from_address(src, n_bytes);
        }
        conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(hd), sizeof(PACKET_HEADER_TYPE));
    };

    if constexpr (role == 0) {
        DeviceZoneScopedN("KMX_READY");
        const uint64_t ready_noc = safe_get_noc_addr(me_x, me_y, ready, 0);
        for (uint32_t d = 0; d < CHIPS; ++d) {
            if (d == r || (tell & (1u << d)) == 0) {
                continue;
            }
            const uint32_t chip = get_arg_val<uint32_t>(chip_ids + d);
            noc_async_writes_flushed();  // (the header is rewritten: its last send has left)
            hr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{ready_noc, 1, true});
            fabric_set_unicast_route(hr, chip, mesh_id);
            send_conn(get_next_hop_router_direction(mesh_id, chip) == dir_f ? fwd : bwd, 0, 0, hr);
        }
        wait_min(ready_l1, expect_ready);
        *go = 1;
    } else {
        wait_min(go, 1);
    }

    if (sends) {
        DeviceZoneScopedN("KMX_SEND");
        const uint64_t data_noc = safe_get_noc_addr(me_x, me_y, data_of(r), 0);
        for (uint32_t i = 0; i < MINE && !failed; ++i) {
            const uint32_t q = 2 * i + role;
            const bool last = i + 1 == MINE;
            const uint32_t src = stage + (i % NB) * SLOT;
            const uint64_t d0 = s_l.get_noc_addr(r * TILES + 2 * q);
            const uint64_t d1 = s_l.get_noc_addr(r * TILES + 2 * q + 1);
            noc_async_read_barrier_with_trid(i % NB + 1);
            noc_async_writes_flushed();  // (the headers are rewritten: their last sends have left)
            for (uint32_t way = 0; way < 2; ++way) {
                if ((way == 0 ? hops_f : hops_b) == 0) {
                    continue;
                }
                Header hd = h[way];
                if (last) {
                    hd->to_noc_fused_unicast_scatter_write_atomic_inc(
                        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader(
                            {d0, d1}, data_noc, {static_cast<uint16_t>(TILE)}, 1, true),
                        SLOT);
                } else {
                    hd->to_noc_unicast_scatter_write(
                        tt::tt_fabric::NocUnicastScatterCommandHeader({d0, d1}, {static_cast<uint16_t>(TILE)}), SLOT);
                }
                send_conn(way == 0 ? fwd : bwd, src, SLOT, hd);
            }
            if (i + NB < MINE) {
                noc_async_writes_flushed();  // slot i mod NB again: once its sends have left L1
                read_packet(i + NB);
            }
        }
    }
    noc_async_read_barrier();  // (a stuck router leaves reads in flight)
    noc_async_read_set_trid(0);
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    if constexpr (role == 1) {
        *done = 1;
    } else {
        DeviceZoneScopedN("KMX_WAIT");
        for (uint32_t s = 0; s < CHIPS; ++s) {  // every packet into this chip has landed: two increments a source
            if (s != r && (from & (1u << s)) != 0) {
                wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_of(s)), 2);
            }
        }
        wait_min(done, 1);
        *ready_l1 = 0;
        for (uint32_t s = 0; s < CHIPS; ++s) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_of(s)) = 0;
        }
        *par = p == 0 ? 1 : 0;
    }
}
