// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The 2D x all-gather with relays (x2d_gather.py, the relay form): a relay RISC.  This chip (mesh row r, column c of
// parity p) gets half p of K of the other 7 rows' tokens from its own column's trunks (x2d_send.cpp: one 2D multicast a
// token pair along the column; trunk role k's pairs on link k, in blocks of G pairs, the last packet of a block fused
// with an increment of the counter (source row, role, way) on the relay core of link k).  A source 1 .. 3 hops back
// sends every block on way 0, one 1 .. 3 hops ahead on way 1, each in order on one path; the one 4 hops away the blocks
// b with (b + k) even on way 0 and the odd ones on way 1.  So a counter tells how many of its blocks have landed, and
// pair i of trunk role k from a source has landed once its counter passes block i / G (or (i / G) / 2 on a way of the
// opposite row).
// The relays pass those halves on to the row neighbours, whose parity is 1 - p and who need exactly that half: trunk
// role k's pairs of even index i (pair q = 2 i + k) to (r, c - 1) (side 0), the odd ones to (r, c + 1) (side 1), on
// link k, a 1-hop unicast each (a scatter write of the pair's two 4 KB halves into the neighbour's x_all at the same
// place); the neighbours in turn give this chip half 1 - p.  The last pair of each (source row, side, link) carries an
// increment, once written, of the relay-in semaphore on the neighbour's first relay core (its flush covers the route's
// earlier packets): 2 sides x 7 rows x 2 links = 28 a chip.
//   A relay core a link, BRISC side 1 and NCRISC side 0 (one relay core for both links, a RISC feeding two routers,
//   sent at ~11 GB/s each, half a link's speed; that form was removed).  Items in order
//   (pair, source row), each read from x_all (DRAM) into one of NB stage slots once landed (trids 1 .. NB) and sent in
//   turn.  At the end the
//   core's BRISC waits for its NCRISC (`done`), on the first relay core for the 28 relay-in increments too, sets the
//   counters of its links (and the relay-in one) back to 0 and flips its core's `par` (the set of counters the calls
//   alternate between).  Every wait is bounded (a lost increment ends the op with wrong values rather than a hang).
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

static_assert(std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "x2d: 2D fabric");

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
using Conn = tt::tt_fabric::WorkerToFabricEdmSender;

namespace {
constexpr uint32_t SPIN_LIMIT = 1u << 27;  // (a bounded wait: well past any call's time)

FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

FORCE_INLINE bool reached(uint32_t addr, uint32_t v) {
    invalidate_l1_cache();
    return *word(addr) >= v;
}

// wait until *addr >= v (bounded); false on a timeout
FORCE_INLINE bool wait_min(uint32_t addr, uint32_t v) {
    for (uint32_t n = 0; n < SPIN_LIMIT; ++n) {
        if (reached(addr, v)) {
            return true;
        }
    }
    return false;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);  // 0 BRISC, 1 NCRISC (its half of the stage)
    constexpr uint32_t T = get_compile_time_arg_val(1);     // tokens a mesh row
    constexpr uint32_t NB = get_compile_time_arg_val(2);    // stage slots a RISC (8 KB each)
    constexpr uint32_t cb_stage = get_compile_time_arg_val(3);
    constexpr uint32_t sem_done = get_compile_time_arg_val(4);  // NCRISC -> BRISC
    constexpr uint32_t G = get_compile_time_arg_val(5);         // pairs a trunk block
    // (tests, the values wrong: bit 0 send without waiting for the trunks; bit 1 read but send nothing, and wait for no
    // relay-in; bit 2 read nothing from DRAM: send the stage as it is)
    constexpr uint32_t DBG = get_compile_time_arg_val(6);
    constexpr uint32_t LINK = get_compile_time_arg_val(7);     // the trunk role (and the link) whose pairs it relays
    constexpr uint32_t SIDE = get_compile_time_arg_val(8);     // 0 (c - 1, the even pairs) or 1 (c + 1, the odd ones)
    constexpr uint32_t WAIT_IN = get_compile_time_arg_val(9);  // (BRISC) wait for the relay-in increments here
    constexpr uint32_t RESET = get_compile_time_arg_val(10);   // (BRISC) the links whose counters it sets back (bits)
    constexpr auto xa_args = TensorAccessorArgs<11>();
    constexpr uint32_t HALF = 4096, SLOT = 2 * HALF, MINE = T / 4, ROWS = 8, OTHERS = ROWS - 1;
    constexpr uint32_t PER = OTHERS;
    constexpr uint32_t TOTAL = (MINE / 2) * OTHERS, EXPECT_IN = 2 * OTHERS * 2;
    static_assert(T % 16 == 0 && NB >= 2 && NB <= 15, "x2d relay: 16 tokens a row at least; transaction ids 1 .. 15");
    static_assert(G >= 1 && MINE >= 2 && SIDE <= 1 && LINK <= 1, "x2d relay: args");

    size_t a = 0;
    const uint32_t xa_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r = get_arg_val<uint32_t>(a++);  // this chip's mesh row
    const uint32_t p = get_arg_val<uint32_t>(a++);  // its column's parity: the half of K it gets from its trunks
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t peer = get_arg_val<uint32_t>(a++);  // this side's row neighbour: (r, c - 1) side 0, (r, c + 1) side 1
    const uint32_t par_addr = get_arg_val<uint32_t>(a++);  // which set of counters this call's (global, this core's)
    const uint32_t in_x = get_arg_val<uint32_t>(a++);      // the first relay core's NoC x, y (the relay-in semaphore)
    const uint32_t in_y = get_arg_val<uint32_t>(a++);
    uint32_t in_sem[2], cnt[2][ROWS][2][2];  // the relay-in semaphores; the trunk counters (set, source row, role, way)
    for (uint32_t s = 0; s < 2; ++s) {
        in_sem[s] = get_arg_val<uint32_t>(a++);
    }
    for (uint32_t s = 0; s < 2; ++s) {
        for (uint32_t q = 0; q < ROWS; ++q) {
            for (uint32_t k = 0; k < 2; ++k) {
                for (uint32_t w = 0; w < 2; ++w) {
                    cnt[s][q][k][w] = get_arg_val<uint32_t>(a++);
                }
            }
        }
    }

    const auto s_xa = TensorAccessor(xa_args, xa_addr);
    const uint32_t stage = get_write_ptr(cb_stage) + role * NB * SLOT;
    volatile tt_l1_ptr uint32_t* par = word(par_addr);
    volatile tt_l1_ptr uint32_t* done = word(get_semaphore(sem_done));
    invalidate_l1_cache();
    const uint32_t set = *par == 0 ? 0 : 1;
    const uint64_t in_noc = safe_get_noc_addr(in_x, in_y, in_sem[set], 0);

    // item j: pair i = 2 (j / PER) + SIDE of trunk role LINK, source row (r + 1 + j mod PER) mod 8
    auto pair = [&](uint32_t j) { return 2 * (j / PER) + SIDE; };
    auto source = [&](uint32_t j) { return (r + 1 + j % PER) % ROWS; };
    // the counter and the value it must reach once item j has landed
    auto need = [&](uint32_t j, uint32_t& addr) -> uint32_t {
        const uint32_t rs = source(j), d = (r + ROWS - rs) % ROWS, k = pair(j) / G;
        if (d < 4) {
            addr = cnt[set][rs][LINK][0];
            return k + 1;
        }
        if (d > 4) {
            addr = cnt[set][rs][LINK][1];
            return k + 1;
        }
        addr = cnt[set][rs][LINK][(k + LINK) & 1];
        return k / 2 + 1;
    };
    auto landed = [&](uint32_t j) {
        if constexpr (DBG & 1) {
            return true;
        }
        uint32_t addr;
        const uint32_t v = need(j, addr);
        return reached(addr, v);
    };
    auto read_item = [&](uint32_t j) {
        const uint32_t q = 2 * pair(j) + LINK, rs = source(j);
        const uint32_t slot = stage + (j % NB) * SLOT;
        if constexpr (DBG & 4) {
            return;
        }
        noc_async_read_set_trid(j % NB + 1);
        noc_async_read(s_xa.get_noc_addr(rs * T + 2 * q, p * HALF), slot, HALF);
        noc_async_read(s_xa.get_noc_addr(rs * T + 2 * q + 1, p * HALF), slot + HALF, HALF);
    };

    PacketHeaderPool::reset();
    Header hd = PacketHeaderPool::allocate_header();  // toward this side's neighbour
    fabric_set_unicast_route(hd, peer, mesh_id);

    bool failed = false;
    auto conn = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);  // toward this side's neighbour on link LINK
    conn.open();
    {
        DeviceZoneScopedN("X2D_RELAY");
        uint32_t issued = 0, sent = 0;
        while (sent < TOTAL) {
            while (issued < TOTAL && issued - sent < NB && landed(issued)) {
                read_item(issued);
                ++issued;
            }
            if (issued == sent) {  // nothing in flight: wait for the next item to land (bounded)
                uint32_t addr;
                const uint32_t v = need(issued, addr);
                if (!wait_min(addr, v)) {
                    failed = true;
                    break;
                }
                continue;
            }
            const uint32_t j = sent, i = pair(j), q = 2 * i + LINK, rs = source(j);
            const bool last = i + 2 >= MINE;  // the last pair of this (source, side)
            const uint32_t src = stage + (j % NB) * SLOT;
            const uint64_t d0 = s_xa.get_noc_addr(rs * T + 2 * q, p * HALF, 0);
            const uint64_t d1 = s_xa.get_noc_addr(rs * T + 2 * q + 1, p * HALF, 0);
            noc_async_read_barrier_with_trid(j % NB + 1);
            if constexpr (DBG & 2) {
                ++sent;
                continue;
            }
            if (last) {
                hd->to_noc_fused_unicast_scatter_write_atomic_inc(
                    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader(
                        {d0, d1}, in_noc, {static_cast<uint16_t>(HALF)}, 1, true),
                    SLOT);
            } else {
                hd->to_noc_unicast_scatter_write(
                    tt::tt_fabric::NocUnicastScatterCommandHeader({d0, d1}, {static_cast<uint16_t>(HALF)}), SLOT);
            }
            uint32_t n = 0;
            while (!conn.edm_has_space_for_packet() && ++n < SPIN_LIMIT) {
            }
            if (n >= SPIN_LIMIT) {  // (the router took nothing for a long time)
                failed = true;
                break;
            }
            conn.send_payload_without_header_non_blocking_from_address(src, SLOT);
            conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(hd), sizeof(PACKET_HEADER_TYPE));
            noc_async_writes_flushed();  // (the header is rewritten next item; the slot is read again NB items on)
            ++sent;
        }
        noc_async_read_barrier();  // (an early end leaves reads in flight)
        noc_async_read_set_trid(0);
    }
    conn.close();
    noc_async_write_barrier();
    if constexpr (role == 1) {
        *done = 1;
    } else {
        DeviceZoneScopedN("X2D_RELAY_WAIT");
        failed |= !wait_min(get_semaphore(sem_done), 1);
        *done = 0;
        if constexpr (WAIT_IN && !(DBG & 2)) {
            failed |= !wait_min(in_sem[set], EXPECT_IN);  // every relayed pair into this chip has landed
        }
        if constexpr (WAIT_IN) {
            *word(in_sem[set]) = 0;
        }
        for (uint32_t q = 0; q < ROWS; ++q) {
            for (uint32_t k = 0; k < 2; ++k) {
                if (RESET & (1u << k)) {
                    for (uint32_t w = 0; w < 2; ++w) {
                        *word(cnt[set][q][k][w]) = 0;
                    }
                }
            }
        }
        *par = *par == 0 ? 1 : 0;
    }
    (void)failed;
}
