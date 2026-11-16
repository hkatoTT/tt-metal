// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The prefill MoE's x all-gather over the mesh column in two dimensions (x2d_gather.py, the relay form), the sender
// core: this chip (mesh row r, column c of parity p = c mod 2) holds its row's x [T, 4096] (row-major bf16, a page a
// token: the same on the row's 4 chips) and the column's x_all [8 T, 4096] gets row r's tokens at r T (the layout of
// ttnn.all_gather over the column; persistent DRAM, the same address on every chip).  This core sends its row's half p
// of K (columns p 2048 ..: 4 KB a token) down its own mesh column, one 2D multicast a packet along the column (4 hops
// one way and 3 the other): the trunks, which the relay cores of the receiving chips pass on along their mesh rows
// (x2d_relay.cpp; the row neighbours, of the other parity, need exactly that half).  The own row's x the copy cores
// write (x2d_copy.cpp).
//   A packet: tokens 2 q, 2 q + 1's half p (2 x 4 KB, a scatter write into their two pages of x_all); each goes out
//   twice (the two trunk ways).  role 0 (BRISC, NoC 0) sends the even pairs on link 0, role 1 (NCRISC, NoC 1) the odd
//   ones on link 1, each reading its pairs' halves into its own stage slots, NB ahead, by transaction ids.  In blocks of
//   G pairs, the hop counts swapping from block to block (block k's: (k + role) mod 2), the last packet of each block
//   (on both ways) fused with an increment of the counter (this row, role, way) on the receivers' relay core (sem0 /
//   sem1 the way-0 counter of its two sets, sem0b / sem1b the way-1 one; me_x / me_y that core).  A receiver 1 .. 3
//   hops away gets every block on one way, in order (one path); the one 4 hops away the even and odd blocks on the two
//   ways, each way's in order with its own counter.  The sender waits for nothing (the relay core does).  The two sets
//   of counters alternate from call to call by `par` (a word of persistent L1, flipped at the end of each call: the
//   same on every chip, and right through traces); role 0 flips it once role 1 is done with it (`done`).
//   (The teeth into the neighbour columns, mode 0, and the test modes 1 / 2 were removed.)
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

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t T = get_compile_time_arg_val(1);   // tokens a mesh row
    constexpr uint32_t NB = get_compile_time_arg_val(2);  // stage slots a role (8 KB each)
    constexpr uint32_t cb_stage = get_compile_time_arg_val(3);
    constexpr uint32_t sem_done = get_compile_time_arg_val(4);  // role 1 -> 0: done with `par`
    constexpr uint32_t G = get_compile_time_arg_val(5);         // an increment every G pairs
    // (tests, the values wrong: bit 0 the packets land in the receivers' sender cores' stage (L1), not x_all)
    constexpr uint32_t SDBG = get_compile_time_arg_val(6);
    // (tests) the role alone on its core (x2d_gather.py `ssplit` 2: a sender core a role): no `done`, it flips its own
    // core's `par`
    constexpr uint32_t SOLO = get_compile_time_arg_val(7);
    constexpr auto xr_args = TensorAccessorArgs<8>();
    constexpr auto xa_args = TensorAccessorArgs<xr_args.next_compile_time_args_offset()>();
    constexpr uint32_t HALF = 4096, SLOT = 2 * HALF, PAIRS = T / 2, MINE = PAIRS / 2;
    static_assert(T % 16 == 0 && NB >= 2 && NB <= 15, "x2d: 16 tokens a row at least; transaction ids 1 .. 15");
    static_assert(G >= 1, "x2d: G");
    constexpr uint32_t SPIN_LIMIT = 1u << 27;  // (bounded waits)

    size_t a = 0;
    const uint32_t xr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t xa_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r = get_arg_val<uint32_t>(a++);  // this chip's mesh row
    const uint32_t p = get_arg_val<uint32_t>(a++);  // its column's parity: the half of K it sends
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t col_fwd = get_arg_val<uint32_t>(a++);  // the ring neighbours' chip ids: (r + 1, c), (r - 1, c)
    const uint32_t col_bwd = get_arg_val<uint32_t>(a++);
    const uint32_t sem0 = get_arg_val<uint32_t>(a++);  // the receivers' way-0 counters (global: the same on every chip)
    const uint32_t sem1 = get_arg_val<uint32_t>(a++);
    const uint32_t par_addr = get_arg_val<uint32_t>(a++);  // which of them this call's (global)
    // the NoC x, y of the core whose semaphores the packets increment (the receivers' relay core of this link)
    const uint32_t me_x = get_arg_val<uint32_t>(a++);
    const uint32_t me_y = get_arg_val<uint32_t>(a++);
    const uint32_t sem0b = get_arg_val<uint32_t>(a++);  // the way-1 counters
    const uint32_t sem1b = get_arg_val<uint32_t>(a++);
    const uint32_t own_x = get_arg_val<uint32_t>(a++);  // (SDBG) this core's NoC x, y
    const uint32_t own_y = get_arg_val<uint32_t>(a++);
    auto fwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);  // toward col_fwd on this role's link
    auto bwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);  // toward col_bwd

    const auto s_xr = TensorAccessor(xr_args, xr_addr);
    const auto s_xa = TensorAccessor(xa_args, xa_addr);
    const uint32_t stage = get_write_ptr(cb_stage) + role * NB * SLOT;
    volatile tt_l1_ptr uint32_t* par = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(par_addr);
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done));
    invalidate_l1_cache();
    const uint32_t sem = *par == 0 ? sem0 : sem1;
    const uint32_t semb = *par == 0 ? sem0b : sem1b;

    // pair q = 2 i + role's halves p into slot i mod NB: [token 2 q | token 2 q + 1]
    auto read_pair = [&](uint32_t i) {
        const uint32_t q = 2 * i + role;
        const uint32_t slot = stage + (i % NB) * SLOT;
        noc_async_read_set_trid(i % NB + 1);
        noc_async_read(s_xr.get_noc_addr(2 * q, p * HALF), slot, HALF);
        noc_async_read(s_xr.get_noc_addr(2 * q + 1, p * HALF), slot + HALF, HALF);
    };

    fwd.open_start();
    bwd.open_start();
    PacketHeaderPool::reset();
    // h[swap][way]: way 0 the trunk toward col_fwd, 1 toward col_bwd; swap 0: 4 hops forward and 3 backward, 1: 3 and 4
    Header h[2][2];
    {
        const auto dir_f = get_next_hop_router_direction(mesh_id, col_fwd);
        const auto dir_b = get_next_hop_router_direction(mesh_id, col_bwd);
        for (uint32_t swap = 0; swap < 2; ++swap) {
            for (uint32_t way = 0; way < 2; ++way) {
                Header hd = PacketHeaderPool::allocate_header();
                uint16_t hops[4] = {0, 0, 0, 0};
                const bool forward = way == 0;
                hops[forward ? dir_f : dir_b] = (forward == (swap == 0)) ? 4 : 3;
                fabric_set_mcast_route(hd, forward ? col_fwd : col_bwd, mesh_id, hops[0], hops[1], hops[2], hops[3]);
                h[swap][way] = hd;
            }
        }
    }
    fwd.open_finish();
    bwd.open_finish();
    for (uint32_t i = 0; i < NB && i < MINE; ++i) {  // (after the connections' own reads)
        read_pair(i);
    }

    const uint64_t sem_noc = safe_get_noc_addr(me_x, me_y, sem, 0);
    const uint64_t semb_noc = safe_get_noc_addr(me_x, me_y, semb, 0);
    bool stuck = false;
    {
    DeviceZoneScopedN("X2D_SEND");
    for (uint32_t i = 0; i < MINE && !stuck; ++i) {
        const uint32_t q = 2 * i + role;
        // this role's pairs i in blocks of G pairs: the swap (block + role) mod 2, the last of a block (or of all)
        // carrying the increment
        const uint32_t swap = (i / G + role) & 1;
        const bool last = (i + 1) % G == 0 || i + 1 == MINE;
        const uint32_t src = stage + (i % NB) * SLOT;
        uint64_t d0 = s_xa.get_noc_addr(r * T + 2 * q, p * HALF, 0);
        uint64_t d1 = s_xa.get_noc_addr(r * T + 2 * q + 1, p * HALF, 0);
        if constexpr (SDBG & 1) {
            d0 = safe_get_noc_addr(own_x, own_y, stage + (i % NB) * SLOT, 0);
            d1 = d0 + HALF;
        }
        noc_async_read_barrier_with_trid(i % NB + 1);
        noc_async_writes_flushed();  // (the headers are rewritten: their last sends have left)
        for (uint32_t way = 0; way < 2; ++way) {
            Header hd = h[swap][way];
            if (last) {
                hd->to_noc_fused_unicast_scatter_write_atomic_inc(
                    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader(
                        {d0, d1}, way == 1 ? semb_noc : sem_noc, {static_cast<uint16_t>(HALF)}, 1, true),
                    SLOT);
            } else {
                hd->to_noc_unicast_scatter_write(
                    tt::tt_fabric::NocUnicastScatterCommandHeader({d0, d1}, {static_cast<uint16_t>(HALF)}), SLOT);
            }
            Conn& conn = way == 0 ? fwd : bwd;
            uint32_t n = 0;
            while (!conn.edm_has_space_for_packet() && ++n < SPIN_LIMIT) {
            }
            if (n >= SPIN_LIMIT) {  // (the router took nothing for a long time: end with wrong values, not a hang)
                stuck = true;
                break;
            }
            conn.send_payload_without_header_non_blocking_from_address(src, SLOT);
            conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(hd), sizeof(PACKET_HEADER_TYPE));
        }
        if (i + NB < MINE) {
            noc_async_writes_flushed();  // slot i mod NB again: once its sends have left L1
            read_pair(i + NB);
        }
    }
    }
    noc_async_read_barrier();  // (a stuck router leaves reads in flight)
    noc_async_read_set_trid(0);
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    if constexpr (SOLO) {
        *par = *par == 0 ? 1 : 0;
    } else if constexpr (role == 1) {
        *done = 1;
    } else {  // (the relay core waits for the packets into this chip)
        DeviceZoneScopedN("X2D_WAIT");
        for (uint32_t n = 0; n < SPIN_LIMIT; ++n) {
            invalidate_l1_cache();
            if (*done != 0) {
                break;
            }
        }
        *done = 0;
        *par = *par == 0 ? 1 : 0;
    }
}
