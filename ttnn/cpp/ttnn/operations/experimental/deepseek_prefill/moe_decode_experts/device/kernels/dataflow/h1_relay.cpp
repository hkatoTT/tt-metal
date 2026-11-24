// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The in-op gather's row relay (MOE_H1_RELAY, MoeDecodeExpertsParams::h1_relay), from 32 tokens a mesh row.  The
// gather's senders (h1_send.cpp) of mesh column c send only half of K down the column, the K slices 4 h .. 4 h + 3
// (h = c mod 2), so the column's links carry half the bytes; the other half of each other mesh row's x comes over the
// mesh row from the row neighbours, whose columns carry it.  This chip (mesh row r, column c) passes its column's
// slices of the other 7 rows on: slices 4 h, 4 h + 1 to (r, c + 1) (relay core 0, `dir` 0, E) and 4 h + 2, 4 h + 3 to
// (r, c - 1) (relay core 1, W), into the neighbour's top cores' CB_XALL at the same place (the same address on every
// chip), each 32 KB group (a tile row q of a mesh row m's slice: 16 tiles) in packets of up to 8 KB alternating
// between the links, the last one on each link fused with an increment, once written, of 1 << 4 m on the top core's
// sem_x: the same counts as the column's (2 a group, h1_send.cpp), so the top cores count a row from either way alike.
// The neighbour, with the other parity, gets slices 4 h .. 4 h + 3 of each row from (r, c - 1) and (r, c + 1): all 4.
//   A group goes once its row has landed here: the slice's top core raises bit m of this core's notice word
//   (RELAY_SEM_N0 + slice mod 2) when mesh row m's x has landed there (gu_ctrl.cpp).  Before the first packet, the
//   handshake with the neighbour's relay core facing this one: this chip's top cores in the program (each raises
//   RELAY_SEM_TOPS here, with CB_XALL's address in RELAY_SEM_MAIL), an increment of its ready word (the global
//   semaphore sem_r on that core, over the fabric), and the wait for this core's (raised by the neighbour likewise).
//   role 0 (BRISC, NoC 0, link 0): the handshake, the groups' reads from the top cores into the stage ring (NB groups,
//          transaction id slot + 1), in the order their rows land, published to role 1 (the group's id in the list
//          after the ring, RELAY_SEM_PUB the count), and the group's packets of link 0;
//   role 1 (NCRISC, NoC 1, link 1, no reads): the published groups' packets of link 1, counted in RELAY_SEM_SENT, which
//          role 0 waits for before it reads into a slot again.
// The same bytes into the same places as the column's gather sends would write: the same x, the same values.
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "moe_decode_experts_common.hpp"

static_assert(std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "row relay: 2D fabric");

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
using Conn = tt::tt_fabric::WorkerToFabricEdmSender;

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t t = get_compile_time_arg_val(1);   // tokens a mesh row (32 or 64)
    constexpr uint32_t ks = get_compile_time_arg_val(2);  // a top core's K slice (tiles)
    constexpr uint32_t cb_stage = get_compile_time_arg_val(3);
    constexpr uint32_t NB = get_compile_time_arg_val(4);    // the stage ring's groups
    constexpr uint32_t maxp = get_compile_time_arg_val(5);  // bytes a packet at most
#ifdef MOE_X_TILE
    constexpr uint32_t xtile = MOE_X_TILE;  // (BFP8_b x: as many packets a group as bf16's, each smaller)
#else
    constexpr uint32_t xtile = 2048;
#endif
    constexpr uint32_t chips = 8, tr = t / 32, gbytes16 = ks * 2048;
    constexpr uint32_t gbytes = ks * xtile, gpk = gbytes16 / (gbytes16 < maxp ? gbytes16 : maxp), pbytes = gbytes / gpk;
    constexpr uint32_t total = 2 * (chips - 1) * tr;  // groups: 2 slices x 7 rows x tr tile rows
    static_assert(t >= 32 && t % 32 == 0 && gbytes % gpk == 0 && pbytes % 64 == 0 && gpk >= 2 && gpk % 2 == 0,
                  "row relay: groups");
    static_assert(NB >= 2 && NB <= 15, "row relay: transaction ids 1 .. 15");

    size_t a = 0;
    const uint32_t r = get_arg_val<uint32_t>(a++);        // this chip's mesh row
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t peer = get_arg_val<uint32_t>(a++);     // the row neighbour (fabric chip id) this core sends to
    const uint32_t sem_x = get_arg_val<uint32_t>(a++);    // the global semaphores' L1 address: the top cores' landings
    const uint32_t sem_r = get_arg_val<uint32_t>(a++);    // ... and this core's (and the peer's) ready word
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);   // the neighbour's relay core facing this one (NoC x, y)
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    uint32_t top_x[2], top_y[2];  // this core's two slices' top cores (NoC x, y: the same on every chip)
    for (uint32_t s = 0; s < 2; ++s) {
        top_x[s] = get_arg_val<uint32_t>(a++);
        top_y[s] = get_arg_val<uint32_t>(a++);
    }
    auto conn = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    const uint32_t stage = get_write_ptr(cb_stage);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + NB * gbytes);
    volatile tt_l1_ptr uint32_t* pub = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(RELAY_SEM_PUB));
    volatile tt_l1_ptr uint32_t* sent = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(RELAY_SEM_SENT));
    volatile tt_l1_ptr uint32_t* mail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(RELAY_SEM_MAIL));

    DeviceZoneScopedN("RL-ALL");
    conn.open_start();
    PacketHeaderPool::reset();
    Header hd = PacketHeaderPool::allocate_header();
    fabric_set_unicast_route(hd, peer, mesh_id);
    conn.open_finish();
    auto put = [&](uint32_t src, uint32_t n) {
        conn.wait_for_empty_write_slot();
        if (n != 0) {
            conn.send_payload_without_header_non_blocking_from_address(src, n);
        }
        conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(hd), sizeof(PACKET_HEADER_TYPE));
    };
    // group id: slice s (0 / 1), source mesh row m, tile row q -> s << 16 | m << 8 | q
    // this role's packets of group i (packet p on link (i + p) mod 2: role `role` takes those with (i + p + role)
    // even), the last one on the link fused with the top core's increment 1 << 4 m
    auto send_group = [&](uint32_t i, uint32_t id, uint32_t base) {
        const uint32_t s = id >> 16, m = (id >> 8) & 0xff, q = id & 0xff;
        const uint32_t src = stage + (i % NB) * gbytes;
        const uint32_t dst = base + (m * tr + q) * ks * xtile;
        const uint64_t sem = safe_get_noc_addr(top_x[s], top_y[s], sem_x, 0);
        for (uint32_t p = (i + role) & 1; p < gpk; p += 2) {
            const uint64_t d = safe_get_noc_addr(top_x[s], top_y[s], dst + p * pbytes, 0);
            noc_async_writes_flushed();  // (the header is rewritten)
            if (p + 2 >= gpk) {
                hd->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{d, sem, 1u << (4 * m), true}, pbytes);
            } else {
                hd->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{d}, pbytes);
            }
            put(src + p * pbytes, pbytes);
        }
    };

    if constexpr (role == 0) {
        {
            DeviceZoneScopedN("RL-READY");  // this chip's top cores in the program, then the neighbour's
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(RELAY_SEM_TOPS)), 8);
            noc_async_writes_flushed();
            hd->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(peer_x, peer_y, sem_r, 0), 1, false});
            put(0, 0);
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_r);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_set(ready, 0);
        }
        invalidate_l1_cache();
        const uint32_t base = *mail;
        volatile tt_l1_ptr uint32_t* notice[2] = {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(RELAY_SEM_N0)),
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(RELAY_SEM_N0 + 1))};
        uint32_t taken[2] = {0, 0};  // the rows whose groups are queued, per slice
        uint32_t queue[total];       // the groups in landing order
        uint32_t queued = 0, issued = 0, done = 0;
        DeviceZoneScopedN("RL-SEND");
        while (done < total) {
            // the rows landed since: their groups queued
            if (queued < total) {
                invalidate_l1_cache();
                for (uint32_t s = 0; s < 2; ++s) {
                    const uint32_t fresh = *notice[s] & ~taken[s];
                    for (uint32_t m = 0; m < chips; ++m) {
                        if ((fresh >> m) & 1u) {
                            for (uint32_t q = 0; q < tr; ++q) {
                                queue[queued++] = s << 16 | m << 8 | q;
                            }
                        }
                    }
                    taken[s] |= fresh;
                }
            }
            // reads ahead into free slots (a slot is free once both roles have sent its group)
            while (issued < queued && issued < done + NB) {
                if (issued >= NB) {
                    invalidate_l1_cache();
                    if (*sent < issued - NB + 1) {  // (role 1's packets of the slot's last group)
                        break;
                    }
                    noc_async_writes_flushed();  // (this role's packets of it: sent, issued - NB < done)
                }
                const uint32_t id = queue[issued], s = id >> 16, m = (id >> 8) & 0xff, q = id & 0xff;
                const uint32_t src = base + (m * tr + q) * ks * xtile;
                noc_async_read_set_trid(issued % NB + 1);
                noc_async_read(get_noc_addr(top_x[s], top_y[s], src), stage + (issued % NB) * gbytes, gbytes);
                ++issued;
            }
            if (done < issued) {
                noc_async_read_barrier_with_trid(done % NB + 1);
                list[done] = queue[done];
                *pub = done + 1;  // (role 1 polls it)
                send_group(done, queue[done], base);
                ++done;
            }
        }
        noc_async_read_set_trid(0);
        // role 1's last packets out (it counts after its writes have left), then the counts back to 0
        noc_async_writes_flushed();
        do {
            invalidate_l1_cache();
        } while (*sent < total);
        for (uint32_t id : {RELAY_SEM_N0, RELAY_SEM_N0 + 1, RELAY_SEM_TOPS}) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)) = 0;
        }
        *pub = 0;
        *sent = 0;
    } else {
        DeviceZoneScopedN("RL-SEND");
        for (uint32_t i = 0; i < total; ++i) {
            do {
                invalidate_l1_cache();
            } while (*pub < i + 1);
            invalidate_l1_cache();
            send_group(i, list[i], *mail);
            noc_async_writes_flushed();  // (the slot's bytes have left: role 0 may read into it again)
            *sent = i + 1;
        }
    }
    conn.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
