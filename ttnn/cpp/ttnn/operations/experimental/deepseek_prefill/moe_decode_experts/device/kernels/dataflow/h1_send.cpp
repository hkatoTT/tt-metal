// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The in-op gather (h1_*): the experts op's sender core, which gathers the mesh column's routing and x for
// the op over the 2D fabric while the op starts.  This chip is mesh row r of the column's ring of 8 chips; each mesh
// row has t tokens (8, 16, 32 or 64), the column's rows r t .. r t + t - 1.
//   The routing: route_only wrote this row's rows into the column's landing w_all [8 t, 32] fp32 here (persistent
//   DRAM, interleaved, the same address on every chip); their piece (a tile from t = 32; faces 0-1 or 2-3 of a tile
//   at t = 16; two 8-row halves of faces at t = 8) goes to the same place on the column's other 7 chips, fused with an
//   increment, once written, of the global semaphore sem_r on the tile's builder core (route_build.cpp waits for it),
//   and the builder here counts it at once.
//   x: straight into the grid columns' top gate | up cores' L1 (from t = 32 their CB_XALL, their K slice of every
//   tile row of tokens, tile (b, i) at (b ks + i) 2048; below, compact, a landing CB the top core then reads into
//   CB_XALL's tiles; the same address on every chip, which the top cores put in this core's mailbox semaphore when
//   they enter the program), so not before every chip of the column is in the op: this core waits for its own 8 top
//   cores, sends a ready increment to the other 7 chips' sender cores (sem_r's word here) and waits for theirs.  A
//   K slice group (a tile row of tokens' ks tiles, contiguous) goes in packets of up to `maxp` bytes, alternating
//   between the links, the last one on a link fused with an increment of the top core's sem_x (the flush before it
//   covers the link's earlier packets of that way: an increment and flush per packet held each receiving router up).
//   The receivers wait for the counts the program factory works out and set their semaphore back to 0; the next call's
//   packets come only after this call's way back, which needs every chip's experts op done.
// Every packet a multicast of 4 hops one way and one of 3 the other, the counts swapping from group to group (a
// receiver 4 hops away gets a group's packets all on one way, before its increment's flush there).
//   role 0 (BRISC, NoC 0): the routing (link 0), the handshake, x read from x_row into the stage a group at a time,
//          a group's even packets on link 0;
//   role 1 (NCRISC, NoC 1, no DRAM reads: the op's multicasts are on NoC 1): a group's odd packets on link 1.
// This row's own x the top cores read from x_row themselves (stream_active.cpp), with their count for it.
// Each role holds a connection to each ring neighbour's router on its link.
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

static_assert(std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "in-op gather: 2D fabric");

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
using Conn = tt::tt_fabric::WorkerToFabricEdmSender;

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t t = get_compile_time_arg_val(1);   // tokens a mesh row
    constexpr uint32_t kd = get_compile_time_arg_val(2);  // x tiles a tile row
    constexpr uint32_t ks = get_compile_time_arg_val(3);  // a top core's K slice (tiles)
    constexpr uint32_t T = kd / ks;                       // top cores (grid columns)
    constexpr uint32_t cb_stage = get_compile_time_arg_val(4);
    constexpr uint32_t sem_ready = get_compile_time_arg_val(5);  // role 0 -> role 1: the groups read (a count)
    constexpr uint32_t sem_tops = get_compile_time_arg_val(6);   // the top cores in the program (a count)
    constexpr uint32_t sem_mail = get_compile_time_arg_val(7);   // their CB_XALL address
    constexpr uint32_t maxp = get_compile_time_arg_val(8);       // bytes a packet at most (4 or 8 KB)
    constexpr auto xr_args = TensorAccessorArgs<9>();
    constexpr auto wa_args = TensorAccessorArgs<xr_args.next_compile_time_args_offset()>();
    // the column's way back in the op (rs_send.cpp): once this role's connections are closed, an increment of the
    // mailbox word of the two return senders on its link (runtime args after the connections': their NoC x, y, then
    // the mailbox's address), which open theirs then
    constexpr bool rs = get_compile_time_arg_val(wa_args.next_compile_time_args_offset()) != 0;
    constexpr uint32_t chips = 8;
    constexpr uint32_t tr = t >= 32 ? t / 32 : 1;           // this row's tile rows of x (and routing tiles)
    constexpr uint32_t pb = t >= 32 ? 2048 : t * 64;        // a tile's bytes of this row's x in the stage
    // below 32 tokens a row x goes compact: a K slice's tiles' rows of this row (pb bytes each) land together in the
    // top core's landing CB (this row's place: r ks pb), which the top core then puts into its CB_XALL's tiles
    constexpr bool compact = t < 32;
    constexpr uint32_t groups = tr * T;                      // (tile row q, K slice j), q-major
    constexpr uint32_t gbytes = ks * pb;                     // a group's bytes (contiguous in the stage and there)
    constexpr uint32_t pbytes = gbytes < maxp ? gbytes : maxp;
    constexpr uint32_t gpk = gbytes / pbytes;                // packets a group
    constexpr uint32_t xtile = 2048, wtile = 4096;
    static_assert(kd % ks == 0 && gbytes % pbytes == 0 && (gpk == 1 || gpk % 2 == 0), "gather: the K slices' packets");

    size_t a = 0;
    const uint32_t xr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t wa_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r = get_arg_val<uint32_t>(a++);  // this chip's mesh row
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t sem_x = get_arg_val<uint32_t>(a++);
    const uint32_t sem_r = get_arg_val<uint32_t>(a++);
    const uint32_t me_x = get_arg_val<uint32_t>(a++);  // this core's NoC x, y (the other sender cores')
    const uint32_t me_y = get_arg_val<uint32_t>(a++);
    const size_t tops = a;  // the top cores' NoC x, y (grid column j)
    a += 2 * T;
    const size_t builders = a;  // the builder cores' NoC x, y (tile row b), 16 of them
    a += 2 * 16;
    auto fwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);
    auto bwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    const auto s_xr = TensorAccessor(xr_args, xr_addr);
    const auto s_wa = TensorAccessor(wa_args, wa_addr);
    const uint32_t stage = get_write_ptr(cb_stage);
    const uint32_t wstage = stage;              // the routing's pieces (2 tiles at most)
    const uint32_t xstage = stage + 2 * wtile;  // x: tile (q, k) at (q kd + k) pb
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready));
    volatile tt_l1_ptr uint32_t* mail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_mail));

    DeviceZoneScopedN("H1-SEND");
    fwd.open_start();
    bwd.open_start();
    PacketHeaderPool::reset();
    // h0: 4 hops forward, h1: 3 backward; h2: 3 forward, h3: 4 backward
    Header h[4];
    {
        const auto dir_f = get_next_hop_router_direction(mesh_id, fwd_chip);
        const auto dir_b = get_next_hop_router_direction(mesh_id, bwd_chip);
        for (uint32_t i = 0; i < 4; ++i) {
            h[i] = PacketHeaderPool::allocate_header();
            uint16_t hops[4] = {0, 0, 0, 0};
            const bool forward = (i & 1) == 0;
            hops[forward ? dir_f : dir_b] = (i == 0 || i == 3) ? chips / 2 : chips - 1 - chips / 2;
            fabric_set_mcast_route(h[i], forward ? fwd_chip : bwd_chip, mesh_id, hops[0], hops[1], hops[2], hops[3]);
        }
    }
    fwd.open_finish();
    bwd.open_finish();

    auto put = [&](Conn& conn, Header hd, uint32_t src, uint32_t n) {
        conn.wait_for_empty_write_slot();
        if (n != 0) {
            conn.send_payload_without_header_non_blocking_from_address(src, n);
        }
        conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(hd), sizeof(PACKET_HEADER_TYPE));
    };
    // one packet of n bytes at src to both ways: chunks (d0, n0) and (d1, n - n0), or one write if n0 == n; `swap`: 3
    // hops forward and 4 backward (else 4 and 3); `inc`: an increment of sem by `val` once its writes have landed
    auto send = [&](uint32_t src, uint64_t d0, uint32_t n0, uint64_t d1, uint32_t n, bool swap, bool inc, uint64_t sem,
                    uint32_t val = 1) {
        noc_async_writes_flushed();  // the headers are rewritten
        Header hf = h[swap ? 2 : 0], hb = h[swap ? 3 : 1];
        for (Header hd : {hf, hb}) {
            if (inc && n0 == n) {
                hd->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{d0, sem, val, true}, n);
            } else if (inc) {
                hd->to_noc_fused_unicast_scatter_write_atomic_inc(
                    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader(
                        {d0, d1}, sem, {static_cast<uint16_t>(n0)}, 1, true),
                    n);
            } else if (n0 == n) {
                hd->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{d0}, n);
            } else {
                hd->to_noc_unicast_scatter_write(
                    tt::tt_fabric::NocUnicastScatterCommandHeader({d0, d1}, {static_cast<uint16_t>(n0)}), n);
            }
        }
        put(fwd, hf, src, n);
        put(bwd, hb, src, n);
    };
    // group (q, j)'s byte o of this row: its place on top core j (`base`: its CB_XALL, the column's tile row r tr + q
    // at (r tr + q) ks tiles; compact, its landing CB, this row's at r gbytes)
    auto xdst = [&](uint32_t base, uint32_t q, uint32_t j, uint32_t o, uint8_t noc) {
        const uint32_t at = compact ? r * gbytes : (r * tr + q) * ks * xtile;
        return safe_get_noc_addr(get_arg_val<uint32_t>(tops + 2 * j), get_arg_val<uint32_t>(tops + 2 * j + 1),
                                 base + at + o, noc);
    };
    auto top_sem = [&](uint32_t j, uint8_t noc) {
        return safe_get_noc_addr(get_arg_val<uint32_t>(tops + 2 * j), get_arg_val<uint32_t>(tops + 2 * j + 1), sem_x, noc);
    };
    // group g's packets on this role's link (packet p on link (g + p) mod 2), the last one with the top core's
    // increment; the hop counts swap from group to group (from one of the link's groups to its next with one packet a
    // group) and between the links.  From 32 tokens a row the increments go into this mesh row's 4-bit field of sem_x
    // (1 << 4 r: at most 2 tile rows x 2 links = 4 a row), so that the top core can pass each row's x down its grid
    // column as soon as that row has landed (gu_ctrl.cpp)
    const uint32_t xval = t >= 32 ? 1u << (4 * r) : 1u;
    auto send_group = [&](uint32_t base, uint32_t g) {
        const uint32_t q = g / T, j = g % T;
        const uint64_t sem = top_sem(j, 0);
        const bool swap = (((gpk == 1 ? g / 2 : g) + role) & 1) != 0;
        const uint32_t src = xstage + (q * kd + j * ks) * pb;
        for (uint32_t p = (g + role) & 1; p < gpk; p += 2) {
            send(src + p * pbytes, xdst(base, q, j, p * pbytes, 0), pbytes, 0, pbytes, swap, p + 2 >= gpk, sem, xval);
        }
    };

    if constexpr (role == 0) {
        // the routing: this row's pieces of w_all (already in place here), each to the other chips, the builder of
        // its tile counting it; and here, its builder counted at once
        constexpr uint32_t pieces = tr;  // t < 32: one piece (one or two chunks)
        uint64_t dst[pieces][2];
        uint32_t tile_of[pieces];
        for (uint32_t q = 0; q < pieces; ++q) {
            uint32_t tile, o0, o1;
            if constexpr (t >= 32) {
                tile = r * tr + q, o0 = 0, o1 = 0;
            } else if constexpr (t == 16) {
                tile = r / 2, o0 = 2048 * (r % 2), o1 = o0;
            } else {
                tile = r / 4, o0 = 2048 * ((r % 4) / 2) + 512 * (r % 2), o1 = o0 + 1024;
            }
            tile_of[q] = tile;
            dst[q][0] = s_wa.get_noc_addr(tile, o0, 0);
            dst[q][1] = s_wa.get_noc_addr(tile, o1, 0);
            const uint32_t l1 = wstage + q * wtile;
            if constexpr (t >= 32) {
                noc_async_read(s_wa.get_noc_addr(tile, 0), l1, wtile);
            } else if constexpr (t == 16) {
                noc_async_read(s_wa.get_noc_addr(tile, o0), l1, 2048);
            } else {
                noc_async_read(s_wa.get_noc_addr(tile, o0), l1, 512);
                noc_async_read(s_wa.get_noc_addr(tile, o1), l1 + 512, 512);
            }
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(builders + 2 * tile), get_arg_val<uint32_t>(builders + 2 * tile + 1),
                             sem_r),
                1);
        }
        noc_async_read_barrier();
        for (uint32_t q = 0; q < pieces; ++q) {
            const uint32_t b = tile_of[q];
            const uint64_t sem = safe_get_noc_addr(
                get_arg_val<uint32_t>(builders + 2 * b), get_arg_val<uint32_t>(builders + 2 * b + 1), sem_r, 0);
            if constexpr (t >= 32) {
                send(wstage + q * wtile, dst[q][0], wtile, 0, wtile, q & 1, true, sem);
            } else if constexpr (t == 16) {
                send(wstage + q * wtile, dst[q][0], 2048, 0, 2048, false, true, sem);
            } else {
                send(wstage + q * wtile, dst[q][0], 512, dst[q][1], 1024, false, true, sem);
            }
        }
        // the handshake (after the routing, which lands in DRAM): this chip's top cores in the program (their landing
        // address in the mailbox), then a ready increment to the other chips' sender cores (theirs waited for before
        // x goes, below)
        {
            DeviceZoneScopedN("H1-READY");
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_tops)), T);
            const uint64_t peer_sem = safe_get_noc_addr(me_x, me_y, sem_r, 0);
            noc_async_writes_flushed();  // (the headers are rewritten)
            for (Header hd : {h[0], h[1]}) {
                hd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_sem, 1, false});
            }
            put(fwd, h[0], 0, 0);
            put(bwd, h[1], 0, 0);
        }
        // x: a group's tiles read with transaction id (g mod W) + 1, W groups ahead (the first ones while the other
        // chips' ready increments come)
        DeviceZoneScopedN("H1-X");
        constexpr uint32_t W = groups < 8 ? groups : 8;
        auto read_group = [&](uint32_t g) {
            const uint32_t q = g / T, j = g % T;
            noc_async_read_set_trid(g % W + 1);
            for (uint32_t i = 0; i < ks; ++i) {
                const uint32_t k = j * ks + i;
                const uint32_t l1 = xstage + (q * kd + k) * pb;
                if constexpr (t >= 32) {
                    noc_async_read(s_xr.get_noc_addr(q * kd + k), l1, xtile);
                } else if constexpr (t == 16) {
                    noc_async_read(s_xr.get_noc_addr(k), l1, 1024);  // rows 0-15: faces 0 and 1
                } else {
                    noc_async_read(s_xr.get_noc_addr(k), l1, 256);              // rows 0-7 of face 0
                    noc_async_read(s_xr.get_noc_addr(k, 512), l1 + 256, 256);  // and of face 1
                }
            }
        };
        for (uint32_t g = 0; g < W; ++g) {
            read_group(g);
        }
        {
            DeviceZoneScopedN("H1-PEERS");  // the other chips' top cores in the program
            volatile tt_l1_ptr uint32_t* peers = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_r);
            noc_semaphore_wait_min(peers, chips - 1);
            noc_semaphore_set(peers, 0);
        }
        const uint32_t base = *mail;
        for (uint32_t g = 0; g < groups; ++g) {
            noc_async_read_barrier_with_trid(g % W + 1);
            *ready = g + 1;  // (role 1 polls it; the handshake is done)
            if (g + W < groups) {
                read_group(g + W);
            }
            send_group(base, g);
        }
        noc_async_read_set_trid(0);
    } else {
        for (uint32_t g = 0; g < groups; ++g) {
            do {
                invalidate_l1_cache();
            } while (*ready < g + 1);
            send_group(*mail, g);
        }
    }
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    if constexpr (rs) {
        const uint32_t mail = get_arg_val<uint32_t>(a + 4);
        for (uint32_t i = 0; i < 2; ++i) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(a + 2 * i), get_arg_val<uint32_t>(a + 2 * i + 1), mail), 1);
        }
        noc_async_atomic_barrier();
    }
}
