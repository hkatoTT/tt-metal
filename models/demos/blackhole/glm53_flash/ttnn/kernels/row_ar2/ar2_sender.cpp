// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two-stage row all-reduce (row_ar2.py), a link's sender core, RISCV_0: a connection to each ring neighbour's router on
// this link.
//   1. the reduce-scatter: each share the reader has put in cb_in, in order, to its owner chip, into this chip's slot
//      of the owner's landing buffer at that core, a write fused with an increment of that core's global semaphore
//      (kind 0: the forward neighbour, 1: the backward one, 2 / 3: the opposite chip two hops through the forward /
//      backward neighbour, which passes it on without delivering);
//   2. the all-gather: each of this chip's owned cores' sums as its bit of the stage's semaphore says it is in the
//      stage, to the same core's output shard on the three other chips (split 0: a 2-hop multicast forward and the
//      backward neighbour, 1: the forward neighbour and a 2-hop multicast backward), fused with the same increment.
// The routes are the 2D fabric's action maps written here (a receiving router reads the X byte of its own column on
// an E / W hop: deliver, pass on, or both), in place of the route table's encode (fabric_set_mcast_route ~0.5 us a
// header); AR2_ROUTE_CHECK: the multicasts' and the forward neighbour's bytes compared
// with the encodes' (DPRINT).
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#ifdef AR2_ROUTE_CHECK
#include "api/debug/dprint.h"
#endif

static_assert(
    std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "row all-reduce: 2D fabric expected");

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
using tt::tt_fabric::Routing2DCodec;

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(1);
    constexpr uint32_t sem_ag = get_compile_time_arg_val(2);
    constexpr uint32_t shard = get_compile_time_arg_val(3);  // bytes a core's sums (bf16): an all-gather packet
    constexpr uint32_t n_rs = get_compile_time_arg_val(4);
    constexpr uint32_t n_own = get_compile_time_arg_val(5);
    constexpr uint32_t shard_in = get_compile_time_arg_val(6);  // bytes a core's input share: a reduce-scatter packet
    static_assert(n_own <= 16, "a bit a core");

    size_t a = 0;
    const uint32_t land_addr = get_arg_val<uint32_t>(a++);  // the landing buffer's shard (this parity)
    const uint32_t sem_addr = get_arg_val<uint32_t>(a++);   // the global semaphore (this parity)
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);   // the output's shard
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t opp_chip = get_arg_val<uint32_t>(a++);
    const uint32_t slot_off = get_arg_val<uint32_t>(a++);  // this chip's slot in a landing shard
    const size_t rs_at = a;                                // n_rs words: NoC x | y << 8 | kind << 16
    a += n_rs;
    const size_t own_at = a;  // n_own words: NoC x | y << 8 | split << 16
    a += n_own;
    auto fwd = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    auto bwd = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);

    DeviceZoneScopedN("AR2-W");
    fwd.open_start();
    bwd.open_start();
    PacketHeaderPool::reset();
    Header h_uf = PacketHeaderPool::allocate_header();  // the forward neighbour
    Header h_ub = PacketHeaderPool::allocate_header();  // the backward neighbour
    Header h_of = PacketHeaderPool::allocate_header();  // the opposite chip through the forward neighbour
    Header h_ob = PacketHeaderPool::allocate_header();  // .. through the backward one
    Header h_mf = PacketHeaderPool::allocate_header();  // 2-hop multicast forward
    Header h_mb = PacketHeaderPool::allocate_header();  // 2-hop multicast backward
    {
        auto* table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
        const uint32_t ny = table->mesh_y_size, nx = table->mesh_x_size;
        const auto dir_f = tt::tt_fabric::get_next_hop_router_direction(mesh_id, fwd_chip);
        const auto dir_b = tt::tt_fabric::get_next_hop_router_direction(mesh_id, bwd_chip);
        const uint8_t act_f = Routing2DCodec::action_bit(dir_f), act_b = Routing2DCodec::action_bit(dir_b);
        const uint32_t xf = fwd_chip % nx, xb = bwd_chip % nx, xo = opp_chip % nx;
        constexpr uint8_t local = Routing2DCodec::ACTION_LOCAL_DELIVER;
        // dst: the header's destination (metadata), hops: its multicast extents; the X bytes at the neighbour and at
        // the opposite chip (0: not reached)
        auto route = [&](Header h, uint32_t dst, uint64_t hops, uint32_t x_mid, uint8_t a_mid, uint32_t x_end) {
            h->dst_start_node_id = (mesh_id << 16) | dst;
            h->mcast_params_64 = hops;
            h->routing_fields.value = 0;
            for (uint32_t i = 0; i < ny + nx; ++i) {
                h->route_buffer[i] = 0;
            }
            h->route_buffer[ny + x_mid] = a_mid;
            if (x_end != x_mid) {
                h->route_buffer[ny + x_end] = local;
            }
        };
        const uint64_t two_f = uint64_t{2} << (16 * dir_f), two_b = uint64_t{2} << (16 * dir_b);
        route(h_uf, fwd_chip, 0, xf, local, xf);
        route(h_ub, bwd_chip, 0, xb, local, xb);
        route(h_of, opp_chip, 0, xf, act_f, xo);
        route(h_ob, opp_chip, 0, xb, act_b, xo);
        route(h_mf, fwd_chip, two_f, xf, act_f | local, xo);
        route(h_mb, bwd_chip, two_b, xb, act_b | local, xo);
#ifdef AR2_ROUTE_CHECK
        {
            Header t = PacketHeaderPool::allocate_header();
            auto differ = [&](Header ours, uint32_t x0, uint32_t x1) {
                return (uint32_t)(t->route_buffer[ny + x0] != ours->route_buffer[ny + x0]) +
                       (uint32_t)(t->route_buffer[ny + x1] != ours->route_buffer[ny + x1]);
            };
            uint16_t hf[4] = {0, 0, 0, 0}, hb[4] = {0, 0, 0, 0};
            hf[dir_f] = 2;
            hb[dir_b] = 2;
            fabric_set_mcast_route(t, fwd_chip, mesh_id, hf[0], hf[1], hf[2], hf[3]);
            const uint32_t d_mf = differ(h_mf, xf, xo);
            fabric_set_mcast_route(t, bwd_chip, mesh_id, hb[0], hb[1], hb[2], hb[3]);
            const uint32_t d_mb = differ(h_mb, xb, xo);
            fabric_set_unicast_route(t, fwd_chip, mesh_id);
            const uint32_t d_uf = (uint32_t)(t->route_buffer[ny + xf] != local);
            fabric_set_unicast_route(t, opp_chip, mesh_id);
            const bool opp_f = tt::tt_fabric::get_next_hop_router_direction(mesh_id, opp_chip) == dir_f;
            const uint32_t d_o = differ(opp_f ? h_of : h_ob, opp_f ? xf : xb, xo);
            DPRINT(
                "AR2-ROUTE-CHECK mesh {} f {} b {} o {} x {} {} {} dirs {} {}: differ mf {} mb {} uf {} o({}) {}\n",
                mesh_id,
                fwd_chip,
                bwd_chip,
                opp_chip,
                xf,
                xb,
                xo,
                (uint32_t)dir_f,
                (uint32_t)dir_b,
                d_mf,
                d_mb,
                d_uf,
                (uint32_t)opp_f,
                d_o);
        }
#endif
    }
    fwd.open_finish();
    bwd.open_finish();

#ifndef AR2_DRY  // (AR2_DRY: a test's, the routes only: nothing sent, the receivers do not wait)
    auto send = [&](tt::tt_fabric::WorkerToFabricEdmSender& conn, Header h, uint32_t src, uint64_t dst, uint64_t sem,
                    uint32_t bytes) {
        h->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, false}, bytes);
        conn.wait_for_empty_write_slot();
        conn.send_payload_without_header_non_blocking_from_address(src, bytes);
        conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(h), sizeof(PACKET_HEADER_TYPE));
    };

    // 1. the reduce-scatter
    {
        DeviceZoneScopedN("AR2-W-RS");
        const uint32_t in = get_read_ptr(cb_in);
        for (uint32_t k = 0; k < n_rs; ++k) {
            cb_wait_front(cb_in, k + 1);
            const uint32_t w = get_arg_val<uint32_t>(rs_at + k);
            const uint32_t x = w & 0xff, y = (w >> 8) & 0xff, kind = w >> 16;
            const uint64_t dst = safe_get_noc_addr(x, y, land_addr + slot_off, 0);
            const uint64_t sem = safe_get_noc_addr(x, y, sem_addr, 0);
            // (an item's packet header is rewritten: its last send has left.  Flushing only when the item's header was
            // sent since the last flush gained nothing)
            noc_async_writes_flushed();
            switch (kind) {
                case 0: send(fwd, h_uf, in + k * shard_in, dst, sem, shard_in); break;
                case 1: send(bwd, h_ub, in + k * shard_in, dst, sem, shard_in); break;
                case 2: send(fwd, h_of, in + k * shard_in, dst, sem, shard_in); break;
                default: send(bwd, h_ob, in + k * shard_in, dst, sem, shard_in); break;
            }
        }
    }

    // 2. the all-gather, each owned core's sums as they come
    {
        DeviceZoneScopedN("AR2-W-AG");
        auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ag));
        const uint32_t stage = get_write_ptr(cb_stage);
        uint32_t pending = (1u << n_own) - 1;
        while (pending != 0) {
            invalidate_l1_cache();
            uint32_t got = *ready & pending;
            if (got == 0) {
                continue;
            }
            pending &= ~got;
            for (; got != 0; got &= got - 1) {
                const uint32_t m = __builtin_ctz(got);
                const uint32_t w = get_arg_val<uint32_t>(own_at + m);
                const uint32_t x = w & 0xff, y = (w >> 8) & 0xff, split = w >> 16;
                const uint64_t dst = safe_get_noc_addr(x, y, out_addr, 0);
                const uint64_t sem = safe_get_noc_addr(x, y, sem_addr, 0);
                noc_async_writes_flushed();
                if (split == 0) {
                    send(fwd, h_mf, stage + m * shard, dst, sem, shard);
                    send(bwd, h_ub, stage + m * shard, dst, sem, shard);
                } else {
                    send(fwd, h_uf, stage + m * shard, dst, sem, shard);
                    send(bwd, h_mb, stage + m * shard, dst, sem, shard);
                }
            }
        }
    }
#endif
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
}
