// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The column's way back inside the experts op: a return sender core (BRISC; one of four:
// link (0, 1) by way (up to mesh row c + 1, down to c - 1)), over the 2D fabric to its ring neighbour, one hop.  It
// sends the blocks of the 8 reducer cores on its link (rs_move.cpp: slot m, NoC x, y and chunk parity in runtime
// args) as they become ready there (the count in this core's semaphore flag_sem + m), ring step by ring step, each into
// the same reducer core of the neighbour: its cb_lo (up) or cb_hi (down) at the block's place in this stream, in packets of
// `maxp` bytes, the last one fused with an increment of that core's word of rs_sem (1 up, 1 << 16 down: rs_move.cpp).
// A stream (by parity and way; step, cb, block):
//   up, even:   (0, own, 0) (1, sum, 0) (2, sum, 1) (3, sum, 3)      up, odd:   (1, own, 1) (2, sum, 2) (3, sum, 4)
//   down, even: (1, own, 2) (2, sum, 2) (3, sum, 4)                  down, odd: (0, own, 0) (1, sum, 0) (2, sum, 1)
//                                                                               (3, sum, 3)
// A block is read from the reducer's L1 into the stage (two slots, transaction ids 1 and 2) and sent; the next one is
// read before this one is sent only when it is ready already (a send never waits for a later block: the ring's steps
// wait on the neighbours' sends).  The connection opens once the in-op gather's sender (h1_send.cpp) has closed its own
// on this link (this core's mailbox word, rs_mail).  The flags are 8 of this core's program semaphores (ids flag_sem ..
// + 7, the dispatcher's zeros at every launch: a reducer's count written before this kernel starts is kept; a clear
// here at the start could erase it), cleared again at the end with the mailbox word.  Every wait is bounded
// (WAIT_LIMIT polls): past it the sender gives up (the rest unsent, its connection still closed: a hang with it open
// wedges the ethernet cores), and the reducers waiting for those blocks give up in turn (rs_move.cpp): wrong values,
// an op that ends.
// BFP8 (GLM53_MOE_RS_BFP8, compile-time arg 10; value-changing, default off): the blocks go in BFP8 (tiles of `tile` =
// 1,088 bytes, in packets of `maxp` = 4,352: half of a tile row's 8 tiles, the factory's), the own blocks from the
// reducer's BFP8 copies (cb_own8: the long chain's at 0, the short one's at 1, rs_compute.cpp), the sums from cb_sum
// (Bfp8_b).
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

static_assert(std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>, "the way back: 2D fabric");

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;
using Conn = tt::tt_fabric::WorkerToFabricEdmSender;

void kernel_main() {
    constexpr uint32_t flag_sem = get_compile_time_arg_val(0);  // the flags' first semaphore id (8 of them)
    constexpr uint32_t cb_stage = get_compile_time_arg_val(1);
    constexpr uint32_t cb_own = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(3);
    constexpr uint32_t cb_lo = get_compile_time_arg_val(4);
    constexpr uint32_t cb_hi = get_compile_time_arg_val(5);
    constexpr uint32_t R = get_compile_time_arg_val(6);
    constexpr uint32_t maxp = get_compile_time_arg_val(7);  // bytes a packet
    constexpr uint32_t tile = get_compile_time_arg_val(8);  // a tile's bytes: 2,048 (bf16) or 1,088 (BFP8)
    constexpr uint32_t cb_own8 = get_compile_time_arg_val(9);
    constexpr bool bfp8 = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t REDUCERS = 8;                         // a link's
    constexpr uint32_t BB = R * 8 * tile;                    // a block's bytes
    static_assert(BB % maxp == 0, "a block: whole packets");
    constexpr uint32_t NPK = BB / maxp;
    constexpr uint32_t WAIT_LIMIT = 1u << 25;

    size_t a = 0;
    const uint32_t down = get_arg_val<uint32_t>(a++);
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t dst_chip = get_arg_val<uint32_t>(a++);
    const uint32_t rs_sem = get_arg_val<uint32_t>(a++);
    const uint32_t rs_mail = get_arg_val<uint32_t>(a++);
    const size_t red = a;  // the reducers: NoC x, y, parity
    a += 3 * REDUCERS;
    auto conn = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    // (transaction ids from 0, their counters clear: as router_mm's kernels)
    noc_async_read_set_trid(0, noc_index);
    noc_async_write_set_trid(0, noc_index);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
    const uint32_t flags = get_semaphore(flag_sem);  // (ids flag_sem + m at 16 m: L1_ALIGNMENT apart)
    static_assert(L1_ALIGNMENT == 16, "the flags 16 bytes apart");
    const uint32_t stage = get_write_ptr(cb_stage);
    const uint32_t own_base = get_write_ptr(bfp8 ? cb_own8 : cb_own), sum_base = get_write_ptr(cb_sum);
    const uint32_t land_base = get_write_ptr(down ? cb_hi : cb_lo);
    const uint32_t inc = down ? (1u << 16) : 1u;
    auto flag = [&](uint32_t m) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags + 16 * m); };

    // the stream's entries in order (ring step, then slot): reducer, source address, place in the stream
    uint32_t e_m[4 * REDUCERS], e_src[4 * REDUCERS], e_o[4 * REDUCERS];
    uint32_t n = 0;
    for (uint32_t step = 0; step < 4; ++step) {
        for (uint32_t m = 0; m < REDUCERS; ++m) {
            const bool even = get_arg_val<uint32_t>(red + 3 * m + 2) == 0;
            // the stream's (first step, own block, sums): the long chain's from step 0, the short one's from step 1
            const bool longc = even != (down != 0);
            const uint32_t first = longc ? 0 : 1;
            if (step < first) {
                continue;
            }
            const uint32_t o = step - first;
            uint32_t src;
            if (o == 0) {
                const uint32_t own = bfp8 ? (longc ? 0 : 1) : (longc ? 0 : (even ? 2 : 1));
                src = own_base + own * BB;
            } else {
                // long: sums 0, 1, 3 after its own block; short: sums 2, 4
                const uint32_t s = longc ? (o == 1 ? 0 : o == 2 ? 1 : 3) : (o == 1 ? 2 : 4);
                src = sum_base + s * BB;
            }
            e_m[n] = m;
            e_src[n] = src;
            e_o[n] = o;
            ++n;
        }
    }

    bool to = false;  // a wait gave up
    {
        DeviceZoneScopedN("RS-OPEN");  // the in-op gather's sender closed its connection on this link
        volatile tt_l1_ptr uint32_t* mail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_mail);
        for (uint32_t k = 0;; ++k) {
            invalidate_l1_cache();
            if (*mail != 0) {
                break;
            }
            if (k == WAIT_LIMIT) {
                to = true;
                break;
            }
        }
    }
    if (to) {  // (no connection: the gather's may still be open)
        for (uint32_t m = 0; m < REDUCERS; ++m) {
            *flag(m) = 0;
        }
        return;
    }
    conn.open_start();
    PacketHeaderPool::reset();
    Header h = PacketHeaderPool::allocate_header();
    fabric_set_unicast_route(h, dst_chip, mesh_id);
    conn.open_finish();

    auto ready = [&](uint32_t i) {
        invalidate_l1_cache();
        return *flag(e_m[i]) >= e_o[i] + 1;
    };
    auto wait_ready = [&](uint32_t i) {
        for (uint32_t k = 0; !ready(i); ++k) {
            if (k == WAIT_LIMIT) {
                to = true;
                return;
            }
        }
    };
    auto read = [&](uint32_t i) {
        const uint32_t m = e_m[i];
        noc_async_read_set_trid((i & 1) + 1);
        noc_async_read(
            get_noc_addr(get_arg_val<uint32_t>(red + 3 * m), get_arg_val<uint32_t>(red + 3 * m + 1), e_src[i]),
            stage + (i & 1) * BB, BB);
    };
    DeviceZoneScopedN("RS-SEND");
    bool issued = false;  // entry i's read issued
    for (uint32_t i = 0; i < n; ++i) {
        if (!issued) {
            wait_ready(i);
            if (to) {
                break;
            }
            read(i);
        }
        noc_async_read_barrier_with_trid((i & 1) + 1);
        issued = false;
        if (i + 1 < n && ready(i + 1)) {  // the next one now, if it is there (its slot's last send flushed first)
            noc_async_writes_flushed();
            read(i + 1);
            issued = true;
        }
        const uint32_t m = e_m[i];
        const uint32_t rx = get_arg_val<uint32_t>(red + 3 * m), ry = get_arg_val<uint32_t>(red + 3 * m + 1);
        const uint32_t src = stage + (i & 1) * BB;
        const uint32_t dst = land_base + e_o[i] * BB;
        for (uint32_t p = 0; p < NPK; ++p) {
            noc_async_writes_flushed();  // (the header is rewritten)
            const uint64_t d = safe_get_noc_addr(rx, ry, dst + p * maxp, 0);
            if (p + 1 == NPK) {
                h->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{d, safe_get_noc_addr(rx, ry, rs_sem, 0), inc,
                                                                         true},
                    maxp);
            } else {
                h->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{d}, maxp);
            }
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(src + p * maxp, maxp);
            conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(h), sizeof(PACKET_HEADER_TYPE));
        }
    }
    noc_async_read_set_trid(0);
    conn.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    for (uint32_t m = 0; m < REDUCERS; ++m) {
        *flag(m) = 0;
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_mail) = 0;
}
