// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

#ifndef AHEAD
#define AHEAD 3  // blocks in flight (the CB holds AHEAD + 1)
#endif

// The router matmul's weights and output (ttnn/router_mm.py; NoC 1, the x multicast's NoC 0 left alone): this core's column tile of w (K tiles, page
// k nt + col of the [K, N] tiled tensor) into CB_W a block of kb tiles at a time (the CB holds `depth` blocks, the
// reads of the next blocks in flight while the compute takes one), then the output tile (tile row `row`, column `col`)
// to DRAM.
//
// SHARE (2 tile rows of x, GLM53_ROUTER_WSHARE): the column's two cores (tile rows 0 and 1, vertical neighbours)
// read each weight block from DRAM once between them: block b by the core of row b % 2, which
// pushes it to its own compute and writes it on into the same slot of the other's CB_W ring (the rings alike, from
// their start); the other tells the reader that the slot is free (its count of such blocks into the reader's SEM_RDY)
// and waits for the block (the reader's count of blocks written on, into its SEM_ARR, after the data's write barrier).
// Counts by plain 4-byte writes of absolute values (one writer each, no atomic), each from its own 16-byte word in
// CB_WS; each core sets its two back to 0 at its end (for the next call).  The compute and the values are those
// without SHARE (each core still sums its tile row's whole K in the same order): only who reads which block from DRAM
// changes (4.7 -> 2.36 MB a call).
//
// router_route (the routing in the same program, `notify` readers): the readers are told once every router core's
// tile is in DRAM, by 4-byte writes only (no atomic: an atomic's response may come after a kernel's end, into the next
// program's count on that NoC).  Router core 0 (the collector) clears its flag slots
// (CB_FLAG, 16 bytes a router core) and then sets each other router core's SEM_GO to 1; a router core, its tile
// written, waits for SEM_GO and writes 1 into its slot at the collector (the same address there: the router cores'
// CBs alike); the collector, its own tile written and every slot 1, sets each reader's SEM_LOGITS to 1.  Every wait is
// bounded (past it the core goes on: wrong values, no hang).
void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t col = get_arg_val<uint32_t>(2);
    const uint32_t row = get_arg_val<uint32_t>(3);
    // router_route: runtime args 4 .. : the readers' count, this core's index k (0: the collector), the router cores'
    // count, then the router cores' NoC x, y (k order), then the readers' NoC x, y
    const uint32_t notify = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t kb = get_compile_time_arg_val(2);  // tiles a block
    constexpr uint32_t nb = get_compile_time_arg_val(3);  // blocks
    constexpr uint32_t nt = get_compile_time_arg_val(4);  // w's (and the output's) column tiles
    constexpr uint32_t sem_logits = get_compile_time_arg_val(5);
    constexpr uint32_t cb_flag = get_compile_time_arg_val(6);
    constexpr uint32_t sem_go = get_compile_time_arg_val(7);
    constexpr uint32_t WAIT_LIMIT = 1u << 25;
#ifdef SHARE
    constexpr uint32_t cb_ws = SHARE_CB;      // the counts' source words (16 bytes each: nb for SEM_RDY, nb for SEM_ARR)
    constexpr uint32_t sem_rdy = SHARE_RDY;   // the other core's slots freed for this core's blocks (its writes)
    constexpr uint32_t sem_arr = SHARE_ARR;   // the other core's blocks written into this core's ring (its writes)
#endif
    constexpr auto w_args = TensorAccessorArgs<8>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const auto s_w = TensorAccessor(w_args, w_addr);
    const auto s_out = TensorAccessor(out_args, out_addr);
    const uint32_t wt = get_local_cb_interface(cb_w).fifo_page_size;
    noc_async_read_set_trid(0, noc_index);  // (transaction ids from a clean state, as rm_x.cpp)
    noc_async_write_set_trid(0, noc_index);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
    const uint32_t k = notify ? get_arg_val<uint32_t>(5) : 0;
    const uint32_t count = notify ? get_arg_val<uint32_t>(6) : 0;
    constexpr uint32_t RA = 7;  // the router cores' NoC x, y; then the readers'
    const uint32_t flags = notify ? get_write_ptr(cb_flag) : 0;
    auto slot = [&](uint32_t i) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags + 16 * i); };
    if (notify && k == 0) {  // the collector: its slots cleared, then the others' go (slot 0: the word 1 they write)
        for (uint32_t i = 0; i < count; ++i) {
            *slot(i) = 0;
        }
        *slot(0) = 1;
        const uint32_t go = get_semaphore(sem_go);
        for (uint32_t i = 1; i < count; ++i) {
            noc_semaphore_set_remote(
                flags, get_noc_addr(get_arg_val<uint32_t>(RA + 2 * i), get_arg_val<uint32_t>(RA + 2 * i + 1), go));
        }
    }
    {
        DeviceZoneScopedN("RM-W");
        // `ahead` blocks in flight (transaction ids 1 .. depth, as decode_matmul's weight stream), the CB a ring of
        // depth blocks from its start (nothing pushed before in the program)
        constexpr uint32_t depth = AHEAD + 1;
        constexpr uint32_t ahead = AHEAD;
        const uint32_t base = get_write_ptr(cb_w);
        uint32_t pushed = 0;
#ifdef SHARE
        // the other core of the column: runtime args after the notify's (or at 5)
        const uint32_t pa = notify ? 7 + 2 * count + 2 * notify : 5;
        const uint32_t px = get_arg_val<uint32_t>(pa), py = get_arg_val<uint32_t>(pa + 1);
        const uint32_t words = get_write_ptr(cb_ws);
        volatile tt_l1_ptr uint32_t* rdy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rdy));
        volatile tt_l1_ptr uint32_t* arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_arr));
        const uint64_t peer_rdy = get_noc_addr(px, py, get_semaphore(sem_rdy));
        const uint64_t peer_arr = get_noc_addr(px, py, get_semaphore(sem_arr));
        auto tell = [&](uint32_t word, uint32_t value, uint64_t dst) {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16 * word);
            *w = value;
            noc_semaphore_set_remote(words + 16 * word, dst);
        };
        // (unbounded, as rm_x's chunk waits)
        auto wait_min = [&](volatile tt_l1_ptr uint32_t* sem, uint32_t value) { noc_semaphore_wait_min(sem, value); };
        uint32_t freed = 0, sent = 0, got = 0;  // slots told free, own blocks written on, the other's blocks here
        uint32_t issued = 0;
        while (pushed < nb) {
            for (; issued < nb && issued - pushed < ahead; ++issued) {
                cb_reserve_back(cb_w, (issued - pushed + 1) * kb);
                if ((issued & 1u) == row) {
                    const uint32_t dst = base + (issued % depth) * kb * wt;
                    noc_async_read_set_trid(issued % depth + 1);
                    for (uint32_t i = 0; i < kb; ++i) {
                        noc_async_read(s_w.get_noc_addr((issued * kb + i) * nt + col), dst + i * wt, wt);
                    }
                } else {
                    ++freed;
                    tell(freed - 1, freed, peer_rdy);
                }
            }
            const uint32_t b = pushed;
            if ((b & 1u) == row) {
                noc_async_read_barrier_with_trid(b % depth + 1);
                cb_push_back(cb_w, kb);
                ++sent;
                wait_min(rdy, sent);
                const uint32_t src = base + (b % depth) * kb * wt;
                noc_async_write(src, get_noc_addr(px, py, src), kb * wt);
                noc_async_write_barrier();
                tell(nb + sent - 1, sent, peer_arr);
            } else {
                ++got;
                wait_min(arr, got);
                cb_push_back(cb_w, kb);
            }
            ++pushed;
        }
        // (the next call's counts from 0, as SEM_GO / SEM_X: every write of the other into these came before the
        // waits above ended)
        noc_semaphore_set(rdy, 0);
        noc_semaphore_set(arr, 0);
        noc_async_write_barrier();
        noc_async_read_set_trid(0);
#else
        for (uint32_t issued = 0; issued < nb; ++issued) {
            while (issued - pushed >= ahead) {
                noc_async_read_barrier_with_trid(pushed % depth + 1);
                cb_push_back(cb_w, kb);
                ++pushed;
            }
            cb_reserve_back(cb_w, (issued - pushed + 1) * kb);
            const uint32_t dst = base + (issued % depth) * kb * wt;
            noc_async_read_set_trid(issued % depth + 1);
            for (uint32_t i = 0; i < kb; ++i) {
                noc_async_read(s_w.get_noc_addr((issued * kb + i) * nt + col), dst + i * wt, wt);
            }
        }
        for (; pushed < nb; ++pushed) {
            noc_async_read_barrier_with_trid(pushed % depth + 1);
            cb_push_back(cb_w, kb);
        }
        noc_async_read_set_trid(0);
#endif
    }
    cb_wait_front(cb_out, 1);
    noc_async_write(get_read_ptr(cb_out), s_out.get_noc_addr(row * nt + col), get_local_cb_interface(cb_out).fifo_page_size);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
    if (notify && k != 0) {  // this core's slot at the collector, once the collector has cleared them
        volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go));
        for (uint32_t i = 0; i < WAIT_LIMIT; ++i) {
            invalidate_l1_cache();
            if (*go != 0) {
                break;
            }
        }
        *go = 0;
        *slot(k) = 1;
        noc_async_write(
            flags + 16 * k, get_noc_addr(get_arg_val<uint32_t>(RA), get_arg_val<uint32_t>(RA + 1), flags + 16 * k), 4);
        noc_async_write_barrier();
    } else if (notify) {  // the collector: every slot, then the readers
        for (uint32_t i = 1; i < count; ++i) {
            for (uint32_t t = 0; t < WAIT_LIMIT; ++t) {
                invalidate_l1_cache();
                if (*slot(i) != 0) {
                    break;
                }
            }
        }
        const uint32_t sem = get_semaphore(sem_logits);
        const uint32_t ra = RA + 2 * count;
        for (uint32_t i = 0; i < notify; ++i) {
            noc_semaphore_set_remote(
                flags, get_noc_addr(get_arg_val<uint32_t>(ra + 2 * i), get_arg_val<uint32_t>(ra + 2 * i + 1), sem));
        }
        noc_async_write_barrier();
    }
#ifdef RM_X8
    // the BFP8 x (RM_X8, rm_compute.cpp): this core's share of the tile row's x as BFP8_b into the experts op's
    // x_row (page row KT + k), after the logits are told (the routing goes on meanwhile); runtime arg RM_X8_ARG:
    // x_row's address
    {
        constexpr uint32_t KT = RM_X8_KT, NT = nt;
        constexpr auto x8_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
        const auto s_x8 = TensorAccessor(x8_args, get_arg_val<uint32_t>(RM_X8_ARG));
        const uint32_t x8t = get_local_cb_interface(RM_X8_CB).fifo_page_size;
        for (uint32_t kk = col; kk < KT; kk += NT) {
            cb_wait_front(RM_X8_CB, 1);
            noc_async_write(get_read_ptr(RM_X8_CB), s_x8.get_noc_addr(row * KT + kk), x8t);
            noc_async_writes_flushed();
            cb_pop_front(RM_X8_CB, 1);
        }
        noc_async_write_barrier();
    }
#endif
}
