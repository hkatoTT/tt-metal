// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// The control kernel (NoC 1) of core (slice s, block b):
// 1. x's K slice (tiles k0 .. k0 + kc - 1 of tile row 0) into CB_X, in chunks of x_chunk tiles: the rectangle's core
//    0 reads each chunk from DRAM (several in flight), multicasts it to the rest of the rectangle (the same address on
//    every core) and then sets their SEM_X to the chunks sent so far (a word per chunk in CB_SCR, never rewritten);
//    the others wait for SEM_X to reach each chunk.  Every chunk is pushed to the compute as it is there.  (Reading
//    each bank's tiles of the slice with one request and multicasting them a bank at a time, the compute and the
//    weights then in that bank-major K order, was slower: x_proj 70.4 -> 77.8 us, the weight rows no longer in order.)
// 2. With more than one slice: this core's fp32 partials (CB_PART, ncb tiles) to the block's owners: tile t goes to
//    the core of slice t mod ks (runtime args: the owners' NoC coordinates) at row s of its CB_RECV (slot s R + t / ks,
//    R its tiles), then bit s added to each owner's SEM_RED; an owner pushes the slices' rows to the compute in the
//    order of K, each as soon as its bit is set (the adds follow the partials as they come).
// 3. This core's output tiles (one slice: all ncb; else the owned tiles s, s + ks, ..; tile t is column n0 + t stride)
//    to DRAM.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t k0 = get_arg_val<uint32_t>(2);
    const uint32_t kc = get_arg_val<uint32_t>(3);
    const uint32_t n0 = get_arg_val<uint32_t>(4);
    const uint32_t ncb = get_arg_val<uint32_t>(5);
    const uint32_t s = get_arg_val<uint32_t>(6);
    const uint32_t ks = get_arg_val<uint32_t>(7);
    const uint32_t rect = get_arg_val<uint32_t>(8);
    const uint32_t sender = get_arg_val<uint32_t>(9);
    const uint32_t x_chunk = get_arg_val<uint32_t>(10);  // x's tiles a multicast
    const uint32_t owned = get_arg_val<uint32_t>(11);
    const uint32_t mc0_x = get_arg_val<uint32_t>(12);
    const uint32_t mc0_y = get_arg_val<uint32_t>(13);
    const uint32_t mc1_x = get_arg_val<uint32_t>(14);
    const uint32_t mc1_y = get_arg_val<uint32_t>(15);
    const uint32_t stride = get_arg_val<uint32_t>(16);
    constexpr uint32_t OWNERS = 17;  // the owners' (x, y) from here

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(4);
    constexpr uint32_t sem_x = get_compile_time_arg_val(5);
    constexpr uint32_t sem_red = get_compile_time_arg_val(6);
    // the weight stream on both NoCs (DECODE_MATMUL_NOCS=2): before this kernel's first NoC 1 transaction, its command
    // buffers' transaction ids to 0 and the NIU's outstanding counters of ids 1 .. 15 cleared (earlier programs'
    // multicasts and tagged writes can leave them non-zero; deepseek_v3_b1's dram_streaming_matmul resets them
    // likewise), as the weight stream waits on ids 1 .. depth of NoC 1 too; then, where this kernel reads nothing (not
    // on x's senders), NoC 1 handed over to the weight stream (sem_noc1)
    constexpr uint32_t sem_noc1 = get_compile_time_arg_val(7);
    constexpr bool both = get_compile_time_arg_val(8) != 0;
    // x's tile rows (2: K whole on the core, ks 1; x's slice k-major in CB_X, tile (r, k) at k mt + r), the K and N
    // tiles of x and the output (their row strides), and the compute's output sub-block columns (its pack order: per
    // sub-block, mt rows of sbw tiles)
    constexpr uint32_t mt = get_compile_time_arg_val(9);
    constexpr uint32_t kt_all = get_compile_time_arg_val(10);
    constexpr uint32_t nt_all = get_compile_time_arg_val(11);
    constexpr uint32_t sb = get_compile_time_arg_val(12);
    constexpr auto x_args = TensorAccessorArgs<13>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    if (ncb == 0) {
        return;  // an idle core of the rectangle: x lands here, unused
    }
    if constexpr (both) {
        noc_async_read_set_trid(0, noc_index);
        noc_async_write_set_trid(0, noc_index);
        ncrisc_noc_set_transaction_id(noc_index, write_reg_cmd_buf, 0);
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
    }

    // 1. x
    {
        DeviceZoneScopedN("X");
        CircularBuffer xcb(cb_x);
        const uint32_t xt = get_local_cb_interface(cb_x).fifo_page_size;
        xcb.reserve_back(mt * kc);
        const uint32_t xbase = xcb.get_write_ptr();
        const uint32_t chunks = (kc + x_chunk - 1) / x_chunk;
        if (sender) {
            const auto s_x = TensorAccessor(x_args, x_addr);
            const uint32_t words = get_local_cb_interface(cb_scr).fifo_wr_ptr;
            const uint32_t sem_addr = get_semaphore(sem_x);
            constexpr uint32_t AHEAD = 4;
            constexpr uint32_t X_TRID0 = 9;  // ids 9 .. 13 (the weight stream's: 1 .. 8, NoC 0 only here)
            auto finish = [&](uint32_t c) {
                noc_async_read_barrier_with_trid(X_TRID0 + c % (AHEAD + 1));
                const uint32_t first = c * x_chunk;
                const uint32_t n = kc - first < x_chunk ? kc - first : x_chunk;
                if (rect > 1) {
                    const uint32_t src = xbase + mt * first * xt;
                    noc_async_write_multicast(
                        src, get_noc_multicast_addr(mc0_x, mc0_y, mc1_x, mc1_y, src), mt * n * xt, rect - 1, true);
                    volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16 * c);
                    *word = c + 1;
                    noc_semaphore_set_multicast(
                        words + 16 * c, get_noc_multicast_addr(mc0_x, mc0_y, mc1_x, mc1_y, sem_addr), rect - 1);
                }
                xcb.push_back(mt * n);
            };
            uint32_t done = 0;
            for (uint32_t c = 0; c < chunks; ++c) {
                if (c - done == AHEAD) {
                    finish(done);
                    ++done;
                }
                noc_async_read_set_trid(X_TRID0 + c % (AHEAD + 1));
                const uint32_t first = c * x_chunk;
                const uint32_t last = kc - first < x_chunk ? kc : first + x_chunk;
                for (uint32_t i = first; i < last; ++i) {
                    if constexpr (mt == 1) {
                        noc_async_read(s_x.get_noc_addr(k0 + i), xbase + i * xt, xt);
                    } else {
                        for (uint32_t r = 0; r < mt; ++r) {
                            noc_async_read(s_x.get_noc_addr(r * kt_all + k0 + i), xbase + (i * mt + r) * xt, xt);
                        }
                    }
                }
            }
            for (; done < chunks; ++done) {
                finish(done);
            }
            noc_async_read_set_trid(0);
        } else {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x));
            for (uint32_t c = 0; c < chunks; ++c) {
                noc_semaphore_wait_min(sem, c + 1);
                const uint32_t first = c * x_chunk;
                xcb.push_back(mt * (kc - first < x_chunk ? kc - first : x_chunk));
            }
            // NoC 1 to the weight stream only now, with x's multicast done (its data was barriered before the last
            // chunk's semaphore): a NoC 1 multicast with other cores' NoC 1 DRAM reads under way jams the NoC now and
            // then (a hang in ~6,000 decode steps when the stream took NoC 1 from the start).  Only with one x
            // sender (ks 1): with more, another rectangle's multicast can still be going on
            if constexpr (both) {
                if (ks == 1) {
                    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_noc1)), 1);
                }
            }
        }
    }

    // 2. the partials
    if (ks > 1) {
        DeviceZoneScopedN("P");
        CircularBuffer part(cb_part), recv(cb_recv);
        const uint32_t pt = get_local_cb_interface(cb_part).fifo_page_size;
        const uint32_t recv_base = get_local_cb_interface(cb_recv).fifo_wr_ptr;  // every owner's too
        const uint32_t owners = ncb < ks ? ncb : ks;
        part.wait_front(ncb);
        const uint32_t src = part.get_read_ptr();
        for (uint32_t t = 0; t < ncb; ++t) {  // tile t to owner t mod ks, at its source s row (s R_o + slot)
            const uint32_t q = t % ks, owner_tiles = (ncb - q + ks - 1) / ks;
            const uint64_t dst = get_noc_addr(
                get_arg_val<uint32_t>(OWNERS + 2 * q),
                get_arg_val<uint32_t>(OWNERS + 2 * q + 1),
                recv_base + (s * owner_tiles + t / ks) * pt);
            noc_async_write(src + t * pt, dst, pt);
        }
        noc_async_write_barrier();
        const uint32_t red_addr = get_semaphore(sem_red);
        for (uint32_t q = 0; q < owners; ++q) {  // bit s: this slice's partials are there
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(OWNERS + 2 * q), get_arg_val<uint32_t>(OWNERS + 2 * q + 1), red_addr),
                1u << s);
        }
        part.pop_front(ncb);
        if (owned > 0) {  // the slices' partials to the compute in the order of K, each as soon as it is there
            volatile tt_l1_ptr uint32_t* red = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_addr);
            recv.reserve_back(owned * ks);
            for (uint32_t q = 0; q < ks; ++q) {
                while (((*red >> q) & 1u) == 0) {
                    invalidate_l1_cache();
                }
                recv.push_back(owned);
            }
        }
    }

    // 3. the output
    CircularBuffer out(cb_out);
    const auto s_out = TensorAccessor(out_args, out_addr);
    const uint32_t ot = get_local_cb_interface(cb_out).fifo_page_size;
    if constexpr (mt == 1) {
        for (uint32_t i = 0; i < owned; ++i) {
            const uint32_t t = ks > 1 ? s + i * ks : i;
            out.wait_front(1);
            noc_async_write(out.get_read_ptr(), s_out.get_noc_addr(n0 + t * stride), ot);
            noc_async_writes_flushed();
            out.pop_front(1);
        }
    } else {  // (ks 1) per sub-block of sbw columns, its mt rows
        for (uint32_t c0 = 0; c0 < ncb; c0 += sb) {
            const uint32_t sbw = ncb - c0 < sb ? ncb - c0 : sb;
            for (uint32_t r = 0; r < mt; ++r) {
                for (uint32_t c = 0; c < sbw; ++c) {
                    out.wait_front(1);
                    noc_async_write(out.get_read_ptr(), s_out.get_noc_addr(r * nt_all + n0 + (c0 + c) * stride), ot);
                    noc_async_writes_flushed();
                    out.pop_front(1);
                }
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
