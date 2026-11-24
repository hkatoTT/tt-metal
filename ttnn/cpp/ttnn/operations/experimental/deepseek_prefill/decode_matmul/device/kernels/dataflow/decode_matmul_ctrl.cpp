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
// The page layouts (decode_matmul_device_operation_types.hpp; the same tiles, other pages): x's tile (r, k) at page
// (k / xw) mt xw + r xw + k mod xw with x_heads (xw its tiles a head), else r kt + k; the output's (r, n) likewise with
// out_heads (ow); with a split, columns from split_t on to the second output (its own row stride nt - split_t).  With
// one tile row (mt 1) the heads' pages are the plain ones.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out2_addr = get_arg_val<uint32_t>(2);
    const uint32_t k0 = get_arg_val<uint32_t>(3);
    const uint32_t kc = get_arg_val<uint32_t>(4);
    const uint32_t n0 = get_arg_val<uint32_t>(5);
    const uint32_t ncb = get_arg_val<uint32_t>(6);
    const uint32_t s = get_arg_val<uint32_t>(7);
    const uint32_t ks = get_arg_val<uint32_t>(8);
    const uint32_t rect = get_arg_val<uint32_t>(9);
    const uint32_t sender = get_arg_val<uint32_t>(10);
    const uint32_t x_chunk = get_arg_val<uint32_t>(11);  // x's tiles a multicast
    const uint32_t owned = get_arg_val<uint32_t>(12);
    const uint32_t mc0_x = get_arg_val<uint32_t>(13);
    const uint32_t mc0_y = get_arg_val<uint32_t>(14);
    const uint32_t mc1_x = get_arg_val<uint32_t>(15);
    const uint32_t mc1_y = get_arg_val<uint32_t>(16);
    const uint32_t stride = get_arg_val<uint32_t>(17);
    constexpr uint32_t OWNERS = 18;  // the owners' (x, y) from here

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
    constexpr uint32_t xw = get_compile_time_arg_val(13);       // x_heads: x's tiles a head (0: [1, 1, M, K])
    constexpr uint32_t ow = get_compile_time_arg_val(14);       // out_heads: the output's tiles a head
    constexpr uint32_t split_t = get_compile_time_arg_val(15);  // the split's first column tile (0: none)
    constexpr auto x_args = TensorAccessorArgs<16>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto out2_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    static_assert(split_t == 0 || ow == 0, "decode_matmul: out_heads and a split together");
#ifdef DM_FG
    // fg (GLM53_KDA_FG_FUSED; K whole on each core: the arguments from 18 on).  On a core without
    // a block: wait for fg's input (SEM_FG: one increment a column, after its mt tiles landed in CB_FGIN, tile (r, j)
    // at j mt + r), hand it to the compute, and write fg's output tiles (CB_FGOUT: per sub-block of up to fg_sb N2
    // columns, its mt x w tiles row-major, the slot fg_sb mt tiles whatever w) to fg_out [M, N2].  On the block of
    // output column nt - fg_k + j (fg_col = j + 1): its output tiles also to every free core's CB_FGIN, then their
    // SEM_FG + 1.
    constexpr uint32_t FG = 18;
    constexpr uint32_t fg_k = get_compile_time_arg_val(out2_args.next_compile_time_args_offset());
    constexpr uint32_t n2t = get_compile_time_arg_val(out2_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t cb_fgin = get_compile_time_arg_val(out2_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t cb_fgout = get_compile_time_arg_val(out2_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t sem_fg = get_compile_time_arg_val(out2_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t fg_sb = get_compile_time_arg_val(out2_args.next_compile_time_args_offset() + 5);
    constexpr auto fgo_args = TensorAccessorArgs<out2_args.next_compile_time_args_offset() + 6>();
    const uint32_t fg_col = get_arg_val<uint32_t>(FG);
    if (ncb == 0) {
        const uint32_t fg_cnt = get_arg_val<uint32_t>(FG + 1);
        if (fg_cnt == 0) {
            return;
        }
        const uint32_t fg_n0 = get_arg_val<uint32_t>(FG + 2);
        const auto s_fg = TensorAccessor(fgo_args, get_arg_val<uint32_t>(FG + 3));
        {
            DeviceZoneScopedN("FGIN");
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_fg)), fg_k);
            CircularBuffer fin(cb_fgin);  // (its tiles written in place by the senders: the CB's start)
            fin.reserve_back(fg_k * mt);
            fin.push_back(fg_k * mt);
        }
        DeviceZoneScopedN("FGOUT");
        if constexpr (both) {  // (as the busy cores: NoC 1's transaction ids cleared before its first transaction)
            noc_async_read_set_trid(0, noc_index);
            noc_async_write_set_trid(0, noc_index);
            ncrisc_noc_set_transaction_id(noc_index, write_reg_cmd_buf, 0);
            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
        }
        CircularBuffer fo(cb_fgout);
        const uint32_t ft = get_local_cb_interface(cb_fgout).fifo_page_size;
        for (uint32_t c0 = 0; c0 < fg_cnt; c0 += fg_sb) {
            const uint32_t w = fg_cnt - c0 < fg_sb ? fg_cnt - c0 : fg_sb;
            fo.wait_front(fg_sb * mt);
            const uint32_t src = fo.get_read_ptr();
            for (uint32_t r = 0; r < mt; ++r) {
                for (uint32_t c = 0; c < w; ++c) {
                    noc_async_write(src + (r * w + c) * ft, s_fg.get_noc_addr(r * n2t + fg_n0 + c0 + c), ft);
                }
            }
            noc_async_writes_flushed();
            fo.pop_front(fg_sb * mt);
        }
        noc_async_write_barrier();
        return;
    }
    // a block of fg's input: its tiles to the free cores' CB_FGIN (the same address on every core)
    const uint32_t fg_n = get_arg_val<uint32_t>(FG + 4);
    const uint32_t fgin_base = get_local_cb_interface(cb_fgin).fifo_wr_ptr;
    auto fg_send = [&](uint32_t src, uint32_t r, uint32_t bytes) {
        for (uint32_t i = 0; i < fg_n; ++i) {
            noc_async_write(
                src,
                get_noc_addr(
                    get_arg_val<uint32_t>(FG + 5 + 2 * i),
                    get_arg_val<uint32_t>(FG + 6 + 2 * i),
                    fgin_base + ((fg_col - 1) * mt + r) * bytes),
                bytes);
        }
    };
#endif
#ifdef DM_CONV
    // conv (GLM53_DFLASH2_CONV_FUSED; K whole on each core: the arguments from 18 on).  On a free core
    // with columns j0 .. j0 + nj - 1: x as on the other cores that receive it (SEM_X, chunk by chunk, all of K: the
    // compute reads x's tiles (i, j) at j mt + i), then its two delta tiles (SEM_CONV: one increment a tap, after the
    // tap's mt tiles landed in CB_CD at 2 i + t), then the output tiles (i, j), row by row, to conv_out [M, xw].  On
    // the block of delta column 16 side + 8 t + q (cv_tap = t + 1): its tiles also to the group's free cores' CB_CD,
    // then their SEM_CONV + 1.
    constexpr uint32_t CV = 18;
    constexpr uint32_t cv_xw = get_compile_time_arg_val(out2_args.next_compile_time_args_offset());
    constexpr uint32_t cb_cd = get_compile_time_arg_val(out2_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t cb_co = get_compile_time_arg_val(out2_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t sem_cv = get_compile_time_arg_val(out2_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t cb_cx = get_compile_time_arg_val(out2_args.next_compile_time_args_offset() + 4);
    constexpr auto cvo_args = TensorAccessorArgs<out2_args.next_compile_time_args_offset() + 5>();
    const uint32_t cv_tap = get_arg_val<uint32_t>(CV);
    if (ncb == 0) {
        const uint32_t cv_nj = get_arg_val<uint32_t>(CV + 1);
        if (cv_nj == 0) {
            return;
        }
        const uint32_t cv_j0 = get_arg_val<uint32_t>(CV + 2);
        const auto s_cv = TensorAccessor(cvo_args, get_arg_val<uint32_t>(CV + 3));
        {
            DeviceZoneScopedN("CVIN");
            CircularBuffer xcb(cb_cx);
            xcb.reserve_back(mt * kc);
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x));
            const uint32_t chunks = (kc + x_chunk - 1) / x_chunk;
            for (uint32_t c = 0; c < chunks; ++c) {
                noc_semaphore_wait_min(sem, c + 1);
                const uint32_t first = c * x_chunk;
                xcb.push_back(mt * (kc - first < x_chunk ? kc - first : x_chunk));
            }
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_cv)), 2);
            CircularBuffer dcb(cb_cd);  // (its tiles written in place by the senders: the CB's start)
            dcb.reserve_back(2 * mt);
            dcb.push_back(2 * mt);
        }
        DeviceZoneScopedN("CVOUT");
        if constexpr (both) {  // (as the busy cores: NoC 1's transaction ids cleared before its first transaction)
            noc_async_read_set_trid(0, noc_index);
            noc_async_write_set_trid(0, noc_index);
            ncrisc_noc_set_transaction_id(noc_index, write_reg_cmd_buf, 0);
            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
        }
        CircularBuffer ocb(cb_co);
        const uint32_t ot = get_local_cb_interface(cb_co).fifo_page_size;
        for (uint32_t i = 0; i < mt; ++i) {
            for (uint32_t j = cv_j0; j < cv_j0 + cv_nj; ++j) {
                ocb.wait_front(1);
                noc_async_write(ocb.get_read_ptr(), s_cv.get_noc_addr(i * cv_xw + j), ot);
                noc_async_writes_flushed();
                ocb.pop_front(1);
            }
        }
        noc_async_write_barrier();
        return;
    }
    // a block of a delta column: its tiles to the group's free cores' CB_CD (the same address on every core)
    const uint32_t cv_n = get_arg_val<uint32_t>(CV + 4);
    const uint32_t cd_base = get_local_cb_interface(cb_cd).fifo_wr_ptr;
    auto cv_send = [&](uint32_t src, uint32_t r, uint32_t bytes) {
        for (uint32_t i = 0; i < cv_n; ++i) {
            noc_async_write(
                src,
                get_noc_addr(
                    get_arg_val<uint32_t>(CV + 5 + 2 * i),
                    get_arg_val<uint32_t>(CV + 6 + 2 * i),
                    cd_base + (2 * r + cv_tap - 1) * bytes),
                bytes);
        }
    };
#endif
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
                            const uint32_t k = k0 + i;
                            const uint32_t page = xw != 0 ? (k / xw) * (mt * xw) + r * xw + k % xw : r * kt_all + k;
                            noc_async_read(s_x.get_noc_addr(page), xbase + (i * mt + r) * xt, xt);
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
    const auto s_out2 = TensorAccessor(out2_args, out2_addr);
    const uint32_t ot = get_local_cb_interface(cb_out).fifo_page_size;
    // output tile (r, n)'s place (the layouts above)
    auto out_noc = [&](uint32_t r, uint32_t n) -> uint64_t {
        if constexpr (split_t != 0) {
            return n >= split_t ? s_out2.get_noc_addr(r * (nt_all - split_t) + n - split_t)
                                : s_out.get_noc_addr(r * split_t + n);
        } else if constexpr (ow != 0) {
            return s_out.get_noc_addr((n / ow) * (mt * ow) + r * ow + n % ow);
        } else {
            return s_out.get_noc_addr(r * nt_all + n);
        }
    };
    if constexpr (mt == 1) {
        for (uint32_t i = 0; i < owned; ++i) {
            const uint32_t t = ks > 1 ? s + i * ks : i;
            out.wait_front(1);
            noc_async_write(out.get_read_ptr(), out_noc(0, n0 + t * stride), ot);
#ifdef DM_FG
            if (fg_col != 0) {  // (one column: its tile is the input's column fg_col - 1)
                fg_send(out.get_read_ptr(), 0, ot);
            }
#endif
#ifdef DM_CONV
            if (cv_tap != 0) {  // (one column: its tile is the group's delta of tap cv_tap - 1)
                cv_send(out.get_read_ptr(), 0, ot);
            }
#endif
            noc_async_writes_flushed();
            out.pop_front(1);
        }
    } else {  // (ks 1) per sub-block of sbw columns, its mt rows
        for (uint32_t c0 = 0; c0 < ncb; c0 += sb) {
            const uint32_t sbw = ncb - c0 < sb ? ncb - c0 : sb;
            for (uint32_t r = 0; r < mt; ++r) {
                for (uint32_t c = 0; c < sbw; ++c) {
                    out.wait_front(1);
                    noc_async_write(out.get_read_ptr(), out_noc(r, n0 + (c0 + c) * stride), ot);
#ifdef DM_FG
                    if (fg_col != 0) {
                        fg_send(out.get_read_ptr(), r, ot);
                    }
#endif
#ifdef DM_CONV
                    if (cv_tap != 0) {
                        cv_send(out.get_read_ptr(), r, ot);
                    }
#endif
                    noc_async_writes_flushed();
                    out.pop_front(1);
                }
            }
        }
    }
    noc_async_write_barrier();
#ifdef DM_FG
    if (fg_col != 0) {  // the tiles landed (the barrier above): each free core's SEM_FG + 1
        const uint32_t sem_addr = get_semaphore(sem_fg);
        for (uint32_t i = 0; i < fg_n; ++i) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(FG + 5 + 2 * i), get_arg_val<uint32_t>(FG + 6 + 2 * i), sem_addr),
                1);
        }
    }
#endif
#ifdef DM_CONV
    if (cv_tap != 0) {  // the tiles landed (the barrier above): each of the group's free cores' SEM_CONV + 1
        const uint32_t sem_addr = get_semaphore(sem_cv);
        for (uint32_t i = 0; i < cv_n; ++i) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(CV + 5 + 2 * i), get_arg_val<uint32_t>(CV + 6 + 2 * i), sem_addr),
                1);
        }
    }
#endif
    noc_async_atomic_barrier();
}
