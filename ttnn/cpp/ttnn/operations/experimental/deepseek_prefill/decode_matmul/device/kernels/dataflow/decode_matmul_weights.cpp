// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// This core's weight block (K tiles k0 .. k0 + kc - 1, N tiles n0 + i stride, i < ncb, of the [K / 32, N / 32] tiles,
// page k N/32 + n) for the compute, in its order: per sub-block of sb columns, per chunk of cr K rows.  A chunk is a
// slot of cr x sb tiles (row r at r sb; a short last sub-block or chunk leaves the rest unread).  With the stride =
// the DRAM banks, a row's tiles of a sub-block are consecutive pages of one bank: one request; with stride 1, one
// request a tile.  Up to `ahead` chunks are in flight, each with its own transaction id, pushed in order as they land.
void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t k0 = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);
    const uint32_t ncb = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);
    const uint32_t kc = get_arg_val<uint32_t>(5);
    const uint32_t cr = get_arg_val<uint32_t>(6);
    const uint32_t sb = get_arg_val<uint32_t>(7);
    const uint32_t ahead = get_arg_val<uint32_t>(8);
    const uint32_t stride = get_arg_val<uint32_t>(9);
    // 2: this core's requests alternate between NoC 0 and NoC 1 once the control kernel hands NoC 1 over (one NoC's
    // reads of a bank top out below the bank's rate); 1: NoC 0 only.  Dedicated NoC mode: both RISCs read with
    // command buffer 1 of their own NoC and share its NIU's counters, so this RISC reads on NoC 1 only on cores where
    // the control kernel (BRISC, NoC 1) does not read (not on x's senders: 1 there), and only after that kernel has
    // set its command buffers' transaction ids to 0 and cleared NoC 1's outstanding counters of ids 1 .. 15 (earlier
    // programs can leave them non-zero) and then set sem_noc1: until then NoC 0 only, with no transaction id set on
    // NoC 1 (a read the control kernel issues would carry it).  The reads here wait on their transaction ids only
    // (1 .. depth; this RISC's firmware keeps no counts of NoC 1).
    const uint32_t nocs = get_arg_val<uint32_t>(10);
    // with nocs 2: at most this many chunks in flight until NoC 1 is handed over (0: `ahead` throughout), so that the
    // banks' queues stay short while x's sender reads x behind them (x then lands sooner, and NoC 1 comes sooner)
    const uint32_t ahead0 = get_arg_val<uint32_t>(11);
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr bool skip = get_compile_time_arg_val(1) != 0;  // a test switch: no reads (the slots pushed as they are)
    constexpr uint32_t sem_noc1 = get_compile_time_arg_val(2);
    constexpr auto w_args = TensorAccessorArgs<3>();
#ifdef DM_FG
    // fg (GLM53_KDA_FG_FUSED), on a core without a block: its N2 tiles n2_0 .. n2_0 + cnt - 1 of
    // fg's weight [fg_k, n2t] tiles into CB_FGW, K row k at k cnt, read at once (NoC 0, one request a tile; the compute
    // waits for all of them)
    constexpr uint32_t fg_k = get_compile_time_arg_val(w_args.next_compile_time_args_offset());
    constexpr uint32_t n2t = get_compile_time_arg_val(w_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t cb_fgw = get_compile_time_arg_val(w_args.next_compile_time_args_offset() + 2);
    constexpr auto fg_args = TensorAccessorArgs<w_args.next_compile_time_args_offset() + 3>();
    if (ncb == 0) {
        const uint32_t fg_addr = get_arg_val<uint32_t>(12);
        const uint32_t n2_0 = get_arg_val<uint32_t>(13);
        const uint32_t cnt = get_arg_val<uint32_t>(14);
        if (cnt == 0) {
            return;
        }
        DeviceZoneScopedN("FGW");
        const auto s_f = TensorAccessor(fg_args, fg_addr);
        CircularBuffer fw(cb_fgw);
        const uint32_t ft = get_local_cb_interface(cb_fgw).fifo_page_size;
        fw.reserve_back(fg_k * cnt);
        const uint32_t dst = fw.get_write_ptr();
        noc_async_read_set_trid(0, noc_index);
        for (uint32_t k = 0; k < fg_k; ++k) {
            for (uint32_t c = 0; c < cnt; ++c) {
                noc_async_read(s_f.get_noc_addr(k * n2t + n2_0 + c), dst + (k * cnt + c) * ft, ft);
            }
        }
        noc_async_read_barrier();
        fw.push_back(fg_k * cnt);
        return;
    }
#endif
#ifdef DM_CONV
#ifdef DM_FG
#error "decode_matmul: fg and conv together"
#endif
    // conv (GLM53_DFLASH2_CONV_FUSED), on a free core with columns j0 .. j0 + nj - 1: per column its E
    // tile E_{j % 16}, its two base tiles (fp32: taps 0 and 1, pages j and xw + j) and the shift S, read at once
    // (NoC 0), as the conv op's reader's first part
    constexpr uint32_t cv_xw = get_compile_time_arg_val(w_args.next_compile_time_args_offset());
    constexpr uint32_t cb_ce = get_compile_time_arg_val(w_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(w_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t cb_cb = get_compile_time_arg_val(w_args.next_compile_time_args_offset() + 3);
    constexpr auto cb_args = TensorAccessorArgs<w_args.next_compile_time_args_offset() + 4>();
    constexpr auto ce_args = TensorAccessorArgs<cb_args.next_compile_time_args_offset()>();
    constexpr auto cs_args = TensorAccessorArgs<ce_args.next_compile_time_args_offset()>();
    if (ncb == 0) {
        const uint32_t nj = get_arg_val<uint32_t>(16);
        if (nj == 0) {
            return;
        }
        DeviceZoneScopedN("CVW");
        constexpr uint32_t BF = 2048, F32 = 4096;
        const auto s_b = TensorAccessor(cb_args, get_arg_val<uint32_t>(12), F32);
        const auto s_e = TensorAccessor(ce_args, get_arg_val<uint32_t>(13), BF);
        const auto s_s = TensorAccessor(cs_args, get_arg_val<uint32_t>(14), BF);
        const uint32_t j0 = get_arg_val<uint32_t>(15);
        noc_async_read_set_trid(0, noc_index);
        cb_reserve_back(cb_cs, 1);
        cb_reserve_back(cb_ce, nj);
        cb_reserve_back(cb_cb, 2 * nj);
        noc_async_read(s_s.get_noc_addr(0), get_write_ptr(cb_cs), BF);
        for (uint32_t a = 0; a < nj; ++a) {
            const uint32_t j = j0 + a;
            noc_async_read(s_e.get_noc_addr(j % 16), get_write_ptr(cb_ce) + a * BF, BF);
            noc_async_read(s_b.get_noc_addr(j), get_write_ptr(cb_cb) + (2 * a) * F32, F32);              // tap 0
            noc_async_read(s_b.get_noc_addr(cv_xw + j), get_write_ptr(cb_cb) + (2 * a + 1) * F32, F32);  // tap 1
        }
        noc_async_read_barrier();
        cb_push_back(cb_cs, 1);
        cb_push_back(cb_ce, nj);
        cb_push_back(cb_cb, 2 * nj);
        return;
    }
#endif
    if (ncb == 0) {
        return;
    }
#ifdef DM_LANDING
    // the landing (GLM53_W1D: the dense weights in L1 ahead of the op): regions A and B at the same addresses
    // on every core, each 64 B of seal and then chunks in their issue order, a whole slot each (cr x sb tiles, as the
    // ring holds them): chunks 0 .. a - 1 in A, a .. a + b - 1 in B (chunk a first), filled by the riders of the
    // programs before this one (ttnn/kernels/w1_dense/dm_rider.cpp), each region taken only when its seal is (tag <<
    // 16) | its last chunk + 1; the chunks from there copied into the ring (a NoC read of this core's own L1), the
    // rest from DRAM as always.  The same bytes into the same slots either way: the values the same.  The seals
    // cleared at the end (a region is filled anew before each use).
    const uint32_t land_a = get_arg_val<uint32_t>(DM_LANDING_RT);
    const uint32_t land_b = get_arg_val<uint32_t>(DM_LANDING_RT + 1);
    const uint32_t land_tag = get_arg_val<uint32_t>(DM_LANDING_RT + 2);
    const uint32_t want_a = get_arg_val<uint32_t>(DM_LANDING_RT + 3);
    const uint32_t want_b = get_arg_val<uint32_t>(DM_LANDING_RT + 4);
    volatile tt_l1_ptr uint32_t* seal_a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(land_a);
    volatile tt_l1_ptr uint32_t* seal_b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(land_b);
    invalidate_l1_cache();
    const uint32_t na = want_a != 0 && seal_a[0] == ((land_tag << 16) | want_a) ? want_a : 0;
    const uint32_t nb = want_b != 0 && land_b != 0 && seal_b[0] == ((land_tag << 16) | (want_a + want_b)) ? want_b : 0;
#else
    constexpr uint32_t land_a = 0, land_b = 0, want_a = 0, na = 0, nb = 0;
#endif
    const auto s_w = TensorAccessor(w_args, w_addr);
    CircularBuffer w(cb_w);
    const uint32_t tile = get_local_cb_interface(cb_w).fifo_page_size;
    const uint32_t slot = cr * sb, depth = ahead + 1, per_sub = (kc + cr - 1) / cr;
    const uint32_t chunks = ((ncb + sb - 1) / sb) * per_sub;
    const uint32_t base = w.get_write_ptr();  // the ring's start: nothing pushed yet
    DeviceZoneScopedN("W");
    uint32_t pushed = 0;
    const uint8_t other = 1 - noc_index;  // NoC 1
    bool both = false;                     // NoC 1 handed over (sem_noc1), from chunk first2 on
    uint32_t first2 = chunks;
    auto landed = [&](uint32_t c) {  // chunk c's reads all here
        noc_async_read_barrier_with_trid(c % depth + 1, noc_index);
        if (c >= first2) {
            noc_async_read_barrier_with_trid(c % depth + 1, other);
        }
    };
    // (DM_LANDING: chunk c's place in this core's L1, 0: DRAM)
    auto from_l1 = [&](uint32_t c) -> uint32_t {
        if (c < na) {
            return land_a + 64 + c * slot * tile;
        }
        if (c >= want_a && c - want_a < nb) {
            return land_b + 64 + (c - want_a) * slot * tile;
        }
        return 0;
    };
    for (uint32_t issued = 0; issued < chunks; ++issued) {
        const uint32_t src = from_l1(issued);
        // (a chunk from L1 is not held back to ahead0: no bank queue to keep short for it)
        const uint32_t lim = both || ahead0 == 0 || src != 0 ? ahead : ahead0;
        while (issued - pushed >= lim) {  // the oldest in flight to the compute
            landed(pushed);
            w.push_back(slot);
            ++pushed;
        }
        w.reserve_back((issued - pushed + 1) * slot);  // room for the ones in flight and this one
        const uint32_t dst = base + (issued % depth) * slot * tile;
        if (nocs == 2 && !both && src == 0) {
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_noc1)) != 0) {
                both = true;
                first2 = issued;
            }
        }
        noc_async_read_set_trid(issued % depth + 1, noc_index);  // ids 1 .. depth
        if (both) {
            noc_async_read_set_trid(issued % depth + 1, other);
        }
        if constexpr (skip) {
            continue;
        }
        if (src != 0) {  // (DM_LANDING: the chunk's slot from this core's own L1, on this RISC's NoC)
            noc_async_read(get_noc_addr(src), dst, slot * tile);
            continue;
        }
        const uint32_t c0 = (issued / per_sub) * sb;
        const uint32_t sbw = ncb - c0 < sb ? ncb - c0 : sb;
        const uint32_t g = issued % per_sub;
        const uint32_t rows = kc - g * cr < cr ? kc - g * cr : cr;  // the slice's last chunk may be short
        const uint32_t row0 = k0 + g * cr;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t page = (row0 + r) * nt + n0 + c0 * stride;
            if (stride != 1) {  // pages page, page + B, ..: adjacent in one bank
                const uint8_t k = both && ((issued * cr + r) & 1) ? other : noc_index;
                noc_async_read(s_w.get_noc_addr(page, 0, k), dst + r * sb * tile, sbw * tile, k);
                continue;
            }
            for (uint32_t c = 0; c < sbw; ++c) {
                noc_async_read(s_w.get_noc_addr(page + c), dst + (r * sb + c) * tile, tile);
            }
        }
    }
    for (; pushed < chunks; ++pushed) {
        landed(pushed);
        w.push_back(slot);
    }
#ifdef DM_LANDING
    if (want_a != 0) {  // (the copies out of the regions landed above)
        seal_a[0] = 0;
    }
    if (want_b != 0) {
        seal_b[0] = 0;
    }
#endif
    noc_async_read_set_trid(0, noc_index);
    if (both) {
        noc_async_read_set_trid(0, other);
    }
    noc_async_atomic_barrier();
}
