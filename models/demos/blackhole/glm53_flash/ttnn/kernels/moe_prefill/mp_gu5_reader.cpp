// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "mp_common.hpp"

// The prefill MoE's gate | up by groups split along K (mp_common.hpp, mp_gu5_*.cpp), the reader, NoC 0: the group's
// experts with pairs in turn, the first one's W1 K block c (all 8 column tiles) into G5_W1 up front; per tile row its
// weights column and this member's K block c of the 32 tokens' x rows into G5_XRM (two rows), the row's x reads with
// the next row's token and weight pages and with an even share of the next expert's W1 blocks (all in by the
// expert's last row: the compute does not wait for W1 between experts), under one barrier; the weights
// column's zeros written once (only column 0 changes).
void kernel_main() {
    using namespace moe_prefill;
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t seg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ptok_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pw_addr = get_arg_val<uint32_t>(a++);
    const uint32_t group = get_arg_val<uint32_t>(a++);
    const uint32_t groups = get_arg_val<uint32_t>(a++);
    const uint32_t c = get_arg_val<uint32_t>(a++);  // member: its K block and its output column tile

    constexpr uint32_t x_row = get_compile_time_arg_val(0);
    constexpr uint32_t w1_tile = get_compile_time_arg_val(1);
    constexpr uint32_t local = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    // timing probes (wrong results; tests only): 8 W1 read for the first expert only (the pages pushed unread after),
    // 16 no x reads (the block pushed unread), 64 the first expert's W1 not read (as if in L1 already)
    constexpr uint32_t probe = get_compile_time_arg_val(p_args.next_compile_time_args_offset());
    // GU5_NOCS 2: of every 4 W1 tiles of a K row (gate, up of 2 column tiles) and every 4 x rows, the
    // first q1 read on NoC 1, once the writer (BRISC, NoC 1, which reads nothing after its seg page) has handed NoC 1
    // over (mp_noc1.hpp, S_NOC1); the NoC 1 reads carry transaction id 1 and are waited for by that id only (this RISC
    // keeps no counts of NoC 1), as decode_matmul's weight stream does
    constexpr uint32_t nocs = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 1);
    // BALANCE: the group's segments of the column's tile rows split evenly (mp_common.hpp), each expert start
    // weighing bal_a rows; 0: the experts group, group + groups, .. whole
    constexpr bool balance = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 2) != 0;
    constexpr uint32_t bal_a = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t bal_pct = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 4);
    // the progressive form (moe_prefill.py PROG): the slots past an expert's base where its correction is (0: none);
    // each expert's K block loaded twice, the base's blocks then the correction's
    constexpr uint32_t corr = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 5);
    // q1 by this core's place on the die: a 2-bit share for each physical (NoC 0) x 0 .. Q1_X - 1 and y 0 .. Q1_Y - 1,
    // y-major, 16 to a word in runtime args 8 .. (moe_prefill.py, GU5_NOC1_QUARTERS / _RULES).  The place is the NIU's
    // NOC_NODE_ID, which counts the harvested columns (my_x / my_y are the translated coordinates, the same on every
    // chip); under all cores reading DRAM, NoC 1's reads end by physical column
    const uint32_t node = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);
    const uint32_t q_at = ((node >> NOC_ADDR_NODE_ID_BITS) & 0x3F) * Q1_X + (node & 0x3F);
    const uint32_t q1 = q_at < Q1_X * Q1_Y ? (get_arg_val<uint32_t>(8 + q_at / 16) >> (2 * (q_at % 16))) & 3 : 2;
    constexpr uint32_t PAGE = ROW_WORDS * 4;
    constexpr uint32_t BLK = KBLK * 64;  // a row's K block: 512 bf16
    constexpr uint32_t W1_HALF = KBLK / 2;  // a load: 2 K rows of the block, all 16 output tiles
    constexpr uint32_t W1_BLOCKS = (corr ? 2 : 1) * W1_HALF;  // (PROG: the base's, then the correction's)

    const auto s_x = TensorAccessor(x_args, x_addr, x_row);
    const auto s_w1 = TensorAccessor(w_args, w1_addr, w1_tile);
    const auto s_seg = TensorAccessor(s_args, seg_addr, SEG_WORDS * 4);
    const auto s_ptok = TensorAccessor(t_args, ptok_addr, PAGE);
    const auto s_pw = TensorAccessor(p_args, pw_addr, PAGE);

    const uint32_t seg_l1 = get_write_ptr(G5_SCR);
    const uint32_t tok_l1 = seg_l1 + SEG_WORDS * 4;
    const uint32_t w_l1 = tok_l1 + 128;
    noc_async_read(s_seg.get_noc_addr(0), seg_l1, SEG_WORDS * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* seg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seg_l1);
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_l1);
    volatile tt_l1_ptr uint32_t* pwt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);

    // the group's segments: expert | first tile row << 8 | tile rows << 20 (12 bits each: an expert's rows are at
    // most N / 32, 256 at a column's 8,192 tokens)
    uint32_t items[local];
    uint32_t n_items = 0;
    if constexpr (balance) {
        BalancedWalk walk = balanced_walk(seg, local, 1, bal_a, group, groups, bal_pct);
        uint32_t item, r0, nr;
        while (next_segment(seg, 1, bal_a, walk, item, r0, nr)) {
            items[n_items++] = item | r0 << 8 | nr << 20;
        }
    } else {
        for (uint32_t e = group; e < local; e += groups) {
            if (seg[SEG_COUNT + e] > 0) {
                items[n_items++] = e | ((seg[SEG_COUNT + e] + 31) / 32) << 20;
            }
        }
    }
    auto it_e = [&](uint32_t i) { return items[i] & 0xFF; };
    auto it_rows = [&](uint32_t i) { return items[i] >> 20; };
    auto it_off = [&](uint32_t i) { return seg[it_e(i)] + 32 * ((items[i] >> 8) & 0xFFF); };  // its first pair row
    const uint8_t other = 1 - noc_index;
    uint8_t n2 = noc_index;  // the NoC of the second half of the reads (NoC 1 once handed over)
    auto barrier = [&]() {
        noc_async_read_barrier();
        if (nocs == 2 && n2 != noc_index) {
            noc_async_read_barrier_with_trid(1, other);
        }
    };
    if constexpr (nocs == 2) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(S_NOC1)), 1);
        noc_async_read_set_trid(1, other);
        n2 = other;
    }
    // W1's (gate, up) tiles t, t + 1 of a K row and column tile (adjacent in the bank: a shard is 16 K rows x 2) into
    // at, at + w1_tile, in one request of 2 tiles (NoC 1 for q1 of every 4 pairs; the two requests before,
    // GLM53_MOE_GU5_W1_PAIRS=0, were removed)
    auto read_pair = [&](uint32_t t, uint32_t at, uint32_t kk, uint32_t cc) {
        const uint8_t q = ((kk * COLS + cc) & 3) < q1 ? n2 : noc_index;
        noc_async_read(s_w1.get_noc_addr(t, 0, q), at, 2 * w1_tile, q);
    };
    uint32_t li = 0, lb = 0;  // the next W1 K block to load: item li, block lb
    auto load_one = [&]() {  // W1(e, c', k, g) for k = 16 c + 2 lb, + 1, every c' and g, at (k - 16 c) 16 + 2 c' + g
        cb_reserve_back(G5_W1, 2 * G5_N);
        const uint32_t dst = get_write_ptr(G5_W1);
        for (uint32_t kk = 0; kk < 2 && !((probe & 8) && li > 0) && !((probe & 64) && li == 0); ++kk) {
            const uint32_t k = c * KBLK + 2 * (lb % W1_HALF) + kk;
            const uint32_t ee = it_e(li) + (lb / W1_HALF) * corr;  // (PROG: the correction's slot)
            for (uint32_t cc = 0; cc < COLS; ++cc) {
                const uint32_t t = ((ee * COLS + cc) * KD + k) * 2;
                const uint32_t at = dst + (kk * G5_N + 2 * cc) * w1_tile;
                read_pair(t, at, kk, cc);
            }
        }
        barrier();
        cb_push_back(G5_W1, 2 * G5_N);
        if (++lb == W1_BLOCKS) {
            lb = 0;
            ++li;
        }
    };
    while (li == 0 && n_items > 0) {
        load_one();
    }

    // the weights columns zeroed once (only column 0 is written per row)
    {
        const uint32_t w0 = get_write_ptr(G5_WCOL);
        const uint32_t bytes = get_local_cb_interface(G5_WCOL).fifo_size;
        for (uint32_t z = 0; z < bytes; z += 512) {
            noc_async_read(zeros, w0 + z, 512);
        }
        noc_async_read_barrier();
    }
    // tok / pw pages of a row, double-buffered: buffer b at tok_l1 + b 256 (tok), + 128 (pw)
    auto pages = [&](uint32_t off, uint32_t b) {
        const uint32_t p0 = off / ROW_WORDS;
        const uint32_t t = tok_l1 + b * 256;
        noc_async_read(s_ptok.get_noc_addr(p0), t, PAGE);
        noc_async_read(s_ptok.get_noc_addr(p0 + 1), t + PAGE, PAGE);
        noc_async_read(s_pw.get_noc_addr(p0), t + 128, PAGE);
        noc_async_read(s_pw.get_noc_addr(p0 + 1), t + 128 + PAGE, PAGE);
    };
    uint32_t b = 0;
    if (n_items > 0) {
        pages(it_off(0), 0);
        noc_async_read_barrier();
    }
    for (uint32_t i = 0; i < n_items; ++i) {
        const uint32_t rows = it_rows(i);
        const uint32_t off = it_off(i);
        for (uint32_t r = 0; r < rows; ++r, b ^= 1) {
            volatile tt_l1_ptr uint32_t* tk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_l1 + b * 256);
            volatile tt_l1_ptr uint32_t* wt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_l1 + b * 256 + 128);
            cb_reserve_back(G5_WCOL, 1);
            volatile tt_l1_ptr uint16_t* wc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(G5_WCOL));
            for (uint32_t y = 0; y < 32; ++y) {
                wc[(y / 16) * 2 * 256 + (y % 16) * 16] = static_cast<uint16_t>(wt[y] >> 16);
            }
            cb_push_back(G5_WCOL, 1);
            {
                DeviceZoneScopedN("G5_RD_XRM");  // the compute two rows behind
                cb_reserve_back(G5_XRM, KBLK);
            }
            const uint32_t blk = get_write_ptr(G5_XRM);
            for (uint32_t y = 0; y < 32 && !(probe & 16); ++y) {
                const uint32_t t = tk[y];
                const uint32_t dst = blk + y * BLK;
                if (t != SENT) {
                    const uint8_t k = (y & 3) < q1 ? n2 : noc_index;
                    noc_async_read(s_x.get_noc_addr(t, c * BLK, k), dst, BLK, k);
                } else {
                    noc_async_read(zeros, dst, 512);
                    noc_async_read(zeros, dst + 512, 512);
                }
            }
            // the next row's pages (this expert's next row, or the next expert's first) into the other buffer
            if (r + 1 < rows) {
                pages(off + 32 * (r + 1), b ^ 1);
            } else if (i + 1 < n_items) {
                pages(it_off(i + 1), b ^ 1);
            }
            // the next expert's W1: an even share of its blocks left over this expert's rows left (one here)
            const bool w1 = li == i + 1 && li < n_items && cb_pages_reservable_at_back(G5_W1, 2 * G5_N);
            uint32_t share = 0;
            if (li == i + 1 && li < n_items) {
                share = (W1_BLOCKS - lb + (rows - r) - 1) / (rows - r);
            }
            if (w1 && share > 0) {
                cb_reserve_back(G5_W1, 2 * G5_N);
                const uint32_t dst = get_write_ptr(G5_W1);
                for (uint32_t kk = 0; kk < 2 && !(probe & 8); ++kk) {
                    const uint32_t k = c * KBLK + 2 * (lb % W1_HALF) + kk;
                    const uint32_t ee = it_e(li) + (lb / W1_HALF) * corr;  // (PROG: the correction's slot)
                    for (uint32_t cc = 0; cc < COLS; ++cc) {
                        const uint32_t t = ((ee * COLS + cc) * KD + k) * 2;
                        const uint32_t at = dst + (kk * G5_N + 2 * cc) * w1_tile;
                        read_pair(t, at, kk, cc);
                    }
                }
            }
            {
                DeviceZoneScopedN("G5_RD_BAR");  // the row's reads in
                barrier();
            }
            cb_push_back(G5_XRM, KBLK);
            if (w1 && share > 0) {
                cb_push_back(G5_W1, 2 * G5_N);
                if (++lb == W1_BLOCKS) {
                    lb = 0;
                    ++li;
                }
                for (uint32_t extra = 1; extra < share && li == i + 1 && li < n_items &&
                                         cb_pages_reservable_at_back(G5_W1, 2 * G5_N);
                     ++extra) {
                    load_one();
                }
            }
        }
        while (li == i + 1 && li < n_items) {
            load_one();
        }
    }
    if (n2 != noc_index) {
        noc_async_read_set_trid(0, other);
    }
}
