// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// split_channels (KDA_PREP_SPLIT): a q / k / v unit's channel tiles c_lo .. c_lo + c_n - 1 only (runtime args 11, 12;
// the gates' all 4)
#ifndef KDA_PREP_SPLIT
#define KDA_PREP_SPLIT 0
#endif

namespace kda_prepare_reader {
// Per unit (part 0 q, 1 k, 2 v, 3 gates; head h; block of 16 users = the rows of one face pair of a [B, ..] tile row):
// the block's rows of each input in rows 0-15 of the part's 4 tiles: the bf16 ones (the conv input, f) as they are
// (the compute takes them through SrcA), the fp32 ones unpacked to Dest.  Per-channel constants (the 4 taps, dt_bias, rate) ride in row 0 (the compute
// broadcasts them down the rows).  q / k / v: the conv input (proj), the 3 history rows and the taps; then this core
// writes the shifted history back (history 0 <- 1, 1 <- 2; the writer writes 2 <- the conv input, which the compute
// packs in fp32) for these channels and users, which no other unit touches.  Gates: f, dt_bias, rate, and the
// block's beta logits for head h at (i, 0).  The history writes are waited for before this core's next reads into
// those CBs, and at the end.
// A tile is 4 faces of 16 x 16: rows 0-15 are faces 0 and 1, rows 16-31 faces 2 and 3.  The history is read for the block's rows
// only (rows x 64 bytes a face: 1 / 16 of the bytes at 1 user).
constexpr uint32_t N = 4;

// one 32-byte bf16 face row: its 8 words loaded back to back, then stored (not a load-store chain)
FORCE_INLINE void move_row32(uint32_t from, uint32_t to) {
    volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(to);
    const uint32_t v0 = f[0], v1 = f[1], v2 = f[2], v3 = f[3], v4 = f[4], v5 = f[5], v6 = f[6], v7 = f[7];
    t[0] = v0, t[1] = v1, t[2] = v2, t[3] = v3, t[4] = v4, t[5] = v5, t[6] = v6, t[7] = v7;
}

// rows row0 .. row0 + n - 1 of the 4 bf16 tiles' faces 0 and 1 at `from` to rows 0 .. n - 1 of those at `to` (the later
// tokens' rows of a face read whole: `packed`)
FORCE_INLINE void copy_rows16(uint32_t from, uint32_t to, uint32_t row0, uint32_t n, uint32_t t16) {
    constexpr uint32_t F16 = 512, R16 = 32;
    for (uint32_t c = 0; c < 4; ++c) {
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t s = from + c * t16 + half * F16 + row0 * R16, d = to + c * t16 + half * F16;
            for (uint32_t r = 0; r < n; ++r) {
                move_row32(s + r * R16, d + r * R16);
            }
        }
    }
}

// one 64-byte face row: the 16 words loaded first, then stored (the loads back to back, not a load-store chain)
FORCE_INLINE void move_row64(uint32_t from, uint32_t to) {
    volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(to);
    const uint32_t v0 = f[0], v1 = f[1], v2 = f[2], v3 = f[3], v4 = f[4], v5 = f[5], v6 = f[6], v7 = f[7];
    const uint32_t v8 = f[8], v9 = f[9], v10 = f[10], v11 = f[11], v12 = f[12], v13 = f[13], v14 = f[14], v15 = f[15];
    t[0] = v0, t[1] = v1, t[2] = v2, t[3] = v3, t[4] = v4, t[5] = v5, t[6] = v6, t[7] = v7;
    t[8] = v8, t[9] = v9, t[10] = v10, t[11] = v11, t[12] = v12, t[13] = v13, t[14] = v14, t[15] = v15;
}

// one bf16 face row (16 values, 32 bytes) as fp32 (64 bytes): exact (the bf16 bits as the high half)
FORCE_INLINE void widen_row(uint32_t from, uint32_t to) {
    volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(to);
    const uint32_t v0 = f[0], v1 = f[1], v2 = f[2], v3 = f[3], v4 = f[4], v5 = f[5], v6 = f[6], v7 = f[7];
    t[0] = v0 << 16, t[1] = v0 & 0xFFFF0000u, t[2] = v1 << 16, t[3] = v1 & 0xFFFF0000u;
    t[4] = v2 << 16, t[5] = v2 & 0xFFFF0000u, t[6] = v3 << 16, t[7] = v3 & 0xFFFF0000u;
    t[8] = v4 << 16, t[9] = v4 & 0xFFFF0000u, t[10] = v5 << 16, t[11] = v5 & 0xFFFF0000u;
    t[12] = v6 << 16, t[13] = v6 & 0xFFFF0000u, t[14] = v7 << 16, t[15] = v7 & 0xFFFF0000u;
}

FORCE_INLINE void fill_ones(uint32_t addr) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        p[i] = 0x3f800000u;
    }
}


// The kernel, its compile-time args from CT on and runtime args from RT on (decode_prepare: 0, 0).
template <uint32_t CT, uint32_t RT>
FORCE_INLINE void run() {
    uint32_t a = RT;
    const uint32_t proj_addr = get_arg_val<uint32_t>(a++);
    const uint32_t fg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h0_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h2_addr = get_arg_val<uint32_t>(a++);
    const uint32_t taps_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gc_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t count = get_arg_val<uint32_t>(a++);
    const uint32_t keep_addr = get_arg_val<uint32_t>(a++);
    const uint32_t saved_addr = get_arg_val<uint32_t>(a++);
    // (after c_lo, c_n:) where this reader writes history 0 and 1 and the checkpoint's rows (the inputs themselves in
    // place; out of place, a ping-pong's other set)
    const uint32_t h0_out_addr = get_arg_val<uint32_t>(a + 2);
    const uint32_t h1_out_addr = get_arg_val<uint32_t>(a + 3);
    const uint32_t saved_out_addr = get_arg_val<uint32_t>(a + 4);
    constexpr bool split = KDA_PREP_SPLIT != 0;
    const uint32_t c_lo = split ? get_arg_val<uint32_t>(a) : 0;  // q / k / v's channel tiles (the gates' all)
    const uint32_t c_hi = split ? c_lo + get_arg_val<uint32_t>(a + 1) : 4;

    constexpr uint32_t cb_cur = get_compile_time_arg_val(CT + 0);
    constexpr uint32_t cb_h0 = get_compile_time_arg_val(CT + 1);
    constexpr uint32_t cb_h1 = get_compile_time_arg_val(CT + 2);
    constexpr uint32_t cb_h2 = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t cb_tap = get_compile_time_arg_val(CT + 4);
    constexpr uint32_t cb_f = get_compile_time_arg_val(CT + 5);
    constexpr uint32_t cb_dt = get_compile_time_arg_val(CT + 6);
    constexpr uint32_t cb_rate = get_compile_time_arg_val(CT + 7);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(CT + 8);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(CT + 9);
    constexpr uint32_t cb_stg = get_compile_time_arg_val(CT + 10);
    constexpr uint32_t heads = get_compile_time_arg_val(CT + 11);
    constexpr uint32_t proj_tiles = get_compile_time_arg_val(CT + 12);
    constexpr uint32_t fg_tiles = get_compile_time_arg_val(CT + 13);
    constexpr uint32_t hist_tiles = get_compile_time_arg_val(CT + 14);
    constexpr uint32_t users = get_compile_time_arg_val(CT + 15);
    constexpr auto proj_args = TensorAccessorArgs<CT + 16>();
    constexpr auto fg_args = TensorAccessorArgs<proj_args.next_compile_time_args_offset()>();
    constexpr auto h0_args = TensorAccessorArgs<fg_args.next_compile_time_args_offset()>();
    constexpr auto h1_args = TensorAccessorArgs<h0_args.next_compile_time_args_offset()>();
    constexpr auto h2_args = TensorAccessorArgs<h1_args.next_compile_time_args_offset()>();
    constexpr auto taps_args = TensorAccessorArgs<h2_args.next_compile_time_args_offset()>();
    constexpr auto gc_args = TensorAccessorArgs<taps_args.next_compile_time_args_offset()>();
    constexpr uint32_t blocks = (users + 15) / 16;
    // a verify step (`choose`): each user's history chosen by its keep code (below).  The accessors are built either
    // way (args are always passed).
    constexpr uint32_t X = gc_args.next_compile_time_args_offset();
    constexpr bool choose = get_compile_time_arg_val(X + 0) != 0;
    constexpr uint32_t cb_keep = get_compile_time_arg_val(X + 1);
    constexpr uint32_t keep_bytes = get_compile_time_arg_val(X + 2);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(X + 3);  // scratch: the faces an unaligned row offset reads
    constexpr uint32_t cb_saved = get_compile_time_arg_val(X + 4);  // scratch: the checkpoint's rows (with choose)
    // `tokens` (1-8) a user (a verify step): token k's rows of proj and fg from row k users on, into tiles k N .. of
    // cb_cur / cb_f and tile k of cb_beta (a unit's tokens pushed together).  The history moves by `tokens`: h_j <-
    // h'_{j + tokens} here (j + tokens < 3), the rest from the writer.  With choose, a user's keep code c reads the
    // history back by (saved rows - c) in (S_0 .. S_{saved rows - 1}, h0, h1, h2), and the checkpoints get S_k <-
    // h'_{k + 1} (S_k `saved_pages` k pages on): tokens - 1 rows (1 with one token), S_0 and S_1 from here, the rest
    // (the earlier tokens' inputs) from the writer
    constexpr uint32_t tokens = get_compile_time_arg_val(X + 5) < 1 ? 1 : get_compile_time_arg_val(X + 5);
    constexpr bool two = tokens > 1;
    constexpr uint32_t saved_pages = get_compile_time_arg_val(X + 6);
    constexpr uint32_t saved_rows = tokens > 1 ? tokens - 1 : 1;
    constexpr auto keep_args = TensorAccessorArgs<X + 7>();
    constexpr auto saved_args = TensorAccessorArgs<keep_args.next_compile_time_args_offset()>();
    // the stacked form (the program factory's `stack_tokens`): stack_g tokens a pass, stack_p passes a unit; token k's
    // users at rows (k - the pass's first) users .., each row's h0 h1 h2 its user's (h0, h1, h2, x_0 ..)[k .. k + 2]
    // (the chosen history from CB_HST, the earlier tokens' inputs widened to fp32 from CB_SEQ)
    constexpr uint32_t Y = saved_args.next_compile_time_args_offset();
    constexpr uint32_t stack_g = get_compile_time_arg_val(Y + 0);
    constexpr uint32_t stack_p = get_compile_time_arg_val(Y + 1);
    constexpr uint32_t cb_seq = get_compile_time_arg_val(Y + 2);
    constexpr uint32_t cb_hst = get_compile_time_arg_val(Y + 3);
    static_assert(stack_g == 0 || (users * tokens <= 32 && stack_g * users <= 16));
    // every token's rows in the first face of its users' rows (up to 16 rows: 8 users a row with two tokens): the later
    // tokens' rows taken from the first token's face once it has landed (no read of their own, no barrier in the loop)
    constexpr bool packed = two && users * tokens <= 16;

    const auto s_proj = TensorAccessor(proj_args, proj_addr);
    const auto s_fg = TensorAccessor(fg_args, fg_addr);
    const auto s_h0 = TensorAccessor(h0_args, h0_addr);
    const auto s_h1 = TensorAccessor(h1_args, h1_addr);
    const auto s_h2 = TensorAccessor(h2_args, h2_addr);
    const auto s_taps = TensorAccessor(taps_args, taps_addr);
    const auto s_gc = TensorAccessor(gc_args, gc_addr);
    const auto s_saved = TensorAccessor(saved_args, saved_addr);
    const auto s_h0_out = TensorAccessor(h0_args, h0_out_addr);
    const auto s_h1_out = TensorAccessor(h1_args, h1_out_addr);
    const auto s_saved_out = TensorAccessor(saved_args, saved_out_addr);

    const uint32_t t16 = get_local_cb_interface(cb_cur).fifo_page_size;  // bf16 tile
    const uint32_t t32 = get_local_cb_interface(cb_h0).fifo_page_size;   // fp32 tile
    constexpr uint32_t F16 = 512, F32 = 1024, R32 = 64, R16 = 32;      // face and face-row bytes

    Noc noc;
    CircularBuffer cur(cb_cur), h0(cb_h0), h1(cb_h1), h2(cb_h2), tap(cb_tap), f(cb_f), dt(cb_dt), rate(cb_rate),
        beta(cb_beta), ones(cb_ones), stg(cb_stg);
    const uint32_t stg_addr = stg.get_write_ptr();  // never pushed: staging only
    ones.reserve_back(1);
    fill_ones(ones.get_write_ptr());
    ones.push_back(1);
    volatile tt_l1_ptr uint32_t* keep = nullptr;
    if constexpr (choose) {
        const auto s_keep = TensorAccessor(keep_args, keep_addr);
        CircularBuffer kb(cb_keep);
        noc.async_read(s_keep, kb, keep_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        keep = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kb.get_write_ptr());
    }

    for (uint32_t u = start; u < start + count; ++u) {
        DeviceZoneScopedN("PR-UNIT");
        const uint32_t block = u % blocks;
        const uint32_t part = (u / blocks) % 4;
        const uint32_t h = u / blocks / 4;
        const uint32_t bt = block / 2;                                 // the tile row
        const uint32_t face32 = (block % 2) * 2 * F32;                 // fp32 faces
        const uint32_t rows = users - block * 16 < 16 ? users - block * 16 : 16;
        // half `half` of tile column `ct` of the block's rows of a bf16 [.., 2 B, ..] input (proj, fg), from row `off` on.
        // Not from a face's start, the rows' faces (at most 2) go whole into the scratch and the RISC-V moves the rows:
        // a 32-byte read from DRAM lands right only where source and destination sit alike in 64 bytes (row 1
        // into row 0 read wrong data)
        auto read16 = [&](const auto& src, CircularBuffer& dst, uint32_t tiles, uint32_t ct, uint32_t half, uint32_t at,
                          uint32_t off) {
            const uint32_t r0 = block * 16 + off;
            if (off % 16 == 0) {
                noc.async_read(
                    src, dst, F16, {.page_id = (r0 / 32) * tiles + ct, .offset_bytes = ((r0 % 32) / 16) * 2 * F16 + half * F16},
                    {.offset_bytes = at});
            } else {
                CircularBuffer scratch(cb_rows);
                const uint32_t f0 = r0 / 16, f1 = (r0 + rows - 1) / 16;
                for (uint32_t fi = f0; fi <= f1; ++fi) {
                    const uint32_t r = fi * 16;
                    noc.async_read(
                        src, scratch, F16, {.page_id = (r / 32) * tiles + ct, .offset_bytes = ((r % 32) / 16) * 2 * F16 + half * F16},
                        {.offset_bytes = (fi - f0) * F16});
                }
                noc.async_read_barrier();
                volatile tt_l1_ptr uint32_t* from = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());
                volatile tt_l1_ptr uint32_t* to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst.get_write_ptr() + at);
                for (uint32_t i = 0; i < rows; ++i) {
                    const uint32_t r = r0 + i;
                    const uint32_t s = ((r / 16 - f0) * F16 + (r % 16) * R16) / 4;
                    move_row32(reinterpret_cast<uint32_t>(from + s), reinterpret_cast<uint32_t>(to + i * (R16 / 4)));
                }
            }
        };

        if constexpr (stack_g != 0) {
            if (part < 3) {
                DeviceZoneScopedN("PR-STACK");
                const uint32_t cbase = part * heads * N + h * N;
                noc_async_write_barrier();  // the previous unit's history writes (from CB_HST)
                CircularBuffer seq(cb_seq), hst(cb_hst);
                const uint32_t seq_l1 = seq.get_write_ptr(), hst_l1 = hst.get_write_ptr();  // scratch, never pushed
                // CB_HST: per channel tile c a column of fp32 rows (2 tile rows): row m users + i = user i's
                // (h0, h1, h2, x_0 .. x_{tokens - 1})[m]; a pass's h_j is then the column's rows (k0 + j) users ..
                auto col = [&](uint32_t c, uint32_t half, uint32_t r) {
                    return hst_l1 + (2 * c + r / 32) * t32 + ((r % 32) / 16) * 2 * F32 + half * F32 + (r % 16) * R32;
                };
                tap.reserve_back(4 * N);
                for (uint32_t c = c_lo; c < c_hi; ++c) {
                    const uint32_t ct = cbase + c;
                    for (uint32_t half = 0; half < 2; ++half) {
                        // every token's input rows (rows k users + i of the tile row; past 16 rows faces 2 and 3)
                        noc.async_read(s_proj, seq, F16, {.page_id = ct, .offset_bytes = half * F16},
                                       {.offset_bytes = c * t16 + half * F16});
                        if constexpr (users * tokens > 16) {
                            noc.async_read(s_proj, seq, F16, {.page_id = ct, .offset_bytes = 2 * F16 + half * F16},
                                           {.offset_bytes = c * t16 + 2 * F16 + half * F16});
                        }
                        // those users' history into the column's rows 0 .. 3 users - 1 (users divides 16: no run
                        // crosses a face) and, to choose from, its checkpoints
                        noc_async_read(s_h0.get_noc_addr(ct, half * F32), col(c, half, 0), rows * R32);
                        noc_async_read(s_h1.get_noc_addr(ct, half * F32), col(c, half, rows), rows * R32);
                        noc_async_read(s_h2.get_noc_addr(ct, half * F32), col(c, half, 2 * rows), rows * R32);
                        if constexpr (choose) {
                            CircularBuffer sv(cb_saved);
                            for (uint32_t k = 0; k < saved_rows; ++k) {
                                noc.async_read(s_saved, sv, rows * R32,
                                               {.page_id = k * saved_pages + ct, .offset_bytes = half * F32},
                                               {.offset_bytes = (k * N + c) * t32 + half * F32});
                            }
                        }
                        for (uint32_t k = 0; k < 4; ++k) {
                            noc.async_read(s_taps, tap, R32, {.page_id = ct, .offset_bytes = half * F32 + k * R32},
                                           {.offset_bytes = (k * N + c) * t32 + half * F32});
                        }
                    }
                }
                noc.async_read_barrier();
                if constexpr (choose) {  // a rejected user's history back by `back` in (S_0 .., h0, h1, h2)
                    const uint32_t sv_l1 = CircularBuffer(cb_saved).get_write_ptr();
                    for (uint32_t i = 0; i < rows; ++i) {
                        const uint32_t code = keep[i];
                        if (code >= saved_rows) {
                            continue;
                        }
                        const uint32_t back = saved_rows - code;
                        for (uint32_t c = c_lo; c < c_hi; ++c) {
                            for (uint32_t half = 0; half < 2; ++half) {
                                for (uint32_t j = 3; j-- > 0;) {
                                    const uint32_t from =
                                        j >= back ? col(c, half, (j - back) * rows + i)
                                                  : sv_l1 + (saved_rows + j - back) * N * t32 + c * t32 + half * F32 + i * R32;
                                    move_row64(from, col(c, half, j * rows + i));
                                }
                            }
                        }
                    }
                }
                // the inputs widened to fp32 after the history: row (3 + m) users + i <- x_m of user i
                for (uint32_t c = c_lo; c < c_hi; ++c) {
                    for (uint32_t half = 0; half < 2; ++half) {
                        for (uint32_t q = 0; q < tokens * rows; ++q) {
                            widen_row(seq_l1 + c * t16 + (q / 16) * 2 * F16 + half * F16 + (q % 16) * R16,
                                      col(c, half, 3 * rows + q));
                        }
                    }
                }
                tap.push_back(4 * N);
                // the shifted history and the checkpoints from the chosen rows, as the unstacked form
                for (uint32_t c = c_lo; c < c_hi; ++c) {
                    const uint32_t page = cbase + c;
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t off = half * F32;
                        if constexpr (tokens == 2) {
                            noc_async_write(col(c, half, 2 * rows), s_h0_out.get_noc_addr(page, off), rows * R32);
                        }
                        if constexpr (choose) {
                            noc_async_write(col(c, half, rows), s_saved_out.get_noc_addr(page, off), rows * R32);
                            if constexpr (saved_rows > 1) {
                                noc_async_write(col(c, half, 2 * rows), s_saved_out.get_noc_addr(saved_pages + page, off), rows * R32);
                            }
                        }
                    }
                }
                for (uint32_t g = 0; g < stack_p; ++g) {
                    const uint32_t k0 = g * stack_g;
                    const uint32_t kn = tokens - k0 < stack_g ? tokens - k0 : stack_g;
                    const uint32_t n = kn * rows;  // the pass's rows (a run of the column: 16-row pieces at most)
                    cur.reserve_back(N);
                    h0.reserve_back(N);
                    h1.reserve_back(N);
                    h2.reserve_back(N);
                    const uint32_t hp[3] = {h0.get_write_ptr(), h1.get_write_ptr(), h2.get_write_ptr()};
                    for (uint32_t c = c_lo; c < c_hi; ++c) {
                        for (uint32_t half = 0; half < 2; ++half) {
                            // the pass's inputs: rows k0 users .. of proj (a face's start)
                            read16(s_proj, cur, proj_tiles, cbase + c, half, c * t16 + half * F16, k0 * rows);
                            for (uint32_t j = 0; j < 3; ++j) {  // h_j: the column's rows (k0 + j) users .. + n
                                const uint32_t r0 = (k0 + j) * rows;
                                for (uint32_t v = 0; v < n;) {  // (pieces within a face of the column)
                                    const uint32_t r = r0 + v, len = n - v < 16 - r % 16 ? n - v : 16 - r % 16;
                                    noc_async_read(get_noc_addr(col(c, half, r)), hp[j] + c * t32 + half * F32 + v * R32,
                                                   len * R32);
                                    v += len;
                                }
                            }
                        }
                    }
                    noc.async_read_barrier();
                    cur.push_back(N);
                    h0.push_back(N);
                    h1.push_back(N);
                    h2.push_back(N);
                }
            } else {
                f.reserve_back(N);
                dt.reserve_back(N);
                rate.reserve_back(N);
                for (uint32_t c = 0; c < N; ++c) {
                    const uint32_t ct = h * N + c;
                    for (uint32_t half = 0; half < 2; ++half) {
                        noc.async_read(s_gc, dt, R32, {.page_id = ct, .offset_bytes = half * F32},
                                       {.offset_bytes = c * t32 + half * F32});
                        noc.async_read(s_gc, rate, R32, {.page_id = ct, .offset_bytes = R32 + half * F32},
                                       {.offset_bytes = c * t32 + half * F32});
                    }
                }
                for (uint32_t g = 0; g < stack_p; ++g) {
                    const uint32_t k0 = g * stack_g;
                    const uint32_t kn = tokens - k0 < stack_g ? tokens - k0 : stack_g;
                    const uint32_t n = kn * rows;
                    if (g > 0) {
                        f.reserve_back(N);
                    }
                    beta.reserve_back(1);
                    for (uint32_t c = 0; c < N; ++c) {  // the pass's rows from row k0 users: a face's start
                        for (uint32_t half = 0; half < 2; ++half) {
                            read16(s_fg, f, fg_tiles, h * N + c, half, c * t16 + half * F16, k0 * rows);
                        }
                    }
                    read16(s_proj, stg, proj_tiles, 3 * heads * N, 0, 0, k0 * rows);
                    noc.async_read_barrier();
                    volatile tt_l1_ptr uint16_t* bs = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stg_addr);
                    volatile tt_l1_ptr uint32_t* bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(beta.get_write_ptr());
                    for (uint32_t v = 0; v < n; ++v) {
                        bp[v * 16] = static_cast<uint32_t>(bs[v * 16 + h]) << 16;  // (v, 0)
                    }
                    if (g == 0) {
                        dt.push_back(N);
                        rate.push_back(N);
                    }
                    f.push_back(N);
                    beta.push_back(1);
                }
            }
            continue;
        }
        if (part < 3) {
            const uint32_t cbase = part * heads * N + h * N;
            noc_async_write_barrier();  // the previous unit's history writes (from these CBs' other half)
            {
            DeviceZoneScopedN("PR-H");  // issuing the reads, to the end of the unit
            cur.reserve_back(tokens * N);
            h0.reserve_back(N);
            h1.reserve_back(N);
            h2.reserve_back(N);
            tap.reserve_back(4 * N);
            for (uint32_t c = c_lo; c < c_hi; ++c) {
                const uint32_t ct = cbase + c;
                for (uint32_t half = 0; half < 2; ++half) {
                    read16(s_proj, cur, proj_tiles, ct, half, c * t16 + half * F16, 0);
                    if constexpr (two && !packed) {
                        for (uint32_t k = 1; k < tokens; ++k) {
                            read16(s_proj, cur, proj_tiles, ct, half, (k * N + c) * t16 + half * F16, k * users);
                        }
                    }
                    noc.async_read(s_h0, h0, rows * R32, {.page_id = bt * hist_tiles + ct, .offset_bytes = face32 + half * F32},
                                   {.offset_bytes = c * t32 + half * F32});
                    noc.async_read(s_h1, h1, rows * R32, {.page_id = bt * hist_tiles + ct, .offset_bytes = face32 + half * F32},
                                   {.offset_bytes = c * t32 + half * F32});
                    noc.async_read(s_h2, h2, rows * R32, {.page_id = bt * hist_tiles + ct, .offset_bytes = face32 + half * F32},
                                   {.offset_bytes = c * t32 + half * F32});
                    if constexpr (choose) {  // the checkpoints' rows too: the rejected users' are shifted in below
                        CircularBuffer sv(cb_saved);
                        for (uint32_t k = 0; k < saved_rows; ++k) {
                            noc.async_read(
                                s_saved, sv, rows * R32,
                                {.page_id = k * saved_pages + bt * hist_tiles + ct, .offset_bytes = face32 + half * F32},
                                {.offset_bytes = (k * N + c) * t32 + half * F32});
                        }
                    }
                    // tap k is row k of the taps' first tile row: each row straight into row 0 of its tile (the
                    // compute broadcasts row 0 only)
                    for (uint32_t k = 0; k < 4; ++k) {
                        noc.async_read(s_taps, tap, R32, {.page_id = ct, .offset_bytes = half * F32 + k * R32},
                                       {.offset_bytes = (k * N + c) * t32 + half * F32});
                    }
                }
            }
            {
            DeviceZoneScopedN("PR-HB");  // the reads issued: their barrier, to the end of the unit
            noc.async_read_barrier();
            {
            DeviceZoneScopedN("PR-AFTER");  // the reads done: the pushes, the history writes issued
            if constexpr (packed) {  // token k's rows k users .. of the first token's face rows 0 ..
                for (uint32_t k = 1; k < tokens; ++k) {
                    copy_rows16(cur.get_write_ptr(), cur.get_write_ptr() + k * N * t16, k * users, users, t16);
                }
            }
            // the write pointers before the pushes: the history writes below read these pages
            const uint32_t h1_l1 = h1.get_write_ptr(), h2_l1 = h2.get_write_ptr();
            if constexpr (choose) {  // a rejected user's history back by `back` in (S_0 .., h0, h1, h2), moved in L1
                const uint32_t h0_l1 = h0.get_write_ptr(), sv_l1 = CircularBuffer(cb_saved).get_write_ptr();
                const uint32_t hl1[3] = {h0_l1, h1_l1, h2_l1};
                auto move = [](uint32_t from, uint32_t to) { move_row64(from, to); };
                for (uint32_t i = 0; i < rows; ++i) {
                    const uint32_t code = keep[block * 16 + i];
                    if (code >= saved_rows) {
                        continue;
                    }
                    const uint32_t back = saved_rows - code;
                    for (uint32_t c = c_lo; c < c_hi; ++c) {
                        for (uint32_t half = 0; half < 2; ++half) {
                            const uint32_t at = c * t32 + half * F32 + i * R32;
                            for (uint32_t j = 3; j-- > 0;) {  // row j <- h_{j - back}, or S_{saved rows + j - back}
                                const uint32_t from =
                                    j >= back ? hl1[j - back] + at : sv_l1 + (saved_rows + j - back) * N * t32 + at;
                                move(from, hl1[j] + at);
                            }
                        }
                    }
                }
            }
            cur.push_back(tokens * N);
            h0.push_back(N);
            h1.push_back(N);
            h2.push_back(N);
            tap.push_back(4 * N);
            // the shifted history, from the pages just filled (the compute only reads them; the next reserve of
            // this half comes after the barrier above): by one (h0 <- h1, h1 <- h2; the writer h2 <- the input), two
            // (h0 <- h2; the writer h1, h2 <- the inputs) or three (the writer all); the checkpoints S_k <- h'_{k + 1}
            for (uint32_t c = c_lo; c < c_hi; ++c) {
                const uint32_t page = bt * hist_tiles + cbase + c;
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t off = face32 + half * F32;
                    const uint32_t l1 = c * t32 + half * F32;
                    if constexpr (tokens == 2) {
                        noc_async_write(h2_l1 + l1, s_h0_out.get_noc_addr(page, off), rows * R32);
                    } else if constexpr (tokens == 1) {
                        noc_async_write(h1_l1 + l1, s_h0_out.get_noc_addr(page, off), rows * R32);
                        noc_async_write(h2_l1 + l1, s_h1_out.get_noc_addr(page, off), rows * R32);
                    }
                    if constexpr (choose) {
                        noc_async_write(h1_l1 + l1, s_saved_out.get_noc_addr(page, off), rows * R32);
                        if constexpr (saved_rows > 1) {
                            noc_async_write(h2_l1 + l1, s_saved_out.get_noc_addr(saved_pages + page, off), rows * R32);
                        }
                    }
                }
            }
            }}}  // the zones' blocks
        } else {
            f.reserve_back(tokens * N);
            dt.reserve_back(N);
            rate.reserve_back(N);
            beta.reserve_back(tokens);
            for (uint32_t c = 0; c < N; ++c) {
                const uint32_t ct = h * N + c;
                for (uint32_t half = 0; half < 2; ++half) {
                    read16(s_fg, f, fg_tiles, ct, half, c * t16 + half * F16, 0);
                    if constexpr (two && !packed) {
                        for (uint32_t k = 1; k < tokens; ++k) {
                            read16(s_fg, f, fg_tiles, ct, half, (k * N + c) * t16 + half * F16, k * users);
                        }
                    }
                    noc.async_read(s_gc, dt, R32, {.page_id = ct, .offset_bytes = half * F32},
                                   {.offset_bytes = c * t32 + half * F32});
                    noc.async_read(s_gc, rate, R32, {.page_id = ct, .offset_bytes = R32 + half * F32},
                                   {.offset_bytes = c * t32 + half * F32});
                }
            }
            // the beta logits sit after q | k | v: tile 3 H N of the row, column h (< 16) of face 0 / 2
            read16(s_proj, stg, proj_tiles, 3 * heads * N, 0, 0, 0);
            noc.async_read_barrier();
            volatile tt_l1_ptr uint16_t* bs = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stg_addr);
            volatile tt_l1_ptr uint32_t* bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(beta.get_write_ptr());
            for (uint32_t i = 0; i < rows; ++i) {
                bp[i * 16] = static_cast<uint32_t>(bs[i * 16 + h]) << 16;  // (i, 0)
            }
            if constexpr (packed) {  // the later tokens' fg rows and logits from the first token's faces
                for (uint32_t k = 1; k < tokens; ++k) {
                    copy_rows16(f.get_write_ptr(), f.get_write_ptr() + k * N * t16, k * users, users, t16);
                    volatile tt_l1_ptr uint32_t* bpk =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(beta.get_write_ptr() + k * t32);
                    for (uint32_t i = 0; i < rows; ++i) {
                        bpk[i * 16] = static_cast<uint32_t>(bs[(k * users + i) * 16 + h]) << 16;
                    }
                }
            }
            for (uint32_t k = 1; k < tokens && !packed; ++k) {  // the later tokens' logits, through the same staging
                read16(s_proj, stg, proj_tiles, 3 * heads * N, 0, 0, k * users);
                noc.async_read_barrier();
                volatile tt_l1_ptr uint32_t* bpk =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(beta.get_write_ptr() + k * t32);
                for (uint32_t i = 0; i < rows; ++i) {
                    bpk[i * 16] = static_cast<uint32_t>(bs[i * 16 + h]) << 16;
                }
            }
            f.push_back(tokens * N);
            dt.push_back(N);
            rate.push_back(N);
            beta.push_back(tokens);
        }
    }
    noc_async_write_barrier();
}

}  // namespace kda_prepare_reader
