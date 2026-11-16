// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// split_channels (KDA_PREP_SPLIT): a q / k / v unit's channel tiles c_lo .. only (runtime args 12, 13), and for q and
// k each pass the half's sum of squares (CB_SQP) into the other half's core's CB_SQX (the same address there; 2 slots,
// a pass in turn) with a count on its semaphore 0, then this core's from it (the count reaching the pass) pushed
#ifndef KDA_PREP_SPLIT
#define KDA_PREP_SPLIT 0
#endif

namespace kda_prepare_writer {
// Per unit (part, head h, block of 16 users): row i of the part's 4 output tiles into pair p = (16 block + i) H + h's
// row of q / k / v / the decay (64-byte pieces: the other pairs of p's 32-row tile row may be on other cores); for the
// gates also sigmoid(beta) at (i, 0) into column 0 of p's row of beta.  For q / k / v first the conv input in fp32
// (packed by the compute) into history 2: these channels' rows of the block, which no other unit touches.

// The kernel, its compile-time args from CT on and runtime args from RT on (decode_prepare: 0, 0).
template <uint32_t CT, uint32_t RT>
FORCE_INLINE void run() {
    uint32_t a = RT;
    const uint32_t q_addr = get_arg_val<uint32_t>(a++);
    const uint32_t k_addr = get_arg_val<uint32_t>(a++);
    const uint32_t v_addr = get_arg_val<uint32_t>(a++);
    const uint32_t e_addr = get_arg_val<uint32_t>(a++);
    const uint32_t beta_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h2_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t count = get_arg_val<uint32_t>(a++);
    // `tokens` a user (MTP's verify step): token k's outputs at pair p + k second, its input into history row
    // 3 - tokens + k (the reader moves the rest)
    const uint32_t h1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t second = get_arg_val<uint32_t>(a++);
    const uint32_t h0_addr = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_out = get_compile_time_arg_val(CT + 0);
    constexpr uint32_t cb_bout = get_compile_time_arg_val(CT + 1);
    constexpr uint32_t cb_wscr = get_compile_time_arg_val(CT + 2);
    constexpr uint32_t heads = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t users = get_compile_time_arg_val(CT + 4);
    constexpr uint32_t cb_curh = get_compile_time_arg_val(CT + 5);
    constexpr uint32_t hist_tiles = get_compile_time_arg_val(CT + 6);
    constexpr auto q_args = TensorAccessorArgs<CT + 7>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto h2_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();
    constexpr uint32_t tokens_ct = get_compile_time_arg_val(h2_args.next_compile_time_args_offset());
    constexpr uint32_t tokens = tokens_ct < 1 ? 1 : tokens_ct;
    // with more than 3 tokens and the checkpoint (`saved`, a verify step that may come back): token k's input, k <
    // tokens - 3 (not in the new history), into checkpoint row k + 2 (S_{k+2} <- h'_{k+3}, `saved_pages` a row)
    constexpr uint32_t W2 = h2_args.next_compile_time_args_offset() + 1;
    constexpr bool keeps = get_compile_time_arg_val(W2) != 0;
    constexpr uint32_t saved_pages = get_compile_time_arg_val(W2 + 1);
    constexpr auto saved_args = TensorAccessorArgs<W2 + 2>();
    const uint32_t saved_addr = get_arg_val<uint32_t>(RT + 11);
    const auto s_saved = TensorAccessor(saved_args, saved_addr);
    // the stacked form (the reader's): stack_p passes a unit of stack_g tokens, token k's users at rows (k - the
    // pass's first) users ..
    constexpr uint32_t stack_g = get_compile_time_arg_val(saved_args.next_compile_time_args_offset());
    constexpr uint32_t stack_p = get_compile_time_arg_val(saved_args.next_compile_time_args_offset() + 1);
    // the outputs by pair (row p tokens + k) in place of token by token (row p + k B H)
    constexpr bool pair_major = get_compile_time_arg_val(saved_args.next_compile_time_args_offset() + 2) != 0;
    constexpr uint32_t cb_sqp = get_compile_time_arg_val(saved_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t cb_sqx = get_compile_time_arg_val(saved_args.next_compile_time_args_offset() + 4);
    constexpr bool split = KDA_PREP_SPLIT != 0;
    const uint32_t c_lo = split ? get_arg_val<uint32_t>(RT + 12) : 0;  // q / k / v's channel tiles (the gates' all)
    const uint32_t c_hi = split ? c_lo + get_arg_val<uint32_t>(RT + 13) : 4;
    const uint32_t px = split ? get_arg_val<uint32_t>(RT + 14) : 0;  // the other half's core (q / k / v)
    const uint32_t py = split ? get_arg_val<uint32_t>(RT + 15) : 0;
    auto pair_row = [&](uint32_t pair, uint32_t k) { return pair_major ? pair * tokens + k : pair + k * second; };
    constexpr uint32_t blocks = (users + 15) / 16;

    const auto s_q = TensorAccessor(q_args, q_addr);
    const auto s_k = TensorAccessor(k_args, k_addr);
    const auto s_v = TensorAccessor(v_args, v_addr);
    const auto s_e = TensorAccessor(e_args, e_addr);
    const auto s_beta = TensorAccessor(beta_args, beta_addr);
    const auto s_h2 = TensorAccessor(h2_args, h2_addr);
    const auto s_h1 = TensorAccessor(h2_args, h1_addr);  // (h2's args: alike)
    const auto s_h0 = TensorAccessor(h2_args, h0_addr);

    const uint32_t t32 = get_local_cb_interface(cb_out).fifo_page_size;
    constexpr uint32_t F32 = 1024, R32 = 64, N = 4;

    Noc noc;
    CircularBuffer out(cb_out), bout(cb_bout), wscr(cb_wscr), curh(cb_curh);
    // never pushed: the beta rows, BETA_RING of them (64 bytes each, in the fp32 tile), so that a pass's rows go out
    // with one barrier (the scratch is reused the next pass): each row is its value in column 0 and zeros; the last
    // slot (split) holds the exchange's flag value
    const uint32_t scr = wscr.get_write_ptr();
    constexpr uint32_t BETA_SLOTS = 64, BETA_RING = BETA_SLOTS - 1;
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    for (uint32_t i = 0; i < 16 * BETA_SLOTS; ++i) {
        row[i] = 0;
    }
    // row v's beta (bp[v * 16]: its column 0) to pair row p's place in beta, from slot v % BETA_RING; the barrier when
    // the slots wrap (and after the last, by the caller)
    auto beta_row = [&](volatile tt_l1_ptr uint32_t* bp, uint32_t v, uint32_t p) {
        const uint32_t slot = v % BETA_RING;
        if (slot == 0 && v > 0) {
            noc_async_write_barrier();
        }
        const uint32_t r = p % 32;
        row[slot * 16] = bp[v * 16];  // (v, 0)
        noc_async_write(scr + slot * R32, s_beta.get_noc_addr(p / 32, (r / 16) * 2 * F32 + (r % 16) * R32), R32);
    };

    // the half's sum of squares to the other half's core, then the other's here (q and k, split; `n`: the pass).  The
    // other's semaphore is set to the pass count by a 4-byte write (no atomic: an atomic's response may come after the
    // kernel's end, into the next program's count on that NoC), from the scratch's last
    // slot (rewritten only after the next pass's write barrier, which covers this write)
    auto exchange = [&](uint32_t n) {
        CircularBuffer sqp(cb_sqp), sqx(cb_sqx);
        const auto& fx = get_local_cb_interface(cb_sqx);
        const uint32_t slot = fx.fifo_limit - fx.fifo_size + (n % 2) * t32;  // (the same on every core)
        const uint32_t sem = get_semaphore(0);
        sqp.wait_front(1);
        noc_async_write(sqp.get_read_ptr(), get_noc_addr(px, py, slot), 2 * F32);  // (faces 0 and 1: the block's rows)
        noc_async_write_barrier();
        const uint32_t flag = scr + BETA_RING * R32;  // (16-byte aligned, as the semaphore)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag) = n + 1;
        noc_semaphore_set_remote(flag, get_noc_addr(px, py, sem));
        sqp.pop_front(1);
        sqx.reserve_back(1);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), n + 1);
        sqx.push_back(1);
    };

    uint32_t part_tiles_lo = 0, part_tiles_end = N;  // the unit's channel tiles (set a unit)
    auto write_rows = [&](const auto& dst, uint32_t i, uint32_t p) {
        const uint32_t r = p % 32;
        const uint32_t lo = (r / 16) * 2 * F32 + (r % 16) * R32;
        const uint32_t src = out.get_read_ptr();
        const uint32_t ce = part_tiles_end;
        for (uint32_t c = part_tiles_lo; c < ce; ++c) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_write(
                    src + c * t32 + half * F32 + i * R32, dst.get_noc_addr((p / 32) * N + c, lo + half * F32), R32);
            }
        }
    };

    for (uint32_t u = start; u < start + count; ++u) {
        const uint32_t block = u % blocks;
        const uint32_t part = (u / blocks) % 4;
        const uint32_t h = u / blocks / 4;
        const uint32_t rows = users - block * 16 < 16 ? users - block * 16 : 16;
        part_tiles_lo = part < 3 ? c_lo : 0;
        part_tiles_end = part < 3 ? c_hi : N;
        const bool swap = split && part < 2;  // (q, k: the norm's halves)
        if constexpr (stack_g != 0) {  // (one block: users <= 8)
            for (uint32_t g = 0; g < stack_p; ++g) {
                const uint32_t k0 = g * stack_g;
                const uint32_t kn = tokens - k0 < stack_g ? tokens - k0 : stack_g;
                if (part < 3) {  // each token's input in fp32 into its history slot or checkpoint row
                    curh.wait_front(N);
                    const uint32_t src = curh.get_read_ptr();
                    for (uint32_t kk = 0; kk < kn; ++kk) {
                        const uint32_t k = k0 + kk;
                        const bool slot_row = k + 3 >= tokens;
                        if (!slot_row && !keeps) {
                            continue;
                        }
                        const uint32_t slot = k + 3 - tokens;
                        const auto& hist = slot == 0 ? s_h0 : (slot == 1 ? s_h1 : s_h2);
                        for (uint32_t c = c_lo; c < c_hi; ++c) {
                            const uint32_t page = part * heads * N + h * N + c;
                            for (uint32_t half = 0; half < 2; ++half) {
                                const uint32_t from = src + c * t32 + half * F32 + kk * rows * R32;
                                if (slot_row) {
                                    noc_async_write(from, hist.get_noc_addr(page, half * F32), rows * R32);
                                } else {
                                    noc_async_write(
                                        from, s_saved.get_noc_addr((k + 2) * saved_pages + page, half * F32), rows * R32);
                                }
                            }
                        }
                    }
                }
                if (swap) {
                    exchange(g);
                }
                out.wait_front(N);
                for (uint32_t v = 0; v < kn * rows; ++v) {
                    const uint32_t p = pair_row((v % rows) * heads + h, k0 + v / rows);
                    if (part == 0) {
                        write_rows(s_q, v, p);
                    } else if (part == 1) {
                        write_rows(s_k, v, p);
                    } else if (part == 2) {
                        write_rows(s_v, v, p);
                    } else {
                        write_rows(s_e, v, p);
                    }
                }
                noc_async_write_barrier();
                out.pop_front(N);
                if (part < 3) {
                    curh.pop_front(N);
                } else {
                    bout.wait_front(1);
                    volatile tt_l1_ptr uint32_t* bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bout.get_read_ptr());
                    for (uint32_t v = 0; v < kn * rows; ++v) {
                        beta_row(bp, v, pair_row((v % rows) * heads + h, k0 + v / rows));
                    }
                    noc_async_write_barrier();  // (the slots are reused by the next pass)
                    bout.pop_front(1);
                }
            }
            continue;
        }
      for (uint32_t pass = 0; pass < tokens; ++pass) {
        if (part < 3) {
            curh.wait_front(N);
            const uint32_t src = curh.get_read_ptr();
            if (pass + 3 >= tokens) {  // history slot pass + 3 - tokens
                const uint32_t slot = pass + 3 - tokens;
                const auto& hist = slot == 0 ? s_h0 : (slot == 1 ? s_h1 : s_h2);
                for (uint32_t c = c_lo; c < c_hi; ++c) {
                    const uint32_t page = (block / 2) * hist_tiles + part * heads * N + h * N + c;
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t off = (block % 2) * 2 * F32 + half * F32;
                        noc_async_write(src + c * t32 + half * F32, hist.get_noc_addr(page, off), rows * R32);
                    }
                }
            } else if constexpr (keeps) {  // checkpoint row pass + 2
                for (uint32_t c = c_lo; c < c_hi; ++c) {
                    const uint32_t page = (pass + 2) * saved_pages + (block / 2) * hist_tiles + part * heads * N + h * N + c;
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t off = (block % 2) * 2 * F32 + half * F32;
                        noc_async_write(src + c * t32 + half * F32, s_saved.get_noc_addr(page, off), rows * R32);
                    }
                }
            }
        }
        if (swap) {
            exchange(pass);
        }
        {
            DeviceZoneScopedN("PW-WAIT");
            out.wait_front(N);
        }
        DeviceZoneScopedN("PW-OUT");  // (to the end of the pass)
        for (uint32_t i = 0; i < rows; ++i) {
            const uint32_t p = pair_row((block * 16 + i) * heads + h, pass);
            if (part == 0) {
                write_rows(s_q, i, p);
            } else if (part == 1) {
                write_rows(s_k, i, p);
            } else if (part == 2) {
                write_rows(s_v, i, p);
            } else {
                write_rows(s_e, i, p);
            }
        }
        noc_async_write_barrier();
        out.pop_front(N);
        if (part < 3) {
            curh.pop_front(N);
        }
        if (part == 3) {
            bout.wait_front(1);
            volatile tt_l1_ptr uint32_t* bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bout.get_read_ptr());
            for (uint32_t i = 0; i < rows; ++i) {
                beta_row(bp, i, pair_row((block * 16 + i) * heads + h, pass));
            }
            noc_async_write_barrier();  // (the slots are reused by the next pass)
            bout.pop_front(1);
        }
      }  // the passes
    }
}

}  // namespace kda_prepare_writer
