// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "mp_common.hpp"

// The prefill MoE's plan, step 2 by pushes (moe_prefill.py `plan`; it replaced each expert core's scan of every route
// row, 0.49 -> 0.13 ms at 4,096 tokens).  From the counts of step 1 (cnt: P pages, each token core's counts per
// expert):
//   role 0 (a token core c, its tile rows tr0 .. tr1 - 1): its pairs of expert e are rows pos0[e] .. pos0[e] +
//          cnt[c][e] - 1 of e's segment (pos0[e]: the pairs of e on the cores before it), in token order; it writes
//          its tokens' pair rows (tpair: off[e] + that row), stages each expert's run of tokens and weights and writes
//          the two runs into expert core e's arrays (L1) at that row, then counts itself in on every expert core
//          (S_PUSHED); core 0 also writes the seg page;
//   role 1 (an expert core e): once all P token cores are in, the padding (SENT, 0) after its n[e] rows, and its
//          segment off[e] .. off[e] + n[e] rounded up to 32 of ptok and pw, a page (16 rows) at a time.
// The arrays are at the same L1 address on every core (the same CB), each run staged at that address's offset
// within 16 bytes (an L1 write's source and destination agree there).  With a mask (as mp_plan_count.cpp's), a token
// whose mask is 0 has no pairs.  `wait_counts` (step 1 in the same program, mp_plan_count.cpp's `signal`): the counts
// read once they are all written.  The gathering core (runtime arg after the expert cores' list: 1) waits for the P
// token cores' increments of S_COUNTED, then sets S_GO and writes it to the other push cores (n, then their NoC
// x | y << 16: plain 4-byte writes from the same semaphore address, no atomics); the others wait for S_GO.  Every
// wait bounded (a lost signal ends the op with wrong tables rather than a hang).
void kernel_main() {
    using namespace moe_prefill;
    uint32_t a = 0;
    const uint32_t route_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(a++);
    const uint32_t seg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ptok_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pw_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tpair_addr = get_arg_val<uint32_t>(a++);
    const uint32_t role = get_arg_val<uint32_t>(a++);
    const uint32_t who = get_arg_val<uint32_t>(a++);  // role 0: the token core c; role 1: the expert e
    const uint32_t tr0 = get_arg_val<uint32_t>(a++);
    const uint32_t tr1 = get_arg_val<uint32_t>(a++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(a++);
    const uint32_t xy0 = a;  // then each expert core's NoC x | y << 16

    constexpr uint32_t n_tiles = get_compile_time_arg_val(0);  // the route's tile rows (N / 32)
    constexpr uint32_t cores = get_compile_time_arg_val(1);    // P (the token cores)
    constexpr uint32_t local = get_compile_time_arg_val(2);
    constexpr uint32_t cb = get_compile_time_arg_val(3);
    constexpr uint32_t stage_words = get_compile_time_arg_val(4);  // a token core's staging per array
    constexpr auto r_args = TensorAccessorArgs<5>();
    constexpr auto c_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t has_mask = get_compile_time_arg_val(p_args.next_compile_time_args_offset());
    constexpr auto m_args = TensorAccessorArgs<p_args.next_compile_time_args_offset() + 1>();
    // (the counts summed in one pass of plain (cached) loads, a token core's own prefix taken on the way: the op -0.015
    // ms at 1,024 tokens a column; tpair a page a token tile row, 32 tokens' 64 bytes, 2 KB, in place of a page a
    // token.  The forms before, GLM53_MOE_PLAN_ONE_PASS=0 and GLM53_MOE_PREFILL_TPAIR_ROWS=0, were removed)
    constexpr bool wait_counts = get_compile_time_arg_val(m_args.next_compile_time_args_offset()) != 0;
    constexpr uint32_t S_COUNTED = get_compile_time_arg_val(m_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t S_GO = get_compile_time_arg_val(m_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t CNT_PAGE = 128 * 4;
    constexpr uint32_t PAGE = ROW_WORDS * 4;
    constexpr uint32_t ARR = (n_tiles * 32 + 32) * 4;  // an expert core's array: every token at most once, + padding
    constexpr uint32_t S_PUSHED = 0;

    const auto s_route = TensorAccessor(r_args, route_addr, 4 * F);
    const auto s_cnt = TensorAccessor(c_args, cnt_addr, CNT_PAGE);
    const auto s_seg = TensorAccessor(s_args, seg_addr, SEG_WORDS * 4);
    const auto s_ptok = TensorAccessor(t_args, ptok_addr, PAGE);
    const auto s_pw = TensorAccessor(w_args, pw_addr, PAGE);
    const auto s_tpair = TensorAccessor(p_args, tpair_addr, 32 * PAGE);
    const auto s_mask = TensorAccessor(m_args, mask_addr, 2048);

    // L1 (cb): the counts (P pages), the seg page, two route tiles, then (64-byte aligned) the expert arrays (tokens,
    // weights) on an expert core, or a token core's buffers
    const uint32_t cnt_l1 = get_write_ptr(cb);
    const uint32_t seg_l1 = cnt_l1 + cores * CNT_PAGE;
    const uint32_t tile_l1 = seg_l1 + SEG_WORDS * 4;
    const uint32_t buf_l1 = (tile_l1 + 2 * 4 * F + 63) & ~63u;
    const uint32_t tok_arr = buf_l1, w_arr = buf_l1 + ARR;
    if constexpr (wait_counts) {
        const uint32_t go_addr = get_semaphore(S_GO);
        volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_addr);
        const uint32_t extra = xy0 + local;  // (after the expert cores' list)
        if (get_arg_val<uint32_t>(extra) != 0) {  // the gathering core
            volatile tt_l1_ptr uint32_t* counted =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(S_COUNTED));
            for (uint32_t spin = 0; *counted < cores && spin < (1u << 27); ++spin) {
                invalidate_l1_cache();
            }
            *go = 1;
            const uint32_t n = get_arg_val<uint32_t>(extra + 1);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t xy = get_arg_val<uint32_t>(extra + 2 + i);
                noc_semaphore_set_remote(go_addr, get_noc_addr(xy & 0xFFFF, xy >> 16, go_addr));
            }
            noc_async_write_barrier();
        } else {
            for (uint32_t spin = 0; *go < 1 && spin < (1u << 27); ++spin) {
                invalidate_l1_cache();
            }
        }
    }
    for (uint32_t i = 0; i < cores; ++i) {
        noc_async_read(s_cnt.get_noc_addr(i), cnt_l1 + i * CNT_PAGE, CNT_PAGE);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_l1);
    volatile tt_l1_ptr uint32_t* seg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seg_l1);
    for (uint32_t i = 0; i < SEG_WORDS; ++i) {
        seg[i] = 0;
    }
    uint32_t total = 0;
    uint32_t before[local];  // a token core's pairs of each expert on the cores before it
    {
        invalidate_l1_cache();
        const uint32_t* cn = reinterpret_cast<const uint32_t*>(cnt_l1);
        const uint32_t mine = role == 0 ? who : cores;
        for (uint32_t e = 0; e < local; ++e) {
            uint32_t n = 0, b = 0;
#pragma GCC unroll 8
            for (uint32_t i = 0; i < cores; ++i) {
                b = i == mine ? n : b;
                n += cn[i * 128 + e];
            }
            before[e] = b;
            seg[e] = total;
            seg[SEG_COUNT + e] = n;
            total += (n + 31) / 32 * 32;
        }
    }
    seg[local] = total;
    const uint32_t pushed_addr = get_semaphore(S_PUSHED);

    if (role == 0) {
        const uint32_t c = who;
        // per expert: pos0 (its first row of e), the staged run's start (word index, offset as pos0 within 16
        // bytes) and the pairs staged so far
        volatile tt_l1_ptr uint32_t* pos0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_l1);
        volatile tt_l1_ptr uint32_t* start = pos0 + 128;
        volatile tt_l1_ptr uint32_t* got = pos0 + 256;
        const uint32_t ring = buf_l1 + 3 * 512;  // 64 tpair pages
        const uint32_t mask_l1 = ring + 64 * PAGE;  // the tile row's mask tile (bf16)
        volatile tt_l1_ptr uint16_t* mask = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_l1);
        const uint32_t tok_st = mask_l1 + 2048;
        const uint32_t w_st = tok_st + stage_words * 4;
        volatile tt_l1_ptr uint32_t* tst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_st);
        volatile tt_l1_ptr uint32_t* wst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_st);
        uint32_t w = 0;
        for (uint32_t e = 0; e < local; ++e) {
            const uint32_t p = before[e];
            pos0[e] = p;
            w = (w + 3) & ~3u;
            w += p % 4;
            start[e] = w;
            got[e] = 0;
            w += cnt[c * 128 + e];
        }

        uint32_t k_tok = 0;
        if (tr0 < tr1) {
            noc_async_read(s_route.get_noc_addr(tr0), tile_l1, 4 * F);
        }
        for (uint32_t tr = tr0; tr < tr1; ++tr) {
            noc_async_read_barrier();
            const uint32_t cur = tile_l1 + ((tr - tr0) % 2) * 4 * F;
            if constexpr (has_mask) {
                noc_async_read(s_mask.get_noc_addr(tr), mask_l1, 2048);
                noc_async_read_barrier();
            }
            if (tr + 1 < tr1) {
                noc_async_read(s_route.get_noc_addr(tr + 1), tile_l1 + ((tr + 1 - tr0) % 2) * 4 * F, 4 * F);
            }
            for (uint32_t r = 0; r < 32; ++r) {
                const bool off = has_mask && mask[(r / 16) * 512 + (r % 16) * 16] == 0;
                const uint32_t row = cur + (r / 16) * 2 * F + (r % 16) * R;
                volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row);
                volatile tt_l1_ptr uint32_t* ws = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row + F);
                const uint32_t t = tr * 32 + r;
                if (k_tok % 64 == 0 && k_tok > 0) {
                    noc_async_write_barrier();  // the ring's slots are reused
                }
                volatile tt_l1_ptr uint32_t* tp =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring + (k_tok % 64) * PAGE);
                uint32_t k = 0;
                for (uint32_t j = 0; j < K; ++j) {
                    const uint32_t e = ids[j];
                    const uint32_t v = ws[j];
                    if (v != 0 && e < local && !off) {
                        const uint32_t i = got[e];
                        got[e] = i + 1;
                        tst[start[e] + i] = t;
                        wst[start[e] + i] = v;
                        tp[k++] = seg[e] + pos0[e] + i;
                    }
                }
                for (; k < ROW_WORDS; ++k) {
                    tp[k] = SENT;
                }
                noc_async_write(ring + (k_tok % 64) * PAGE, s_tpair.get_noc_addr(t / 32, (t % 32) * PAGE), PAGE);
                ++k_tok;
            }
        }
        noc_async_read_barrier();
        for (uint32_t e = 0; e < local; ++e) {
            const uint32_t n = got[e];
            if (n == 0) {
                continue;
            }
            const uint32_t xy = get_arg_val<uint32_t>(xy0 + e);
            const uint32_t x = xy & 0xFFFF, y = xy >> 16;
            noc_async_write(tok_st + start[e] * 4, get_noc_addr(x, y, tok_arr + pos0[e] * 4), n * 4);
            noc_async_write(w_st + start[e] * 4, get_noc_addr(x, y, w_arr + pos0[e] * 4), n * 4);
        }
        if (c == 0) {
            noc_async_write(seg_l1, s_seg.get_noc_addr(0), SEG_WORDS * 4);
        }
        noc_async_write_barrier();  // the runs are in place before the expert cores hear of it
        for (uint32_t e = 0; e < local; ++e) {
            const uint32_t xy = get_arg_val<uint32_t>(xy0 + e);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, pushed_addr), 1);
        }
        noc_async_atomic_barrier();
        return;
    }

    // role 1: expert e's segment
    const uint32_t e = who;
    const uint32_t n = seg[SEG_COUNT + e], rows = (n + 31) / 32 * 32;
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pushed_addr), cores);
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_arr);
    volatile tt_l1_ptr uint32_t* wts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_arr);
    for (uint32_t i = n; i < rows; ++i) {
        tok[i] = SENT;
        wts[i] = 0;
    }
    const uint32_t page0 = seg[e] / ROW_WORDS;
    for (uint32_t i = 0; i < rows / ROW_WORDS; ++i) {
        noc_async_write(tok_arr + i * PAGE, s_ptok.get_noc_addr(page0 + i), PAGE);
        noc_async_write(w_arr + i * PAGE, s_pw.get_noc_addr(page0 + i), PAGE);
    }
    noc_async_write_barrier();
}
