// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "rt_common.hpp"

// The routing's writer (NoC 0): first the bias in the block layout (rt_common.hpp) for the compute, unless the reader
// brings it (few tokens): its row 0 read from DRAM by a few seed cores and written on to the others (unicast), then
// copied in L1 to the core's rows (every core's reader read it again for each row before: the same 18 DRAM words a
// row on every core); then this core's token rows start .. start + count - 1 (with `share`, the first half of them,
// rounded up: the reader routes the rest), as moe_decode_route writes them for a column's experts (`route_row`), each
// row its own staging (no wait for the earlier rows' writes); and the zero rows tokens .. n - 1 dealt out over the
// cores (row tokens + i, + n_workers, ..).  With `live`, a row whose token is idle in the live mask (a seat without
// a request: moe_route.LiveTokens; the reader reads it with the logits) is written zero, choosing no expert.
void kernel_main() {
    using namespace moe_route;
    const uint32_t bias_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const uint32_t widx = get_arg_val<uint32_t>(5);
    const uint32_t n_workers = get_arg_val<uint32_t>(6);
    const uint32_t tokens = get_arg_val<uint32_t>(7);
    const uint32_t n_rows = get_arg_val<uint32_t>(8);
    // the bias: 1, a receiver (its seed writes it here); else a seed, writing it on to nrecv receivers (their NoC x, y
    // from runtime arg 13 on)
    const bool receiver = get_arg_val<uint32_t>(9) != 0;
    const uint32_t nrecv = get_arg_val<uint32_t>(10);
    // moe.H1 (moe.py `_routed_h1`): this mesh row's tokens are rows out_base .. of the column's w (the shared expert's
    // span over the column's span_tokens; n_rows then `tokens`: no zero rows); else 0 and tokens
    const uint32_t out_base = get_arg_val<uint32_t>(11);
    const uint32_t span_tokens = get_arg_val<uint32_t>(12);

    constexpr uint32_t cb_bias = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t et = get_compile_time_arg_val(4);
    constexpr uint32_t k_arg = get_compile_time_arg_val(5);  // k
    constexpr uint32_t k = k_arg & 0xffu;
    constexpr bool globals = ((k_arg >> 16) & 1u) != 0;  // (rt_common.hpp stage_row: | 1 << 16)
    constexpr uint32_t local = get_compile_time_arg_val(6);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(7);
    constexpr bool with_shared = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t base_page = get_compile_time_arg_val(9);
    constexpr bool share = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(11);  // the bias's row 0 (2 et pieces of 64 B)
    constexpr uint32_t sem_bias = get_compile_time_arg_val(12);
    constexpr bool bias_here = get_compile_time_arg_val(13) != 0;  // the reader brings the bias (rt_reader.cpp)
    constexpr bool trisc = get_compile_time_arg_val(14) != 0;      // rows routed by the compute (rows_of)
    constexpr bool live = get_compile_time_arg_val(15) != 0;       // idle tokens choose nothing (the live mask)
    constexpr auto bias_args = TensorAccessorArgs<16>();
    constexpr auto base_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    constexpr uint32_t E = et * 32;

    const auto s_bias = TensorAccessor(bias_args, bias_addr);
    const auto s_base = TensorAccessor(base_args, base_addr);
    const auto s_w = TensorAccessor(w_args, w_addr, 4 * F);
    const uint32_t scr = align64(get_write_ptr(cb_scr));
    const uint32_t np = (count * E + 1023) / 1024;
    {
        DeviceZoneScopedN("W-BIAS");
        // the bias's row 0 (piece b: tile b / 2, face b % 2) as 2 et pieces of 64 B at `raw` (the same address on every
        // core): read from DRAM by a few seed cores only, each then writing it on to its receivers (unicast) and adding
        // 1 to their sem_bias (every core reading the same 18 DRAM words took 9.5 us at 256 tokens); then copied
        // to the core's rows in the block layout
        const uint32_t b0 = get_write_ptr(cb_bias);
        const uint32_t raw = get_write_ptr(cb_tmp);
        if constexpr (!bias_here) {
            cb_reserve_back(cb_bias, np);
        }
        if (!bias_here && !receiver) {
            for (uint32_t b = 0; b < 2 * et; ++b) {
                noc_async_read(s_bias.get_noc_addr(b / 2, (b % 2) * F), raw + b * R, R);
            }
        }
        noc_async_read(s_base.get_noc_addr(0), scr + WBASE, base_page);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + WZERO);
        for (uint32_t i = 0; i < 16; ++i) {
            z[i] = 0;
        }
        noc_async_read_barrier();
        if constexpr (bias_here) {
            // (nothing more: the base page and the zero row only)
        } else if (receiver) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_bias)), 1);
        } else if (nrecv > 0) {
            for (uint32_t i = 0; i < nrecv; ++i) {
                const uint32_t x = get_arg_val<uint32_t>(13 + 2 * i), y = get_arg_val<uint32_t>(14 + 2 * i);
                noc_async_write(raw, get_noc_addr(x, y, raw), 2 * et * R);
            }
            noc_async_writes_flushed();
            for (uint32_t i = 0; i < nrecv; ++i) {
                noc_semaphore_inc(
                    get_noc_addr(
                        get_arg_val<uint32_t>(13 + 2 * i), get_arg_val<uint32_t>(14 + 2 * i), get_semaphore(sem_bias)),
                    1);
            }
        }
        invalidate_l1_cache();
        for (uint32_t b = 0; b < 2 * et && !bias_here; ++b) {
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw + b * R);
            volatile tt_l1_ptr uint32_t* dst0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b0 + b * count * R);
            for (uint32_t i = 0; i < 16; i += 4) {
                const uint32_t v0 = src[i], v1 = src[i + 1], v2 = src[i + 2], v3 = src[i + 3];
                for (uint32_t j = 0; j < count; ++j) {
                    volatile tt_l1_ptr uint32_t* dst = dst0 + 16 * j;
                    dst[i] = v0;
                    dst[i + 1] = v1;
                    dst[i + 2] = v2;
                    dst[i + 3] = v3;
                }
            }
        }
        if constexpr (!bias_here) {
            cb_push_back(cb_bias, np);
        }
    }
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + WBASE);
    const uint32_t base = page[0];
    const Span sh = shared_span(with_shared, page[1], page[2], page[3], out_base + start, count, span_tokens);
    const float scale = as_float(scale_bits);
    const Rows rows = rows_of(count, trisc, share);
    volatile tt_l1_ptr uint32_t* sync = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + TSYNC);
    if (rows.reader < count) {  // the compute's rows: the base for it, its count zeroed, then the scratch CB pushed
        sync[0] = base;
        sync[1] = sh.lo;
        sync[3] = sh.hi;
        sync[2] = 0;
        cb_reserve_back(cb_scr, 1);
        cb_push_back(cb_scr, 1);
    }
    auto row_addr = [&](uint32_t r, uint32_t half) {  // row r's columns 16 half .. 16 half + 15 in w
        r += out_base;
        const uint32_t y = r % 32;
        return s_w.get_noc_addr(r / 32, (y / 16) * 2 * F + half * F + (y % 16) * R);
    };
    auto idle = [&](uint32_t r) {  // row r's token without a request: its row written zero
        if constexpr (live) {
            if (!live_row(scr + LIVE, out_base + r)) {  // (the column's token)
                noc_async_write(scr + WZERO, row_addr(r, 0), R);
                noc_async_write(scr + WZERO, row_addr(r, 1), R);
                return true;
            }
        }
        return false;
    };

    {
        DeviceZoneScopedN("W-WAIT");
        cb_wait_front(cb_sc, np);
        cb_wait_front(cb_ch, np);
    }
    asm volatile("" ::: "memory");
    if constexpr (live) {
        invalidate_l1_cache();  // (the mask: the reader's, in before the logits it pushed)
    }
    const uint32_t sc_base = get_read_ptr(cb_sc), ch_base = get_read_ptr(cb_ch);
    {
        DeviceZoneScopedN("W-ROUTE");
        for (uint32_t j = 0; j < rows.writer; ++j) {
            if (idle(start + j)) {
                continue;
            }
            route_row<k, E, globals>(
                reinterpret_cast<const uint32_t*>(sc_base + j * R),
                reinterpret_cast<const uint32_t*>(ch_base + j * R),
                16 * count,
                base,
                local,
                scale,
                j >= sh.lo && j < sh.hi,
                scr + WOUT + j * 2 * R,
                row_addr,
                start + j);
        }
    }
    if (rows.reader < count) {  // the compute's rows, once staged
        DeviceZoneScopedN("W-TRISC");
        do {
            invalidate_l1_cache();
        } while (sync[2] != count - rows.reader);
        for (uint32_t j = rows.reader; j < count; ++j) {
            if (idle(start + j)) {
                continue;
            }
            const uint32_t stage = scr + WOUT + j * 2 * R;
            noc_async_write(stage, row_addr(start + j, 0), R);
            noc_async_write(stage + R, row_addr(start + j, 1), R);
        }
    }
    DeviceZoneScopedN("W-TAIL");
    for (uint32_t r = tokens + widx; r < n_rows; r += n_workers) {
        noc_async_write(scr + WZERO, row_addr(r, 0), R);
        noc_async_write(scr + WZERO, row_addr(r, 1), R);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();  // (the bias seed's increments' acks in before the kernel ends)
    // With `share` the reader waits on the scores and keys too (rt_reader.cpp) and nothing orders its wait_front
    // (which reads tiles_acked once) after this pop: a pop first would leave it waiting for good.
    // Nothing else uses these CBs in the program, and the counts are zeroed after it.
    if constexpr (!share) {
        cb_pop_front(cb_sc, np);
        cb_pop_front(cb_ch, np);
    }
}
