// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "rt_common.hpp"

// The routing's reader: this core's token rows start .. start + count - 1 of the logits (a tile row is `stride` tiles;
// the first et are read) in the block layout (rt_common.hpp): the rows of one 16-row group of a tile row are adjacent
// face rows, so one read a block (tile t, face half h) a group, not one a row (the DRAM's count of small reads was
// what took the time: 256 tokens, 3 rows a core, 21.5 us of reads).  With `share`, then the second half of the
// rows' routing (the writer routes the first half, rounded up), once the compute has made the scores and keys; with
// `live`, a row whose token is idle in the live mask written zero (rt_writer.cpp).
void kernel_main() {
    using namespace moe_route;
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const uint32_t bias_addr = get_arg_val<uint32_t>(5);
    const uint32_t tokens = get_arg_val<uint32_t>(6);  // (the shared expert's rows, `shared_span`)
    const uint32_t live_addr = get_arg_val<uint32_t>(7);  // the live mask (with `live`) and its bytes
    const uint32_t live_bytes = get_arg_val<uint32_t>(8);
    // H1 (moe.py `_routed_h1`): this mesh row's tokens are rows out_base .. of the column's w (the shared expert's
    // span over the column's span_tokens); else 0 and tokens
    const uint32_t out_base = get_arg_val<uint32_t>(9);
    const uint32_t span_tokens = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_log = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t et = get_compile_time_arg_val(4);
    constexpr uint32_t stride = get_compile_time_arg_val(5);
    constexpr bool share = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t k_arg = get_compile_time_arg_val(7);  // k
    constexpr uint32_t k = k_arg & 0xffu;
    constexpr bool globals = ((k_arg >> 16) & 1u) != 0;  // (rt_common.hpp stage_row: | 1 << 16)
    constexpr uint32_t local = get_compile_time_arg_val(8);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(9);
    constexpr bool with_shared = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t base_page = get_compile_time_arg_val(11);
    // the bias read here, row by row into the block layout (few tokens: few reads, and the compute then waits for no
    // seed); else by the writer
    constexpr bool bias_here = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t cb_bias = get_compile_time_arg_val(13);
    constexpr bool trisc = get_compile_time_arg_val(14) != 0;  // rows routed by the compute (rows_of)
    constexpr bool live = get_compile_time_arg_val(15) != 0;   // idle tokens choose nothing (the live mask)
    // the router and the routing in one program (ttnn/router_mm.py `router_route`): the router's
    // core 0 sets this core's sem_logits to 1 once every router core's logits tile is in DRAM (rm_w.cpp: by writes,
    // no atomic); wait for it first (`fused`: the router's cores, 0: route_only alone, the logits there already)
    constexpr uint32_t fused = get_compile_time_arg_val(16);
    constexpr uint32_t sem_logits = get_compile_time_arg_val(17);
    constexpr auto logits_args = TensorAccessorArgs<18>();
    constexpr auto base_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto live_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr uint32_t E = et * 32;

    const auto s_logits = TensorAccessor(logits_args, logits_addr);
    const uint32_t np = (count * E + 1023) / 1024;
    if constexpr (fused != 0) {
        DeviceZoneScopedN("R-LOGITS");
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_logits));
        noc_semaphore_wait_min(sem, 1);
        noc_semaphore_set(sem, 0);  // (the next call's from 0)
    }
    {
        DeviceZoneScopedN("R-READ");
        cb_reserve_back(cb_log, np);
        const uint32_t l0 = get_write_ptr(cb_log);
        for (uint32_t j0 = 0; j0 < count;) {
            const uint32_t r0 = start + j0;
            const uint32_t len = count - j0 < 16 - r0 % 16 ? count - j0 : 16 - r0 % 16;  // to the group's end
            const uint32_t y = r0 % 32, page = (r0 / 32) * stride;
            for (uint32_t t = 0; t < et; ++t) {
                for (uint32_t h = 0; h < 2; ++h) {  // face 2 (y / 16) + h, rows y % 16 ..
                    noc_async_read(
                        s_logits.get_noc_addr(page + t, ((y / 16) * 2 + h) * F + (y % 16) * R),
                        l0 + ((2 * t + h) * count + j0) * R,
                        len * R);
                }
            }
            j0 += len;
        }
        if constexpr (live) {  // the live mask, for this RISC and the writer (rt_common.hpp LIVE)
            const uint32_t scr = align64(get_write_ptr(cb_scr));
            noc_async_read(TensorAccessor(live_args, live_addr).get_noc_addr(0), scr + LIVE, live_bytes);
        }
        if constexpr (bias_here) {
            const auto s_bias = TensorAccessor(bias_args, bias_addr);
            cb_reserve_back(cb_bias, np);
            const uint32_t b0 = get_write_ptr(cb_bias);
            for (uint32_t j = 0; j < count; ++j) {
                for (uint32_t b = 0; b < 2 * et; ++b) {  // tile b / 2, face b % 2, row 0
                    noc_async_read(s_bias.get_noc_addr(b / 2, (b % 2) * F), b0 + (b * count + j) * R, R);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_log, np);
        if constexpr (bias_here) {
            cb_push_back(cb_bias, np);
        }
    }

    if constexpr (share) {
        // (the writer pops the scores and keys after its own rows; nothing is pushed after them)
        const Rows rows = rows_of(count, trisc, share);
        const uint32_t from = rows.writer;
        if (from < rows.reader) {
            DeviceZoneScopedN("R-ROUTE");
            const uint32_t scr = align64(get_write_ptr(cb_scr));
            const auto s_base = TensorAccessor(base_args, base_addr);
            const auto s_w = TensorAccessor(w_args, w_addr, 4 * F);
            noc_async_read(s_base.get_noc_addr(0), scr + RBASE, base_page);
            if constexpr (live) {
                volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + RZERO);
                for (uint32_t i = 0; i < 16; ++i) {
                    z[i] = 0;
                }
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + RBASE);
            const uint32_t base = page[0];
            const Span sh =
                shared_span(with_shared, page[1], page[2], page[3], out_base + start, count, span_tokens);
            auto row_addr = [&](uint32_t r, uint32_t half) {
                r += out_base;
                const uint32_t y = r % 32;
                return s_w.get_noc_addr(r / 32, (y / 16) * 2 * F + half * F + (y % 16) * R);
            };
            {
                DeviceZoneScopedN("R-WAIT");
                cb_wait_front(cb_sc, np);
                cb_wait_front(cb_ch, np);
            }
            asm volatile("" ::: "memory");
            const uint32_t sc_base = get_read_ptr(cb_sc), ch_base = get_read_ptr(cb_ch);
            for (uint32_t j = from; j < rows.reader; ++j) {
                if constexpr (live) {
                    if (!live_row(scr + LIVE, out_base + start + j)) {  // (the column's token)
                        noc_async_write(scr + RZERO, row_addr(start + j, 0), R);
                        noc_async_write(scr + RZERO, row_addr(start + j, 1), R);
                        continue;
                    }
                }
                route_row<k, E, globals>(
                    reinterpret_cast<const uint32_t*>(sc_base + j * R),
                    reinterpret_cast<const uint32_t*>(ch_base + j * R),
                    16 * count,
                    base,
                    local,
                    as_float(scale_bits),
                    j >= sh.lo && j < sh.hi,
                    scr + WOUT + j * 2 * R,
                    row_addr,
                    start + j);
            }
            noc_async_write_barrier();
        }
    }
}
