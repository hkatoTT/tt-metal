// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// The bias once, then each tile row of logits this core's rows (start .. start + count - 1) touch.  When those rows of
// a tile row are all in one half (rows 0-15 or 16-31: faces 0-1 or 2-3), only that half is read, into faces 0-1 (the
// compute then takes only those, the writer reads them there).  Packed (`pack`): only this core's rows, each row's E
// values (2 x 64 B of each of its et tiles) in a row of E fp32 of the tile's 1024, rpt rows a tile (row j of the core
// at word (j mod rpt) E of tile j / rpt), and the bias's row 0 the same way into each of one tile's rpt rows.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t logits_addr = get_arg_val<uint32_t>(a++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t count = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_log = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t et = get_compile_time_arg_val(2);
    constexpr bool pack = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t rpt = get_compile_time_arg_val(4);  // packed: rows a tile
    constexpr auto logits_args = TensorAccessorArgs<5>();
    constexpr auto bias_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();

    const auto s_logits = TensorAccessor(logits_args, logits_addr);
    const auto s_bias = TensorAccessor(bias_args, bias_addr);
    const uint32_t tile = get_local_cb_interface(cb_log).fifo_page_size;

    Noc noc;
    CircularBuffer log(cb_log), bias(cb_bias);
    if constexpr (pack) {
        constexpr uint32_t F = 1024, R = 64, E = et * 32;  // fp32 face, face row (bytes), values a row
        // a row's E values: tile t's columns 16 h .. 16 h + 15 (face 2 (y / 16) + h, row y % 16) at word 32 t + 16 h
        auto read_row = [&](const auto& src, uint32_t first_page, uint32_t y, uint32_t dst) {
            for (uint32_t t = 0; t < et; ++t) {
                for (uint32_t h = 0; h < 2; ++h) {
                    noc_async_read(src.get_noc_addr(first_page + t, ((y / 16) * 2 + h) * F + (y % 16) * R),
                                   dst + (32 * t + 16 * h) * 4, R);
                }
            }
        };
        bias.reserve_back(1);
        const uint32_t b0 = bias.get_write_ptr();
        read_row(s_bias, 0, 0, b0);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* bw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b0);
        for (uint32_t i = E; i < rpt * E; ++i) {
            bw[i] = bw[i - E];
        }
        bias.push_back(1);
        if (count == 0) {
            return;
        }
        const uint32_t np = (count + rpt - 1) / rpt;
        log.reserve_back(np);
        const uint32_t l0 = log.get_write_ptr();
        for (uint32_t j = 0; j < count; ++j) {
            const uint32_t r = start + j;
            read_row(s_logits, (r / 32) * et, r % 32, l0 + (j / rpt) * tile + (j % rpt) * E * 4);
        }
        noc_async_read_barrier();
        log.push_back(np);
        return;
    }
    bias.reserve_back(et);
    for (uint32_t t = 0; t < et; ++t) {
        noc.async_read(s_bias, bias, tile, {.page_id = t}, {.offset_bytes = t * tile});
    }
    noc.async_read_barrier();
    bias.push_back(et);

    for (uint32_t tr = start / 32; tr <= (start + count - 1) / 32; ++tr) {
        const uint32_t r0 = tr * 32 > start ? tr * 32 : start;
        const uint32_t r1 = (tr + 1) * 32 < start + count ? (tr + 1) * 32 : start + count;
        const bool half = (r0 % 32) / 16 == ((r1 - 1) % 32) / 16;
        const uint32_t from = half ? ((r0 % 32) / 16) * tile / 2 : 0;
        log.reserve_back(et);
        for (uint32_t t = 0; t < et; ++t) {
            noc.async_read(
                s_logits, log, half ? tile / 2 : tile, {.page_id = tr * et + t, .offset_bytes = from},
                {.offset_bytes = t * tile});
        }
        noc.async_read_barrier();
        log.push_back(et);
    }
}
