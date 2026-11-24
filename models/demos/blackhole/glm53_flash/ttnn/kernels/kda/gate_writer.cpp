// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// The KDA prefill's decay gate, the writer: the core's tiles idx in [i0, i1) (column-major, as the reader's) to tile
// (r, c) of out [1, T, W] (bf16).
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t i0 = get_arg_val<uint32_t>(1);
    const uint32_t i1 = get_arg_val<uint32_t>(2);
    const uint32_t real_addr = get_arg_val<uint32_t>(3);
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t W_COLS = get_compile_time_arg_val(1);
    constexpr uint32_t CB_O = get_compile_time_arg_val(2);
    constexpr uint32_t CB_M = CB_O + 1;
    constexpr auto o_args = TensorAccessorArgs<3>();
    // (the program cache's key [R, W_COLS, per] at o_args' next offset), then has_real and real's accessor: the rows
    // whose mask is 0 come out times 0 (the bf16 sign bit kept: x * 0), as ttnn.multiply(gate, real)
    constexpr uint32_t key_at = o_args.next_compile_time_args_offset();
    constexpr bool has_real = get_compile_time_arg_val(key_at + 3) != 0;
    constexpr auto r_args = TensorAccessorArgs<key_at + 4>();
    const uint32_t o_tile = get_tile_size(CB_O);
    const auto s_o = TensorAccessor(o_args, out_addr, o_tile);
    const auto s_r = TensorAccessor(r_args, real_addr, 2048);
    uint32_t mask_r = ~0U, zero_rows = 0;
    const uint32_t scr = get_write_ptr(CB_M);  // (the mask's scratch, 1 KB: this writer's alone)
    for (uint32_t idx = i0; idx < i1; ++idx) {
        const uint32_t c = idx / R, r = idx % R;
        cb_wait_front(CB_O, 1);
        if constexpr (has_real) {
            if (r != mask_r) {  // the tile row's mask: column 0 of its faces 0 and 2 (rows 0-15, 16-31)
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_read(s_r.get_noc_addr(r, half * 1024), scr + half * 512, 512);
                }
                noc_async_read_barrier();
                const auto* values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
                zero_rows = 0;
                for (uint32_t y = 0; y < 32; ++y) {
                    if ((values[(y / 16) * 256 + (y % 16) * 16] & 0x7FFF) == 0) {
                        zero_rows |= 1U << y;
                    }
                }
                mask_r = r;
            }
            const uint32_t tile = get_read_ptr(CB_O);
            // (8 words loaded before any is stored back, so that the loads overlap: a tile wholly masked, a row
            // without a request's, took ~17 cycles a word word by word; the same AND)
            const auto mask_words = [](volatile tt_l1_ptr uint32_t* words, uint32_t n) {
                for (uint32_t i = 0; i < n; i += 8) {
                    const uint32_t w0 = words[i], w1 = words[i + 1], w2 = words[i + 2], w3 = words[i + 3];
                    const uint32_t w4 = words[i + 4], w5 = words[i + 5], w6 = words[i + 6], w7 = words[i + 7];
                    words[i] = w0 & 0x80008000U;
                    words[i + 1] = w1 & 0x80008000U;
                    words[i + 2] = w2 & 0x80008000U;
                    words[i + 3] = w3 & 0x80008000U;
                    words[i + 4] = w4 & 0x80008000U;
                    words[i + 5] = w5 & 0x80008000U;
                    words[i + 6] = w6 & 0x80008000U;
                    words[i + 7] = w7 & 0x80008000U;
                }
            };
            if (zero_rows == 0xFFFFFFFFU) {  // every row: the tile's 512 words in one run
                mask_words(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile), 512);
            } else {
                for (uint32_t y = 0, rows = zero_rows; rows != 0; ++y, rows >>= 1) {
                    if (rows & 1) {
                        for (uint32_t side = 0; side < 2; ++side) {
                            mask_words(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                    tile + ((y / 16) * 2 + side) * 512 + (y % 16) * 32),
                                8);
                        }
                    }
                }
            }
        }
        noc_async_write(get_read_ptr(CB_O), s_o.get_noc_addr(r * W_COLS + c), o_tile);
        noc_async_writes_flushed();
        cb_pop_front(CB_O, 1);
    }
    noc_async_write_barrier();
}
