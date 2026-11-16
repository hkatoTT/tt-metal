// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// o_norm (row 0 of 4 tiles) and a ones tile, once; then per piece (unit u = (head h, block of 16 users), its rows
// row0 .. row0 + rows - 1, row0 a multiple of 4): user 16 block + row0 + i's o row (pair (16 block + row0 + i) H + h)
// into row i of 4 tiles, and the piece's gate rows (bf16, whole 4-row groups of the faces: 128-byte pieces at
// 128-byte offsets) into rows 0 .. of faces 0-1 of 4 tiles.
namespace {
constexpr uint32_t N = 4;
}  // namespace

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t fg_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);  // the pieces, then (unit, row0, rows) each

    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(3);
    constexpr uint32_t heads = get_compile_time_arg_val(4);
    constexpr uint32_t users = get_compile_time_arg_val(5);
    constexpr uint32_t fg_tiles = get_compile_time_arg_val(6);
    constexpr auto o_args = TensorAccessorArgs<7>();
    constexpr auto fg_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<fg_args.next_compile_time_args_offset()>();
    constexpr uint32_t blocks = (users + 15) / 16;

    const auto s_o = TensorAccessor(o_args, o_addr);
    const auto s_fg = TensorAccessor(fg_args, fg_addr);
    const auto s_w = TensorAccessor(w_args, w_addr);
    const uint32_t t16 = get_local_cb_interface(cb_g).fifo_page_size;
    const uint32_t t32 = get_local_cb_interface(cb_o).fifo_page_size;
    constexpr uint32_t F16 = 512, F32 = 1024, R32 = 64;

    Noc noc;
    CircularBuffer o(cb_o), g(cb_g), w(cb_w), ones(cb_ones);

    DeviceZoneScopedN("FR-ALL");  // (zones to the end of their scope)
    w.reserve_back(N);
    for (uint32_t c = 0; c < N; ++c) {
        for (uint32_t half = 0; half < 2; ++half) {
            noc.async_read(s_w, w, R32, {.page_id = c, .offset_bytes = half * F32}, {.offset_bytes = c * t32 + half * F32});
        }
    }
    ones.reserve_back(1);  // filled while o_norm is read
    volatile tt_l1_ptr uint32_t* one = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones.get_write_ptr());
    for (uint32_t i = 0; i < 1024; ++i) {
        one[i] = 0x3f800000u;
    }
    ones.push_back(1);
    noc.async_read_barrier();
    w.push_back(N);

    for (uint32_t pc = 0; pc < count; ++pc) {
        DeviceZoneScopedN("FR-UNIT");
        const uint32_t u = get_arg_val<uint32_t>(4 + 3 * pc);
        const uint32_t row0 = get_arg_val<uint32_t>(5 + 3 * pc);
        const uint32_t rows = get_arg_val<uint32_t>(6 + 3 * pc);
        const uint32_t block = u % blocks;
        const uint32_t h = u / blocks;
        const uint32_t g_bytes = ((rows + 3) / 4) * 4 * 32;  // the piece's 4-row groups of a bf16 face
        o.reserve_back(N);
        g.reserve_back(N);
        for (uint32_t i = 0; i < rows; ++i) {
            const uint32_t p = (block * 16 + row0 + i) * heads + h;
            const uint32_t r = p % 32;
            const uint32_t lo = (r / 16) * 2 * F32 + (r % 16) * R32;
            for (uint32_t c = 0; c < N; ++c) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc.async_read(s_o, o, R32, {.page_id = (p / 32) * N + c, .offset_bytes = lo + half * F32},
                                   {.offset_bytes = c * t32 + half * F32 + i * R32});
                }
            }
        }
        for (uint32_t c = 0; c < N; ++c) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc.async_read(s_fg, g, g_bytes,
                               {.page_id = (block / 2) * fg_tiles + heads * N + h * N + c,
                                .offset_bytes = (block % 2) * 2 * F16 + half * F16 + row0 * 32},
                               {.offset_bytes = c * t16 + half * F16});
            }
        }
        noc.async_read_barrier();
        o.push_back(N);
        g.push_back(N);
    }
}
