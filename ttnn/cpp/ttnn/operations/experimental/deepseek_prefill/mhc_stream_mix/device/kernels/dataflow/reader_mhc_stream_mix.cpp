// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// The 13 mix constants and the 8 Sinkhorn constants (resident), then the partials in the compute's order: per row
// tile, per column tile (proj 0-2, Gram), every slice's tile; `chunk` tiles per barrier, the first with the constants.
void kernel_main() {
    const uint32_t partial_addr = get_arg_val<uint32_t>(0);
    const uint32_t mix_consts_addr = get_arg_val<uint32_t>(1);
    const uint32_t consts_addr = get_arg_val<uint32_t>(2);
    const uint32_t rt0 = get_arg_val<uint32_t>(3);  // the group's first row tile of the partials

    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(2);
    constexpr uint32_t slices = get_compile_time_arg_val(3);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t chunk = get_compile_time_arg_val(5);
    constexpr uint32_t total_rt = get_compile_time_arg_val(6);  // the partials' row tiles (every group's)
    constexpr auto partial_args = TensorAccessorArgs<7>();
    constexpr auto mix_consts_args = TensorAccessorArgs<partial_args.next_compile_time_args_offset()>();
    constexpr auto consts_args = TensorAccessorArgs<mix_consts_args.next_compile_time_args_offset()>();

    const uint32_t tile = get_local_cb_interface(cb_part).fifo_page_size;
    const auto s_partial = TensorAccessor(partial_args, partial_addr);
    const auto s_mix_consts = TensorAccessor(mix_consts_args, mix_consts_addr);
    const auto s_consts = TensorAccessor(consts_args, consts_addr);

    Noc noc;
    CircularBuffer part(cb_part), mc(cb_mc), sc(cb_sc);
    mc.reserve_back(13);
    for (uint32_t i = 0; i < 13; ++i) {
        noc.async_read(s_mix_consts, mc, tile, {.page_id = i}, {.offset_bytes = i * tile});
    }
    sc.reserve_back(8);
    for (uint32_t i = 0; i < 8; ++i) {
        noc.async_read(s_consts, sc, tile, {.page_id = i}, {.offset_bytes = i * tile});
    }

    constexpr uint32_t total = row_tiles * 4 * slices;
    for (uint32_t first = 0; first < total; first += chunk) {
        const uint32_t n_tiles = total - first < chunk ? total - first : chunk;
        part.reserve_back(n_tiles);
        for (uint32_t k = 0; k < n_tiles; ++k) {
            const uint32_t i = first + k;  // (rt, n, s) in the compute's order
            const uint32_t s = i % slices;
            const uint32_t n = (i / slices) % 4;
            const uint32_t rt = i / slices / 4;
            noc.async_read(s_partial, part, tile, {.page_id = (s * total_rt + rt0 + rt) * 4 + n}, {.offset_bytes = k * tile});
        }
        noc.async_read_barrier();
        if (first == 0) {
            mc.push_back(13);
            sc.push_back(8);
        }
        part.push_back(n_tiles);
    }
}
