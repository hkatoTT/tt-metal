// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per unit (K slice s, row tile rt): the streams tiles (rt, s K .. s K + K - 1) and the fn tiles of those K rows
// (3 per row of tiles, k-major).
void kernel_main() {
    const uint32_t streams_addr = get_arg_val<uint32_t>(0);
    const uint32_t fn_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_s = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fn = get_compile_time_arg_val(1);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(4);
    constexpr auto streams_args = TensorAccessorArgs<5>();
    constexpr auto fn_args = TensorAccessorArgs<streams_args.next_compile_time_args_offset()>();
    constexpr uint32_t proj_tiles = 3;

    const uint32_t tile = get_local_cb_interface(cb_s).fifo_page_size;
    const auto s_streams = TensorAccessor(streams_args, streams_addr);
    const auto s_fn = TensorAccessor(fn_args, fn_addr);

    Noc noc;
    CircularBuffer s_cb(cb_s), fn(cb_fn);
    for (uint32_t u = start; u < start + count; ++u) {
        const uint32_t slice = u / row_tiles;
        const uint32_t rt = u % row_tiles;
        s_cb.reserve_back(k_tiles);
        fn.reserve_back(k_tiles * proj_tiles);
        for (uint32_t k = 0; k < k_tiles; ++k) {
            const uint32_t kt = slice * k_tiles + k;
            noc.async_read(s_streams, s_cb, tile, {.page_id = rt * d_tiles + kt}, {.offset_bytes = k * tile});
            for (uint32_t n = 0; n < proj_tiles; ++n) {
                noc.async_read(
                    s_fn, fn, tile, {.page_id = kt * proj_tiles + n}, {.offset_bytes = (k * proj_tiles + n) * tile});
            }
        }
        noc.async_read_barrier();
        s_cb.push_back(k_tiles);
        fn.push_back(k_tiles * proj_tiles);
    }
}
