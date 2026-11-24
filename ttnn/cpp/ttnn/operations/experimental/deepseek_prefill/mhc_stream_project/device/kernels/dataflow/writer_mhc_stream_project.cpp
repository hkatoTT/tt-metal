// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per unit (s, rt): the 4 partial tiles into row tile rt of slice s of the [S, R, 128] output.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr uint32_t out_tiles = 4;

    const uint32_t tile = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s_out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer out(cb_out);
    for (uint32_t u = start; u < start + count; ++u) {
        const uint32_t slice = u / row_tiles;
        const uint32_t rt = u % row_tiles;
        out.wait_front(out_tiles);
        for (uint32_t n = 0; n < out_tiles; ++n) {
            noc.async_write(
                out, s_out, tile, {.offset_bytes = n * tile}, {.page_id = (slice * row_tiles + rt) * out_tiles + n});
        }
        noc.async_write_barrier();
        out.pop_front(out_tiles);
    }
}
