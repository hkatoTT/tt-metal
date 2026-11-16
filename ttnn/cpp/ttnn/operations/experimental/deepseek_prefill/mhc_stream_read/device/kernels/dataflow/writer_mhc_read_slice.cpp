// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Slice s: its partials into the mix core's CB_PART (tile (rt, n) at (rt 4 + n) S + s, the order mhc_stream_mix sums
// them in), followed by a semaphore increment there once written; then its columns of the normed block input to the
// output.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_part = get_compile_time_arg_val(2);
    constexpr uint32_t slices = get_compile_time_arg_val(3);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t sem_part = get_compile_time_arg_val(6);
    constexpr uint32_t mix_x = get_compile_time_arg_val(7);
    constexpr uint32_t mix_y = get_compile_time_arg_val(8);
    constexpr auto out_args = TensorAccessorArgs<9>();

    const uint32_t t32 = get_local_cb_interface(cb_out).fifo_page_size;
    const uint32_t t16 = get_local_cb_interface(cb_y).fifo_page_size;
    const uint32_t part_base = get_local_cb_interface(cb_part).fifo_wr_ptr;  // the mix core's too
    const auto s_out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer out(cb_out), y(cb_y);
    out.wait_front(4 * row_tiles);
    for (uint32_t i = 0; i < 4 * row_tiles; ++i) {  // i = rt 4 + n
        noc_async_write(
            out.get_read_ptr() + i * t32, get_noc_addr(mix_x, mix_y, part_base + (i * slices + s) * t32), t32);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(mix_x, mix_y, get_semaphore(sem_part)), 1);
    out.pop_front(4 * row_tiles);

    y.wait_front(k_tiles);
    for (uint32_t c = 0; c < k_tiles; ++c) {
        noc.async_write(y, s_out, t16, {.offset_bytes = c * t16}, {.page_id = s * k_tiles + c});
    }
    noc.async_write_barrier();
    y.pop_front(k_tiles);
}
