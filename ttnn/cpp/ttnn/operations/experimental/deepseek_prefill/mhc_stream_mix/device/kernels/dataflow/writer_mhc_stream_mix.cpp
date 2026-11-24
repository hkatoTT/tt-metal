// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t post_addr = get_arg_val<uint32_t>(0);
    const uint32_t comb_addr = get_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_arg_val<uint32_t>(2);
    const uint32_t g = get_arg_val<uint32_t>(3);  // the group: token tile row g of each output

    constexpr uint32_t cb_post = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(1);
    constexpr uint32_t cb_p = get_compile_time_arg_val(2);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(3);
    constexpr auto post_args = TensorAccessorArgs<4>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();

    const uint32_t tile = get_local_cb_interface(cb_post).fifo_page_size;
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);
    const auto s_p = TensorAccessor(p_args, p_addr);

    Noc noc;
    CircularBuffer post(cb_post), comb(cb_comb), p(cb_p);
    comb.wait_front(1);
    noc.async_write(comb, s_comb, tile, {}, {.page_id = g});
    post.wait_front(1);
    noc.async_write(post, s_post, tile, {}, {.page_id = g});
    p.wait_front(row_tiles);
    for (uint32_t rt = 0; rt < row_tiles; ++rt) {
        noc.async_write(p, s_p, tile, {.offset_bytes = rt * tile}, {.page_id = g * row_tiles + rt});
    }
    noc.async_write_barrier();
    comb.pop_front(1);
    post.pop_front(1);
    p.pop_front(row_tiles);
}
