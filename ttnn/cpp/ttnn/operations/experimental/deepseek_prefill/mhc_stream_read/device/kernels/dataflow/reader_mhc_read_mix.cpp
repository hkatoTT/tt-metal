// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// The mix core: the 13 mix constants and the 8 Sinkhorn constants; then the slices' partials, which the slice cores
// write into CB_PART (at its base: this program's first push) and count on a semaphore.
// MHC_CP (mhc_stream_write_read's decode forms: the CB index): first the write side's comb and post pages (its
// inputs: pages each, comb's then post's), for the writer to multicast to the write cores (one mix core: `cp` set).
void kernel_main() {
    const uint32_t mix_consts_addr = get_arg_val<uint32_t>(0);
    const uint32_t consts_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(2);
    constexpr uint32_t slices = get_compile_time_arg_val(3);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t sem_part = get_compile_time_arg_val(5);
    constexpr auto mix_consts_args = TensorAccessorArgs<6>();
    constexpr auto consts_args = TensorAccessorArgs<mix_consts_args.next_compile_time_args_offset()>();

    const uint32_t tile = get_local_cb_interface(cb_mc).fifo_page_size;
    const auto s_mix_consts = TensorAccessor(mix_consts_args, mix_consts_addr);
    const auto s_consts = TensorAccessor(consts_args, consts_addr);

    Noc noc;
    CircularBuffer part(cb_part), mc(cb_mc), sc(cb_sc);
#ifdef MHC_CP
    {
        constexpr uint32_t pages = get_compile_time_arg_val(consts_args.next_compile_time_args_offset());
        constexpr auto comb_args = TensorAccessorArgs<consts_args.next_compile_time_args_offset() + 1>();
        constexpr auto post_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
        if (get_arg_val<uint32_t>(4) != 0) {
            const auto s_comb = TensorAccessor(comb_args, get_arg_val<uint32_t>(2));
            const auto s_post = TensorAccessor(post_args, get_arg_val<uint32_t>(3));
            CircularBuffer cp(MHC_CP);
            cp.reserve_back(2 * pages);
            for (uint32_t i = 0; i < pages; ++i) {
                noc.async_read(s_comb, cp, tile, {.page_id = i}, {.offset_bytes = i * tile});
                noc.async_read(s_post, cp, tile, {.page_id = i}, {.offset_bytes = (pages + i) * tile});
            }
            noc.async_read_barrier();
            cp.push_back(2 * pages);
        }
    }
#endif
    mc.reserve_back(13);
    for (uint32_t i = 0; i < 13; ++i) {
        noc.async_read(s_mix_consts, mc, tile, {.page_id = i}, {.offset_bytes = i * tile});
    }
    sc.reserve_back(8);
    for (uint32_t i = 0; i < 8; ++i) {
        noc.async_read(s_consts, sc, tile, {.page_id = i}, {.offset_bytes = i * tile});
    }
    part.reserve_back(row_tiles * 4 * slices);
    noc.async_read_barrier();
    mc.push_back(13);
    sc.push_back(8);

#ifdef MHC_WR_PART_FLAGS
    for (uint32_t s = 0; s < slices; ++s) {  // (MHC_WR_UNICAST: each slice's flag, set by a write)
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MHC_WR_PART_FLAGS + s)), 1);
    }
#else
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_part)), slices);
#endif
    part.push_back(row_tiles * 4 * slices);
}
