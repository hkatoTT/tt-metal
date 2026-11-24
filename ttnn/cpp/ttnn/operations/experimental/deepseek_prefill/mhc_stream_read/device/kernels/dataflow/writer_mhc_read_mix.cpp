// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "mhc_cp.hpp"

// The mix core: P multicast to the slice cores (into CB_PIN at its base, the same address there), followed by a
// semaphore set to 1 once written; the block input's norm scale likewise (CB_NSO into CB_NS, its own semaphore: the
// slices start on P while it is made); then comb and post to their outputs (the compute makes them after both).
void kernel_main() {
    const uint32_t post_addr = get_arg_val<uint32_t>(0);
    const uint32_t comb_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_post = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(1);
    constexpr uint32_t cb_p = get_compile_time_arg_val(2);
    constexpr uint32_t cb_pin = get_compile_time_arg_val(3);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t slices = get_compile_time_arg_val(5);
    constexpr uint32_t sem_p = get_compile_time_arg_val(6);
    constexpr uint32_t lo_x = get_compile_time_arg_val(7);
    constexpr uint32_t lo_y = get_compile_time_arg_val(8);
    constexpr uint32_t hi_x = get_compile_time_arg_val(9);
    constexpr uint32_t hi_y = get_compile_time_arg_val(10);
    constexpr uint32_t cb_nso = get_compile_time_arg_val(11);
    constexpr uint32_t cb_ns = get_compile_time_arg_val(12);
    constexpr uint32_t sem_ns = get_compile_time_arg_val(13);
    constexpr auto post_args = TensorAccessorArgs<14>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    const uint32_t tile = get_local_cb_interface(cb_post).fifo_page_size;
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);
    const uint32_t pin_base = get_local_cb_interface(cb_pin).fifo_wr_ptr;
    const uint32_t ns_base = get_local_cb_interface(cb_ns).fifo_wr_ptr;
    // the slice cores' rectangle, its corners in the order this NoC's multicast takes them
    auto mcast = [](uint32_t addr) {
        return noc_index == 0 ? get_noc_multicast_addr(lo_x, lo_y, hi_x, hi_y, addr)
                              : get_noc_multicast_addr(hi_x, hi_y, lo_x, lo_y, addr);
    };
    auto send = [&](uint32_t src, uint32_t dst, uint32_t bytes, uint32_t sem) {
        noc_async_write_multicast(src, mcast(dst), bytes, slices);
        noc_async_write_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem)) = 1;
        noc_semaphore_set_multicast(get_semaphore(sem), mcast(get_semaphore(sem)), slices);
    };

    Noc noc;
    CircularBuffer post(cb_post), comb(cb_comb), p(cb_p), nso(cb_nso);
#ifndef MHC_FINISHER  // (GLM53_MHC_FINISH: a finisher core writes comb and post only)
    mhc_send_comb_post(2);  // (mhc_stream_write_read's decode forms: the write cores' comb and post, first)
#ifdef MHC_FIN_AWAY
    // GLM53_MHC_FINISH: the mixes (CB MHC_FIN_MIX) into the group's finisher core's same CB (at
    // the same address; its NoC x, y the runtime args after the comb / post multicast's), its semaphore MHC_FIN_SEM
    // incremented once they landed; then the compute may drop them (one page into CB MHC_FIN_AWAY).  MHC_FIN_SPLIT:
    // the increment after P's send, whose barrier covers the mixes' write (P not held back by a barrier of its own)
    const uint32_t fa = 2 + 1 + 5 * get_arg_val<uint32_t>(2);
    const uint32_t fx = get_arg_val<uint32_t>(fa), fy = get_arg_val<uint32_t>(fa + 1);
    CircularBuffer mixes(MHC_FIN_MIX), sent(MHC_FIN_AWAY);
    auto fin_signal = [&]() {
        noc_semaphore_inc(get_noc_addr(fx, fy, get_semaphore(MHC_FIN_SEM)), 1);
        noc_async_atomic_barrier();
        sent.reserve_back(1);
        sent.push_back(1);
    };
    mixes.wait_front(1);
    noc_async_write(mixes.get_read_ptr(), get_noc_addr(fx, fy, mixes.get_read_ptr()),
                    get_local_cb_interface(MHC_FIN_MIX).fifo_page_size);
#ifndef MHC_FIN_SPLIT
    noc_async_write_barrier();
    fin_signal();
#endif
#endif
    p.wait_front(row_tiles);
    send(p.get_read_ptr(), pin_base, row_tiles * tile, sem_p);
    p.pop_front(row_tiles);
#if defined(MHC_FIN_AWAY) && defined(MHC_FIN_SPLIT)
    fin_signal();  // (P's send's barrier covered the mixes' write)
#endif
    nso.wait_front(1);
    send(nso.get_read_ptr(), ns_base, tile, sem_ns);
    nso.pop_front(1);
#endif
#ifndef MHC_FIN_AWAY  // (GLM53_MHC_FINISH: the finisher core writes them)
    comb.wait_front(1);
    noc.async_write(comb, s_comb, tile, {}, {.page_id = 0});
    post.wait_front(1);
    noc.async_write(post, s_post, tile, {}, {.page_id = 0});
    noc_async_write_barrier();
    comb.pop_front(1);
    post.pop_front(1);
#endif
}
