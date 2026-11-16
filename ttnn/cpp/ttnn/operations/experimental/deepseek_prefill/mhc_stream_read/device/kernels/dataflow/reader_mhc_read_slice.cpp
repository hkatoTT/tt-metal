// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Slice s (K tiles kt = s K .. s K + K - 1 of the hidden dimension): the streams' tiles of every row tile and fn's rows
// (3 tiles each, k-major), all requested at once with a transaction id per k (1 + k % 8) and pushed in k order as each
// k's tiles land (the compute starts the projection on the first while the rest arrive), then gamma's tiles; then P
// and the block input's norm scale as the mix core sends them (its multicasts write them at CB_PIN's and CB_NS's base,
// the same address on every core: this program's first push).
void kernel_main() {
    const uint32_t streams_addr = get_arg_val<uint32_t>(0);
    const uint32_t fn_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t s = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_s = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fn = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_pin = get_compile_time_arg_val(3);
    constexpr uint32_t cb_ns = get_compile_time_arg_val(4);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t sem_p = get_compile_time_arg_val(8);
    constexpr uint32_t sem_ns = get_compile_time_arg_val(9);
    // mhc_stream_write_read: the streams' tiles come from the write cores into CB_S (each followed by an increment of
    // sem_s), not from DRAM
    constexpr bool from_l1 = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t sem_s = get_compile_time_arg_val(11);
    constexpr auto streams_args = TensorAccessorArgs<12>();
    constexpr auto fn_args = TensorAccessorArgs<streams_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<fn_args.next_compile_time_args_offset()>();
    constexpr uint32_t proj_tiles = 3;

    const uint32_t tile = get_local_cb_interface(cb_s).fifo_page_size;  // bf16
    const auto s_streams = TensorAccessor(streams_args, streams_addr);
    const auto s_fn = TensorAccessor(fn_args, fn_addr);
    const auto s_gamma = TensorAccessor(gamma_args, gamma_addr);

    Noc noc;
    CircularBuffer s_cb(cb_s), fn(cb_fn), g(cb_g), pin(cb_pin), ns(cb_ns);
    s_cb.reserve_back(row_tiles * k_tiles);
    fn.reserve_back(k_tiles * proj_tiles);
    g.reserve_back(k_tiles);
    const uint32_t s_base = s_cb.get_write_ptr(), fn_base = fn.get_write_ptr();
    for (uint32_t k = 0; k < k_tiles; ++k) {
        const uint32_t kt = s * k_tiles + k;
        noc_async_read_set_trid(1 + k % 8);
        for (uint32_t rt = 0; rt < row_tiles && !from_l1; ++rt) {
            noc_async_read(s_streams.get_noc_addr(rt * d_tiles + kt), s_base + (rt * k_tiles + k) * tile, tile);
        }
        for (uint32_t n = 0; n < proj_tiles; ++n) {
            noc_async_read(s_fn.get_noc_addr(kt * proj_tiles + n), fn_base + (k * proj_tiles + n) * tile, tile);
        }
    }
    noc_async_read_set_trid(0);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        noc.async_read(s_gamma, g, tile, {.page_id = s * k_tiles + k}, {.offset_bytes = k * tile});
    }
    if constexpr (from_l1) {  // every streams tile of this slice landed
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_s)), row_tiles * k_tiles);
    }
    for (uint32_t k = 0; k < k_tiles; ++k) {  // k's tiles at positions rt K + k: the compute waits for (k + 1) R
        noc_async_read_barrier_with_trid(1 + k % 8);
        s_cb.push_back(row_tiles);
        fn.push_back(proj_tiles);
    }
    noc.async_read_barrier();
    g.push_back(k_tiles);

    pin.reserve_back(row_tiles);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_p)), 1);
    pin.push_back(row_tiles);
    ns.reserve_back(1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ns)), 1);
    ns.push_back(1);
}
