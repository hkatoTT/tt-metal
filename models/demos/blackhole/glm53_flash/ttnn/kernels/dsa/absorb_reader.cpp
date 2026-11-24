// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// The DSA decode's per-head absorb matmul's reads (ttnn/dsa_absorb.py; RISCV_0, NoC 0): this core's head h, its tile
// rows r0 .. r0 + R - 1 of a (R x K tiles, a's pages (h RT + r0 + r) K + k) into CB_A, then its column slice of w
// (K x NC tiles, pages (h K + k) N + n0 + j) into CB_W, every read in flight at once: a under transaction id 1, w's k
// rows in chunks of KC under ids 2, 3, ..; a pushed when it has landed, then each chunk of w in order as it lands (the
// compute starts on the first k rows while the rest arrive).  The CBs hold the whole of both (nothing else is pushed in
// the program).
void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t h = get_arg_val<uint32_t>(2);
    const uint32_t n0 = get_arg_val<uint32_t>(3);
    const uint32_t r0 = get_arg_val<uint32_t>(4);  // this core's first tile row (GLM53_DSA_ABSORB_RSPLIT)

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);   // this core's tile rows of a
    constexpr uint32_t K = get_compile_time_arg_val(3);   // inner tiles
    constexpr uint32_t N = get_compile_time_arg_val(4);   // w's column tiles (a head's)
    constexpr uint32_t NC = get_compile_time_arg_val(5);  // this core's column tiles
    constexpr uint32_t KC = get_compile_time_arg_val(6);  // w's k rows a chunk
    constexpr uint32_t RT = get_compile_time_arg_val(7);  // a's tile rows (the tokens / 32)
    constexpr auto a_args = TensorAccessorArgs<8>();
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t CHUNKS = (K + KC - 1) / KC;
    static_assert(CHUNKS + 1 <= 15, "transaction ids 1 .. CHUNKS + 1");

    const auto s_a = TensorAccessor(a_args, a_addr);
    const auto s_w = TensorAccessor(w_args, w_addr);
    const uint32_t at = get_local_cb_interface(cb_a).fifo_page_size;
    const uint32_t wt = get_local_cb_interface(cb_w).fifo_page_size;
    noc_async_read_set_trid(0, noc_index);  // (transaction ids from a clean state, as router/rm_w.cpp)
    noc_async_write_set_trid(0, noc_index);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);

    cb_reserve_back(cb_a, R * K);
    cb_reserve_back(cb_w, K * NC);
    const uint32_t a_l1 = get_write_ptr(cb_a);
    const uint32_t w_l1 = get_write_ptr(cb_w);
    noc_async_read_set_trid(1);
    for (uint32_t t = 0; t < R * K; ++t) {
        noc_async_read(s_a.get_noc_addr((h * RT + r0) * K + t), a_l1 + t * at, at);
    }
    for (uint32_t c = 0; c < CHUNKS; ++c) {
        noc_async_read_set_trid(c + 2);
        const uint32_t k1 = (c + 1) * KC < K ? (c + 1) * KC : K;
        for (uint32_t k = c * KC; k < k1; ++k) {
            for (uint32_t j = 0; j < NC; ++j) {
                noc_async_read(s_w.get_noc_addr((h * K + k) * N + n0 + j), w_l1 + (k * NC + j) * wt, wt);
            }
        }
    }
    noc_async_read_barrier_with_trid(1);
    cb_push_back(cb_a, R * K);
    for (uint32_t c = 0; c < CHUNKS; ++c) {
        noc_async_read_barrier_with_trid(c + 2);
        const uint32_t k1 = (c + 1) * KC < K ? (c + 1) * KC : K;
        cb_push_back(cb_w, (k1 - c * KC) * NC);
    }
    noc_async_read_set_trid(0);
}
