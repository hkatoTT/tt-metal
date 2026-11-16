// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"

// recurrent_decode_step's chunked verify form: per pair batch (as the reader), per unit (pair p, value block j): its
// O' tile's rows 16 + T + k (token k's o, k < T) to row p + k P of o's tile j (64-byte pieces), then its 4
// tiles of the step's start state (after the redone tokens: the next step redoes from it) over the state.
namespace {
constexpr uint32_t FACE = 1024;
constexpr uint32_t ROWB = 64;
constexpr uint32_t N = 4;
constexpr uint32_t CB_O = 24, CB_SO = 25;
}  // namespace

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t pairs = get_compile_time_arg_val(1);
    constexpr auto state_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();

    const auto s_state = TensorAccessor(state_args, state_addr);
    const auto s_out = TensorAccessor(out_args, out_addr);
    CircularBuffer o(CB_O), so(CB_SO);
    const uint32_t s_tile = get_local_cb_interface(CB_SO).fifo_page_size;

    for (uint32_t u = start; u < start + count;) {
        const uint32_t p = u / N;
        const uint32_t j0 = u % N;
        const uint32_t nb = N - j0 < start + count - u ? N - j0 : start + count - u;
        for (uint32_t b = 0; b < nb; ++b) {
            const uint32_t j = j0 + b;
            {
                DeviceZoneScopedN("CW-WAIT");
                o.wait_front(1);
            }
            const uint32_t oa = o.get_read_ptr();
            for (uint32_t k = 0; k < T; ++k) {
                const uint32_t row = p + k * pairs, rr = row % 32;
                const uint32_t lo = (rr / 16) * 2 * FACE + (rr % 16) * ROWB;
                const uint32_t page = (row / 32) * N + j;
                const uint32_t src = oa + 2 * FACE + (T + k) * ROWB;  // O' row 16 + T + k: faces 2 and 3
                noc_async_write(src, s_out.get_noc_addr(page, lo), ROWB);
                noc_async_write(src + FACE, s_out.get_noc_addr(page, lo + FACE), ROWB);
            }
            noc_async_writes_flushed();
            o.pop_front(1);
            so.wait_front(N);
            const uint32_t sa = so.get_read_ptr();
            for (uint32_t i = 0; i < N; ++i) {
                noc_async_write(sa + i * s_tile, s_state.get_noc_addr(p * N * N + i * N + j), s_tile);
            }
            noc_async_writes_flushed();
            so.pop_front(N);
        }
        u += nb;
    }
    noc_async_write_barrier();
}
