// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per work unit (pair p, value block j): the 4 new state tiles S'_ij back in place, and row 0 of the output tile into
// the pair's row of output tile j (64-byte pieces: the pairs of a 32-row tile row may be on other cores, each writing
// only its own row).
namespace {
constexpr uint32_t FACE_BYTES = 1024;
constexpr uint32_t FACE_ROW_BYTES = 64;
constexpr uint32_t N = 4;
}  // namespace

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_sout = get_compile_time_arg_val(0);
    constexpr uint32_t cb_o = get_compile_time_arg_val(1);
    constexpr auto state_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();

    const uint32_t tile = get_local_cb_interface(cb_sout).fifo_page_size;
    const auto s_state = TensorAccessor(state_args, state_addr);
    const auto s_out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer sout(cb_sout), o(cb_o);

#ifdef KDA_STEP_ROTATE  // (form bit 128: a pair's value blocks in the reader's order, rotated by the core)
    const uint32_t rot = count > 0 ? (start / count) % N : 0;
#endif
    for (uint32_t u = start; u < start + count; ++u) {
        const uint32_t p = u / N;
#ifdef KDA_STEP_ROTATE
        // the reader's batch of u: the pair's units here, from `first` (the pair's first, or the core's), nb of them
        const uint32_t first = p * N > start ? p * N : start;
        const uint32_t nb = (p + 1) * N < start + count ? (p + 1) * N - first : start + count - first;
        const uint32_t j = (u % N - (first % N) + rot) % nb + first % N;
#else
        const uint32_t j = u % N;
#endif
        const uint32_t r = p % 32;
        const uint32_t lo = (r / 16) * 2 * FACE_BYTES + (r % 16) * FACE_ROW_BYTES;
        const uint32_t hi = lo + FACE_BYTES;
        const uint32_t page = (p / 32) * N + j;

        {
            DeviceZoneScopedN("W-WAIT");
            sout.wait_front(N);
        }
        for (uint32_t i = 0; i < N; ++i) {
            noc.async_write(sout, s_state, tile, {.offset_bytes = i * tile}, {.page_id = p * N * N + i * N + j});
        }
        o.wait_front(1);
        noc.async_write(o, s_out, FACE_ROW_BYTES, {.offset_bytes = 0}, {.page_id = page, .offset_bytes = lo});
        noc.async_write(o, s_out, FACE_ROW_BYTES, {.offset_bytes = FACE_BYTES}, {.page_id = page, .offset_bytes = hi});
        {
            DeviceZoneScopedN("W-BAR");
            noc.async_write_barrier();
        }
        sout.pop_front(N);
        o.pop_front(1);
    }
}
