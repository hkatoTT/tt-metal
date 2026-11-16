// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per piece (unit = (head h, block of 16 users), rows row0 .. row0 + rows - 1, row0 a multiple of 4): rows 0 .. of
// faces 0-1 of the 4 bf16 output tiles, written as the piece's whole 4-row groups of the output's faces (columns
// h 128 ..; 128-byte pieces at 128-byte offsets: no 64-byte span is shared with another piece).
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);  // the pieces, then (unit, row0, rows) each

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t users = get_compile_time_arg_val(1);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(2);  // per tile row
    constexpr auto out_args = TensorAccessorArgs<3>();
    constexpr uint32_t blocks = (users + 15) / 16;
    constexpr uint32_t N = 4, F16 = 512;

    const auto s_out = TensorAccessor(out_args, out_addr);
    const uint32_t t16 = get_local_cb_interface(cb_out).fifo_page_size;

    CircularBuffer out(cb_out);
    for (uint32_t pc = 0; pc < count; ++pc) {
        const uint32_t u = get_arg_val<uint32_t>(2 + 3 * pc);
        const uint32_t row0 = get_arg_val<uint32_t>(3 + 3 * pc);
        const uint32_t rows = get_arg_val<uint32_t>(4 + 3 * pc);
        const uint32_t g_bytes = ((rows + 3) / 4) * 4 * 32;
        const uint32_t block = u % blocks;
        const uint32_t h = u / blocks;
        {
            DeviceZoneScopedN("FW-WAIT");
            out.wait_front(N);
        }
        DeviceZoneScopedN("FW-OUT");
        const uint32_t src = out.get_read_ptr();
        for (uint32_t c = 0; c < N; ++c) {
            const uint32_t page = (block / 2) * out_tiles + h * N + c;
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_write(
                    src + c * t16 + half * F16,
                    s_out.get_noc_addr(page, (block % 2) * 2 * F16 + half * F16 + row0 * 32),
                    g_bytes);
            }
        }
        noc_async_write_barrier();
        out.pop_front(N);
    }
}
