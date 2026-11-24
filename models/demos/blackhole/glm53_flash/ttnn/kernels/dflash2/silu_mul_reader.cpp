// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's MLP gate (dflash2_ops.silu_mul), the reader: per output tile t of the core's run [t0, t1) (tile row i =
// t / IT, column j = t % IT) the gate's tile (i, j) and the up's (i, IT + j) of gate | up [M, 2 I] bf16 tiled.
void kernel_main() {
    const uint32_t gu_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t IT = get_compile_time_arg_val(0);  // I / 32
    constexpr uint32_t CB_G = get_compile_time_arg_val(1);
    constexpr uint32_t CB_U = get_compile_time_arg_val(2);
    constexpr auto gu_args = TensorAccessorArgs<3>();
    constexpr uint32_t BF = 2048;
    const auto s_gu = TensorAccessor(gu_args, gu_addr, BF);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t i = t / IT, j = t % IT;
        cb_reserve_back(CB_G, 1);
        cb_reserve_back(CB_U, 1);
        noc_async_read(s_gu.get_noc_addr(i * 2 * IT + j), get_write_ptr(CB_G), BF);
        noc_async_read(s_gu.get_noc_addr(i * 2 * IT + IT + j), get_write_ptr(CB_U), BF);
        noc_async_read_barrier();
        cb_push_back(CB_G, 1);
        cb_push_back(CB_U, 1);
    }
}
