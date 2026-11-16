// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's dynamic convolution (dflash2_ops.conv, the rows form), the writer: the output tiles (i, j) in the compute's
// order (tile row by tile row), each flushed before its page is freed, the acks waited once at the end.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const uint32_t j1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t XW = get_compile_time_arg_val(1);
    constexpr uint32_t CB_O = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    constexpr uint32_t BF = 2048;
    const auto s_o = TensorAccessor(o_args, out_addr, BF);
    for (uint32_t i = 0; i < R; ++i) {
        for (uint32_t j = j0; j < j1; ++j) {
            cb_wait_front(CB_O, 1);
            noc_async_write(get_read_ptr(CB_O), s_o.get_noc_addr(i * XW + j), BF);
            noc_async_writes_flushed();
            cb_pop_front(CB_O, 1);
        }
    }
    noc_async_write_barrier();
}
