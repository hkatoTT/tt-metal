// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's MLP gate (dflash2_ops.silu_mul), the writer: the output tiles t of the core's run, in order.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t CB_O = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr uint32_t BF = 2048;
    const auto s_o = TensorAccessor(o_args, out_addr, BF);
    for (uint32_t t = t0; t < t1; ++t) {
        cb_wait_front(CB_O, 1);
        noc_async_write(get_read_ptr(CB_O), s_o.get_noc_addr(t), BF);
        noc_async_writes_flushed();
        cb_pop_front(CB_O, 1);
    }
    noc_async_write_barrier();
}
