// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's selector terms (dflash2_ops.sel_scores), the writer: the S tile (fp32) of each item of [i0, i1).
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t i0 = get_arg_val<uint32_t>(1);
    const uint32_t i1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t CB_O = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr uint32_t F32 = 4096;
    const auto s_o = TensorAccessor(o_args, out_addr, F32);
    for (uint32_t item = i0; item < i1; ++item) {
        cb_wait_front(CB_O, 1);
        noc_async_write(get_read_ptr(CB_O), s_o.get_noc_addr(item), F32);
        noc_async_write_barrier();
        cb_pop_front(CB_O, 1);
    }
}
