// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's head norm weight and RoPE (dflash2_ops.rope), the writer: the output tile of each item (tile row r,
// column tile t) of [i0, i1), at tile r * 4 + t; with halves (> 0: the compile-time arg after the key) the rows from
// tile row `halves` on (k's) into the second output (runtime arg 3) from its tile 0.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t i0 = get_arg_val<uint32_t>(1);
    const uint32_t i1 = get_arg_val<uint32_t>(2);
    const uint32_t out2_addr = get_arg_val<uint32_t>(3);
    constexpr uint32_t CB_O = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr uint32_t halves = get_compile_time_arg_val(o_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t BF = 2048, WT = 4;
    const auto s_o = TensorAccessor(o_args, out_addr, BF);
    [[maybe_unused]] const auto s_o2 = TensorAccessor(o_args, out2_addr, BF);
    for (uint32_t item = i0; item < i1; ++item) {
        cb_wait_front(CB_O, 1);
        if (halves && item >= halves * WT) {
            noc_async_write(get_read_ptr(CB_O), s_o2.get_noc_addr(item - halves * WT), BF);
        } else {
            noc_async_write(get_read_ptr(CB_O), s_o.get_noc_addr(item), BF);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_O, 1);
    }
}
