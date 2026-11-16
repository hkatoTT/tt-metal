// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// The TP4 MLP's clamped SwiGLU (swiglu_split.py), the writer: the core's output tiles idx in [i0, i1) (row-major, as
// the reader's) from CB_O to page idx of h [.., S, Q tiles].
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t i0 = get_arg_val<uint32_t>(1);
    const uint32_t i1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t CB_O = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const uint32_t tile = get_tile_size(CB_O);
    const auto s = TensorAccessor(o_args, out_addr, tile);
    for (uint32_t idx = i0; idx < i1; ++idx) {
        cb_wait_front(CB_O, 1);
        noc_async_write(get_read_ptr(CB_O), s.get_noc_addr(idx), tile);
        noc_async_writes_flushed();
        cb_pop_front(CB_O, 1);
    }
    noc_async_write_barrier();
}
