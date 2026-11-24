// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dsa_indexer.bfp_cast, the writer: CB_OUT's tiles (the output's format) to tiles [t0, t1) of the output.
//   CT: CB_OUT, then the output's accessor.  RT: the address, t0, t1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    for (uint32_t t = t0; t < t1; ++t) {
        cb_wait_front(CB_OUT, 1);
        noc_async_write_page(t, out, get_read_ptr(CB_OUT));
        noc_async_writes_flushed();
        cb_pop_front(CB_OUT, 1);
    }
    noc_async_write_barrier();
}
