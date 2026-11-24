// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dsa_indexer.bfp_cast, the reader: tiles [t0, t1) of a bf16 tiled tensor into CB_IN, one at a time.
//   CT: CB_IN, then the input's accessor.  RT: the address, t0, t1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
    constexpr auto in_args = TensorAccessorArgs<1>();
    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0));
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    for (uint32_t t = t0; t < t1; ++t) {
        cb_reserve_back(CB_IN, 1);
        noc_async_read_page(t, in, get_write_ptr(CB_IN));
        noc_async_read_barrier();
        cb_push_back(CB_IN, 1);
    }
}
