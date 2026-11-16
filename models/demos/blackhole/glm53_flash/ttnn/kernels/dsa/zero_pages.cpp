// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dsa_indexer.zeros_bytes: pages [p0, p1) of a tensor (any dtype: FP8 rows, BFP tiles) set to zero bytes (0.0 in
// every one of those formats), from a zeroed page of this core's L1.
//   CT: CB (a page of scratch), the page's bytes, then the tensor's accessor.  RT: the address, p0, p1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB = get_compile_time_arg_val(0);
    constexpr uint32_t PAGE = get_compile_time_arg_val(1);
    constexpr auto args = TensorAccessorArgs<2>();
    const auto t = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t p0 = get_arg_val<uint32_t>(1);
    const uint32_t p1 = get_arg_val<uint32_t>(2);
    const uint32_t l1 = get_write_ptr(CB);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < PAGE / 4; ++i) {
        z[i] = 0;
    }
    for (uint32_t p = p0; p < p1; ++p) {
        noc_async_write_page(p, t, l1);
        if ((p - p0) % 64 == 63) {
            noc_async_writes_flushed();
        }
    }
    noc_async_write_barrier();
}
