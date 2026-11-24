// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The split sparse attention's combine (dsa_indexer.sparse_combine), the writer: the unit's C out tiles into the
// output [1, S, 16, V_DHT * 32] (tiles; a token's 32 query rows, the first 16 the heads, one tile row a token).
//   CT: S, V_DHT, C, Q, CB_OUT, then the output's accessor.  RT: the output's address, the unit (>= S * Q: idle)

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr uint32_t Q = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t u_first = get_arg_val<uint32_t>(1);  // this core's units u_first .. u_first + n_units - 1
    const uint32_t n_units = get_arg_val<uint32_t>(2);
    constexpr uint32_t tile_bytes = tt::constants::TILE_HW * 2;
    static_assert(out_args.AlignedPageSize == tile_bytes && C * Q == vDHt);
    const auto out = TensorAccessor(out_args, out_addr);
    for (uint32_t k = 0; k < n_units; ++k) {
        const uint32_t unit = u_first + k;
        if (unit >= S * Q) {
            continue;
        }
        const uint32_t tok = unit / Q;
        const uint32_t col0 = (unit % Q) * C;
        for (uint32_t c = 0; c < C; ++c) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(tok * vDHt + col0 + c, out, get_read_ptr(cb_out), tile_bytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
