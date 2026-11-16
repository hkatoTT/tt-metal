// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dsa_indexer.bfp_pack: bf16 tiles -> BFP8_b / BFP4_b tiles, rounded to nearest (ties away from zero) by the core
// (bfp_rows.hpp; the packer's BFP4 truncates), for the pools' keys a prefill chunk writes into a BFP pools cache.  Tiles
// t0 .. t1 - 1 of the input [.., H, W] (interleaved pages) to the same pages of the output; a bf16 tile read ahead
// while the previous one is converted, two output slots written in turn.  The op runs this kernel on both data
// movement cores of a core, each its own range and its own circular buffer.
//   CT: BITS (8 / 4), CB (4 KB of bf16 slots, then 2 output slots), the accessors of the input and of the output.
//   RT: the input's and the output's addresses, t0, t1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "bfp_rows.hpp"

void kernel_main() {
    constexpr uint32_t BITS = get_compile_time_arg_val(0);
    constexpr uint32_t CB = get_compile_time_arg_val(1);
    constexpr auto in_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t IN = 2048, OUT = dsa_bfp::tile_bytes(BITS);
    static_assert(in_args.AlignedPageSize == IN && out_args.AlignedPageSize == OUT);
    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(1));
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t t1 = get_arg_val<uint32_t>(3);
    if (t1 <= t0) {
        return;
    }
    const uint32_t l1 = get_write_ptr(CB);
    const uint32_t src[2] = {l1, l1 + IN};
    const uint32_t dst[2] = {l1 + 2 * IN, l1 + 2 * IN + OUT};
    noc_async_read_page(t0, in, src[0]);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t k = (t - t0) % 2;
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (t + 1 < t1) {
            noc_async_read_page(t + 1, in, src[1 - k]);
        }
        noc_async_writes_flushed();  // (slot k's write, two tiles back, out of L1)
        dsa_bfp::pack_tile<BITS>(src[k], dst[k]);
        noc_async_write_page(t, out, dst[k]);
    }
    noc_async_write_barrier();
}
