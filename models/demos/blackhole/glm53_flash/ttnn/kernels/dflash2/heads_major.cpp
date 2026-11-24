// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// dflash2_ops.heads_major (GLM53_DFLASH2_CTX_HEADS): out's tiles [i0, i1), tile i = (h RT + rt) 4 + c the source's
// tile rt src_wt + 4 h + c (bf16), all of this core's reads first, one barrier, then the writes.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t i0 = get_arg_val<uint32_t>(2);
    const uint32_t i1 = get_arg_val<uint32_t>(3);
    constexpr uint32_t RT = get_compile_time_arg_val(0);      // the source's tile rows
    constexpr uint32_t src_wt = get_compile_time_arg_val(1);  // the source's tiles across
    constexpr uint32_t cb = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048, WT = 4;
    const auto s_src = TensorAccessor(src_args, src_addr, TILE);
    const auto s_out = TensorAccessor(out_args, out_addr, TILE);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t i = i0; i < i1; ++i) {
        const uint32_t c = i % WT, hr = i / WT, h = hr / RT, rt = hr % RT;
        noc_async_read(s_src.get_noc_addr(rt * src_wt + h * WT + c), l1 + (i - i0) * TILE, TILE);
    }
    noc_async_read_barrier();
    for (uint32_t i = i0; i < i1; ++i) {
        noc_async_write(l1 + (i - i0) * TILE, s_out.get_noc_addr(i), TILE);
    }
    noc_async_write_barrier();
}
