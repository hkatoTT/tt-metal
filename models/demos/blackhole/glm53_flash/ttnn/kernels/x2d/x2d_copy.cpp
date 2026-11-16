// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The 2D x all-gather (x2d_gather.py, x2d_send.cpp), a copy core's RISC: this chip's own mesh row's tokens t0 ..
// t0 + n - 1 of x [T, 4096] (row-major, a page a token) into x_all's pages r T + t (whole 8 KB pages), on this RISC's
// NoC, through NB slots of this RISC's half of the core's CB, read NB ahead by transaction ids.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);  // (its half of the CB)
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t cb_copy = get_compile_time_arg_val(3);
    constexpr auto xr_args = TensorAccessorArgs<4>();
    constexpr auto xa_args = TensorAccessorArgs<xr_args.next_compile_time_args_offset()>();
    constexpr uint32_t PAGE = 8192;
    static_assert(NB >= 1 && NB <= 15, "x2d copy: transaction ids 1 .. 15");

    const uint32_t xr_addr = get_arg_val<uint32_t>(0);
    const uint32_t xa_addr = get_arg_val<uint32_t>(1);
    const uint32_t r = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t n = get_arg_val<uint32_t>(4);
    const auto s_xr = TensorAccessor(xr_args, xr_addr);
    const auto s_xa = TensorAccessor(xa_args, xa_addr);
    const uint32_t buf = get_write_ptr(cb_copy) + role * NB * PAGE;

    auto read = [&](uint32_t i) {
        noc_async_read_set_trid(i % NB + 1);
        noc_async_read(s_xr.get_noc_addr(t0 + i), buf + (i % NB) * PAGE, PAGE);
    };
    for (uint32_t i = 0; i < NB && i < n; ++i) {
        read(i);
    }
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read_barrier_with_trid(i % NB + 1);
        noc_async_write(buf + (i % NB) * PAGE, s_xa.get_noc_addr(r * T + t0 + i), PAGE);
        if (i + NB < n) {
            noc_async_writes_flushed();  // (the slot again)
            read(i + NB);
        }
    }
    noc_async_read_set_trid(0);
    noc_async_write_barrier();
}
