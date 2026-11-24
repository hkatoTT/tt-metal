// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// The DSA decode's per-head absorb matmul's output (ttnn/dsa_absorb.py; RISCV_1, NoC 1): this core's subblocks of CW
// tiles, tile row r0 + r of head h, columns n0 + sb CW .., to the output's pages (h RT + r0 + r) N + n (interleaved,
// DRAM or L1).
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t h = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);
    const uint32_t r0 = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr uint32_t CW = get_compile_time_arg_val(4);
    constexpr uint32_t RT = get_compile_time_arg_val(5);  // the output's tile rows (a head's)
    constexpr auto o_args = TensorAccessorArgs<6>();
    const auto s_o = TensorAccessor(o_args, out_addr);
    const uint32_t ot = get_local_cb_interface(cb_out).fifo_page_size;
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t sb = 0; sb < NC / CW; ++sb) {
            cb_wait_front(cb_out, CW);
            const uint32_t l1 = get_read_ptr(cb_out);
            const uint32_t page = (h * RT + r0 + r) * N + n0 + sb * CW;
            for (uint32_t j = 0; j < CW; ++j) {
                noc_async_write(l1 + j * ot, s_o.get_noc_addr(page + j), ot);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, CW);
        }
    }
    noc_async_write_barrier();
}
