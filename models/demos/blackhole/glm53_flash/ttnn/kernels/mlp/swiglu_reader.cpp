// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// The TP4 MLP's clamped SwiGLU from its gate | up matmul's output (swiglu_split.py), the reader: the core's output tiles
// idx in [i0, i1) in row-major order (tile row r = idx / Q, column c = idx % Q), for each the gate tile (r, c) of
// gate_up [.., S, 2 Q tiles] into CB_A and the up tile (r, Q + c) into CB_B (binary_ng's c_0 and c_1: its compute
// kernel runs unchanged on them).
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t i0 = get_arg_val<uint32_t>(1);
    const uint32_t i1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t Q = get_compile_time_arg_val(0);  // gate's (and up's) tile columns
    constexpr uint32_t CB_A = get_compile_time_arg_val(1);
    constexpr uint32_t CB_B = get_compile_time_arg_val(2);
    constexpr auto s_args = TensorAccessorArgs<3>();
    const uint32_t tile = get_tile_size(CB_A);
    const auto s = TensorAccessor(s_args, src_addr, tile);
    for (uint32_t idx = i0; idx < i1; ++idx) {
        const uint32_t page = (idx / Q) * 2 * Q + idx % Q;
        cb_reserve_back(CB_A, 1);
        cb_reserve_back(CB_B, 1);
        noc_async_read(s.get_noc_addr(page), get_write_ptr(CB_A), tile);
        noc_async_read(s.get_noc_addr(page + Q), get_write_ptr(CB_B), tile);
        noc_async_read_barrier();
        cb_push_back(CB_A, 1);
        cb_push_back(CB_B, 1);
    }
}
