// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's selector terms (dflash2_ops.sel_scores), the reader: per item (tile row) of [i0, i1), its K tiles of the
// predecessors' rows, the spread proj and the successors' rows ([rows, K 32] bf16 tiled each).
void kernel_main() {
    const uint32_t pr_addr = get_arg_val<uint32_t>(0);
    const uint32_t pj_addr = get_arg_val<uint32_t>(1);
    const uint32_t su_addr = get_arg_val<uint32_t>(2);
    const uint32_t i0 = get_arg_val<uint32_t>(3);
    const uint32_t i1 = get_arg_val<uint32_t>(4);
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t CB_PR = get_compile_time_arg_val(1);
    constexpr uint32_t CB_PJ = get_compile_time_arg_val(2);
    constexpr uint32_t CB_SU = get_compile_time_arg_val(3);
    constexpr auto a_args = TensorAccessorArgs<4>();
    constexpr uint32_t BF = 2048;
    const auto s_pr = TensorAccessor(a_args, pr_addr, BF);
    const auto s_pj = TensorAccessor(a_args, pj_addr, BF);
    const auto s_su = TensorAccessor(a_args, su_addr, BF);
    for (uint32_t item = i0; item < i1; ++item) {
        cb_reserve_back(CB_PR, KT);
        cb_reserve_back(CB_PJ, KT);
        cb_reserve_back(CB_SU, KT);
        for (uint32_t k = 0; k < KT; ++k) {
            const uint32_t t = item * KT + k;
            noc_async_read(s_pr.get_noc_addr(t), get_write_ptr(CB_PR) + k * BF, BF);
            noc_async_read(s_pj.get_noc_addr(t), get_write_ptr(CB_PJ) + k * BF, BF);
            noc_async_read(s_su.get_noc_addr(t), get_write_ptr(CB_SU) + k * BF, BF);
        }
        noc_async_read_barrier();
        cb_push_back(CB_PR, KT);
        cb_push_back(CB_PJ, KT);
        cb_push_back(CB_SU, KT);
    }
}
