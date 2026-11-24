// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2's selector terms with the codebook rows gathered here (dflash2_ops.sel_scores_ids, GLM53_DFLASH2_SEL_FUSED),
// the reader: per item of [i0, i1), its 32 predecessor ids and 32 candidate ids (the
// rows of pred / cand [1, items 32] uint32), those rows of the two codebooks ([V, rank] bf16 row-major: a page a row)
// into the row-major blocks CB_PRR / CB_SUR (32 rows of rank, as ttnn.embedding gathers them; the compute tilizes
// them), and the item's K tiles of the spread proj.
void kernel_main() {
    const uint32_t pid_addr = get_arg_val<uint32_t>(0);
    const uint32_t pj_addr = get_arg_val<uint32_t>(1);
    const uint32_t sid_addr = get_arg_val<uint32_t>(2);
    const uint32_t ptab_addr = get_arg_val<uint32_t>(3);
    const uint32_t stab_addr = get_arg_val<uint32_t>(4);
    const uint32_t i0 = get_arg_val<uint32_t>(5);
    const uint32_t i1 = get_arg_val<uint32_t>(6);
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t CB_PRR = get_compile_time_arg_val(1);
    constexpr uint32_t CB_PJ = get_compile_time_arg_val(2);
    constexpr uint32_t CB_SUR = get_compile_time_arg_val(3);
    constexpr uint32_t CB_IDS = get_compile_time_arg_val(4);
    constexpr uint32_t IDS_PAGE = get_compile_time_arg_val(5);  // the ids' one page (items 32 x 4 bytes)
    constexpr auto pj_args = TensorAccessorArgs<6>();
    constexpr auto id_args = TensorAccessorArgs<pj_args.next_compile_time_args_offset()>();
    constexpr auto tab_args = TensorAccessorArgs<id_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048, ROW = KT * 64;  // a tile; a codebook row (rank bf16)
    const auto s_pj = TensorAccessor(pj_args, pj_addr, BF);
    const auto s_pid = TensorAccessor(id_args, pid_addr, IDS_PAGE);
    const auto s_sid = TensorAccessor(id_args, sid_addr, IDS_PAGE);
    const auto s_ptab = TensorAccessor(tab_args, ptab_addr, ROW);
    const auto s_stab = TensorAccessor(tab_args, stab_addr, ROW);
    const uint32_t ids = get_write_ptr(CB_IDS);  // [pred 32 | cand 32] uint32 of the item
    for (uint32_t item = i0; item < i1; ++item) {
        noc_async_read(s_pid.get_noc_addr(0) + item * 128, ids, 128);
        noc_async_read(s_sid.get_noc_addr(0) + item * 128, ids + 128, 128);
        cb_reserve_back(CB_PJ, KT);
        for (uint32_t k = 0; k < KT; ++k) {
            noc_async_read(s_pj.get_noc_addr(item * KT + k), get_write_ptr(CB_PJ) + k * BF, BF);
        }
        noc_async_read_barrier();
        const volatile tt_l1_ptr uint32_t* id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids);
        cb_reserve_back(CB_PRR, KT);
        cb_reserve_back(CB_SUR, KT);
        const uint32_t pr = get_write_ptr(CB_PRR), su = get_write_ptr(CB_SUR);
        for (uint32_t j = 0; j < 32; ++j) {
            noc_async_read(s_ptab.get_noc_addr(id[j]), pr + j * ROW, ROW);
            noc_async_read(s_stab.get_noc_addr(id[32 + j]), su + j * ROW, ROW);
        }
        noc_async_read_barrier();
        cb_push_back(CB_PJ, KT);
        cb_push_back(CB_PRR, KT);
        cb_push_back(CB_SUR, KT);
    }
}
