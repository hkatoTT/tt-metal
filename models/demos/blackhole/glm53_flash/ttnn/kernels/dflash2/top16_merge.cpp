// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "top16_common.hpp"

// DFlash2's candidates over the mesh (dflash2_ops.top16_merge): the 32 chips' lists (top16's tiles, gathered: chip k's
// block of N rows [chips, N, 32] fp32 tiled, row n: the 16 values in columns 0-15 and the 16 ids in 16-31, chip k's
// ids k W ..) -> for each of this core's slot rows i = u K + s of this mesh row's users (gathered row row_base + i),
// the 16 largest values (descending; equal values: the smaller id first) and their ids.  Outputs (this mesh row's):
//   sel [U K, 32] uint32 row-major: the 16 values (fp32 bits), then the 16 ids;
//   cand [1, U K 32] uint32 row-major: per slot row its 16 ids and 16 zeros (the successor codebook's rows);
//   pred [1, U K 32] uint32 row-major: per slot row the ids before it, slot 0 its user's bonus token 16 times, slot
//       s > 0 slot s - 1's candidates, and 16 zeros (the predecessor codebook's rows): slot s's core writes its
//       candidates as slot s + 1's predecessors.
void kernel_main() {
    const uint32_t g_addr = get_arg_val<uint32_t>(0);
    const uint32_t bonus_addr = get_arg_val<uint32_t>(1);
    const uint32_t sel_addr = get_arg_val<uint32_t>(2);
    const uint32_t cand_addr = get_arg_val<uint32_t>(3);
    const uint32_t pred_addr = get_arg_val<uint32_t>(4);
    const uint32_t i0 = get_arg_val<uint32_t>(5);
    const uint32_t i1 = get_arg_val<uint32_t>(6);
    const uint32_t row_base = get_arg_val<uint32_t>(7);

    constexpr uint32_t N = get_compile_time_arg_val(0);  // rows a chip's block of the gathered lists
    constexpr uint32_t K = get_compile_time_arg_val(1);  // mask slots a user
    constexpr uint32_t U = get_compile_time_arg_val(2);  // users of the mesh row
    constexpr uint32_t chips = get_compile_time_arg_val(3);
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr auto g_args = TensorAccessorArgs<5>();
    constexpr auto bonus_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto sel_args = TensorAccessorArgs<bonus_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();
    constexpr uint32_t ROW = 128, FROW = 64, FACE = 1024;  // a list row (32 x 4 bytes), an fp32 face row, face
    constexpr uint32_t bonus_bytes = U * 4;
    constexpr uint32_t bonus_slot = (bonus_bytes + 63) & ~63u;
    constexpr uint32_t ids_page = U * K * ROW;
    constexpr uint32_t tiles_a_chip = N / 32;

    const auto s_g = TensorAccessor(g_args, g_addr, 4096);
    const auto s_sel = TensorAccessor(sel_args, sel_addr, ROW);
    const auto s_cand = TensorAccessor(ids_args, cand_addr, ids_page);
    const auto s_pred = TensorAccessor(ids_args, pred_addr, ids_page);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t lists = l1 + bonus_slot;    // the item's 32 chips' rows: values (64 B) then ids (64 B)
    const uint32_t outs = lists + chips * ROW;  // sel, cand, pred rows
    noc_async_read(TensorAccessor(bonus_args, bonus_addr, bonus_bytes).get_noc_addr(0), l1, bonus_bytes);

    for (uint32_t i = i0; i < i1; ++i) {
        const uint32_t u = i / K, s = i % K;
        const uint32_t n_row = row_base + i;
        const uint32_t tile = n_row / 32, r = n_row % 32;
        const uint32_t vals_off = (r / 16) * 2 * FACE + (r % 16) * FROW;
        for (uint32_t k = 0; k < chips; ++k) {
            const uint32_t page = k * tiles_a_chip + tile;
            noc_async_read(s_g.get_noc_addr(page, vals_off), lists + k * ROW, FROW);
            noc_async_read(s_g.get_noc_addr(page, vals_off + FACE), lists + k * ROW + FROW, FROW);
        }
        noc_async_read_barrier();
        const uint32_t* bonus = reinterpret_cast<const uint32_t*>(l1);
        int32_t key[TOPN];
        uint32_t col[TOPN];
        uint32_t n = 0;
        for (uint32_t k = 0; k < chips; ++k) {
            const uint32_t* v = reinterpret_cast<const uint32_t*>(lists + k * ROW);
            const float* id = reinterpret_cast<const float*>(lists + k * ROW) + TOPN;
            for (uint32_t j = 0; j < TOPN; ++j) {
                const int32_t kk = key_of(v[j]);
                if (n == TOPN && kk <= key[TOPN - 1]) {
                    break;  // the chip's list is sorted
                }
                n = list_insert(key, col, n, kk, static_cast<uint32_t>(id[j]));
            }
        }
        uint32_t* sel = reinterpret_cast<uint32_t*>(outs);
        uint32_t* cand = reinterpret_cast<uint32_t*>(outs + ROW);
        uint32_t* pred0 = reinterpret_cast<uint32_t*>(outs + 2 * ROW);
        for (uint32_t j = 0; j < TOPN; ++j) {
            sel[j] = bits_of(key[j]);
            sel[TOPN + j] = col[j];
            cand[j] = col[j];
            cand[TOPN + j] = 0;
            pred0[j] = bonus[u];
            pred0[TOPN + j] = 0;
        }
        noc_async_write(outs, s_sel.get_noc_addr(i), ROW);
        noc_async_write(outs + ROW, s_cand.get_noc_addr(0, i * ROW), ROW);
        if (s + 1 < K) {  // this slot's candidates: the next slot's predecessors
            noc_async_write(outs + ROW, s_pred.get_noc_addr(0, (i + 1) * ROW), ROW);
        }
        if (s == 0) {
            noc_async_write(outs + 2 * ROW, s_pred.get_noc_addr(0, i * ROW), ROW);
        }
        noc_async_write_barrier();  // (the next item reuses the L1)
    }
}
