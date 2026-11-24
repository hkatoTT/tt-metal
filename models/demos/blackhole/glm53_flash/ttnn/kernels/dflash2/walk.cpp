// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "top16_common.hpp"

// DFlash2's selector walk (dflash2_ops.walk): for each of this core's users u, from the anchor (the bonus token),
// slot s = 0 .. K - 1: the scores of the edges from the previous pick p to the slot's 16 candidates c are
// unary[c] + S[u K + s][p][c] (fp32, added on the RISC-V as torch adds them), the pick is the first largest, the
// draft its id.  S [1, U K, 32, 32] fp32 tiled (pairwise terms: rows the predecessors, columns the candidates; face 0
// holds the 16 x 16), sel [U K, 32] uint32 row-major (top16_merge's: the 16 values, then the ids) -> drafts [U, 8]
// uint32 row-major: the bonus token, the K drafts, zeros.  Slot 0's predecessors are all the anchor (pick 0).
void kernel_main() {
    const uint32_t s_addr = get_arg_val<uint32_t>(0);
    const uint32_t sel_addr = get_arg_val<uint32_t>(1);
    const uint32_t bonus_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t u0 = get_arg_val<uint32_t>(4);
    const uint32_t u1 = get_arg_val<uint32_t>(5);

    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t U = get_compile_time_arg_val(1);
    constexpr uint32_t cb = get_compile_time_arg_val(2);
    constexpr auto s_args = TensorAccessorArgs<3>();
    constexpr auto sel_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto bonus_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<bonus_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096, FACE = 1024, ROW = 128, OUT = 32;
    constexpr uint32_t bonus_bytes = U * 4;
    constexpr uint32_t bonus_slot = (bonus_bytes + 63) & ~63u;

    const auto s_s = TensorAccessor(s_args, s_addr, TILE);
    const auto s_sel = TensorAccessor(sel_args, sel_addr, ROW);
    const auto s_out = TensorAccessor(out_args, out_addr, OUT);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t faces = l1 + bonus_slot;  // K faces 0 (16 x 16 fp32)
    const uint32_t sels = faces + K * FACE;  // K sel rows
    const uint32_t out = sels + K * ROW;     // 64-byte slot
    noc_async_read(TensorAccessor(bonus_args, bonus_addr, bonus_bytes).get_noc_addr(0), l1, bonus_bytes);
    noc_async_read_barrier();
    const uint32_t* bonus = reinterpret_cast<const uint32_t*>(l1);

    for (uint32_t u = u0; u < u1; ++u) {
        for (uint32_t s = 0; s < K; ++s) {
            noc_async_read(s_s.get_noc_addr(u * K + s), faces + s * FACE, FACE);
            noc_async_read(s_sel.get_noc_addr(u * K + s), sels + s * ROW, ROW);
        }
        noc_async_read_barrier();
        uint32_t* o = reinterpret_cast<uint32_t*>(out);
        o[0] = bonus[u];
        uint32_t prev = 0;
        for (uint32_t s = 0; s < K; ++s) {
            const float* row = reinterpret_cast<const float*>(faces + s * FACE) + prev * 16;
            const float* unary = reinterpret_cast<const float*>(sels + s * ROW);
            const uint32_t* ids = reinterpret_cast<const uint32_t*>(sels + s * ROW) + TOPN;
            uint32_t best = 0;
            int32_t best_key = 0;
            for (uint32_t c = 0; c < TOPN; ++c) {
                const float score = unary[c] + row[c];
                uint32_t bits;
                __builtin_memcpy(&bits, &score, 4);
                const int32_t k = key_of(bits);
                if (c == 0 || k > best_key) {
                    best_key = k;
                    best = c;
                }
            }
            o[1 + s] = ids[best];
            prev = best;
        }
        for (uint32_t j = 1 + K; j < 8; ++j) {
            o[j] = 0;
        }
        noc_async_write(out, s_out.get_noc_addr(u), OUT);
        noc_async_write_barrier();
    }
}
