// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "mp_common.hpp"

// The prefill MoE's combine, the reader (mp_common.hpp): per item (token tile row, column block), the 32 tokens' pair
// rows; for each slot j below the most pairs any of them has, a block of the CB_S ring (reserved whole each item):
// token y's j-th pair's Y row (the block's 512 fp32) at row y, zeros for a token with fewer pairs.
void kernel_main() {
    using namespace moe_prefill;
    uint32_t a = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tpair_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_items = get_arg_val<uint32_t>(a++);

    constexpr uint32_t y_row = get_compile_time_arg_val(0);
    constexpr auto y_args = TensorAccessorArgs<1>();
    constexpr auto p_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    // (the program cache key N at p_args' next offset), then DNCB_NOCS 2: the Y rows of the tokens y with y mod 4
    // below q1 (DNCB_NOC1_QUARTERS) read on NoC 1, once the writer (BRISC, NoC 1, which reads nothing) has handed NoC 1
    // over (mp_noc1.hpp, S_NOC1); they carry transaction id 1 and are waited for by it
    constexpr uint32_t nocs = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t q1 = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 2);
    // (tpair a page a token tile row, 2 KB: the item's 32 tokens' pair rows in one read)
    // Y_BFP8: Y8 records (mp_dn_writer.cpp: a pair row's 1,024 columns a flat bfp8 piece); an item is a block of
    // CB_COLS8 columns and CB_S's slot j holds token y's record in piece 32 j + y: one read of Y8_REC bytes (the
    // record, and the piece from a CB's 64-byte aligned address: offsets 0 mod 64)
    constexpr bool y8 = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 3) != 0;
    constexpr uint32_t COLS = y8 ? CB_COLS8 : CB_COLS;
    constexpr uint32_t PIECES = y8 ? 32 : CB_PIECES;  // CB_S's pieces a slot
    constexpr uint32_t PAGE = ROW_WORDS * 4;
    constexpr uint32_t BLK = y_row / (4096 / CB_COLS);  // a row's block: 512 values (fp32, or bf16: Y_BF16)
    constexpr uint32_t PIECE = y_row / 4;               // a flat piece: 1,024 values

    const auto s_y = TensorAccessor(y_args, y_addr, y_row);
    const auto s_p = TensorAccessor(p_args, tpair_addr, 32 * PAGE);
    const uint32_t tp_l1 = get_write_ptr(CB_SCR);
    volatile tt_l1_ptr uint32_t* tp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tp_l1);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    const uint64_t zeros64 = get_noc_addr((MEM_ZEROS_BASE + 63) & ~63u);  // (Y_BFP8's 32-byte reads: offsets mod 64)
    const uint8_t other = 1 - noc_index;
    uint8_t n2 = noc_index;
    if constexpr (nocs == 2) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(S_NOC1)), 1);
        noc_async_read_set_trid(1, other);
        n2 = other;
    }

    for (uint32_t i = 0; i < n_items; ++i) {
        const uint32_t item = get_arg_val<uint32_t>(a + i);
        const uint32_t tr = item / (4096 / COLS), blk = item % (4096 / COLS);
        noc_async_read(s_p.get_noc_addr(tr), tp_l1, 32 * PAGE);  // the 32 tokens' pair rows
        noc_async_read_barrier();
        uint32_t slots = 0;
        for (uint32_t y = 0; y < 32; ++y) {
            uint32_t k = 0;
            while (k < K && tp[y * ROW_WORDS + k] != SENT) {
                ++k;
            }
            slots = k > slots ? k : slots;
        }
        cb_reserve_back(CB_SL, 1);  // the slots to the writer (for the compute)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_SL)) = slots;
        cb_push_back(CB_SL, 1);
        // the whole ring every item (an item's slots must not wrap around its end: the compute indexes them)
        cb_reserve_back(CB_S, K * PIECES);
        const uint32_t ring = get_write_ptr(CB_S);
        for (uint32_t j = 0; j < slots && y8; ++j) {
            for (uint32_t y = 0; y < 32; ++y) {
                const uint32_t piece = ring + (j * PIECES + y) * Y8_PIECE;
                const uint32_t p = tp[y * ROW_WORDS + j];
                if (p != SENT) {
                    const uint8_t q = (y & 3) < q1 ? n2 : noc_index;
                    noc_async_read(s_y.get_noc_addr(p, blk * Y8_REC, q), piece, Y8_REC, q);
                } else {  // (zeros: the exponents, then the mantissas)
                    noc_async_read(zeros64, piece, 64);
                    noc_async_read(zeros, piece + 64, 512);
                    noc_async_read(zeros, piece + 576, 512);
                }
            }
        }
        for (uint32_t j = 0; j < slots && !y8; ++j) {
            const uint32_t dst = ring + j * CB_PIECES * PIECE;
            for (uint32_t y = 0; y < 32; ++y) {
                const uint32_t p = tp[y * ROW_WORDS + j];
                if (p != SENT) {
                    const uint8_t q = (y & 3) < q1 ? n2 : noc_index;
                    noc_async_read(s_y.get_noc_addr(p, blk * BLK, q), dst + y * BLK, BLK, q);
                } else {
                    for (uint32_t z = 0; z < BLK; z += 512) {
                        noc_async_read(zeros, dst + y * BLK + z, 512);
                    }
                }
            }
        }
        noc_async_read_barrier();
        if constexpr (nocs == 2) {
            noc_async_read_barrier_with_trid(1, other);
        }
        cb_push_back(CB_S, K * PIECES);
    }
    if constexpr (nocs == 2) {
        noc_async_read_set_trid(0, other);
    }
}
