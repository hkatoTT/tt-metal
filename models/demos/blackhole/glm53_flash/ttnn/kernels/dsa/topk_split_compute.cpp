// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The decode indexer's top 512 of a long row (dsa_indexer.topk_split, GLM53_DSA_TOPK_SPLIT), the
// compute: ttnn.experimental.topk_large_indices' Classic body (K = 512, more than 32 chunks of 512) with its chunks'
// local sorts spread over helper cores.  A chunk's sort (the copy, the LSB indices, the local sort, the indices
// separated with the chunk's base) depends only on the chunk, so a helper sorts it as the row's core would and sends
// the sorted chunk (the value and index tiles, fp32 / u32 bits) to the row's core; the row's core folds the chunks in
// the unsplit op's order with its merge and rebuild: the same picks in the same order, ties included.
//
// ROLE 0 (a row's core): chunks 0 .. OWN - 1 sorted here as in the unsplit op, chunks OWN .. CHUNKS - 1 unpacked to
//   Dest from CB_RECV (the helpers' sorted tiles), each followed by the inits the unsplit op's sort leaves before the
//   merge; then the unsplit op's end (the -inf picks marked, the indices in rank order, untilized to CB_IDX).
// ROLE 1 (a helper): chunks FIRST, FIRST + H, .. of its row (runtime args: FIRST, COUNT), each sorted as the unsplit
//   op's chunk at Dest slot 2 (the row's incoming slot) and separated with its base (LREG12: init_static once, then
//   the base advanced K a chunk as the unsplit op's row core advances it), its value and index tiles packed to
//   CB_SEND.
// A helper ends with one more Dest section, a transpose_dest of Dest tile 1, as the row's core (and
// topk_large_indices) end.  The local sort's transposes are Dest moves (MOVD2B / MOVB2D) inside topk_xl's transpose CFG
// block, under ALU_ACC_CTRL_Zero_Flag_disabled_dst = 1; a kernel that ends after them, with no Dest move of the
// Matrix Unit once the bit is back to 0, leaves the next program's first Dest-reading Matrix Unit instruction ignoring
// Dest's zero flags: the next matmul's first output tile's rows 0-7 x 16 columns get the stale Dest data its ZEROACC
// only flagged zero.  Any plain Dest move clears it: the closing transpose_dest is one.
// Compile-time args: ROLE, CB_IN, CB_IDX, CB_SEND, CB_RECV, OWN, H, CHUNKS.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// (as topk_large_indices' compute.cpp: an index whose value is -inf becomes 0xFFFFFFFF)
inline void _topk_split_mark_neginf_indices_init_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);
}

template <uint32_t K>
inline void _topk_split_mark_neginf_indices_() {
    static_assert(K == 512, "K must be 512");
    constexpr uint32_t tiles_per_sequence = 1;
    constexpr uint32_t indices_offset = tiles_per_sequence * 64;
    constexpr uint32_t iterations = 16;

    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0xFFFF);

    for (uint32_t i = 0; i < iterations; ++i) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

}  // namespace ckernel::sfpu
#endif

namespace {

constexpr uint32_t K = 512;
constexpr uint32_t SURVIVOR = 0;  // the row's top K: value tile 0, index tile 1
constexpr uint32_t INCOMING = 2;  // a chunk being merged: value tile 2, index tile 3

template <uint32_t CB_IN>
FORCE_INLINE void sort_chunk(uint32_t dst, bool ascending) {
    cb_wait_front(CB_IN, 1);
    topk_xl_copy_tile_init(CB_IN);
    topk_xl_copy_tile<K>(CB_IN, dst, 0, K);
    cb_pop_front(CB_IN, 1);

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<K, 0>(dst);
    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst, ascending);
}

FORCE_INLINE void merge_incoming() {
    topk_xl_init<K, false>();
    topk_xl_merge<K, false>(SURVIVOR);
    topk_xl_rebuild<K, false>(SURVIVOR, false);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t CB_IN = get_compile_time_arg_val(1);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(2);
    constexpr uint32_t CB_SEND = get_compile_time_arg_val(3);
    constexpr uint32_t CB_RECV = get_compile_time_arg_val(4);
    constexpr uint32_t OWN = get_compile_time_arg_val(5);
    constexpr uint32_t H = get_compile_time_arg_val(6);
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(7);
    static_assert(OWN >= 1 && OWN < CHUNKS && CHUNKS > 32, "the Classic body's rows: chunk 0 on the row's core");

    if constexpr (ROLE == 0) {
        compute_kernel_hw_startup(CB_IN, CB_IDX);
        pack_untilize_dest_init<1, 1>(CB_IDX);

        tile_regs_acquire();
        topk_xl_separate_indices_row_major_init_static<0, 0>();
        // chunk 0 (descending: the survivor) and chunks 1 .. OWN - 1 here, as in the unsplit op
        sort_chunk<CB_IN>(SURVIVOR, false);
        topk_xl_separate_indices_row_major_reinit();
        topk_xl_separate_indices_row_major<K>(SURVIVOR);
        topk_xl_separate_indices_row_major_advance_chunk_base<K>();
        for (uint32_t chunk = 1; chunk < OWN; ++chunk) {
            sort_chunk<CB_IN>(INCOMING, true);
            topk_xl_separate_indices_row_major_reinit();
            topk_xl_separate_indices_row_major<K>(INCOMING);
            topk_xl_separate_indices_row_major_advance_chunk_base<K>();
            merge_incoming();
        }
        // chunks OWN .. CHUNKS - 1: sorted by the helpers, in the unsplit op's order
        for (uint32_t chunk = OWN; chunk < CHUNKS; ++chunk) {
            cb_wait_front(CB_RECV, 2);
            reconfig_data_format_srca(CB_IN, CB_RECV);
            copy_init(CB_RECV);
            copy_tile(CB_RECV, 0, INCOMING);
            copy_tile(CB_RECV, 1, INCOMING + 1);
            cb_pop_front(CB_RECV, 2);
            // the state the unsplit op's sort and separate leave before the merge's init
            topk_xl_copy_tile_init(CB_IN);
            topk_xl_add_lsb_indices_init();
            topk_xl_init<K, true>();
            topk_xl_separate_indices_row_major_reinit();
            merge_incoming();
        }

        MATH((ckernel::sfpu::_topk_split_mark_neginf_indices_init_()));
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_topk_split_mark_neginf_indices_<K>, SURVIVOR, VectorMode::RC_custom)));
        transpose_dest_init<true, false>(CB_IDX);
        transpose_dest<true, false>(SURVIVOR + 1);

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_IDX, 1);
        pack_untilize_dest<1, 1>(CB_IDX, 1, 0, SURVIVOR + 1);
        cb_push_back(CB_IDX, 1);
        tile_regs_release();
    } else {
        const uint32_t first = get_arg_val<uint32_t>(0);
        const uint32_t count = get_arg_val<uint32_t>(1);
        compute_kernel_hw_startup(CB_IN, CB_SEND);
        for (uint32_t t = 0; t < count; ++t) {
            tile_regs_acquire();
            if (t == 0) {  // the row's state as the unsplit row core sets it (init_static), the base advanced to FIRST
                topk_xl_separate_indices_row_major_init_static<0, 0>();
                for (uint32_t c = 0; c < first; ++c) {
                    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
                }
            }
            sort_chunk<CB_IN>(INCOMING, true);
            // (the chunk base in LREG12 as the unsplit op's at this chunk: advanced K a chunk, never loaded from a
            // runtime value - the LLK's base init takes it as an asm immediate)
            topk_xl_separate_indices_row_major_reinit();
            topk_xl_separate_indices_row_major<K>(INCOMING);
            for (uint32_t c = 0; c < H; ++c) {  // the next chunk of this helper: H chunks on
                topk_xl_separate_indices_row_major_advance_chunk_base<K>();
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_SEND, 2);
            pack_tile(INCOMING, CB_SEND);
            pack_tile(INCOMING + 1, CB_SEND);
            cb_push_back(CB_SEND, 2);
            tile_regs_release();
        }
    }
    if constexpr (ROLE == 1) {  // (the row's core ends in transpose_dest already)
        tile_regs_acquire();
        transpose_dest_init<true, false>(CB_SEND);
        transpose_dest<true, false>(SURVIVOR + 1);
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
    }
}
