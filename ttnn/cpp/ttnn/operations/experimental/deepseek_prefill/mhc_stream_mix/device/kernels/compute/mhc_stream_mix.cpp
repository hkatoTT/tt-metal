// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The mHC read after the projection, for one token tile (T <= 32; R = 4 Tp stream rows in RT row tiles):
//   1. proj, Gram per row tile = the sum of the slices' partials (I @ partial, accumulated in fp32)
//   2. own-block masks: row (t, i) keeps projection columns 24 i .. 24 i + 23; the Gram diagonal
//   3. mixes[t, m] = sum_i proj[(t, i), 24 i + m] = sum_n (sum_rt TS_rt @ masked_{rt,n}) @ BS_n
//      squares[t] = (sum_rt TS_rt @ diag_rt) @ ONES   (a token's 4 rows' sums of squares, in every column)
//   4. mixes * rsqrt(squares / (n D) + norm_eps)
//   5. pre, post, comb as mhc_split_sinkhorn (fused form)
//   6. P_rt = (pre @ REPEAT) * TS_rt: each token's pre on its own 4 rows (the block input is P @ streams)
// MHC_READ_NORM (mhc_stream_read's mix core): pre and P before comb and post, as the slice cores wait for P; then the
// block input's RMS norm scale for them, from P and the streams' Gram tiles of phase 1 (x_t = sum_i P[t, i] s_i, so
// |x_t|^2 = sum_rt ((P_rt @ G_rt) * P_rt) @ ONES: the slices need not exchange anything for it)
// The 0/1 matmuls take their fp32 operands at the matrix engine's precision, as the composite read's matmuls do.

#include "api/debug/kernel_profiler.hpp"
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "sfpi.h"
#endif

namespace {
constexpr uint32_t CB_PART = 0;
constexpr uint32_t CB_MC = 1;
constexpr uint32_t CB_SC = 2;
constexpr uint32_t CB_POST = 3;
constexpr uint32_t CB_COMB = 4;
constexpr uint32_t CB_P = 5;
constexpr uint32_t CB_PROJ = 16;
constexpr uint32_t CB_MASK = 17;
constexpr uint32_t CB_DIAG = 18;
constexpr uint32_t CB_OWN = 19;
constexpr uint32_t CB_SQP = 20;
constexpr uint32_t CB_RAW = 21;
constexpr uint32_t CB_SCALE = 22;
constexpr uint32_t CB_MIX = 23;
constexpr uint32_t CB_MA = 24;
constexpr uint32_t CB_MB = 25;
constexpr uint32_t CB_PRE = 26;
constexpr uint32_t CB_SPREAD = 27;
#ifdef MHC_READ_NORM
constexpr uint32_t CB_NSO = 7;   // the block input's norm scale, for the slice cores
constexpr uint32_t CB_P2 = 11;   // P again (CB_P goes to the slice cores)
constexpr uint32_t CB_PG = 15;   // P_rt @ G_rt
constexpr uint32_t CB_Q = 30;    // (P_rt @ G_rt) * P_rt
#endif

// mix constants (tiles of CB_MC)
constexpr uint32_t OB0 = 0, BS0 = 3, ID = 6, ONES = 7, REPEAT = 8, TS0 = 9;
// Sinkhorn constants (tiles of CB_SC), as mhc_split_sinkhorn
constexpr uint32_t SEL_PRE = 0, SEL_POST = 1, SEL_COMB = 2;
constexpr uint32_t BASE_PRE = 3, BASE_POST = 4, BASE_COMB = 5;
constexpr uint32_t RB = 6, CB_COL = 7;
constexpr uint32_t TWO_BITS = 0x40000000u;  // 2.0f
constexpr uint32_t OP_EXP = 0, OP_SIGMOID = 1, OP_RECIP = 2, OP_ADD_EPS = 3, OP_MUL2 = 4, OP_RECIP_EPS = 5, OP_COPY = 6;

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb, uint32_t index) { pack_tile<true>(dst, cb, index); }

#ifdef TRISC_MATH
// ---- the Sinkhorn of comb on the SFPU, on Dest tile 0 holding comb transposed (entry 4 i + j of a token's 4 x 4
// matrix in tile row 4 i + j, the tokens along the columns) ----
// SFPLOAD at address A moves tile rows (A & ~3) .. + 3, columns (lane & 7) * 2 + (A & 2 ? 1 : 0) of the face, so
// matrix row i of 8 tokens is one register (its 4 entries in the register's 4 rows) at 4 i + {0, 2, 16, 18} for the
// tokens {0, 2, .., 14}, {1, .., 15}, {16, .., 30}, {17, .., 31} (a slot each).  A slot's 4 rows stay in LREG4-7
// for all the iterations: a column sum is the sum of the 4 registers; for the row sums a copy in LREG0-3 is
// transposed (SFPTRANSP transposes LREG4-7 along with it, and the second transpose restores them).  The
// reciprocal is SFPARECIP and two Newton steps (LREG12 = 2.0), on LREG0 with LREG1-3 as scratch.  LREG13 = eps.
inline void recip_lreg0() {
    TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);   // y
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);                                // -x
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LREG3, 0);  // t = 2 - x y
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // y = y t
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
}

// rows: m / rowsum + eps on the first pass, m / (rowsum + eps) after (as mhc_split_sinkhorn)
template <bool FIRST>
inline void sinkhorn_rows() {
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LREG7, p_sfpu::LREG3, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);  // LREG<j> = matrix column j (row i in register row i)
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG0, 0);  // register row i: rowsum i
    if constexpr (!FIRST) {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG13, p_sfpu::LREG0, 0);
    }
    recip_lreg0();
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);  // LREG<i> = 1 / rowsum i in every register row; LREG4-7 the rows again
    constexpr uint32_t add = FIRST ? p_sfpu::LREG13 : p_sfpu::LCONST_0;
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, add, p_sfpu::LREG4, 0);
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG1, add, p_sfpu::LREG5, 0);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG2, add, p_sfpu::LREG6, 0);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG3, add, p_sfpu::LREG7, 0);
}

// columns: m / (colsum + eps)
inline void sinkhorn_cols() {
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG13, p_sfpu::LREG0, 0);
    recip_lreg0();
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);
}

template <uint32_t A, uint32_t ITERS>
inline void sinkhorn_slot() {
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, A);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, A + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, ADDR_MOD_7, A + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, A + 12);
    sinkhorn_rows<true>();
    sinkhorn_cols();
#pragma GCC unroll 0
    for (uint32_t it = 1; it < ITERS; ++it) {
        sinkhorn_rows<false>();
        sinkhorn_cols();
    }
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, A);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, A + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, ADDR_MOD_7, A + 8);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, A + 12);
}

// the slots holding tokens (the rest of the tile keeps exp(base): padding nobody reads)
template <uint32_t SLOTS, uint32_t ITERS, uint32_t EPS_BITS>
inline void sinkhorn_sfpu() {
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x4000);  // 2.0
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, EPS_BITS >> 16);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, EPS_BITS & 0xffff);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
    sinkhorn_slot<0, ITERS>();
    if constexpr (SLOTS > 1) {
        sinkhorn_slot<2, ITERS>();
    }
    if constexpr (SLOTS > 2) {
        sinkhorn_slot<16, ITERS>();
    }
    if constexpr (SLOTS > 3) {
        sinkhorn_slot<18, ITERS>();
    }
}
#endif

// ---- Sinkhorn steps in one Dest pass each (as mhc_split_sinkhorn's fused form) ----
FORCE_INLINE void sfpu_in_dst(uint32_t op, uint32_t scalar_bits) {
    if (op == OP_EXP) {
        unary_min_tile_init();
        unary_min_tile(0, 0x42a00000u);  // min(x, 80.0f)
        exp_tile_init();
        exp_tile(0);
    } else if (op == OP_SIGMOID) {
        sigmoid_tile_init();
        sigmoid_tile(0);
    } else if (op == OP_RECIP) {
        recip_tile_init();
        recip_tile(0);
    } else if (op == OP_RECIP_EPS) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, scalar_bits);
        recip_tile_init();
        recip_tile(0);
    } else if (op == OP_ADD_EPS) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, scalar_bits);
    } else if (op == OP_MUL2) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scalar_bits);
    }
}

FORCE_INLINE void pack_dst(uint32_t out) {
    CircularBuffer cb_out(out);
    tile_regs_commit();
    tile_regs_wait();
    cb_out.reserve_back(1);
    pack_tile(0, out);
    cb_out.push_back(1);
    tile_regs_release();
}

// out <- op2(op1(mixes @ SC[sel] + SC[base])).  Reads CB_MIX[0]; pops nothing.
FORCE_INLINE void project_fused(
    uint32_t sel, uint32_t base, uint32_t op1, uint32_t op2, uint32_t scalar_bits, uint32_t out) {
    tile_regs_acquire();
    matmul_init(CB_MIX, CB_SC);
    matmul_tiles(CB_MIX, CB_SC, 0, sel, 0);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_SC);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_SC, base, 0);
    sfpu_in_dst(op1, scalar_bits);
    sfpu_in_dst(op2, scalar_bits);
    pack_dst(out);
}

// m_out <- m_in / (m_in @ SC[k] [+ eps]) [+ eps].  m_in is popped.
FORCE_INLINE void normalize_fused(
    uint32_t m_in, uint32_t k_tile, bool use_eps, bool add_eps_after, uint32_t eps_bits, uint32_t m_out) {
    CircularBuffer in(m_in);
    in.wait_front(1);
    tile_regs_acquire();
    matmul_init(m_in, CB_SC);
    matmul_tiles(m_in, CB_SC, 0, k_tile, 0);
    sfpu_in_dst(use_eps ? OP_RECIP_EPS : OP_RECIP, eps_bits);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(m_in);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(m_in, 0, 0);
    if (add_eps_after) {
        sfpu_in_dst(OP_ADD_EPS, eps_bits);
    }
    pack_dst(m_out);
    in.pop_front(1);
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_MC, CB_PART, CB_PROJ);
    constexpr uint32_t slices = get_compile_time_arg_val(0);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t iters = get_compile_time_arg_val(2);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(3);
    constexpr uint32_t norm_eps_bits = get_compile_time_arg_val(4);
    constexpr uint32_t inv_count_bits = get_compile_time_arg_val(5);
    constexpr uint32_t tokens = get_compile_time_arg_val(6);
    constexpr bool sfpu_sinkhorn = get_compile_time_arg_val(7) != 0;

    CircularBuffer part(CB_PART), mc(CB_MC), sc(CB_SC), proj(CB_PROJ), mask(CB_MASK), diag(CB_DIAG), own(CB_OWN),
        sqp(CB_SQP), raw(CB_RAW), scale(CB_SCALE), mix(CB_MIX), pre(CB_PRE), spread(CB_SPREAD), p(CB_P);
    DeviceZoneScopedN("M-ALL");  // zones to the end of kernel_main (a block each): the phases' times are differences
    mc.wait_front(13);
    sc.wait_front(8);

    // 1. the slices' sums: Dest n = sum_s I @ partial_s[rt][n]
    proj.reserve_back(4 * row_tiles);
    for (uint32_t rt = 0; rt < row_tiles; ++rt) {
        tile_regs_acquire();
        matmul_init(CB_MC, CB_PART);
        for (uint32_t n = 0; n < 4; ++n) {
            for (uint32_t s = 0; s < slices; ++s) {
                part.wait_front(1);
                matmul_tiles(CB_MC, CB_PART, ID, 0, n);
                part.pop_front(1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t n = 0; n < 4; ++n) {
            pack_to(n, CB_PROJ, rt * 4 + n);
        }
        tile_regs_release();
    }
    proj.push_back(4 * row_tiles);

    {
    DeviceZoneScopedN("M-2");
    // 2. own-block masks and the Gram diagonals
    proj.wait_front(4 * row_tiles);
    mask.reserve_back(3 * row_tiles);
    diag.reserve_back(row_tiles);
    mul_init(CB_PROJ, CB_MC);
    for (uint32_t rt = 0; rt < row_tiles; ++rt) {
        for (uint32_t n = 0; n < 4; ++n) {
            tile_regs_acquire();
            mul_tiles(CB_PROJ, CB_MC, rt * 4 + n, n < 3 ? OB0 + n : ID, 0);
            tile_regs_commit();
            tile_regs_wait();
            if (n < 3) {
                pack_to(0, CB_MASK, rt * 3 + n);
            } else {
                pack_to(0, CB_DIAG, rt);
            }
            tile_regs_release();
        }
    }
    mask.push_back(3 * row_tiles);
    diag.push_back(row_tiles);
#ifndef MHC_READ_NORM  // the read op keeps the Gram tiles for the block input's norm (make_ns)
    proj.pop_front(4 * row_tiles);
#endif

    // 3. own_n = sum_rt TS_rt @ masked_{rt,n}; mixes = sum_n own_n @ BS_n
    mask.wait_front(3 * row_tiles);
    own.reserve_back(3);
    tile_regs_acquire();
    matmul_init(CB_MC, CB_MASK);
    for (uint32_t n = 0; n < 3; ++n) {
        for (uint32_t rt = 0; rt < row_tiles; ++rt) {
            matmul_tiles(CB_MC, CB_MASK, TS0 + rt, rt * 3 + n, n);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t n = 0; n < 3; ++n) {
        pack_to(n, CB_OWN, n);
    }
    tile_regs_release();
    own.push_back(3);
    mask.pop_front(3 * row_tiles);

    own.wait_front(3);
    raw.reserve_back(1);
    tile_regs_acquire();
    matmul_init(CB_OWN, CB_MC);
    for (uint32_t n = 0; n < 3; ++n) {
        matmul_tiles(CB_OWN, CB_MC, n, BS0 + n, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_RAW, 0);
    tile_regs_release();
    raw.push_back(1);
    own.pop_front(3);

    // squares: (sum_rt TS_rt @ diag_rt) @ ONES, then the scale rsqrt(squares / (n D) + norm_eps)
    diag.wait_front(row_tiles);
    sqp.reserve_back(1);
    tile_regs_acquire();
    matmul_init(CB_MC, CB_DIAG);
    for (uint32_t rt = 0; rt < row_tiles; ++rt) {
        matmul_tiles(CB_MC, CB_DIAG, TS0 + rt, rt, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_SQP, 0);
    tile_regs_release();
    sqp.push_back(1);
    diag.pop_front(row_tiles);

    sqp.wait_front(1);
    scale.reserve_back(1);
    tile_regs_acquire();
    matmul_init(CB_SQP, CB_MC);
    matmul_tiles(CB_SQP, CB_MC, 0, ONES, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_count_bits);
    add_unary_tile(0, norm_eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_SCALE, 0);
    tile_regs_release();
    scale.push_back(1);
    sqp.pop_front(1);

    // 4. mixes * scale
    raw.wait_front(1);
    scale.wait_front(1);
    mix.reserve_back(1);
    tile_regs_acquire();
    mul_init(CB_RAW, CB_SCALE);
    mul_tiles(CB_RAW, CB_SCALE, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_MIX, 0);
    tile_regs_release();
    mix.push_back(1);
    raw.pop_front(1);
    scale.pop_front(1);

    {
    DeviceZoneScopedN("M-5");
    // 5. comb (Sinkhorn), pre and post; 6. P_rt = (pre @ REPEAT) * TS_rt.  The read op (MHC_READ_NORM) makes pre and P
    // first: P needs pre only and the slice cores wait for it, while comb and post only go to the outputs
    mix.wait_front(1);
    auto make_p = [&]() {
        pre.wait_front(1);
        spread.reserve_back(1);
        tile_regs_acquire();
        matmul_init(CB_PRE, CB_MC);
        matmul_tiles(CB_PRE, CB_MC, 0, REPEAT, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_SPREAD, 0);
        tile_regs_release();
        spread.push_back(1);
        pre.pop_front(1);

        spread.wait_front(1);
        p.reserve_back(row_tiles);
#ifdef MHC_READ_NORM
        CircularBuffer p2(CB_P2);
        p2.reserve_back(row_tiles);
#endif
        mul_init(CB_SPREAD, CB_MC);
        for (uint32_t rt = 0; rt < row_tiles; ++rt) {
            tile_regs_acquire();
            mul_tiles(CB_SPREAD, CB_MC, 0, TS0 + rt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, CB_P, rt);
#ifdef MHC_READ_NORM
            pack_to(0, CB_P2, rt);
#endif
            tile_regs_release();
        }
        p.push_back(row_tiles);
#ifdef MHC_READ_NORM
        p2.push_back(row_tiles);
#endif
        spread.pop_front(1);
    };
#ifdef MHC_READ_NORM
    // the block input's norm scale rsqrt(|x_t|^2 / D + eps) in every column of row t (as the slices computed it from
    // the bf16 block input before: here from x before its rounding to bf16)
    auto make_ns = [&]() {
        constexpr uint32_t inv_hidden_bits = get_compile_time_arg_val(8);
        constexpr uint32_t rms_eps_bits = get_compile_time_arg_val(9);
        CircularBuffer p2(CB_P2), pg(CB_PG), q(CB_Q), nso(CB_NSO);
        p2.wait_front(row_tiles);
        pg.reserve_back(row_tiles);
        matmul_init(CB_P2, CB_PROJ);
        for (uint32_t rt = 0; rt < row_tiles; ++rt) {
            tile_regs_acquire();
            matmul_tiles(CB_P2, CB_PROJ, rt, rt * 4 + 3, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, CB_PG, rt);
            tile_regs_release();
        }
        pg.push_back(row_tiles);
        proj.pop_front(4 * row_tiles);

        pg.wait_front(row_tiles);
        q.reserve_back(row_tiles);
        mul_init(CB_PG, CB_P2);
        for (uint32_t rt = 0; rt < row_tiles; ++rt) {
            tile_regs_acquire();
            mul_tiles(CB_PG, CB_P2, rt, rt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, CB_Q, rt);
            tile_regs_release();
        }
        q.push_back(row_tiles);
        pg.pop_front(row_tiles);
        p2.pop_front(row_tiles);

        q.wait_front(row_tiles);
        nso.reserve_back(1);
        tile_regs_acquire();
        matmul_init(CB_Q, CB_MC);
        for (uint32_t rt = 0; rt < row_tiles; ++rt) {
            matmul_tiles(CB_Q, CB_MC, rt, ONES, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_hidden_bits);
        add_unary_tile(0, rms_eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_NSO, 0);
        tile_regs_release();
        nso.push_back(1);
        q.pop_front(row_tiles);
    };
    project_fused(SEL_PRE, BASE_PRE, OP_SIGMOID, OP_ADD_EPS, eps_bits, CB_PRE);
    make_p();
    make_ns();
#endif
    if constexpr (sfpu_sinkhorn) {
        // comb = exp(min(mixes @ SEL + BASE, 80)) in Dest, transposed (fp32), the iterations on the SFPU, transposed
        // back, packed once
        constexpr uint32_t slots = tokens > 17 ? 4 : tokens > 16 ? 3 : tokens > 1 ? 2 : 1;
        tile_regs_acquire();
        matmul_init(CB_MIX, CB_SC);
        matmul_tiles(CB_MIX, CB_SC, 0, SEL_COMB, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_SC);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_SC, BASE_COMB, 0);
        sfpu_in_dst(OP_EXP, 0);
        transpose_dest_init<true>();
        transpose_dest<true>(0);
        recip_tile_init();  // the SFPU state (ADDR_MOD_7: no increments)
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            sinkhorn_sfpu<slots, iters, eps_bits>, 0, VectorMode::RC_custom)));
        transpose_dest_init<true>();
        transpose_dest<true>(0);
        pack_dst(CB_COMB);
    } else {
        project_fused(SEL_COMB, BASE_COMB, OP_EXP, OP_COPY, 0, CB_MA);
        normalize_fused(CB_MA, RB, false, true, eps_bits, CB_MB);
        normalize_fused(CB_MB, CB_COL, true, false, eps_bits, iters == 1 ? CB_COMB : CB_MA);
        for (uint32_t i = 1; i < iters; ++i) {
            normalize_fused(CB_MA, RB, true, false, eps_bits, CB_MB);
            normalize_fused(CB_MB, CB_COL, true, false, eps_bits, i + 1 == iters ? CB_COMB : CB_MA);
        }
    }
#ifndef MHC_READ_NORM
    project_fused(SEL_PRE, BASE_PRE, OP_SIGMOID, OP_ADD_EPS, eps_bits, CB_PRE);
#endif
    project_fused(SEL_POST, BASE_POST, OP_SIGMOID, OP_MUL2, TWO_BITS, CB_POST);
    mix.pop_front(1);
#ifndef MHC_READ_NORM
    make_p();
#endif
    }}  // the zones' blocks
}
