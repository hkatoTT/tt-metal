// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A prefill chunk's pools' keys in one op (dsa_indexer.pool_keys, `TtDsa._chunk_pool_keys`), the compute.  An item is
// one tile of the indexer keys / gates [1, 1, T, 128]: 32 tokens x 32 features = 8 pools of 4 rows (pool k in rows
// 4 k .. 4 k + 3), and gives the item's 8 pools' keys.  The ttnn ops it replaces work on one pool a tile (rows 0-3,
// rows 4-31 padding: 0 for the key and gate, -inf for the slot bias), and this computes every value as they do:
//   t = fp32(gate) + bias (add_binary_tile, RNE; the bias tile holds the 4 slot biases in rows 4 k + i),
//   m = the pool's max of t, e = exp(t - m) (sub_binary_tile, exp_tile<0>), s = the pool's sum of e,
//   w = bf16(e / s) (div_binary_tile, typecast_tile fp32 -> bf16, RNE), p = bf16(w k) (the bf16-Dest multiply:
//   RNE, 0 if either is 0), all elementwise on the full tile; the pools' max and sum on the SFPU (pool_groups below)
//   in the order of the column reduce on a padded tile: sfpu_reduce's column sum adds the rows 4 apart (all 0 but the
//   pool's: exact) and then the 4 rows as r0 + (r1 + (r2 + r3)); the max selects (no rounding);
//   then the key = sum of the pool's 4 rows of p on the FPU (reduce_tile SUM over the columns, HiFi4, fp32 Dest, as
//   the bf16 sum over dim -2): the reader copies each pool's 4 rows of p to rows 0-3 of a tile whose rows 4-31 are 0
//   (the padded tile the ttnn sum reads), 8 tiles an item; the writer takes their row 0.
// Compute runs an item's SFPU part and then the previous item's reduces, so the reader's copies overlap.
//   CT: CB_IG, CB_IK, CB_BIAS, CB_SCALER, CB_PROD, CB_PAD, CB_KEY, then the shape key.  RT: the items [i0, i1).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t CB_IG = get_compile_time_arg_val(0);
constexpr uint32_t CB_IK = get_compile_time_arg_val(1);
constexpr uint32_t CB_BIAS = get_compile_time_arg_val(2);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(3);
constexpr uint32_t CB_PROD = get_compile_time_arg_val(4);
constexpr uint32_t CB_PAD = get_compile_time_arg_val(5);
constexpr uint32_t CB_KEY = get_compile_time_arg_val(6);
constexpr uint32_t COLS = 4;   // 128 features: 4 tiles
constexpr uint32_t POOLS = 8;  // pools a tile

#ifdef TRISC_MATH
// Dest tile 0 (the base) -> tile 2 (128 rows on): each pool's max (MAX) or sum of its 4 rows, in all 4 of its rows.
// An SFPU load at address A moves rows (A & ~3) .. + 3 of a face's even (A & 2 = 0) or odd columns, so a column half
// of a face holds 4 pools in 4 registers (rows 0-3, 4-7, 8-11, 12-15); faces 0 / 1 have pools 0-3, faces 2 / 3 pools
// 4-7 (in LREG4-7).  After SFPTRANSP, register j row k is pool k's row j: the 4 registers' max or sum is each pool's
// in its own row; copied to the 4 registers and transposed back, every row of a pool holds it.  The sum is
// LREG0 + (LREG1 + (LREG2 + LREG3)), sfpu_reduce's last step (init_reduce_sum_avg's half reduce); the max takes the
// larger into the first operand of SFPSWAP (the default direction, as calculate_reduce_max_min's last step).
template <bool MAX, uint32_t U>
inline void pool_groups_half() {
    constexpr uint32_t L = U + 32;     // the face below (2 or 3)
    constexpr uint32_t OUT = 2 * 64;   // Dest tile 2
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, U);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, U + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, U + 8);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, U + 12);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, L);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, L + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, ADDR_MOD_7, L + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, L + 12);
    TTI_SFPTRANSP(0, 0, 0, 0);
    if constexpr (MAX) {
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG2, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG6, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
    } else {
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);
        TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, 0);
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0);
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    }
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG6, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG7, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, OUT + U);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, OUT + U + 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, OUT + U + 8);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, OUT + U + 12);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, OUT + L);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, OUT + L + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, ADDR_MOD_7, OUT + L + 8);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, OUT + L + 12);
}

template <bool MAX>
inline void pool_groups() {
    if constexpr (MAX) {
        TTI_SFPCONFIG(0, 0xF, 1);  // SFPSWAP's default direction (the larger to the first operand)
    }
    pool_groups_half<MAX, 0>();   // face 0 even columns (and face 2)
    pool_groups_half<MAX, 2>();   // face 0 odd
    pool_groups_half<MAX, 16>();  // face 1 even (and face 3)
    pool_groups_half<MAX, 18>();  // face 1 odd
}
#endif

// t, m, e, s, w and p of item tile column j (the bias tile j) -> CB_PROD (bf16)
FORCE_INLINE void weights(uint32_t j) {
    cb_wait_front(CB_IG, 1);
    cb_wait_front(CB_IK, 1);
    cb_reserve_back(CB_PROD, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(CB_IG);
    copy_tile_to_dst_init_short(CB_IG);
    copy_tile(CB_IG, 0, 0);
    reconfig_data_format_srca(CB_IG, CB_BIAS);
    copy_tile_to_dst_init_short(CB_BIAS);
    copy_tile(CB_BIAS, j, 1);
    add_binary_tile_init();
    add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
    MATH((_llk_math_eltwise_unary_sfpu_params_(pool_groups<true>, 0, VectorMode::RC_custom)));
    sub_binary_tile_init();
    sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 2, 0);
    exp_tile_init<0u>();
    exp_tile<0u>(0);
    MATH((_llk_math_eltwise_unary_sfpu_params_(pool_groups<false>, 0, VectorMode::RC_custom)));
    div_binary_tile_init();
    div_binary_tile(0, 2, 0);
    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
    reconfig_data_format_srca(CB_BIAS, CB_IK);
    copy_tile_to_dst_init_short(CB_IK);
    copy_tile(CB_IK, 0, 1);
    // ttnn.multiply of the bf16 w and key: mul_binary_tile with a bf16 Dest (RNE, and 0 where either is 0)
    mul_binary_tile_init();
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        true,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, false),
        0,
        1,
        0,
        VectorMode::RC)));
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_PROD);
    pack_tile(0, CB_PROD);
    tile_regs_release();
    cb_push_back(CB_PROD, 1);
    cb_pop_front(CB_IG, 1);
    cb_pop_front(CB_IK, 1);
}

// the item's 8 padded tiles of p (from the reader) -> their column sums (row 0) in CB_KEY
FORCE_INLINE void keys() {
    cb_wait_front(CB_PAD, POOLS);
    cb_reserve_back(CB_KEY, POOLS);
    reconfig_data_format(CB_PAD, CB_SCALER);
    pack_reconfig_data_format(CB_KEY);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(CB_PAD, CB_SCALER, CB_KEY);
    for (uint32_t k = 0; k < POOLS; ++k) {
        tile_regs_acquire();
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(CB_PAD, CB_SCALER, k, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, CB_KEY, k);
        tile_regs_release();
    }
    reduce_uninit();
    cb_push_back(CB_KEY, POOLS);
    cb_pop_front(CB_PAD, POOLS);
}
}  // namespace

void kernel_main() {
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    const uint32_t i1 = get_arg_val<uint32_t>(1);
    if (i1 <= i0) {
        return;
    }
    compute_kernel_hw_startup(CB_IG, CB_BIAS, CB_PROD);
    cb_wait_front(CB_BIAS, COLS);
    cb_wait_front(CB_SCALER, 1);
    for (uint32_t idx = i0; idx < i1; ++idx) {
        weights(idx % COLS);
        if (idx > i0) {
            keys();
        }
    }
    keys();
}
