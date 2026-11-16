// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// SFPU operations on a tile's first ROWS rows only (ROWS <= 16), for the KDA decode ops, which hold a block's users in
// rows 0-15 (faces 0 and 1) and use only those rows.  An SFPU load covers 4 rows of a face's even or odd columns, so
// a face's rows 0 .. 4 n - 1 take 2 n of its 8 iterations (as prepare_chunk_recurrence's exp of row 0); faces 2 and 3
// are skipped (VectorMode::R).  The other rows keep what Dest held; each function computes its rows exactly as the
// full-tile one does.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/typecast.h"

namespace ckernel::kda_rows {

template <uint32_t ROWS>
constexpr int iterations() {
    static_assert(ROWS >= 1 && ROWS <= 16, "the rows of faces 0 and 1");
    return 2 * static_cast<int>((ROWS + 3) / 4);
}

template <uint32_t ROWS>
ALWI void mul(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        true,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, iterations<ROWS>(), true),
        a,
        b,
        out,
        VectorMode::R)));
}

template <uint32_t ROWS>
ALWI void add(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        true,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, iterations<ROWS>(), true, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::R)));
}

template <uint32_t ROWS>
ALWI void sub(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        true,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, iterations<ROWS>(), true, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::R)));
}

template <uint32_t ROWS>
ALWI void silu(uint32_t i) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, true, calculate_silu, (true, iterations<ROWS>()), i, VectorMode::R));
}

template <uint32_t ROWS>
ALWI void sigmoid(uint32_t i) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, true, calculate_sigmoid, (false, true, iterations<ROWS>()), i, VectorMode::R));
}

template <uint32_t ROWS>
ALWI void exp(uint32_t i) {
    exp_tile<false, false, InputClamping::ClampToNegative, iterations<ROWS>()>(i, VectorMode::R);
}

template <uint32_t ROWS>
ALWI void rsqrt(uint32_t i) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, true, calculate_rsqrt, (APPROX, iterations<ROWS>(), true, false), i, VectorMode::R));
}

// fp32 rounded to bf16 (held in fp32), as typecast_tile<Float32, Float16_b>
template <uint32_t ROWS>
ALWI void round_bf16(uint32_t i) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, true, calculate_typecast_fp32_to_fp16b, (APPROX, iterations<ROWS>()), i, VectorMode::R));
}

template <uint32_t ROWS>
ALWI void add_scalar(uint32_t i, uint32_t bits) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        true,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, iterations<ROWS>(), true),
        i,
        VectorMode::R,
        bits));
}

template <uint32_t ROWS>
ALWI void mul_scalar(uint32_t i, uint32_t bits) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        true,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, iterations<ROWS>(), true),
        i,
        VectorMode::R,
        bits));
}

}  // namespace ckernel::kda_rows
