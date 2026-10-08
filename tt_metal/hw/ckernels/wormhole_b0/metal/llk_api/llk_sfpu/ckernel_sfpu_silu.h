// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_tanh.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#if defined(HK_SILU_FAST) && HK_SILU_FAST == 2
    if constexpr (ITERATIONS % 2 == 0) {
        // EXPERIMENT (hkato): silu(x) = x * (0.5 + 0.5 * tanh(x / 2)), bf16 tanh polynomial, two rows in lockstep
        sfpi::vFloat c1 = TANH_POLY_C1;
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS / 2; d++) {
            sfpi::vFloat x0 = sfpi::dst_reg[0];
            sfpi::vFloat x1 = sfpi::dst_reg[1];
            sfpi::vFloat t0, t1;
            _sfpu_tanh_polynomial_x2_(t0, t1, 0.5f * x0, 0.5f * x1, c1);
            x0 = sfpi::dst_reg[0];
            x1 = sfpi::dst_reg[1];
            sfpi::vFloat h0 = 0.5f * x0;
            sfpi::vFloat h1 = 0.5f * x1;
            h0 = h0 * t0 + h0;
            h1 = h1 * t1 + h1;
            if constexpr (!is_fp32_dest_acc_en) {
                h0 = sfpi::convert<sfpi::vFloat16b>(h0, sfpi::RoundMode::Nearest);
                h1 = sfpi::convert<sfpi::vFloat16b>(h1, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = h0;
            sfpi::dst_reg[1] = h1;
            sfpi::dst_reg += 2;
        }
        return;
    }
#endif
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
#if defined(HK_SILU_FAST) && HK_SILU_FAST == 1
        sfpi::vFloat result = x * _sfpu_sigmoid_<false>(x);  // EXPERIMENT (hkato): bf16-path sigmoid
#else
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
#endif

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu always uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must
    // use non-approx sigmoid_init regardless of APPROXIMATION_MODE.
#if defined(HK_SILU_FAST) && HK_SILU_FAST == 2
    tanh_init<false, false>();  // EXPERIMENT (hkato): bf16 tanh polynomial constants
#else
    sigmoid_init<false>();
#endif
}

}  // namespace ckernel::sfpu
