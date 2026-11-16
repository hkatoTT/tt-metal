// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// matmul_init / matmul_tiles / matmul_block_init / matmul_block (api/compute/matmul.h) at math fidelity F in place of
// the kernel's (HiFi4), for the packed units' gather and scatter, whose in0 (SrcB: matmul's in0 -> SrcB, in1 -> SrcA
// here) is a 0/1 tile: its mantissas are 0, so the fidelity phases on SrcB's low bits only add zeros and fewer phases
// give the same sums (MOE_EXPERTS_ONEHOT_FIDELITY; A/B'd bit for bit against HiFi4).  Blackhole's firmware-controlled
// matmul throttle is kept as matmul_block has it.

#include "api/compute/matmul.h"

namespace ckernel {

template <MathFidelity F>
ALWI void onehot_math_block(uint32_t in0, uint32_t in1, uint32_t idst, uint32_t ct, uint32_t rt) {
#ifdef TRISC_MATH
#ifdef ARCH_BLACKHOLE
    volatile std::uint32_t mm_throttle_en = *(throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (throttled_mop_status != 1) {
            llk_math_matmul_init<F, MM_THROTTLE_MAX>(in0, in1, 0, ct, rt);
            throttled_mop_status = 1;
        }
        llk_math_matmul<F, MM_THROTTLE_MAX>(idst, ct, rt);
    } else {
        if (throttled_mop_status != 0) {
            llk_math_matmul_init<F, MM_THROTTLE>(in0, in1, 0, ct, rt);
            throttled_mop_status = 0;
        }
        llk_math_matmul<F, MM_THROTTLE>(idst, ct, rt);
    }
#else
    llk_math_matmul<F, MM_THROTTLE>(idst, ct, rt);
#endif
#endif
}

template <MathFidelity F>
ALWI void onehot_block_init(uint32_t in0, uint32_t in1, uint32_t ct, uint32_t rt, uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0, ct, rt, kt)));
    MATH((llk_math_matmul_init<F, MM_THROTTLE>(in0, in1, 0, ct, rt)));
#ifdef ARCH_BLACKHOLE
    MATH((throttled_mop_status = 0));
#endif
}

template <MathFidelity F>
ALWI void onehot_block(
    uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst, uint32_t ct, uint32_t rt, uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct, rt, kt)));
    MATH((onehot_math_block<F>(in0, in1, idst, ct, rt)));
}

template <MathFidelity F>
ALWI void onehot_init(uint32_t in0, uint32_t in1) {
    state_configure(in1, in0, __builtin_LINE());
    MATH((llk_math_matmul_init<F, MM_THROTTLE>(in0, in1, 0)));
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0)));
}

template <MathFidelity F>
ALWI void onehot_tiles(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1)));
    MATH((llk_math_matmul<F, MM_THROTTLE>(idst)));
}

}  // namespace ckernel
