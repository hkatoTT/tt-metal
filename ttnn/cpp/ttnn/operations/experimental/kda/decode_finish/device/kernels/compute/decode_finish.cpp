// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KDA decode after the delta rule, per piece (the rows of a (head, block of 16 users) unit a core has: rows 0 .. ROWS - 1
// of 4 tiles; ttnn/kda.py `_output`, the composite form's order and roundings): o rounded to bf16; o * rsqrt(mean(o^2)
// + eps) * o_norm (broadcast down the rows) * sigmoid(gate), rounded to bf16.  Two stages a piece (the same values as
// four stages with a whole unit a core):
//   A: o rounded to bf16 (Dest 0-3), its squares (Dest 4-7) summed over the 4 tiles (Dest 4); packed: R, the sum
//   B: the ones matmul into Dest 4, rsqrt(mean + eps) there, R back into Dest 0-3, then the norm, o_norm and the gate
// (no CB round trip of R into the squares, nor of the norm's inverse).  The squares are summed exactly over the 4 tiles, then across the row by a matmul
// with ones.  The fp32 inputs are unpacked to Dest, the bf16 gate through SrcA (exact); the roundings are done in Dest
// and the output is packed as bf16 (exact: already rounded).  The SFPU computes only the block's rows (rows_sfpu.hpp);
// the other rows of the output's faces are padding of the [B, ..] tile row.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/rows_sfpu.hpp"

namespace {
constexpr uint32_t CB_O = 0;
constexpr uint32_t CB_G = 1;
constexpr uint32_t CB_W = 2;
constexpr uint32_t CB_ONES = 3;
constexpr uint32_t CB_R = 16;
constexpr uint32_t CB_SQ = 17;
constexpr uint32_t CB_INV = 18;
constexpr uint32_t CB_OUT = 19;
constexpr uint32_t N = 4;
constexpr uint32_t INV_DIM_BITS = 0x3c000000u;  // 1 / 128
constexpr uint32_t FMT_FP32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t FMT_BF16 = static_cast<uint32_t>(DataFormat::Float16_b);
constexpr uint32_t ROWS = get_compile_time_arg_val(1);  // a piece's rows (at most 16)

FORCE_INLINE void pack_four(uint32_t cb) {
    for (uint32_t i = 0; i < N; ++i) {
        pack_tile<true>(i, cb, i);
    }
}

FORCE_INLINE void copy_four(uint32_t cb, uint32_t first_dst) {
    copy_init(cb);
    for (uint32_t c = 0; c < N; ++c) {
        copy_tile(cb, c, first_dst + c);
    }
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_SQ, CB_ONES, CB_INV);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(0);
    const uint32_t count = get_arg_val<uint32_t>(0);

    CircularBuffer o(CB_O), g(CB_G), w(CB_W), ones(CB_ONES), r(CB_R), sq(CB_SQ), out(CB_OUT);
    ones.wait_front(1);
    w.wait_front(N);

    DeviceZoneScopedN("FC-ALL");
    for (uint32_t unit = 0; unit < count; ++unit) {
        DeviceZoneScopedN("FC-UNIT");
        {
            DeviceZoneScopedN("FC-SA");
            // A: R = o rounded to bf16 (Dest 0-3); the squares (Dest 4-7) summed over the 4 tiles into Dest 4
            o.wait_front(N);
            r.reserve_back(N);
            sq.reserve_back(1);
            tile_regs_acquire();
            if (unit > 0) {
                // SrcA back to the startup's configuration (CB_SQ, fp32 through SrcA) for B's ones matmul
                reconfig_data_format_srca(CB_G, CB_SQ);
            }
            copy_four(CB_O, 0);
            typecast_tile_init<FMT_FP32, FMT_BF16>();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::round_bf16<ROWS>(c);
            }
            mul_binary_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::mul<ROWS>(c, c, N + c);
            }
            add_binary_tile_init();
            kda_rows::add<ROWS>(N, N + 1, N);
            kda_rows::add<ROWS>(N + 2, N + 3, N + 2);
            kda_rows::add<ROWS>(N, N + 2, N);
            tile_regs_commit();
            tile_regs_wait();
            if (unit > 0) {
                pack_reconfig_data_format(CB_OUT, CB_R);
            }
            pack_four(CB_R);
            pack_tile<true>(N, CB_SQ, 0);
            tile_regs_release();
            r.push_back(N);
            sq.push_back(1);
            o.pop_front(N);
        }
        {
            DeviceZoneScopedN("FC-SB");
            // B: rsqrt(mean + eps) (every column of each row) in Dest 4, then ((R * inv) * o_norm) * sigmoid(gate),
            // rounded to bf16
            sq.wait_front(1);
            r.wait_front(N);
            g.wait_front(N);
            out.reserve_back(N);
            tile_regs_acquire();
            matmul_init(CB_SQ, CB_ONES);
            matmul_tiles(CB_SQ, CB_ONES, 0, 0, N);
            binop_with_scalar_tile_init();
            kda_rows::mul_scalar<ROWS>(N, INV_DIM_BITS);
            kda_rows::add_scalar<ROWS>(N, eps_bits);
            rsqrt_tile_init();
            kda_rows::rsqrt<ROWS>(N);
            copy_four(CB_R, 0);
            mul_binary_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::mul<ROWS>(c, N, c);
            }
            unary_bcast_init<BroadcastType::ROW>(CB_W);
            for (uint32_t c = 0; c < N; ++c) {
                unary_bcast<BroadcastType::ROW>(CB_W, c, N + c);
            }
            mul_binary_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::mul<ROWS>(c, N + c, c);
            }
            reconfig_data_format_srca(CB_W, CB_G);
            copy_four(CB_G, N);
            sigmoid_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::sigmoid<ROWS>(N + c);
            }
            mul_binary_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::mul<ROWS>(c, N + c, c);
            }
            typecast_tile_init<FMT_FP32, FMT_BF16>();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::round_bf16<ROWS>(c);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_SQ, CB_OUT);
            pack_four(CB_OUT);
            tile_regs_release();
            out.push_back(N);
            sq.pop_front(1);
            r.pop_front(N);
            g.pop_front(N);
        }
    }
}
