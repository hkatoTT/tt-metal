// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A slice core of mhc_stream_read: K tiles (k) of the hidden dimension, every row tile (rt) of the streams.
//   A. per row tile: Dest 0-2 = sum_k S[rt, k] @ fn[k] (the projection), Dest 3 = sum_k S[rt, k] @ S[rt, k]^T (the Gram
//      tile), as mhc_stream_project, for the mix core
//   C. with P from the mix core: the block input's tiles x_c = bf16(sum_rt P[rt] @ S[rt, c]) (as the composite's
//      P @ streams matmul to bf16)
//   D. with the norm's scale rsqrt(mean + eps) of each row (in every column) from the mix core: y_c = bf16((x_c *
//      scale) * gamma_c), the first product on the matrix engine (the scale at its 19 bits, as ttnn's norm applies its
//      rsqrt), gamma's row 0 broadcast down the rows (through SrcB) and the second product on the SFPU
// MHC_WR_MERGED (mhc_stream_write_read's prefill groups with the write on the slice cores): first, W. per column k
//   and row tile rt the new streams tile bf16(W_rt @ S_old) + bf16(Q_rt @ BO_k), as mhc_stream_write.cpp computes it
//   (each product summed in fp32, packed to bf16, the two added), into CB_S in the order made: column major, tile
//   k R + rt (the read's streams tiles, A and C index them so; 4 a column pushed, the writer also writes them to DRAM);
//   W and Q of the group's 4 row tiles from the reader.
// Matmul operands: in0 in SrcB, in1 in SrcA; elementwise: icb0 in SrcA, icb1 in SrcB.  S, fn, x and gamma are bf16, P
// and the scale fp32: SrcB's format is switched for each.

#include "api/debug/kernel_profiler.hpp"
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/rows_sfpu.hpp"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t CB_Y = 6;
constexpr uint32_t CB_PIN = 9;
constexpr uint32_t CB_NS = 10;
constexpr uint32_t CB_S = 12;
constexpr uint32_t CB_FN = 13;
constexpr uint32_t CB_G = 14;
constexpr uint32_t CB_OUT = 28;
constexpr uint32_t CB_X = 29;
constexpr uint32_t PROJ_TILES = 3;

}  // namespace

void kernel_main() {
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(1);
#ifdef MHC_WR_MERGED
    constexpr uint32_t CB_SOLD = 20, CB_BOW = 21, CB_W = 22, CB_Q = 23, CB_T1 = 24, CB_T2 = 25, CB_READY = 26;
    compute_kernel_hw_startup(CB_W, CB_SOLD, CB_T1);  // (as mhc_stream_write.cpp)
    // the streams tile (rt, k) in CB_S: column major as the write side made them
    auto s_at = [](uint32_t rt, uint32_t k) { return k * row_tiles + rt; };
#else
    compute_kernel_hw_startup(CB_S, CB_FN, CB_OUT);
    auto s_at = [](uint32_t rt, uint32_t k) { return rt * k_tiles + k; };  // the reader's order
#endif

    CircularBuffer s(CB_S), fn(CB_FN), out(CB_OUT), pin(CB_PIN), x(CB_X), ns(CB_NS), g(CB_G), y(CB_Y);
    // the tokens' rows of the SFPU's gamma product (the padding rows keep x * scale)
    constexpr uint32_t tokens = get_compile_time_arg_val(2);

    DeviceZoneScopedN("S-ALL");  // zones to the end of kernel_main (a block each): the phases' times are differences
#ifdef MHC_WR_MERGED
    {  // W. the new streams tiles of this slice
        DeviceZoneScopedN("S-W");
        CircularBuffer sold(CB_SOLD), bow(CB_BOW), wt(CB_W), qt(CB_Q), t1(CB_T1), t2(CB_T2);
        wt.wait_front(row_tiles);
        qt.wait_front(row_tiles);
        for (uint32_t k = 0; k < MHC_WR_KOWN; ++k) {  // its own columns
            s.reserve_back(row_tiles);
            sold.wait_front(row_tiles);
            bow.wait_front(1);
            // two row tiles a Dest acquire: W @ S into Dest 0, 1 and Q @ BO into 2, 3 (one init each), packed to bf16;
            // then the two sums (one init) packed into CB_S in the row tiles' order
            for (uint32_t rt = 0; rt < row_tiles; rt += 2) {
                t1.reserve_back(2);
                t2.reserve_back(2);
                tile_regs_acquire();
#ifndef MHC_WR_PROBE_NOMATH
                matmul_init(CB_W, CB_SOLD);
                matmul_tiles(CB_W, CB_SOLD, rt, rt, 0);
                matmul_tiles(CB_W, CB_SOLD, rt + 1, rt + 1, 1);
                matmul_init(CB_Q, CB_BOW);
                matmul_tiles(CB_Q, CB_BOW, rt, 0, 2);
                matmul_tiles(CB_Q, CB_BOW, rt + 1, 0, 3);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_T1);
                pack_tile(1, CB_T1);
                pack_tile(2, CB_T2);
                pack_tile(3, CB_T2);
                tile_regs_release();
                t1.push_back(2);
                t2.push_back(2);
                t1.wait_front(2);
                t2.wait_front(2);
                tile_regs_acquire();
#ifndef MHC_WR_PROBE_NOMATH
                add_init(CB_T1, CB_T2);
                add_tiles(CB_T1, CB_T2, 0, 0, 0);
                add_tiles(CB_T1, CB_T2, 1, 1, 1);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_S);
                pack_tile(1, CB_S);
                tile_regs_release();
                t1.pop_front(2);
                t2.pop_front(2);
            }
            s.push_back(row_tiles);
            sold.pop_front(row_tiles);
            bow.pop_front(1);
        }
        wt.pop_front(row_tiles);
        qt.pop_front(row_tiles);
        // the helper cores' columns (written into CB_S at their places; their flags waited for by the reader)
        if constexpr (MHC_WR_KOWN < k_tiles) {
            CircularBuffer ready(CB_READY);
            ready.wait_front(1);
            s.reserve_back(row_tiles * (k_tiles - MHC_WR_KOWN));
            s.push_back(row_tiles * (k_tiles - MHC_WR_KOWN));
            ready.pop_front(1);
        }
        // A's state: S and fn (bf16) unpacked, the partials packed fp32
        reconfig_data_format(CB_S, CB_FN);
        pack_reconfig_data_format(CB_OUT);
    }
#endif
    // A. the partials, each k as the reader pushes it (its tiles at positions rt K + k: (k + 1) R tiles pushed)
    {
    DeviceZoneScopedN("S-A");
    out.reserve_back(4 * row_tiles);
    for (uint32_t rt = 0; rt < row_tiles; ++rt) {
        tile_regs_acquire();
        matmul_init(CB_S, CB_FN);
        for (uint32_t k = 0; k < k_tiles; ++k) {
            if (rt == 0) {
                s.wait_front((k + 1) * row_tiles);
                fn.wait_front((k + 1) * PROJ_TILES);
            }
            for (uint32_t n = 0; n < PROJ_TILES; ++n) {
                matmul_tiles(CB_S, CB_FN, s_at(rt, k), k * PROJ_TILES + n, n);
            }
        }
        matmul_init(CB_S, CB_S, 1);
        for (uint32_t k = 0; k < k_tiles; ++k) {
            matmul_tiles(CB_S, CB_S, s_at(rt, k), s_at(rt, k), PROJ_TILES);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t n = 0; n <= PROJ_TILES; ++n) {
            pack_tile<true>(n, CB_OUT, rt * 4 + n);
        }
        tile_regs_release();
    }
    out.push_back(4 * row_tiles);
    fn.pop_front(k_tiles * PROJ_TILES);

    // C. the block input's tiles of this slice
    {
    DeviceZoneScopedN("S-PWAIT");
    pin.wait_front(row_tiles);
    {
    DeviceZoneScopedN("S-C");
    x.reserve_back(k_tiles);
    reconfig_data_format_srcb(CB_S, CB_PIN);
    pack_reconfig_data_format(CB_OUT, CB_X);
    matmul_init(CB_PIN, CB_S);
    for (uint32_t c = 0; c < k_tiles; ++c) {
        tile_regs_acquire();
        for (uint32_t rt = 0; rt < row_tiles; ++rt) {
            matmul_tiles(CB_PIN, CB_S, rt, s_at(rt, c), 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, CB_X, c);
        tile_regs_release();
    }
    x.push_back(k_tiles);
    pin.pop_front(row_tiles);
    s.pop_front(row_tiles * k_tiles);

    reconfig_data_format_srcb(CB_PIN, CB_X);  // D's starting state

    // D. y_c = bf16((x_c * scale) * gamma_c)
    {
    DeviceZoneScopedN("S-NWAIT");
    x.wait_front(k_tiles);
    ns.wait_front(1);
    {
    DeviceZoneScopedN("S-D");
    g.wait_front(k_tiles);
    y.reserve_back(k_tiles);
    pack_reconfig_data_format(CB_X, CB_Y);
    for (uint32_t c = 0; c < k_tiles; ++c) {
        tile_regs_acquire();
        reconfig_data_format_srcb(CB_X, CB_NS);
        mul_init(CB_X, CB_NS, false);
        mul_tiles(CB_X, CB_NS, c, 0, 0);
        reconfig_data_format_srcb(CB_NS, CB_G);
        unary_bcast_init<BroadcastType::ROW>(CB_G);
        unary_bcast<BroadcastType::ROW>(CB_G, c, 1);
        mul_binary_tile_init();
        if constexpr (tokens <= 16) {  // the tokens' rows; the padding rows keep x * scale
            kda_rows::mul<tokens>(0, 1, 0);
        } else {
            mul_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, CB_Y, c);
        tile_regs_release();
        reconfig_data_format_srcb(CB_G, CB_X);
    }
    y.push_back(k_tiles);
    ns.pop_front(1);
    g.pop_front(k_tiles);
    x.pop_front(k_tiles);
    }}}}}  // the zones' blocks
}
