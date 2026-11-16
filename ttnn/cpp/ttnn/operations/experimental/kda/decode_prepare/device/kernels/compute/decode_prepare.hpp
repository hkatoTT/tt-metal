// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// KDA decode before the delta rule, per (part, head, block of 16 users), those users in rows 0-15 of the part's 4 tiles
// (ttnn/kda.py `step`, the same order and roundings; the per-channel taps, dt_bias and rate broadcast down the rows):
//   q / k / v: acc = h0 tap0 + h1 tap1 + h2 tap2 + cur tap3 (fp32, exact); SiLU; rounded to bf16 (in Dest);
//              q, k: x * rsqrt(sum(x^2) + 1e-6) (q times 1/sqrt(128) too)
//   gates:     e = exp(lower * sigmoid((f + dt_bias) * rate)); beta = sigmoid(logit) (rounded to bf16)
// The squares are summed exactly over the 4 tiles, then across the row by a matmul with ones.  The fp32 inputs are
// unpacked to Dest; the bf16 ones (the conv input, f) go through SrcA (exact), with SrcA switched to bf16 for those
// copies only (the ones matmul needs it in fp32).  Every output is fp32 (the bf16 roundings done in Dest).  The conv
// input is also copied to Dest 6-7 with its term and packed in fp32 for the writer, which writes it into history 2:
// after this core's history 2 was read and used (an earlier pack let the writer overwrite it before the read).
// The SFPU computes only the block's rows (rows_sfpu.hpp): the writer takes those rows only.

#include "api/debug/kernel_profiler.hpp"
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/rows_sfpu.hpp"

// With 2 or 3 tokens a user (KDA_PREP_TOKENS, MTP's verify step): each unit's pass k takes the k-th token's rows
// (CB_CUR2 / CB_F2 / CB_BETA2, CB_CUR3 / CB_F3 / CB_BETA3) and conv history, the last 3 of (h0, h1, h2, x_0 ..
// x_{k-1}): the earlier tokens' inputs in fp32 in CB_XT (x_j at tile j N, packed by pass j), after the earlier
// passes' outputs.
#ifndef KDA_PREP_TOKENS
#define KDA_PREP_TOKENS 1
#endif
// The stacked form (KDA_PREP_STACK passes a unit, the program factory's `stack_tokens`): each pass one token's work over
// USERS rows (a few tokens' users stacked by the reader, each row with its own history), its h0 h1 h2 and input pushed
// a pass, the taps (and the gates' dt_bias and rate) once a unit
#ifndef KDA_PREP_STACK
#define KDA_PREP_STACK 1
#endif
// split_channels (KDA_PREP_SPLIT): an item a core, a q / k / v unit's channel tiles c_lo .. c_lo + 1 (runtime args 2,
// 3; the gates' all 4).  q and k's L2 norm: the half's sum of squares (t0 + t1 or t2 + t3) to CB_SQP (the writer sends
// it to the other half's core), the other's from CB_SQX, the whole (t0 + t1) + (t2 + t3) as the unsplit form's (either
// order: the addition commutes)
#ifndef KDA_PREP_SPLIT
#define KDA_PREP_SPLIT 0
#endif
namespace kda_prepare_compute {
constexpr uint32_t CB_CUR = 0;
constexpr uint32_t CB_H0 = 1;
constexpr uint32_t CB_H1 = 2;
constexpr uint32_t CB_H2 = 3;
constexpr uint32_t CB_TAP = 4;
constexpr uint32_t CB_F = 5;
constexpr uint32_t CB_DT = 6;
constexpr uint32_t CB_RATE = 7;
constexpr uint32_t CB_BETA = 8;
constexpr uint32_t CB_ONES = 9;
constexpr uint32_t CB_QKV = 16;
constexpr uint32_t CB_SQ = 17;
constexpr uint32_t CB_INV = 18;
constexpr uint32_t CB_OUT = 19;
constexpr uint32_t CB_BOUT = 20;
constexpr uint32_t CB_CURH = 22;
constexpr uint32_t CB_CUR2 = 11;  // tokens 2 and 3: the conv input, f and beta logit; the earlier inputs in fp32
constexpr uint32_t CB_XT = 12;
constexpr uint32_t CB_F2 = 13;
constexpr uint32_t CB_BETA2 = 14;
constexpr uint32_t CB_CUR3 = 15;
constexpr uint32_t CB_F3 = 26;
constexpr uint32_t CB_BETA3 = 27;
constexpr uint32_t CB_SQP = 30;  // split: the half's sum of squares (out), the other half's (in)
constexpr uint32_t CB_SQX = 31;
constexpr bool SPLIT = KDA_PREP_SPLIT != 0;
constexpr uint32_t TOKENS = KDA_PREP_TOKENS;
constexpr uint32_t STACK = KDA_PREP_STACK;
static_assert(STACK == 1 || TOKENS == 1, "the stacked form takes one token a pass");
constexpr uint32_t N = 4;
constexpr uint32_t L2_EPS_BITS = 0x358637bdu;  // 1e-6f
constexpr uint32_t FMT_FP32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t FMT_BF16 = static_cast<uint32_t>(DataFormat::Float16_b);
constexpr uint32_t USERS = get_compile_time_arg_val(2);
constexpr uint32_t ROWS = USERS < 16 ? USERS : 16;  // a block's rows

FORCE_INLINE void pack_four(uint32_t cb) {
    for (uint32_t i = 0; i < N; ++i) {
        pack_tile<true>(i, cb, i);
    }
}

// Dest[j] += src[at + slot] * tap[term][slot] for slots c0, c0 + 1 (Dest 0, 1 the sums; 2, 3 the term; 4, 5 the taps,
// row 0 broadcast down the rows); the conv input also to Dest 6, 7 (for the history)
FORCE_INLINE void conv_term(uint32_t src, uint32_t term, uint32_t c0, bool first, uint32_t at = 0) {
    const bool input = src == CB_CUR;  // a token's bf16 conv input
    if (input) {  // SrcA as the startup's (CB_SQ, fp32 through SrcA) otherwise: the ones matmul needs it
        reconfig_data_format_srca(CB_SQ, src);
    }
    copy_init(src);
    copy_tile(src, at + c0, first ? 0 : 2);
    copy_tile(src, at + c0 + 1, first ? 1 : 3);
    if (input) {
        copy_tile(src, at + c0, 6);
        copy_tile(src, at + c0 + 1, 7);
        reconfig_data_format_srca(src, CB_SQ);
    }
    unary_bcast_init<BroadcastType::ROW>(CB_TAP);
    unary_bcast<BroadcastType::ROW>(CB_TAP, term * N + c0, 4);
    unary_bcast<BroadcastType::ROW>(CB_TAP, term * N + c0 + 1, 5);
    mul_binary_tile_init();
    if (first) {
        kda_rows::mul<ROWS>(0, 4, 0);
        kda_rows::mul<ROWS>(1, 5, 1);
    } else {
        kda_rows::mul<ROWS>(2, 4, 2);
        kda_rows::mul<ROWS>(3, 5, 3);
        add_binary_tile_init();
        kda_rows::add<ROWS>(0, 2, 0);
        kda_rows::add<ROWS>(1, 3, 1);
    }
}


FORCE_INLINE void run() {
    compute_kernel_hw_startup(CB_SQ, CB_ONES, CB_INV);
    constexpr uint32_t lower_bits = get_compile_time_arg_val(0);
    constexpr uint32_t q_scale_bits = get_compile_time_arg_val(1);
    constexpr uint32_t blocks = (USERS + 15) / 16;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t c_lo = SPLIT ? get_arg_val<uint32_t>(2) : 0;  // the item's channel tiles
    const uint32_t c_n = SPLIT ? get_arg_val<uint32_t>(3) : N;

    CircularBuffer cur(CB_CUR), h0(CB_H0), h1(CB_H1), h2(CB_H2), tap(CB_TAP), f(CB_F), dt(CB_DT), rate(CB_RATE),
        beta(CB_BETA), ones(CB_ONES), qkv(CB_QKV), sq(CB_SQ), inv(CB_INV), out(CB_OUT), bout(CB_BOUT), curh(CB_CURH);
    ones.wait_front(1);

    for (uint32_t u = start; u < start + count; ++u) {
        DeviceZoneScopedN("PC-UNIT");  // zones to the end of their scope: the phases' times are differences
        const uint32_t part = (u / blocks) % 4;
        if (part < 3) {
          for (uint32_t sp = 0; sp < STACK; ++sp) {
          for (uint32_t pass = 0; pass < TOKENS; ++pass) {
            DeviceZoneScopedN("PC-QKV");
            // the conv and SiLU, two tiles per pass, rounded to bf16 into CB_QKV; the conv input in fp32 into CB_CURH
            // (and, but the last token's, into CB_XT: the later tokens' history)
            const uint32_t cb_in = CB_CUR;  // (a token's tiles at the front in turn)
            const bool later = pass + 1 < TOKENS;  // a later token's history takes this input
            CircularBuffer in(cb_in), xt(CB_XT);
            in.wait_front(N);
            if (pass == 0) {
                h0.wait_front(N);
                h1.wait_front(N);
                h2.wait_front(N);
                tap.wait_front(4 * N);
            } else {
                xt.wait_front(pass * N);
            }
            qkv.reserve_back(N);
            curh.reserve_back(N);
            if (later) {
                xt.reserve_back(N);
            }
            for (uint32_t c0 = c_lo; c0 < c_lo + c_n; c0 += 2) {
                tile_regs_acquire();
                for (uint32_t term = 0; term < 4; ++term) {  // (h0, h1, h2, x_0 .. x_pass) from pass on
                    const uint32_t i = pass + term;
                    if (i < 3) {
                        conv_term(i == 0 ? CB_H0 : (i == 1 ? CB_H1 : CB_H2), term, c0, term == 0);
                    } else if (i - 3 < pass) {
                        conv_term(CB_XT, term, c0, term == 0, (i - 3) * N);
                    } else {
                        conv_term(cb_in, term, c0, term == 0);
                    }
                }
                silu_tile_init();
                kda_rows::silu<ROWS>(0);
                kda_rows::silu<ROWS>(1);
                typecast_tile_init<FMT_FP32, FMT_BF16>();
                kda_rows::round_bf16<ROWS>(0);
                kda_rows::round_bf16<ROWS>(1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, CB_QKV, c0);
                pack_tile<true>(1, CB_QKV, c0 + 1);
                pack_tile<true>(6, CB_CURH, c0);
                pack_tile<true>(7, CB_CURH, c0 + 1);
                if (later) {
                    pack_tile<true>(6, CB_XT, c0);
                    pack_tile<true>(7, CB_XT, c0 + 1);
                }
                tile_regs_release();
            }
            qkv.push_back(N);
            curh.push_back(N);
            in.pop_front(N);
            if (later) {  // h_pass leaves the window; the rest and the taps are the later passes' too
                xt.push_back(N);
                if (pass < 3) {
                    (pass == 0 ? h0 : (pass == 1 ? h1 : h2)).pop_front(N);
                }
            } else {
                if (pass > 0) {
                    xt.pop_front(pass * N);
                }
                if (pass == 0) {
                    h0.pop_front(N);
                }
                if (pass <= 1) {
                    h1.pop_front(N);
                }
                if (pass <= 2) {
                    h2.pop_front(N);
                }
                if (sp + 1 == STACK) {
                    tap.pop_front(4 * N);
                }
            }

            {
            DeviceZoneScopedN("PC-NORM");
            qkv.wait_front(N);
            out.reserve_back(N);
            if (part == 2) {  // v: the rounded values
                tile_regs_acquire();
                copy_init(CB_QKV);
                for (uint32_t c = 0; c < c_n; ++c) {
                    copy_tile(CB_QKV, c_lo + c, c);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t c = 0; c < c_n; ++c) {
                    pack_tile<true>(c, CB_OUT, c_lo + c);
                }
                tile_regs_release();
            } else {  // q, k: the L2 norm
                if constexpr (SPLIT) {
                    // the half's sum of squares, for the other half's core
                    CircularBuffer sqp(CB_SQP), sqx(CB_SQX);
                    sqp.reserve_back(1);
                    tile_regs_acquire();
                    copy_init(CB_QKV);
                    copy_tile(CB_QKV, c_lo, 0);
                    copy_tile(CB_QKV, c_lo + 1, 1);
                    mul_binary_tile_init();
                    kda_rows::mul<ROWS>(0, 0, 0);
                    kda_rows::mul<ROWS>(1, 1, 1);
                    add_binary_tile_init();
                    kda_rows::add<ROWS>(0, 1, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, CB_SQP);
                    tile_regs_release();
                    sqp.push_back(1);
                    // the whole: this half's again plus the other's
                    sqx.wait_front(1);
                    sq.reserve_back(1);
                    tile_regs_acquire();
                    copy_init(CB_QKV);
                    copy_tile(CB_QKV, c_lo, 0);
                    copy_tile(CB_QKV, c_lo + 1, 1);
                    mul_binary_tile_init();
                    kda_rows::mul<ROWS>(0, 0, 0);
                    kda_rows::mul<ROWS>(1, 1, 1);
                    add_binary_tile_init();
                    kda_rows::add<ROWS>(0, 1, 0);
                    copy_init(CB_SQX);
                    copy_tile(CB_SQX, 0, 1);
                    add_binary_tile_init();
                    kda_rows::add<ROWS>(0, 1, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, CB_SQ);
                    tile_regs_release();
                    sq.push_back(1);
                    sqx.pop_front(1);
                } else {
                sq.reserve_back(1);
                tile_regs_acquire();
                copy_init(CB_QKV);
                for (uint32_t c = 0; c < N; ++c) {
                    copy_tile(CB_QKV, c, c);
                }
                mul_binary_tile_init();
                for (uint32_t c = 0; c < N; ++c) {
                    kda_rows::mul<ROWS>(c, c, c);
                }
                add_binary_tile_init();
                kda_rows::add<ROWS>(0, 1, 0);
                kda_rows::add<ROWS>(2, 3, 2);
                kda_rows::add<ROWS>(0, 2, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_SQ);
                tile_regs_release();
                sq.push_back(1);
                }

                sq.wait_front(1);
                inv.reserve_back(1);
                tile_regs_acquire();
                matmul_init(CB_SQ, CB_ONES);
                matmul_tiles(CB_SQ, CB_ONES, 0, 0, 0);  // row 0: the sum in every column
                binop_with_scalar_tile_init();
                kda_rows::add_scalar<ROWS>(0, L2_EPS_BITS);
                rsqrt_tile_init();
                kda_rows::rsqrt<ROWS>(0);
                if (part == 0) {
                    binop_with_scalar_tile_init();
                    kda_rows::mul_scalar<ROWS>(0, q_scale_bits);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_INV);
                tile_regs_release();
                inv.push_back(1);
                sq.pop_front(1);

                inv.wait_front(1);
                tile_regs_acquire();
                copy_init(CB_QKV);
                for (uint32_t c = 0; c < c_n; ++c) {
                    copy_tile(CB_QKV, c_lo + c, c);
                }
                copy_init(CB_INV);
                copy_tile(CB_INV, 0, N);
                mul_binary_tile_init();
                for (uint32_t c = 0; c < c_n; ++c) {
                    kda_rows::mul<ROWS>(c, N, c);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t c = 0; c < c_n; ++c) {
                    pack_tile<true>(c, CB_OUT, c_lo + c);
                }
                tile_regs_release();
                inv.pop_front(1);
            }
            out.push_back(N);
            qkv.pop_front(N);
            }  // the zone's block
          }  // the passes
          }  // the stacked passes
        } else {
          for (uint32_t sp = 0; sp < STACK; ++sp) {
          for (uint32_t pass = 0; pass < TOKENS; ++pass) {
            DeviceZoneScopedN("PC-GATE");
            const uint32_t cb_f = CB_F;  // (a token's tiles at the front in turn)
            const uint32_t cb_beta = CB_BETA;
            CircularBuffer fx(cb_f), bx(cb_beta);
            // e = exp(lower * sigmoid((f + dt_bias) * rate))
            fx.wait_front(N);
            if (pass == 0 && sp == 0) {
                dt.wait_front(N);
                rate.wait_front(N);
            }
            out.reserve_back(N);
            tile_regs_acquire();
            reconfig_data_format_srca(CB_SQ, cb_f);
            copy_init(cb_f);
            for (uint32_t c = 0; c < N; ++c) {
                copy_tile(cb_f, c, c);
            }
            reconfig_data_format_srca(cb_f, CB_SQ);
            unary_bcast_init<BroadcastType::ROW>(CB_DT);
            for (uint32_t c = 0; c < N; ++c) {
                unary_bcast<BroadcastType::ROW>(CB_DT, c, N + c);
            }
            add_binary_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::add<ROWS>(c, N + c, c);
            }
            unary_bcast_init<BroadcastType::ROW>(CB_RATE);
            for (uint32_t c = 0; c < N; ++c) {
                unary_bcast<BroadcastType::ROW>(CB_RATE, c, N + c);
            }
            mul_binary_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::mul<ROWS>(c, N + c, c);
            }
            sigmoid_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::sigmoid<ROWS>(c);
            }
            binop_with_scalar_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::mul_scalar<ROWS>(c, lower_bits);
            }
            exp_tile_init();
            for (uint32_t c = 0; c < N; ++c) {
                kda_rows::exp<ROWS>(c);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_four(CB_OUT);
            tile_regs_release();
            out.push_back(N);
            fx.pop_front(N);
            if (pass + 1 == TOKENS && sp + 1 == STACK) {
                dt.pop_front(N);
                rate.pop_front(N);
            }

            // beta = sigmoid(logit), rounded to bf16 as the composite's bf16 sigmoid (held in fp32)
            bx.wait_front(1);
            bout.reserve_back(1);
            tile_regs_acquire();
            copy_init(cb_beta);
            copy_tile(cb_beta, 0, 0);
            sigmoid_tile_init();
            kda_rows::sigmoid<ROWS>(0);
            typecast_tile_init<FMT_FP32, FMT_BF16>();
            kda_rows::round_bf16<ROWS>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_BOUT);
            tile_regs_release();
            bout.push_back(1);
            bx.pop_front(1);
          }  // the passes
          }  // the stacked passes
        }
    }
}

}  // namespace kda_prepare_compute
