// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A gate | up core, one tile row of tokens, per item (the chosen experts in groups of 64 / T; the count from BRISC's
// mailbox): g = x @ gate column tile, u = x @ up column tile of slot p / T's expert over the K blocks of CB_W1 (fp32 in
// Dest, HiFi4), each rounded to bf16 as the matmul outputs were; then h = silu(min(g, limit)) * clamp(u, -limit,
// limit), times each token's routing weight for this expert (the routing column broadcast across the columns), rounded
// to bf16, to CB_HOUT.  The W1 blocks come starting at block `rot` (runtime arg 0; the cores start in different DRAM
// banks), so x's K tiles are taken in that order.  Packed units (2-16 tile rows; this core's K slice j and column j of
// slot p / T): per item, the unit's K slice of x = sum over the token rows r of its 0/1 tile r times x's slice of row r
// (exact copies, bf16, by matmul_block over the 8 K tiles of a pass: one 0/1 tile in SrcB per token row); the partial
// gate and up of the T columns over the slice (the slice's W1 block of each column, fp32, 4 columns at a time in Dest)
// to CB_PART for BRISC to send; then this column's T pairs from the row (CB_PIN, fp32 unpacked to Dest) added in fp32
// in the order of the slices, and h made from them as above.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/cb_api.h"
#include "onehot_mm.hpp"
#include "api/compute/experimental/reg_api.h"

namespace {
constexpr uint32_t CB_X = 0;
constexpr uint32_t CB_W1 = 1;
constexpr uint32_t CB_WCOL = 2;
constexpr uint32_t CB_XALL = 5;  // packed units: the tokens' K slice, the unit's 0/1 tiles, its gathered K slice,
constexpr uint32_t CB_P = 6;     // the T columns' partials, the row's pairs of this column
constexpr uint32_t CB_XS = 7;
constexpr uint32_t CB_HOUT = 16;
constexpr uint32_t CB_PART = 19;
constexpr uint32_t CB_PIN = 20;
constexpr uint32_t FMT_FP32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t FMT_BF16 = static_cast<uint32_t>(DataFormat::Float16_b);
}  // namespace

void kernel_main() {
    constexpr uint32_t kd = get_compile_time_arg_val(0);     // K tiles
    constexpr uint32_t block = get_compile_time_arg_val(1);  // K tiles per CB_W1 block (2 tiles each: gate, up)
    constexpr uint32_t limit_bits = get_compile_time_arg_val(2);
    constexpr uint32_t neg_limit_bits = get_compile_time_arg_val(3);
    constexpr uint32_t rows = get_compile_time_arg_val(4);    // tile rows of tokens
    constexpr bool units = get_compile_time_arg_val(5) != 0;  // packed units (block = the K slice)
    constexpr uint32_t T = get_compile_time_arg_val(6);       // an expert's column tiles on the chip
    // packed units: the gather over only the token rows the unit has tokens in (gu_ctrl.cpp's mask after its 0/1
    // tiles; the others' tiles are zero, so the sums are the same)
    constexpr bool gather_mask = get_compile_time_arg_val(7) != 0;
    // the gather's 0/1 matmuls at this fidelity (onehot_mm.hpp; 4: the kernel's HiFi4 through matmul.h)
    constexpr auto OF = static_cast<MathFidelity>(get_compile_time_arg_val(8));
    constexpr bool hifi4 = OF == MathFidelity::HiFi4;
    // packed units: each Dest section cleared by MATH as it is acquired (ZEROACC, reg_api.h's math clear, for every
    // section of the units' path) in place of the SFPU's zero fills before each accumulation
    constexpr bool mclear = get_compile_time_arg_val(9) != 0;
    auto acquire_zeros = [&](uint32_t tiles) {  // Dest tiles 0 .. tiles - 1 zero
        if constexpr (mclear) {
            tile_regs_acquire_math_clear();
        } else {
            tile_regs_acquire();
            fill_tile_init();  // Dest is not cleared between programs
            for (uint32_t t = 0; t < tiles; ++t) {
                fill_tile(t, 0.0f);
            }
        }
    };
    auto release = [&]() {
        if constexpr (mclear) {
            tile_regs_release_math_clear();
        } else {
            tile_regs_release();
        }
    };
    const uint32_t rot = get_arg_val<uint32_t>(0);

    uint32_t n = 0;
    UNPACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    MATH(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    PACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)

    CircularBuffer x(CB_X), w1(CB_W1), wcol(CB_WCOL), hout(CB_HOUT);
    if constexpr (units) {
        constexpr uint32_t ks = block;
        compute_kernel_hw_startup<SrcOrder::Reverse>(CB_P, CB_XALL, CB_XS);  // matmul: in0 -> SrcB, in1 -> SrcA
        CircularBuffer xall(CB_XALL), pcb(CB_P), xs(CB_XS), part(CB_PART), pin(CB_PIN);
        xall.wait_front(rows * ks);
        // item i's K slice of x and the T columns' partials to CB_PART (the config on entry and after: the gather's)
#ifdef MOE_PROG
        constexpr uint32_t p_tiles = rows + 1;  // (gu_ctrl.cpp: word 1 of the last, the unit read with its correction)
#else
        constexpr uint32_t p_tiles = rows + (gather_mask ? 1 : 0);
#endif
        auto partials = [&](uint32_t i) {
            pcb.wait_front(p_tiles);
            uint32_t hit = (1u << rows) - 1;
            if constexpr (gather_mask) {
                hit = read_tile_value(CB_P, rows, 0);
            }
            uint32_t corr = 0;  // the progressive form: each column's correction block after its base, same Dest
#ifdef MOE_PROG
            corr = read_tile_value(CB_P, rows, 1);
#endif
            xs.reserve_back(ks);
            for (uint32_t k0 = 0; k0 < ks; k0 += 8) {  // the unit's K slice of x
                acquire_zeros(8);
                if constexpr (hifi4) {
                    matmul_block_init(CB_P, CB_XALL, 0, 8, 1, rows);
                } else {
                    onehot_block_init<OF>(CB_P, CB_XALL, 8, 1, rows);
                }
                for (uint32_t r = 0; r < rows; ++r) {  // 0/1 tile r times x's tiles (r, k0 ..)
                    if ((hit >> r) & 1u) {
                        if constexpr (hifi4) {
                            matmul_block(CB_P, CB_XALL, r, r * ks + k0, 0, 0, 8, 1, rows);
                        } else {
                            onehot_block<OF>(CB_P, CB_XALL, r, r * ks + k0, 0, 8, 1, rows);
                        }
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < 8; ++t) {
                    pack_tile(t, CB_XS);
                }
                release();
            }
            xs.push_back(ks);
            pcb.pop_front(p_tiles);

            // the T columns' partial (gate, up) over the slice, 4 columns at a time
            xs.wait_front(ks);
            part.reserve_back(2 * T);
            reconfig_data_format_srca(CB_XALL, CB_W1);
            reconfig_data_format_srcb(CB_P, CB_XS);
            pack_reconfig_data_format(CB_XS, CB_PART);
            for (uint32_t c0 = 0; c0 < T; c0 += 4) {
                acquire_zeros(8);
                matmul_block_init(CB_XS, CB_W1, 0, 2, 1, ks);
                for (uint32_t c = 0; c < 4; ++c) {
                    for (uint32_t part = 0; part < (corr ? 2u : 1u); ++part) {  // (the base, then the correction)
                        w1.wait_front(2 * ks);
                        for (uint32_t k = 0; k < ks; ++k) {  // x tile k times the row's (gate, up) tiles
                            matmul_block(CB_XS, CB_W1, k, 2 * k, 2 * c, 0, 2, 1, ks);
                        }
                        w1.pop_front(2 * ks);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < 8; ++t) {
                    pack_tile<true>(t, CB_PART, 2 * c0 + t);
                }
                release();
            }
            part.push_back(2 * T);
            xs.pop_front(ks);
            reconfig_data_format_srca(CB_W1, CB_XALL);
            reconfig_data_format_srcb(CB_XS, CB_P);
            pack_reconfig_data_format(CB_PART, CB_XS);
        };
        // item i's pairs of this column added in the slices' order (fp32), then h (the config on entry and after:
        // the gather's)
        auto sum_h = [&](uint32_t i) {
            pin.wait_front(2 * T);
            wcol.wait_front(1);
            reconfig_data_format_srca(CB_XALL, CB_PIN);
            pack_reconfig_data_format(CB_XS, CB_HOUT);
            if constexpr (mclear) {
                tile_regs_acquire_math_clear();
            } else {
                tile_regs_acquire();
            }
            copy_init(CB_PIN);
            copy_tile(CB_PIN, 0, 0);
            copy_tile(CB_PIN, 1, 1);
            for (uint32_t j = 1; j < T; ++j) {
                copy_init(CB_PIN);
                copy_tile(CB_PIN, 2 * j, 2);
                copy_tile(CB_PIN, 2 * j + 1, 3);
                add_binary_tile_init();
                add_binary_tile(0, 2, 0);
                add_binary_tile(1, 3, 1);
            }
            typecast_tile_init<FMT_FP32, FMT_BF16>();
            typecast_tile<FMT_FP32, FMT_BF16>(0);
            typecast_tile<FMT_FP32, FMT_BF16>(1);
            unary_min_tile_init();
            unary_min_tile(0, limit_bits);
            silu_tile_init();
            silu_tile(0);
            hardtanh_tile_init();
            hardtanh_tile(1, neg_limit_bits, limit_bits);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            unary_bcast_init<BroadcastType::COL>(CB_WCOL);
            unary_bcast<BroadcastType::COL>(CB_WCOL, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            typecast_tile_init<FMT_FP32, FMT_BF16>();
            typecast_tile<FMT_FP32, FMT_BF16>(0);
            tile_regs_commit();
            tile_regs_wait();
            hout.reserve_back(1);
            pack_tile(0, CB_HOUT);
            hout.push_back(1);
            release();
            pin.pop_front(2 * T);
            wcol.pop_front(1);
            reconfig_data_format_srca(CB_PIN, CB_XALL);
            pack_reconfig_data_format(CB_HOUT, CB_XS);
        };
        // per item, its partials, then its sum (making item i + 1's partials before item i's sum, to put the row's
        // exchange under the matmuls, held item i's h back by a matmul and the down cores' W2 stream with it: 128 /
        // 256 tokens 661 / 780 -> 684 / 835 us)
        for (uint32_t i = 0; i < n; ++i) {
            partials(i);
            sum_h(i);
        }
        xall.pop_front(rows * ks);
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_W1, CB_HOUT);  // matmul: in0 -> SrcB, in1 -> SrcA
    static_assert(units || rows == 1, "gu_compute: more than one tile row of tokens in packed units only");
    x.wait_front(kd);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        fill_tile_init();  // Dest is not cleared between programs
        fill_tile(0, 0.0f);
        fill_tile(1, 0.0f);
        matmul_block_init(CB_X, CB_W1, 0, 2, 1, block);
        for (uint32_t j = 0; j < kd / block; ++j) {
            const uint32_t b = ((j + rot) % (kd / block)) * block;
            w1.wait_front(2 * block);
            for (uint32_t k = 0; k < block; ++k) {  // x tile b + k times the row's (gate, up) tiles
                matmul_block(CB_X, CB_W1, b + k, 2 * k, 0, 0, 2, 1, block);
            }
            w1.pop_front(2 * block);
        }
        typecast_tile_init<FMT_FP32, FMT_BF16>();
        typecast_tile<FMT_FP32, FMT_BF16>(0);
        typecast_tile<FMT_FP32, FMT_BF16>(1);
        unary_min_tile_init();
        unary_min_tile(0, limit_bits);
        silu_tile_init();
        silu_tile(0);
        hardtanh_tile_init();
        hardtanh_tile(1, neg_limit_bits, limit_bits);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        wcol.wait_front(1);
        unary_bcast_init<BroadcastType::COL>(CB_WCOL);
        unary_bcast<BroadcastType::COL>(CB_WCOL, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);
        typecast_tile_init<FMT_FP32, FMT_BF16>();
        typecast_tile<FMT_FP32, FMT_BF16>(0);
        tile_regs_commit();
        tile_regs_wait();
        hout.reserve_back(1);
        pack_tile(0, CB_HOUT);
        hout.push_back(1);
        tile_regs_release();
        wcol.pop_front(1);
    }
    x.pop_front(kd);
}
