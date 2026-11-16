// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A down core: one row of tokens: y[:, its C column tiles] = sum over the items (groups of up to 64 / T chosen
// experts; the chosen experts' count from BRISC's mailbox) of h'_j @ down_j[:, columns], h' the gate | up cores'
// weighted h (64 tiles a slot of CB_H), all in one Dest accumulation (fp32, HiFi4), rounded to bf16 once at the end
// (zero when no expert is active).  Packed units (2-16 tile rows; G = 64 / T slots an item, each one expert's tokens
// packed in a tile row): per slot, y_s = its T h tiles times its expert's T rows of W2 (fp32, to CB_YS), then for each
// token row r, its 0/1 tile transposed times y_s (y_s at SrcA's 19 bits; the products exact) added to that row's sums
// in L1 (fp32, the packer's accumulation) over the slots and items; at the end each row's sums rounded to bf16 once.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/cb_api.h"
#include "onehot_mm.hpp"
#include "api/compute/experimental/reg_api.h"

namespace {
constexpr uint32_t CB_W2 = 1;
constexpr uint32_t CB_H = 4;
constexpr uint32_t CB_PT = 6;  // packed units: the slots' 0/1 tiles transposed, a slot's y (fp32)
constexpr uint32_t CB_YS = 7;
constexpr uint32_t CB_Y = 17;
constexpr uint32_t CB_ACC = 18;
constexpr uint32_t FMT_FP32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t FMT_BF16 = static_cast<uint32_t>(DataFormat::Float16_b);
}  // namespace

void kernel_main() {
    constexpr uint32_t ki = get_compile_time_arg_val(0);     // K tiles (h tiles per expert)
    constexpr uint32_t block = get_compile_time_arg_val(1);  // K tiles per CB_W2 block
    constexpr uint32_t cols = get_compile_time_arg_val(2);   // output column tiles
    constexpr uint32_t rows = get_compile_time_arg_val(3);   // tile rows of tokens (1, or 2-16 with packed units)
    constexpr bool units = get_compile_time_arg_val(4) != 0;  // packed units
    constexpr uint32_t T = get_compile_time_arg_val(5);       // an expert's column tiles on the chip
    // packed units: slots per scatter pass (1: each slot's y into the rows' sums on its own).  More: the pass's y
    // tiles kept (CB_YS) and each token row's contributions of the pass summed in Dest by one matmul_block over the
    // slots (the 0/1 tiles token-row-major, dn_ctrl.cpp), then added to the row's sums in L1 once (fp32 in another
    // order: the ~4,600 packer accumulations of 16 rows a call down to a quarter at 4)
    constexpr uint32_t HS = get_compile_time_arg_val(6);
    // ... and with the item's incidence (CB_PT's tile G rows after its 0/1 tiles: word r, bit s set when slot s has a
    // token in token row r; dn_ctrl.cpp): a row's pass takes only the slots with a token in it, a row no slot of the
    // pass touches is left out, and the rows' sums start from zeros packed once (the packer always accumulates)
    constexpr bool masked = get_compile_time_arg_val(7) != 0;
    // the scatter's 0/1 matmuls at this fidelity (onehot_mm.hpp; 4: the kernel's HiFi4 through matmul.h)
    constexpr auto OF = static_cast<MathFidelity>(get_compile_time_arg_val(8));
    constexpr bool hifi4 = OF == MathFidelity::HiFi4;
    // packed units: each Dest section cleared by MATH as it is acquired (ZEROACC, reg_api.h's math clear, for every
    // section of the kernel) in place of the SFPU's zero fills before each accumulation
    constexpr bool mclear = get_compile_time_arg_val(9) != 0;
    // packed units: the units' count from BRISC's mailbox (in place of the items'), and the last item's slots past it
    // left out (their W2 blocks zeros and their 0/1 tiles empty: they add zeros; the last item's work is the op's
    // tail, after its h has come).  Slots per scatter pass 1 only
    constexpr bool live_only = get_compile_time_arg_val(10) != 0;
    // (The rows' sums rounded to bf16 by the packer in place of the typecast: 256 / 512 tokens 611.6 / 634.2 -> 606.2
    // / 626.1 us, but 1 bf16 ulp off in places: the packer does not round as the typecast does.)  Packed
    // units' finish by the packer's rounding (MOE_EXPERTS_FIN_PACK=1, a value-changing option;
    // 0: typecast.h's, the default)
    constexpr bool fin_pack = get_compile_time_arg_val(11) != 0;
    // the column's way back in the op (rs_move.cpp): packed units' rows rounded slice by
    // slice (a slice: a mesh row's rs_R tile rows) in the order the ring needs them, slices c + 4, c + 3, c + 5, c + 2,
    // c + 6, c + 1, c + 7, c (c: this chip's mesh row, runtime arg 0), in place of 0 .. rows - 1 (the same values)
    constexpr bool rs = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t rs_R = get_compile_time_arg_val(13);
    auto row_of = [&](uint32_t r) -> uint32_t {
        if constexpr (rs) {
            constexpr uint32_t off[8] = {4, 3, 5, 2, 6, 1, 7, 0};
            const uint32_t c = get_arg_val<uint32_t>(0);
            return ((c + off[r / rs_R]) & 7) * rs_R + r % rs_R;
        } else {
            return r;
        }
    };
    auto acquire = [&]() {
        if constexpr (mclear) {
            tile_regs_acquire_math_clear();
        } else {
            tile_regs_acquire();
        }
    };
    auto release = [&]() {
        if constexpr (mclear) {
            tile_regs_release_math_clear();
        } else {
            tile_regs_release();
        }
    };
    auto acquire_zeros = [&](uint32_t tiles) {  // Dest tiles 0 .. tiles - 1 zero
        acquire();
        if constexpr (!mclear) {
            fill_tile_init();  // Dest is not cleared between programs
            for (uint32_t c = 0; c < tiles; ++c) {
                fill_tile(c, 0.0f);
            }
        }
    };

    uint32_t n = 0;
    UNPACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    MATH(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    PACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    static_assert(units || rows == 1, "dn_compute: more than one tile row of tokens in packed units only");
    // one row of tokens: the chosen experts' count came; the items are groups of up to G of them, sized as
    // sliced_slot does (moe_decode_experts_common.hpp: the first chosen % items one larger)
    constexpr uint32_t GS = 64 / T;
    const uint32_t chosen = n;
    if constexpr (!units) {
        n = (chosen + GS - 1) / GS;
    }
    if constexpr (units && live_only) {  // (chosen: the units)
        n = (chosen + GS - 1) / GS;
    }
    // K tiles of item i to compute (one row of tokens): its live slots' (a slot past the item's size has zero h and
    // zero W2: skipping it leaves the sums as they were)
    auto live = [&](uint32_t i) -> uint32_t {
        const uint32_t size = chosen / n + (i < chosen % n ? 1 : 0);
        return size * T;
    };

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_H, CB_W2, CB_Y);  // matmul: in0 -> SrcB, in1 -> SrcA
    CircularBuffer w2(CB_W2), h(CB_H), y(CB_Y);
    auto start = [&]() {
        acquire_zeros(cols);
        matmul_block_init(CB_H, CB_W2, 0, cols, 1, block);
    };
    auto accumulate = [&](uint32_t kt) {  // Dest[0 .. cols - 1] += h @ the item's down columns, its first kt K tiles
        h.wait_front(ki);
        for (uint32_t b = 0; b < ki; b += block) {
            w2.wait_front(cols * block);
            for (uint32_t k = 0; k < block && b + k < kt; ++k) {  // h tile b + k times the row's C tiles
                matmul_block(CB_H, CB_W2, b + k, k * cols, 0, 0, cols, 1, block);
            }
            w2.pop_front(cols * block);
        }
        h.pop_front(ki);
    };
    auto finish = [&]() {  // rounded to bf16, to CB_Y
        typecast_tile_init<FMT_FP32, FMT_BF16>();
        for (uint32_t c = 0; c < cols; ++c) {
            typecast_tile<FMT_FP32, FMT_BF16>(c);
        }
        tile_regs_commit();
        tile_regs_wait();
        y.reserve_back(cols);
        for (uint32_t c = 0; c < cols; ++c) {
            pack_tile(c, CB_Y);
        }
        y.push_back(cols);
        release();
    };

    if constexpr (units && HS > 1) {
        constexpr uint32_t G = ki / T;
        static_assert(G % (HS > 0 ? HS : 1) == 0, "the scatter passes must hold whole slots");
        CircularBuffer pt(CB_PT), ys(CB_YS), acc(CB_ACC);
#ifdef MOE_PROG
        constexpr uint32_t pt_tiles = G * rows + 1;  // (dn_ctrl.cpp: word 31 of the last, the slots' corrections)
#else
        constexpr uint32_t pt_tiles = G * rows + (masked ? 1 : 0);
#endif
        acc.reserve_back(rows * cols);
        pack_reconfig_data_format(CB_Y, CB_YS);  // fp32, as CB_ACC
        if constexpr (masked) {  // the rows' sums from zeros
            pack_reconfig_l1_acc(0);
            for (uint32_t r = 0; r < rows; ++r) {
                acquire();
                fill_tile_init();  // (written in full: packed)
                for (uint32_t c = 0; c < cols; ++c) {
                    fill_tile(c, 0.0f);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t c = 0; c < cols; ++c) {
                    pack_tile<true>(c, CB_ACC, r * cols + c);
                }
                release();
            }
        }
        for (uint32_t i = 0; i < n; ++i) {
            h.wait_front(ki);
            pt.wait_front(pt_tiles);
            uint32_t marked = 0;  // the progressive form: a block a slot, its base's T rows, then its correction's
#ifdef MOE_PROG
            marked = read_tile_value(CB_PT, G * rows, 31);
#endif
            for (uint32_t s0 = 0; s0 < G; s0 += HS) {
                for (uint32_t s = s0; s < s0 + HS; ++s) {  // y_s = its T h tiles times its T rows of W2 (fp32)
#ifdef MOE_PROG
                    w2.wait_front(cols * block);  // slot s's block
#else
                    if ((s * T) % block == 0) {
                        w2.wait_front(cols * block);  // the block holding slot s
                    }
#endif
                    acquire_zeros(cols);
                    matmul_block_init(CB_H, CB_W2, 0, cols, 1, block);
                    for (uint32_t c = 0; c < T; ++c) {
                        const uint32_t k = s * T + c;
#ifdef MOE_PROG
                        matmul_block(CB_H, CB_W2, k, c * cols, 0, 0, cols, 1, block);
                        if ((marked >> s) & 1u) {
                            matmul_block(CB_H, CB_W2, k, (T + c) * cols, 0, 0, cols, 1, block);
                        }
#else
                        matmul_block(CB_H, CB_W2, k, (k % block) * cols, 0, 0, cols, 1, block);
#endif
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    ys.reserve_back(cols);
                    for (uint32_t c = 0; c < cols; ++c) {
                        pack_tile(c, CB_YS);
                    }
                    ys.push_back(cols);
                    release();
#ifdef MOE_PROG
                    w2.pop_front(cols * block);
#else
                    if ((s * T + T) % block == 0) {
                        w2.pop_front(cols * block);
                    }
#endif
                }
                ys.wait_front(HS * cols);
                reconfig_data_format_srca(CB_W2, CB_YS);
                reconfig_data_format_srcb(CB_H, CB_PT);
                for (uint32_t r = 0; r < rows; ++r) {  // row r's 0/1 tiles of the pass's slots times their y
                    uint32_t hits = (1u << HS) - 1;
                    if constexpr (masked) {
                        hits = (read_tile_value(CB_PT, G * rows, r) >> s0) & ((1u << HS) - 1);
                        if (hits == 0) {
                            continue;
                        }
                    }
                    acquire_zeros(cols);
                    if constexpr (hifi4) {
                        matmul_block_init(CB_PT, CB_YS, 0, cols, 1, HS);
                    } else {
                        onehot_block_init<OF>(CB_PT, CB_YS, cols, 1, HS);
                    }
                    for (uint32_t k = 0; k < HS; ++k) {
                        if ((hits >> k) & 1u) {
                            if constexpr (hifi4) {
                                matmul_block(CB_PT, CB_YS, r * G + s0 + k, k * cols, 0, 0, cols, 1, HS);
                            } else {
                                onehot_block<OF>(CB_PT, CB_YS, r * G + s0 + k, k * cols, 0, cols, 1, HS);
                            }
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_l1_acc(masked || i > 0 || s0 > 0 ? 1 : 0);
                    for (uint32_t c = 0; c < cols; ++c) {
                        pack_tile<true>(c, CB_ACC, r * cols + c);
                    }
                    release();
                }
                pack_reconfig_l1_acc(0);
                ys.pop_front(HS * cols);
                reconfig_data_format_srca(CB_YS, CB_W2);
                reconfig_data_format_srcb(CB_PT, CB_H);
            }
            h.pop_front(ki);
            pt.pop_front(pt_tiles);
        }
        acc.push_back(rows * cols);  // the packer's sums to the unpacker
        acc.wait_front(rows * cols);
        pack_reconfig_data_format(CB_YS, CB_Y);
        reconfig_data_format_srca(CB_W2, CB_ACC);
        for (uint32_t r = 0; r < rows; ++r) {
            acquire();
            if (n == 0) {  // no unit: zeros
                fill_tile_init();
                for (uint32_t c = 0; c < cols; ++c) {
                    fill_tile(c, 0.0f);
                }
            } else {
                copy_init(CB_ACC);
                for (uint32_t c = 0; c < cols; ++c) {
                    copy_tile(CB_ACC, row_of(r) * cols + c, c);
                }
            }
            if constexpr (!fin_pack) {
                typecast_tile_init<FMT_FP32, FMT_BF16>();
                for (uint32_t c = 0; c < cols; ++c) {
                    typecast_tile<FMT_FP32, FMT_BF16>(c);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            y.reserve_back(cols);
            for (uint32_t c = 0; c < cols; ++c) {
                pack_tile(c, CB_Y);
            }
            y.push_back(cols);
            release();
        }
        acc.pop_front(rows * cols);
        return;
    }
    if constexpr (units) {
        constexpr uint32_t G = ki / T;
        CircularBuffer pt(CB_PT), ys(CB_YS), acc(CB_ACC);
        acc.reserve_back(rows * cols);
        pack_reconfig_data_format(CB_Y, CB_YS);  // fp32, as CB_ACC
#ifdef MOE_PROG
        constexpr uint32_t pt_tiles = G * rows + 1;  // (dn_ctrl.cpp: word 31 of the last, the slots' corrections)
#else
        constexpr uint32_t pt_tiles = G * rows;
#endif
        for (uint32_t i = 0; i < n; ++i) {
            h.wait_front(ki);
            pt.wait_front(pt_tiles);
            uint32_t marked = 0;  // the progressive form: a block a slot, its base's T rows, then its correction's
#ifdef MOE_PROG
            marked = read_tile_value(CB_PT, G * rows, 31);
#endif
            const uint32_t live_s = live_only && i + 1 == n ? chosen - i * G : G;  // the item's slots with a unit
            for (uint32_t s = 0; s < G; ++s) {
#ifdef MOE_PROG
                w2.wait_front(cols * block);  // slot s's block
                if (s >= live_s) {
                    w2.pop_front(cols * block);
                    continue;
                }
#else
                if ((s * T) % block == 0) {
                    w2.wait_front(cols * block);  // the block holding slot s
                }
                if (s >= live_s) {
                    if ((s * T + T) % block == 0) {
                        w2.pop_front(cols * block);
                    }
                    continue;
                }
#endif
                acquire_zeros(cols);
                matmul_block_init(CB_H, CB_W2, 0, cols, 1, block);
                for (uint32_t c = 0; c < T; ++c) {  // h tile s T + c times the block's row of C tiles
                    const uint32_t k = s * T + c;
#ifdef MOE_PROG
                    matmul_block(CB_H, CB_W2, k, c * cols, 0, 0, cols, 1, block);
                    if ((marked >> s) & 1u) {
                        matmul_block(CB_H, CB_W2, k, (T + c) * cols, 0, 0, cols, 1, block);
                    }
#else
                    matmul_block(CB_H, CB_W2, k, (k % block) * cols, 0, 0, cols, 1, block);
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                ys.reserve_back(cols);
                for (uint32_t c = 0; c < cols; ++c) {
                    pack_tile(c, CB_YS);
                }
                ys.push_back(cols);
                release();
#ifdef MOE_PROG
                w2.pop_front(cols * block);
#else
                if ((s * T + T) % block == 0) {
                    w2.pop_front(cols * block);
                }
#endif
                ys.wait_front(cols);
                reconfig_data_format_srca(CB_W2, CB_YS);
                reconfig_data_format_srcb(CB_H, CB_PT);
                for (uint32_t r = 0; r < rows; ++r) {
                    acquire_zeros(cols);
                    if constexpr (hifi4) {
                        matmul_init(CB_PT, CB_YS);
                        for (uint32_t c = 0; c < cols; ++c) {
                            matmul_tiles(CB_PT, CB_YS, s * rows + r, c, c);
                        }
                    } else {
                        onehot_init<OF>(CB_PT, CB_YS);
                        for (uint32_t c = 0; c < cols; ++c) {
                            onehot_tiles<OF>(CB_PT, CB_YS, s * rows + r, c, c);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_l1_acc(i > 0 || s > 0 ? 1 : 0);
                    for (uint32_t c = 0; c < cols; ++c) {
                        pack_tile<true>(c, CB_ACC, r * cols + c);
                    }
                    release();
                }
                pack_reconfig_l1_acc(0);
                ys.pop_front(cols);
                reconfig_data_format_srca(CB_YS, CB_W2);
                reconfig_data_format_srcb(CB_PT, CB_H);
            }
            h.pop_front(ki);
            pt.pop_front(pt_tiles);
        }
        acc.push_back(rows * cols);  // the packer's sums to the unpacker
        acc.wait_front(rows * cols);
        pack_reconfig_data_format(CB_YS, CB_Y);
        reconfig_data_format_srca(CB_W2, CB_ACC);
        for (uint32_t r = 0; r < rows; ++r) {
            acquire();
            if (n == 0) {  // no unit: zeros
                fill_tile_init();
                for (uint32_t c = 0; c < cols; ++c) {
                    fill_tile(c, 0.0f);
                }
            } else {
                copy_init(CB_ACC);
                for (uint32_t c = 0; c < cols; ++c) {
                    copy_tile(CB_ACC, row_of(r) * cols + c, c);
                }
            }
            if constexpr (!fin_pack) {
                typecast_tile_init<FMT_FP32, FMT_BF16>();
                for (uint32_t c = 0; c < cols; ++c) {
                    typecast_tile<FMT_FP32, FMT_BF16>(c);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            y.reserve_back(cols);
            for (uint32_t c = 0; c < cols; ++c) {
                pack_tile(c, CB_Y);
            }
            y.push_back(cols);
            release();
        }
        acc.pop_front(rows * cols);
        return;
    }
    {
        start();
        for (uint32_t i = 0; i < n; ++i) {
            accumulate(live(i));
        }
        finish();
    }
}
