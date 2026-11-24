// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This core's partial x[:, K slice] @ w[K slice, N block]: per sub-block of up to 8 columns (full-sync Dest: 8 fp32
// tiles), matmul_block over the K slice's tiles in order, accumulated in Dest (fp32, HiFi4), packed to CB_PART (fp32)
// or, with one slice, to CB_OUT in the output dtype.  On an owner, then: the ks slices' partials of its owned tiles,
// unpacked to Dest as fp32 and added in fp32 in the order of the slices (SFPU) as they come, packed once to CB_OUT.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

namespace {
constexpr uint32_t CB_X = 0;
constexpr uint32_t CB_W = 1;
constexpr uint32_t CB_RECV = 2;
constexpr uint32_t CB_PART = 16;
constexpr uint32_t CB_OUT = 17;
constexpr uint32_t CB_FGW = 4;     // fg's weight block (DM_FG)
constexpr uint32_t CB_FGIN = 5;    // fg's input, tile (r, k) at k mt + r
constexpr uint32_t CB_FGP = 18;    // fg's partials over its K blocks (fp32; unpacked to Dest as fp32)
constexpr uint32_t CB_FGOUT = 19;  // fg's output tiles (bf16)
constexpr uint32_t CB_CD = 6;      // conv's two delta tiles (DM_CONV): tap t of row i at 2 i + t
constexpr uint32_t CB_CE = 7;      // conv's E tiles, one a column
constexpr uint32_t CB_CS = 8;      // conv's shift S
constexpr uint32_t CB_CB = 9;      // conv's base tiles (fp32; unpacked to Dest as fp32), taps 0 and 1 a column
constexpr uint32_t CB_CO = 20;     // conv's output tiles (bf16)
}  // namespace

void kernel_main() {
    const uint32_t ncb = get_arg_val<uint32_t>(0);
    const uint32_t kc = get_arg_val<uint32_t>(1);
    const uint32_t cr = get_arg_val<uint32_t>(2);
    const uint32_t sb = get_arg_val<uint32_t>(3);
    const uint32_t ks = get_arg_val<uint32_t>(4);
    const uint32_t owned = get_arg_val<uint32_t>(5);
    constexpr bool no_matmul = get_compile_time_arg_val(0) != 0;  // a test switch: the weights only consumed
    // x's tile rows: with 2 (K whole, one slice) CB_X holds tile (r, k) at k mt + r and the output sub-blocks are mt x
    // sbw tiles in Dest (row-major), packed in that order
    constexpr uint32_t mt = get_compile_time_arg_val(1);
#ifdef DM_FG
    // fg (GLM53_KDA_FG_FUSED) on a core without a block: its cnt N2 columns of fg = in @ W2, each
    // output tile as ttnn.linear's default 1D config makes it (bmm_large_block_zm_fused_bias_activation.cpp, no bias,
    // packer L1 accumulation, fp32 Dest): K in blocks of 2 tiles; block 0 from zero into the partials (fp32), blocks
    // 1 .. nb - 2 from zero added into them by the packer, the last on the partials unpacked to Dest as fp32, packed
    // once to bf16.  The tiles are independent, so the sub-blocks (up to fg_sb columns x mt rows: Dest's 8 fp32 tiles)
    // may come in any order.  The partials' and the output's slots are whole CBs (fg_sb mt tiles) whatever the width,
    // so that every block packs to the same place.
    if (ncb == 0) {
        constexpr uint32_t fg_k = get_compile_time_arg_val(2);
        constexpr uint32_t fg_sb = get_compile_time_arg_val(3);
        constexpr uint32_t nblk = fg_k / 2, slot = fg_sb * mt;
        const uint32_t cnt = get_arg_val<uint32_t>(6);
        if (cnt == 0) {
            return;
        }
        DeviceZoneScopedN("FG");
        compute_kernel_hw_startup<SrcOrder::Reverse>(CB_FGIN, CB_FGW, CB_FGP);  // matmul: in0 -> SrcB, in1 -> SrcA
        CircularBuffer fin(CB_FGIN), fw(CB_FGW), fp(CB_FGP), fo(CB_FGOUT);
        fw.wait_front(fg_k * cnt);
        fin.wait_front(fg_k * mt);
        for (uint32_t c0 = 0; c0 < cnt; c0 += fg_sb) {
            const uint32_t w = cnt - c0 < fg_sb ? cnt - c0 : fg_sb, tiles = mt * w;
            for (uint32_t blk = 0; blk < nblk; ++blk) {
                const bool last = blk + 1 == nblk;
                tile_regs_acquire();
                if (last) {  // the partials into Dest as fp32 (ttnn's reload_from_cb_to_dst)
                    reconfig_data_format_srca(CB_FGW, CB_FGP);
                    copy_init(CB_FGP);
                    fp.wait_front(slot);
                    for (uint32_t i = 0; i < tiles; ++i) {
                        copy_tile(CB_FGP, i, i);
                    }
                    fp.pop_front(slot);
                    reconfig_data_format_srca(CB_FGP, CB_FGW);
                } else {
                    fill_tile_init();  // (Dest is not cleared between programs)
                    for (uint32_t i = 0; i < tiles; ++i) {
                        fill_tile(i, 0.0f);
                    }
                }
                matmul_block_init(CB_FGIN, CB_FGW, 0, w, mt, 1);
                for (uint32_t k = 2 * blk; k < 2 * blk + 2; ++k) {  // in tiles (0 .. mt - 1, k) adjacent
                    matmul_block(CB_FGIN, CB_FGW, k * mt, k * cnt + c0, 0, 0, w, mt, 1);
                }
                tile_regs_commit();
                tile_regs_wait();
                if (last) {
                    pack_reconfig_data_format(CB_FGOUT);
                    pack_reconfig_l1_acc(0);
                    fo.reserve_back(slot);
                    for (uint32_t i = 0; i < tiles; ++i) {
                        pack_tile(i, CB_FGOUT);
                    }
                    fo.push_back(slot);
                } else {
                    pack_reconfig_data_format(CB_FGP);
                    pack_reconfig_l1_acc(blk == 0 ? 0 : 1);  // block 0 written, the next added in place
                    fp.reserve_back(slot);
                    for (uint32_t i = 0; i < tiles; ++i) {
                        pack_tile(i, CB_FGP);
                    }
                    fp.push_back(slot);
                    if (blk + 2 < nblk) {  // back to the slot's start: the next block packs onto these
                        fp.wait_front(slot);
                        fp.pop_front(slot);
                    }
                }
                tile_regs_release();
            }
        }
        pack_reconfig_l1_acc(0);
        return;
    }
#endif
#ifdef DM_CONV
    // conv (GLM53_DFLASH2_CONV_FUSED) on a free core: its columns j0 .. j0 + nj - 1 of DFlash2's dynamic
    // convolution, each output tile (i, j) as dflash2's conv_rows_compute.cpp makes it (the same calls in the same
    // order on the same tiles: c_t = D_t E + base_t, xs = S x, bf16(c0 x + c1 xs)); x's tile (i, j) from CB_X (all of
    // x, k-major: j mt + i), the delta tiles from CB_CD (2 i + t)
    if (ncb == 0) {
        const uint32_t cj0 = get_arg_val<uint32_t>(6);
        const uint32_t nj = get_arg_val<uint32_t>(7);
        if (nj == 0) {
            return;
        }
        DeviceZoneScopedN("CV");
        compute_kernel_hw_startup(CB_CD, CB_CE, CB_CO);
        cb_wait_front(CB_CS, 1);
        cb_wait_front(CB_CE, nj);
        cb_wait_front(CB_CB, 2 * nj);
        cb_wait_front(CB_X, mt * kc);
        cb_wait_front(CB_CD, 2 * mt);
        for (uint32_t i = 0; i < mt; ++i) {
            for (uint32_t a = 0; a < nj; ++a) {
                const uint32_t at = (cj0 + a) * mt + i;
                const uint32_t dt = 2 * i;
                tile_regs_acquire();
                // the products (srcA = in1, srcB = in0)
                reconfig_data_format(CB_CE, CB_CD);
                matmul_init(CB_CD, CB_CE, 0);
                matmul_tiles(CB_CD, CB_CE, dt, a, 0);
                matmul_tiles(CB_CD, CB_CE, dt + 1, a, 1);
                reconfig_data_format(CB_X, CB_CS);
                matmul_init(CB_CS, CB_X, 0);
                matmul_tiles(CB_CS, CB_X, 0, at, 2);
                // the coefficients
                reconfig_data_format_srca(CB_CB);
                copy_init(CB_CB);
                copy_tile(CB_CB, 2 * a, 3);
                add_binary_tile_init();
                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 3, 0);
                copy_init(CB_CB);
                copy_tile(CB_CB, 2 * a + 1, 3);
                add_binary_tile_init();
                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(1, 3, 1);
                // c0 x + c1 xs
                reconfig_data_format_srca(CB_X);
                copy_init(CB_X);
                copy_tile(CB_X, at, 3);
                mul_binary_tile_init();
                mul_binary_tile(0, 3, 0);
                mul_binary_tile(1, 2, 1);
                add_binary_tile_init();
                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
                typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
                typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
                tile_regs_commit();
                cb_reserve_back(CB_CO, 1);
                tile_regs_wait();
                pack_tile(0, CB_CO);
                tile_regs_release();
                cb_push_back(CB_CO, 1);
            }
        }
        cb_pop_front(CB_CS, 1);
        cb_pop_front(CB_CE, nj);
        cb_pop_front(CB_CB, 2 * nj);
        cb_pop_front(CB_CD, 2 * mt);
        cb_pop_front(CB_X, mt * kc);
        return;
    }
#endif
    if (ncb == 0) {
        return;
    }
    const uint32_t first_out = ks > 1 ? CB_PART : CB_OUT;
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_W, first_out);  // matmul: in0 -> SrcB, in1 -> SrcA
    CircularBuffer x(CB_X), w(CB_W), recv(CB_RECV), out(CB_OUT);
    const uint32_t slot = cr * sb, per_sub = (kc + cr - 1) / cr;

    {
        DeviceZoneScopedN("MM");
        for (uint32_t c0 = 0; c0 < ncb; c0 += sb) {
            const uint32_t sbw = ncb - c0 < sb ? ncb - c0 : sb;
            tile_regs_acquire();
            fill_tile_init();  // Dest is not cleared between programs
            for (uint32_t i = 0; i < mt * sbw; ++i) {
                fill_tile(i, 0.0f);
            }
            if constexpr (mt == 1) {
                matmul_block_init(CB_X, CB_W, 0, sbw, 1, kc);
            } else {
                matmul_block_init(CB_X, CB_W, 0, sbw, mt, 1);
            }
            for (uint32_t g = 0; g < per_sub; ++g) {
                const uint32_t rows = kc - g * cr < cr ? kc - g * cr : cr;  // the slice's last chunk may be short
                if (c0 == 0) {
                    x.wait_front(mt * (g * cr + rows));  // x arrives in chunks
                }
                w.wait_front(slot);
                for (uint32_t r = 0; r < rows && !no_matmul; ++r) {  // x tile k times row k's sbw tiles of the chunk
                    if constexpr (mt == 1) {
                        matmul_block(CB_X, CB_W, g * cr + r, r * sb, 0, 0, sbw, 1, kc);
                    } else {  // x tiles (0 .. mt - 1, k), adjacent
                        matmul_block(CB_X, CB_W, (g * cr + r) * mt, r * sb, 0, 0, sbw, mt, 1);
                    }
                }
                w.pop_front(slot);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(first_out, mt * sbw);
            for (uint32_t i = 0; i < mt * sbw; ++i) {
                pack_tile(i, first_out);
            }
            cb_push_back(first_out, mt * sbw);
            tile_regs_release();
        }
        x.pop_front(mt * kc);
    }

    if (ks == 1 || owned == 0) {
        return;
    }
    DeviceZoneScopedN("ADD");
    reconfig_data_format_srca(CB_W, CB_RECV);
    pack_reconfig_data_format(CB_PART, CB_OUT);
    // the owned tiles' sums in Dest 0 .. owned - 1 (owned <= 4), each slice's partials unpacked to Dest owned .. as
    // fp32 and added in the order of K as the slices' rows come (decode_matmul_ctrl.cpp)
    tile_regs_acquire();
    for (uint32_t q = 0; q < ks; ++q) {
        recv.wait_front(owned);
        copy_init(CB_RECV);
        for (uint32_t i = 0; i < owned; ++i) {
            copy_tile(CB_RECV, i, q == 0 ? i : owned + i);
        }
        if (q > 0) {
            add_binary_tile_init();
            for (uint32_t i = 0; i < owned; ++i) {
                add_binary_tile(i, owned + i, i);
            }
        }
        recv.pop_front(owned);
    }
    tile_regs_commit();
    tile_regs_wait();
    out.reserve_back(owned);
    for (uint32_t i = 0; i < owned; ++i) {
        pack_tile(i, CB_OUT);
    }
    out.push_back(owned);
    tile_regs_release();
}
