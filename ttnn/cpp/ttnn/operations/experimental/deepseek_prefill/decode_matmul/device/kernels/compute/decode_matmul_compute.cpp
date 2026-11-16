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
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

namespace {
constexpr uint32_t CB_X = 0;
constexpr uint32_t CB_W = 1;
constexpr uint32_t CB_RECV = 2;
constexpr uint32_t CB_PART = 16;
constexpr uint32_t CB_OUT = 17;
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
