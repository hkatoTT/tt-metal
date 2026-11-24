// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/debug/kernel_profiler.hpp"
#include "mp_common.hpp"

// The prefill MoE's gate | up by groups split along K (mp_common.hpp, mp_gu5_*.cpp), the compute: the group's rows
// in all (the writer's first mailbox value), then per item its rows' count; per row k (the item's W1 K block kept):
// this member's x block tilized, its partial sums of all 16 output tiles (16 K tiles each, fp32 Dest, HiFi4, 4 tiles a
// pass) packed to G5_PO; then row k - 1 finished: its 8 members' partials of this member's column tile (G5_RED,
// unpacked to Dest in fp32) summed in fp32 (member 0 .. 7), then g, u rounded to bf16 (as HF's bf16 matmuls), h =
// silu(min(g, limit)) x clamp(u, -limit, limit) x the pair's weight, rounded to bf16 once.
namespace {
constexpr uint32_t FMT_FP32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t FMT_BF16 = static_cast<uint32_t>(DataFormat::Float16_b);
}  // namespace

void kernel_main() {
    using namespace moe_prefill;
    constexpr uint32_t limit_bits = get_compile_time_arg_val(0);
    constexpr uint32_t neg_limit_bits = get_compile_time_arg_val(1);
    // timing probes (wrong results; tests only): 1 no tilize (the x block passed through untouched), 2 no finish (h a
    // zero tile), 4 no matmul (the partials zero tiles)
    constexpr uint32_t probe = get_compile_time_arg_val(2);
    // BALANCE: the group's segments' count from the writer (after the rows in all) in place of the runtime arg
    constexpr bool balance = get_compile_time_arg_val(3) != 0;
    // the progressive form: 2 (each expert's K block of the base, then of the correction, x's K block times both into
    // the same Dest), else 1
    constexpr uint32_t sets = get_compile_time_arg_val(4);
    constexpr uint32_t members = 8;
    uint32_t n_items = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(G5_XT, G5_W1, G5_PO);  // matmul: in0 -> SrcB, in1 -> SrcA
    uint32_t total = 0;  // the group's rows
    UNPACK(total = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    MATH(total = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    PACK(total = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    if constexpr (balance) {
        UNPACK(n_items = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(n_items = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(n_items = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    }

    auto partials = [&]() {
        if constexpr (probe & 1) {
            cb_wait_front(G5_XRM, KBLK);
            cb_pop_front(G5_XRM, KBLK);
            cb_reserve_back(G5_XT, KBLK);
            cb_push_back(G5_XT, KBLK);
        } else {
            DeviceZoneScopedN("G5_TILIZE");
            compute_kernel_lib::tilize<KBLK, G5_XRM, G5_XT>(1);
        }
        DeviceZoneScopedN("G5_MM");
        cb_wait_front(G5_XT, KBLK);
        reconfig_data_format(G5_W1, G5_XT);
        pack_reconfig_data_format(G5_PO);
        cb_reserve_back(G5_PO, G5_N);
        for (uint32_t j = 0; j < G5_N / G5_CHUNK; ++j) {
            tile_regs_acquire();
            fill_tile_init();  // Dest is not cleared between programs
            for (uint32_t d = 0; d < G5_CHUNK; ++d) {
                fill_tile(d, 0.0f);
            }
            if constexpr ((probe & 4) == 0) {
                matmul_block_init(G5_XT, G5_W1, 0, G5_CHUNK, 1, KBLK);
                for (uint32_t k = 0; k < sets * KBLK; ++k) {  // x tile k times output tiles 4 j .. 4 j + 3 of K row k
                    matmul_block(G5_XT, G5_W1, k % KBLK, k * G5_N + j * G5_CHUNK, 0, 0, G5_CHUNK, 1, KBLK);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < G5_CHUNK; ++d) {
                pack_tile<true>(d, G5_PO, j * G5_CHUNK + d);
            }
            tile_regs_release();
        }
        cb_push_back(G5_PO, G5_N);
        cb_pop_front(G5_XT, KBLK);
    };
    auto finish = [&]() {
        {
            DeviceZoneScopedN("G5_WAIT_RED");  // the 8 members' partials of the row
            cb_wait_front(G5_RED, G5_N);
            cb_wait_front(G5_WCOL, 1);
        }
        DeviceZoneScopedN("G5_FINISH");
        if constexpr (probe & 2) {
            pack_reconfig_data_format(G5_H);
            tile_regs_acquire();
            fill_tile_init();
            fill_tile(0, 0.0f);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(G5_H, 1);
            pack_tile(0, G5_H);
            cb_push_back(G5_H, 1);
            tile_regs_release();
            cb_pop_front(G5_RED, G5_N);
            cb_pop_front(G5_WCOL, 1);
            pack_reconfig_data_format(G5_PO);
            return;
        }
        reconfig_data_format_srca(G5_W1, G5_RED);
        pack_reconfig_data_format(G5_H);
        tile_regs_acquire();
        for (uint32_t g = 0; g < 2; ++g) {  // Dest g: the sum of the members' partials of gate (0), up (1)
            copy_tile_to_dst_init_short(G5_RED);
            copy_tile(G5_RED, g, g);
            for (uint32_t s = 1; s < members; ++s) {
                copy_tile_to_dst_init_short(G5_RED);
                copy_tile(G5_RED, 2 * s + g, 2);
                add_binary_tile_init();
                add_binary_tile(g, 2, g);
            }
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
        reconfig_data_format_srca(G5_RED, G5_WCOL);
        unary_bcast_init<BroadcastType::COL>(G5_WCOL);
        unary_bcast<BroadcastType::COL>(G5_WCOL, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        typecast_tile_init<FMT_FP32, FMT_BF16>();
        typecast_tile<FMT_FP32, FMT_BF16>(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(G5_H, 1);
        pack_tile(0, G5_H);
        cb_push_back(G5_H, 1);
        tile_regs_release();
        cb_pop_front(G5_RED, G5_N);
        cb_pop_front(G5_WCOL, 1);
    };

    uint32_t row = 0;  // the group's rows so far
    for (uint32_t i = 0; i < n_items; ++i) {
        uint32_t rows = 0;
        UNPACK(rows = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(rows = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(rows = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        if (rows == 0) {
            continue;
        }
        {
            DeviceZoneScopedN("G5_WAIT_W1");
            cb_wait_front(G5_W1, sets * G5_N * KBLK);
        }
        for (uint32_t r = 0; r < rows; ++r, ++row) {
            partials();
            if (row > 0) {
                finish();  // row - 1
            }
        }
        cb_pop_front(G5_W1, sets * G5_N * KBLK);
    }
    if (total > 0) {
        finish();  // the last row
    }
}
