// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per tile row of logits (fp32, unpacked to Dest): the scores sigmoid(logits) (ttnn.sigmoid's kernel) and the choice
// keys scores + bias (the bias's row 0 broadcast down the rows), both fp32, four tiles a pass (Dest in full sync).  When this core's rows of
// the tile row are all in one half, the reader put only that half in faces 0-1, and the sigmoid takes only those.
// Packed: the reader's tiles of this core's rows only, the bias already in each row (added value by value).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t CB_LOG = 0;
constexpr uint32_t CB_BIAS = 1;
constexpr uint32_t CB_SC = 16;
constexpr uint32_t CB_CH = 17;
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_LOG, CB_SC);
    constexpr uint32_t et = get_compile_time_arg_val(0);
    constexpr bool pack = get_compile_time_arg_val(1) != 0;  // the reader's packed rows (rpt a tile): np tiles
    constexpr uint32_t rpt = get_compile_time_arg_val(2);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    CircularBuffer log(CB_LOG), bias(CB_BIAS), sc(CB_SC), ch(CB_CH);
    if constexpr (pack) {  // the same per value: the bias tile holds row 0 in each packed row (no broadcast)
        bias.wait_front(1);
        const uint32_t np = (count + rpt - 1) / rpt;
        if (np == 0) {
            return;
        }
        log.wait_front(np);
        sc.reserve_back(np);
        ch.reserve_back(np);
        for (uint32_t t = 0; t < np; t += 4) {
            const uint32_t n = np - t < 4 ? np - t : 4;  // Dest 0-3 the scores, 4-7 the keys
            tile_regs_acquire();
            copy_init(CB_LOG);
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(CB_LOG, t + i, i);
            }
            sigmoid_tile_init();
            for (uint32_t i = 0; i < n; ++i) {
                sigmoid_tile(i);
            }
            copy_init(CB_BIAS);
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(CB_BIAS, 0, 4 + i);
            }
            add_binary_tile_init();
            for (uint32_t i = 0; i < n; ++i) {
                add_binary_tile(i, 4 + i, 4 + i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile<true>(i, CB_SC, t + i);
                pack_tile<true>(4 + i, CB_CH, t + i);
            }
            tile_regs_release();
        }
        sc.push_back(np);
        ch.push_back(np);
        log.pop_front(np);
        return;
    }
    bias.wait_front(et);
    for (uint32_t tr = start / 32; tr <= (start + count - 1) / 32; ++tr) {
        const uint32_t r0 = tr * 32 > start ? tr * 32 : start;
        const uint32_t r1 = (tr + 1) * 32 < start + count ? (tr + 1) * 32 : start + count;
        const bool half = (r0 % 32) / 16 == ((r1 - 1) % 32) / 16;
        log.wait_front(et);
        sc.reserve_back(et);
        ch.reserve_back(et);
        for (uint32_t t = 0; t < et; t += 4) {
            const uint32_t n = et - t < 4 ? et - t : 4;  // Dest 0-3 the scores, 4-7 the keys
            tile_regs_acquire();
            copy_init(CB_LOG);
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(CB_LOG, t + i, i);
            }
            sigmoid_tile_init();
            for (uint32_t i = 0; i < n; ++i) {
                if (half) {
                    sigmoid_tile<VectorMode::R>(i);
                } else {
                    sigmoid_tile(i);
                }
            }
            unary_bcast_init<BroadcastType::ROW>(CB_BIAS);
            for (uint32_t i = 0; i < n; ++i) {
                unary_bcast<BroadcastType::ROW>(CB_BIAS, t + i, 4 + i);
            }
            add_binary_tile_init();
            for (uint32_t i = 0; i < n; ++i) {
                add_binary_tile(i, 4 + i, 4 + i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile<true>(i, CB_SC, t + i);
                pack_tile<true>(4 + i, CB_CH, t + i);
            }
            tile_regs_release();
        }
        sc.push_back(et);
        ch.push_back(et);
        log.pop_front(et);
    }
}
