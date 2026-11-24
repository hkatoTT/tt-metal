// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per output tile: bf16(W @ S) + bf16(Q @ BO), each product summed in fp32 and rounded once, the sum in bf16 (the two
// bf16 terms HF forms, added as it adds them).  W and Q change with the row tile.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {
// the CBs (compile-time args 1 .. 7: mhc_stream_write's own, or mhc_stream_write_read's indices on its write cores)
constexpr uint32_t CB_S = get_compile_time_arg_val(1);
constexpr uint32_t CB_BO = get_compile_time_arg_val(2);
constexpr uint32_t CB_W = get_compile_time_arg_val(3);
constexpr uint32_t CB_Q = get_compile_time_arg_val(4);
constexpr uint32_t CB_T1 = get_compile_time_arg_val(5);
constexpr uint32_t CB_T2 = get_compile_time_arg_val(6);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(7);

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_W, CB_S, CB_T1);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(0);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    CircularBuffer s(CB_S), bo(CB_BO), w(CB_W), q(CB_Q), t1(CB_T1), t2(CB_T2), out(CB_OUT);
    uint32_t current = 0xFFFFFFFFu;
    for (uint32_t u = start; u < start + count; ++u) {
        const uint32_t rt = u / d_tiles;
        if (rt != current) {
            if (current != 0xFFFFFFFFu) {
                w.pop_front(1);
                q.pop_front(1);
            }
            current = rt;
            w.wait_front(1);
            q.wait_front(1);
        }
        s.wait_front(1);
        bo.wait_front(1);
        // both products in one Dest acquire (tiles 0 and 1), each packed to bf16 on its own
        t1.reserve_back(1);
        t2.reserve_back(1);
        tile_regs_acquire();
        matmul_init(CB_W, CB_S);
        matmul_tiles(CB_W, CB_S, 0, 0, 0);
        matmul_init(CB_Q, CB_BO);
        matmul_tiles(CB_Q, CB_BO, 0, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_T1);
        pack_tile(1, CB_T2);
        tile_regs_release();
        t1.push_back(1);
        t2.push_back(1);
        s.pop_front(1);
        bo.pop_front(1);

        t1.wait_front(1);
        t2.wait_front(1);
        out.reserve_back(1);
        tile_regs_acquire();
        add_init(CB_T1, CB_T2);
        add_tiles(CB_T1, CB_T2, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        out.push_back(1);
        t1.pop_front(1);
        t2.pop_front(1);
    }
    if (current != 0xFFFFFFFFu) {
        w.pop_front(1);
        q.pop_front(1);
    }
}
