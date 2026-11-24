// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mhc_stream_write_read's prefill groups with the write spread over the cores (MHC_WR_MERGED): a helper core's compute.
// Per unit (a slice's column: 4 row tiles), the new streams tiles bf16(W_rt @ S_old) + bf16(Q_rt @ BO), as
// mhc_stream_write.cpp and mhc_read_slice.cpp's W phase compute them (each product summed in fp32, packed to bf16, the
// two added; two row tiles a Dest acquire), packed in the row tiles' order to CB_OUT for the writer.  W and Q of the
// unit's group (4 tiles each) change with the group.
//   CT: CB_SOLD, CB_BO, CB_W, CB_Q, CB_T1, CB_T2, CB_OUT, k_tiles, k_own, slices.  RT: the first unit, the units.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t CB_SOLD = get_compile_time_arg_val(0);
    constexpr uint32_t CB_BOW = get_compile_time_arg_val(1);
    constexpr uint32_t CB_W = get_compile_time_arg_val(2);
    constexpr uint32_t CB_Q = get_compile_time_arg_val(3);
    constexpr uint32_t CB_T1 = get_compile_time_arg_val(4);
    constexpr uint32_t CB_T2 = get_compile_time_arg_val(5);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(6);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t k_own = get_compile_time_arg_val(8);
    constexpr uint32_t slices = get_compile_time_arg_val(9);
    constexpr uint32_t helped = k_tiles - k_own;
    constexpr uint32_t row_tiles = 4;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(CB_W, CB_SOLD, CB_T1);  // (as mhc_stream_write.cpp)
    CircularBuffer sold(CB_SOLD), bow(CB_BOW), wt(CB_W), qt(CB_Q), t1(CB_T1), t2(CB_T2), out(CB_OUT);
    uint32_t wq_g = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t g = ((start + i) / helped) / slices;
        if (g != wq_g) {
            if (wq_g != 0xFFFFFFFFu) {
                wt.pop_front(row_tiles);
                qt.pop_front(row_tiles);
            }
            wq_g = g;
            wt.wait_front(row_tiles);
            qt.wait_front(row_tiles);
        }
        sold.wait_front(row_tiles);
        bow.wait_front(1);
        out.reserve_back(row_tiles);
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
            pack_tile(0, CB_OUT);
            pack_tile(1, CB_OUT);
            tile_regs_release();
            t1.pop_front(2);
            t2.pop_front(2);
        }
        out.push_back(row_tiles);
        sold.pop_front(row_tiles);
        bow.pop_front(1);
    }
    if (wq_g != 0xFFFFFFFFu) {
        wt.pop_front(row_tiles);
        qt.pop_front(row_tiles);
    }
}
