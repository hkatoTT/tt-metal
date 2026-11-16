// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// col_row_reduce_scatter's add stage, a core's compute: out = a + b, the FPU's
// ELWADD with a 16-bit Dst (fp32_dest_acc off), as ttnn.add of two bf16 tile tensors (the same bits), chunks of
// up to CHUNK tiles (the reader pushes CHUNK slots a chunk; only the first k hold tiles).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t batches = get_compile_time_arg_val(3);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(4);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    add_init(cb_a, cb_b);
    for (uint32_t b = 0; b < batches; ++b) {
        for (uint32_t i = start; i < end; i += CHUNK) {
            const uint32_t k = end - i < CHUNK ? end - i : CHUNK;
            cb_wait_front(cb_a, CHUNK);
            cb_wait_front(cb_b, CHUNK);
            cb_reserve_back(cb_out, CHUNK);
            tile_regs_acquire();
            for (uint32_t j = 0; j < k; ++j) {
                add_tiles(cb_a, cb_b, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < k; ++j) {
                pack_tile<true>(j, cb_out, j);
            }
            tile_regs_release();
            cb_push_back(cb_out, CHUNK);
            cb_pop_front(cb_a, CHUNK);
            cb_pop_front(cb_b, CHUNK);
        }
    }
}
