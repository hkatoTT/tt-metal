// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two-stage row all-reduce (row_ar2.py), a link's sender core, RISCV_1: the reduce-scatter's shares (the input shards
// of this link's cores that the row's other chips own), in the writer's order, into cb_in (a page each, all of them
// at once), `depth` reads in flight (a transaction id each), each page pushed as it lands.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t shard = get_compile_time_arg_val(1);  // bytes a core (an input shard: bf16 or BFP8_b)
    constexpr uint32_t n = get_compile_time_arg_val(2);      // shares
    constexpr uint32_t depth = 8;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);  // the input's shard (the same address on every core)
    DeviceZoneScopedN("AR2-R");
    cb_reserve_back(cb_in, n);
    const uint32_t base = get_write_ptr(cb_in);
    for (uint32_t k = 0; k < n + depth; ++k) {
        if (k >= depth) {
            noc_async_read_barrier_with_trid((k - depth) % depth + 1);
            cb_push_back(cb_in, 1);
        }
        if (k < n) {
            const uint32_t w = get_arg_val<uint32_t>(1 + k);  // the share's core: NoC x | y << 8
            noc_async_read_set_trid(k % depth + 1);
            noc_async_read(get_noc_addr(w & 0xff, (w >> 8) & 0xff, x_addr), base + k * shard, shard);
        }
    }
    noc_async_read_set_trid(0);
}
