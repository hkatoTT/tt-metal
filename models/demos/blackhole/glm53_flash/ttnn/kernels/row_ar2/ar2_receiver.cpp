// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two-stage row all-reduce (row_ar2.py), an output core, RISCV_1.
//   owner (this chip makes this core's sums): its own share into its slot of the landing buffer; the other 3 chips'
//     shares (the global semaphore at 3); the compute (all_reduce_async's reduction.cpp) adds the 4 slots into the
//     output; the sums to its sender's stage, then its bit of the sender's semaphore (the all-gather).
//   not the owner: the owner chip's sums land in the output directly (the global semaphore at 1).
// The global semaphore back to 0 once seen (a parity's next call is two calls on, as all_reduce_async's).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_land = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t sem_ag = get_compile_time_arg_val(3);
    constexpr uint32_t shard = get_compile_time_arg_val(4);  // bytes a core
    constexpr uint32_t tiles = get_compile_time_arg_val(5);  // tiles a core
    constexpr uint32_t ring = get_compile_time_arg_val(6);   // the row's chips
    constexpr uint32_t shard_in = get_compile_time_arg_val(8);  // bytes a core's input share (BFP8_b: GLM53_AR_BFP8)

#ifdef AR2_DRY  // (a test's: the sender sends nothing)
    return;
#endif
    size_t a = 0;
    const uint32_t owner = get_arg_val<uint32_t>(a++);
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    if (owner == 0) {
        DeviceZoneScopedN("AR2-RX-AG");
        noc_semaphore_wait(sem, 1);
        noc_semaphore_set(sem, 0);
        return;
    }
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);    // this core's input shard
    const uint32_t slot_off = get_arg_val<uint32_t>(a++);  // this chip's slot in the landing shard
    const uint32_t sender_x = get_arg_val<uint32_t>(a++);
    const uint32_t sender_y = get_arg_val<uint32_t>(a++);
    const uint32_t m = get_arg_val<uint32_t>(a++);  // this core's place in the sender's stage
    DeviceZoneScopedN("AR2-RX-OWN");
    noc_async_read(get_noc_addr(x_addr), get_write_ptr(cb_land) + slot_off, shard_in);
    noc_async_read_barrier();
    {
        DeviceZoneScopedN("AR2-RX-RS");
        noc_semaphore_wait(sem, ring - 1);
    }
    noc_semaphore_set(sem, 0);
    cb_push_back(cb_land, ring * tiles);
    {
        DeviceZoneScopedN("AR2-RX-SUM");
        cb_wait_front(cb_out, tiles);
    }
    const uint32_t stage = get_write_ptr(cb_stage) + m * shard;
    noc_async_write(get_read_ptr(cb_out), get_noc_addr(sender_x, sender_y, stage), shard);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(sender_x, sender_y, get_semaphore(sem_ag)), 1u << m);
    noc_async_atomic_barrier();
}
