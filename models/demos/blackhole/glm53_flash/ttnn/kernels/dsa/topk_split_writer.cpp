// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dsa_indexer.topk_split, the writer (topk_split_compute.cpp has the plan).
// ROLE 0 (a row's core): the row's 512 indices (CB_IDX's 2,048-byte page, untilized) to its output page.
// ROLE 1 (a helper): each sorted chunk c (CB_SEND's two fp32 tiles, 8 KB) written into the row core's CB_RECV slot
//   (c - OWN) mod S once the row's core has given this chunk's credit (SEM_CREDIT here reaching t + 1: the credits
//   come in this helper's chunk order), the write acknowledged, then the row core's SEM_ARRIVE0 + J incremented.
//   CB_RECV is allocated at the same address on every core of the program: this core's own (unused) CB_RECV start is
//   the row core's ring start.
// Compile-time args: ROLE, CB_IDX, CB_SEND, CB_RECV, OWN, H, S, SEM_CREDIT, SEM_ARRIVE0, then the output's accessor.
// Runtime args: ROLE 0: the output's address, the row; ROLE 1: the row core's NoC x, y, J, FIRST, COUNT.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(1);
    constexpr uint32_t CB_SEND = get_compile_time_arg_val(2);
    constexpr uint32_t CB_RECV = get_compile_time_arg_val(3);
    constexpr uint32_t OWN = get_compile_time_arg_val(4);
    constexpr uint32_t H = get_compile_time_arg_val(5);
    constexpr uint32_t S = get_compile_time_arg_val(6);
    constexpr uint32_t SEM_CREDIT = get_compile_time_arg_val(7);
    constexpr uint32_t SEM_ARRIVE0 = get_compile_time_arg_val(8);
    constexpr auto out_args = TensorAccessorArgs<9>();
    constexpr uint32_t ROW_BYTES = 512 * 4;
    constexpr uint32_t SLOT_BYTES = 2 * 32 * 32 * 4;  // a sorted chunk: the value and index tiles, fp32

    if constexpr (ROLE == 0) {
        const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
        const uint32_t row = get_arg_val<uint32_t>(1);
        cb_wait_front(CB_IDX, 1);
        noc_async_write(get_read_ptr(CB_IDX), out.get_noc_addr(row), ROW_BYTES);
        noc_async_write_barrier();
        cb_pop_front(CB_IDX, 1);
    } else {
        const uint32_t fx = get_arg_val<uint32_t>(0);
        const uint32_t fy = get_arg_val<uint32_t>(1);
        const uint32_t j = get_arg_val<uint32_t>(2);
        const uint32_t first = get_arg_val<uint32_t>(3);
        const uint32_t count = get_arg_val<uint32_t>(4);
        const uint32_t ring = get_write_ptr(CB_RECV);
        volatile tt_l1_ptr uint32_t* credit =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_CREDIT));
        const uint64_t arrive = get_noc_addr(fx, fy, get_semaphore(SEM_ARRIVE0 + j));
        for (uint32_t t = 0; t < count; ++t) {
            const uint32_t chunk = first + t * H;
            cb_wait_front(CB_SEND, 2);
            noc_semaphore_wait_min(credit, t + 1);
            const uint64_t slot = get_noc_addr(fx, fy, ring + ((chunk - OWN) % S) * SLOT_BYTES);
            noc_async_write(get_read_ptr(CB_SEND), slot, SLOT_BYTES);
            noc_async_write_barrier();
            noc_semaphore_inc(arrive, 1);
            cb_pop_front(CB_SEND, 2);
        }
        noc_semaphore_set(credit, 0);  // (every credit counted: ready for the next launch)
        noc_async_atomic_barrier();
    }
}
