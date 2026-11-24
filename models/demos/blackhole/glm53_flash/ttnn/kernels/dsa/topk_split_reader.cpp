// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dsa_indexer.topk_split, the reader (topk_split_compute.cpp has the plan).  A chunk is its row's 512 bf16 scores
// (1,024 bytes) into a 2,048-byte CB_IN page, as topk_large_indices' reader puts it.
// ROLE 0 (a row's core): chunks 0 .. OWN - 1 from DRAM; then for each chunk c >= OWN, in order: CB_RECV's next two
//   pages reserved (the slot (c - OWN) mod S free: compute popped chunk c - S), a credit to chunk c's helper
//   (j = (c - OWN) mod H: its SEM_CREDIT, counted in its chunk order), its arrival awaited (SEM_ARRIVE0 + j here
//   reaching its t + 1, t = (c - OWN) / H), the pages pushed.
// ROLE 1 (a helper): its chunks FIRST, FIRST + H, .. from DRAM.
// Compile-time args: ROLE, CB_IN, CB_RECV, OWN, H, CHUNKS, SEM_CREDIT, SEM_ARRIVE0, then the input's accessor.
// Runtime args: the input's address, the row; ROLE 0: the H helpers' NoC x then y (4 each); ROLE 1: FIRST, COUNT.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t CB_IN = get_compile_time_arg_val(1);
    constexpr uint32_t CB_RECV = get_compile_time_arg_val(2);
    constexpr uint32_t OWN = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(5);
    constexpr uint32_t SEM_CREDIT = get_compile_time_arg_val(6);
    constexpr uint32_t SEM_ARRIVE0 = get_compile_time_arg_val(7);
    constexpr auto in_args = TensorAccessorArgs<8>();
    constexpr uint32_t CHUNK_BYTES = 512 * 2;
    static_assert(H >= 1 && H <= 4);

    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0));
    const uint32_t row = get_arg_val<uint32_t>(1);

    auto read_chunk = [&](uint32_t chunk) {
        cb_reserve_back(CB_IN, 1);
        noc_async_read(in.get_noc_addr(row, chunk * CHUNK_BYTES), get_write_ptr(CB_IN), CHUNK_BYTES);
        noc_async_read_barrier();
        cb_push_back(CB_IN, 1);
    };

    if constexpr (ROLE == 0) {
        uint32_t hx[4], hy[4];
        volatile tt_l1_ptr uint32_t* arrive[4];
        for (uint32_t j = 0; j < H; ++j) {
            hx[j] = get_arg_val<uint32_t>(2 + j);
            hy[j] = get_arg_val<uint32_t>(6 + j);
            arrive[j] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ARRIVE0 + j));
        }
        const uint32_t credit = get_semaphore(SEM_CREDIT);
        for (uint32_t chunk = 0; chunk < OWN; ++chunk) {
            read_chunk(chunk);
        }
        uint32_t j = 0, t = 0;  // chunk c's helper and its place in that helper's chunks
        for (uint32_t chunk = OWN; chunk < CHUNKS; ++chunk) {
            cb_reserve_back(CB_RECV, 2);
            noc_semaphore_inc(get_noc_addr(hx[j], hy[j], credit), 1);
            noc_semaphore_wait_min(arrive[j], t + 1);
            cb_push_back(CB_RECV, 2);
            if (++j == H) {
                j = 0;
                ++t;
            }
        }
        for (uint32_t k = 0; k < H; ++k) {
            noc_semaphore_set(arrive[k], 0);  // (every arrival counted: ready for the next launch)
        }
        noc_async_atomic_barrier();
    } else {
        const uint32_t first = get_arg_val<uint32_t>(2);
        const uint32_t count = get_arg_val<uint32_t>(3);
        for (uint32_t t = 0; t < count; ++t) {
            read_chunk(first + t * H);
        }
    }
}
