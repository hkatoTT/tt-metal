// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused form (GLM53_KDA_D2=1): the KDA verify step's writer: this core's decode_prepare item's outputs
// (writer_decode_prepare.hpp, args from 0), then, once they have landed, one increment of semaphore D2_SEM on each core
// whose recurrence units read this item's head (runtime args from D2_SIG_RT: the count, then x, y each), then this
// core's chunked-form units' o and state (d2_chunk_writer.hpp).
#if defined(D2_DM_OS)  // (the data mover at -Os: a smaller binary; the same values)
#pragma GCC optimize("Os")
#endif
#include "ttnn/cpp/ttnn/operations/experimental/kda/decode_prepare/device/kernels/dataflow/writer_decode_prepare.hpp"
#include "d2_chunk_writer.hpp"

#ifndef D2_SEM
#define D2_SEM 2
#endif

void kernel_main() {
#if D2_PREP
    kda_prepare_writer::run<0, 0>();
#endif
    {
        DeviceZoneScopedN("D2-SIG");
        noc_async_write_barrier();  // (the item's q / k / v / decay / beta rows landed)
        const uint32_t n = get_arg_val<uint32_t>(D2_SIG_RT);
        const uint32_t sem = get_semaphore(D2_SEM);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t x = get_arg_val<uint32_t>(D2_SIG_RT + 1 + 2 * i);
            const uint32_t y = get_arg_val<uint32_t>(D2_SIG_RT + 2 + 2 * i);
            noc_semaphore_inc(get_noc_addr(x, y, sem), 1);
        }
        // and this core's own: its prepare work done (its compute's last outputs written out, its CBs free: the chunk
        // CBs that share their memory (D2_ALIAS) may be written from here on)
        noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], sem), 1);
    }
    d2_chunk_writer();
    noc_async_atomic_barrier();  // (the increments' responses before the kernel ends)
}
