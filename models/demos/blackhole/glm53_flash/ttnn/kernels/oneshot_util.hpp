// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the one-shot fabric ops (row_all_reduce.py, col_collectives.py).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// zeros into this core's L1 [addr, addr + bytes) (bytes a multiple of MEM_ZEROS_SIZE), complete on return
inline void oneshot_zero(uint32_t addr, uint32_t bytes) {
    for (uint32_t a = addr; a < addr + bytes; a += MEM_ZEROS_SIZE) {
        noc_async_write(MEM_ZEROS_BASE, get_noc_addr(a), MEM_ZEROS_SIZE);
    }
    noc_async_write_barrier();
}
