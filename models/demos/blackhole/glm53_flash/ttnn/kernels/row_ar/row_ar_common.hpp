// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Row all-reduce (row_all_reduce.py): output tiles j0 .. j1 - 1 of a worker's slice.  The summed rows (`src`, the
// compute's output: per tile, face 0's first `rows` rows, then face 1's, `seg` bytes each) into this core's staging
// tiles (this program's, zeroed earlier in the call by the same RISC), then the tiles to the output.
template <uint32_t seg, uint32_t rows, typename Accessor>
inline void row_ar_put_tiles(
    const Accessor& s_out, uint32_t src, uint32_t staging, uint32_t tile0, uint32_t j0, uint32_t j1) {
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_write(src + j * 2 * seg, get_noc_addr(staging + j * 2048), rows * 32);
        noc_async_write(src + j * 2 * seg + seg, get_noc_addr(staging + j * 2048 + 512), rows * 32);
    }
    noc_async_write_barrier();
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_write(staging + j * 2048, s_out.get_noc_addr(tile0 + j), 2048);
    }
    noc_async_write_barrier();
}
