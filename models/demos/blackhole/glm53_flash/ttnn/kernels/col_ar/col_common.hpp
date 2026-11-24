// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The MoE's column collectives (col_collectives.py), shared by a worker's reader (NCRISC) and writer (BRISC).

// scatter: rows d t .. d t + t - 1 of the untilized input (`rm`, rows of `row_bytes`) for chips d0 .. d1 - 1 into this
// chip's receive slot (d = my; only by the reader, which pushes the slots to the compute: the writer passes
// own_slot 0 and skips it) or the sender core's stage (slot d workers + worker)
inline void col_stage_rows(
    uint32_t rm,
    uint32_t slice,
    uint32_t d0,
    uint32_t d1,
    uint32_t my,
    uint32_t own_slot,
    uint32_t stage,
    uint32_t workers,
    uint32_t worker,
    uint32_t sender_x,
    uint32_t sender_y) {
    for (uint32_t d = d0; d < d1; ++d) {
        if (d == my) {
            if (own_slot != 0) {
                noc_async_write(rm + d * slice, get_noc_addr(own_slot), slice);
            }
        } else {
            noc_async_write(
                rm + d * slice, get_noc_addr(sender_x, sender_y, stage + (d * workers + worker) * slice), slice);
        }
    }
}

// scatter: output tiles j0 .. j1 - 1: the summed rows (`sum`: `rows` rows of `row_bytes`, row-major) into this core's
// staging tiles (a circular buffer, zeroed earlier in the call by the same RISC), then the tiles to the output
template <typename Accessor>
inline void col_put_rows(
    const Accessor& s_out,
    uint32_t sum,
    uint32_t rows,
    uint32_t row_bytes,
    uint32_t staging,
    uint32_t tile0,
    uint32_t j0,
    uint32_t j1) {
    for (uint32_t j = j0; j < j1; ++j) {
        for (uint32_t i = 0; i < rows; ++i) {
            const uint32_t src = sum + i * row_bytes + j * 64;
            noc_async_write(src, get_noc_addr(staging + j * 2048 + i * 32), 32);
            noc_async_write(src + 32, get_noc_addr(staging + j * 2048 + 512 + i * 32), 32);
        }
    }
    noc_async_write_barrier();
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_write(staging + j * 2048, s_out.get_noc_addr(tile0 + j), 2048);
    }
    noc_async_write_barrier();
}

// the padded scatter: output tiles j0 .. j1 - 1: the summed slice (`sum`: tile j's face 0 then face 1 first `rows`
// rows, 32 rows bytes each, at j 64 rows) into faces 0 and 1 of this core's staging tiles (zeroed earlier in the call
// by the same RISC), then the tiles to the output
template <typename Accessor>
inline void col_put_faces(
    const Accessor& s_out, uint32_t sum, uint32_t rows, uint32_t staging, uint32_t tile0, uint32_t j0, uint32_t j1) {
    const uint32_t half = rows * 32;
    for (uint32_t j = j0; j < j1; ++j) {
        const uint32_t src = sum + j * 2 * half;
        noc_async_write(src, get_noc_addr(staging + j * 2048), half);
        noc_async_write(src + half, get_noc_addr(staging + j * 2048 + 512), half);
    }
    noc_async_write_barrier();
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_write(staging + j * 2048, s_out.get_noc_addr(tile0 + j), 2048);
    }
    noc_async_write_barrier();
}

// the output tiles j0 .. j1 - 1 from a CB of tiles
template <typename Accessor>
inline void col_put_tiles(const Accessor& s_out, uint32_t tiles_l1, uint32_t tile0, uint32_t j0, uint32_t j1) {
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_write(tiles_l1 + j * 2048, s_out.get_noc_addr(tile0 + j), 2048);
    }
    noc_async_write_barrier();
}
