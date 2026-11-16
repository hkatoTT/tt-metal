// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// The work split: K in k_split slices of K / 32 / k_split tiles, slice s on the rectangle of rect_w x rect_h cores at
// ((s % (grid_x / rect_w)) rect_w, (s / (grid_x / rect_w)) rect_h); N in n_blocks blocks of up to n_per_core tiles
// (decode_matmul_blocks.hpp), block i on core floor(i P / n_blocks) of each rectangle of P cores (row-major in it: the
// busy cores spread over the rectangle).  The other cores only receive x.
struct DecodeMatmulPlan {
    uint32_t k_split;
    uint32_t n_per_core;
    uint32_t n_blocks;
    uint32_t rect_w;
    uint32_t rect_h;
};

// The split for K / 32 = k_tiles and N / 32 = n_tiles on a grid_x x grid_y compute grid with `banks` DRAM banks
// (k_split 0: K whole on each core and blocks of 2 columns from N / 32 = 64 on, else 1; K in 4 slices when that leaves
// fewer than 24 blocks: the measured best of the 9 decode shapes; else that many K slices, the fewest columns a
// block that fit).  The N blocks are decode_matmul_blocks(n_tiles, banks, n_per_core, decode_matmul_strided()).
DecodeMatmulPlan decode_matmul_plan(
    uint32_t k_tiles,
    uint32_t n_tiles,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t weight_tile_bytes,
    uint32_t k_split,
    uint32_t banks);
// whether the N blocks are strided by the DRAM banks (DECODE_MATMUL_STRIDED=0, a test switch: consecutive columns)
bool decode_matmul_strided();

struct DecodeMatmulParams {
    tt::tt_metal::DataType output_dtype;
    uint32_t k_split;
    uint32_t n_per_core;
    uint32_t rect_w;
    uint32_t rect_h;
    uint32_t strided;  // the N blocks strided by the DRAM banks (decode_matmul_strided())
    uint32_t debug;    // DECODE_MATMUL_DEBUG, a test switch (wrong results): 1 no weight reads, 2 no matmuls
    // the weight stream's chunks in flight and tiles a chunk (DECODE_MATMUL_AHEAD / _CHUNK, test switches; 0: the
    // defaults, 2 and 8)
    uint32_t ahead;
    uint32_t chunk;
    // the weight stream's NoCs: 2 (the default) alternates its requests between NoC 0 and NoC 1 (KDA x_proj 70.6 ->
    // 64.4, DSA o_proj 47.6 -> 41.1 us), NoC 1 only once x's multicast is done (the control kernel's sem_noc1
    // after its last x chunk, single-sender shapes: taken from the start it hung once in ~6,000 decode steps, the x
    // sender's NoC 1 multicast stuck while other cores read DRAM on NoC 1; so 16,000 steps clean);
    // 1, NoC 0 only (DECODE_MATMUL_NOCS=1)
    uint32_t nocs;
    // where the output goes: DRAM interleaved (the default), or e.g. L1 width-sharded for the all-reduce that takes it
    // (the writer writes its tiles through a TensorAccessor either way)
    tt::tt_metal::MemoryConfig output_memory_config;
};

// x: [1, 1, M, K] BFLOAT16 TILE (M <= 64; above 32, two tile rows: K whole on each core).  w: [1, 1, K, N] BFLOAT8_B / BFLOAT16 TILE.  Both DRAM interleaved.
struct DecodeMatmulTensorArgs {
    const Tensor& x;
    const Tensor& w;
};

}  // namespace ttnn::experimental::prim
