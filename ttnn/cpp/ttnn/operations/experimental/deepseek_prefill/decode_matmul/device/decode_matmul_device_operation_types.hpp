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
    // Page layouts (the same tiles, other places): x_heads H, x as [1, H, M, K / H] (heads first: x's tile (r, k) at
    // page (k / W) Mt W + r W + k mod W, W = K / H / 32, Mt its tile rows), as the heads' [1, H, M, d] tiles lie;
    // out_heads H, the output as [1, H, M, N / H] likewise; split_n S (a multiple of 32), the output's columns from S
    // on into split_out ([1, 1, M, N - S], its own memory config), the output itself [1, 1, M, S].  0: none.  With one
    // tile row of x (M <= 32) the heads' pages are x's / the output's own.
    uint32_t x_heads = 0;
    uint32_t out_heads = 0;
    uint32_t split_n = 0;
    // fg (KDA's fg made in x_proj's op, GLM53_KDA_FG_FUSED): fg_k K tiles of fg_w ([1, 1, 32 fg_k, N2]): the output's
    // last fg_k column tiles times fg_w into fg_out ([1, 1, M, N2], allocated by the caller), as ttnn.linear's default
    // 1D config makes it (K blocks of 2 tiles, packer L1 accumulation in fp32, the last block on the reloaded partials,
    // HiFi4, fp32 Dest), on the rectangle's cores left without an N block; those last columns each in a block of its
    // own.  0: none.  K whole on each core (k_split 1) only.
    uint32_t fg_k = 0;
    // conv (GLM53_DFLASH2_CONV_FUSED): 1 + side: also DFlash2's dynamic convolution of that side (dflash2_ops.conv) of
    // x itself with the output as its delta (output column 16 side + 8 t + q: tap t of x's columns 16 q .. 16 q + 15)
    // into conv_out ([1, 1, M, K]), the conv op's arithmetic tile for tile, on the rectangle's cores left without an N
    // block; the side's 16 delta columns each in a block of its own.  0: none.
    uint32_t conv = 0;
    // landing (GLM53_W1D, ttnn/w1_dense.py: the dense weights in L1 ahead of the op): empty, none; else [A, B, tag,
    // then (a, b) for each core in decode_matmul_streams' order]: the core's weight chunks 0 .. a - 1 from its L1 at A
    // (64 B of seal, then the chunks in issue order, a slot each: cr x sb tiles as the ring holds them) and chunks a ..
    // a + b - 1 from B (likewise, chunk a first), each region only when its seal is (tag << 16) | (its last chunk + 1),
    // as the riders that filled it wrote it (ttnn/w1_dense.py), else from DRAM as always; the seals cleared after.  The
    // same bytes into the same slots either way: the values the same.  Any form (plain, fg, conv)
    std::vector<uint32_t> landing;
};

// x: [1, 1, M, K] BFLOAT16 TILE (M <= 64; above 32, two tile rows: K whole on each core), or [1, H, M, K / H] with
// x_heads; DRAM interleaved, or any memory a TensorAccessor reads (e.g. the L1 width-sharded output of a norm).  w: [1,
// 1, K, N] BFLOAT8_B / BFLOAT16 TILE, DRAM interleaved.  split_out: with split_n, the output's columns from split_n on
// (allocated by the caller).
struct DecodeMatmulTensorArgs {
    const Tensor& x;
    const Tensor& w;
    std::optional<Tensor> split_out;
    std::optional<Tensor> fg_w;    // with fg_k: [1, 1, 32 fg_k, N2] BFLOAT8_B / BFLOAT16 TILE, DRAM interleaved
    std::optional<Tensor> fg_out;  // with fg_k: [1, 1, M, N2] BFLOAT16 TILE, DRAM interleaved
    // with conv: the conv op's operands (dflash2_ops.conv_base / conv_constants) and its output, DRAM interleaved:
    // base [1, 2, 32, K] FLOAT32, E [1, 1, 512, 32] and S [1, 1, 32, 32] BFLOAT16, conv_out [1, 1, M, K] BFLOAT16
    std::optional<Tensor> conv_base;
    std::optional<Tensor> conv_e;
    std::optional<Tensor> conv_s;
    std::optional<Tensor> conv_out;
};

}  // namespace ttnn::experimental::prim
