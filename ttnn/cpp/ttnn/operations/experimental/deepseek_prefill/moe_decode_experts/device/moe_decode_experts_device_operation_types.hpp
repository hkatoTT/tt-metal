// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MoeDecodeExpertsParams {
    uint32_t local;  // experts (slots) on the chip: a mesh column's, and the shared expert's
    uint32_t inter;  // an expert's intermediate width I on the chip: 256 (its columns 256 r .. on mesh row r)
    float limit;     // the SwiGLU clamp
    uint32_t row_tokens;  // 8 / 16: y by mesh row, each row_tokens rows at the top of a tile row of its own (0: as x)
    // The in-op gather (h1_*): the column's x and routing gathered by the op itself over the fabric.  The
    // L1 address of two global semaphores (every core of the grid): the x arrivals (the grid columns' top gate | up
    // cores count them) and the routing's (the builder cores); 0: off (x and weights are already the column's)
    uint32_t h1_sem_x = 0;
    uint32_t h1_sem_r = 0;
    // the column's way back in the op (with the in-op gather, 32 / 64 tokens a mesh row): the L1 address
    // of two global semaphores (every core of the grid): the reducer cores' fabric landings (low 16 bits from mesh row
    // r - 1, high from r + 1) and a mailbox word (the reducers' own partials come; the down cores' landing address;
    // the return senders' go).  The op then returns this mesh row's t tokens summed over the column [1, 1, t, D], the
    // bits of ttnn.reduce_scatter over the column of the partial sums it would return otherwise; 0: off
    uint32_t rs_sem = 0;
    uint32_t rs_mail = 0;
    // ... and where that output goes (e.g. the row all-reduce's L1 shards, for it to take as it is); none: DRAM
    std::optional<tt::tt_metal::MemoryConfig> rs_memory_config = std::nullopt;
};

// x: [.., N, D] BFLOAT16 TILE (N = 32: every token of the mesh).  weights: [.., N, 32] FLOAT32 TILE
// (moe_decode_route's).  w1: [.., P L (D / 32) 32, 64] the gates | ups, core p's (gate column tile p, up column tile p)
// of expert j at tile rows ((p L + j) D / 32 + k).  w2: [.., Q L (I / 32) 32, C 32] the downs, core q's C column tiles
// of expert j at tile rows ((q L + j) I / 32 + k).  P = I / 32 = 64 gate | up cores, Q = 32 down cores, C = D / 32 / Q.
// The in-op gather (h1_sem_x != 0): x and weights are the column's landing buffers [8 t, D] / [8 t, 32] (persistent,
// the same address on every chip; weights already holds this mesh row's rows r t .. r t + t - 1, written there by the
// routing), and x_row [t, D] this mesh row's tokens: the op sends this row's routing rows and x rows to the column's
// other 7 chips (and copies x_row into x), and reads the column's once they have come.
struct MoeDecodeExpertsTensorArgs {
    const Tensor& x;
    const Tensor& weights;
    const Tensor& w1;
    const Tensor& w2;
    std::optional<Tensor> x_row = std::nullopt;
};

}  // namespace ttnn::experimental::prim
