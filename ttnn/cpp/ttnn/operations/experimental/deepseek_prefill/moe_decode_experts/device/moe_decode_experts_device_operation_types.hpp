// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// MoeDecodeExpertsParams::w1_recv's places
constexpr uint32_t W1_SPEC_TAG = 0, W1_SPEC_C = 1, W1_SPEC_LANDING = 2, W1_SPEC_SLOT = 3, W1_SPEC_MAILBOX = 4,
                   W1_SPEC_DRISCS = 5, W1_SPEC_DN_DEPTH = 21, W1_SPEC_H_SLOTS = 22, W1_SPEC_EXPERTS = 23;

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
    // ... and its dtype: BFLOAT16, or BFLOAT8_B (GLM53_AR_BFP8: the row all-reduce's input share in BFP8_b, the
    // reducers' final adds packed into a Bfp8_b CB_OUT)
    tt::tt_metal::DataType rs_dtype = tt::tt_metal::DataType::BFLOAT16;
    // The in-op gather on the 2D fabric: the sender core's 4 packet headers' multicast routes (4 / 3 hops each way
    // along the mesh column) made on the host (tt::tt_fabric::get_2d_mcast_route_maps) and passed as runtime args, in
    // place of encoding them in h1_send.cpp before its first packet (~0.6 us a header, ~2.45 us a call).  The same
    // header bytes: the same sends.  1 on, 2 check (the kernel encodes them as before, compares and prints with
    // DPRINT whether the host's bytes are the same), 0 off; set from MOE_H1_HOST_ROUTES at each call (with the gather
    // only)
    uint32_t h1_host_routes = 0;
    // The in-op gather from 32 tokens a mesh row: each mesh column carries half of K's slices of x and the row
    // neighbours pass the other half on over the mesh row (the row relay: h1_relay.cpp, MOE_H1_RELAY): half the bytes
    // on the column's links.  The same values.  2 (the default, MOE_H1_RELAY unset) on at 64 tokens a mesh row only, 1
    // on from 32, 0 off; set from MOE_H1_RELAY at each call (with the gather only)
    uint32_t h1_relay = 0;
    // The weight landing (GLM53_W1, packed units only): the landing a DRISC a bank fills with this layer's
    // chosen experts' weight blocks in the layer before's end window, read here in place of DRAM when a block's seal
    // (tag | expert << 16, before and after it) says it is this layer's and that expert's.  The spec, a decode call's
    // only (ttnn/w1_landing.py's op_spec; empty: off; in the program cache's key, so a layer a program), at the
    // W1_SPEC_* places below: the layer's tag (1 ..), C (slots, 1 .. 8), the landing's L1 address, a slot's bytes, the
    // DRISCs' mailbox L1 address, the 8 DRISCs' NoC x, y (bank b at 2 b), the down cores' W2 ring depth and h slots
    // (0: the op's defaults 3 / 3), then the slots' experts (local ids) by mesh column: column c's C at c C
    std::vector<uint32_t> w1_recv = {};
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
