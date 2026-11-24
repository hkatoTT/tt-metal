// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RecurrentDecodeStepParams {
    uint32_t pairs;      // P: (user, head) pairs, one 128 x 128 state each
    uint32_t heads = 0;  // (the chunked form) pairs a user: pair p is user p / heads's
    uint32_t tokens = 1;  // tokens a pair: 1 (the plain step), 2-8 (the chunked form)
    // the plain step's form (bits; the values the same): 128 a pair's value blocks in a per-core rotated order (the DRAM
    // banks of the cores' state reads and writes spread), 256 the cores of the grid's top 3 rows (virtual y 2-4, served
    // last by NoC 0's returns) read on NoC 1 (their reader on BRISC, their writer on NCRISC
    // with NoC 0), 1024 the units over the cores in proportion to their grid row's speed and with 256 the top 5 rows
    uint32_t form = 0;
    // the chunked form: q .. beta by pair (token k of pair p at row p tokens + k, decode_prepare's pair_major) rather
    // than by token (row p + k pairs)
    bool pair_major = false;
};

// state: [.., 128, 128] BFLOAT16 TILE, P states (the update in fp32, the state rounded to bf16 once a step), updated in
// place.  q, k, v, decay: [.., P, 128] FLOAT32 TILE, one row per pair (q, k L2-normalized, q scaled; decay = exp(g)).
// beta: [.., P, 1+] FLOAT32 TILE, column 0 used.
struct RecurrentDecodeStepTensorArgs {
    const Tensor& state;
    const Tensor& q;
    const Tensor& k;
    const Tensor& v;
    const Tensor& decay;
    const Tensor& beta;
    // the chunked form, out of place (ping-pong): the last step's start state read from
    // `source` and its rows from `saved_source` (default: `state` / `saved` themselves); this step's start and rows
    // still go to `state` and `saved`
    std::optional<Tensor> source;
    // the chunked form: the keep codes, one a user, uint32 [1, B] row-major (B = P / heads)
    std::optional<Tensor> keep;
    std::optional<Tensor> output;  // o to write into (made otherwise)
    // the chunked verify form (`saved`, with `keep`: a verify step of `tokens` tokens a user):
    // the state holds the start state of the last step and `saved` its T tokens' rows (k, decay, beta, v: FLOAT32
    // TILE, a unit's slot of 1536 T + 64 bytes in its own whole tiles, the tiles' bytes as they are; a batch's slot in
    // its first unit's); a user's keep code c redoes the first c + 1 of them (c = T - 1: all) before this step's tokens.
    // The step writes its own start state (after the redone tokens) over the state and its tokens' rows to `saved`: one
    // state written a step
    std::optional<Tensor> saved;
    std::optional<Tensor> saved_source;
};

}  // namespace ttnn::experimental::prim
