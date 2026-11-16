// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DecodePrepareParams {
    uint32_t heads;     // H per chip
    float lower;        // the gate's lower bound (g = lower * sigmoid(..))
    float q_scale;      // 1 / sqrt(head dim)
    // a verify step: which users this call takes (0: proj's rows); the history has these users' rows
    uint32_t users = 0;
    // tokens a user (1-8): token k's rows k B .. (k + 1) B (its conv after the earlier ones'); the outputs have
    // tokens B H rows (token 0's pairs, then token 1's ..); the history moves by `tokens`
    uint32_t tokens = 1;
    // the outputs by pair, its tokens in turn (row p tokens + k: a pair's tokens one run of rows, the T-token delta
    // rule's reads), in place of token by token (row p + k B H)
    bool pair_major = false;
    // the outputs L1 interleaved, in place of DRAM (a delta rule reading them next: no DRAM write and read)
    bool l1_outputs = false;
    // q / k / v's units (part, head, 16 users) each split over 2 cores by channel halves (2 of the 4 tiles each; q and
    // k's L2 norm sums the halves' squares across the pair of cores), one unit-half a core: the same values, about
    // half the compute a core.  Off where the cores are too few
    bool split_channels = false;
    // the plain step's call (one token, no keep, all rows, not split, more units than cores): the top 6 grid rows'
    // reads on NoC 1 (the others' on NoC 0, where alone the lower rows starve) and the units over the cores in
    // proportion to their grid row's speed: the same values
    bool row_split = false;
};

// proj: [.., B, >= 3 H 128 + 32] BFLOAT16 TILE (q | k | v channels, then the H beta logits).
// fg: [.., B, >= H 128] BFLOAT16 TILE (f first).  history0-2: [.., B, 3 H 128] FLOAT32 TILE (oldest first, updated).
// taps: [.., 4, 3 H 128] FLOAT32 TILE (the conv taps, oldest first).  gate_consts: [.., 2, H 128] FLOAT32 TILE
// (dt_bias, rate).
struct DecodePrepareTensorArgs {
    const Tensor& proj;
    const Tensor& fg;
    const Tensor& history0;
    const Tensor& history1;
    const Tensor& history2;
    const Tensor& taps;
    const Tensor& gate_consts;
    // a verify step of `tokens` (2-8) tokens a user: each user's keep code c (uint32 [1, B] row-major) below the saved
    // rows n = tokens - 1 reads its history back by n - c in (saved 0 .. saved n - 1, history0, history1, history2),
    // undoing its rejected drafts' shift; saved k gets row k + 1 of (the read history, the tokens' inputs) (the
    // checkpoint the next step may come back to)
    std::optional<Tensor> keep;
    std::optional<Tensor> saved;
    // out of place (ping-pong): the new history and checkpoint rows written here in
    // place of history0-2 / saved (read as before): the call can be run again from the same inputs.  All four, or none
    std::optional<Tensor> history0_out;
    std::optional<Tensor> history1_out;
    std::optional<Tensor> history2_out;
    std::optional<Tensor> saved_out;
};

}  // namespace ttnn::experimental::prim
