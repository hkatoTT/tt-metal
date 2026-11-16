// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DSA indexer pool scores (dsa_indexer.py `pool_scores`): the circular buffers and the two loop orders.
//   per query (decode): each query its own pools (its user's pages): for each query, its Q and W, then its blocks;
//   shared (prefill, and decode's groups: a user's T queries): the core's queries share the pools (one user): their Q
//   and W once, then for each block, every query.  The core's queries: n = na + i stride, i < count.
// VISIBLE: only the blocks with a pool the queries see (block j < ceil(((pos + 1) / 4) / 32), the largest pos of the
// core's queries when they share) are read and computed; the writer sets the rest -inf as it masked them.
// A unit (query n, block j: pools 32 j .. 32 j + 31): S = relu(Q_n K_j^T) [32 heads x 32 pools] (fp32 Dest), then
// W_n S, whose row 0 (W_n's row 0: the query's head weights) is the block's scores.
namespace dsa_scores {
constexpr uint32_t CB_Q = 0;    // a query's 32 heads x 128, 4 tiles (bf16)
constexpr uint32_t CB_W = 1;    // a query's head weights in row 0 of a tile, zeros elsewhere (bf16)
constexpr uint32_t CB_K = 2;    // a block's 32 pools x 128, 4 tiles (bf16)
constexpr uint32_t CB_S = 3;    // relu(Q K^T) (fp32)
constexpr uint32_t CB_OUT = 4;  // W S (bf16; row 0 read by the writer)
constexpr uint32_t CB_RS = 5;   // the reader's scratch: its queries' params rows, a head-weights tile
constexpr uint32_t CB_WS = 6;   // the writer's scratch: its queries' params rows, a row of -inf
constexpr uint32_t CB_QRM = 7;  // Q_RM (decode): a query's 32 heads x 128 row-major (32 rows), tilized into CB_Q
constexpr uint32_t CB_CNT = 8;  // VISIBLE: the reader's count of blocks to compute (a 16-byte page)
constexpr uint32_t PARAMS_WORDS = 16;
constexpr uint32_t SENT = 0xFFFFFFFFu;

// the end of the visible blocks of a query at `pos` (SENT: none), kpool tokens a pool, 32 pools a block
inline uint32_t visible_blocks(uint32_t pos, uint32_t kpool) {
    if (pos == SENT) {
        return 0;
    }
    const uint32_t vis = (pos + 1) / kpool;
    return (vis + 31) / 32;
}

// the local blocks j (global block j mul + add) below the global block `end`: with mul 1, add 0 `end` itself
inline uint32_t local_blocks(uint32_t end, uint32_t mul, uint32_t add) { return end > add ? (end - add + mul - 1) / mul : 0; }
}  // namespace dsa_scores
