// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared by the split sparse attention's kernels (split_*.cpp, combine_*.cpp; dsa_indexer.sparse_split).

#pragma once

#include <stdint.h>

namespace dsa_split {

// A gather request's SPLIT word for a core whose share is empty: no rows, no partial written.
constexpr uint32_t EMPTY = 0xFFFFFFFFu;

// The first active chunk of split j of G over n chunks: n - floor((G - j) n / G).  Split 0 starts at 0 and, for
// n >= 1, always ends past it; the shares differ by at most one chunk.
inline uint32_t share_begin(uint32_t n, uint32_t g, uint32_t j) { return n - ((g - j) * n) / g; }

// A unit's partial in the partials tensor: the out tiles [32 rows, V_DHT tiles] (unnormalized), then the row sums
// (column 0) and the row maxima (column 0, unscaled scores): PART_TILES = V_DHT + 2 tiles, page unit * PART_TILES.
constexpr uint32_t part_tiles(uint32_t v_dht) { return v_dht + 2; }

}  // namespace dsa_split
