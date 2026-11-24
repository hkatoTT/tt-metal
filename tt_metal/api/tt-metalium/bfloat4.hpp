// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <optional>
#include <vector>

template <typename T>
std::vector<uint32_t> pack_as_bfp4_tiles(
    ttsl::Span<const T> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

// The concatenation of transposed bfloat16 slices packed into BFP4 tiles (32 x 32, the tilized order) in one pass: the
// bytes ttnn.from_torch(torch.cat([s.T for s in slices], dim), bfp4) gives.  Each slice a row-major [rows, cols] block
// of bfloat16 bits with its own row stride (values); rows and cols multiples of 32.  dim 1: the result is
// [cols, slices x rows]; dim 0: [slices x cols, rows].
std::vector<uint32_t> pack_transposed_slices_as_bfp4_tiles(
    ttsl::Span<const uint16_t* const> slices, ttsl::Span<const size_t> row_strides, size_t rows, size_t cols, int dim);

std::vector<float> unpack_bfp4_tiles_into_float_vec(
    ttsl::Span<const uint32_t> bfp_tiles,
    bool row_major_output,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
