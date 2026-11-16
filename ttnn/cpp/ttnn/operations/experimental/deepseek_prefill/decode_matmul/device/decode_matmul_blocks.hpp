// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ttnn::experimental::prim {

// An N block: tiles n0, n0 + stride, .., n0 + (ncb - 1) stride.
struct DecodeMatmulBlock {
    uint32_t n0;
    uint32_t ncb;
};

// The N blocks of at most nc tiles for N / 32 = nt.  Strided (stride = the DRAM banks B): a block is a run of nc
// columns of one residue class mod B, so that in every K row its tiles are consecutive pages of one bank (page p and
// p + B are adjacent in bank p mod B), read as one request; the blocks in order of their run, then residue (the
// cores side by side read different banks).  Not strided: runs of nc consecutive columns (one request a tile).
inline std::vector<DecodeMatmulBlock> decode_matmul_blocks(uint32_t nt, uint32_t banks, uint32_t nc, bool strided) {
    std::vector<DecodeMatmulBlock> out;
    if (!strided) {
        for (uint32_t n0 = 0; n0 < nt; n0 += nc) {
            out.push_back({n0, std::min(nc, nt - n0)});
        }
        return out;
    }
    for (uint32_t j0 = 0;; j0 += nc) {
        bool any = false;
        for (uint32_t r = 0; r < banks && r < nt; ++r) {
            const uint32_t count = (nt - r + banks - 1) / banks;  // columns r, r + B, .. below nt
            if (j0 < count) {
                out.push_back({r + banks * j0, std::min(nc, count - j0)});
                any = true;
            }
        }
        if (!any) {
            break;
        }
    }
    return out;
}

}  // namespace ttnn::experimental::prim
