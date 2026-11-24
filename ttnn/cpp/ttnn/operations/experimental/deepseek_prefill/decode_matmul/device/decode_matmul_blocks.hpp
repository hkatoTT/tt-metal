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
// tail (strided only; decode_matmul's fg): the last `tail` columns each in a block of its own, after the runs of their
// class's other columns (a block of one column ends about halfway through the op: fg's input is there early).
inline std::vector<DecodeMatmulBlock> decode_matmul_blocks(
    uint32_t nt, uint32_t banks, uint32_t nc, bool strided, uint32_t tail = 0) {
    std::vector<DecodeMatmulBlock> out;
    if (strided && tail != 0) {
        std::vector<std::vector<DecodeMatmulBlock>> per(std::min(banks, nt));
        for (uint32_t r = 0; r < per.size(); ++r) {
            const uint32_t count = (nt - r + banks - 1) / banks;  // columns r, r + B, .. below nt
            uint32_t head = count;                                // the class's columns before the tail
            while (head > 0 && r + banks * (head - 1) >= nt - tail) {
                --head;
            }
            for (uint32_t j0 = 0; j0 < head; j0 += nc) {
                per[r].push_back({r + banks * j0, std::min(nc, head - j0)});
            }
            for (uint32_t j = head; j < count; ++j) {
                per[r].push_back({r + banks * j, 1});
            }
        }
        for (uint32_t i = 0;; ++i) {  // the blocks in order of their place in the class, then residue
            bool any = false;
            for (const auto& blocks : per) {
                if (i < blocks.size()) {
                    out.push_back(blocks[i]);
                    any = true;
                }
            }
            if (!any) {
                break;
            }
        }
        return out;
    }
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

// A core's weight stream (decode_matmul_weights.cpp's runtime arguments 1 .. 9 and its chunks): K tiles k0 .. k0 + kc
// - 1, N tiles n0 + i stride (i < ncb), in chunks of cr K rows x sb columns (per sub-block of sb columns, per chunk
// of cr rows; a short last chunk / sub-block leaves the rest of its slot unread), `chunks` of them (0 without a block).
struct DecodeMatmulStream {
    uint32_t x, y;  // the logical core
    uint32_t k0, kc, n0, ncb;
    uint32_t chunks;
};
struct DecodeMatmulStreams {
    uint32_t cr, sb, stride;
    std::vector<DecodeMatmulStream> cores;  // every core of the rectangles, slice by slice, row-major in each
};

// The factory's split for these shapes (decode_matmul_program_factory.cpp, which checks itself against this): K /
// 32 = kt, N / 32 = nt, x's tile rows mt, the plan's k_split ks / n_per_core nc / rectangle cw x rh on a grid gx
// wide, `banks` DRAM banks, strided blocks, fg's tail fg_k, chunk_target tiles a chunk (0: 8).
inline DecodeMatmulStreams decode_matmul_streams(
    uint32_t kt,
    uint32_t nt,
    uint32_t mt,
    uint32_t ks,
    uint32_t nc,
    uint32_t cw,
    uint32_t rh,
    uint32_t gx,
    uint32_t banks,
    bool strided,
    uint32_t fg_k,
    uint32_t chunk_target) {
    auto div_up = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
    const auto blocks = decode_matmul_blocks(nt, banks, nc, strided, fg_k);
    const uint32_t nb = static_cast<uint32_t>(blocks.size()), rect = cw * rh, rects_x = gx / cw;
    const uint32_t kc_max = div_up(kt, ks);
    const uint32_t nsub = div_up(mt * nc, 8), sb = div_up(nc, nsub);
    const uint32_t cr = std::min(kc_max, std::max(1u, (chunk_target ? chunk_target : 8u) / sb));
    DecodeMatmulStreams out{cr, sb, strided ? banks : 1u, {}};
    std::vector<int32_t> block_at(rect, -1);
    for (uint32_t i = 0; i < nb; ++i) {
        block_at[i * rect / nb] = static_cast<int32_t>(i);
    }
    for (uint32_t s = 0; s < ks; ++s) {
        const uint32_t ox = (s % rects_x) * cw, oy = (s / rects_x) * rh;
        const uint32_t k0 = s * kt / ks, kc = (s + 1) * kt / ks - k0;
        for (uint32_t p = 0; p < rect; ++p) {
            const int32_t b = block_at[p];
            const uint32_t n0 = b >= 0 ? blocks[b].n0 : 0, ncb = b >= 0 ? blocks[b].ncb : 0;
            out.cores.push_back({ox + p % cw, oy + p / cw, k0, kc, n0, ncb, ncb ? div_up(ncb, sb) * div_up(kc, cr) : 0});
        }
    }
    return out;
}

}  // namespace ttnn::experimental::prim
