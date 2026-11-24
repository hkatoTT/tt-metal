// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The split sparse attention's combine (dsa_indexer.sparse_combine), the reader: a unit is (token, column group q
// of C out tiles).  From the token's index row it finds the valid keys and so the splits that wrote a partial (an
// empty share writes none), then reads, for each of those, its row maxima, its row sums and its C out tiles, and
// makes the rows' maxima over the splits, M (column 0 of a bf16 tile: integer compares of the bf16 patterns).
//   CT: S, G, V_DHT, C, Q, TOPK, K_CHUNK, CB_M, CB_L, CB_O, CB_CTRL, CB_IDX, CB_MAX, then the accessors of the partials
//       (tiles, split_common.hpp) and the indices [1, 1, S, TOPK] (row-major)
//   RT: the partials' and the indices' addresses, the first unit (token * Q + q) and the count of this core's units

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "split_common.hpp"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t Q = get_compile_time_arg_val(4);
    constexpr uint32_t topk = get_compile_time_arg_val(5);
    constexpr uint32_t k_chunk = get_compile_time_arg_val(6);
    constexpr uint32_t cb_m = get_compile_time_arg_val(7);
    constexpr uint32_t cb_l = get_compile_time_arg_val(8);
    constexpr uint32_t cb_o = get_compile_time_arg_val(9);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(10);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(11);
    constexpr uint32_t cb_max = get_compile_time_arg_val(12);
    constexpr auto part_args = TensorAccessorArgs<13>();
    constexpr auto idx_args = TensorAccessorArgs<part_args.next_compile_time_args_offset()>();
    // XCHIP (the split latents: a long cache's latents over the row's 4 chips by token): the partials are the row's
    // exchange ([1, 4, R, PART x 32]: dsa_indexer.shard_repack, then all_to_all), every one written: chip c's split j
    // of token tok is item m = tok XG + j of chip c's plane, tile row m / 2, rows 16 (m mod 2) .. (read into a tile's
    // top half); G = 4 XG
    constexpr uint32_t xchip = get_compile_time_arg_val(idx_args.next_compile_time_args_offset());
    constexpr uint32_t XG = get_compile_time_arg_val(idx_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t PLANE_ROWS = get_compile_time_arg_val(idx_args.next_compile_time_args_offset() + 2);

    const uint32_t part_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t u_first = get_arg_val<uint32_t>(2);  // this core's units u_first .. u_first + n_units - 1
    const uint32_t n_units = get_arg_val<uint32_t>(3);
    for (uint32_t k = 0; k < n_units; ++k) {
    const uint32_t unit = u_first + k;
    if (unit >= S * Q) {
        continue;
    }
    const uint32_t tok = unit / Q;
    const uint32_t col0 = (unit % Q) * C;
    constexpr uint32_t part_tiles = dsa_split::part_tiles(vDHt);
    constexpr uint32_t tile_bytes = tt::constants::TILE_HW * 2;
    static_assert(part_args.AlignedPageSize == tile_bytes && idx_args.AlignedPageSize == topk * 4);

    const auto part = TensorAccessor(part_args, part_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    uint32_t n_active = 0, count = G;
    if constexpr (!xchip) {
        cb_reserve_back(cb_idx, 1);
        const uint32_t idx_l1 = get_write_ptr(cb_idx);
        noc_async_read_page(tok, idx, idx_l1);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
        uint32_t lo = 0, hi = topk;  // the first sentinel = the valid keys
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (idx_ptr[mid] == 0xFFFFFFFFu) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        const uint32_t nv = lo == 0 ? 1 : lo;
        n_active = (nv + k_chunk - 1) / k_chunk;
        count = 0;
        for (uint32_t j = 0; j < G; ++j) {
            count += dsa_split::share_begin(n_active, G, j + 1) > dsa_split::share_begin(n_active, G, j);
        }
    }
    cb_reserve_back(cb_ctrl, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctrl))[0] = count;
    cb_push_back(cb_ctrl, 1);

    cb_reserve_back(cb_m, count);
    cb_reserve_back(cb_l, count);
    cb_reserve_back(cb_o, count * C);
    uint32_t m_l1 = get_write_ptr(cb_m), l_l1 = get_write_ptr(cb_l), o_l1 = get_write_ptr(cb_o);
    for (uint32_t j = 0; j < G; ++j) {
        if constexpr (xchip) {  // chip j / XG's split j mod XG: a tile's half
            const uint32_t m = tok * XG + j % XG;
            const uint32_t page = ((j / XG) * PLANE_ROWS + m / 2) * part_tiles;
            const uint32_t off = (m % 2) * (tile_bytes / 2);
            noc_async_read(part.get_noc_addr(page + vDHt + 1, off), m_l1, tile_bytes / 2);
            noc_async_read(part.get_noc_addr(page + vDHt, off), l_l1, tile_bytes / 2);
            for (uint32_t c = 0; c < C; ++c) {
                noc_async_read(part.get_noc_addr(page + col0 + c, off), o_l1 + c * tile_bytes, tile_bytes / 2);
            }
        } else {
            if (dsa_split::share_begin(n_active, G, j + 1) == dsa_split::share_begin(n_active, G, j)) {
                continue;
            }
            const uint32_t page = (tok * G + j) * part_tiles;
            noc_async_read_page(page + vDHt + 1, part, m_l1);
            noc_async_read_page(page + vDHt, part, l_l1);
            for (uint32_t c = 0; c < C; ++c) {
                noc_async_read_page(page + col0 + c, part, o_l1 + c * tile_bytes);
            }
        }
        m_l1 += tile_bytes;
        l_l1 += tile_bytes;
        o_l1 += C * tile_bytes;
    }
    noc_async_read_barrier();

    // M: per row r, the largest of the splits' maxima (column 0: element (r, 0) of the tile's faces 0 / 2)
    cb_reserve_back(cb_max, 1);
    volatile tt_l1_ptr uint16_t* mx = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_max));
    volatile tt_l1_ptr uint16_t* ms = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_m));
    for (uint32_t i = 0; i < tt::constants::TILE_HW; ++i) {
        mx[i] = 0;
    }
    for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
        const uint32_t at = (r / tt::constants::FACE_HEIGHT) * 2 * tt::constants::FACE_HW +
                            (r % tt::constants::FACE_HEIGHT) * tt::constants::FACE_WIDTH;
        uint32_t best = 0, best_key = 0;
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t v = ms[k * tt::constants::TILE_HW + at];
            const uint32_t key = (v & 0x8000u) ? (~v & 0xFFFFu) : (v | 0x8000u);  // orders as the floats
            if (k == 0 || key > best_key) {
                best = v;
                best_key = key;
            }
        }
        mx[at] = static_cast<uint16_t>(best);
    }
    cb_push_back(cb_max, 1);
    cb_push_back(cb_m, count);
    cb_push_back(cb_l, count);
    cb_push_back(cb_o, count * C);
    }  // units
}
