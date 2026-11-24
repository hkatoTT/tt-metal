// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "scores_common.hpp"

// DSA indexer pool scores, the reader (scores_common.hpp): queries na + i stride (i < count), blocks ja .. jc - 1 (jc:
// the end of the computed blocks of the core's range; VISIBLE: also of the visible ones, the count sent to compute
// through CB_CNT).  Q_n: tiles 4 n .. 4 n + 3 of qi [N, 1, 32, 128]; W_n: row n of hw [1, 1, N, 32] (tile n / 32) into
// row 0 of a zeroed tile; K_j of query n: page (params[n][2] + j) of the pools cache [P, 1, 32, 128], 4 tiles of
// K_TILE bytes (bf16, or BFP8_b / BFP4_b: compute's unpacker reads CB_K in the cache's format).
void kernel_main() {
    using namespace dsa_scores;
    const uint32_t qi_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t hw_addr = get_arg_val<uint32_t>(2);
    const uint32_t params_addr = get_arg_val<uint32_t>(3);
    const uint32_t na = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    const uint32_t stride = get_arg_val<uint32_t>(6);
    const uint32_t ja = get_arg_val<uint32_t>(7);
    const uint32_t jc = get_arg_val<uint32_t>(8);

    constexpr bool shared = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t tile = get_compile_time_arg_val(1);  // bf16 tile bytes
    // Q_RM: qi is [1, 1, N, 4096] row-major (the projection untilized, no reshape): row n -> CB_QRM, compute tilizes
    constexpr bool q_rm = get_compile_time_arg_val(2) != 0;
    constexpr bool visible = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t kpool = get_compile_time_arg_val(4);
    constexpr uint32_t k_tile = get_compile_time_arg_val(5);  // the pools cache's tile bytes
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto c_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    // the blocks: local block j is the pools' block j MUL + ADD (1, 0: all; MUL 4, ADD c: mesh column c's share of a
    // row's pools, dsa_indexer.pool_scores' shard)
    constexpr uint32_t MUL = get_compile_time_arg_val(p_args.next_compile_time_args_offset());
    constexpr uint32_t ADD = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t P_BYTES = PARAMS_WORDS * 4;

    const auto s_q = TensorAccessor(q_args, qi_addr, q_rm ? 4 * tile : tile);
    const auto s_c = TensorAccessor(c_args, cache_addr, k_tile);
    const auto s_h = TensorAccessor(h_args, hw_addr, tile);
    const auto s_p = TensorAccessor(p_args, params_addr, P_BYTES);
    if (count == 0 || jc <= ja) {
        return;
    }
    // the scratch: params rows of the queries, then the head-weights tile (of tile row `hrow`)
    const uint32_t params_l1 = get_write_ptr(CB_RS);
    const uint32_t h_l1 = params_l1 + count * P_BYTES;
    for (uint32_t i = 0; i < count; ++i) {
        noc_async_read(s_p.get_noc_addr(na + i * stride), params_l1 + i * P_BYTES, P_BYTES);
    }
    noc_async_read_barrier();
    uint32_t hrow = SENT;

    // W tiles: zero the circular buffer once (only row 0 of a slot is written later)
    {
        const uint32_t w0 = get_write_ptr(CB_W);
        const uint32_t w_bytes = get_local_cb_interface(CB_W).fifo_size;
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < w_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, w0 + off, MEM_ZEROS_SIZE < w_bytes - off ? MEM_ZEROS_SIZE : w_bytes - off);
        }
        noc_async_read_barrier();
    }
    auto push_q_w = [&](uint32_t n) {
        const uint32_t q_cb = q_rm ? CB_QRM : CB_Q;
        const uint32_t q_pages = q_rm ? 32 : 4;
        cb_reserve_back(q_cb, q_pages);
        const uint32_t q_l1 = get_write_ptr(q_cb);
        if constexpr (q_rm) {
            noc_async_read(s_q.get_noc_addr(n), q_l1, 4 * tile);  // the 32 heads' 128 values, row-major
        } else {
            for (uint32_t c = 0; c < 4; ++c) {
                noc_async_read(s_q.get_noc_addr(n * 4 + c), q_l1 + c * tile, tile);
            }
        }
        if (n / 32 != hrow) {
            hrow = n / 32;
            noc_async_read(s_h.get_noc_addr(hrow), h_l1, tile);
        }
        noc_async_read_barrier();
        cb_push_back(q_cb, q_pages);
        // row r of the head weights: faces (r / 16) 2 and + 1, row r mod 16 -> faces 0 and 1, row 0
        cb_reserve_back(CB_W, 1);
        const uint32_t r = n % 32;
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            h_l1 + (r / 16) * 2 * 512 + (r % 16) * 32);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_W));
        for (uint32_t i = 0; i < 8; ++i) {
            dst[i] = src[i];
            dst[128 + i] = src[128 + i];  // the next face: 512 bytes on
        }
        cb_push_back(CB_W, 1);
    };
    auto push_k = [&](uint32_t page) {
        cb_reserve_back(CB_K, 4);
        const uint32_t k_l1 = get_write_ptr(CB_K);
        for (uint32_t c = 0; c < 4; ++c) {
            noc_async_read(s_c.get_noc_addr(page * 4 + c), k_l1 + c * k_tile, k_tile);
        }
        noc_async_read_barrier();
        cb_push_back(CB_K, 4);
    };
    auto param = [&](uint32_t i, uint32_t w) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(params_l1 + i * P_BYTES)[w];
    };
    // the end of the blocks to read and compute: jc, or with VISIBLE the visible ones' (at least ja), sent to compute
    auto end_of = [&](uint32_t i0, uint32_t i1) {
        if constexpr (!visible) {
            return jc;
        }
        uint32_t jv = 0;
        for (uint32_t i = i0; i < i1; ++i) {
            const uint32_t v = local_blocks(visible_blocks(param(i, 0), kpool), MUL, ADD);
            jv = v > jv ? v : jv;
        }
        const uint32_t je = jv < jc ? jv : jc;
        const uint32_t n_blocks = je > ja ? je - ja : 0;
        cb_reserve_back(CB_CNT, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_CNT))[0] = n_blocks;
        cb_push_back(CB_CNT, 1);
        return ja + n_blocks;
    };

    if constexpr (shared) {
        const uint32_t je = end_of(0, count);
        for (uint32_t i = 0; i < count; ++i) {
            push_q_w(na + i * stride);
        }
        const uint32_t base = param(0, 2);
        for (uint32_t j = ja; j < je; ++j) {
            push_k(base + j * MUL + ADD);
        }
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t je = end_of(i, i + 1);
            push_q_w(na + i * stride);
            const uint32_t base = param(i, 2);
            for (uint32_t j = ja; j < je; ++j) {
                push_k(base + j * MUL + ADD);
            }
        }
    }
}
