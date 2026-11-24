// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DSA decode's sparse attention with each query's keys split over G cores (dsa_indexer.sparse_split): the reader
// (from ttnn's sparse_sdpa_reader.cpp, bf16 latent cache, natural order). A core's unit is (token, split j): it reads
// the token's Q rows and index row, counts the valid keys nv (the sentinels are a contiguous tail), and takes its
// share of the active chunks: [c0, c1) with c0 = n - floor((G - j) n / G), so split 0 always has one.  Per chunk the
// K rows are gathered over both NoCs (half here, half by the writer: cb_kreq / cb_kack) and the boundary mask tile
// is made for the token's last chunk.  An empty share (n < G) tells compute 0 chunks and the writer EMPTY.
//   CT: H, S, TOPK, K_CHUNK, K_DIM, G, CB_Q_RM, CB_K_RM, CB_MASK_PART, CB_IDX, CB_CTRL, CB_KREQ, CB_KACK, Q_TILED,
//       CB_Q_IN, then the accessors of q ([1, H, S, K_DIM] row-major, or the split latents' [1, UPQ H, S / UPQ, K_DIM];
//       Q_TILED: [1, S, 16, K_DIM] tiles, a token's tile row its heads, read straight into cb_q_in), kv [1, 1, T,
//       K_DIM], indices [1, 1, S / UPQ, TOPK] (row-major), UPQ, ZERO_KEYS
//   RT: q, kv, indices addresses, the first unit (token * G + j) and the count of this core's units (each in turn: a
//       prefill's 2 S G units over the cores; >= S * G: skipped)

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"
#include "split_common.hpp"

constexpr uint32_t sentinel = 0xFFFFFFFFu;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t topk = get_compile_time_arg_val(2);
    constexpr uint32_t k_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t k_dim = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(6);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(7);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(8);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(9);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(10);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(11);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(12);
    constexpr bool q_tiled = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(14);
    constexpr uint32_t kv_bytes = get_compile_time_arg_val(15);  // the latent cache's datum: 2 (bf16) or 1 (fp8 e4m3)
    constexpr auto q_args = TensorAccessorArgs<16>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    // units a query (split latents: its 64 heads as two 32-head units, tokens 2 s and 2 s + 1 of q): the
    // index row tok / UPQ
    constexpr uint32_t UPQ = get_compile_time_arg_val(idx_args.next_compile_time_args_offset());
    // split latents (the factory's write_empty): a chip may hold none of a query's keys (positions 0-2), its units then
    // empty
    constexpr bool ZERO_KEYS = get_compile_time_arg_val(idx_args.next_compile_time_args_offset() + 1) != 0;

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t u_first = get_arg_val<uint32_t>(3);  // this core's units u_first .. u_first + n_units - 1
    const uint32_t n_units = get_arg_val<uint32_t>(4);
    for (uint32_t k = 0; k < n_units; ++k) {
    const uint32_t unit = u_first + k;
    if (unit >= S * G) {
        continue;
    }
    const uint32_t tok = unit / G;
    const uint32_t split = unit % G;

    constexpr uint32_t row_bytes = k_dim * 2;  // bf16 Q rows
    constexpr uint32_t kv_row_bytes = k_dim * kv_bytes;  // K (latent) rows
    constexpr uint32_t idx_row_bytes = topk * 4;
    // the accessors' pages as the factory meant them (a misplaced argument would show here, not as stray reads)
    constexpr uint32_t tile_bytes = tt::constants::TILE_HW * 2;
    constexpr uint32_t q_tiles = k_dim / tt::constants::TILE_WIDTH;
    static_assert(q_args.AlignedPageSize == (q_tiled ? tile_bytes : row_bytes) && kv_args.AlignedPageSize == kv_row_bytes);
    static_assert(idx_args.AlignedPageSize == idx_row_bytes);

    Noc noc;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_rm), mask_part_cb(cb_mask_part), idx_cb(cb_idx), ctrl_cb(cb_ctrl);
    experimental::CB kreq_cb(cb_kreq), kack_cb(cb_kack);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    // Q: H head rows (row (h, tok) = page h * S + tok) -> cb_q_rm; tiled, the token's tile row -> cb_q_in.  The split
    // latents' row-major q [1, UPQ H, S / UPQ, K_DIM] (a query's 64 heads, no transpose): unit-token tok = query tok /
    // UPQ's heads H (tok % UPQ) .., row (h, tok) = page (H (tok % UPQ) + h) S / UPQ + tok / UPQ (UPQ 1: h S + tok)
    experimental::CB q_in_cb(cb_q_in);
    if constexpr (q_tiled) {
        q_in_cb.reserve_back(q_tiles);
        for (uint32_t t = 0; t < q_tiles; ++t) {
            noc.async_read(q, q_in_cb, tile_bytes, {.page_id = tok * q_tiles + t}, {.offset_bytes = t * tile_bytes});
        }
    } else {
        q_cb.reserve_back(H);
        const uint32_t q_row0 = H * (tok % UPQ) * (S / UPQ) + tok / UPQ;
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_read(q, q_cb, row_bytes, {.page_id = q_row0 + h * (S / UPQ)}, {.offset_bytes = h * row_bytes});
        }
    }
    noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = tok / UPQ}, {.offset_bytes = 0});
    noc.async_read_barrier();
    if constexpr (q_tiled) {
        q_in_cb.push_back(q_tiles);
    } else {
        q_cb.push_back(H);
    }

    uint32_t nv = topk;  // the first sentinel = the valid keys
    {
        uint32_t lo = 0, hi = topk;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (idx_ptr[mid] == sentinel) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        // the unsplit cache: at least one key by contract (a query's own position is always among them); the split
        // latents (ZERO_KEYS, its 32-head units or 64-head units alike): a chip may hold none of a query's keys
        // (positions 0-2: only the tail's tokens 0 .. p, on chips 0 .. p), then every share is empty and the writer
        // writes neutral partials (WRITE_EMPTY), which the combine weighs 0
        nv = lo == 0 ? (ZERO_KEYS ? 0 : 1) : lo;
    }
    const uint32_t n_active = (nv + k_chunk - 1) / k_chunk;
    const uint32_t c0 = dsa_split::share_begin(n_active, G, split);
    const uint32_t c1 = dsa_split::share_begin(n_active, G, split + 1);

    // compute's control: its chunks and the valid keys from its first chunk on (the last chunk's boundary)
    ctrl_cb.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
        const uint32_t end = nv < c1 * k_chunk ? nv : c1 * k_chunk;
        cp[sparse_sdpa::control_message::ACTIVE_CHUNKS] = c1 - c0;
        cp[sparse_sdpa::control_message::VALID_KEYS] = c1 > c0 ? end - c0 * k_chunk : 0;
    }
    ctrl_cb.push_back(1);

    if (c1 == c0) {  // nothing to gather: the writer only acknowledges
        kreq_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* request =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
            request[sparse_sdpa::gather_request::BASE] = 0;
            request[sparse_sdpa::gather_request::SPLIT] = dsa_split::EMPTY;
            request[sparse_sdpa::gather_request::IS_LAST] = 1;
            request[sparse_sdpa::gather_request::DST_L1] = 0;
        }
        kreq_cb.push_back(1);
        kack_cb.wait_front(1);
        kack_cb.pop_front(1);
        continue;
    }

    for (uint32_t chunk = c0; chunk < c1; ++chunk) {
        const uint32_t base = chunk * k_chunk;
        const uint32_t valid = (nv - base) < k_chunk ? (nv - base) : k_chunk;  // (0, k_chunk]
        k_cb.reserve_back(k_chunk);
        const uint32_t k_write_ptr = k_cb.get_write_ptr();
        const uint32_t half = valid >> 1;  // the writer gathers rows [0, half) on its NoC, this core [half, valid)
        kreq_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* request =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
            request[sparse_sdpa::gather_request::BASE] = base;
            request[sparse_sdpa::gather_request::SPLIT] = half;
            request[sparse_sdpa::gather_request::IS_LAST] = chunk == c1 - 1;
            request[sparse_sdpa::gather_request::DST_L1] = k_write_ptr;
        }
        kreq_cb.push_back(1);
        sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::K_TRID_RING>(
            noc, kv, k_write_ptr, idx_ptr, base, half, valid, kv_row_bytes, 0);
        kack_cb.wait_front(1);
        kack_cb.pop_front(1);
        if (valid < k_chunk) {  // the sentinel suffix: defined zeros (compute adds -inf there)
            noc.async_write_zeros(k_cb, (k_chunk - valid) * kv_row_bytes, {.offset_bytes = valid * kv_row_bytes});
            noc.write_zeros_l1_barrier();
        }
        k_cb.push_back(k_chunk);

        // the token's last chunk with a boundary inside a key tile: that tile's row 0 is -inf from the boundary on
        if (chunk == n_active - 1 && (valid % tt::constants::TILE_WIDTH) != 0) {
            mask_part_cb.reserve_back(1);
            {
                const uint32_t first_masked_col = valid % tt::constants::TILE_WIDTH;
                volatile tt_l1_ptr uint16_t* mask_tile =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_part_cb.get_write_ptr());
                for (uint32_t col = 0; col < tt::constants::TILE_WIDTH; ++col) {
                    const uint32_t elem_off = (col < tt::constants::FACE_WIDTH)
                                                  ? col
                                                  : (tt::constants::FACE_HW + col - tt::constants::FACE_WIDTH);
                    mask_tile[elem_off] = (col >= first_masked_col) ? neg_inf_bf16 : (uint16_t)0;
                }
            }
            mask_part_cb.push_back(1);
        }
    }
    }  // units
}
