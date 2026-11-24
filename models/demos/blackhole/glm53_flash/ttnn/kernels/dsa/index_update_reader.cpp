// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DSA decode's indexer update in one op (dsa_indexer.index_update), the reader / writer: a core per user u, its
// segments in turn (a verify step's: the token's, then each draft's one position on), for virtual user v = s USERS + u
// at position pos: row pos mod RING of user u's recent pages (rkey / rgate page u, a ring of RING / 4 pools: read
// once, kept in L1 over the segments, written back in the last) <- row v of the step's indexer key and gate (ik / ig);
// the pool's 4 rows (ring rows 4 ((pos / 4) mod (RING / 4)) ..) to rows 0-3 of compute's tiles (rows 4-31 zero), which
// makes the pool's key; row (pos / 4) mod 32 of u's pool page (params[v][2] + pos / 128) <- the key's row 0, the
// page written back.  The ring keeps a committed token's row while the rejected drafts of a verify step (up to RING - 2
// tokens a user, a pool or two on) write theirs: with rows pos mod 4 a draft one pool on would overwrite the row of a
// committed token of the next step's pool.  Whole 2 KB tiles are read and written (no sub-tile DRAM
// writes); the row copies are 32-byte face rows in L1.
//   CT: USERS, SEGS, CB_RK, CB_RG, CB_KEY, CB_POOL, CB_IK, CB_IG, CB_BIAS, CB_SCALER, CB_PARAMS, CB_RING, RING,
//       then the accessors of ik, ig [1, 1, B, 128] (tiles), params [1, 1, B, 16] (uint32 rows), rkey, rgate
//       [*, 1, 32, 128], pools [*, 1, 32, 128] (tiles), bias [1, 1, 32, 128] (fp32 tiles)
//   RT: the 7 addresses, the unit (a user; >= USERS: idle)

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t TILE = tt::constants::TILE_HW * 2;  // a bf16 tile
constexpr uint32_t COLS = 4;                            // 128 columns: 4 tiles

// row `src_row` of 4 tiles at src -> row `dst_row` of 4 tiles at dst (bf16, 32 x 32 tiles of 4 16 x 16 faces)
FORCE_INLINE void copy_row(uint32_t src, uint32_t src_row, uint32_t dst, uint32_t dst_row) {
    const uint32_t so = (src_row / 16) * 2 * 512 + (src_row % 16) * 32;
    const uint32_t d_o = (dst_row / 16) * 2 * 512 + (dst_row % 16) * 32;
    for (uint32_t t = 0; t < COLS; ++t) {
        for (uint32_t f = 0; f < 2; ++f) {  // the row's two faces
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + t * TILE + so + f * 512);
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + t * TILE + d_o + f * 512);
            for (uint32_t w = 0; w < 8; ++w) {
                d[w] = s[w];
            }
        }
    }
}

// a pool's 4 rows of the ring (rows `first` .., first a multiple of 4 below 16) -> rows 0-3 of 4 tiles at dst, by local
// NoC reads: in a 32 x 32 tile the 4 rows are 128 contiguous bytes of face 0 and of face 1 (8 reads; a RISC-V copy of
// the 4 rows was ~0.7 us a row)
FORCE_INLINE void copy_pool_rows(uint32_t ring, uint32_t first, uint32_t dst) {
    for (uint32_t t = 0; t < COLS; ++t) {
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_read(get_noc_addr(ring + t * TILE + f * 512 + first * 32), dst + t * TILE + f * 512, 4 * 32);
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t USERS = get_compile_time_arg_val(0);
    constexpr uint32_t SEGS = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rk = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rg = get_compile_time_arg_val(3);
    constexpr uint32_t cb_key = get_compile_time_arg_val(4);
    constexpr uint32_t cb_pool = get_compile_time_arg_val(5);
    constexpr uint32_t cb_ik = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ig = get_compile_time_arg_val(7);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(8);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(9);
    constexpr uint32_t cb_params = get_compile_time_arg_val(10);
    constexpr uint32_t cb_ring = get_compile_time_arg_val(11);
    constexpr uint32_t RING = get_compile_time_arg_val(12);
    static_assert(RING % 4 == 0 && RING <= 16, "the recent pages' ring: whole pools in a tile's top faces");
    constexpr auto ik_args = TensorAccessorArgs<13>();
    constexpr auto ig_args = TensorAccessorArgs<ik_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<ig_args.next_compile_time_args_offset()>();
    constexpr auto rk_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto rg_args = TensorAccessorArgs<rk_args.next_compile_time_args_offset()>();
    constexpr auto pool_args = TensorAccessorArgs<rg_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();
    static_assert(ik_args.AlignedPageSize == TILE && ig_args.AlignedPageSize == TILE && p_args.AlignedPageSize == 64);
    static_assert(rk_args.AlignedPageSize == TILE && rg_args.AlignedPageSize == TILE && pool_args.AlignedPageSize == TILE);
    static_assert(bias_args.AlignedPageSize == 2 * TILE);

    const auto ik = TensorAccessor(ik_args, get_arg_val<uint32_t>(0));
    const auto ig = TensorAccessor(ig_args, get_arg_val<uint32_t>(1));
    const auto params = TensorAccessor(p_args, get_arg_val<uint32_t>(2));
    const auto rkey = TensorAccessor(rk_args, get_arg_val<uint32_t>(3));
    const auto rgate = TensorAccessor(rg_args, get_arg_val<uint32_t>(4));
    const auto pools = TensorAccessor(pool_args, get_arg_val<uint32_t>(5));
    const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(6));
    const uint32_t u = get_arg_val<uint32_t>(7);
    if (u >= USERS) {
        return;
    }

    // compute's constants: the slot bias (4 fp32 tiles) and the reduce scaler
    cb_reserve_back(cb_bias, COLS);
    for (uint32_t t = 0; t < COLS; ++t) {
        noc_async_read_page(t, bias, get_write_ptr(cb_bias) + t * 2 * TILE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_bias, COLS);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL,
        /*reduce_factor=*/1>();

    cb_reserve_back(cb_params, 1);
    cb_reserve_back(cb_ik, COLS);
    cb_reserve_back(cb_ig, COLS);
    cb_reserve_back(cb_pool, COLS);
    cb_reserve_back(cb_ring, 2 * COLS);
    const uint32_t p_l1 = get_write_ptr(cb_params), ik_l1 = get_write_ptr(cb_ik), ig_l1 = get_write_ptr(cb_ig);
    const uint32_t pool_l1 = get_write_ptr(cb_pool);
    const uint32_t ring_k = get_write_ptr(cb_ring), ring_g = ring_k + COLS * TILE;  // user u's recent pages
    for (uint32_t t = 0; t < COLS; ++t) {
        noc_async_read_page(u * COLS + t, rkey, ring_k + t * TILE);
        noc_async_read_page(u * COLS + t, rgate, ring_g + t * TILE);
    }
    // compute's tiles: rows 0-3 a segment's pool, rows 4-31 zero.  Zeroed once: CB_RK / CB_RG hold exactly COLS pages,
    // so every segment's reserve gives the same address (and compute only reads them)
    cb_reserve_back(cb_rk, COLS);
    cb_reserve_back(cb_rg, COLS);
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t rk0 = get_write_ptr(cb_rk), rg0 = get_write_ptr(cb_rg);
        for (uint32_t o = 0; o < COLS * TILE; o += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, rk0 + o, MEM_ZEROS_SIZE);
            noc_async_read(zeros, rg0 + o, MEM_ZEROS_SIZE);
        }
    }
    noc_async_read_barrier();
    uint32_t block = 0xFFFFFFFF;  // the tile row of ik / ig in L1 (a few users: every segment's the same)
    for (uint32_t s = 0; s < SEGS; ++s) {
        const uint32_t v = s * USERS + u;
        noc_async_read_page(v, params, p_l1);
        if (v / 32 != block) {
            block = v / 32;
            for (uint32_t t = 0; t < COLS; ++t) {
                noc_async_read_page(block * COLS + t, ik, ik_l1 + t * TILE);
                noc_async_read_page(block * COLS + t, ig, ig_l1 + t * TILE);
            }
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p_l1);
        const uint32_t pos = p[0], pool = pos / 4;
        const uint32_t page = p[2] + pool / 32;
        copy_row(ik_l1, v % 32, ring_k, pos % RING);
        copy_row(ig_l1, v % 32, ring_g, pos % RING);
        if (s + 1 == SEGS) {  // the recent pages back, while compute makes the last key (the pool's write waits for it)
            for (uint32_t t = 0; t < COLS; ++t) {
                noc_async_write_page(u * COLS + t, rkey, ring_k + t * TILE);
                noc_async_write_page(u * COLS + t, rgate, ring_g + t * TILE);
            }
        }
        cb_reserve_back(cb_rk, COLS);
        cb_reserve_back(cb_rg, COLS);
        const uint32_t rk_l1 = get_write_ptr(cb_rk), rg_l1 = get_write_ptr(cb_rg);
        const uint32_t first = (pool % (RING / 4)) * 4;  // the pool's 4 rows in the ring
        copy_pool_rows(ring_k, first, rk_l1);
        copy_pool_rows(ring_g, first, rg_l1);
        noc_async_read_barrier();
        cb_push_back(cb_rk, COLS);
        cb_push_back(cb_rg, COLS);

        for (uint32_t t = 0; t < COLS; ++t) {
            noc_async_read_page(page * COLS + t, pools, pool_l1 + t * TILE);
        }
        noc_async_read_barrier();
        cb_wait_front(cb_key, COLS);
        copy_row(get_read_ptr(cb_key), 0, pool_l1, pool % 32);
        for (uint32_t t = 0; t < COLS; ++t) {
            noc_async_write_page(page * COLS + t, pools, pool_l1 + t * TILE);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_key, COLS);
    }
}
