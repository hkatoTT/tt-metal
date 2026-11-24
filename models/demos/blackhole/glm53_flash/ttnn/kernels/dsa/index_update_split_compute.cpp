// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DSA decode's indexer update, the split form (dsa_indexer.index_update; index_update_split_reader.cpp), the compute:
// the count of keys from the reader (CB_CTL), then per key the pool's key from its 4 recent rows (rows 0-3 of the
// rkey / rgate tiles, rows 4-31 zero; the slot bias -inf below), the same tile ops as index_update_compute.cpp:
// per feature t = gate + bias (fp32), e = exp(t - max over the rows), w = e / sum e (rounded to bf16), key = sum over
// the rows of bf16(w rk) -> row 0 of 4 bf16 tiles.  BATCH 4: each op on the 4 column tiles in one acquire (DST 0-3,
// fp32: half of the register), the ops and their order on a tile as BATCH 1's (a tile op an acquire).
//   CT: CB_RK, CB_RG, CB_KEY, CB_BIAS, CB_SCALER, CB_T, CB_M, CB_E, CB_RS, CB_W, CB_P, CB_CTL, BATCH.  RT: none.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"

namespace {
// the reader's count (word 0 of CB_CTL's page) on every thread
ALWI uint32_t read_count(uint32_t cb) {
    UNPACK(asm volatile("fence" ::: "memory");)  // (the RISC's small L1 cache: a line of an earlier step's count)
    return read_tile_value(cb, 0, 0);
}

// DST 0 .. n - 1 -> n tiles of out_cb (at its write pointer, in order)
template <uint32_t out_cb, uint32_t n>
ALWI void pack_n() {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(out_cb);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(i, out_cb);
    }
    tile_regs_release();
}

// one key: the 4 column tiles c0 .. c0 + N - 1 at a time (N = 1: c0 the column, the tiles of the step's CBs at 0)
template <uint32_t N, uint32_t cb_rk, uint32_t cb_rg, uint32_t cb_key, uint32_t cb_bias, uint32_t cb_scaler,
          uint32_t cb_t, uint32_t cb_m, uint32_t cb_e, uint32_t cb_rs, uint32_t cb_w, uint32_t cb_p>
ALWI void columns(uint32_t c0) {
    // t = gate + bias
    cb_reserve_back(cb_t, N);
    tile_regs_acquire();
    reconfig_data_format(cb_rg, cb_bias);
    add_init(cb_rg, cb_bias);
    for (uint32_t i = 0; i < N; ++i) {
        add_tiles(cb_rg, cb_bias, c0 + i, c0 + i, i);
    }
    pack_n<cb_t, N>();
    cb_push_back(cb_t, N);
    cb_wait_front(cb_t, N);
    // m = max over the rows (row 0)
    cb_reserve_back(cb_m, N);
    tile_regs_acquire();
    reconfig_data_format(cb_t, cb_scaler);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_t, cb_scaler, cb_m);
    for (uint32_t i = 0; i < N; ++i) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_t, cb_scaler, i, 0, i);
    }
    reduce_uninit();
    pack_n<cb_m, N>();
    cb_push_back(cb_m, N);
    cb_wait_front(cb_m, N);
    // e = exp(t - m)
    cb_reserve_back(cb_e, N);
    tile_regs_acquire();
    reconfig_data_format(cb_t, cb_m);
    sub_bcast_rows_init(cb_t, cb_m);
    for (uint32_t i = 0; i < N; ++i) {
        sub_tiles_bcast_rows(cb_t, cb_m, i, i, i);
    }
    exp_tile_init<false>();
    for (uint32_t i = 0; i < N; ++i) {
        exp_tile<false>(i);
    }
    pack_n<cb_e, N>();
    cb_push_back(cb_e, N);
    cb_pop_front(cb_t, N);
    cb_pop_front(cb_m, N);
    cb_wait_front(cb_e, N);
    // 1 / sum over the rows
    cb_reserve_back(cb_rs, N);
    tile_regs_acquire();
    reconfig_data_format(cb_e, cb_scaler);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_e, cb_scaler, cb_rs);
    for (uint32_t i = 0; i < N; ++i) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_e, cb_scaler, i, 0, i);
    }
    reduce_uninit();
    recip_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        recip_tile(i);
    }
    pack_n<cb_rs, N>();
    cb_push_back(cb_rs, N);
    cb_wait_front(cb_rs, N);
    // w = e / sum (bf16)
    cb_reserve_back(cb_w, N);
    tile_regs_acquire();
    reconfig_data_format(cb_e, cb_rs);
    mul_bcast_rows_init(cb_e, cb_rs);
    for (uint32_t i = 0; i < N; ++i) {
        mul_tiles_bcast_rows(cb_e, cb_rs, i, i, i);
    }
    pack_n<cb_w, N>();
    cb_push_back(cb_w, N);
    cb_pop_front(cb_e, N);
    cb_pop_front(cb_rs, N);
    cb_wait_front(cb_w, N);
    // bf16(w rk)
    cb_reserve_back(cb_p, N);
    tile_regs_acquire();
    reconfig_data_format(cb_w, cb_rk);
    mul_init(cb_w, cb_rk);
    for (uint32_t i = 0; i < N; ++i) {
        mul_tiles(cb_w, cb_rk, i, c0 + i, i);
    }
    pack_n<cb_p, N>();
    cb_push_back(cb_p, N);
    cb_pop_front(cb_w, N);
    cb_wait_front(cb_p, N);
    // the key: sum over the rows (row 0), into the key's tiles in column order
    tile_regs_acquire();
    reconfig_data_format(cb_p, cb_scaler);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_p, cb_scaler, cb_key);
    for (uint32_t i = 0; i < N; ++i) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_p, cb_scaler, i, 0, i);
    }
    reduce_uninit();
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_key);
    for (uint32_t i = 0; i < N; ++i) {
        pack_tile<true>(i, cb_key, c0 + i);
    }
    tile_regs_release();
    cb_pop_front(cb_p, N);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_rk = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rg = get_compile_time_arg_val(1);
    constexpr uint32_t cb_key = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_t = get_compile_time_arg_val(5);
    constexpr uint32_t cb_m = get_compile_time_arg_val(6);
    constexpr uint32_t cb_e = get_compile_time_arg_val(7);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(8);
    constexpr uint32_t cb_w = get_compile_time_arg_val(9);
    constexpr uint32_t cb_p = get_compile_time_arg_val(10);
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(11);
    constexpr uint32_t BATCH = get_compile_time_arg_val(12);
    static_assert(BATCH == 1 || BATCH == 4, "a tile an op, or the 4 columns");

    compute_kernel_hw_startup(cb_rg, cb_bias, cb_t);
    cb_wait_front(cb_ctl, 1);
    const uint32_t keys = read_count(cb_ctl);
    cb_pop_front(cb_ctl, 1);
    if (keys == 0) {
        return;
    }
    cb_wait_front(cb_bias, 4);
    cb_wait_front(cb_scaler, 1);
    for (uint32_t k = 0; k < keys; ++k) {
        cb_wait_front(cb_rk, 4);
        cb_wait_front(cb_rg, 4);
        cb_reserve_back(cb_key, 4);
        if constexpr (BATCH == 4) {
            columns<4, cb_rk, cb_rg, cb_key, cb_bias, cb_scaler, cb_t, cb_m, cb_e, cb_rs, cb_w, cb_p>(0);
        } else {
            for (uint32_t c = 0; c < 4; ++c) {
                columns<1, cb_rk, cb_rg, cb_key, cb_bias, cb_scaler, cb_t, cb_m, cb_e, cb_rs, cb_w, cb_p>(c);
            }
        }
        cb_push_back(cb_key, 4);
        cb_pop_front(cb_rk, 4);
        cb_pop_front(cb_rg, 4);
    }
}
