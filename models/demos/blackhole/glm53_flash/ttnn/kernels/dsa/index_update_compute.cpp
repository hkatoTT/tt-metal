// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DSA decode's indexer update in one op (dsa_indexer.index_update), the compute: per segment, the pool's key from
// the pool's 4 recent rows (rows 0-3 of the rkey / rgate tiles the reader fills from its user's recent pages, rows 4-31
// zero; the slot bias is -inf below), as `TtDsa._compress`:
// per feature t = gate + bias (fp32), e = exp(t - max over the rows), w = e / sum e (rounded to bf16), key = sum over
// the rows of bf16(w rk) -> row 0 of 4 bf16 tiles.  One tile op an acquire (fp32 intermediates in fp32 CBs).
//   CT: SEGS, CB_RK, CB_RG, CB_KEY, CB_BIAS, CB_SCALER, CB_T, CB_M, CB_E, CB_RS, CB_W, CB_P.  RT: 1 (a user) or 0.

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
template <uint32_t out_cb>
ALWI void pack_one(uint32_t at = 0, bool ordered = false) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(out_cb);
    if (ordered) {
        pack_tile<true>(0, out_cb, at);
    } else {
        pack_tile(0, out_cb);
    }
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t SEGS = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rk = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rg = get_compile_time_arg_val(2);
    constexpr uint32_t cb_key = get_compile_time_arg_val(3);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t cb_t = get_compile_time_arg_val(6);
    constexpr uint32_t cb_m = get_compile_time_arg_val(7);
    constexpr uint32_t cb_e = get_compile_time_arg_val(8);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(9);
    constexpr uint32_t cb_w = get_compile_time_arg_val(10);
    constexpr uint32_t cb_p = get_compile_time_arg_val(11);
    if (get_arg_val<uint32_t>(0) == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_rg, cb_bias, cb_t);
    cb_wait_front(cb_bias, 4);
    cb_wait_front(cb_scaler, 1);

    for (uint32_t s = 0; s < SEGS; ++s) {
        cb_wait_front(cb_rk, 4);
        cb_wait_front(cb_rg, 4);
        cb_reserve_back(cb_key, 4);
        for (uint32_t c = 0; c < 4; ++c) {
            // t = gate + bias
            cb_reserve_back(cb_t, 1);
            tile_regs_acquire();
            reconfig_data_format(cb_rg, cb_bias);
            add_init(cb_rg, cb_bias);
            add_tiles(cb_rg, cb_bias, c, c, 0);
            pack_one<cb_t>();
            cb_push_back(cb_t, 1);
            cb_wait_front(cb_t, 1);
            // m = max over the rows (row 0)
            cb_reserve_back(cb_m, 1);
            tile_regs_acquire();
            reconfig_data_format(cb_t, cb_scaler);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_t, cb_scaler, cb_m);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_t, cb_scaler, 0, 0, 0);
            reduce_uninit();
            pack_one<cb_m>();
            cb_push_back(cb_m, 1);
            cb_wait_front(cb_m, 1);
            // e = exp(t - m)
            cb_reserve_back(cb_e, 1);
            tile_regs_acquire();
            reconfig_data_format(cb_t, cb_m);
            sub_bcast_rows_init(cb_t, cb_m);
            sub_tiles_bcast_rows(cb_t, cb_m, 0, 0, 0);
            exp_tile_init<false>();
            exp_tile<false>(0);
            pack_one<cb_e>();
            cb_push_back(cb_e, 1);
            cb_pop_front(cb_t, 1);
            cb_pop_front(cb_m, 1);
            cb_wait_front(cb_e, 1);
            // 1 / sum over the rows
            cb_reserve_back(cb_rs, 1);
            tile_regs_acquire();
            reconfig_data_format(cb_e, cb_scaler);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_e, cb_scaler, cb_rs);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_e, cb_scaler, 0, 0, 0);
            reduce_uninit();
            recip_tile_init();
            recip_tile(0);
            pack_one<cb_rs>();
            cb_push_back(cb_rs, 1);
            cb_wait_front(cb_rs, 1);
            // w = e / sum (bf16)
            cb_reserve_back(cb_w, 1);
            tile_regs_acquire();
            reconfig_data_format(cb_e, cb_rs);
            mul_bcast_rows_init(cb_e, cb_rs);
            mul_tiles_bcast_rows(cb_e, cb_rs, 0, 0, 0);
            pack_one<cb_w>();
            cb_push_back(cb_w, 1);
            cb_pop_front(cb_e, 1);
            cb_pop_front(cb_rs, 1);
            cb_wait_front(cb_w, 1);
            // bf16(w rk)
            cb_reserve_back(cb_p, 1);
            tile_regs_acquire();
            reconfig_data_format(cb_w, cb_rk);
            mul_init(cb_w, cb_rk);
            mul_tiles(cb_w, cb_rk, 0, c, 0);
            pack_one<cb_p>();
            cb_push_back(cb_p, 1);
            cb_pop_front(cb_w, 1);
            cb_wait_front(cb_p, 1);
            // the key: sum over the rows (row 0)
            tile_regs_acquire();
            reconfig_data_format(cb_p, cb_scaler);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_p, cb_scaler, cb_key);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_p, cb_scaler, 0, 0, 0);
            reduce_uninit();
            pack_one<cb_key>(c, true);
            cb_pop_front(cb_p, 1);
        }
        cb_push_back(cb_key, 4);
        cb_pop_front(cb_rk, 4);
        cb_pop_front(cb_rg, 4);
    }
}
