// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// EXPERIMENT (hkato, optional, default off; TT_MMRS_RING=1): compute kernel of the matmul + ring reduce-scatter +
// addcmul that runs entirely on the matmul cores. At ring step k the core computes its M rows x N_block_tiles
// columns of slice s(k) (one K loop per M block). Steps k > 0 add the partial the neighbour device computed for the
// same tiles at step k-1 (recv CB); steps k < ring_size - 1 hand the sum to the relay (send CB); the last step
// writes out = a + scalar * (mm + recv) * b (final CB).

#include <cstdint>
#include <algorithm>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

namespace {

// The kernel's math fidelity is the matmul's (LoFi in Wan); the gate multiply needs full precision, as in the
// reduce-scatter's own compute kernel (HiFi4): LoFi keeps 5 significant bits of the partial sum.
ALWI void hk_mul_bcast_rows_init_hifi4(uint32_t icb0, uint32_t icb1) {
    state_configure(icb0, icb1, __builtin_LINE());
    MATH(
        (llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, MathFidelity::HiFi4>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0, icb1)));
}

ALWI void hk_mul_tiles_bcast_rows_hifi4(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MathFidelity::HiFi4,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1)));
}

// Same as minimal_matmul's compute.cpp
void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb,
                    in1_cb,
                    in0_index,
                    in1_index,
                    dst_index,
                    false /*transpose*/,
                    subblock_w,
                    subblock_h,
                    K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile_id = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_start + w;
                    uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                    dst_index++;
                }
            }
            tile_regs_release();

            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t ring_size = get_compile_time_arg_val(7);
    constexpr bool scalar_is_one = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t dst_batch = get_compile_time_arg_val(9);  // tiles per DST half (8 bf16 / 4 fp32)

    const uint32_t M_start_tile = get_arg_val<uint32_t>(0);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(1);   // this core's (padded) row range end
    const uint32_t M_valid_end = get_arg_val<uint32_t>(2);  // min(M_end_tile, M_tiles)
    const uint32_t scalar_bits = get_arg_val<uint32_t>(3);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t send_cb = tt::CBIndex::c_2;
    constexpr uint32_t interm_cb = tt::CBIndex::c_3;
    constexpr uint32_t final_cb = tt::CBIndex::c_4;
    constexpr uint32_t a_cb = tt::CBIndex::c_5;
    constexpr uint32_t b_cb = tt::CBIndex::c_6;
    constexpr uint32_t recv_cb = tt::CBIndex::c_7;
#ifdef HK_RING_BIAS
    constexpr uint32_t bias_cb = tt::CBIndex::c_10;
#endif

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t block_tiles = M_block_tiles * N_block_tiles;

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, interm_cb);
    matmul_init(in0_cb, in1_cb);

    for (uint32_t k = 0; k < ring_size; k++) {
        const bool last_step = k == ring_size - 1;
        if (last_step) {
            // final_cb aliases send_cb's L1: wait until every block handed to the relay has been released
            cb_wait_front(b_cb, N_block_tiles);
#ifdef HK_RING_BIAS
            cb_wait_front(bias_cb, N_block_tiles);
#endif
        }
        for (uint32_t mb = 0; mb < M_blocks_per_core; mb++) {
            const uint32_t m_tile = M_start_tile + mb * M_block_tiles;
            const uint32_t cur_rows = std::min(M_block_tiles, M_end_tile - m_tile);
            const uint32_t cur_sub_h = std::min(cur_rows, subblock_h);
            const uint32_t valid_rows = m_tile >= M_valid_end ? 0 : std::min(M_block_tiles, M_valid_end - m_tile);

            // Reconfig before init: the previous block's epilogue left other operands configured (see #55052)
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(interm_cb);
            matmul_block_init(in0_cb, in1_cb, false /*transpose*/, subblock_w, cur_sub_h, K_block_tiles);
            cb_reserve_back(interm_cb, block_tiles);
            for (uint32_t kb = 0; kb < K_num_blocks; kb++) {
                cb_wait_front(in0_cb, in0_block_num_tiles);
                cb_wait_front(in1_cb, in1_block_num_tiles);
                matmul_blocks(
                    in0_cb,
                    in1_cb,
                    interm_cb,
                    cur_rows,
                    N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    cur_sub_h,
                    subblock_w);
                cb_pop_front(in0_cb, in0_block_num_tiles);
                cb_pop_front(in1_cb, in1_block_num_tiles);
                if (kb == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            cb_push_back(interm_cb, block_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));
            cb_wait_front(interm_cb, block_tiles);

            if (!last_step) {
                // send = mm (+ recv), the whole block (rows past valid_rows are padding the relay skips)
                cb_reserve_back(send_cb, block_tiles);
                if (k == 0) {
                    reconfig_data_format_srca(interm_cb);
                    copy_init(interm_cb);
                } else {
                    cb_wait_front(recv_cb, block_tiles);
                    reconfig_data_format(interm_cb, recv_cb);
                    add_init(interm_cb, recv_cb);
                }
                pack_reconfig_data_format(send_cb);
                for (uint32_t t = 0; t < block_tiles; t += dst_batch) {
                    const uint32_t n = std::min(dst_batch, block_tiles - t);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < n; i++) {
                        if (k == 0) {
                            copy_tile(interm_cb, t + i, i);
                        } else {
                            add_tiles(interm_cb, recv_cb, t + i, t + i, i);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < n; i++) {
                        pack_tile(i, send_cb);
                    }
                    tile_regs_release();
                }
                cb_push_back(send_cb, block_tiles);
                if (k > 0) {
                    cb_pop_front(recv_cb, block_tiles);
                }
                cb_pop_front(interm_cb, block_tiles);
            } else {
                const uint32_t valid_tiles = valid_rows * N_block_tiles;
                // 1) interm += recv, in place
                cb_wait_front(recv_cb, block_tiles);
                reconfig_data_format(interm_cb, recv_cb);
                pack_reconfig_data_format(interm_cb);
                add_init(interm_cb, recv_cb);
                for (uint32_t t = 0; t < valid_tiles; t += dst_batch) {
                    const uint32_t n = std::min(dst_batch, valid_tiles - t);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < n; i++) {
                        add_tiles(interm_cb, recv_cb, t + i, t + i, i);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < n; i++) {
                        pack_tile<true>(i, interm_cb, t + i);
                    }
                    tile_regs_release();
                }
                cb_pop_front(recv_cb, block_tiles);
                // make the packer's in-place writes visible to the unpacker
                cb_pop_front(interm_cb, block_tiles);
                cb_reserve_back(interm_cb, block_tiles);
                cb_push_back(interm_cb, block_tiles);
                cb_wait_front(interm_cb, block_tiles);
#ifdef HK_RING_BIAS
                // 1b) interm += bias (row broadcast), in place
                reconfig_data_format(interm_cb, bias_cb);
                pack_reconfig_data_format(interm_cb);
                add_bcast_rows_init(interm_cb, bias_cb);
                for (uint32_t m = 0; m < valid_rows; m++) {
                    for (uint32_t n0 = 0; n0 < N_block_tiles; n0 += dst_batch) {
                        const uint32_t n1 = std::min(N_block_tiles, n0 + dst_batch);
                        tile_regs_acquire();
                        for (uint32_t n = n0; n < n1; n++) {
                            add_tiles_bcast<BroadcastType::ROW>(interm_cb, bias_cb, m * N_block_tiles + n, n, n - n0);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t n = n0; n < n1; n++) {
                            pack_tile<true>(n - n0, interm_cb, m * N_block_tiles + n);
                        }
                        tile_regs_release();
                    }
                }
                cb_pop_front(interm_cb, block_tiles);
                cb_reserve_back(interm_cb, block_tiles);
                cb_push_back(interm_cb, block_tiles);
                cb_wait_front(interm_cb, block_tiles);
#endif

                // 2) interm *= b (row broadcast) * scalar, in place
                reconfig_data_format(interm_cb, b_cb);
                pack_reconfig_data_format(interm_cb);
                hk_mul_bcast_rows_init_hifi4(interm_cb, b_cb);
                if constexpr (!scalar_is_one) {
                    binop_with_scalar_tile_init();
                }
                for (uint32_t m = 0; m < valid_rows; m++) {
                    for (uint32_t n0 = 0; n0 < N_block_tiles; n0 += dst_batch) {
                        const uint32_t n1 = std::min(N_block_tiles, n0 + dst_batch);
                        tile_regs_acquire();
                        for (uint32_t n = n0; n < n1; n++) {
                            hk_mul_tiles_bcast_rows_hifi4(interm_cb, b_cb, m * N_block_tiles + n, n, n - n0);
                            if constexpr (!scalar_is_one) {
                                mul_unary_tile(n - n0, scalar_bits);
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t n = n0; n < n1; n++) {
                            pack_tile<true>(n - n0, interm_cb, m * N_block_tiles + n);
                        }
                        tile_regs_release();
                    }
                }
                cb_pop_front(interm_cb, block_tiles);
                cb_reserve_back(interm_cb, block_tiles);
                cb_push_back(interm_cb, block_tiles);
                cb_wait_front(interm_cb, block_tiles);

                // 3) out = interm + a, one row at a time
                reconfig_data_format(interm_cb, a_cb);
                pack_reconfig_data_format(final_cb);
                add_init(interm_cb, a_cb);
                for (uint32_t m = 0; m < valid_rows; m++) {
                    cb_wait_front(a_cb, N_block_tiles);
                    cb_reserve_back(final_cb, N_block_tiles);
                    for (uint32_t n0 = 0; n0 < N_block_tiles; n0 += dst_batch) {
                        const uint32_t n1 = std::min(N_block_tiles, n0 + dst_batch);
                        tile_regs_acquire();
                        for (uint32_t n = n0; n < n1; n++) {
                            add_tiles(interm_cb, a_cb, m * N_block_tiles + n, n, n - n0);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t n = n0; n < n1; n++) {
                            pack_tile(n - n0, final_cb);
                        }
                        tile_regs_release();
                    }
                    cb_push_back(final_cb, N_block_tiles);
                    cb_pop_front(a_cb, N_block_tiles);
                }
                cb_pop_front(interm_cb, block_tiles);
            }
        }
    }
    cb_pop_front(b_cb, N_block_tiles);
#ifdef HK_RING_BIAS
    cb_pop_front(bias_cb, N_block_tiles);
#endif
}
