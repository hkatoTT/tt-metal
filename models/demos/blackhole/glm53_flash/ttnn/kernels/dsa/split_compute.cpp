// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DSA decode's sparse attention split over cores (dsa_indexer.sparse_split): the compute, ttnn's sparse_sdpa compute
// (flash / online softmax over the key chunks) for one unit (token, split): its share of the chunks, then instead of
// normalizing, the partial (split_common.hpp): the unnormalized out tiles, the row sums (the column-partial sums
// times the column identity) and the row maxima, for the combine (combine_*.cpp).  bf16 latent K, one query
// tile row (H = 32; the split latents' 64-head unit H = 64: two, a partial each), no scaled cache, no sink.
//   CT: split_ct below.  RT: the core's count of units (each in turn; 0: idle).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"  // add_tiles_bcast_rows (mask add)
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
namespace split_ct {
enum : uint32_t {
    H = 0, DHT, V_DHT, SKT, SCALE, CB_Q_RM, CB_Q_IN, CB_K_RM, CB_K_IN, CB_NEGINF, CB_MASK_PART, CB_SCALE, CB_QK_IM,
    CB_MAX_A, CB_MAX_B, CB_SUM_A, CB_SUM_B, CB_OUT_A, CB_OUT_B, CB_CORR, CB_PART, CB_CTRL, CB_COL_IDENTITY,
    MATH_APPROX_MODE, QUERY_SUBBLOCK, Q_TILED, END
};
}
constexpr bool EXP_APPROX_MODE = get_compile_time_arg_val(split_ct::MATH_APPROX_MODE) != 0;
// compute_streaming.hpp needs declarations from compute_common.hpp (LightweightMaskContext, reduce helpers,
// DEST_AUTO_LIMIT); include it first. Only compute_streaming primitives are used.
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#include "split_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/dataflow/circular_buffer.h"  // CircularBuffer: COMPILE_FOR_TRISC-aware CB lifecycle
#include <tt-metalium/constants.hpp>       // tt::constants::TILE_HEIGHT

// In-place scores[tile] += row-broadcast of mask row 0. cb_qk_im is held (hold_wr_ptr), so re-pack at the
// absolute position. Caller has init'd the bcast-rows op + single-tile pack.
ALWI void add_bcast_row_mask_tile(uint32_t scores_cb, uint32_t mask_cb, uint32_t score_tile) {
    tile_regs_acquire();
    add_tiles_bcast_rows(scores_cb, mask_cb, score_tile, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, scores_cb, score_tile);
    tile_regs_release();
}

// Make in-place PACK writes to a held CB visible to the next UNPACK read (after the in-place mask/sub_exp
// writes to cb_qk_im and the V-matmul writes to out_cur, none of which do a cb_push_back).
ALWI void pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

ALWI void swap_cb(CircularBuffer& a, CircularBuffer& b) {
    const CircularBuffer t = a;
    a = b;
    b = t;
}

// The unit's partials into cb_part (split_common.hpp), one a query tile row (Sqt: 1, or 2 for a 64-head unit): the out
// tiles as they are (unnormalized), the row sums (the column-partial sums times the column identity: column 0) and the
// row maxima.  Pops the sums, the outs and the maxima.
template <uint32_t Sqt, uint32_t vDHt, uint32_t dst_size, uint32_t cb_col_identity, uint32_t cb_part>
ALWI void pack_partial(uint32_t sum_cb, uint32_t out_cb, uint32_t max_cb) {
    CircularBuffer part(cb_part), sum(sum_cb), out(out_cb), mx(max_cb), ident(cb_col_identity);
    out.wait_front(Sqt * vDHt);
    ident.wait_front(1);
    sum.wait_front(Sqt);
    mx.wait_front(Sqt);
    constexpr uint32_t batch = vDHt < dst_size ? vDHt : dst_size;
    static_assert(vDHt % batch == 0);
    for (uint32_t r = 0; r < Sqt; ++r) {
        part.reserve_back(vDHt + 2);
        reconfig_data_format_srca(out_cb);
        pack_reconfig_data_format(cb_part);
        copy_tile_to_dst_init_short(out_cb);
        configure_single_tile_pack(cb_part);
        for (uint32_t base = 0; base < vDHt; base += batch) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < batch; ++j) {
                copy_tile(out_cb, r * vDHt + base + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < batch; ++j) {
                pack_tile<true>(j, cb_part, base + j);
            }
            tile_regs_release();
        }
        matmul_block_init(sum_cb, cb_col_identity, 0, 1, 1, 1);
        reconfig_data_format(cb_col_identity, sum_cb);
        tile_regs_acquire();
        matmul_block(sum_cb, cb_col_identity, r, 0, 0, 0, 1, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(cb_part);
        pack_tile<true>(0, cb_part, vDHt);
        tile_regs_release();
        reconfig_data_format_srca(max_cb);
        copy_tile_to_dst_init_short(max_cb);
        tile_regs_acquire();
        copy_tile(max_cb, r, 0);
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(cb_part);
        pack_tile<true>(0, cb_part, vDHt + 1);
        tile_regs_release();
        part.push_back(vDHt + 2);
    }
    sum.pop_front(Sqt);
    out.pop_front(Sqt * vDHt);
    mx.pop_front(Sqt);
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(split_ct::H);
    constexpr uint32_t DHt = get_compile_time_arg_val(split_ct::DHT);
    constexpr uint32_t vDHt = get_compile_time_arg_val(split_ct::V_DHT);
    constexpr uint32_t Skt = get_compile_time_arg_val(split_ct::SKT);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(split_ct::SCALE);

    // CB ids from the factory (the SparseCB enum is the single source); order matches the factory's compute
    // compile-arg block (5..24). They feed the templates + format helpers; the CB objects below wrap them.
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(split_ct::CB_Q_RM);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(split_ct::CB_Q_IN);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(split_ct::CB_K_RM);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(split_ct::CB_K_IN);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(split_ct::CB_NEGINF);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(split_ct::CB_MASK_PART);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(split_ct::CB_SCALE);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(split_ct::CB_QK_IM);
    constexpr uint32_t cb_max_a = get_compile_time_arg_val(split_ct::CB_MAX_A);
    constexpr uint32_t cb_max_b = get_compile_time_arg_val(split_ct::CB_MAX_B);
    constexpr uint32_t cb_sum_a = get_compile_time_arg_val(split_ct::CB_SUM_A);
    constexpr uint32_t cb_sum_b = get_compile_time_arg_val(split_ct::CB_SUM_B);
    constexpr uint32_t cb_out_a = get_compile_time_arg_val(split_ct::CB_OUT_A);
    constexpr uint32_t cb_out_b = get_compile_time_arg_val(split_ct::CB_OUT_B);
    constexpr uint32_t cb_corr = get_compile_time_arg_val(split_ct::CB_CORR);
    constexpr uint32_t cb_part = get_compile_time_arg_val(split_ct::CB_PART);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(split_ct::CB_CTRL);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(split_ct::CB_COL_IDENTITY);

    constexpr uint32_t qsb =
        get_compile_time_arg_val(split_ct::QUERY_SUBBLOCK);  // query tile-rows per DST group
    constexpr uint32_t Sqt = H / tt::constants::TILE_HEIGHT;  // total query tile-rows (32 heads each)
    constexpr bool q_tiled = get_compile_time_arg_val(split_ct::Q_TILED) != 0;
    constexpr uint32_t q_groups = Sqt / qsb;                  // DST-bound work runs in this many query-row passes
    constexpr uint32_t KT_stride = Skt;                       // cb_qk_im physical row width
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    // sub_exp packs qsb*sbw tiles into DST: full Skt width when one group fits, else one key-tile column at a time.
    constexpr uint32_t exp_sbw = (qsb * Skt <= dst_size) ? Skt : 1;

    // CB wrappers for the fixed (non-ping-pong) buffers' lifecycle verbs.
    CircularBuffer q_in_cb(cb_q_in), k_in_cb(cb_k_in), qk_cb(cb_qk_im), scale_cb(cb_scale), ctrl_cb(cb_ctrl);
    CircularBuffer neginf_cb(cb_neginf), mask_part_cb(cb_mask_part), corr_cb(cb_corr);
    // one query tile row (H = 32), or the split latents' 64-head unit (H = 64: a query's two tile rows over the keys
    // read once)
    static_assert((Sqt == 1 || Sqt == 2) && qsb == 1, "split sparse attention: H = 32 or 64");

    const uint32_t n_units = get_arg_val<uint32_t>(0);  // this core's units, each in turn (0: an idle core)
    if (n_units == 0) {
        return;
    }

    // Q tilize is the first operation, so startup must configure raw-Q -> tiled-Q.
    compute_kernel_hw_startup(cb_q_rm, cb_q_in);

    scale_cb.wait_front(1);  // persistent reduce scaler; the streaming reduce assumes it is ready

    for (uint32_t unit = 0; unit < n_units; ++unit) {
        // tilize Q rows -> [Sqt, DHt]
        if constexpr (!q_tiled) {  // (tiled: the reader filled cb_q_in)
            compute_kernel_lib::tilize<DHt, cb_q_rm, cb_q_in>(/*num_blocks=*/Sqt, /*total_input_pages=*/H);
        }

        // Per-token control from the reader: num_active_chunks (>=1; all-sentinel chunks are skipped) and
        // num_valid_keys (last chunk's mask boundary). read_tile_value mailbox-distributes the UNPACK read so
        // all three threads see the same loop bound / mask branches.
        ctrl_cb.wait_front(1);
        const uint32_t num_active_chunks = ckernel::read_tile_value(
            cb_ctrl, /*tile=*/0, /*element_offset=*/sparse_sdpa::control_message::ACTIVE_CHUNKS);
        const uint32_t num_valid_keys =
            ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/sparse_sdpa::control_message::VALID_KEYS);
        ctrl_cb.pop_front(1);
        if (num_active_chunks == 0) {  // an empty share (fewer chunks than splits): no partial
            q_in_cb.wait_front(Sqt * DHt);
            q_in_cb.pop_front(Sqt * DHt);
            continue;
        }

        // Flash running state, ping-pong. Reset every token; all buffers start empty.
        CircularBuffer max_prev(cb_max_a), max_cur(cb_max_b);
        CircularBuffer sum_prev(cb_sum_a), sum_cur(cb_sum_b);
        CircularBuffer out_prev(cb_out_a), out_cur(cb_out_b);

        for (uint32_t chunk = 0; chunk < num_active_chunks; ++chunk) {
            // tilize K rows -> [Skt, DHt] (waits on reader cb_k_rm -> absorbs the K-read stall)
            compute_kernel_lib::tilize<DHt, cb_k_rm, cb_k_in>(
                /*num_blocks=*/Skt, /*total_input_pages=*/Skt * tt::constants::TILE_HEIGHT);
            // fp8 K tilize leaves srcA in fp8. QK reads K (transposed -> srcA) and Q (srcB), so restore
            // srcA=cb_k_in (bfp8 for fp8 K), srcB=cb_q_in. Reconfigure the tiled descriptor geometry/strides;
            // mm_no_mop_init_short only programs the matmul MOP.
            reconfig_full_operand(cb_k_in, cb_q_in);
            // K tilize also leaves the packer in cb_k_in's format+strides (bfp8 for fp8). Restore bf16 once per
            // chunk for the downstream packs (cb_qk_im/max/sum/out share its geometry); configure_pack_width in
            // the qg loop refreshes only the MOP. No-op for bf16.
            pack_reconfig_data_format(cb_qk_im);

            const bool is_first = (chunk == 0);
            const bool is_last = (chunk == num_active_chunks - 1);
            const bool has_mask = is_last;  // only the last active chunk can hold sentinels

            // cb_qk_im / sum_cur / out_cur span all Sqt rows; the qg loop fills them band-by-band (sub_exp & PV
            // pack at absolute row offsets; reduce/corr reserve+push per band).
            qk_cb.reserve_back(Sqt * KT_stride);
            sum_cur.reserve_back(Sqt);
            out_cur.reserve_back(Sqt * vDHt);
            k_in_cb.wait_front(Skt * DHt);  // shared by every query group (QK + PV)
            q_in_cb.wait_front(Sqt * DHt);

            // Boundary geometry (last chunk, same for every row): keys [valid_last, k_chunk) are sentinels ->
            // -inf. Key tiles [full_start, Skt) are fully masked (cb_neginf); a mid-tile boundary masks the
            // straddling tile part_t via cb_mask_part.
            const uint32_t k_chunk = Skt * tt::constants::TILE_WIDTH;
            const uint32_t valid_last = num_valid_keys - (num_active_chunks - 1) * k_chunk;
            const uint32_t full_start = (valid_last + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
            const uint32_t part_t = valid_last / tt::constants::TILE_WIDTH;
            const bool has_part_mask = has_mask && (valid_last % tt::constants::TILE_WIDTH != 0);

            // DST holds qsb query tile-rows; process Sqt rows in q_groups passes (one pass when qsb==Sqt).
            for (uint32_t qg = 0; qg < q_groups; ++qg) {
                const uint32_t row_base = qg * qsb;  // first query tile-row of this group
                // a later group's QK (the split latents' 64-head unit: the second tile row): the previous group's PV
                // and mask left srcA / srcB in bf16 (cb_qk_im, cb_scale), so restore K's (bfp8 for an fp8 latent cache)
                // and Q's as before the first group (bf16 K: the same formats, the same values)
                if (qg > 0) {
                    reconfig_full_operand(cb_k_in, cb_q_in);
                }

                // Set exp to the softmax scale; salad's correction below re-inits it to unit scale.
                exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();

                // ===== Phase 1: Q@Kᵀ -> cb_qk_im band, mask, running row-max =====
                {
                    // scores[q,sk]=ΣQ·K. in1=K, transpose=true (within-tile transpose => Kᵀ).
                    mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, qsb, DHt);
                    configure_row_pack_width(cb_qk_im, 1);
                    for (uint32_t kt = 0; kt < Skt; ++kt) {
                        blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                            cb_q_in,
                            cb_k_in,
                            cb_qk_im,
                            /*in0_index_start=*/row_base * DHt,
                            /*in1_index_start=*/kt * DHt,
                            /*row_subblock_idx=*/qg,
                            /*out_col_offset=*/kt,
                            /*subblock_w=*/1,
                            /*subblock_h=*/qsb,
                            /*inner_dim=*/DHt,
                            /*matmul_stride=*/DHt,
                            /*skip_pack_configure=*/true);
                    }
                    // Publish the band to UNPACK while holding wr_ptr (for the in-place mask/sub_exp).
                    cb_push_back_hold_wr_ptr(cb_qk_im, qsb * KT_stride);
                }

                {
                    // QK left srcA=cb_k_in, srcB=cb_q_in. The mask/reduce/sub_exp read cb_qk_im (srcA) + bf16
                    // scalers (srcB); restore both to bf16. No-op for bf16 Q/K.
                    reconfig_data_format(cb_qk_im, cb_scale);
                    if (has_mask) {
                        qk_cb.wait_front((qg + 1) * qsb * KT_stride);  // band visible before the in-place mask
                        // Same mask for every query row: stamp each masked key tile across this group's rows
                        // (cb_qk_im row row_base+r, key tile t = (row_base+r)*Skt + t).
                        if (full_start < Skt) {
                            neginf_cb.wait_front(1);
                            add_bcast_rows_init(cb_qk_im, cb_neginf);
                            configure_single_tile_pack(cb_qk_im);
                            for (uint32_t r = 0; r < qsb; ++r) {
                                for (uint32_t t = full_start; t < Skt; ++t) {
                                    add_bcast_row_mask_tile(cb_qk_im, cb_neginf, (row_base + r) * Skt + t);
                                }
                            }
                        }
                        if (has_part_mask) {
                            mask_part_cb.wait_front(1);  // persistent until popped after the group loop
                            add_bcast_rows_init(cb_qk_im, cb_mask_part);
                            configure_single_tile_pack(cb_qk_im);
                            for (uint32_t r = 0; r < qsb; ++r) {
                                add_bcast_row_mask_tile(cb_qk_im, cb_mask_part, (row_base + r) * Skt + part_t);
                            }
                        }
                        pack_to_unpack_sync();  // mask PACK writes must be visible to the row-max UNPACK
                    }
                    // running row-max (MAX-only; eltwise-max against prev on chunk>0)
                    max_cur.reserve_back(qsb);
                    configure_single_tile_pack(max_cur.get_cb_id());
                    reduce_c_row_group<cb_qk_im, cb_scale, KT_stride>(
                        max_cur.get_cb_id(),
                        max_prev.get_cb_id(),
                        /*row_group_index=*/qg,
                        /*do_eltwise_max=*/!is_first,
                        qsb,
                        Skt);
                    max_cur.push_back(qsb);

                    // sub_exp in place: cb_qk_im = exp((cb_qk_im - max)*scale); partial row-sum -> sum_cur (L1-acc).
                    // Walk key-tile columns in exp_sbw steps (one step when the group fits DST); global_col_base
                    // drives the L1-accumulate across steps.
                    for (uint32_t kc = 0; kc < Skt; kc += exp_sbw) {
                        sub_exp_block_bcast_cols<false, scale_fp32>(
                            cb_qk_im,
                            max_cur.get_cb_id(),
                            sum_cur.get_cb_id(),
                            /*cols_in_row=*/KT_stride,
                            /*q_subblock=*/qg,
                            /*global_col_base=*/kc,
                            /*sbh=*/qsb,
                            /*sbw=*/exp_sbw);
                    }
                    pack_to_unpack_sync();  // sub_exp PACK writes must be visible to the V-matmul UNPACK
                }

                // ===== Phase 2: probs@V -> out_cur band =====
                {
                    qk_cb.wait_front((qg + 1) * qsb * KT_stride);
                    // out[q,vd]=Σprobs·K, V = the latent feature cols. PV reads V into srcA and
                    // probs (cb_qk_im) into srcB (operands swap); set srcA=cb_k_in, srcB=cb_qk_im.
                    reconfig_data_format(cb_k_in, cb_qk_im);
                    mm_no_mop_init_short(cb_qk_im, cb_k_in, /*transpose=*/false, 1, qsb, Skt);
                    configure_row_pack_width(out_cur.get_cb_id(), 1);
                    for (uint32_t vd = 0; vd < vDHt; ++vd) {
                        blocked_matmul_and_pack<false, /*in1_stride=*/DHt, /*out_num_cols=*/vDHt>(
                            cb_qk_im,
                            cb_k_in,
                            out_cur.get_cb_id(),
                            /*in0_index_start=*/row_base * Skt,
                            /*in1_index_start=*/vd,
                            /*row_subblock_idx=*/qg,
                            /*out_col_offset=*/vd,
                            /*subblock_w=*/1,
                            /*subblock_h=*/qsb,
                            /*inner_dim=*/Skt,
                            /*matmul_stride=*/KT_stride,
                            /*skip_pack_configure=*/true);
                    }
                    pack_to_unpack_sync();                // flush PV's HELD out_cur packs (push_back deferred to
                                                          // after the group loop) so SALAD's L1-accumulate (!is_first)
                                                          // reads them; normalize reads post-push_back, so covered
                    reconfig_data_format_srca(cb_qk_im);  // PV left srcA in cb_k_in's format; restore bf16
                }

                // ===== SALAD flash combine (skip on the first chunk) =====
                if (!is_first) {
                    // correction = exp((prev_max - cur_max)*scale) -> cb_corr col 0. Re-init exp at unit scale
                    // (sub_exp_first_col_blocks applies scale itself; the group's init baked scale_fp32).
                    exp_packthread_tile_init<EXP_APPROX_MODE>();
                    corr_cb.reserve_back(qsb);
                    sub_exp_first_col_blocks<false, scale_fp32>(
                        max_prev.get_cb_id(), max_cur.get_cb_id(), cb_corr, /*q_subblock=*/qg, qsb);
                    corr_cb.push_back(qsb);
                    // cur.out += prev.out*corr ; cur.sum += prev.sum*corr (L1-acc). salad packs width=dst_size,
                    // so restore Default packer geometry first (the per-chunk tilize left it in Tilize layout).
                    PACK((
                        llk_pack_init<ckernel::PackMode::Default, false, false, false>(out_cur.get_cb_id(), dst_size)));
                    pack_reconfig_l1_acc(1);
                    // out_prev & cb_corr consumed front-first per group (popped below); sum_prev cumulative (qg).
                    salad_correct_fused<qsb, vDHt, dst_size>(
                        out_prev.get_cb_id(),
                        sum_prev.get_cb_id(),
                        cb_corr,
                        out_cur.get_cb_id(),
                        sum_cur.get_cb_id(),
                        /*ob_q_subblock=*/0,
                        /*sum_q_subblock=*/qg,
                        /*write_q_subblock=*/qg);
                    pack_reconfig_l1_acc(0);
                    corr_cb.pop_front(qsb);
                    out_prev.pop_front(qsb * vDHt);  // this band's prev.out consumed into cur
                }
            }  // query groups

            // Pop cb_mask_part once: every group bcast-added the same straddling-tile mask.
            if (has_part_mask) {
                mask_part_cb.pop_front(1);
            }
            // prev max/sum consumed into cur (out_prev popped per band above).
            if (!is_first) {
                max_prev.pop_front(Sqt);
                sum_prev.pop_front(Sqt);
            }

            // Publish cur.sum / cur.out (running state for the next chunk, or the final result).
            sum_cur.push_back(Sqt);
            out_cur.push_back(Sqt * vDHt);

            if (is_last) {  // the partial for the combine instead of the normalized out
                pack_partial<Sqt, vDHt, dst_size, cb_col_identity, cb_part>(
                    sum_cur.get_cb_id(), out_cur.get_cb_id(), max_cur.get_cb_id());
            }

            // Release the held cb_qk_im rows + this chunk's K (consumed by QK and PV).
            qk_cb.pop_front(Sqt * KT_stride);
            k_in_cb.pop_front(Skt * DHt);

            swap_cb(max_prev, max_cur);  // prev <-> cur for the next chunk
            swap_cb(sum_prev, sum_cur);
            swap_cb(out_prev, out_cur);
        }

        q_in_cb.pop_front(Sqt * DHt);
    }
}
