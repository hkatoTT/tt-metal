// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "scores_common.hpp"

// DSA indexer pool scores, the writer (scores_common.hpp): each unit's row 0 (faces 0 and 1, 16 bf16 each) gathered
// into the tile's first 64 bytes, the pools a query does not see (p >= (pos + 1) / 4) set to -inf, and written to
// row n, columns 32 j .. 32 j + 31 of out [1, 1, N, P] bf16 row-major; blocks jc .. jb - 1 (not computed) -inf.
void kernel_main() {
    using namespace dsa_scores;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t params_addr = get_arg_val<uint32_t>(1);
    const uint32_t na = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t stride = get_arg_val<uint32_t>(4);
    const uint32_t ja = get_arg_val<uint32_t>(5);
    const uint32_t jb = get_arg_val<uint32_t>(6);
    const uint32_t jc = get_arg_val<uint32_t>(7);

    constexpr bool shared = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);  // out's row: P bf16
    constexpr uint32_t kpool = get_compile_time_arg_val(2);
    constexpr bool visible_only = get_compile_time_arg_val(3) != 0;  // VISIBLE: the reader's blocks, the rest -inf
    constexpr auto o_args = TensorAccessorArgs<4>();
    constexpr auto p_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    // local block j is the pools' block j MUL + ADD (the reader's): its pools' visibility by their global numbers
    constexpr uint32_t MUL = get_compile_time_arg_val(p_args.next_compile_time_args_offset());
    constexpr uint32_t ADD = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t P_BYTES = PARAMS_WORDS * 4;
    constexpr uint32_t NEG_INF2 = 0xFF80FF80u;  // two bf16 -inf

    const auto s_out = TensorAccessor(o_args, out_addr, row_bytes);
    const auto s_p = TensorAccessor(p_args, params_addr, P_BYTES);
    if (count == 0 || jb <= ja) {
        return;
    }
    const uint32_t params_l1 = get_write_ptr(CB_WS);
    const uint32_t inf_l1 = params_l1 + count * P_BYTES;
    for (uint32_t i = 0; i < count; ++i) {
        noc_async_read(s_p.get_noc_addr(na + i * stride), params_l1 + i * P_BYTES, P_BYTES);
    }
    volatile tt_l1_ptr uint32_t* inf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inf_l1);
    for (uint32_t i = 0; i < 16; ++i) {
        inf[i] = NEG_INF2;
    }
    noc_async_read_barrier();
    auto pos_of = [&](uint32_t i) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(params_l1 + i * P_BYTES)[0]; };
    auto visible = [&](uint32_t i) {
        const uint32_t pos = pos_of(i);
        return pos == SENT ? 0u : (pos + 1) / kpool;
    };
    // the end of the computed blocks: jc, or with VISIBLE the reader's (the visible ones' of queries i0 .. i1 - 1)
    auto end_of = [&](uint32_t i0, uint32_t i1) {
        if constexpr (!visible_only) {
            return jc;
        }
        uint32_t jv = 0;
        for (uint32_t i = i0; i < i1; ++i) {
            const uint32_t v = local_blocks(visible_blocks(pos_of(i), kpool), MUL, ADD);
            jv = v > jv ? v : jv;
        }
        const uint32_t je = jv < jc ? jv : jc;
        return je > ja ? je : ja;
    };

    auto unit = [&](uint32_t n, uint32_t j, uint32_t vis) {
        cb_wait_front(CB_OUT, 1);
        const uint32_t t = get_read_ptr(CB_OUT);
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t);
        for (uint32_t i = 0; i < 8; ++i) {
            w[8 + i] = w[128 + i];  // face 1's row 0 after face 0's
        }
        const uint32_t first = (j * MUL + ADD) * 32;  // the block's first pool (global)
        if (vis < first + 32) {  // pools vis .. of this block unseen
            for (uint32_t p = (vis > first ? vis - first : 0); p < 32; ++p) {
                volatile tt_l1_ptr uint16_t* h = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(t);
                h[p] = 0xFF80;
            }
        }
        noc_async_write(t, s_out.get_noc_addr(n, j * 64), 64);
        noc_async_writes_flushed();
        cb_pop_front(CB_OUT, 1);
    };

    if constexpr (shared) {
        const uint32_t je = end_of(0, count);
        for (uint32_t j = ja; j < je; ++j) {
            for (uint32_t i = 0; i < count; ++i) {
                unit(na + i * stride, j, visible(i));
            }
        }
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t j = je; j < jb; ++j) {
                noc_async_write(inf_l1, s_out.get_noc_addr(na + i * stride, j * 64), 64);
            }
        }
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t vis = visible(i), je = end_of(i, i + 1);
            for (uint32_t j = ja; j < je; ++j) {
                unit(na + i * stride, j, vis);
            }
            for (uint32_t j = je; j < jb; ++j) {
                noc_async_write(inf_l1, s_out.get_noc_addr(na + i * stride, j * 64), 64);
            }
        }
    }
    noc_async_write_barrier();
}
