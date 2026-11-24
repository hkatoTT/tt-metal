// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DSA indexer (dsa_indexer.py): per user u0 .. u1 - 1, its row of token rows for sparse_sdpa from its top pools
// (topk_large_indices' row: pool ids, best first, 0xFFFFFFFF after the visible ones) and its position (params row u:
// [pos, base], pos 0xFFFFFFFF for a user with no query): base + 4 p + j for each pool p and j < kpool, then the tail
// (the visible tokens after the last complete pool: (pos + 1) - (pos + 1) mod kpool .. pos), then 0xFFFFFFFF to the
// row's end (sparse_sdpa stops at the first one, so they come last).  With PARTS > 1 a core writes part `part` of
// one user's row (pools [part k / PARTS, (part + 1) k / PARTS); the last part also the rest of the row): each part
// finds the visible pools itself (the sentinels are a contiguous suffix of the row: a binary search).
//   CT: k, width, kpool, params page, cb, PARTS, then the accessors; MAP, then (MAP) the candidates' accessor.
//   RT: pools, params, out addresses, u0, u1, part; (MAP) the candidates' address.
// MAP (the split scores: a long cache's decode indexer over the row's 4 chips): `pools` holds topk_large_indices' picks
// among the row's chips' candidates (dsa_indexer.shard_candidates, gathered: [1, 2, U, CHIPS k], plane 1 their
// columns): pick x is chip x / k's column c = plane 1's [u][x], the pool 32 (CHIPS (c / 32) + x / k) + c mod 32.
void kernel_main() {
    const uint32_t pools_addr = get_arg_val<uint32_t>(0);
    const uint32_t params_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t u1 = get_arg_val<uint32_t>(4);
    const uint32_t part = get_arg_val<uint32_t>(5);

    constexpr uint32_t k = get_compile_time_arg_val(0);       // pools a row
    constexpr uint32_t width = get_compile_time_arg_val(1);   // token rows a row of the output
    constexpr uint32_t kpool = get_compile_time_arg_val(2);   // tokens a pool
    constexpr uint32_t params_page = get_compile_time_arg_val(3);
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr uint32_t parts = get_compile_time_arg_val(5);
    constexpr auto p_args = TensorAccessorArgs<6>();
    constexpr auto q_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr uint32_t map = get_compile_time_arg_val(o_args.next_compile_time_args_offset());
    constexpr uint32_t chips = 4;
    constexpr auto c_args = TensorAccessorArgs<o_args.next_compile_time_args_offset() + 1>();
    // MUL, ADD (the split latents; 1, 0: every token): the latents of a user's tokens t with t mod MUL = ADD only,
    // token row r at r / MUL (rm_rows_write's): a pool's one row here (base + MUL p + ADD) / MUL, the tail's rows of
    // this residue
    constexpr uint32_t MUL = get_compile_time_arg_val(c_args.next_compile_time_args_offset());
    constexpr uint32_t ADD = get_compile_time_arg_val(c_args.next_compile_time_args_offset() + 1);
    static_assert(MUL == 1 || (MUL == kpool && parts == 1), "select_tokens: a sharded cache a pool's token a chip, one part");
    // PREFIX_Q (the gathered prefix; 0: off): rows of a user's prefix gathered over the row's chips
    // (dsa_indexer.prefix_rows, then all_gather: chip c's PREFIX_Q rows, its tokens t = c mod 4, at c PREFIX_Q): token
    // pos at (pos mod 4) PREFIX_Q + pos / 4
    constexpr uint32_t PREFIX_Q = get_compile_time_arg_val(c_args.next_compile_time_args_offset() + 2);
    static_assert(PREFIX_Q == 0 || (MUL == 1 && parts == 1), "select_tokens: the prefix's rows, one part");
    constexpr uint32_t SENT = 0xFFFFFFFFu;

    const auto s_pools = TensorAccessor(p_args, pools_addr, k * 4);
    const auto s_params = TensorAccessor(q_args, params_addr, params_page);
    const auto s_out = TensorAccessor(o_args, out_addr, width * 4);
    const uint32_t pools_l1 = get_write_ptr(cb);
    const uint32_t params_l1 = pools_l1 + k * 4;
    const uint32_t out_l1 = params_l1 + params_page;
    volatile tt_l1_ptr uint32_t* pools = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pools_l1);
    volatile tt_l1_ptr uint32_t* params = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(params_l1);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    const uint32_t cols_l1 = out_l1 + width * 4;
    for (uint32_t u = u0; u < u1; ++u) {
        noc_async_read(s_pools.get_noc_addr(u), pools_l1, k * 4);
        noc_async_read(s_params.get_noc_addr(u), params_l1, params_page);
        if constexpr (map) {
            const uint32_t n_users = get_arg_val<uint32_t>(7);
            const auto s_cand = TensorAccessor(c_args, get_arg_val<uint32_t>(6), chips * k * 2);
            noc_async_read(s_cand.get_noc_addr(n_users + u), cols_l1, chips * k * 2);  // plane 1: the columns
        }
        noc_async_read_barrier();
        if constexpr (map) {
            const uint16_t* cols = reinterpret_cast<const uint16_t*>(cols_l1);
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t x = pools[i];
                if (x != SENT) {
                    const uint32_t c = cols[x];
                    pools[i] = 32 * (chips * (c >> 5) + x / k) + (c & 31);
                }
            }
        }
        const uint32_t pos = params[0], base = params[1];
        if constexpr (parts > 1) {  // this part's slice of the row
            static_assert((k / parts) * kpool * 4 % 64 == 0, "a part's slice must start DRAM-aligned");
            uint32_t lo = 0, hi = k;  // the visible pools: the first sentinel
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (pools[mid] == SENT) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            const uint32_t n_pools = pos == SENT ? 0 : lo;
            const uint32_t visible = pos + 1, tail = pos == SENT ? 0 : visible % kpool;
            const uint32_t begin = part * (k / parts) * kpool;
            const uint32_t end = part == parts - 1 ? width : (part + 1) * (k / parts) * kpool;
            for (uint32_t n = begin; n < end; ++n) {
                uint32_t v = SENT;
                if (n < n_pools * kpool) {
                    v = base + pools[n / kpool] * kpool + n % kpool;
                } else if (n < n_pools * kpool + tail) {
                    v = base + visible - tail + (n - n_pools * kpool);
                }
                out[n - begin] = v;
            }
            noc_async_write(out_l1, s_out.get_noc_addr(u, begin * 4), (end - begin) * 4);
            noc_async_write_barrier();
            continue;
        }
        uint32_t n = 0;
        if (pos != SENT && MUL > 1) {  // (split latents: this chip's token of each pool, and of the tail)
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t p = pools[i];
                if (p == SENT) {
                    break;
                }
                out[n++] = (base + p * kpool + ADD) / MUL;
            }
            const uint32_t visible = pos + 1, tail = visible % kpool;
            for (uint32_t j = 0; j < tail; ++j) {
                const uint32_t r = base + visible - tail + j;
                if (r % MUL == ADD) {
                    out[n++] = r / MUL;
                }
            }
        } else if (pos != SENT && PREFIX_Q > 0) {  // (the gathered prefix's rows, in the unsplit cache's token order)
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t p = pools[i];
                if (p == SENT) {
                    break;
                }
                for (uint32_t j = 0; j < kpool; ++j) {
                    out[n++] = j * PREFIX_Q + p;  // token 4 p + j
                }
            }
            const uint32_t visible = pos + 1, tail = visible % kpool;
            for (uint32_t j = 0; j < tail; ++j) {
                const uint32_t t = visible - tail + j;
                out[n++] = (t % kpool) * PREFIX_Q + t / kpool;
            }
        } else if (pos != SENT) {
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t p = pools[i];
                if (p == SENT) {
                    break;
                }
                for (uint32_t j = 0; j < kpool; ++j) {
                    out[n++] = base + p * kpool + j;
                }
            }
            const uint32_t visible = pos + 1, tail = visible % kpool;
            for (uint32_t j = 0; j < tail; ++j) {
                out[n++] = base + visible - tail + j;
            }
        }
        for (; n < width; ++n) {
            out[n] = SENT;
        }
        noc_async_write(out_l1, s_out.get_noc_addr(u), width * 4);
        noc_async_write_barrier();
    }
}
