// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "rope_kv_items.hpp"

// DFlash2's RoPE with the block's K / V into the caches (dflash2_ops.rope_kv), the
// writer, per item of [i0, i1) in the compute's order:
//   q items (0 .. 4 H - 1): CB_O's tile to q at tile `item` (rope_writer.cpp's first output);
//   k items (4 H .. 8 H - 1: virtual user g = item / 4 - H = u heads + h, column tile t): CB_KV's BFP8_b tile, its
//   rows pos < B (token r = u B + pos) into the K cache;
//   v items (8 H ..: tile row R, head h, column tile t): CB_KV's tile, its rows rb (token r = 32 R + rb < tokens) into
//   the V cache.
// A token r's row goes where kv_write.cpp (mode 0) puts it: user u = r / B, position p = idx[u] + r % B (none: p < 0,
// p past the pages unless RING, then p mod pages x 32), cache page page_base + (u heads + h) pages + p / 32, row
// p % 32, column tile t (none past the last user either); per face row its 16 bytes of mantissas and its exponent byte,
// each staged at the destination's offset mod 64 in a 64-byte slot (a NoC write under 64 bytes: the same offset mod
// 64 at both ends).
// RKV_RUNS (GLM53_DFLASH2_ROPE_KV_RUNS=1): a user's rows that land on consecutive rows of one face of one page go as
// one run: per face column half its n x 16 bytes of mantissas and its n exponent bytes, each staged at the
// destination's offset mod 64 (the same bytes in the same places as the row-by-row writes, 4 writes a run in place
// of 4 n).  The source rows of a user never cross a face (B divides 16).
#ifndef RKV_RUNS
#define RKV_RUNS 0
#endif
void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t kc_addr = get_arg_val<uint32_t>(1);
    const uint32_t vc_addr = get_arg_val<uint32_t>(2);
    const uint32_t idx_addr = get_arg_val<uint32_t>(3);
    const uint32_t i0 = get_arg_val<uint32_t>(4);
    const uint32_t i1 = get_arg_val<uint32_t>(5);
    const uint32_t page_base = get_arg_val<uint32_t>(6);
    constexpr uint32_t CB_O = get_compile_time_arg_val(0);
    constexpr uint32_t CB_KV = get_compile_time_arg_val(1);
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr uint32_t heads = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);
    constexpr uint32_t n_idx = get_compile_time_arg_val(6);  // idx's entries (one a user)
    constexpr uint32_t tokens = get_compile_time_arg_val(7);
    constexpr uint32_t pages = get_compile_time_arg_val(8);  // a (user, head)'s pages
    constexpr uint32_t RING = get_compile_time_arg_val(9);
    constexpr uint32_t n_users = get_compile_time_arg_val(10);
    constexpr auto q_args = TensorAccessorArgs<11>();
    constexpr auto kc_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048, TILE = 1088, WT = 4, SLOT = 64, FACE = 256, FROW = 16, BASE = 64;
    constexpr uint32_t QITEMS = 4 * H, ROPE = 8 * H;
    constexpr uint32_t idx_bytes = n_idx * 4;
    constexpr uint32_t idx_slot = (idx_bytes + 63) & ~63u;

    const auto s_q = TensorAccessor(q_args, q_addr, BF);
    const auto s_kc = TensorAccessor(kc_args, kc_addr, TILE);
    const auto s_vc = TensorAccessor(vc_args, vc_addr, TILE);
    const uint32_t l1 = get_write_ptr(CB_SCR);
    const uint32_t stage0 = l1 + idx_slot;
    bool kv_here = false;
    for (uint32_t phys = i0; phys < i1; ++phys) {
        kv_here = kv_here || rkv_logical<ROPE>(phys) >= QITEMS;
    }
    if (kv_here) {  // (the positions only where there are K / V rows)
        noc_async_read(TensorAccessor(idx_args, idx_addr, idx_bytes).get_noc_addr(0), l1, idx_bytes);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr int32_t* idx = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1);

    for (uint32_t phys = i0; phys < i1; ++phys) {
        const uint32_t item = rkv_logical<ROPE>(phys);
        if (item < QITEMS) {
            cb_wait_front(CB_O, 1);
            noc_async_write(get_read_ptr(CB_O), s_q.get_noc_addr(item), BF);
            noc_async_writes_flushed();
            cb_pop_front(CB_O, 1);
            continue;
        }
        const bool is_k = item < ROPE;
        uint32_t t, h, r_first, rows;  // the tile's column tile and head, its first token and its rows
        uint32_t rb_first;             // the first token's row in the tile
        if (is_k) {
            const uint32_t g = item / WT - H, u = g / heads;
            t = item % WT;
            h = g % heads;
            r_first = u * B;
            rows = B;
            rb_first = 0;
        } else {
            const uint32_t vi = item - ROPE, R = vi / (WT * heads);
            t = vi % WT;
            h = (vi / WT) % heads;
            r_first = R * 32;
            rows = tokens > r_first ? (tokens - r_first < 32 ? tokens - r_first : 32) : 0;
            rb_first = 0;
        }
        cb_wait_front(CB_KV, 1);
        const uint32_t src = get_read_ptr(CB_KV);
        if constexpr (RKV_RUNS != 0) {
            constexpr uint32_t RSLOT = 256, MSLOT = 192;  // a run's stage a face column half: mantissas, then exponents
            uint32_t nrun = 0;
            uint32_t j = 0;
            while (j < rows) {
                const uint32_t u = (r_first + j) / B;
                const uint32_t jend = (u + 1) * B - r_first < rows ? (u + 1) * B - r_first : rows;
                const int32_t p0 = u < n_idx ? idx[u] : -1;
                if (p0 < 0 || u >= n_users) {
                    j = jend;
                    continue;
                }
                while (j < jend) {
                    const int32_t pj = p0 + static_cast<int32_t>((r_first + j) % B);
                    if (RING == 0 && pj >= static_cast<int32_t>(pages * 32)) {
                        j = jend;  // (its user's later rows are past the pages too)
                        break;
                    }
                    const uint32_t pos = RING != 0 ? static_cast<uint32_t>(pj) % (pages * 32) : static_cast<uint32_t>(pj);
                    const uint32_t page = page_base + (u * heads + h) * pages + pos / 32, row = pos % 32;
                    uint32_t n = 1;
                    while (j + n < jend && (row + n) % 16 != 0) {
                        const int32_t pn = p0 + static_cast<int32_t>((r_first + j + n) % B);
                        if (RING == 0 && pn >= static_cast<int32_t>(pages * 32)) {
                            break;
                        }
                        const uint32_t posn =
                            RING != 0 ? static_cast<uint32_t>(pn) % (pages * 32) : static_cast<uint32_t>(pn);
                        if (posn != pos + n) {
                            break;
                        }
                        ++n;
                    }
                    const uint32_t rb = rb_first + j;
                    for (uint32_t f = 0; f < 2; ++f) {
                        const uint32_t src_off = BASE + ((rb / 16) * 2 + f) * FACE + (rb % 16) * FROW;
                        const uint32_t dst_off = BASE + ((row / 16) * 2 + f) * FACE + (row % 16) * FROW;
                        const uint32_t slot = stage0 + (nrun * 2 + f) * RSLOT;
                        const uint32_t dst = slot + dst_off % SLOT;
                        volatile tt_l1_ptr uint32_t* from = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + src_off);
                        volatile tt_l1_ptr uint32_t* to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                        for (uint32_t w = 0; w < n * FROW / 4; ++w) {
                            to[w] = from[w];
                        }
                        const uint64_t tile_addr = is_k ? s_kc.get_noc_addr(page * WT + t, dst_off)
                                                        : s_vc.get_noc_addr(page * WT + t, dst_off);
                        noc_async_write(dst, tile_addr, n * FROW);
                        const uint32_t es = ((rb / 16) * 2 + f) * 16 + rb % 16;
                        const uint32_t ed = ((row / 16) * 2 + f) * 16 + row % 16;
                        const uint32_t edst = slot + MSLOT + ed % SLOT;
                        for (uint32_t b = 0; b < n; ++b) {
                            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(edst)[b] =
                                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src + es)[b];
                        }
                        const uint64_t exp_addr =
                            is_k ? s_kc.get_noc_addr(page * WT + t, ed) : s_vc.get_noc_addr(page * WT + t, ed);
                        noc_async_write(edst, exp_addr, n);
                    }
                    ++nrun;
                    j += n;
                }
            }
            noc_async_writes_flushed();  // (the staged runs read out before the slots and the tile are reused)
            cb_pop_front(CB_KV, 1);
            continue;
        }
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t r = r_first + j, rb = rb_first + j;
            const uint32_t u = r / B;
            int32_t p = u < n_idx ? idx[u] : -1;
            if (p < 0 || u >= n_users) {
                continue;
            }
            p += static_cast<int32_t>(r % B);
            if (RING == 0 && p >= static_cast<int32_t>(pages * 32)) {
                continue;
            }
            const uint32_t pos = RING != 0 ? static_cast<uint32_t>(p) % (pages * 32) : static_cast<uint32_t>(p);
            const uint32_t page = page_base + (u * heads + h) * pages + pos / 32, row = pos % 32;
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t src_off = BASE + ((rb / 16) * 2 + f) * FACE + (rb % 16) * FROW;
                const uint32_t dst_off = BASE + ((row / 16) * 2 + f) * FACE + (row % 16) * FROW;
                const uint32_t slot = stage0 + (j * 2 + f) * 2 * SLOT;
                const uint32_t dst = slot + dst_off % SLOT;
                volatile tt_l1_ptr uint32_t* from = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + src_off);
                volatile tt_l1_ptr uint32_t* to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                for (uint32_t w = 0; w < FROW / 4; ++w) {
                    to[w] = from[w];
                }
                const uint64_t tile_addr = is_k ? s_kc.get_noc_addr(page * WT + t, dst_off)
                                                : s_vc.get_noc_addr(page * WT + t, dst_off);
                noc_async_write(dst, tile_addr, FROW);
                // the row's exponent byte, at its destination's offset mod 64 in its own slot
                const uint32_t es = ((rb / 16) * 2 + f) * 16 + rb % 16;
                const uint32_t ed = ((row / 16) * 2 + f) * 16 + row % 16;
                const uint32_t edst = slot + SLOT + ed % SLOT;
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(edst)[0] =
                    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src + es)[0];
                const uint64_t exp_addr =
                    is_k ? s_kc.get_noc_addr(page * WT + t, ed) : s_vc.get_noc_addr(page * WT + t, ed);
                noc_async_write(edst, exp_addr, 1);
            }
        }
        noc_async_writes_flushed();  // (the staged rows read out before the slots and the tile are reused)
        cb_pop_front(CB_KV, 1);
    }
    noc_async_write_barrier();
}
