// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2 (dflash2_ops.kv_write): head rows of a tiled bf16 source into their rows of the paged cache
// [pages, 1, 32, 128] (tiled).  For token r of this core's range, layer l < layers and KV head h < heads:
//   source row (r / B) sU + (r % B) sP + l sL + h sH, columns 128 (c0 + l cL + h cH) .. + 127;
//   user u, position p: mode 0 (the drafter's block) u = r / B, p = idx[u] + r % B; mode 1 (context rows)
//   p = idx[r], u = users[r];
//   cache page page_base + l layer_pages + (u heads + h) pages + p / 32, row p % 32 (skipped: p < 0, p past the pages,
//   u past the last user).  RING (dflash2.RING, the drafter's last positions only): p taken mod pages x 32 (the
//   paged sdpa decode's cache_position_modulo), none past the pages.
// A head row is 4 tiles x 2 face rows of 32 bytes; each is read into a 64-byte slot of L1 at the source's offset mod
// 64 and written from the destination's (moved within the slot when they differ: a NoC transfer under 64 bytes needs
// the same offset mod 64 at both ends).  In batches of `batch` head rows.
// BITS 8 / 4 (the cache in BFP8_b / BFP4_b: source and cache, tiles of 64 exponent bytes then the faces'
// mantissas): a face row is its 16 / 8 bytes of mantissas and its exponent byte, each moved as above in its own slot.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t users_addr = get_arg_val<uint32_t>(3);
    const uint32_t r0 = get_arg_val<uint32_t>(4);
    const uint32_t r1 = get_arg_val<uint32_t>(5);
    const uint32_t page_base = get_arg_val<uint32_t>(6);  // the first layer's first page

    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t n_idx = get_compile_time_arg_val(2);  // entries of idx (and users)
    constexpr uint32_t n_users = get_compile_time_arg_val(3);
    constexpr uint32_t layers = get_compile_time_arg_val(4);
    constexpr uint32_t heads = get_compile_time_arg_val(5);
    constexpr uint32_t src_wt = get_compile_time_arg_val(6);  // tiles across the source
    constexpr uint32_t sU = get_compile_time_arg_val(7);
    constexpr uint32_t sP = get_compile_time_arg_val(8);
    constexpr uint32_t sL = get_compile_time_arg_val(9);
    constexpr uint32_t sH = get_compile_time_arg_val(10);
    constexpr uint32_t c0 = get_compile_time_arg_val(11);
    constexpr uint32_t cL = get_compile_time_arg_val(12);
    constexpr uint32_t cH = get_compile_time_arg_val(13);
    constexpr uint32_t pages = get_compile_time_arg_val(14);        // a (user, head)'s pages
    constexpr uint32_t layer_pages = get_compile_time_arg_val(15);  // a layer's pages
    constexpr uint32_t batch = get_compile_time_arg_val(16);        // head rows staged at a time
    constexpr uint32_t cb = get_compile_time_arg_val(17);
    constexpr uint32_t BITS = get_compile_time_arg_val(18);  // 0 bf16, 8 / 4 BFP8_b / BFP4_b
    constexpr uint32_t RING = get_compile_time_arg_val(19);  // 1: positions mod pages x 32
    constexpr auto src_args = TensorAccessorArgs<20>();
    constexpr auto cache_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto users_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = BITS == 0 ? 2048 : (BITS == 8 ? 1088 : 576), SLOT = 64, WT = 4;
    constexpr uint32_t FACE = BITS == 0 ? 512 : BITS * 32;    // a face's bytes (BFP: its mantissas)
    constexpr uint32_t FROW = BITS == 0 ? 32 : BITS * 2;      // a face row's bytes (BFP: its mantissas)
    constexpr uint32_t BASE = BITS == 0 ? 0 : 64;             // the faces' first byte (BFP: after the exponents)
    constexpr uint32_t NS = BITS == 0 ? 1 : 2;                // slots a face row: its values (BFP: and its exponent)
    constexpr uint32_t idx_bytes = n_idx * 4;
    constexpr uint32_t idx_slot = (idx_bytes + 63) & ~63u;
    constexpr uint32_t row_slots = WT * 2 * NS * SLOT;  // a head row's slots

    const auto s_src = TensorAccessor(src_args, src_addr, TILE);
    const auto s_cache = TensorAccessor(cache_args, cache_addr, TILE);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t users_l1 = l1 + idx_slot;
    const uint32_t stage0 = users_l1 + (mode == 1 ? idx_slot : 0);
    noc_async_read(TensorAccessor(idx_args, idx_addr, idx_bytes).get_noc_addr(0), l1, idx_bytes);
    if (mode == 1) {
        noc_async_read(TensorAccessor(users_args, users_addr, idx_bytes).get_noc_addr(0), users_l1, idx_bytes);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* idx = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1);
    volatile tt_l1_ptr int32_t* usr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(users_l1);

    // the head rows of the range in order (r, l, h), `batch` at a time: their destinations, then reads, then writes
    constexpr uint32_t per_r = layers * heads;
    uint32_t item = r0 * per_r;
    const uint32_t end = r1 * per_r;
    while (item < end) {
        const uint32_t n = end - item < batch ? end - item : batch;
        // reads
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t it = item + j;
            const uint32_t r = it / per_r, l = (it % per_r) / heads, h = it % heads;
            const uint32_t srow = (r / B) * sU + (r % B) * sP + l * sL + h * sH;
            const uint32_t col = c0 + l * cL + h * cH;
            const uint32_t stage = stage0 + j * row_slots;
            const uint32_t rb = srow % 32;
            for (uint32_t t = 0; t < WT; ++t) {
                const uint32_t tile = (srow / 32) * src_wt + col * WT + t;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t face = (rb / 16) * 2 + f;
                    const uint32_t off = BASE + face * FACE + (rb % 16) * FROW;
                    const uint32_t slot = stage + (t * 2 + f) * NS * SLOT;
                    noc_async_read(s_src.get_noc_addr(tile, off), slot + off % SLOT, FROW);
                    if constexpr (BITS != 0) {  // the row's exponent byte
                        const uint32_t e = face * 16 + rb % 16;
                        noc_async_read(s_src.get_noc_addr(tile, e), slot + SLOT + e % SLOT, 1);
                    }
                }
            }
        }
        noc_async_read_barrier();
        // writes
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t it = item + j;
            const uint32_t r = it / per_r, l = (it % per_r) / heads, h = it % heads;
            int32_t p;
            uint32_t u;
            if (mode == 0) {
                u = r / B;
                p = (u < n_idx ? idx[u] : -1);
                if (p >= 0) {
                    p += static_cast<int32_t>(r % B);
                }
            } else {
                p = r < n_idx ? idx[r] : -1;
                u = r < n_idx ? static_cast<uint32_t>(usr[r]) : n_users;
            }
            if (p < 0 || (RING == 0 && p >= static_cast<int32_t>(pages * 32)) || u >= n_users) {
                continue;
            }
            const uint32_t pos = RING != 0 ? static_cast<uint32_t>(p) % (pages * 32) : static_cast<uint32_t>(p);
            const uint32_t page = page_base + l * layer_pages + (u * heads + h) * pages + pos / 32, row = pos % 32;
            const uint32_t srow = (r / B) * sU + (r % B) * sP + l * sL + h * sH;
            const uint32_t rb = srow % 32;
            const uint32_t stage = stage0 + j * row_slots;
            for (uint32_t t = 0; t < WT; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t src_off = BASE + ((rb / 16) * 2 + f) * FACE + (rb % 16) * FROW;
                    const uint32_t dst_off = BASE + ((row / 16) * 2 + f) * FACE + (row % 16) * FROW;
                    const uint32_t slot = stage + (t * 2 + f) * NS * SLOT;
                    const uint32_t dst = slot + dst_off % SLOT;
                    if (src_off % SLOT != dst_off % SLOT) {
                        volatile tt_l1_ptr uint32_t* from =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + src_off % SLOT);
                        volatile tt_l1_ptr uint32_t* to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                        for (uint32_t w = 0; w < FROW / 4; ++w) {
                            to[w] = from[w];
                        }
                    }
                    noc_async_write(dst, s_cache.get_noc_addr(page * WT + t, dst_off), FROW);
                    if constexpr (BITS != 0) {  // the exponent byte, at its destination's offset mod 64 in its slot
                        const uint32_t es = ((rb / 16) * 2 + f) * 16 + rb % 16;
                        const uint32_t ed = ((row / 16) * 2 + f) * 16 + row % 16;
                        const uint32_t edst = slot + SLOT + ed % SLOT;
                        if (es % SLOT != ed % SLOT) {
                            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(edst)[0] =
                                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(slot + SLOT + es % SLOT)[0];
                        }
                        noc_async_write(edst, s_cache.get_noc_addr(page * WT + t, ed), 1);
                    }
                }
            }
        }
        noc_async_write_barrier();
        item += n;
    }
}
