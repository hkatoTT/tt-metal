// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DSA decode (dsa_latent.py): this core's users' tokens' latents (rows of the tiles [1, 1, B, D] bf16) into their rows
// of the paged latent cache [pages, 1, 32, D] (tiled) at their positions, in place of reshaping each token into a tile
// of its own and paged_update_cache's read, untilize, tilize and write of the whole page.  Token v takes user
// v mod `owners`'s pages (owner * `pages` + position / 32: the page tables `TtDsa.alloc_positions` makes); position
// -1: no token.  Per tile across D a token's row is two 32-byte face rows; each is read into a 64-byte slot of L1 at
// the source's offset mod 64 and written from the destination's (moved within the slot when the two differ: a NoC
// transfer under 64 bytes needs the same offset mod 64 at both ends).  A user's tokens (MTP's token and drafts, rows
// next to each other) are on one core for each range of tiles; the ranges on cores of their own (dsa_latent.py).
void kernel_main() {
    const uint32_t lat_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t o0 = get_arg_val<uint32_t>(3);
    const uint32_t o1 = get_arg_val<uint32_t>(4);
    const uint32_t t0 = get_arg_val<uint32_t>(5);  // this core's tiles across D: [t0, t1)
    const uint32_t t1 = get_arg_val<uint32_t>(6);

    constexpr uint32_t tokens = get_compile_time_arg_val(0);    // B
    constexpr uint32_t owners = get_compile_time_arg_val(1);    // users whose pages the tokens take
    constexpr uint32_t pages = get_compile_time_arg_val(2);     // a user's pages
    constexpr uint32_t tiles = get_compile_time_arg_val(3);     // tiles across D
    constexpr uint32_t pos_page = get_compile_time_arg_val(4);  // the positions' page (bytes)
    constexpr uint32_t cb = get_compile_time_arg_val(5);
    constexpr auto lat_args = TensorAccessorArgs<6>();
    constexpr auto cache_args = TensorAccessorArgs<lat_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048, FACE = 512, FROW = 32, SLOT = 64;
    const uint32_t stage_bytes = 2 * (t1 - t0) * SLOT;  // a token's slots (this core's tiles)

    const auto s_lat = TensorAccessor(lat_args, lat_addr, TILE);
    const auto s_cache = TensorAccessor(cache_args, cache_addr, TILE);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, pos_page);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t stage0 = l1 + ((pos_page + SLOT - 1) & ~(SLOT - 1));
    noc_async_read(s_pos.get_noc_addr(0), l1, pos_page);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1);

    uint32_t n = 0;
    for (uint32_t o = o0; o < o1; ++o) {
        for (uint32_t v = o; v < tokens; v += owners, ++n) {
            if (pos[v] < 0) {
                continue;
            }
            const uint32_t stage = stage0 + n * stage_bytes;
            const uint32_t tr = v / 32, rb = v % 32;
            for (uint32_t t = t0; t < t1; ++t) {
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t off = ((rb / 16) * 2 + h) * FACE + (rb % 16) * FROW;
                    noc_async_read(
                        s_lat.get_noc_addr(tr * tiles + t, off), stage + ((t - t0) * 2 + h) * SLOT + off % SLOT, FROW);
                }
            }
        }
    }
    noc_async_read_barrier();
    n = 0;
    for (uint32_t o = o0; o < o1; ++o) {
        for (uint32_t v = o; v < tokens; v += owners, ++n) {
            if (pos[v] < 0) {
                continue;
            }
            const uint32_t p = static_cast<uint32_t>(pos[v]);
            const uint32_t page = o * pages + p / 32, r = p % 32;
            const uint32_t stage = stage0 + n * stage_bytes;
            const uint32_t rb = v % 32;
            for (uint32_t t = t0; t < t1; ++t) {
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t src_off = ((rb / 16) * 2 + h) * FACE + (rb % 16) * FROW;
                    const uint32_t dst_off = ((r / 16) * 2 + h) * FACE + (r % 16) * FROW;
                    const uint32_t slot = stage + ((t - t0) * 2 + h) * SLOT;
                    const uint32_t dst = slot + dst_off % SLOT;
                    if (src_off % SLOT != dst_off % SLOT) {  // the other half of the slot
                        volatile tt_l1_ptr uint32_t* from =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + src_off % SLOT);
                        volatile tt_l1_ptr uint32_t* to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                        for (uint32_t w = 0; w < FROW / 4; ++w) {
                            to[w] = from[w];
                        }
                    }
                    noc_async_write(dst, s_cache.get_noc_addr(page * tiles + t, dst_off), FROW);
                }
            }
        }
    }
    noc_async_write_barrier();
}
