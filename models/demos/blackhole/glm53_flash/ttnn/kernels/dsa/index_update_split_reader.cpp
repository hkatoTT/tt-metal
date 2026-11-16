// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DSA decode's indexer update, the split form (dsa_indexer.index_update; the one-core-a-user form:
// index_update_reader.cpp), the reader / writer.  The same caches as that form leaves them, bit for bit, with a user's
// segments over cores and only the keys that stay computed:
//  - segment s's key (the one-core form's, made after rows pos mod RING of segments 0 .. s went into its user's recent
//    pages) reads the pool's 4 ring rows; each is the row of ik / ig of the last segment s' <= s that wrote it, else
//    the ring as the step found it (DRAM).  Both are inputs: a core makes any segment's key without the others' writes.
//  - the one-core form writes every segment's key into its pool page's row; the step leaves the last segment's that
//    writes a (page, row).  The others' keys (a pool written again later in the step) are not made (SKIP; SKIP = 0
//    makes and sends every key, applied as the last per row all the same: a timing control).
//  - the owner core (its user's last part: segment SEGS - 1, always kept) alone writes DRAM: the pool pages (read,
//    the parts' keys put in their rows, written back) and the recent pages (the ring as the step leaves it: each row
//    the last segment's that wrote it).  The other parts send their keys' rows (256 B a key) to the owner's slot of
//    the segment, then add their count to the owner's semaphore after the writes are acknowledged; the owner waits for
//    the count, which also orders its ring write after the parts' reads of the old ring.  Unicast writes and a
//    semaphore increment only, all on the reader's NoC (no multicast).
// Rows are copied in L1 by local NoC reads of 32-byte face rows; DRAM is read and written in whole tiles.
// A BFP pools cache (BITS 8 / 4, dsa.POOLS_DTYPE): the owner reads and writes its pages' BFP tiles and puts each kept
// key's row in by bfp_rows.hpp (the bf16 row from its slot rounded to nearest on its face rows' own exponents: the other
// rows' bytes stay as they were); the parts still send bf16 rows.
//   CT: USERS, SEGS, CB_RK, CB_RG, CB_KEY, CB_POOL, CB_IK, CB_IG, CB_BIAS, CB_SCALER, CB_PARAMS, CB_RING, CB_CTL,
//       CB_SLOT, RING, SKIP, SEM, NBLK (ik / ig tile rows a core keeps), NPB (pool pages the owner reads ahead), BITS
//       (the pools' format: 0 bf16, 8 BFP8_b, 4 BFP4_b), then the accessors of ik, ig [1, 1, B, 128] (tiles), params
//       [1, 1, B, 16] (uint32 rows), rkey, rgate [*, 1, 32, 128], pools [*, 1, 32, 128] (tiles), bias [1, 1, 32,
//       128] (fp32 tiles)
//   RT: the 7 addresses, user u, its segments s0 .. s1 - 1 here, owner (1: s1 = SEGS), the owner's NoC x, y

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "bfp_rows.hpp"

namespace {
constexpr uint32_t TILE = tt::constants::TILE_HW * 2;  // a bf16 tile
constexpr uint32_t COLS = 4;                            // 128 columns: 4 tiles
constexpr uint32_t FACE = 512;                          // a bf16 16 x 16 face
constexpr uint32_t ROWB = 32;                           // a face row
constexpr uint32_t SLOT = COLS * 2 * ROWB;              // a key's row: 4 tiles x 2 faces
constexpr uint32_t NONE = 0xFFFFFFFF;

FORCE_INLINE uint32_t row_off(uint32_t row) { return (row / 16) * 2 * FACE + (row % 16) * ROWB; }

// row src_row of 4 tiles at src -> row dst_row of 4 tiles at dst, both in this core's L1 (8 face rows)
FORCE_INLINE void copy_row(uint32_t src, uint32_t src_row, uint32_t dst, uint32_t dst_row) {
    const uint32_t so = row_off(src_row), d_o = row_off(dst_row);
    for (uint32_t t = 0; t < COLS; ++t) {
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_read(get_noc_addr(src + t * TILE + so + f * FACE), dst + t * TILE + d_o + f * FACE, ROWB);
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
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(12);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(13);
    constexpr uint32_t RING = get_compile_time_arg_val(14);
    constexpr bool SKIP = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t SEM = get_compile_time_arg_val(16);
    constexpr uint32_t NBLK = get_compile_time_arg_val(17);
    constexpr uint32_t NPB = get_compile_time_arg_val(18);
    constexpr uint32_t BITS = get_compile_time_arg_val(19);
    constexpr uint32_t PTILE = dsa_bfp::tile_bytes(BITS);  // a pools tile
    static_assert(BITS == 0 || BITS == 8 || BITS == 4);
    static_assert(RING % 4 == 0 && RING <= 16, "the recent pages' ring: whole pools in a tile's top faces");
    static_assert(SEGS >= 1 && SEGS <= 16 && NPB >= 1);
    constexpr auto ik_args = TensorAccessorArgs<20>();
    constexpr auto ig_args = TensorAccessorArgs<ik_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<ig_args.next_compile_time_args_offset()>();
    constexpr auto rk_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto rg_args = TensorAccessorArgs<rk_args.next_compile_time_args_offset()>();
    constexpr auto pool_args = TensorAccessorArgs<rg_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();
    static_assert(ik_args.AlignedPageSize == TILE && ig_args.AlignedPageSize == TILE && p_args.AlignedPageSize == 64);
    static_assert(rk_args.AlignedPageSize == TILE && rg_args.AlignedPageSize == TILE);
    static_assert(pool_args.AlignedPageSize == PTILE);
    static_assert(bias_args.AlignedPageSize == 2 * TILE);

    const auto ik = TensorAccessor(ik_args, get_arg_val<uint32_t>(0));
    const auto ig = TensorAccessor(ig_args, get_arg_val<uint32_t>(1));
    const auto params = TensorAccessor(p_args, get_arg_val<uint32_t>(2));
    const auto rkey = TensorAccessor(rk_args, get_arg_val<uint32_t>(3));
    const auto rgate = TensorAccessor(rg_args, get_arg_val<uint32_t>(4));
    const auto pools = TensorAccessor(pool_args, get_arg_val<uint32_t>(5));
    const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(6));
    const uint32_t u = get_arg_val<uint32_t>(7);
    const uint32_t s0 = get_arg_val<uint32_t>(8);
    const uint32_t s1 = get_arg_val<uint32_t>(9);
    const bool owner = get_arg_val<uint32_t>(10) != 0;
    const uint32_t owner_x = get_arg_val<uint32_t>(11), owner_y = get_arg_val<uint32_t>(12);

    // user u's segments' params (rows v = s USERS + u): position, pool page base
    cb_reserve_back(cb_params, 1);
    const uint32_t p_l1 = get_write_ptr(cb_params);
    for (uint32_t s = 0; s < SEGS; ++s) {
        noc_async_read_page(s * USERS + u, params, p_l1 + s * 64);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();  // (lines of an earlier step's params)
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p_l1);
    uint32_t pos[SEGS], page[SEGS];
    for (uint32_t s = 0; s < SEGS; ++s) {
        pos[s] = p[s * 16];
        page[s] = p[s * 16 + 2] + (pos[s] / 4) / 32;
    }
    // kept[s]: no later segment writes its (page, row), so its key is the one the step leaves; made[s]: computed
    bool kept[SEGS], made[SEGS];
    uint32_t mine = 0, total = 0;
    for (uint32_t s = 0; s < SEGS; ++s) {
        kept[s] = true;
        for (uint32_t s2 = s + 1; s2 < SEGS; ++s2) {
            if (page[s2] == page[s] && (pos[s2] / 4) % 32 == (pos[s] / 4) % 32) {
                kept[s] = false;
            }
        }
        made[s] = SKIP ? kept[s] : true;
        if (made[s]) {
            ++total;
            mine += (s >= s0 && s < s1) ? 1 : 0;
        }
    }
    cb_reserve_back(cb_ctl, 1);  // compute's count of keys
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl))[0] = mine;
    cb_push_back(cb_ctl, 1);
    if (mine == 0) {  // (never the owner: segment SEGS - 1 is kept)
        return;
    }

    // compute's constants; the ring as the step found it; u's tile rows of ik / ig; the pool pages ahead
    cb_reserve_back(cb_bias, COLS);
    for (uint32_t t = 0; t < COLS; ++t) {
        noc_async_read_page(t, bias, get_write_ptr(cb_bias) + t * 2 * TILE);
    }
    cb_reserve_back(cb_ring, 2 * COLS);
    const uint32_t ring_k = get_write_ptr(cb_ring), ring_g = ring_k + COLS * TILE;
    for (uint32_t t = 0; t < COLS; ++t) {
        noc_async_read_page(u * COLS + t, rkey, ring_k + t * TILE);
        noc_async_read_page(u * COLS + t, rgate, ring_g + t * TILE);
    }
    const uint32_t b0 = u / 32, b1 = ((SEGS - 1) * USERS + u) / 32;
    cb_reserve_back(cb_ik, NBLK * COLS);
    cb_reserve_back(cb_ig, NBLK * COLS);
    const uint32_t ik_l1 = get_write_ptr(cb_ik), ig_l1 = get_write_ptr(cb_ig);
    for (uint32_t b = b0; b <= b1; ++b) {
        for (uint32_t t = 0; t < COLS; ++t) {
            noc_async_read_page(b * COLS + t, ik, ik_l1 + ((b - b0) * COLS + t) * TILE);
            noc_async_read_page(b * COLS + t, ig, ig_l1 + ((b - b0) * COLS + t) * TILE);
        }
    }
    uint32_t pages[SEGS], npages = 0;  // the owner's distinct pool pages, in segment order
    cb_reserve_back(cb_pool, NPB * COLS);
    const uint32_t pool_l1 = get_write_ptr(cb_pool);
    if (owner) {
        for (uint32_t s = 0; s < SEGS; ++s) {
            bool seen = !kept[s];
            for (uint32_t k = 0; k < npages && !seen; ++k) {
                seen = pages[k] == page[s];
            }
            if (!seen) {
                pages[npages++] = page[s];
            }
        }
        for (uint32_t k = 0; k < npages && k < NPB; ++k) {
            for (uint32_t t = 0; t < COLS; ++t) {
                noc_async_read_page(pages[k] * COLS + t, pools, pool_l1 + (k * COLS + t) * PTILE);
            }
        }
    }
    // compute's tiles: rows 0-3 a key's pool, rows 4-31 zero.  Two keys' worth (CB_RK / CB_RG 2 COLS pages: the
    // reserves alternate between two addresses), zeroed once (compute only reads them)
    cb_reserve_back(cb_rk, 2 * COLS);
    cb_reserve_back(cb_rg, 2 * COLS);
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t rk0 = get_write_ptr(cb_rk), rg0 = get_write_ptr(cb_rg);
        for (uint32_t o = 0; o < 2 * COLS * TILE; o += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, rk0 + o, MEM_ZEROS_SIZE);
            noc_async_read(zeros, rg0 + o, MEM_ZEROS_SIZE);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_bias, COLS);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL,
        /*reduce_factor=*/1>();

    // the L1 tile row holding row v of ik (ig: the same offset from ig_l1)
    auto block_of = [&](uint32_t s) -> uint32_t { return ((s * USERS + u) / 32 - b0) * COLS * TILE; };
    // ring row r as segment s's key reads it: the last segment s' <= s that wrote it, else NONE (the old ring)
    auto writer_of = [&](uint32_t r, uint32_t s) -> uint32_t {
        uint32_t w = NONE;
        for (uint32_t s2 = 0; s2 <= s; ++s2) {
            if (pos[s2] % RING == r) {
                w = s2;
            }
        }
        return w;
    };
    const uint32_t slot_l1 = get_write_ptr(cb_slot);
    // a key out of compute: its row 0 into segment s's slot (the owner's own, or sent to the owner's)
    auto take_key = [&](uint32_t s) {
        cb_wait_front(cb_key, COLS);
        const uint32_t key = get_read_ptr(cb_key);
        const uint32_t slot = slot_l1 + s * SLOT;
        for (uint32_t t = 0; t < COLS; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t src = key + t * TILE + f * FACE, dst = slot + (t * 2 + f) * ROWB;
                if (owner) {
                    noc_async_read(get_noc_addr(src), dst, ROWB);
                } else {
                    noc_async_write(src, get_noc_addr(owner_x, owner_y, dst), ROWB);
                }
            }
        }
        if (owner) {
            noc_async_read_barrier();
        } else {
            noc_async_writes_flushed();  // (compute's next key reuses the pages)
        }
        cb_pop_front(cb_key, COLS);
    };

    uint32_t pending = NONE;  // the segment whose key compute is making
    for (uint32_t s = s0; s < s1; ++s) {
        if (!made[s]) {
            continue;
        }
        cb_reserve_back(cb_rk, COLS);
        cb_reserve_back(cb_rg, COLS);
        const uint32_t rk_l1 = get_write_ptr(cb_rk), rg_l1 = get_write_ptr(cb_rg);
        const uint32_t first = ((pos[s] / 4) % (RING / 4)) * 4;  // the pool's 4 rows in the ring
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t w = writer_of(first + j, s);
            if (w == NONE) {
                copy_row(ring_k, first + j, rk_l1, j);
                copy_row(ring_g, first + j, rg_l1, j);
            } else {
                const uint32_t row = (w * USERS + u) % 32;
                copy_row(ik_l1 + block_of(w), row, rk_l1, j);
                copy_row(ig_l1 + block_of(w), row, rg_l1, j);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_rk, COLS);
        cb_push_back(cb_rg, COLS);
        if (pending != NONE) {
            take_key(pending);
        }
        pending = s;
    }
    if (owner) {  // the ring as the step leaves it (while compute makes the last key)
        for (uint32_t r = 0; r < RING; ++r) {
            const uint32_t w = writer_of(r, SEGS - 1);
            if (w != NONE) {
                const uint32_t row = (w * USERS + u) % 32;
                copy_row(ik_l1 + block_of(w), row, ring_k, r);
                copy_row(ig_l1 + block_of(w), row, ring_g, r);
            }
        }
        noc_async_read_barrier();
    }
    take_key(pending);

    if (!owner) {
        noc_async_write_barrier();  // (the keys in the owner's slots before the count)
        noc_semaphore_inc(get_noc_addr(owner_x, owner_y, get_semaphore(SEM)), mine);
        noc_async_atomic_barrier();
        return;
    }
    // the owner: the parts' keys, then the caches
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
    noc_semaphore_wait(sem, total - mine);
    noc_semaphore_set(sem, 0);
    for (uint32_t t = 0; t < COLS; ++t) {  // (after the parts' reads of the old ring: their counts came after them)
        noc_async_write_page(u * COLS + t, rkey, ring_k + t * TILE);
        noc_async_write_page(u * COLS + t, rgate, ring_g + t * TILE);
    }
    for (uint32_t k = 0; k < npages; ++k) {
        const uint32_t buf = pool_l1 + (k % NPB) * COLS * PTILE;
        if (k >= NPB) {  // (a page past the ones read ahead: its buffer's last write out first)
            noc_async_write_barrier();
            for (uint32_t t = 0; t < COLS; ++t) {
                noc_async_read_page(pages[k] * COLS + t, pools, buf + t * PTILE);
            }
            noc_async_read_barrier();
        }
        if constexpr (BITS != 0) {
            noc_async_read_barrier();  // (the pages read ahead)
            invalidate_l1_cache();     // (the slots: the parts' NoC writes, the owner's own copies)
        }
        for (uint32_t s = 0; s < SEGS; ++s) {
            if (!kept[s] || page[s] != pages[k]) {
                continue;
            }
            const uint32_t slot = slot_l1 + s * SLOT, row = (pos[s] / 4) % 32, d_o = row_off(row);
            for (uint32_t t = 0; t < COLS; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    if constexpr (BITS == 0) {
                        noc_async_read(get_noc_addr(slot + (t * 2 + f) * ROWB), buf + t * TILE + d_o + f * FACE, ROWB);
                    } else {  // the face row's 16 values rounded into the BFP tile's face row
                        dsa_bfp::pack_group<(BITS == 0 ? 8 : BITS)>(  // (BITS 0: not taken)
                            slot + (t * 2 + f) * ROWB, buf + t * PTILE, (row / 16) * 2 + f, row % 16);
                    }
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < COLS; ++t) {
            noc_async_write_page(pages[k] * COLS + t, pools, buf + t * PTILE);
        }
    }
    noc_async_write_barrier();
}
