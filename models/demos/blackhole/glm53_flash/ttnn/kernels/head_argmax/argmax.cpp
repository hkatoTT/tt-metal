// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// The greedy head's per-chip top (head_argmax.py): for each row of the logits [1, 1, M, W] fp32 (tiled), the largest
// of the chip's valid columns and the smallest global id where it is hit, in one pass, in place of the mask add, max,
// compare, multiply, id add and min.  A core takes one tile row's run of column tiles; its NCRISC (half 0) the rows
// 0-15 (faces 0-1, a tile's first 2 KB), its BRISC (half 1) the rows 16-31.  Each RISC scans its rows in column order
// on the RISC-V (fp32 compared as ordered integers, -0.0 as +0.0, as the masked add made it; a strictly larger value
// replaces, so the first, smallest column of the maximum stays), then writes its rows' (value, id) into the tile row's
// root core (the run of column 0) and increments the root's semaphore.  The root's NCRISC folds the runs in column
// order and writes the value and id tiles (column 0; the rest zero).
FORCE_INLINE int32_t key_of(uint32_t bits) {
    // negative floats flipped below the positive ones; -0.0 (key -1) onto +0.0 (key 0); branch-free
    const int32_t s = static_cast<int32_t>(bits);
    return (s ^ ((s >> 31) & 0x7FFFFFFF)) + static_cast<int32_t>(bits == 0x80000000u);
}

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t value_addr = get_arg_val<uint32_t>(1);
    const uint32_t ids_addr = get_arg_val<uint32_t>(2);
    const uint32_t tr = get_arg_val<uint32_t>(3);         // the tile row
    const uint32_t ct0 = get_arg_val<uint32_t>(4);        // this core's first column tile
    const uint32_t ct1 = get_arg_val<uint32_t>(5);        // .. past its last
    const uint32_t run = get_arg_val<uint32_t>(6);        // this core's run in the tile row (0: the root)
    const uint32_t root_x = get_arg_val<uint32_t>(7);     // the root core (NoC coordinates)
    const uint32_t root_y = get_arg_val<uint32_t>(8);
    const uint32_t valid_cols = get_arg_val<uint32_t>(9);  // the chip's columns before its padding
    const uint32_t id_base = get_arg_val<uint32_t>(10);    // the chip's first global id

    constexpr uint32_t half = get_compile_time_arg_val(0);
    constexpr uint32_t wt = get_compile_time_arg_val(1);      // column tiles
    constexpr uint32_t runs = get_compile_time_arg_val(2);    // runs a tile row
    constexpr uint32_t rows = get_compile_time_arg_val(3);    // M (valid rows)
    constexpr uint32_t max_run = get_compile_time_arg_val(4);  // column tiles of the longest run
    constexpr uint32_t cb = get_compile_time_arg_val(5);
    constexpr uint32_t sem_id = get_compile_time_arg_val(6);
    // GLM53_HEAD_ARGMAX_FAST (head_argmax.py, on by default): a full face of 16
    // first by its raw signed bits (a load and a compare a value); its maximum m > 0 is a positive float's bits, so
    // key_of(m) = m, every other value's key <= m (the positives' keys are their bits; the negatives' and -0.0's keys
    // are < 0 < m), and the first column of m is the first of the face's best key: the same (key, column) as the keyed
    // scan.  m <= 0 (no positive value in the face): the keyed scan
    constexpr bool fast = get_compile_time_arg_val(7) != 0;
    constexpr auto logits_args = TensorAccessorArgs<8>();
    constexpr auto value_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096, HALF = 2048, FACE = 256;  // bytes, bytes, floats
    constexpr uint32_t SLOT = 128;  // a RISC's 16 rows of (key, id)

    const uint32_t tile_rows_here = rows - tr * 32 < 32 ? rows - tr * 32 : 32;
    const uint32_t my_rows = half == 0 ? (tile_rows_here < 16 ? tile_rows_here : 16)
                                       : (tile_rows_here > 16 ? tile_rows_here - 16 : 0);
    const uint32_t halves = tile_rows_here > 16 ? 2 : 1;

    // L1: each RISC's tile halves, the runs' slots (the root's), the output tiles
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t data = l1 + half * max_run * HALF;
    const uint32_t slots = l1 + 2 * max_run * HALF;
    const uint32_t out_tiles = slots + runs * 2 * SLOT;
    const uint32_t sem_addr = get_semaphore(sem_id);

    if (my_rows > 0) {
        const auto s_logits = TensorAccessor(logits_args, logits_addr, TILE);
        for (uint32_t t = ct0; t < ct1; ++t) {
            noc_async_read(s_logits.get_noc_addr(tr * wt + t, half * HALF), data + (t - ct0) * HALF, HALF);
        }
        noc_async_read_barrier();
        // this RISC's rows' best (key, id), staged at its own slot address, then sent to the root's
        const uint32_t mine = slots + (run * 2 + half) * SLOT;
        volatile tt_l1_ptr int32_t* out = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mine);
        // the data landed (barrier above): plain loads the compiler may schedule; a face row's 16 values at a time,
        // its best found first and compared with the row's best once (the first column of the face row's best kept)
        for (uint32_t r = 0; r < my_rows; ++r) {
            int32_t best = static_cast<int32_t>(0x80000000u);
            uint32_t best_col = 0xFFFFFFFFu;
            for (uint32_t t = ct0; t < ct1; ++t) {
                const uint32_t col0 = t * 32;
                if (col0 >= valid_cols) {
                    break;
                }
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t base = col0 + f * 16;
                    if (base >= valid_cols) {
                        break;
                    }
                    const uint32_t* v = reinterpret_cast<const uint32_t*>(data + (t - ct0) * HALF) + f * FACE + r * 16;
                    if (valid_cols - base >= 16) {
                        int32_t fb = 0;
                        uint32_t fj = 0;
                        bool keyed = true;
                        if constexpr (fast) {
                            const int32_t* sv = reinterpret_cast<const int32_t*>(v);
                            int32_t mx = sv[0];
                            uint32_t mj = 0;
#pragma GCC unroll 15
                            for (uint32_t j = 1; j < 16; ++j) {
                                if (sv[j] > mx) {
                                    mx = sv[j];
                                    mj = j;
                                }
                            }
                            if (mx > 0) {
                                fb = mx;
                                fj = mj;
                                keyed = false;
                            }
                        }
                        if (keyed) {
                            fb = key_of(v[0]);
                            fj = 0;
#pragma GCC unroll 15
                            for (uint32_t j = 1; j < 16; ++j) {
                                const int32_t k = key_of(v[j]);
                                if (k > fb) {
                                    fb = k;
                                    fj = j;
                                }
                            }
                        }
                        if (fb > best || best_col == 0xFFFFFFFFu) {
                            best = fb;
                            best_col = base + fj;
                        }
                    } else {
                        for (uint32_t j = 0; j < valid_cols - base; ++j) {
                            const int32_t k = key_of(v[j]);
                            if (k > best || best_col == 0xFFFFFFFFu) {
                                best = k;
                                best_col = base + j;
                            }
                        }
                    }
                }
            }
            out[2 * r] = best;
            out[2 * r + 1] = static_cast<int32_t>(best_col);
        }
        const uint64_t root_slot = get_noc_addr(root_x, root_y, mine);
        if (run != 0) {
            noc_async_write(mine, root_slot, SLOT);
            noc_async_write_barrier();
        }
        noc_semaphore_inc(get_noc_addr(root_x, root_y, sem_addr), 1);
    }

    if (half != 0 || run != 0) {
        noc_async_atomic_barrier();  // (the root's increment's ack in before the kernel ends)
        return;
    }
    // the root: fold the runs in column order, then the value and id tiles of this tile row
    // the output tiles zeroed from the core's zeros region while the runs come in
    for (uint32_t off = 0; off < 2 * TILE; off += MEM_ZEROS_SIZE) {
        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), out_tiles + off, MEM_ZEROS_SIZE);
    }
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, runs * halves);
    noc_async_read_barrier();
    uint32_t* vt = reinterpret_cast<uint32_t*>(out_tiles);
    float* it = reinterpret_cast<float*>(out_tiles + TILE);
    for (uint32_t r = 0; r < tile_rows_here; ++r) {
        const uint32_t h = r / 16, rr = r % 16;
        int32_t best = static_cast<int32_t>(0x80000000u);
        uint32_t best_col = 0xFFFFFFFFu;
        for (uint32_t k = 0; k < runs; ++k) {
            const int32_t* s = reinterpret_cast<const int32_t*>(slots + (k * 2 + h) * SLOT);
            const int32_t key = s[2 * rr];
            const uint32_t col = static_cast<uint32_t>(s[2 * rr + 1]);
            if (col != 0xFFFFFFFFu && (best_col == 0xFFFFFFFFu || key > best)) {
                best = key;
                best_col = col;
            }
        }
        const uint32_t bits = best < 0 ? static_cast<uint32_t>(best ^ 0x7FFFFFFF) : static_cast<uint32_t>(best);
        const uint32_t at = h * 2 * FACE + rr * 16;  // column 0 of row r: face 0 (rows 0-15) or 2
        vt[at] = bits;
        it[at] = static_cast<float>(id_base + best_col);
    }
    noc_semaphore_set(sem, 0);
    const auto s_value = TensorAccessor(value_args, value_addr, TILE);
    const auto s_ids = TensorAccessor(ids_args, ids_addr, TILE);
    noc_async_write(out_tiles, s_value.get_noc_addr(tr), TILE);
    noc_async_write(out_tiles + TILE, s_ids.get_noc_addr(tr), TILE);
    noc_async_write_barrier();
    noc_async_atomic_barrier();  // (this core's own increment above)
}
