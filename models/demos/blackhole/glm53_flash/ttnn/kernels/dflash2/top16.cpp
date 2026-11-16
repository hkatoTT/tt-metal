// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

#include "top16_common.hpp"

// DFlash2's candidates (dflash2_ops.top16): for each live row of the logits [1, 1, M, W] fp32 (tiled), the 16 largest
// of the chip's valid columns, in descending order (equal values: the smaller column first), with their global ids ->
// out [1, 1, M, 32] fp32 tiled (a row: the 16 values in columns 0-15, the 16 ids as floats, exact below 2^24, in
// 16-31; the rows not live are not written).  The work is split into items (a group of rows ra..rb of one half of a
// tile row, a run of column tiles) over both RISCs of the cores (this kernel is one RISC's: `risc`).  Each RISC reads
// its rows of its tiles (two face rows a row and tile), keeps a sorted list of 16 (key, column) per row in its local
// memory while it scans (a face row with no value above the list's last is skipped: with a non-negative last, a plain
// signed compare of the bits), writes the lists into its group's root (the item of run 0) and increments the root's
// semaphore; the root merges the runs of its rows in column order and writes the rows.
void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t tr = get_arg_val<uint32_t>(2);  // tile row
    const uint32_t h = get_arg_val<uint32_t>(3);   // its half: rows 0-15 (faces 0, 1) or 16-31 (faces 2, 3)
    const uint32_t ra = get_arg_val<uint32_t>(4);  // the group's rows of the half
    const uint32_t rb = get_arg_val<uint32_t>(5);
    const uint32_t ct0 = get_arg_val<uint32_t>(6);  // the run's column tiles
    const uint32_t ct1 = get_arg_val<uint32_t>(7);
    const uint32_t run = get_arg_val<uint32_t>(8);
    const uint32_t root_x = get_arg_val<uint32_t>(9);
    const uint32_t root_y = get_arg_val<uint32_t>(10);
    const uint32_t root_risc = get_arg_val<uint32_t>(11);
    const uint32_t valid_cols = get_arg_val<uint32_t>(12);
    const uint32_t id_base = get_arg_val<uint32_t>(13);

    constexpr uint32_t risc = get_compile_time_arg_val(0);  // this kernel's RISC of the core: its L1 and semaphore
    constexpr uint32_t wt = get_compile_time_arg_val(1);
    constexpr uint32_t runs = get_compile_time_arg_val(2);  // runs a group
    constexpr uint32_t per = get_compile_time_arg_val(3);   // column tiles a run, at most
    constexpr uint32_t G = get_compile_time_arg_val(4);     // rows a group, at most (<= 16)
    constexpr uint32_t cb = get_compile_time_arg_val(5);
    constexpr uint32_t sem_id = get_compile_time_arg_val(6);  // RISC 0's; RISC 1's is sem_id + 1
    // (compile-time args 7-9: the rows, period and live rows of the split)
    constexpr auto logits_args = TensorAccessorArgs<10>();
    constexpr auto out_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr uint32_t FROW = 64, FACE_B = 1024;  // bytes: an fp32 face row, face
    constexpr uint32_t LIST = TOPN * 8;                       // a row's 16 (key, column)
    constexpr uint32_t TILE_B = 2 * G * FROW;                 // a tile's rows here: its left face's, then its right's
    constexpr uint32_t DATA = per * TILE_B;
    constexpr uint32_t SL = runs * G * LIST;  // a root's lists: run k's rows at k G LIST
    static_assert(G >= 1 && G <= 16);

    const uint32_t g = rb - ra;
    if (g == 0) {
        return;
    }
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t data = l1 + risc * DATA;
    const uint32_t slots = l1 + 2 * DATA + risc * SL;  // this RISC's: a root's lists (a sender's own at its run)
    const uint32_t root_slots = l1 + 2 * DATA + root_risc * SL;
    const uint32_t stage = l1 + 2 * DATA + 2 * SL + risc * G * 2 * FROW;
    const uint32_t face0 = 2 * h * FACE_B + ra * FROW;  // the group's first row in the left face of the half

    const auto s_logits = TensorAccessor(logits_args, logits_addr, 4096);
    {
        DeviceZoneScopedN("T16_READ");
        for (uint32_t t = ct0; t < ct1; ++t) {
            const uint32_t at = data + (t - ct0) * TILE_B;
            noc_async_read(s_logits.get_noc_addr(tr * wt + t, face0), at, g * FROW);
            noc_async_read(s_logits.get_noc_addr(tr * wt + t, face0 + FACE_B), at + G * FROW, g * FROW);
        }
        noc_async_read_barrier();
    }
    const uint32_t mine = slots + run * G * LIST;
    {
        DeviceZoneScopedN("T16_SCAN");
        for (uint32_t r = 0; r < g; ++r) {
            int32_t key[TOPN];
            uint32_t col[TOPN];
            uint32_t n = 0;
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
                    const int32_t* v =
                        reinterpret_cast<const int32_t*>(data + (t - ct0) * TILE_B + f * G * FROW) + r * 16;
                    const uint32_t cnt = valid_cols - base >= 16 ? 16 : valid_cols - base;
                    if (n == TOPN && cnt == 16) {
                        const int32_t thr = key[TOPN - 1];
                        bool any = false;
                        if (thr >= 0) {  // (a key >= 0 is its float's bits: a larger value has larger signed bits)
#pragma GCC unroll 16
                            for (uint32_t j = 0; j < 16; ++j) {
                                any |= v[j] > thr;
                            }
                        } else {
#pragma GCC unroll 16
                            for (uint32_t j = 0; j < 16; ++j) {
                                any |= key_of(static_cast<uint32_t>(v[j])) > thr;
                            }
                        }
                        if (!any) {
                            continue;
                        }
                    }
                    for (uint32_t j = 0; j < cnt; ++j) {
                        n = list_insert(key, col, n, key_of(static_cast<uint32_t>(v[j])), base + j);
                    }
                }
            }
            for (uint32_t j = n; j < TOPN; ++j) {
                col[j] = EMPTY;
                key[j] = 0;
            }
            uint32_t* dst = reinterpret_cast<uint32_t*>(mine + r * LIST);
            for (uint32_t j = 0; j < TOPN; ++j) {
                dst[j] = static_cast<uint32_t>(key[j]);
                dst[TOPN + j] = col[j];
            }
        }
    }
    const uint32_t root_sem = get_semaphore(sem_id + root_risc);
    if (run != 0) {
        noc_async_write(mine, get_noc_addr(root_x, root_y, root_slots + run * G * LIST), g * LIST);
        noc_async_write_barrier();
    }
    noc_semaphore_inc(get_noc_addr(root_x, root_y, root_sem), 1);

    if (run != 0) {
        return;
    }
    // the group's root: merge the runs of its rows in column order, then write them
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(root_sem);
    {
        DeviceZoneScopedN("T16_WAIT");
        noc_semaphore_wait(sem, runs);
    }
    DeviceZoneScopedN("T16_MERGE");
    const auto s_out = TensorAccessor(out_args, out_addr, 4096);
    for (uint32_t rr = 0; rr < g; ++rr) {
        int32_t key[TOPN];
        uint32_t col[TOPN];
        uint32_t n = 0;
        for (uint32_t k = 0; k < runs; ++k) {
            const uint32_t at = slots + k * G * LIST + rr * LIST;
            const int32_t* sk = reinterpret_cast<const int32_t*>(at);
            const uint32_t* sc = reinterpret_cast<const uint32_t*>(at + TOPN * 4);
            for (uint32_t j = 0; j < TOPN && sc[j] != EMPTY; ++j) {
                if (n == TOPN && sk[j] <= key[TOPN - 1]) {
                    break;  // the run's list is sorted: the rest cannot enter
                }
                n = list_insert(key, col, n, sk[j], sc[j]);
            }
        }
        uint32_t* o = reinterpret_cast<uint32_t*>(stage + rr * 2 * FROW);     // the values' face row
        float* of = reinterpret_cast<float*>(stage + rr * 2 * FROW + FROW);  // the ids'
        for (uint32_t j = 0; j < TOPN; ++j) {
            const bool ok = j < n;
            o[j] = ok ? bits_of(key[j]) : 0xFF800000u;  // (fewer than 16 valid columns: -inf)
            of[j] = ok ? static_cast<float>(id_base + col[j]) : 0.0f;
        }
        noc_async_write(stage + rr * 2 * FROW, s_out.get_noc_addr(tr, face0 + rr * FROW), FROW);
        noc_async_write(stage + rr * 2 * FROW + FROW, s_out.get_noc_addr(tr, face0 + FACE_B + rr * FROW), FROW);
    }
    noc_semaphore_set(sem, 0);
    noc_async_write_barrier();
}
