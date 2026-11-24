// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2 (dflash2_ops.kv_write, GLM53_DFLASH2_KVW_RUNS=1; the context rows, mode 1): kv_write.cpp's writes - token r
// (position idx[r], user users[r]), layer l, KV head h: source row r sU + l sL + h sH, columns 128 (c0 + l cL + h cH)
// .. + 127, into cache page page_base + l layer_pages + (u heads + h) pages + p / 32, row p % 32 (none: p < 0, p past
// the pages unless RING, u past the last user) - with the NoC transfers merged:
//   - a unit is (group g of T tokens, layers LG l' .. LG l' + LG - 1, column tiles TPU q .. TPU q + TPU - 1), all heads;
//     this core takes the units [u0, u1) (at most PER);
//   - reads, every unit of the core first: per column tile t and face column half f, the unit's head rows in source
//     order (INNER_H: token-major, then the layers and heads, as the K rows (r, l, h) lie; else layer- and head-major,
//     the tokens inner, as V's rows r lie), rows that follow each other in one face of one tile read as one segment
//     (its n x FROW bytes of values and, BFP, its n exponent bytes), each staged at the source's offset mod 64 in a
//     slot of RB; then one barrier;
//   - writes: per unit, layer, head, column tile and face column half, the tokens whose rows land on consecutive rows
//     of one face of one page (the same user, the next position) go as one run, copied from RB to a slot of WB at the
//     destination's offset mod 64, n x FROW bytes and n exponent bytes (a NoC transfer under 64 bytes: the same offset
//     mod 64 at both ends); WB reused once its writes are flushed.
// The same bytes in the same places as kv_write.cpp.  At most 8 rows a segment or a run.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t users_addr = get_arg_val<uint32_t>(3);
    const uint32_t u0 = get_arg_val<uint32_t>(4);
    const uint32_t u1 = get_arg_val<uint32_t>(5);
    const uint32_t page_base = get_arg_val<uint32_t>(6);

    constexpr uint32_t T = get_compile_time_arg_val(0);      // tokens a group
    constexpr uint32_t n_idx = get_compile_time_arg_val(1);  // tokens (idx's and users' entries)
    constexpr uint32_t n_users = get_compile_time_arg_val(2);
    constexpr uint32_t layers = get_compile_time_arg_val(3);
    constexpr uint32_t heads = get_compile_time_arg_val(4);
    constexpr uint32_t src_wt = get_compile_time_arg_val(5);
    constexpr uint32_t sU = get_compile_time_arg_val(6);
    constexpr uint32_t sL = get_compile_time_arg_val(7);
    constexpr uint32_t sH = get_compile_time_arg_val(8);
    constexpr uint32_t c0 = get_compile_time_arg_val(9);
    constexpr uint32_t cL = get_compile_time_arg_val(10);
    constexpr uint32_t cH = get_compile_time_arg_val(11);
    constexpr uint32_t pages = get_compile_time_arg_val(12);
    constexpr uint32_t layer_pages = get_compile_time_arg_val(13);
    constexpr uint32_t cb = get_compile_time_arg_val(14);
    constexpr uint32_t BITS = get_compile_time_arg_val(15);
    constexpr uint32_t RING = get_compile_time_arg_val(16);
    constexpr uint32_t INNER_H = get_compile_time_arg_val(17);
    constexpr uint32_t NRB = get_compile_time_arg_val(18);  // RB's slots (every segment of the core's units)
    constexpr uint32_t NWB = get_compile_time_arg_val(19);  // WB's slots
    constexpr uint32_t LG = get_compile_time_arg_val(20);   // layers a unit
    constexpr uint32_t TPU = get_compile_time_arg_val(21);  // column tiles a unit
    constexpr uint32_t PER = get_compile_time_arg_val(22);  // units a core at most
    constexpr auto src_args = TensorAccessorArgs<23>();
    constexpr auto cache_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto users_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = BITS == 0 ? 2048 : (BITS == 8 ? 1088 : 576), SLOT = 64, WT = 4;
    constexpr uint32_t FACE = BITS == 0 ? 512 : BITS * 32;
    constexpr uint32_t FROW = BITS == 0 ? 32 : BITS * 2;
    constexpr uint32_t BASE = BITS == 0 ? 0 : 64;
    constexpr uint32_t MAX_ROWS = 8;                            // a segment's or a run's rows at most
    constexpr uint32_t MAREA = SLOT + MAX_ROWS * FROW;          // a slot's values (staged at an offset mod 64 < 64)
    constexpr uint32_t RSLOT = MAREA + (BITS != 0 ? SLOT : 0);  // then its exponent bytes (within one face: < 64)
    constexpr uint32_t idx_bytes = n_idx * 4;
    constexpr uint32_t idx_slot = (idx_bytes + 63) & ~63u;
    constexpr uint32_t ROWS_U = T * LG * heads * TPU * 2;  // a unit's (token, layer, head, column tile, half) rows
    constexpr uint32_t QS = WT / TPU;                       // column tile groups
    constexpr uint32_t UG = (layers / LG) * QS;             // units a token group
    constexpr uint32_t PMAX = pages * 32;
    static_assert(T <= MAX_ROWS && heads * LG <= MAX_ROWS && layers % LG == 0 && WT % TPU == 0, "kv_write_runs");

    const auto s_src = TensorAccessor(src_args, src_addr, TILE);
    const auto s_cache = TensorAccessor(cache_args, cache_addr, TILE);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t users_l1 = l1 + idx_slot;
    const uint32_t table = users_l1 + idx_slot;  // per row of the core's units: its values' address in RB, its exponent's
    const uint32_t rb = table + ((PER * ROWS_U * 8 + 63) & ~63u);
    const uint32_t wb = rb + NRB * RSLOT;
    noc_async_read(TensorAccessor(idx_args, idx_addr, idx_bytes).get_noc_addr(0), l1, idx_bytes);
    noc_async_read(TensorAccessor(users_args, users_addr, idx_bytes).get_noc_addr(0), users_l1, idx_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* idx = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1);
    volatile tt_l1_ptr int32_t* usr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(users_l1);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table);

    // token r's cache position (PMAX: not written)
    auto position = [&](uint32_t r) -> uint32_t {
        const int32_t p = r < n_idx ? idx[r] : -1;
        const uint32_t u = r < n_idx ? static_cast<uint32_t>(usr[r]) : n_users;
        if (p < 0 || (RING == 0 && p >= static_cast<int32_t>(PMAX)) || u >= n_users) {
            return PMAX;
        }
        return RING != 0 ? static_cast<uint32_t>(p) % PMAX : static_cast<uint32_t>(p);
    };
    auto entry = [&](uint32_t k, uint32_t j, uint32_t li, uint32_t h, uint32_t tl, uint32_t f) -> uint32_t {
        return k * ROWS_U + ((((j * LG + li) * heads + h) * TPU + tl) * 2 + f);
    };

    // reads: segments of consecutive source rows, every unit of the core
    uint32_t nseg = 0;
    for (uint32_t unit = u0; unit < u1; ++unit) {
        const uint32_t k = unit - u0;
        const uint32_t g = unit / UG, rem = unit % UG;
        const uint32_t l0 = (rem / QS) * LG, t0 = (rem % QS) * TPU;
        const uint32_t j0 = g * T;
        const uint32_t nt = n_idx - j0 < T ? n_idx - j0 : T;
        for (uint32_t tl = 0; tl < TPU; ++tl) {
            const uint32_t t = t0 + tl;
            for (uint32_t f = 0; f < 2; ++f) {
                uint32_t s_tile = 0, s_srow = 0, s_cnt = 0, s_slot = 0, s_off = 0, s_es = 0;
                auto close = [&]() {
                    if (s_cnt == 0) {
                        return;
                    }
                    noc_async_read(s_src.get_noc_addr(s_tile, s_off), s_slot + s_off % SLOT, s_cnt * FROW);
                    if constexpr (BITS != 0) {
                        noc_async_read(s_src.get_noc_addr(s_tile, s_es), s_slot + MAREA + s_es % SLOT, s_cnt);
                    }
                    s_cnt = 0;
                };
                auto take = [&](uint32_t j, uint32_t li, uint32_t h) {
                    const uint32_t r = j0 + j, l = l0 + li;
                    const uint32_t srow = r * sU + l * sL + h * sH;
                    const uint32_t col = c0 + l * cL + h * cH;
                    const uint32_t tile = (srow / 32) * src_wt + col * WT + t;
                    const uint32_t rr = srow % 32;
                    if (s_cnt == 0 || tile != s_tile || srow != s_srow + s_cnt || rr % 16 == 0 || s_cnt == MAX_ROWS) {
                        close();
                        s_tile = tile;
                        s_srow = srow;
                        s_slot = rb + nseg * RSLOT;
                        s_off = BASE + ((rr / 16) * 2 + f) * FACE + (rr % 16) * FROW;
                        s_es = ((rr / 16) * 2 + f) * 16 + rr % 16;
                        ++nseg;
                    }
                    const uint32_t e = entry(k, j, li, h, tl, f);
                    tab[2 * e] = s_slot + s_off % SLOT + s_cnt * FROW;
                    tab[2 * e + 1] = s_slot + MAREA + s_es % SLOT + s_cnt;
                    ++s_cnt;
                };
                if constexpr (INNER_H != 0) {
                    for (uint32_t j = 0; j < nt; ++j) {
                        if (position(j0 + j) == PMAX) {
                            close();
                            continue;
                        }
                        for (uint32_t li = 0; li < LG; ++li) {
                            for (uint32_t h = 0; h < heads; ++h) {
                                take(j, li, h);
                            }
                        }
                    }
                } else {
                    for (uint32_t li = 0; li < LG; ++li) {
                        for (uint32_t h = 0; h < heads; ++h) {
                            for (uint32_t j = 0; j < nt; ++j) {
                                if (position(j0 + j) == PMAX) {
                                    close();
                                    continue;
                                }
                                take(j, li, h);
                            }
                            close();
                        }
                    }
                }
                close();
            }
        }
    }
    noc_async_read_barrier();

    // writes: runs of consecutive cache rows
    uint32_t nrun = 0;
    for (uint32_t unit = u0; unit < u1; ++unit) {
        const uint32_t k = unit - u0;
        const uint32_t g = unit / UG, rem = unit % UG;
        const uint32_t l0 = (rem / QS) * LG, t0 = (rem % QS) * TPU;
        const uint32_t j0 = g * T;
        const uint32_t nt = n_idx - j0 < T ? n_idx - j0 : T;
        for (uint32_t li = 0; li < LG; ++li) {
            const uint32_t l = l0 + li;
            for (uint32_t h = 0; h < heads; ++h) {
                for (uint32_t tl = 0; tl < TPU; ++tl) {
                    const uint32_t t = t0 + tl;
                    for (uint32_t f = 0; f < 2; ++f) {
                        uint32_t j = 0;
                        while (j < nt) {
                            const uint32_t pos = position(j0 + j);
                            if (pos == PMAX) {
                                ++j;
                                continue;
                            }
                            const uint32_t u = static_cast<uint32_t>(usr[j0 + j]);
                            const uint32_t row = pos % 32;
                            uint32_t n = 1;
                            while (j + n < nt && (row + n) % 16 != 0 && position(j0 + j + n) == pos + n &&
                                   static_cast<uint32_t>(usr[j0 + j + n]) == u) {
                                ++n;
                            }
                            if (nrun == NWB) {
                                noc_async_writes_flushed();  // (WB read out before it is staged again)
                                nrun = 0;
                            }
                            const uint32_t page = page_base + l * layer_pages + (u * heads + h) * pages + pos / 32;
                            const uint32_t dst_off = BASE + ((row / 16) * 2 + f) * FACE + (row % 16) * FROW;
                            const uint32_t ed = ((row / 16) * 2 + f) * 16 + row % 16;
                            const uint32_t slot = wb + nrun * RSLOT;
                            volatile tt_l1_ptr uint32_t* to =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + dst_off % SLOT);
                            for (uint32_t m = 0; m < n; ++m) {
                                const uint32_t e = entry(k, j + m, li, h, tl, f);
                                volatile tt_l1_ptr uint32_t* from =
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab[2 * e]);
                                for (uint32_t w = 0; w < FROW / 4; ++w) {
                                    to[m * (FROW / 4) + w] = from[w];
                                }
                                if constexpr (BITS != 0) {
                                    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(slot + MAREA + ed % SLOT)[m] =
                                        reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tab[2 * e + 1])[0];
                                }
                            }
                            noc_async_write(
                                slot + dst_off % SLOT, s_cache.get_noc_addr(page * WT + t, dst_off), n * FROW);
                            if constexpr (BITS != 0) {
                                noc_async_write(slot + MAREA + ed % SLOT, s_cache.get_noc_addr(page * WT + t, ed), n);
                            }
                            ++nrun;
                            j += n;
                        }
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
}
