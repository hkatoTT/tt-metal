// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/debug/kernel_profiler.hpp"
#include "moe_decode_experts_common.hpp"

// the h slots of the down cores' CB_H (the program factory's h_slots: GLM53_MOE_H_SLOTS or its shapes' defaults; 3
// when not set)
#ifndef MOE_H_SLOTS
#define MOE_H_SLOTS 3
#endif

#ifdef MOE_DN_PT_FAST
// The program factory's MOE_DN_PT_FAST (GLM53_MOE_DN_PT_FAST, read at the program's creation; on by default with
// MOE_DN_RELEASE_EARLY, below): an item's 0/1 tiles made in two walks of the records a slot, not three (item i - 2's
// entries cleared; item i's written, the scatter's row marks gathered in the same walk and written once at the end),
// with plain reads of the records: they no longer change once all have come (sem_rec, then invalidate_l1_cache, whose
// fence comes before these reads in the kernel's code).  The same bytes to the 0/1 tiles in the same order, the same
// marks.  As rec_unit_picks (moe_decode_experts_common.hpp)
template <uint32_t local, uint32_t rows, typename F>
inline void rec_unit_picks_plain(uint32_t recs, const PackedUnit& u, F&& fn) {
    uint32_t skip = u.start, i = 0;
    for (uint32_t b = 0; b < rows && i < u.size; ++b) {
        const uint32_t rec = recs + b * rec_bytes(local);
        const uint32_t c = reinterpret_cast<const uint8_t*>(rec)[u.expert];
        if (skip >= c) {
            skip -= c;
            continue;
        }
        const uint32_t first = reinterpret_cast<const uint16_t*>(rec + rec_off(local))[u.expert] + skip;
        const uint32_t* p = reinterpret_cast<const uint32_t*>(rec + rec_hdr(local)) + first;
        uint32_t take = c - skip;
        if (take > u.size - i) {
            take = u.size - i;
        }
        for (uint32_t k = 0; k < take; ++k) {
            fn(i++, p[k]);
        }
        skip = 0;
    }
}
// rec_onehot8's store (transposed: token row x packed column i) at the mantissas of the token's tile (`mant`: the
// slot's first tile's mantissas, 64 bytes past its exponents)
inline void onehot8_put(uint32_t mant, uint32_t pick, uint32_t i, uint32_t row_stride, uint8_t value) {
    const uint32_t n = pick & 0xffff;
    const uint32_t y = n % 32, x = i;
    volatile tt_l1_ptr uint8_t* t8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(mant + (n / 32) * row_stride);
    t8[((y / 16) * 2 + x / 16) * 256 + (y % 16) * 16 + x % 16] = value;
}
#endif

// A down core (column block q: runtime arg 2), on BRISC / NoC 1: the work items' count to the compute (mailbox); per
// item i, its slot (i mod 3) of the h buffer: reserved (the compute has let go of item i - 3), let go to its writers,
// then pushed once its 64 tiles have arrived.  h comes by unicasts in two steps (no NoC multicast, for the weight
// streams reading on NOC1 too; gu_ctrl.cpp): the 8 gate | up cores of this core's grid row write their tiles to the
// row's 4 down cores (arrivals on sem_arrive0 + slot), which pass the row's 8 tiles (16 KB) on to the 7 other down cores
// of their grid column (arrivals on sem_hvalid0 + slot).  A slot is let go to its writers by unicast: to the row's
// gate | up cores (their sem_free) and the column's down cores (their sem_dfree).  The compute's C output tiles (bf16)
// to y once at the end.  One tile row of tokens: an item is a group of up to 64 / T chosen experts
// (moe_decode_experts_common.hpp), told the x sender this core is in the program.  Packed units (2-16 tile rows): one h
// unit per item, and before it the 0/1 tiles of its G slots' units transposed for the compute (from the builders'
// records).
void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(1);  // (runtime arg 0: the routing, for the program cache's addresses)
    const uint32_t q = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);
    constexpr uint32_t local = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t sender_x = get_compile_time_arg_val(5);
    constexpr uint32_t sender_y = get_compile_time_arg_val(6);
    constexpr uint32_t h_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t sem_ready = get_compile_time_arg_val(8);
    constexpr uint32_t sem_free = get_compile_time_arg_val(9);
    constexpr uint32_t sem_arrive0 = get_compile_time_arg_val(10);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(11);  // y's tiles per tile row
    constexpr uint32_t T = get_compile_time_arg_val(12);        // an expert's column tiles on the chip
    constexpr uint32_t rows = get_compile_time_arg_val(13);     // tile rows of tokens (1, or 2-16 with packed units)
    constexpr uint32_t sem_hvalid0 = get_compile_time_arg_val(14);
    constexpr uint32_t sem_dfree = get_compile_time_arg_val(15);
    constexpr uint32_t num_dn = get_compile_time_arg_val(16);
    constexpr bool units = get_compile_time_arg_val(17) != 0;
    constexpr uint32_t cb_pt = get_compile_time_arg_val(18);
    // the weight stream reads on NOC1 too (stream_active.cpp) once this kernel has read the routing: semaphore sem_noc1
    constexpr uint32_t sem_noc1 = get_compile_time_arg_val(19);
    constexpr bool both = get_compile_time_arg_val(20) != 0;
    // one row of tokens: the routing multicast here by the x sender (gu_ctrl.cpp), semaphore sem_route
    constexpr uint32_t sem_route = get_compile_time_arg_val(21);
    // 8 / 16: y by mesh row, each row_tokens rows of tokens at the top of a tile row of its own, the rest zero (the
    // row collectives' padded layout, in place of the 0/1 matmul that spread them; 0: as the tokens)
    constexpr uint32_t row_tokens = get_compile_time_arg_val(22);
    // packed units: the compute's slots per scatter pass (dn_compute.cpp); above 1 the slots' 0/1 tiles token-row-major
    // (token row r, slot s at r G + s) for its matmul_block over the slots, else slot-major (s rows + r)
    constexpr uint32_t scatter_slots = get_compile_time_arg_val(23);
    // ... and the item's incidence after its 0/1 tiles (a tile: word r, bit s when slot s has a token in token row r)
    constexpr bool scatter_mask = get_compile_time_arg_val(24) != 0;
    // packed units from builder cores (route_build.cpp): in the program (builder 0's sem_rdy), then the `rows` records
    // (sem_rec) and the units made from them (moe_decode_experts_common.hpp's rec_*); the slots' 0/1 tiles in bfp8_b,
    // two items' (rec_onehot8): item i's made in place of item i - 2's while the compute is on item i - 1 (one item's
    // bf16 tiles held the next item's back until the compute was done)
    constexpr uint32_t b0_x = get_compile_time_arg_val(25);
    constexpr uint32_t b0_y = get_compile_time_arg_val(26);
    constexpr uint32_t sem_rdy = get_compile_time_arg_val(27);
    constexpr uint32_t sem_rec = get_compile_time_arg_val(28);
    // row_tokens with packed units: the zeros of y's padded layout (each tile's rows past the mesh row's tokens) written
    // first, while the records come (they do not depend on the routing), and at the end only the tokens' rows (two
    // writes a tile in place of six at 8 tokens a row, four at 16: the end's writes ~8 / ~4 us at 64 / 128 tokens)
    constexpr bool zeros_first = row_tokens != 0 && units;
    // packed units: the units' count to the compute (dn_compute.cpp's live_only: the last item's slots past it left
    // out) in place of the items'
    constexpr bool live_only = get_compile_time_arg_val(29) != 0 && units;
    constexpr auto y_args = TensorAccessorArgs<30>();
    const auto s_y = TensorAccessor(y_args, y_addr);
    // the column's way back in the op: the rounded tile rows go to the reducers from this
    // core's NCRISC (stream_active.cpp, NoC 0), not to y: nothing here then, CB_Y is its
    constexpr bool rs = get_compile_time_arg_val(y_args.next_compile_time_args_offset()) != 0;
    static_assert(!rs || (cols == 4 && units), "the way back in the op: 4 columns a down core, packed units");
    static_assert(units || rows == 1, "dn_ctrl: more than one tile row of tokens in packed units only");

    CircularBuffer hbuf(cb_h), y(cb_y), scr(cb_scr);
    // row_tokens: tile row `row` of the compute's y is mesh rows row * per .. row * per + per - 1, each into a tile row
    // of its own, its tokens in faces 0 and 1's top rows: the zeros after them there, and faces 2 and 3
    constexpr uint32_t per = 32 / (row_tokens ? row_tokens : 32), R = 32, F = 512;
    auto y_zeros = [&](uint32_t row, uint32_t c) {
        for (uint32_t i = 0; i < per; ++i) {
            const uint64_t dst = s_y.get_noc_addr((row * per + i) * d_tiles + q * cols + c);
            if constexpr (row_tokens < 16) {
                noc_async_write(MEM_ZEROS_BASE, dst + row_tokens * R, (16 - row_tokens) * R);
                noc_async_write(MEM_ZEROS_BASE, dst + F + row_tokens * R, (16 - row_tokens) * R);
            }
            noc_async_write(MEM_ZEROS_BASE, dst + 2 * F, F);
            noc_async_write(MEM_ZEROS_BASE, dst + 3 * F, F);
        }
    };
    uint32_t n = 0, n_units = 0, chosen = 0;
    constexpr uint32_t G = 64 / T;
    const uint32_t recs_l1 = scr.get_write_ptr();  // packed units: the records, then BRISC's unit lists
    const UnitLists ul = rec_lists(recs_l1 + rec_brisc(rows, local), rows, local);
    // one row of tokens, in the program: the x sender waits for it once (first thing, so that the sender can multicast
    // x before its routing, gu_ctrl.cpp)
    if constexpr (!units) {
        noc_semaphore_inc(get_noc_addr(sender_x, sender_y, get_semaphore(sem_ready)), 1);
    }
    {
        DeviceZoneScopedN("DC-ROUTE");
        if constexpr (both) {
            claim_noc1_ids();
        }
        if constexpr (units) {
            noc_semaphore_inc(get_noc_addr(b0_x, b0_y, get_semaphore(sem_rdy)), 1);
            if constexpr (zeros_first) {  // (the end's barrier waits for them)
                for (uint32_t r = 0; r < rows; ++r) {
                    for (uint32_t c = 0; c < cols; ++c) {
                        y_zeros(r, c);
                    }
                }
            }
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rec)), rows);
            invalidate_l1_cache();
            n_units = rec_units<local, rows>(recs_l1, recs_l1 + rec_brisc(rows, local));
            n = (n_units + G - 1) / G;
        } else {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_route)), 1);
            chosen = chosen_in<local, rows>(scr.get_write_ptr(), scr.get_write_ptr() + scr_brisc_list(rows));
            n = sliced_items<G>(chosen);
        }
    }
    if constexpr (both) {  // no more reads here (they share NOC1's read command buffer and counters)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_noc1)) = 1;
    }
    // to the compute: one row of tokens, the chosen experts' count (it makes the items and their sizes from it, as
    // sliced_slot); packed units, the items' (or the units', live_only)
    const uint32_t mail = !units ? chosen : (live_only ? n_units : n);
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, mail);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, mail);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, mail);

    const uint32_t tile = get_local_cb_interface(cb_y).fifo_page_size;
    auto write_y = [&](uint32_t row) {  // the compute's C tiles of y tile row `row`
        y.wait_front(cols);
        for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t src = y.get_read_ptr() + c * tile;
            if constexpr (row_tokens == 0) {
                noc_async_write(src, s_y.get_noc_addr(row * d_tiles + q * cols + c), tile);
            } else {
                // mesh row m's rows (within one half: faces 2 h and 2 h + 1, bf16 face rows of 32 bytes) into faces 0
                // and 1 of tile row m, then zeros (unless written first)
                for (uint32_t i = 0; i < per; ++i) {
                    const uint32_t y0 = i * row_tokens;
                    const uint32_t f = src + (y0 / 16) * 2 * F + (y0 % 16) * R;
                    const uint64_t dst = s_y.get_noc_addr((row * per + i) * d_tiles + q * cols + c);
                    noc_async_write(f, dst, row_tokens * R);
                    noc_async_write(f + F, dst + F, row_tokens * R);
                }
                if constexpr (!zeros_first) {
                    y_zeros(row, c);
                }
            }
        }
        noc_async_write_barrier();
        y.pop_front(cols);
    };
    uint32_t released = 0;  // the items whose slot was let go to its writers
    CircularBuffer pt(cb_pt);
    for (uint32_t i = 0; i < n; ++i) {  // the items
        DeviceZoneScopedN("DC-H");  // (one instance per item)
#ifdef MOE_DN_RELEASE_EARLY
        // The program factory's MOE_DN_RELEASE_EARLY (GLM53_MOE_DN_RELEASE_EARLY, read at the program's creation): item
        // i's h slot reserved and let go to its writers (the row's gate | up cores, the column's down cores) before its
        // 0/1 tiles are made, not after: the same slot on the same condition (the compute has popped item
        // i - MOE_H_SLOTS) to the same receivers, so the same values; the h's arrival and the column's exchange then
        // overlap the 0/1 tiles' making (~16 us an item at 16 tile rows) instead of following it, and the gate | up
        // cores do not wait for it.  Safe only while the compute never waits for item j's 0/1 tiles holding an h of item
        // j - MOE_H_SLOTS or before (dn_compute.cpp pops each item's h and 0/1 tiles together at the item's end, before
        // it waits for the next item's): a compute that popped its 0/1 tiles first and kept an item's h until a later
        // item's 0/1 tiles would stall here, the reserve waiting for that pop while the compute waits for this item's
        // 0/1 tiles.  (The release below then finds item i let go already.)
        {
            DeviceZoneScopedN("DC-RESE");  // (waiting for the compute to free an h slot, early)
            hbuf.reserve_back(h_tiles);
        }
        {
            constexpr uint32_t gu_row = 8, dn_rows = 8;  // (as below)
            const uint32_t gy = q / (num_dn / dn_rows);
            for (uint32_t j = 0; j < gu_row; ++j) {
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3 + 2 * j), get_arg_val<uint32_t>(4 + 2 * j),
                                               get_semaphore(sem_free)),
                                  1);
            }
            for (uint32_t r = 0; r < dn_rows; ++r) {
                if (r != gy) {
                    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(19 + 2 * r), get_arg_val<uint32_t>(20 + 2 * r),
                                                   get_semaphore(sem_dfree)),
                                      1);
                }
            }
            released = i + 1;
        }
#endif
#ifdef MOE_PROG
        // (the progressive form: one more tile, its word 31 the item's slots read with their correction, bit s)
        constexpr uint32_t pt_tiles = G * rows + 1;
#else
        constexpr uint32_t pt_tiles = G * rows + (scatter_mask ? 1 : 0);
#endif
        constexpr uint32_t ptile = 1088;  // (bfp8_b)
        if constexpr (units) {  // two CB slots: item i - 2's entries cleared
            pt.reserve_back(pt_tiles);
            constexpr uint32_t slot_stride = scatter_slots > 1 ? ptile : rows * ptile;
            constexpr uint32_t row_stride = scatter_slots > 1 ? G * ptile : ptile;
            if (i < 2) {  // the slot's first use: zero mantissas, exponents 127
                zero_tiles<both>(pt.get_write_ptr(), G * rows * ptile);
                for (uint32_t t = 0; t < G * rows; ++t) {
                    volatile tt_l1_ptr uint32_t* e =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt.get_write_ptr() + t * ptile);
                    for (uint32_t w = 0; w < 16; ++w) {
                        e[w] = 0x7f7f7f7fu;
                    }
                }
            }
#ifdef MOE_DN_PT_FAST
            {
                uint32_t marks[rows];  // the scatter's row marks (bit s: a pick of slot s in that tile row of tokens)
                for (uint32_t r = 0; r < rows; ++r) {
                    marks[r] = 0;
                }
                for (uint32_t s = 0; s < G; ++s) {
                    const uint32_t mant = pt.get_write_ptr() + s * slot_stride + 64;
                    if (i >= 2) {
                        rec_unit_picks_plain<local, rows>(
                            recs_l1, packed_unit<G>(ul, n_units, i - 2, s),
                            [&](uint32_t k, uint32_t pick) { onehot8_put(mant, pick, k, row_stride, 0); });
                    }
                    rec_unit_picks_plain<local, rows>(
                        recs_l1, packed_unit<G>(ul, n_units, i, s), [&](uint32_t k, uint32_t pick) {
                            onehot8_put(mant, pick, k, row_stride, 0x40);
                            marks[(pick & 0xffff) / 32] |= 1u << s;
                        });
                }
                if constexpr (scatter_mask) {
                    volatile tt_l1_ptr uint32_t* hits =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt.get_write_ptr() + G * rows * ptile);
                    for (uint32_t r = 0; r < rows; ++r) {
                        hits[r] = marks[r];
                    }
                }
            }
#else
            for (uint32_t s = 0; s < G; ++s) {
                const uint32_t dst = pt.get_write_ptr() + s * slot_stride;
                if (i >= 2) {
                    rec_onehot8<local, rows>(dst, recs_l1, packed_unit<G>(ul, n_units, i - 2, s), true, 0, row_stride);
                }
                rec_onehot8<local, rows>(dst, recs_l1, packed_unit<G>(ul, n_units, i, s), true, 0x40, row_stride);
            }
            if constexpr (scatter_mask) {
                volatile tt_l1_ptr uint32_t* hits =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt.get_write_ptr() + G * rows * ptile);
                for (uint32_t r = 0; r < rows; ++r) {
                    hits[r] = 0;
                }
                for (uint32_t s = 0; s < G; ++s) {
                    rec_unit_picks<local, rows>(recs_l1, packed_unit<G>(ul, n_units, i, s), [&](uint32_t, uint32_t pick) {
                        hits[(pick & 0xffff) / 32] |= 1u << s;
                    });
                }
            }
#endif
#ifdef MOE_PROG
            {
                uint32_t marked = 0;
                for (uint32_t s = 0; s < G; ++s) {
                    const PackedUnit u = packed_unit<G>(ul, n_units, i, s);
                    marked |= (u.live && prog_corrected<local>(recs_l1, u.expert) ? 1u : 0u) << s;
                }
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt.get_write_ptr() + G * rows * ptile)[31] = marked;
            }
#endif
            pt.push_back(pt_tiles);
        }
#ifndef MOE_DN_RELEASE_EARLY
        {
            DeviceZoneScopedN("DC-RES");  // (waiting for the compute to free an h slot)
            hbuf.reserve_back(h_tiles);
        }
#endif
        volatile tt_l1_ptr uint32_t* arrived =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_arrive0 + i % MOE_H_SLOTS));
        {
            constexpr uint32_t gu_row = 8, dn_rows = 8, row_tiles_h = h_tiles / dn_rows;  // a grid row's h tiles
            const uint32_t gy = q / (num_dn / dn_rows);  // this core's grid row
            auto release_one = [&]() {  // the next item's slot is free here: to its writers
                for (uint32_t j = 0; j < gu_row; ++j) {
                    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3 + 2 * j), get_arg_val<uint32_t>(4 + 2 * j),
                                                   get_semaphore(sem_free)),
                                      1);
                }
                for (uint32_t r = 0; r < dn_rows; ++r) {
                    if (r != gy) {
                        noc_semaphore_inc(
                            get_noc_addr(get_arg_val<uint32_t>(19 + 2 * r), get_arg_val<uint32_t>(20 + 2 * r),
                                         get_semaphore(sem_dfree)),
                            1);
                    }
                }
                ++released;
            };
            while (released <= i) {  // item i (its slot reserved above)
                release_one();
            }
            {
                DeviceZoneScopedN("DC-ARR");  // (waiting for the grid row's gate | up cores' tiles)
                noc_semaphore_wait_min(arrived, gu_row * (i / MOE_H_SLOTS + 1));  // the row's tiles
            }
            noc_semaphore_wait_min(  // the column's down cores have let this slot go
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_dfree)), (dn_rows - 1) * (i + 1));
            const uint32_t t = get_local_cb_interface(cb_h).fifo_page_size;
            const uint32_t src = hbuf.get_write_ptr() + gy * row_tiles_h * t;
            for (uint32_t r = 0; r < dn_rows; ++r) {
                if (r != gy) {
                    noc_async_write(
                        src, get_noc_addr(get_arg_val<uint32_t>(19 + 2 * r), get_arg_val<uint32_t>(20 + 2 * r), src),
                        row_tiles_h * t);
                }
            }
            noc_async_write_barrier();
            const uint32_t col_sem = get_semaphore(sem_hvalid0 + i % MOE_H_SLOTS);
            for (uint32_t r = 0; r < dn_rows; ++r) {
                if (r != gy) {
                    noc_semaphore_inc(
                        get_noc_addr(get_arg_val<uint32_t>(19 + 2 * r), get_arg_val<uint32_t>(20 + 2 * r), col_sem), 1);
                }
            }
            DeviceZoneScopedN("DC-COL");  // (waiting for the other grid rows' tiles)
            noc_semaphore_wait_min(  // the other rows' tiles from the column
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(col_sem), (dn_rows - 1) * (i / MOE_H_SLOTS + 1));
        }
        hbuf.push_back(h_tiles);
    }

    if constexpr (!rs) {  // (with the way back in the op the rows go from stream_active.cpp)
        DeviceZoneScopedN("DC-Y");
        for (uint32_t r = 0; r < rows; ++r) {
            write_y(r);
        }
    }
    noc_async_atomic_barrier();
}
