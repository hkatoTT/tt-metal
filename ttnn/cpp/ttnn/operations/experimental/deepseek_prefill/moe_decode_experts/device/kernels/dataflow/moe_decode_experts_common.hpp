// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The control kernels (BRISC, NOC1) with the weight stream reading on NOC1 too (MOE_EXPERTS_BOTH_NOCS,
// stream_active.cpp), before their first NOC1 transaction: this RISC's command buffers' transaction ids to 0 and the
// NIU's outstanding counters of ids 1-15 cleared.  Earlier programs' multicasts and tagged writes can leave those
// counters non-zero (deepseek_v3_b1's dram_streaming_matmul resets them likewise), and the stream waits on ids
// 1 .. ahead + 1 of NOC1 (it hung there otherwise).  The stream takes NOC1 only after this (semaphore sem_noc1).
inline void claim_noc1_ids() {
    noc_async_read_set_trid(0, noc_index);
    noc_async_write_set_trid(0, noc_index);
    ncrisc_noc_set_transaction_id(noc_index, write_reg_cmd_buf, 0);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
}

// The in-op gather's row relay (MOE_H1_RELAY, h1_relay.cpp): the relay cores' semaphores (ids, the same on both relay
// cores): the top cores in the program (each raises it) and their CB_XALL's address (the mailbox); the notices of the
// rows landed of the core's two K slices (bit m: mesh row m's, from the slice's top core); role 0 -> role 1 the groups
// read (a count), role 1 -> role 0 the groups sent (a count)
constexpr uint32_t RELAY_SEM_TOPS = 0, RELAY_SEM_MAIL = 1, RELAY_SEM_N0 = 2, RELAY_SEM_PUB = 4, RELAY_SEM_SENT = 5;
constexpr uint32_t RELAY_SEMS = 6;

// The column layout (each chip holds I = 32 T columns, T = 8, of every one of its mesh column's experts) with one tile
// row of tokens: the routing is moe_decode_route's id tile (the column's range: the chosen experts' ids in columns 0-15
// from column 0 on, their weights in 16-31, then zero weights; all zero in the rows of no token), multicast to
// `scratch` by the op's x sender (gu_ctrl.cpp).  Lists the experts some token chose, in increasing order, as uint16 at
// `active`, and returns how many.
//
// The scratch CB (CB_SCR) with one tile row of tokens: the routing tile at 0 (room for two), then BRISC's list of
// chosen experts (1 KB), NCRISC's, and the down stream's zero tiles.
constexpr uint32_t scr_brisc_list(uint32_t rows) { return 2 * rows * 4096; }
constexpr uint32_t scr_ncrisc_list(uint32_t rows) { return 2 * rows * 4096 + 1024; }
constexpr uint32_t scr_zeros(uint32_t rows) { return 2 * rows * 4096 + 2048; }
template <uint32_t experts, uint32_t rows>
inline uint32_t chosen_in(uint32_t scratch, uint32_t active) {
    uint32_t mask[(experts + 31) / 32];
    for (uint32_t w = 0; w < (experts + 31) / 32; ++w) {
        mask[w] = 0;
    }
    for (uint32_t r = 0; r < 32 * rows; ++r) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + (r / 32) * 4096);
        const uint32_t row = (r % 32 / 16) * 2 * 256 + (r % 16) * 16;  // columns 0-15; 16-31 a face (256) further
        for (uint32_t c = 0; c < 16 && static_cast<int32_t>(t[row + 256 + c]) > 0; ++c) {  // (from column 0 on)
            const uint32_t e = t[row + c];
            mask[e / 32] |= 1u << (e % 32);
        }
    }
    volatile tt_l1_ptr uint16_t* a = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(active);
    uint32_t n = 0;
    for (uint32_t w = 0; w < (experts + 31) / 32; ++w) {
        for (uint32_t b = 0; b < 32; ++b) {
            if ((mask[w] >> b) & 1u) {
                a[n++] = w * 32 + b;
            }
        }
    }
    return n;
}

// The work items with one tile row of tokens, for slices of T column tiles (I = 32 T columns of each expert on the
// chip): the n chosen experts in groups of at most G = 64 / T (64 gate | up column tiles, 64 down K tiles), as even as
// can be.  Slot s of
// item i (gate | up cores T s .. T s + T - 1, the down cores' K tiles likewise) is expert `first + s` of the list; a
// slot past its group's size repeats the group's first expert, not live (zero routing weights: its h is zero, and its
// weights are finite).
struct Slot {
    uint32_t expert;
    bool live;
};
template <uint32_t G>
inline uint32_t sliced_items(uint32_t n) {
    return (n + G - 1) / G;
}
template <uint32_t G>
inline Slot sliced_slot(uint32_t active, uint32_t n, uint32_t i, uint32_t s) {
    const uint32_t items = sliced_items<G>(n), base = n / items, extra = n % items;
    const uint32_t first = i * base + (i < extra ? i : extra);
    const uint32_t size = base + (i < extra ? 1 : 0);
    volatile tt_l1_ptr uint16_t* a = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(active);
    return {a[first + (s < size ? s : 0)], s < size};
}


// The packed units, for 2-16 tile rows of tokens (`rows` id tiles of moe_decode_route with the column's range: ids
// e < local in columns 0-15, weights in 16-31): every chosen expert's tokens in increasing order, in units of up to 32
// (a tile row of packed tokens), the units in order of expert, then chunk; item i's slot s is unit G i + s.  The units
// come from the builders' records (rec_units, below); a RISC's lists hold per unit its first pick, its size and its
// expert (uint16 each).
constexpr uint32_t unit_picks(uint32_t rows) { return rows * 32 * 16; }
constexpr uint32_t unit_max(uint32_t rows, uint32_t local) { return local + unit_picks(rows) / 32; }
struct UnitLists {
    uint32_t start, size, expert;
};
struct PackedUnit {
    uint32_t start, size, expert;
    bool live;
};
// Unit G i + s of the lists (not live past the last one: no tokens, the first unit's expert)
template <uint32_t G>
inline PackedUnit packed_unit(const UnitLists& l, uint32_t n, uint32_t i, uint32_t s) {
    const uint32_t u = G * i + s;
    volatile tt_l1_ptr uint16_t* start = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l.start);
    volatile tt_l1_ptr uint16_t* size = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l.size);
    volatile tt_l1_ptr uint16_t* expert = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l.expert);
    if (u >= n) {
        return {0, 0, expert[0], false};
    }
    return {start[u], size[u], expert[u], true};
}
// The packed units from builder cores (route_build.cpp): in place of every RISC of the op reading all the id tiles
// twice and scanning them (256 / 512 tokens: 70-90 / 116-160 us before the weight streams start), builder b reads id
// tile b once, scans its 32 tokens and multicasts its record to every gate | up and down core, at record b of their
// scratch: cnt[e] (uint8, the tile's picks of local expert e: at most 32), off[e] (uint16 from rec_off, where e's picks
// start among the record's), then the picks by expert, each in token order, a uint32 each (the weight's bf16 bits << 16
// | the token).  An expert's picks over the records in order are its picks in token order: the same units, the same
// tokens in the same packed rows.  (The scans count in registers or on the stack: read-modify-writes of L1 words, the
// earlier per-RISC lists' counts, took ~70 cycles each, 20 / 80 us of making the units at 256 / 512 tokens with uint16
// sums in L1.)
constexpr uint32_t rec_picks() { return 32 * 16; }  // a tile row's picks at most (16 columns a token)
constexpr uint32_t rec_off(uint32_t local) { return (local + 3) / 4 * 4; }
constexpr uint32_t rec_hdr(uint32_t local) { return (rec_off(local) + 2 * local + 63) / 64 * 64; }
constexpr uint32_t rec_bytes(uint32_t local) { return rec_hdr(local) + 4 * rec_picks(); }
// a RISC's unit lists: per unit its first pick among its expert's, its size and its expert (uint16 each), then the
// experts' totals (uint16)
constexpr uint32_t rec_lists_bytes(uint32_t rows, uint32_t local) {
    return (6 * unit_max(rows, local) + 2 * local + 63) / 64 * 64;
}
// the scratch with builders: the records, BRISC's unit lists, NCRISC's, the down stream's zero tiles (a builder's id
// tile where BRISC's lists are)
constexpr uint32_t rec_brisc(uint32_t rows, uint32_t local) { return rows * rec_bytes(local); }
constexpr uint32_t rec_ncrisc(uint32_t rows, uint32_t local) {
    return rec_brisc(rows, local) + rec_lists_bytes(rows, local);
}
constexpr uint32_t rec_zeros(uint32_t rows, uint32_t local) {
    return rec_ncrisc(rows, local) + rec_lists_bytes(rows, local);
}
constexpr UnitLists rec_lists(uint32_t lists, uint32_t rows, uint32_t local) {
    const uint32_t start = lists;
    return {start, start + 2 * unit_max(rows, local), start + 4 * unit_max(rows, local)};
}
// a pick's weight (fp32 bits, a positive bf16 value: (128 + m) 2^(E - 134)) in fixed point, 2^20 a unit, truncated
// (below 2^-13 bits drop): a column's sums at most 512 x 2.5 < 2^11 units fit 32 bits, and integer adds need no
// soft float (the progressive form's budget, route_build.cpp)
inline uint32_t fixed_weight(uint32_t w) {
    const uint32_t e = (w >> 23) & 0xffu, m = 128u | ((w >> 16) & 0x7fu);
    return e >= 114u ? (e - 114u < 24u ? m << (e - 114u) : 0xffffffffu) : (114u - e < 32u ? m >> (114u - e) : 0u);
}
constexpr uint32_t mask_words(uint32_t local) { return (local + 31) / 32; }
// The progressive form (MOE_PROG, a quality-changing option, off unless the program
// factory defines it from GLM53_MOE_PROG_F): each slot's weights are a base (BFP4(W)) and a correction (BFP4(W -
// base)), the corrections after the bases (W1: slot e + local; W2: a down core's slots 0 .. 2 local - 1, the
// corrections from local on).  Builder 0 marks the column's experts read with their correction this call (a budget of
// bytes, route_build.cpp) in record 0's header padding, after the counts and offsets; a marked expert's units are
// computed with both, x (h) times the base, then times the correction, into the same Dest.
constexpr uint32_t prog_mask_off(uint32_t local) { return (rec_off(local) + 2 * local + 3) / 4 * 4; }
static_assert(prog_mask_off(73) + 4 * mask_words(73) <= rec_hdr(73), "the progressive mask in record 0's header");
static_assert(prog_mask_off(72) + 4 * mask_words(72) <= rec_hdr(72), "the progressive mask in record 0's header");
template <uint32_t local>
inline bool prog_corrected(uint32_t recs, uint32_t e) {
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recs + prog_mask_off(local));
    return (m[e / 32] >> (e % 32)) & 1u;
}
// The units from the records at `recs` (once all `rows` have come): each expert's picks over the records in chunks of
// up to 32, the units in order of expert, then chunk; a unit's start is its first pick among its expert's.  Returns the number of units.
template <uint32_t local, uint32_t rows>
inline uint32_t rec_units(uint32_t recs, uint32_t lists) {
    const UnitLists l = rec_lists(lists, rows, local);
    // the uint8 counts summed four a word in two 16-bit lanes each (at most 16 x 32 a lane), on the stack
    constexpr uint32_t W = rec_off(local) / 4;
    uint32_t lo[W], hi[W];
    for (uint32_t w = 0; w < W; ++w) {
        lo[w] = 0;
        hi[w] = 0;
    }
    for (uint32_t b = 0; b < rows; ++b) {
        volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recs + b * rec_bytes(local));
        for (uint32_t w = 0; w < W; ++w) {
            const uint32_t v = cnt[w];
            lo[w] += v & 0x00ff00ffu;
            hi[w] += (v >> 8) & 0x00ff00ffu;
        }
    }
    volatile tt_l1_ptr uint16_t* start = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l.start);
    volatile tt_l1_ptr uint16_t* size = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l.size);
    volatile tt_l1_ptr uint16_t* expert = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l.expert);
    uint32_t n = 0;
    for (uint32_t e = 0; e < local; ++e) {
        const uint32_t lane = (e % 2 ? hi[e / 4] : lo[e / 4]) >> (e % 4 >= 2 ? 16 : 0);
        const uint32_t t = lane & 0xffff;
        for (uint32_t s = 0; s < t; s += 32) {
            start[n] = s;
            size[n] = t - s < 32 ? t - s : 32;
            expert[n] = e;
            ++n;
        }
    }
    return n;
}
// fn(i, pick) for the unit's picks i = 0 .. size - 1 (pick: the weight's bf16 bits << 16 | the token)
template <uint32_t local, uint32_t rows, typename F>
inline void rec_unit_picks(uint32_t recs, const PackedUnit& u, F&& fn) {
    uint32_t skip = u.start, i = 0;
    for (uint32_t b = 0; b < rows && i < u.size; ++b) {
        const uint32_t rec = recs + b * rec_bytes(local);
        const uint32_t c = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(rec)[u.expert];
        if (skip >= c) {
            skip -= c;
            continue;
        }
        const uint32_t first = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rec + rec_off(local))[u.expert] + skip;
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec + rec_hdr(local)) + first;
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
// A unit's bf16 0/1 tiles, one per tile row r of the tokens (2 KB each from `dst`): `transposed` false, packed row i x
// token column (the gather, P); true, token row x packed column i (the scatter back, P^T).  Writes `value` (1.0 or 0)
// at the unit's entries only: the tiles start zero (zero_tiles) and each use clears the entries of the unit before it.
// (`row_stride`: the bytes between the tiles of consecutive token rows, 2048 for adjacent tiles)
template <uint32_t local, uint32_t rows>
inline void rec_onehot(
    uint32_t dst, uint32_t recs, const PackedUnit& u, bool transposed, uint16_t value, uint32_t row_stride = 2048) {
    rec_unit_picks<local, rows>(recs, u, [&](uint32_t i, uint32_t pick) {
        const uint32_t n = pick & 0xffff;
        const uint32_t y = transposed ? n % 32 : i, x = transposed ? i : n % 32;
        volatile tt_l1_ptr uint16_t* t16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst + (n / 32) * row_stride);
        t16[((y / 16) * 2 + x / 16) * 256 + (y % 16) * 16 + x % 16] = value;
    });
}
// rec_onehot into bfp8_b tiles (1088 bytes: 64 shared exponents, then the 4 faces' mantissa bytes; with exponents 127
// a mantissa byte 0x40 is 1.0 and 0 is 0, both exact): the down cores' 0/1 tiles, half the size of bf16 ones, so that
// two items' fit in L1 (the next item's made while the compute is on this one)
template <uint32_t local, uint32_t rows>
inline void rec_onehot8(
    uint32_t dst, uint32_t recs, const PackedUnit& u, bool transposed, uint8_t value, uint32_t row_stride) {
    rec_unit_picks<local, rows>(recs, u, [&](uint32_t i, uint32_t pick) {
        const uint32_t n = pick & 0xffff;
        const uint32_t y = transposed ? n % 32 : i, x = transposed ? i : n % 32;
        volatile tt_l1_ptr uint8_t* t8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst + (n / 32) * row_stride + 64);
        t8[((y / 16) * 2 + x / 16) * 256 + (y % 16) * 16 + x % 16] = value;
    });
}

// `bytes` (a multiple of 512) of zeros at `dst`, read over the NoC from the core's own zero region: the RISC-V's
// stores took ~1 us a KB (35 KB of the down stream's zero tiles: a third of the op's fixed time)
// by_write: written (to this core's own L1) instead, for the control kernels once they have handed NOC1's reads to
// the weight stream (claim_noc1_ids): a read barrier there counts the stream's NOC1 reads too and never ends
template <bool by_write = false>
inline void zero_tiles(uint32_t dst, uint32_t bytes) {
    if constexpr (by_write) {
        for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
            noc_async_write(MEM_ZEROS_BASE, get_noc_addr(dst + o), MEM_ZEROS_SIZE);
        }
        noc_async_write_barrier();
        return;
    }
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + o, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
}
