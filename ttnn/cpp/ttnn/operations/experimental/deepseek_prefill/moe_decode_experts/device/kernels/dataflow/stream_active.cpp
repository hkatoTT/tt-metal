// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"
#include "moe_decode_experts_common.hpp"

// This core's slice (runtime arg 1) of each work item's experts, pushed `block` K tile rows (of `row_tiles` tiles) at a
// time for the compute.  Each chip holds I = 32 T columns of every expert of its mesh column, a core's slice one shard
// (its blocks contiguous in one bank).  The CB holds `depth` blocks; up to `ahead` of them are in flight at once (each
// its own transaction id), across the experts' boundaries.  One tile row of tokens (the items are groups of up to
// 64 / T chosen experts, moe_decode_experts_common.hpp): role 1, a gate | up core's (gate, up) column tiles p mod T of
// slot p / T's expert, at tile rows (T e + p mod T) rows + k, a block a shard (block j in bank j), the blocks from
// block `rot` on (runtime arg 2; so the 64 cores read from all the banks at once; the compute takes x in the same
// order); role 2, a down core's columns of the T rows of each slot's expert, at tile rows (core experts + e) T + k mod
// T.  A slot past its group's size reads nothing: the gate | up core's blocks stay as they were (its h is sent as
// zeros), and the down core copies zeros from its own L1 (the scratch's zero tiles) so that zero h times them stays
// zero.  Packed units (2-16 tile rows of tokens; the items are groups of G units from the builders' records,
// moe_decode_experts_common.hpp): role 1, a gate | up core's K slice p mod T (one block, a shard) of each of the T
// columns of slot p / T's expert; role 2, a slot's expert by its unit.
void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t core = get_arg_val<uint32_t>(1);
    const uint32_t rot = get_arg_val<uint32_t>(2);  // gate | up, one tile row: the first K block
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr uint32_t local = get_compile_time_arg_val(4);
    constexpr uint32_t block = get_compile_time_arg_val(5);
    constexpr uint32_t depth = get_compile_time_arg_val(6);
    constexpr uint32_t sliced = get_compile_time_arg_val(7);       // 1 / 2: the gate | up / down stream
    constexpr uint32_t token_rows = get_compile_time_arg_val(8);   // the routing's tile rows
    constexpr uint32_t ahead = get_compile_time_arg_val(9);        // blocks in flight (< depth)
    constexpr uint32_t T = get_compile_time_arg_val(10);           // an expert's column tiles on the chip
    constexpr uint32_t G = 64 / T;                                 // experts (units) per item
    constexpr bool units = get_compile_time_arg_val(11) != 0;      // packed units
    constexpr bool skip = get_compile_time_arg_val(12) != 0;       // a test switch: read nothing (the blocks pushed)
    // the runs' requests (of 8 tiles) alternate between NOC0 and NOC1: one NOC's reads cap at ~47 GB/s a DRAM bank,
    // both reach its ~64 (64 cores each reading runs of 8 banks: 412-428 -> 461-472 GB/s).  With dedicated NOCs this
    // RISC (on NOC0) and the control kernel (BRISC, on NOC1) both read through command buffer 1 of their own NOC, and a
    // NOC's read counters are shared: so NOC1 only once the control kernel has done its reads (it sets semaphore
    // sem_noc1: the routing, and x on the cores reading it once; a core whose control kernel keeps reading x never sets
    // it), its transaction ids set from then on, and waited on by the ids only (the kernel's entry syncs NOC0's
    // software counters only)
    constexpr bool both = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t sem_noc1 = get_compile_time_arg_val(14);
    // one row of tokens: the routing multicast to CB_SCR's start by the x sender (gu_ctrl.cpp), semaphore sem_route
    constexpr uint32_t sem_route = get_compile_time_arg_val(15);
    // packed units from builder cores (route_build.cpp): the units made from the `rows` records once they have come
    // (semaphore sem_rec), into NCRISC's own lists
    constexpr uint32_t sem_rec = get_compile_time_arg_val(16);
    // a gate | up stream on the grid column's top core (slot 0) with packed units: x's slice of the column (all the
    // tokens' K slice core mod T, a tile row's ks tiles at r ks) read into the control kernel's CB_XALL on NOC0 while
    // the records come (runtime arg 3: x), and semaphore sem_xread set for gu_ctrl.cpp to multicast it down the column
    constexpr bool xread = get_compile_time_arg_val(17) != 0;
    constexpr uint32_t cb_xall = get_compile_time_arg_val(18);
    constexpr uint32_t sem_xread = get_compile_time_arg_val(19);
    constexpr bool xskip = get_compile_time_arg_val(20) != 0;
    // NOC1 also only once every NoC multicast of the op is done (semaphore sem_go1, raised by the op's coordinator:
    // gu_ctrl.cpp, route_build.cpp), so that no NOC1 multicast meets NOC1 weight reads (they jammed now and then)
    constexpr uint32_t sem_go1 = get_compile_time_arg_val(21);
    // the in-op gather (h1_send.cpp, h1_sem: the global semaphore sem_x's L1 address, 0 off): the top core's x slice
    // comes over the fabric into its landing CB h1_land (CB_XALL from 32 tokens a mesh row, h1_t; below, the compact
    // one gu_ctrl.cpp reads into CB_XALL), the other mesh rows' by the other chips' sender cores, this mesh row's
    // (runtime args 4, 5: x_row, the mesh row) read here, then sem_x raised by h1_inc (gu_ctrl.cpp waits for the total)
    constexpr uint32_t h1_sem = get_compile_time_arg_val(22);
    constexpr uint32_t h1_t = get_compile_time_arg_val(23);
    constexpr uint32_t h1_land = get_compile_time_arg_val(24);
    constexpr uint32_t h1_inc = get_compile_time_arg_val(25);
    // packed units: a unit whose expert is the one before's (an expert of more than 32 tokens: its units next to each
    // other) takes that unit's weights from L1 in place of reading them from DRAM again (the same bytes; before, each
    // unit read its expert's W1 and W2 again, ~8.2 us a unit at 512 tokens).  Bit 1 (down, sliced 2): from this core's
    // own ring, the slot before's rows (in this block, or the block before: still in the ring), copied once the block's
    // reads are in, before it is pushed.  Bit 2 (gate | up, sliced 1): the same K slice of the same column is at the
    // same ring address on the core of the slot before (the core above: same grid column), copied from there once that
    // core has pushed the block (its semaphore dd_sem raised by it), and that core told when the copy is in (its
    // dd_sem + 1), before it reuses the ring place; within an item only (the slot before's ring has moved on by the
    // next item).  Runtime args 6, 7: the cores above and below (NoC x | y << 16; ~0: none)
    constexpr auto w_args = TensorAccessorArgs<26>();
    constexpr auto x_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto xr_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t dedupe = get_compile_time_arg_val(xr_args.next_compile_time_args_offset());  // (after them)
    constexpr uint32_t dd_sem = get_compile_time_arg_val(xr_args.next_compile_time_args_offset() + 1);
    // the column's way back in the op (rs_move.cpp), down cores: once the W2 stream is
    // done, each rounded tile row (the compute's order, 8 rs_R of them) from CB_Y into reducer q / 2's cb_own (its
    // address in this core's mailbox word; this core's 4 tiles at 4 (q % 2)), then an increment of the reducer's
    // mailbox word (the low 16 bits from the even core, the high 16 from the odd one), on NoC 0 (y-increasing:
    // straight down to the reducer rows; dn_ctrl's NoC 1 went around the grid column).  Runtime args 8 .. 10:
    // the reducer's NoC x, y and the mailbox's address
    constexpr bool rs_rows = get_compile_time_arg_val(xr_args.next_compile_time_args_offset() + 2) != 0;
    constexpr uint32_t rs_R = get_compile_time_arg_val(xr_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t cb_y = get_compile_time_arg_val(xr_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t ycols = get_compile_time_arg_val(xr_args.next_compile_time_args_offset() + 5);
    // (the progressive form: the same, by item and slot: its own branches below)
    constexpr bool dd_dn = units && sliced == 2 && (dedupe & 1) != 0;
    constexpr bool dd_gu = units && sliced == 1 && (dedupe & 2) != 0;
    const auto s_w = TensorAccessor(w_args, w_addr);
    constexpr uint32_t block_tiles = block * row_tiles;
    bool use1 = false;  // NOC1 handed over by the control kernel
    // a contiguous run of `bytes` from weight page `page` on (one bank): one request, or with NOC1 handed over requests
    // of 8 tiles alternating between the NOCs
    auto read_run = [&](uint32_t page, uint32_t dst, uint32_t bytes) {
        const uint64_t src = s_w.get_noc_addr(page, 0, 0);
        if (both && use1) {  // NOC1's address of the same bytes, for every other request
            const uint32_t step = 8 * get_local_cb_interface(cb_w).fifo_page_size;
            const uint64_t src1 = s_w.get_noc_addr(page, 0, 1);
            uint32_t i = 0;
            for (uint32_t o = 0; o < bytes; o += step, ++i) {
                const uint32_t n = bytes - o < step ? bytes - o : step;
                if (i & 1) {
                    noc_async_read(src1 + o, dst + o, n, 1);
                } else {
                    noc_async_read(src + o, dst + o, n, 0);
                }
            }
        } else {
            noc_async_read(src, dst, bytes);
        }
    };

    CircularBuffer w(cb_w), scr(cb_scr);
    const uint32_t active = scr.get_write_ptr() + scr_ncrisc_list(token_rows);  // one row: the chosen experts (uint16)
    uint32_t chosen = 0, n = 0;
    // down: T rows of zero tiles
    const uint32_t zeros = scr.get_write_ptr() + (units ? rec_zeros(token_rows, local) : scr_zeros(token_rows));
    const uint32_t rec_mine = scr.get_write_ptr() + rec_ncrisc(token_rows, local);  // packed units: NCRISC's lists
    const UnitLists ul = rec_lists(rec_mine, token_rows, local);
    DeviceZoneScopedN("ST-ALL");
    {
    DeviceZoneScopedN("ST-ROUTE");
    {
        if constexpr (units) {
            if constexpr (xread && h1_sem != 0) {
                if (core / T == 0) {  // the in-op gather: this mesh row's x slice, into its place in the landing CB
                    DeviceZoneScopedN("ST-H1");
                    const auto s_xr = TensorAccessor(xr_args, get_arg_val<uint32_t>(4));
                    const uint32_t mrow = get_arg_val<uint32_t>(5);
                    constexpr uint32_t ks = rows / T;  // (rows: W1's K tiles here)
                    constexpr uint32_t tr = h1_t >= 32 ? h1_t / 32 : 1, pb = h1_t >= 32 ? 2048 : 64 * h1_t;
                    const uint32_t j = core % T, land = get_write_ptr(h1_land);
                    for (uint32_t q = 0; q < tr; ++q) {
                        for (uint32_t i = 0; i < ks; ++i) {
                            const uint32_t page = q * rows + j * ks + i;
                            if constexpr (h1_t >= 32) {  // CB_XALL's tile (mrow tr + q, i)
                                noc_async_read(s_xr.get_noc_addr(page), land + ((mrow * tr + q) * ks + i) * pb, pb);
                            } else if constexpr (h1_t == 16) {  // rows 0-15: faces 0 and 1
                                noc_async_read(s_xr.get_noc_addr(page), land + (mrow * ks + i) * pb, pb);
                            } else {  // rows 0-7 of faces 0 and 1
                                const uint32_t d = land + (mrow * ks + i) * pb;
                                noc_async_read(s_xr.get_noc_addr(page), d, 256);
                                noc_async_read(s_xr.get_noc_addr(page, 512), d + 256, 256);
                            }
                        }
                    }
                    noc_async_read_barrier();
                    // (from 32 tokens a row: into this mesh row's 4-bit field, as the other rows' come, h1_send.cpp)
                    noc_semaphore_inc(get_noc_addr(h1_sem), h1_t >= 32 ? h1_inc << (4 * mrow) : h1_inc);
                }
            } else if constexpr (xread) {
                if (core / T == 0) {
                    if constexpr (!xskip) {
                        const auto s_x = TensorAccessor(x_args, get_arg_val<uint32_t>(3));
                        CircularBuffer xall(cb_xall);
                        constexpr uint32_t ks = rows / T;  // (rows: W1's K tiles here)
                        const uint32_t xt = get_local_cb_interface(cb_xall).fifo_page_size;
                        for (uint32_t r = 0; r < token_rows; ++r) {
                            for (uint32_t k = 0; k < ks; ++k) {
                                noc_async_read(s_x.get_noc_addr(r * rows + (core % T) * ks + k),
                                               xall.get_write_ptr() + (r * ks + k) * xt, xt);
                            }
                        }
                        noc_async_read_barrier();
                    }
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xread)) = 1;
                }
            }
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rec)), token_rows);
            invalidate_l1_cache();
            chosen = rec_units<local, token_rows>(scr.get_write_ptr(), rec_mine);
            n = (chosen + G - 1) / G;
        } else {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_route)), 1);
            chosen = chosen_in<local, token_rows>(scr.get_write_ptr(), active);  // BRISC's copy, read only
            n = sliced_items<G>(chosen);
        }
        if constexpr (sliced == 2) {
            zero_tiles(zeros, T * row_tiles * get_local_cb_interface(cb_w).fifo_page_size);
        }
    }
    }

    const uint32_t tile = get_local_cb_interface(cb_w).fifo_page_size;
    const uint32_t base = w.get_write_ptr();  // the FIFO's start: nothing pushed yet
    // gate | up: this core's grid row, the slot it takes of each item
    const uint32_t row = units && sliced == 1 ? core / T : 0;
    auto slot_of = [&](uint32_t) { return row; };
    uint32_t blocks = n * (rows / block);
    constexpr uint32_t bpi = rows / block;  // blocks an item
#ifdef MOE_PROG
    // the progressive form (moe_decode_experts_common.hpp's prog_*; packed units from builders only): gate | up, per
    // item and column c the base's K slice block, then the correction's if the slot's expert is marked (W1's slot
    // e + local); down, a block per slot (2 T rows: the base's T rows, then the correction's if marked, from the
    // core's slot local + e: the core's slice holds 2 local slots).  Dedupe as without it, by item: an item's blocks
    // start at a different ring index on each core (the items before took T or 2 T blocks), so a gate | up copy from
    // the core above takes its block from that core's own ring index (a_first, below), and the down cores copy the
    // slot before's whole block (base, and correction if marked: the same expert's) from the place before
    const uint32_t recs_l1 = scr.get_write_ptr();
    if constexpr (units && sliced == 1) {
        blocks = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const PackedUnit u = packed_unit<G>(ul, chosen, i, slot_of(i));
            blocks += T * (u.live && prog_corrected<local>(recs_l1, u.expert) ? 2 : 1);
        }
    } else if constexpr (units) {
        blocks = n * G;
    }
    uint32_t p_item = 0, p_col = 0, p_part = 0;  // gate | up: the next block's item, column, base / correction
    bool p_corr = false;                          // ... and whether its item's unit is marked
    if constexpr (units && sliced == 1) {
        if (n > 0) {
            const PackedUnit u = packed_unit<G>(ul, chosen, 0, slot_of(0));
            p_corr = u.live && prog_corrected<local>(recs_l1, u.expert);
        }
    }
#endif
    // dedupe (above): whether unit q (item q / G, slot q % G) takes its weights from unit q - 1
    auto same_as_before = [&](uint32_t q) {
        if (q == 0) {
            return false;
        }
        const PackedUnit u = packed_unit<G>(ul, chosen, q / G, q % G);
        const PackedUnit v = packed_unit<G>(ul, chosen, (q - 1) / G, (q - 1) % G);
        return u.live && v.live && u.expert == v.expert;
    };
#ifdef MOE_PROG
    // gate | up: the item of the block at each ring place (set as it is issued; still there when it is pushed), and
    // (dedupe) the core above's ring index of its first block of item p_item: its items before took T or 2 T each
    uint32_t blk_item[depth];
    uint32_t a_first = 0;
    auto unit_corr = [&](uint32_t i, uint32_t s) {
        const PackedUnit u = packed_unit<G>(ul, chosen, i, s);
        return u.live && prog_corrected<local>(recs_l1, u.expert);
    };
    auto item_of = [&](uint32_t b) { return blk_item[b % depth]; };
#else
    auto item_of = [&](uint32_t b) { return b / bpi; };
#endif
    // gate | up: block b (item b / bpi) copied from the core above (the slot before, on the grid row above), or copied
    // by the core below
    auto gu_from_above = [&](uint32_t b) {
        const uint32_t i = item_of(b), s = slot_of(i);
        return dd_gu && row > 0 && s > 0 && same_as_before(i * G + s);
    };
    auto gu_to_below = [&](uint32_t b) {
        const uint32_t i = item_of(b), s = slot_of(i);
        return dd_gu && row + 1 < G && s + 1 < G && same_as_before(i * G + s + 1);
    };
    const uint32_t above = get_arg_val<uint32_t>(6), below = get_arg_val<uint32_t>(7);
    volatile tt_l1_ptr uint32_t* dd_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dd_sem));
    volatile tt_l1_ptr uint32_t* dd_copied =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dd_sem + 1));
    uint32_t dd_in = 0, dd_out = 0;  // gate | up: blocks copied from above so far, issued to be copied below so far
    uint32_t dd_place[depth];  // gate | up: per ring place, the count dd_out of the block there if copied below, else 0
    for (uint32_t i = 0; i < depth; ++i) {
        dd_place[i] = 0;
    }
    constexpr uint32_t DD_TRID = ahead + 2;  // down: the local copies' transaction id
    // block b's reads are in: (down) its copied slots filled from the slot before, then pushed; (gate | up) the core
    // below told (if it copies b), the core above told (if b was copied from it)
    auto push_one = [&](uint32_t b) {
        noc_async_read_barrier_with_trid(b % (ahead + 1) + 1, 0);
        if constexpr (both) {
            noc_async_read_barrier_with_trid(b % (ahead + 1) + 1, 1);
        }
#ifdef MOE_PROG
        if constexpr (dd_dn && units) {  // a block a slot: the slot before's (base, correction if marked)
            if (same_as_before(b)) {
                const PackedUnit u = packed_unit<G>(ul, chosen, b / G, b % G);
                const uint32_t bytes = (prog_corrected<local>(recs_l1, u.expert) ? 2 : 1) * T * row_tiles * tile;
                noc_async_read_set_trid(DD_TRID, 0);
                noc_async_read(get_noc_addr(base + ((b - 1) % depth) * block_tiles * tile),
                               base + (b % depth) * block_tiles * tile, bytes, 0);
                noc_async_read_barrier_with_trid(DD_TRID, 0);
            }
        } else
#endif
        if constexpr (dd_dn) {
            const uint32_t k0 = (b % bpi) * block;
            for (uint32_t k = 0; k < block; k += T) {
                const uint32_t q = (b / bpi) * G + (k0 + k) / T;
                if (!same_as_before(q)) {
                    continue;
                }
                const uint32_t pk = (q - 1) * T;  // the slot before's first row among the item's (all items' rows)
                const uint32_t src =
                    base + ((pk / block) % depth) * block_tiles * tile + (pk % block) * row_tiles * tile;
                const uint32_t dst = base + (b % depth) * block_tiles * tile + k * row_tiles * tile;
                noc_async_read_set_trid(DD_TRID, 0);
                noc_async_read(get_noc_addr(src), dst, T * row_tiles * tile, 0);
                noc_async_read_barrier_with_trid(DD_TRID, 0);  // (the next slot may copy this one)
            }
        }
        w.push_back(block_tiles);
        if constexpr (dd_gu) {
            if (gu_to_below(b)) {
                noc_semaphore_inc(get_noc_addr(below & 0xFFFF, below >> 16, get_semaphore(dd_sem)), 1);
            }
            if (gu_from_above(b)) {
                noc_semaphore_inc(get_noc_addr(above & 0xFFFF, above >> 16, get_semaphore(dd_sem + 1)), 1);
            }
        }
    };
    uint32_t pushed = 0;
    for (uint32_t issued = 0; issued < blocks; ++issued) {
        if (issued - pushed == ahead) {  // the oldest in flight to the compute
            push_one(pushed);
            ++pushed;
        }
        w.reserve_back((issued - pushed + 1) * block_tiles);  // room for the ones in flight and this one
#ifdef MOE_PROG
        if constexpr (units && sliced == 1) {
            blk_item[issued % depth] = p_item;
        }
#endif
        if constexpr (dd_gu) {  // the block here before: copied by the core below by now
            const uint32_t place = issued % depth;
            if (dd_place[place] != 0) {
                noc_semaphore_wait_min(dd_copied, dd_place[place]);
            }
            dd_place[place] = gu_to_below(issued) ? ++dd_out : 0;
        }
        const uint32_t dst = base + (issued % depth) * block_tiles * tile;
        noc_async_read_set_trid(issued % (ahead + 1) + 1, 0);  // ids 1 .. ahead + 1
        if constexpr (both) {
            if (!use1) {
                invalidate_l1_cache();
                use1 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_noc1)) != 0 &&
                       *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go1)) != 0;
            }
            if (use1) {
                noc_async_read_set_trid(issued % (ahead + 1) + 1, 1);
            }
        }
#ifdef MOE_PROG
        if constexpr (units && sliced == 1) {  // (before the test switch: the item's state goes on)
            const PackedUnit u = packed_unit<G>(ul, chosen, p_item, slot_of(p_item));
            if (gu_from_above(issued)) {  // the slot above's expert: its block, at that core's ring index for it
                if constexpr (!skip) {
                    noc_semaphore_wait_min(dd_ready, ++dd_in);
                    const uint32_t at = (a_first + p_col * (p_corr ? 2 : 1) + p_part) % depth;
                    const uint32_t src = base + at * block_tiles * tile;
                    noc_async_read(get_noc_addr(above & 0xFFFF, above >> 16, src), dst, block_tiles * tile, 0);
                }
            } else if (u.live && !skip) {
                const uint32_t e = u.expert + (p_part ? local : 0);
                read_run(((e * T + p_col) * rows + (core % T) * block) * row_tiles, dst, block_tiles * tile);
            }
            if (p_part == 0 && p_corr) {
                p_part = 1;
            } else {
                p_part = 0;
                if (++p_col == T) {
                    p_col = 0;
                    if (dd_gu && row > 0) {  // (the core above's slot of this item)
                        a_first += T * (unit_corr(p_item, row - 1) ? 2 : 1);
                    }
                    if (++p_item < n) {
                        p_corr = unit_corr(p_item, slot_of(p_item));
                    }
                }
            }
            continue;
        }
#endif
        if constexpr (skip) {
            continue;
        }
#ifdef MOE_PROG
        if constexpr (units && sliced == 2) {
            const PackedUnit u = packed_unit<G>(ul, chosen, issued / G, issued % G);
            const uint32_t half = T * row_tiles * tile;
            if (dd_dn && same_as_before(issued)) {
                // (the slot before's block copied into this one's place once its reads are in: push_one)
            } else if (u.live) {
                read_run((core * 2 * local + u.expert) * T * row_tiles, dst, half);
                if (prog_corrected<local>(recs_l1, u.expert)) {
                    read_run((core * 2 * local + local + u.expert) * T * row_tiles, dst + half, half);
                }
            } else {
                noc_async_read(get_noc_addr(zeros), dst, half);
            }
            continue;
        }
#endif
        if constexpr (units && sliced == 1) {  // block = the K slice: column c's slice of the slot's expert
            const uint32_t c = issued % (rows / block);
            const PackedUnit u = packed_unit<G>(ul, chosen, issued / (rows / block), slot_of(issued / (rows / block)));
            if (gu_from_above(issued)) {  // the same bytes at the same place of the core above, once it has them
                noc_semaphore_wait_min(dd_ready, ++dd_in);
                noc_async_read(get_noc_addr(above & 0xFFFF, above >> 16, dst), dst, block_tiles * tile, 0);
            } else if (u.live) {
                const uint32_t first = (u.expert * T + c) * rows + (core % T) * block;
                read_run(first * row_tiles, dst, block_tiles * tile);
            }
            continue;
        } else if constexpr (units && sliced == 2) {  // the block's slots, each its unit's expert's T rows
            const uint32_t k0 = (issued % (rows / block)) * block;
            for (uint32_t k = 0; k < block; k += T) {
                const PackedUnit u = packed_unit<G>(ul, chosen, issued / (rows / block), (k0 + k) / T);
                if (dd_dn && same_as_before((issued / (rows / block)) * G + (k0 + k) / T)) {
                    continue;  // (filled from the slot before when the block is pushed)
                }
                if (u.live) {
                    read_run((core * local + u.expert) * T * row_tiles, dst + k * row_tiles * tile, T * row_tiles * tile);
                } else {
                    noc_async_read(get_noc_addr(zeros), dst + k * row_tiles * tile, T * row_tiles * tile);
                }
            }
            continue;
        } else if constexpr (sliced == 1) {
            const Slot slot = sliced_slot<G>(active, chosen, issued / (rows / block), core / T);
            const uint32_t j = (issued + (token_rows == 1 ? rot : 0)) % (rows / block);
            const uint32_t first = (slot.expert * T + core % T) * rows + j * block;
            if (slot.live) {
                read_run(first * row_tiles, dst, block_tiles * tile);
            }
            continue;
        } else if constexpr (sliced == 2) {
            const uint32_t k0 = (issued % (rows / block)) * block;
            // the slots' experts are in increasing order: consecutive ones are adjacent rows, read as one run
            for (uint32_t k = 0; k < block;) {
                const Slot slot = sliced_slot<G>(active, chosen, issued / (rows / block), (k0 + k) / T);
                uint32_t run = T;
                if (!slot.live) {
                    noc_async_read(get_noc_addr(zeros), dst + k * row_tiles * tile, T * row_tiles * tile);
                } else {
                    while (k + run < block) {
                        const Slot next = sliced_slot<G>(active, chosen, issued / (rows / block), (k0 + k + run) / T);
                        if (!next.live || next.expert != slot.expert + run / T) {
                            break;
                        }
                        run += T;
                    }
                    read_run(
                        (core * local + slot.expert) * T * row_tiles, dst + k * row_tiles * tile, run * row_tiles * tile);
                }
                k += run;
            }
            continue;
        }
    }
    for (; pushed < blocks; ++pushed) {
        push_one(pushed);
    }
    if constexpr (dd_gu) {  // the core below has copied all it takes from here (before this L1 goes)
        noc_semaphore_wait_min(dd_copied, dd_out);
        noc_async_atomic_barrier();
    }
    noc_async_read_set_trid(0, 0);
    if (both && use1) {
        noc_async_read_set_trid(0, 1);
    }
    if constexpr (rs_rows && sliced == 2) {
        DeviceZoneScopedN("ST-RS");
        const uint32_t rx = get_arg_val<uint32_t>(8), ry = get_arg_val<uint32_t>(9), rmail = get_arg_val<uint32_t>(10);
        volatile tt_l1_ptr uint32_t* box = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rmail);
        uint32_t own_base = 0;
        for (uint32_t k = 0; k < (1u << 25) && own_base == 0; ++k) {  // (bounded, as rs_move.cpp)
            invalidate_l1_cache();
            own_base = *box;
        }
        *box = 0;  // (the next call's address may differ: a t of its own)
        const uint32_t tile = get_tile_size(cb_y);
        const uint64_t count = get_noc_addr(rx, ry, rmail);
        for (uint32_t r = 0; r < 8 * rs_R; ++r) {
            cb_wait_front(cb_y, ycols);
            if (own_base != 0) {  // (else no address came: the rows dropped, the reducer gives up)
                const uint32_t at = own_base + (r * 8 + 4 * (core & 1)) * tile;  // (block r / rs_R, its row r % rs_R)
                noc_async_write(get_read_ptr(cb_y), get_noc_addr(rx, ry, at), ycols * tile);
                noc_async_write_barrier();
                noc_semaphore_inc(count, (core & 1) ? 1u << 16 : 1u);  // (each of the two down cores its own field)
            }
            cb_pop_front(cb_y, ycols);
            DeviceTimestampedData("ST-ROW", r);  // (a profiler build only: the row written)
        }
        noc_async_atomic_barrier();
    }
}
