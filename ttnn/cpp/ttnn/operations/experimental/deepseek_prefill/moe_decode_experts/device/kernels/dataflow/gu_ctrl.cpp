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

// A gate | up core (column tile p: runtime arg 2 + 2 num_dn), on BRISC / NoC 1: the work items' count to the compute
// (mailbox).  Each chip holds I = 32 T columns of every expert of its mesh column; an item is a group of up to 64 / T
// chosen experts, and this core takes column p mod T of the group's expert p / T (moe_decode_experts_common.hpp).
// One tile row of tokens: x into L1 (core 0 reads it and multicasts it, and the routing, to the other cores, once);
// per item the routing column (bf16, column 0 = this slot's expert's weight per token, from the chosen ids and weights)
// for the compute; then the compute's h tile written into slot (i mod 3) of the h buffer of the 4 down cores of this
// core's grid row at tile p (they pass the row's tiles on down their grid columns, dn_ctrl.cpp), once all 4 have let
// the slot go (their count on S_FREE), and their arrival count for the slot raised; a slot past its group's size sends
// zeros (its weights were not read).  Packed units (2-16 tile rows of tokens; this core slot s = p / T's K slice
// j = p mod T and column j): the tokens' K slice into L1 once (the grid column's top core's, multicast down the
// column); per item its unit's 0/1 tiles and routing column for the compute (from the builders' records); then the
// compute's partial (gate, up) pairs of the T columns, each to its column's core in the row (slot i mod 2 of their
// CB_PIN, pair j; its own too), the arrivals counted on S_PARRIVE (per slot) and the slot reused once every receiver
// has summed item i - 2 (S_PFREE); the T pairs of this core's column to the compute, and its h tile on as above.
void kernel_main() {
    const uint32_t routing_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_wcol = get_compile_time_arg_val(1);
    constexpr uint32_t cb_hout = get_compile_time_arg_val(2);
    constexpr uint32_t cb_h = get_compile_time_arg_val(3);  // (the down cores' only)
    constexpr uint32_t cb_scr = get_compile_time_arg_val(4);
    constexpr uint32_t local = get_compile_time_arg_val(5);
    constexpr uint32_t kd = get_compile_time_arg_val(6);  // x tiles
    constexpr uint32_t sender_x = get_compile_time_arg_val(7);
    constexpr uint32_t sender_y = get_compile_time_arg_val(8);
    constexpr uint32_t gu_x0 = get_compile_time_arg_val(9);
    constexpr uint32_t gu_y0 = get_compile_time_arg_val(10);
    constexpr uint32_t gu_x1 = get_compile_time_arg_val(11);
    constexpr uint32_t gu_y1 = get_compile_time_arg_val(12);
    constexpr uint32_t num_gu = get_compile_time_arg_val(13);
    constexpr uint32_t num_dn = get_compile_time_arg_val(14);
    constexpr uint32_t h_tiles = get_compile_time_arg_val(15);  // h tiles per expert (= num_gu)
    constexpr uint32_t sem_ready = get_compile_time_arg_val(16);
    constexpr uint32_t sem_xvalid = get_compile_time_arg_val(17);
    constexpr uint32_t sem_free = get_compile_time_arg_val(18);
    constexpr uint32_t sem_arrive0 = get_compile_time_arg_val(19);
    constexpr uint32_t gu_x_offset = get_compile_time_arg_val(20);
    constexpr uint32_t T = get_compile_time_arg_val(21);  // an expert's column tiles on the chip
    constexpr uint32_t G = 64 / T;                        // experts (units) per item
    constexpr uint32_t rows = get_compile_time_arg_val(22);  // tile rows of tokens (1, or 2-16 with packed units)
    constexpr bool units = get_compile_time_arg_val(23) != 0;
    constexpr uint32_t cb_xall = get_compile_time_arg_val(24);
    constexpr uint32_t cb_p = get_compile_time_arg_val(25);
    constexpr uint32_t cb_part = get_compile_time_arg_val(26);
    constexpr uint32_t cb_pin = get_compile_time_arg_val(27);
    constexpr uint32_t sem_parrive0 = get_compile_time_arg_val(28);
    constexpr uint32_t sem_pfree0 = get_compile_time_arg_val(29);  // per slot
    constexpr bool xskip = get_compile_time_arg_val(30) != 0;  // a test switch: x not read from DRAM (the times)
    // the weight stream reads on NOC1 too (stream_active.cpp) once this kernel has done its reads: semaphore sem_noc1
    // (one tile row of tokens: the gate | up streams; packed units: the down ones only)
    constexpr uint32_t sem_noc1 = get_compile_time_arg_val(31);
    constexpr bool both = get_compile_time_arg_val(32) != 0;
    // one row of tokens: the x sender reads the routing once and multicasts it to every core of the op (at CB_SCR's
    // start on each), which wait for semaphore sem_route, in place of each RISC reading it from DRAM (~290 readers of
    // one 4 KB page: 8-15 us).  The down cores' rectangle (NoC 1) for that
    constexpr uint32_t sem_route = get_compile_time_arg_val(33);
    constexpr uint32_t dn_x0 = get_compile_time_arg_val(34);
    constexpr uint32_t dn_y0 = get_compile_time_arg_val(35);
    constexpr uint32_t dn_x1 = get_compile_time_arg_val(36);
    constexpr uint32_t dn_y1 = get_compile_time_arg_val(37);
    // packed units: after a unit's 0/1 tiles, a tile whose word 0 marks the token rows the unit has tokens in (bit r),
    // for the compute's gather to leave the others out (their tiles are zero: the same sums)
    constexpr bool gather_mask = get_compile_time_arg_val(38) != 0;
    // packed units from builder cores (route_build.cpp): this core says it is in the program (builder 0's sem_rdy),
    // then waits for the `rows` records (sem_rec) and makes its units from them (moe_decode_experts_common.hpp's rec_*)
    constexpr uint32_t b0_x = get_compile_time_arg_val(39);
    constexpr uint32_t b0_y = get_compile_time_arg_val(40);
    constexpr uint32_t sem_rdy = get_compile_time_arg_val(41);
    constexpr uint32_t sem_rec = get_compile_time_arg_val(42);
    // packed units: the grid column's top core's x slice read by the core's weight stream on NOC0 while it waits for
    // the records (stream_active.cpp; semaphore sem_xread when they are in): no NOC1 reads of DRAM next to the records'
    // multicasts (256 / 512 tokens 611.7 / 635.1 us against 622.4 / 653.3 read here before them)
    constexpr uint32_t sem_xread = get_compile_time_arg_val(43);
    // h by unicasts in two steps (no NoC multicast: the weight streams may read on NOC1 too): this core's tile to the 4
    // down cores of its grid row (they pass the row's tiles on down their columns, dn_ctrl.cpp), once all 4 have let
    // the slot go (sem_free, one each a unit).  The weight streams read on NOC1 only once every NoC multicast of the op
    // is done: one row of tokens, the x sender (the op's only multicasts: the routing and x) then raises every core's
    // sem_go1 (runtime args 87 ..: the gate | up cores' NoC coordinates, then the down cores' from 2); packed units,
    // each grid column's top core raises builder 0's sem_xdone once its x slice's multicast is done, and builder 0 then
    // every core's sem_go1 (route_build.cpp)
    constexpr uint32_t sem_go1 = get_compile_time_arg_val(44);
    constexpr uint32_t sem_xdone = get_compile_time_arg_val(45);
    // the in-op gather (h1_send.cpp; h1_sem: the global semaphore sem_x's L1 address, 0 off): x comes over the fabric
    // straight into the grid column's top core's CB_XALL.  The top core first puts that address in the sender core's
    // mailbox semaphore (h1_mail; the sender's NoC x, y in runtime args 87, 88) and raises its count (h1_tops), then,
    // in place of reading x, waits for h1_count increments of h1_sem (and sets it back to 0)
    constexpr uint32_t h1_sem = get_compile_time_arg_val(46);
    constexpr uint32_t h1_count = get_compile_time_arg_val(47);
    constexpr uint32_t h1_tops = get_compile_time_arg_val(48);
    constexpr uint32_t h1_mail = get_compile_time_arg_val(49);
    // ... below 32 tokens a mesh row (h1_t: the tokens a row), x lands compact in CB h1_land (mesh row s's K slice rows
    // at s 16 pb, pb = 64 h1_t a tile), put into CB_XALL's tiles here
    constexpr uint32_t h1_t = get_compile_time_arg_val(50);
    constexpr uint32_t h1_land = get_compile_time_arg_val(51);
    constexpr auto r_args = TensorAccessorArgs<52>();
    constexpr auto x_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    const auto s_x = TensorAccessor(x_args, x_addr);
    static_assert(units || rows == 1, "gu_ctrl: more than one tile row of tokens in packed units only");

    const uint32_t p = get_arg_val<uint32_t>(2 + 2 * num_dn);
    const bool sender = get_absolute_logical_y() == 0 && get_absolute_logical_x() == gu_x_offset;
    CircularBuffer x(cb_x), wcol(cb_wcol), hout(cb_hout), scr(cb_scr);
    const uint32_t routing_l1 = scr.get_write_ptr();
    const uint32_t active = scr.get_write_ptr() + scr_brisc_list(rows);  // one row: the chosen experts (uint16)
    uint32_t chosen = 0, n = 0;
    const uint32_t recs_l1 = routing_l1;  // packed units: the builders' records, then BRISC's unit lists
    const UnitLists ul = rec_lists(recs_l1 + rec_brisc(rows, local), rows, local);
    constexpr uint32_t x_tiles = kd;  // one row of tokens: x (CB_X)
    const uint32_t x_tile = get_local_cb_interface(cb_x).fifo_page_size;
    volatile tt_l1_ptr uint32_t* xvalid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xvalid));
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready));
    // One row of tokens with x read once: the sender reads and multicasts x before the routing, which x does not
    // depend on (the routing's read and scan took 8-19 us of the cores' critical path), and the receivers tell
    // it they are in the program first thing (then wait for x after their routing).  The sender's x reads go out
    // before it waits for them (into its own L1), in flight while the other cores enter the program.  (x read and
    // multicast by the first row's 8 cores, 32 KB each, was slower: 8 multicasts into one rectangle queue up, x
    // reached the others ~40 us in against ~24.)
    constexpr bool x_first = !units;
    if constexpr (x_first) {
        DeviceZoneScopedN("GC-X");
        if (sender) {
            if constexpr (both) {
                claim_noc1_ids();
            }
            x.reserve_back(x_tiles);
            const uint32_t dst = x.get_write_ptr();
            if constexpr (!xskip) {
                for (uint32_t t = 0; t < kd; ++t) {
                    noc_async_read(s_x.get_noc_addr(t), dst + t * x_tile, x_tile);
                }
            }
            noc_semaphore_wait_min(ready, num_dn + num_gu - 1);  // every core in the program (semaphores set up)
            {  // the routing first
                const auto s_r = TensorAccessor(r_args, routing_addr);
                for (uint32_t t = 0; t < rows; ++t) {
                    noc_async_read(s_r.get_noc_addr(t), routing_l1 + t * 4096, 4096);
                }
                noc_async_read_barrier();
                noc_async_write_multicast(routing_l1, get_noc_multicast_addr(gu_x0, gu_y0, gu_x1, gu_y1, routing_l1),
                                          rows * 4096, num_gu - 1);
                noc_async_write_multicast(routing_l1, get_noc_multicast_addr(dn_x0, dn_y0, dn_x1, dn_y1, routing_l1),
                                          rows * 4096, num_dn);
                noc_async_write_barrier();
                const uint32_t sem = get_semaphore(sem_route);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) = 1;
                noc_semaphore_set_multicast(sem, get_noc_multicast_addr(gu_x0, gu_y0, gu_x1, gu_y1, sem), num_gu - 1);
                noc_semaphore_set_multicast(sem, get_noc_multicast_addr(dn_x0, dn_y0, dn_x1, dn_y1, sem), num_dn);
            }
            noc_async_write_multicast(
                dst, get_noc_multicast_addr(gu_x0, gu_y0, gu_x1, gu_y1, dst), x_tiles * x_tile, num_gu - 1);
            noc_async_write_barrier();
            *xvalid = 1;
            noc_semaphore_set_multicast(
                get_semaphore(sem_xvalid),
                get_noc_multicast_addr(gu_x0, gu_y0, gu_x1, gu_y1, get_semaphore(sem_xvalid)),
                num_gu - 1);
            {  // the op's multicasts are done: NOC1's reads may start everywhere
                noc_async_write_barrier();
                const uint32_t go = get_semaphore(sem_go1);
                for (uint32_t c = 0; c < num_gu; ++c) {
                    noc_semaphore_inc(
                        get_noc_addr(get_arg_val<uint32_t>(87 + 2 * c), get_arg_val<uint32_t>(88 + 2 * c), go), 1);
                }
                for (uint32_t q = 0; q < num_dn; ++q) {
                    noc_semaphore_inc(
                        get_noc_addr(get_arg_val<uint32_t>(2 + 2 * q), get_arg_val<uint32_t>(3 + 2 * q), go), 1);
                }
            }
            x.push_back(x_tiles);
        } else {
            noc_semaphore_inc(get_noc_addr(sender_x, sender_y, get_semaphore(sem_ready)), 1);
        }
    }
    if constexpr (h1_sem != 0 && units) {
        if (p / T == 0) {  // a top core: in the program, its CB_XALL's address
            const uint32_t sx = get_arg_val<uint32_t>(87), sy = get_arg_val<uint32_t>(88);
            noc_inline_dw_write(
                get_noc_addr(sx, sy, get_semaphore(h1_mail)), get_write_ptr(h1_t < 32 ? h1_land : cb_xall));
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(h1_tops)), 1);
#ifdef MOE_H1_RELAY
            // ... and to the 2 relay cores (h1_relay.cpp; runtime args 91 .. 94): CB_XALL's address, in the program
            for (uint32_t d = 0; d < 2; ++d) {
                const uint32_t rx = get_arg_val<uint32_t>(91 + 2 * d), ry = get_arg_val<uint32_t>(92 + 2 * d);
                noc_inline_dw_write(get_noc_addr(rx, ry, get_semaphore(RELAY_SEM_MAIL)), get_write_ptr(cb_xall));
            }
            noc_async_write_barrier();
            for (uint32_t d = 0; d < 2; ++d) {
                const uint32_t rx = get_arg_val<uint32_t>(91 + 2 * d), ry = get_arg_val<uint32_t>(92 + 2 * d);
                noc_semaphore_inc(get_noc_addr(rx, ry, get_semaphore(RELAY_SEM_TOPS)), 1);
            }
#endif
        }
    }
    bool x_early = false;  // (MOE_H1_RELAY: x passed down the column from the records' wait on, row by row)
    {
        DeviceZoneScopedN("GC-ROUTE");
        if constexpr (both) {
            if (!(x_first && sender)) {
                claim_noc1_ids();
            }
        }
        if constexpr (units) {
            noc_semaphore_inc(get_noc_addr(b0_x, b0_y, get_semaphore(sem_rdy)), 1);
#ifdef MOE_H1_RELAY
            // MOE_H1_RELAY (the row relay, MoeDecodeExpertsParams::h1_relay, from 32 tokens a mesh row): a top core
            // whose slice this column carries (h1_send.cpp: slices 4 half .. 4 half + 3) tells the relay core of its
            // slice (h1_relay.cpp) of each other mesh row's as soon as it has landed (its own 4-bit field of sem_x),
            // for it to pass on to a row neighbour, while the records come; and each row's tile rows go down the grid
            // column once the records are in and that row has landed (as GC-H1's below).  Not before the records: these
            // multicasts (NoC 1) and builder 0's multicast of the records (route_build.cpp, NoC 1) into the same cores
            // at once jammed NoC 1 (the builder's multicast never done, its acks short by 96, with the column's order
            // of slices that lands rows early); with the records in, the builder's multicasts of them are done (its
            // write barrier before the semaphore), as in GC-H1.  The column's other cores are in the program by then
            // (this one is, and the program before is done on every core); their x valid flag only after the records,
            // as before.  The same writes of the same bytes to the same places: the same x
            static_assert(h1_sem != 0 && h1_t >= 32, "MOE_H1_RELAY: the in-op gather from 32 tokens a mesh row");
            if (p / T == 0) {
                DeviceZoneScopedN("GC-RX");
                constexpr uint32_t per_row = h1_count / 8, ks = kd / T;
                static_assert(per_row * 8 == h1_count && per_row < 16, "MOE_H1_RELAY: a row's count in 4 bits");
                const uint32_t base = get_write_ptr(cb_xall);
                // runtime args 89, 90: this chip's mesh row and its column's parity; 91 .. 94: the relay cores (E,
                // W), slices 4 half + 2 d, 4 half + 2 d + 1 at relay core d, their notices in RELAY_SEM_N0 + (j mod 2)
                const uint32_t mrow = get_arg_val<uint32_t>(89), half = get_arg_val<uint32_t>(90);
                const uint32_t jj = p;  // (a top core: its slice)
                const bool carried = jj / 4 == half;
                const uint32_t dd = (jj % 4) / 2;
                const uint64_t notice = get_noc_addr(get_arg_val<uint32_t>(91 + 2 * dd),
                                                     get_arg_val<uint32_t>(92 + 2 * dd),
                                                     get_semaphore(RELAY_SEM_N0 + jj % 2));
                // (GLM53_MOE_X_BFP8: CB_XALL's tiles BFP8_b)
                const uint32_t tr = h1_t / 32, rbytes = tr * ks * get_local_cb_interface(cb_xall).fifo_page_size;
                const uint32_t a = 3 + 2 * num_dn + 2 * T;  // the column's rectangle (the runtime args after the peers)
                volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h1_sem);
                volatile tt_l1_ptr uint32_t* recs_in =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rec));
                uint32_t done = 0, told = 0;  // (the rows passed down the column; those told to the relay core)
                bool parsed = false;
                while (done != 0xffu || !parsed) {
                    invalidate_l1_cache();
                    const uint32_t v = *landed;
                    for (uint32_t m = 0; m < 8; ++m) {
                        if (((done >> m) & 1u) != 0 || ((v >> (4 * m)) & 0xfu) < per_row) {
                            continue;
                        }
                        if (carried && m != mrow && ((told >> m) & 1u) == 0) {  // (first: the relay takes longest)
                            noc_semaphore_inc(notice, 1u << m);
                            told |= 1u << m;
                        }
                        if (!parsed) {  // (the column's multicast only with the records in: above)
                            continue;
                        }
                        noc_async_write_multicast(
                            base + m * rbytes,
                            get_noc_multicast_addr(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1),
                                                   get_arg_val<uint32_t>(a + 2), get_arg_val<uint32_t>(a + 3),
                                                   base + m * rbytes),
                            rbytes,
                            G - 1);
                        done |= 1u << m;
                    }
                    if (!parsed && *recs_in >= rows) {
                        invalidate_l1_cache();
                        chosen = rec_units<local, rows>(recs_l1, recs_l1 + rec_brisc(rows, local));
                        n = (chosen + G - 1) / G;
                        parsed = true;
                    }
                }
                noc_async_write_barrier();
                noc_semaphore_set(landed, 0);
                x_early = true;
            } else
#endif
            {
                {
                    DeviceZoneScopedN("GC-RWAIT");
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rec)), rows);
                }
                invalidate_l1_cache();
                chosen = rec_units<local, rows>(recs_l1, recs_l1 + rec_brisc(rows, local));
                n = (chosen + G - 1) / G;
            }
        } else {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_route)), 1);
            chosen = chosen_in<local, rows>(routing_l1, active);
            n = sliced_items<G>(chosen);
        }
    }
    // packed units: the slot this core takes of each item (its grid row)
    const uint32_t g_row = p / T;
    auto slot_of = [&](uint32_t) { return g_row; };
    auto free_noc1 = [&]() {  // no more reads here (they share NOC1's read command buffer and counters)
        if constexpr (both) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_noc1)) = 1;
        }
    };
    if (!units && !sender) {  // x comes by multicast
        free_noc1();
    }
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, n);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, n);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, n);


    // the routing column of item i for the compute (one row of tokens): column 0 of a bf16 tile, token r's weight for
    // the item's slot p / T (zero for a slot past its group's size)
    volatile tt_l1_ptr uint32_t* routing = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(routing_l1);
    auto push_wcol = [&](uint32_t i) {
        const Slot slot = sliced_slot<G>(active, chosen, i, p / T);
        wcol.reserve_back(1);
        zero_tiles<both>(wcol.get_write_ptr(), 2048);  // a bf16 tile, zero
        volatile tt_l1_ptr uint16_t* col16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wcol.get_write_ptr());
        for (uint32_t r = 0; r < 32 && slot.live; ++r) {
            const uint32_t row = (r / 16) * 2 * 256 + (r % 16) * 16;  // (r, 0)
            // the row's picks from column 0 on, then zero weights
            for (uint32_t c = 0; c < 16 && static_cast<int32_t>(routing[row + 256 + c]) > 0; ++c) {
                if (routing[row + c] == slot.expert) {
                    col16[row] = routing[row + 256 + c] >> 16;
                }
            }
        }
        wcol.push_back(1);
    };

    // the compute's h tile of item u into slot u mod 3 of the h buffer of the 4 down cores of this core's grid row (q =
    // 4 y + x), at tile p, and the slot's arrival count raised; a slot past its group's size sends zeros (its weights
    // were not read)
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_free));
    const uint32_t h_tile = get_local_cb_interface(cb_hout).fifo_page_size;
    // the down cores' h buffer: CB_H there starts where CB_SCR (on every core, at the same address) ends
    const uint32_t h_base = get_local_cb_interface(cb_scr).fifo_limit;
    auto send_h = [&](uint32_t u, bool live) {
        hout.wait_front(1);
        if (!live) {
            volatile tt_l1_ptr uint32_t* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hout.get_read_ptr());
            for (uint32_t e = 0; e < h_tile / 4; ++e) {
                h[e] = 0;
            }
        }
        const uint32_t dst = h_base + ((u % MOE_H_SLOTS) * h_tiles + p) * h_tile;
        constexpr uint32_t dn_row = num_dn / 8;
        {
            DeviceZoneScopedN("GC-FREE");  // (waiting for the down cores to let the slot go)
            noc_semaphore_wait_min(freed, dn_row * (u + 1));
        }
        const uint32_t q0 = (p / (num_gu / 8)) * dn_row;
        for (uint32_t q = q0; q < q0 + dn_row; ++q) {
            noc_async_write(hout.get_read_ptr(),
                            get_noc_addr(get_arg_val<uint32_t>(2 + 2 * q), get_arg_val<uint32_t>(3 + 2 * q), dst),
                            h_tile);
        }
        noc_async_write_barrier();
        const uint32_t arrive = get_semaphore(sem_arrive0 + u % MOE_H_SLOTS);
        for (uint32_t q = q0; q < q0 + dn_row; ++q) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(2 + 2 * q), get_arg_val<uint32_t>(3 + 2 * q), arrive), 1);
        }
        hout.pop_front(1);
    };
#ifdef MOE_ZERO_H_FIRST
    // (the program factory's GLM53_MOE_ZERO_H_FIRST) a dead slot of the last item (no unit:
    // its h is zeros) sends its zero h at the item's start, from the zeros region, in place of after its compute's
    // partials: the down cores then do not wait on this grid row (the upper rows run behind on the item before, and a
    // dead row's own sum of the item was the last h in).  Its compute's h is taken later and dropped.  The same
    // zeros in the same place, so the same values.
    auto send_zero_h = [&](uint32_t u) {
        const uint32_t dst = h_base + ((u % MOE_H_SLOTS) * h_tiles + p) * h_tile;
        constexpr uint32_t dn_row = num_dn / 8;
        {
            DeviceZoneScopedN("GC-FREE");  // (waiting for the down cores to let the slot go)
            noc_semaphore_wait_min(freed, dn_row * (u + 1));
        }
        const uint32_t q0 = (p / (num_gu / 8)) * dn_row;
        for (uint32_t q = q0; q < q0 + dn_row; ++q) {
            for (uint32_t o = 0; o < h_tile; o += MEM_ZEROS_SIZE) {
                noc_async_write(MEM_ZEROS_BASE,
                                get_noc_addr(get_arg_val<uint32_t>(2 + 2 * q), get_arg_val<uint32_t>(3 + 2 * q), dst + o),
                                MEM_ZEROS_SIZE);
            }
        }
        noc_async_write_barrier();
        const uint32_t arrive = get_semaphore(sem_arrive0 + u % MOE_H_SLOTS);
        for (uint32_t q = q0; q < q0 + dn_row; ++q) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(2 + 2 * q), get_arg_val<uint32_t>(3 + 2 * q), arrive), 1);
        }
    };
#endif

    if constexpr (units) {
        constexpr uint32_t ks = kd / T;  // the K slice
        const uint32_t s = p / T, j = p % T;
        CircularBuffer xall(cb_xall), pcb(cb_p), part(cb_part), pin(cb_pin);
        auto peer = [&](uint32_t c, uint32_t addr) {  // core (s, c) of the row (the runtime args after p)
            const uint32_t a = 3 + 2 * num_dn + 2 * c;
            return get_noc_addr(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1), addr);
        };
        const uint32_t xt = get_local_cb_interface(cb_xall).fifo_page_size;  // bf16 (no CB_X here)
        xall.reserve_back(rows * ks);  // tile (r, j ks + k) of x at r ks + k
        bool xsent = x_early && h1_t >= 32;  // (in-op gather from 32 tokens a row: each row's x passed down the column)
        if (h1_sem != 0 && s == 0) {  // the in-op gather: x landed here over the fabric
#ifndef MOE_H1_RELAY  // (MOE_H1_RELAY: passed down the column above, row by row)
            DeviceZoneScopedN("GC-H1");
            volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h1_sem);
            if constexpr (h1_t >= 32) {
                // each mesh row's increments in its own 4-bit field (h1_send.cpp, stream_active.cpp): its tile rows
                // go down the grid column (the other G - 1 cores, as the whole slice did) as soon as they are all
                // here, while the other rows still come; the column's valid flag once every row has gone (below)
                constexpr uint32_t per_row = h1_count / 8, tr = h1_t / 32;
                const uint32_t a = 3 + 2 * num_dn + 2 * T;  // the column's rectangle (the runtime args after the peers)
                const uint32_t rbytes = tr * ks * xt, base = xall.get_write_ptr();
                uint32_t done = 0;
                while (done != 0xffu) {
                    invalidate_l1_cache();
                    const uint32_t v = *landed;
                    for (uint32_t m = 0; m < 8; ++m) {
                        if (((done >> m) & 1u) == 0 && ((v >> (4 * m)) & 0xfu) >= per_row) {
                            const uint32_t src = base + m * rbytes;
                            noc_async_write_multicast(
                                src,
                                get_noc_multicast_addr(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1),
                                                       get_arg_val<uint32_t>(a + 2), get_arg_val<uint32_t>(a + 3), src),
                                rbytes,
                                G - 1);
                            done |= 1u << m;
                        }
                    }
                }
                xsent = true;
            } else {
                noc_semaphore_wait_min(landed, h1_count);
            }
            noc_semaphore_set(landed, 0);
            if constexpr (h1_t < 32) {  // the compact rows into the tiles: mesh row m's t rows of tile (m t / 32, i)
                constexpr uint32_t pb = 64 * h1_t;
                const uint32_t land = get_write_ptr(h1_land), base = xall.get_write_ptr();
                for (uint32_t m = 0; m < 8; ++m) {
                    for (uint32_t i = 0; i < ks; ++i) {
                        const uint32_t src = land + (m * ks + i) * pb;
                        const uint32_t tile = base + ((m * h1_t / 32) * ks + i) * xt;
                        if constexpr (h1_t == 16) {  // faces 0-1 or 2-3
                            noc_async_write(src, get_noc_addr(tile + 1024 * (m % 2)), 1024);
                        } else {  // rows 8 (m mod 4) .. of faces 0-1 or 2-3: two halves of 8 face rows
                            const uint32_t d = tile + 1024 * ((m % 4) / 2) + 256 * (m % 2);
                            noc_async_write(src, get_noc_addr(d), 256);
                            noc_async_write(src + 256, get_noc_addr(d + 512), 256);
                        }
                    }
                }
                noc_async_write_barrier();
            }
#endif
        } else if (s == 0) {  // read by the stream
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xread)), 1);
        }
        free_noc1();
        {  // slice j for the grid column's other G - 1 cores (a slot a grid row): from the top one
            DeviceZoneScopedN("GC-XCOL");
            const uint32_t a = 3 + 2 * num_dn + 2 * T;  // the column's rectangle (the runtime args after the peers)
            auto column = [&](uint32_t addr) {
                return get_noc_multicast_addr(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1),
                                              get_arg_val<uint32_t>(a + 2), get_arg_val<uint32_t>(a + 3), addr);
            };
            if (s == 0) {
                if (!xsent) {
                    noc_async_write_multicast(
                        xall.get_write_ptr(), column(xall.get_write_ptr()), rows * ks * xt, G - 1);
                }
                noc_async_write_barrier();
                *xvalid = 1;
                noc_semaphore_set_multicast(get_semaphore(sem_xvalid), column(get_semaphore(sem_xvalid)), G - 1);
                noc_async_write_barrier();  // this column's multicasts are done (builder 0 counts them)
                noc_semaphore_inc(get_noc_addr(b0_x, b0_y, get_semaphore(sem_xdone)), 1);
            } else {
                noc_semaphore_wait_min(xvalid, 1);
            }
        }
        xall.push_back(rows * ks);
        const uint32_t part_tile = get_local_cb_interface(cb_part).fifo_page_size;  // fp32
        const uint32_t pin_base = pin.get_write_ptr();  // the same address on every gate | up core
        // item i's unit for the compute: its 0/1 tiles (CB slot i mod 2, which item i - 2's unit left: its entries
        // cleared, zero before the first use) and its routing column (packed row a's weight in column 0, likewise)
#ifdef MOE_PROG
        // (the progressive form: one more tile, its word 1 whether the unit's expert is read with its correction)
        constexpr uint32_t p_tiles = rows + 1;
#else
        constexpr uint32_t p_tiles = rows + (gather_mask ? 1 : 0);
#endif
        auto feed_p = [&](uint32_t i) {
            pcb.reserve_back(p_tiles);
            if (i < 2) {
                zero_tiles<both>(pcb.get_write_ptr(), rows * 2048);
            } else {
                const PackedUnit old = packed_unit<G>(ul, chosen, i - 2, slot_of(i - 2));
                rec_onehot<local, rows>(pcb.get_write_ptr(), recs_l1, old, false, 0);
            }
            const PackedUnit u = packed_unit<G>(ul, chosen, i, slot_of(i));
            rec_onehot<local, rows>(pcb.get_write_ptr(), recs_l1, u, false, 0x3f80);
            if constexpr (gather_mask) {
                uint32_t hit = 0;
                rec_unit_picks<local, rows>(
                    recs_l1, u, [&](uint32_t, uint32_t pick) { hit |= 1u << ((pick & 0xffff) / 32); });
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pcb.get_write_ptr() + rows * 2048) = hit;
            }
#ifdef MOE_PROG
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pcb.get_write_ptr() + rows * 2048)[1] =
                u.live && prog_corrected<local>(recs_l1, u.expert) ? 1u : 0u;
#endif
            pcb.push_back(p_tiles);
        };
        auto feed_w = [&](uint32_t i) {
            wcol.reserve_back(1);
            volatile tt_l1_ptr uint16_t* col16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wcol.get_write_ptr());
            if (i < 2) {
                zero_tiles<both>(wcol.get_write_ptr(), 2048);
            } else {
                const PackedUnit old = packed_unit<G>(ul, chosen, i - 2, slot_of(i - 2));
                for (uint32_t a = 0; a < old.size; ++a) {
                    col16[(a / 16) * 2 * 256 + (a % 16) * 16] = 0;
                }
            }
            const PackedUnit u = packed_unit<G>(ul, chosen, i, slot_of(i));
            rec_unit_picks<local, rows>(recs_l1, u, [&](uint32_t a, uint32_t pick) {
                col16[(a / 16) * 2 * 256 + (a % 16) * 16] = pick >> 16;
            });
            wcol.push_back(1);
        };
        // Per item i: the compute's partials out (once every receiver has summed item i - 2: S_PFREE of the slot,
        // one per item of the slot a peer has summed) and the row's in (S_PARRIVE, per slot), item i + 1's unit for
        // the compute while it sums item i, then item i's h on.
        if (n > 0) {
            feed_p(0);
            feed_w(0);
        }
        for (uint32_t i = 0; i < n; ++i) {
#ifdef MOE_ZERO_H_FIRST
            const bool dead_last = i + 1 == n && !packed_unit<G>(ul, chosen, i, slot_of(i)).live;
            if (dead_last) {
                send_zero_h(i);
            }
#endif
            {
                DeviceZoneScopedN("GC-PART");  // (waiting for this core's compute)
                part.wait_front(2 * T);
            }
            if (i >= 1) {  // the compute summed item i - 1 before this item's gather: that slot is free
                for (uint32_t c = 0; c < T; ++c) {
                    if (c != j) {
                        noc_semaphore_inc(peer(c, get_semaphore(sem_pfree0 + (i - 1) % 2)), 1);
                    }
                }
            }
            if (i >= 2) {
                volatile tt_l1_ptr uint32_t* pfree =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_pfree0 + i % 2));
                noc_semaphore_wait_min(pfree, (T - 1) * (i / 2));
            }
            const uint32_t dst = pin_base + ((i % 2) * T + j) * 2 * part_tile;
            for (uint32_t c = 0; c < T; ++c) {
                noc_async_write(part.get_read_ptr() + 2 * c * part_tile, peer(c, dst), 2 * part_tile);
            }
            noc_async_write_barrier();
            for (uint32_t c = 0; c < T; ++c) {
                if (c != j) {
                    noc_semaphore_inc(peer(c, get_semaphore(sem_parrive0 + i % 2)), 1);
                }
            }
            part.pop_front(2 * T);
            pin.reserve_back(2 * T);
            {
                DeviceZoneScopedN("GC-PIN");  // (waiting for the row's other cores' partials)
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_parrive0 + i % 2)),
                    (T - 1) * (i / 2 + 1));
            }
            pin.push_back(2 * T);
            if (i + 1 < n) {  // while the compute sums this item's pairs
                feed_p(i + 1);
                feed_w(i + 1);
            }
#ifdef MOE_ZERO_H_FIRST
            if (dead_last) {  // (its zero h sent already)
                hout.wait_front(1);
                hout.pop_front(1);
                continue;
            }
#endif
            send_h(i, packed_unit<G>(ul, chosen, i, slot_of(i)).live);
        }
    } else {
        if (!sender) {  // the sender's own x is pushed already
            DeviceZoneScopedN("GC-XWAIT");
            x.reserve_back(x_tiles);
            noc_semaphore_wait_min(xvalid, 1);
            x.push_back(x_tiles);
        } else {  // x read once
            free_noc1();
        }
        DeviceZoneScopedN("GC-ITEMS");
        // per item: its routing column for the compute, then this core's h tile to the down cores
        for (uint32_t i = 0; i < n; ++i) {
            push_wcol(i);
            send_h(i, sliced_slot<G>(active, chosen, i, p / T).live);
        }
    }
    noc_async_atomic_barrier();
}
