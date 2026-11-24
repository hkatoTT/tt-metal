// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "mp_common.hpp"
#include "mp_noc1.hpp"

// The prefill MoE's gate | up by groups split along K (mp_common.hpp, mp_gu5_*.cpp), the writer, NoC 1: the group's
// rows in all to the compute (mailbox), then per expert of the group its tile rows' count; per tile row k: G5_RED
// reserved (two rows: the compute has finished row k - 2) and S_READY counted in on the 7 others; once the compute
// has this member's partials of row k and the 7 others have reserved theirs, each member m's two partials (gate m,
// up m: 8 KB fp32) written into slot c of member m's G5_RED (this member's own locally) and S_XVALID counted in on the
// others; G5_RED pushed once the 7 others' partials are in; then row k - 1's h to H.
void kernel_main() {
    using namespace moe_prefill;
    uint32_t a = 0;
    const uint32_t h_addr = get_arg_val<uint32_t>(a++);
    const uint32_t seg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t group = get_arg_val<uint32_t>(a++);
    const uint32_t groups = get_arg_val<uint32_t>(a++);
    const uint32_t c = get_arg_val<uint32_t>(a++);
    const uint32_t xy0 = a;  // then the 8 members' NoC x | y << 16, member 0 .. 7 (this one too)

    constexpr uint32_t h_tile = get_compile_time_arg_val(0);
    constexpr uint32_t local = get_compile_time_arg_val(1);
    constexpr uint32_t members = get_compile_time_arg_val(2);  // 8
    constexpr auto h_args = TensorAccessorArgs<3>();
    constexpr auto s_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    // timing probe (wrong results; tests only): 32 no exchange (the partials dropped, G5_RED pushed unwritten)
    constexpr uint32_t probe = get_compile_time_arg_val(s_args.next_compile_time_args_offset());
    constexpr uint32_t nocs = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 1);  // GU5_NOCS
    // BALANCE (mp_common.hpp): the group's segments, as the reader walks them; the compute hears their count first
    constexpr bool balance = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 2) != 0;
    constexpr uint32_t bal_a = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t bal_pct = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t P_TILE = 4096;          // a partial tile (fp32)
    constexpr uint32_t PAIR = 2 * P_TILE;      // a column tile's two partials (gate, up)

    const auto s_h = TensorAccessor(h_args, h_addr, h_tile);
    const auto s_seg = TensorAccessor(s_args, seg_addr, SEG_WORDS * 4);
    const uint32_t seg_l1 = get_write_ptr(G5_WSCR);
    noc_async_read(s_seg.get_noc_addr(0), seg_l1, SEG_WORDS * 4);
    noc_async_read_barrier();
    if constexpr (nocs == 2) {  // NoC 1 handed over to the reader's DRAM reads: no more reads here (writes only)
        hand_over_noc1();
    }
    volatile tt_l1_ptr uint32_t* seg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seg_l1);
    const uint32_t ready_addr = get_semaphore(S_READY);
    const uint32_t xvalid_addr = get_semaphore(S_XVALID);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* xvalid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xvalid_addr);
    auto member = [&](uint32_t m, uint32_t addr) {
        const uint32_t xy = get_arg_val<uint32_t>(xy0 + m);
        return get_noc_addr(xy & 0xFFFF, xy >> 16, addr);
    };
    auto count_in = [&](uint32_t sem_addr) {
        for (uint32_t m = 0; m < members; ++m) {
            if (m != c) {
                noc_semaphore_inc(member(m, sem_addr), 1);
            }
        }
    };
    auto write_h = [&](uint32_t page) {
        DeviceZoneScopedN("G5_WAIT_H");
        cb_wait_front(G5_H, 1);
        noc_async_write(get_read_ptr(G5_H), s_h.get_noc_addr(page), h_tile);
        noc_async_writes_flushed();
        cb_pop_front(G5_H, 1);
    };

    auto mail = [](uint32_t v) {
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, v);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, v);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, v);
    };
    uint32_t total = 0, n_segs = 0;
    if constexpr (balance) {
        BalancedWalk walk = balanced_walk(seg, local, 1, bal_a, group, groups, bal_pct);
        uint32_t item, r0, nr;
        while (next_segment(seg, 1, bal_a, walk, item, r0, nr)) {
            total += nr;
            ++n_segs;
        }
    } else {
        for (uint32_t e = group; e < local; e += groups) {
            total += (seg[SEG_COUNT + e] + 31) / 32;
        }
    }
    mail(total);
    if constexpr (balance) {
        mail(n_segs);
    }

    uint32_t k = 0;
    uint32_t pending = SENT;  // the h page of the row before
    BalancedWalk walk{};
    if constexpr (balance) {
        walk = balanced_walk(seg, local, 1, bal_a, group, groups, bal_pct);
    }
    for (uint32_t e0 = group; balance || e0 < local; e0 += groups) {
        uint32_t e = e0, r0 = 0, rows = 0;
        if constexpr (balance) {
            if (!next_segment(seg, 1, bal_a, walk, e, r0, rows)) {
                break;
            }
        } else {
            rows = (seg[SEG_COUNT + e] + 31) / 32;
        }
        mail(rows);
        const uint32_t row0 = seg[e] / 32 + r0;
        for (uint32_t r = 0; r < rows; ++r, ++k) {
            {
                DeviceZoneScopedN("G5_X");
                cb_reserve_back(G5_RED, G5_N);  // the compute has finished row k - 2
                if constexpr (probe & 32) {
                    cb_wait_front(G5_PO, G5_N);
                    cb_pop_front(G5_PO, G5_N);
                    cb_push_back(G5_RED, G5_N);
                    if (pending != SENT) {
                        write_h(pending);
                    }
                    pending = (row0 + r) * COLS + c;
                    continue;
                }
                const uint32_t slot = get_write_ptr(G5_RED) + c * PAIR;  // this member's slot, the same everywhere
                count_in(ready_addr);
                cb_wait_front(G5_PO, G5_N);
                noc_semaphore_wait_min(ready, (members - 1) * (k + 1));
                const uint32_t po = get_read_ptr(G5_PO);
                for (uint32_t m = 0; m < members; ++m) {
                    noc_async_write(po + m * PAIR, member(m, slot), PAIR);
                }
                noc_async_write_barrier();  // the partials are in before the others hear of it
                cb_pop_front(G5_PO, G5_N);
                count_in(xvalid_addr);
                noc_semaphore_wait_min(xvalid, (members - 1) * (k + 1));
                cb_push_back(G5_RED, G5_N);
            }
            if (pending != SENT) {
                write_h(pending);
            }
            pending = (row0 + r) * COLS + c;
        }
    }
    if (pending != SENT) {
        write_h(pending);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
