// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "mp_common.hpp"
#include "mp_noc1.hpp"

// The prefill MoE's down, the writer (mp_common.hpp): per item, its tile rows' count to the compute (mailbox), then
// each untilized tile row's 32 rows of its 4 column blocks side by side (512 values a row) to rows off[e] + 32 r + y
// of Y, columns 512 qq ..; only the rows of the expert's pairs (the last tile row's padding rows are left unwritten:
// nothing reads them).  Y_BFP8 (y8): from DN_Y8's flat bfp8 pieces (piece y / 2, half y mod 2: its 32 exponents at
// 32 (y mod 2), its 512 mantissas at 64 + 512 (y mod 2)), row y's half h = qq mod 2 of its record (off[e] + 32 r + y,
// columns 1,024 (qq / 2) ..: a flat bfp8 piece of Y8_REC bytes): the mantissas at 64 + 512 h, the exponents at 32 h
// (copied to DN_E8 at 32 h first: the write's L1 and DRAM offsets equal mod 64).
void kernel_main() {
    using namespace moe_prefill;
    uint32_t a = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t seg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_items = get_arg_val<uint32_t>(a++);

    constexpr uint32_t y_row = get_compile_time_arg_val(0);  // bytes of a Y row (4096 fp32, or bf16: Y_BF16)
    constexpr auto y_args = TensorAccessorArgs<1>();
    constexpr auto s_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t nocs = get_compile_time_arg_val(s_args.next_compile_time_args_offset());  // DNCB_NOCS
    // BALANCE: as the reader (the core count and index in the runtime args); the compute hears the count first
    constexpr bool balance = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 1) != 0;
    constexpr uint32_t bal_a = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t bal_pct = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 3);
    constexpr bool y8 = get_compile_time_arg_val(s_args.next_compile_time_args_offset() + 4) != 0;  // Y_BFP8
    constexpr uint32_t PIECE = y_row / 32;  // a row's 128 values
    constexpr uint32_t CB_IN = y8 ? DN_Y8 : DN_Y;
    const uint32_t e8 = y8 ? get_write_ptr(DN_E8) : 0;  // (64-byte aligned: a CB's address)

    const auto s_y = TensorAccessor(y_args, y_addr, y_row);
    const auto s_seg = TensorAccessor(s_args, seg_addr, SEG_WORDS * 4);
    const uint32_t seg_l1 = get_write_ptr(DN_WSCR);
    noc_async_read(s_seg.get_noc_addr(0), seg_l1, SEG_WORDS * 4);
    noc_async_read_barrier();
    if constexpr (nocs == 2) {
        hand_over_noc1();
    }
    volatile tt_l1_ptr uint32_t* seg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seg_l1);

    auto mail = [](uint32_t v) {
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, v);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, v);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, v);
    };
    constexpr uint32_t PER = 32 / DN_Q;  // items an expert
    const uint32_t core = balance ? get_arg_val<uint32_t>(a) : 0;
    if constexpr (balance) {
        BalancedWalk count = balanced_walk(seg, 72, PER, bal_a, core, n_items, bal_pct);
        uint32_t item, r0, nr, n = 0;
        while (next_segment(seg, PER, bal_a, count, item, r0, nr)) {
            ++n;
        }
        mail(n);
    }
    BalancedWalk walk{};
    if constexpr (balance) {
        walk = balanced_walk(seg, 72, PER, bal_a, core, n_items, bal_pct);
    }
    for (uint32_t i = 0; balance || i < n_items; ++i) {
        uint32_t item = 0, r0 = 0, rows = 0;
        if constexpr (balance) {
            if (!next_segment(seg, PER, bal_a, walk, item, r0, rows)) {
                break;
            }
        } else {
            item = get_arg_val<uint32_t>(a + i);
            rows = (seg[SEG_COUNT + item / PER] + 31) / 32;
        }
        const uint32_t e = item / PER, qq = item % PER;
        mail(rows);
        const uint32_t off = seg[e] + 32 * r0, pairs = seg[SEG_COUNT + e] - 32 * r0;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t used = pairs - 32 * r < 32 ? pairs - 32 * r : 32;
            cb_wait_front(CB_IN, DN_Q * DN_N);
            const uint32_t src = get_read_ptr(CB_IN);
            if constexpr (y8) {
                const uint32_t h = qq & 1;
                for (uint32_t y = 0; y < used; ++y) {
                    const uint32_t piece = src + (y >> 1) * Y8_PIECE, half = y & 1;
                    const uint64_t dst = s_y.get_noc_addr(off + 32 * r + y, (qq >> 1) * Y8_REC);
                    noc_async_write(piece + 64 + half * 512, dst + 64 + 512 * h, 512);
                    volatile tt_l1_ptr uint32_t* ex = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(piece + 32 * half);
                    volatile tt_l1_ptr uint32_t* st =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(e8 + 64 * y + 32 * h);
                    for (uint32_t w = 0; w < 8; ++w) {
                        st[w] = ex[w];
                    }
                    noc_async_write(e8 + 64 * y + 32 * h, dst + 32 * h, 32);
                }
            } else {
                for (uint32_t y = 0; y < used; ++y) {
                    noc_async_write(
                        src + y * DN_Q * PIECE, s_y.get_noc_addr(off + 32 * r + y, qq * DN_Q * PIECE), DN_Q * PIECE);
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(CB_IN, DN_Q * DN_N);
        }
    }
    noc_async_write_barrier();
}
