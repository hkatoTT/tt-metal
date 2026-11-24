// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The L1 landing of the next MoE layer's chosen experts' weights (GLM53_W1, ttnn/w1_landing.py): a bank's DRISC
// (role 0) fills the experts op's landing, in each MoE layer's end window, with the next MoE layer's chosen experts'
// weight blocks from its own bank.
//
// Window w (the w-th experts op since the start; the layers in order, `layers` of them, cyclic): its blocks are those
// of layer L' = (w + 1) mod layers.
//   1. Once the 8 gate | up weight streams of this bank's K slice (grid column 4 + b) have raised mailbox word 2 (8 a
//      window: the op's end-window signal): for each slot s, expert e = the layer's s-th chosen, its 8 W1 blocks
//      (columns c = 0 .. 7, contiguous in this bank: shards 8 e + c of the layer's w1) to gate | up core (4 + b, c),
//      slot s.
//   2. Once the 4 down streams of this bank (cores b, b + 8, b + 16, b + 24, their W2 slices in this bank) have raised
//      word 3: for each slot s, each of those cores' block of e (shard q / 8 of the layer's w2, block e) to that core,
//      slot s.
//   Each block: GDDR -> this DRISC's L1 (DMA, waited for in full: a pattern seen to keep the data whole) -> NoC 0
//   posted writes: a zero seal over the slot's seal, the block, then the slot's 64-byte seal (word 0: L''s tag |
//   the slot's expert << 16), from the same place on the same NoC (writes from one sender to one place arrive in
//   order).  The reader reads seal, block, seal (stream_active.cpp MOE_W1_RECV): its (tag, expert) both times means
//   the block was whole, not rewritten while read, and the expert it asked for (this chip's table: its mesh column's
//   experts).
//   A slot is written only after the op that read it has said so (the words above); a window whose next window's
//   signals are already in (this DRISC behind: the layer it fills has been read) is skipped, counted in word 8.
//
// Mailbox (L1, `mailbox`): words 0 stop (host), 1 windows done, 2 gate | up signals, 3 down signals, 4 / 5 the first
// timestamp, 6 / 7 the last, 8 windows skipped (behind: the next window's signals already in), 9 the launch's
// generation (the host's, written before the launch: the kernel returns once it changes, as on a stop: a launch
// left behind never outlives the next), 12 the gate | up streams' starts (GLM53_W1_CUT: 8 an op), 13 windows cut
// short, 14 the cut switch (the host's: 1, a window stops at the next op's start), 11 the late slots (the host's,
// GLM53_W1_LATE: 0); from byte 64 a ring of 128 windows: start, W1 part's cycles, W2 part's cycles, the window's
// layer.  Byte 2176: the zero seal (64 B), 2240 + 64 s: slot s's seal (MAX_C slots).  (A process killed with the
// kernel running leaves it polling these words, harmless: only the experts op with GLM53_W1 raises them, and tt-metal
// puts the DRAM cores' RISCs in reset at the next device open, RiscFirmwareInitializer::reset_cores.)  The table (the
// host's, `table`): layers, C, the slot's bytes, a block's bytes, a W2 shard's bytes, then per layer: tag, w1
// address, w2 address (bank-local), max(8, C) experts.
//
// C above 8 (up to MAX_C, the landing as large as the CBs allow): the copy runs on past the end window, beside the
// next layer's chain (the DRAM-reading ops there slow down a little).  With the cut switch (word 14), it stops at the
// next experts op's start (word 12 at 8 (w + 2): that op's gate | up streams have started; the slots not written
// keep their old seals, so the op reads them from DRAM), the blocks in flight finished: the carrier never reads
// beside the op that would read the same bytes.
//
// Compile-time args: stage (L1: two blocks), mailbox, table.
// Runtime args: the landing's L1 address, then the gate | up cores of this bank's column (rows 0 .. 7: NoC x, y),
// then the 4 down cores (NoC x, y).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/gddr_dma.h"

void kernel_main() {
    constexpr uint32_t stage = get_compile_time_arg_val(0);
    constexpr uint32_t mailbox = get_compile_time_arg_val(1);
    constexpr uint32_t TABLE = get_compile_time_arg_val(2);
    constexpr uint32_t ZERO = mailbox + 2176;
    constexpr uint32_t SEALS = mailbox + 2240;  // slot s's at SEALS + 64 s
    constexpr uint32_t RING = 128;
    constexpr uint32_t MAX_C = 24;
    static_assert(2240 + 64 * MAX_C <= 4096, "the seals within the mailbox's 4 KB (the stage after it)");
    const uint32_t landing = get_arg_val<uint32_t>(0);

    volatile tt_l1_ptr uint32_t* box = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox);
    volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox + 64);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TABLE);
    volatile tt_l1_ptr uint32_t* seals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SEALS);
    volatile tt_l1_ptr uint32_t* zero = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ZERO);
    invalidate_l1_cache();
    const uint32_t layers = tab[0], C = tab[1], slot = tab[2], blk = tab[3], w2_shard = tab[4];
    const uint32_t gen = box[9];
    const uint32_t stride = 3 + (C > 8 ? C : 8);  // a layer's words: 0 tag, 1 w1, 2 w2, 3.. experts
    auto entry = [&](uint32_t layer, uint32_t k) { return tab[5 + layer * stride + k]; };
    const bool cut_on = box[14] != 0;
    // word 11 (GLM53_W1_LATE, the host's; 0: none): slots k .. C - 1 (the layer's experts of its last items:
    // w1_landing.py's chosen_columns) copied whole, W1 and W2 parts, not in this window but once the next experts op
    // (their reader) has started (its word 12: with the cut switch only), in expert order (the op's item order), never
    // cut: a block the op reaches first it reads from DRAM (its slot's old seal), so the values are the same either way
    // (the copy then sits in that op's start, x not yet in, where the DRAM is idle, in place of this window's end
    // beside the next layer's chain and the drafter)
    const uint32_t late = cut_on ? box[11] : 0;
    const uint32_t now_c = late != 0 && late < C ? late : C;  // slots copied in this window

    const uint64_t t_begin = get_timestamp();
    box[4] = static_cast<uint32_t>(t_begin);
    box[5] = static_cast<uint32_t>(t_begin >> 32);
    for (uint32_t i = 0; i < 16; ++i) {
        zero[i] = 0;
    }
    for (uint32_t i = 0; i < 16 * MAX_C; ++i) {
        seals[i] = 0;
    }
    uint32_t cuts = 0;
    uint32_t cut_at = 0xFFFFFFFFu;  // word 12's count at which the window in hand stops (the next op's start)
    auto cut = [&]() {
        if (!cut_on) {
            return false;
        }
        invalidate_l1_cache();
        return box[12] >= cut_at;
    };

    // blocks i = 0 .. n - 1 of a part: src(i) -> dst(i), sealed with slot seal_of(i)'s seal; the next block's DMA while
    // this one's write goes out
    // (cut: no block after the next op's start; returns false then)
    auto copy = [&](uint32_t n, auto src, auto dst, auto seal_of) {
        if (n == 0) {
            return true;
        }
        if (cut()) {
            return false;
        }
        experimental::dma_async_read(0, src(0), stage, blk);
        for (uint32_t i = 0; i < n; ++i) {
            experimental::dma_async_read_barrier(0);  // block i is in slot i & 1
            const bool more = i + 1 < n && !cut();
            if (more) {
                noc_async_posted_writes_flushed();  // block i - 1's write has left slot (i + 1) & 1
                experimental::dma_async_read(0, src(i + 1), stage + ((i + 1) & 1) * blk, blk);
            }
            const uint64_t d = dst(i);
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, /*posted=*/true>(ZERO, d + blk, 64);
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, /*posted=*/true>(stage + (i & 1) * blk, d, blk);
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, /*posted=*/true>(SEALS + 64 * seal_of(i), d + blk, 64);
            if (!more) {
                noc_async_posted_writes_flushed();
                return i + 1 == n;
            }
        }
        noc_async_posted_writes_flushed();
        return true;
    };
    // wait until word `k` has reached `need`; false: stopped
    auto wait_for = [&](uint32_t k, uint32_t need) {
        for (;;) {
            invalidate_l1_cache();
            if (box[k] >= need) {
                return true;
            }
            if (box[0] != 0 || box[9] != gen) {
                return false;
            }
        }
    };

    uint32_t behind = 0;
    for (uint32_t w = 0;; ++w) {
        if (!wait_for(2, 8 * (w + 1))) {
            break;
        }
        if (box[2] >= 8 * (w + 2)) {  // the layer this window fills has been read: skipped
            ++behind;
            box[8] = behind;
            box[1] = w + 1;
            continue;
        }
        const uint32_t L = (w + 1) % layers;
        cut_at = 8 * (w + 2);  // op w + 1 (the reader of layer L) started: its 8 gate | up streams of this bank
        for (uint32_t s = 0; s < C; ++s) {  // (the window before's seal writes have left: flushed at its end)
            seals[16 * s] = entry(L, 0) | (entry(L, 3 + s) << 16);
        }
        const uint32_t w1 = entry(L, 1), w2 = entry(L, 2);
        const uint64_t t0 = get_timestamp();
        // 1. W1: slot s, column c: block (s, c) = i (8 a slot)
        bool whole = copy(
            now_c * 8,
            [&](uint32_t i) {
                return static_cast<uint64_t>(w1) + static_cast<uint64_t>(8 * entry(L, 3 + i / 8) + i % 8) * blk;
            },
            [&](uint32_t i) {
                return get_noc_addr(get_arg_val<uint32_t>(1 + 2 * (i % 8)), get_arg_val<uint32_t>(2 + 2 * (i % 8)),
                                    landing + (i / 8) * slot);
            },
            [](uint32_t i) { return i / 8; });
        const uint64_t t1 = get_timestamp();
        if (!wait_for(3, 4 * (w + 1))) {
            break;
        }
        // 2. W2: slot s, down core k (q = b + 8 k): block (s, k) = i (4 a slot)
        whole = copy(
            now_c * 4,
            [&](uint32_t i) {
                return static_cast<uint64_t>(w2) + static_cast<uint64_t>(i % 4) * w2_shard +
                       static_cast<uint64_t>(entry(L, 3 + i / 4)) * blk;
            },
            [&](uint32_t i) {
                return get_noc_addr(get_arg_val<uint32_t>(17 + 2 * (i % 4)), get_arg_val<uint32_t>(18 + 2 * (i % 4)),
                                    landing + (i / 4) * slot);
            },
            [](uint32_t i) { return i / 4; }) && whole;
        if (!whole) {
            box[13] = ++cuts;
        }
        const uint64_t t2 = get_timestamp();
        uint32_t late_cycles = 0;
        if (now_c < C) {  // 3. (GLM53_W1_LATE) slots k .. C - 1, once op w + 1 (their reader) has started
            if (!wait_for(12, 8 * (w + 2))) {
                break;
            }
            const uint64_t t3 = get_timestamp();
            uint32_t ls[MAX_C];
            const uint32_t n = C - now_c;
            for (uint32_t j = 0; j < n; ++j) {  // in expert order (insertion: C <= 24)
                uint32_t v = now_c + j, x = j;
                while (x > 0 && entry(L, 3 + ls[x - 1]) > entry(L, 3 + v)) {
                    ls[x] = ls[x - 1];
                    --x;
                }
                ls[x] = v;
            }
            cut_at = 0xFFFFFFFFu;  // (never cut: the op reads a block not yet here from DRAM)
            copy(
                n * 8,
                [&](uint32_t i) {
                    return static_cast<uint64_t>(w1) + static_cast<uint64_t>(8 * entry(L, 3 + ls[i / 8]) + i % 8) * blk;
                },
                [&](uint32_t i) {
                    return get_noc_addr(get_arg_val<uint32_t>(1 + 2 * (i % 8)), get_arg_val<uint32_t>(2 + 2 * (i % 8)),
                                        landing + ls[i / 8] * slot);
                },
                [&](uint32_t i) { return ls[i / 8]; });
            copy(
                n * 4,
                [&](uint32_t i) {
                    return static_cast<uint64_t>(w2) + static_cast<uint64_t>(i % 4) * w2_shard +
                           static_cast<uint64_t>(entry(L, 3 + ls[i / 4])) * blk;
                },
                [&](uint32_t i) {
                    return get_noc_addr(get_arg_val<uint32_t>(17 + 2 * (i % 4)),
                                        get_arg_val<uint32_t>(18 + 2 * (i % 4)), landing + ls[i / 4] * slot);
                },
                [&](uint32_t i) { return ls[i / 4]; });
            const uint64_t d = (get_timestamp() - t3) >> 4;
            late_cycles = d > 0xFFFFu ? 0xFFFFu : static_cast<uint32_t>(d);
        }
        const uint32_t r = 4 * (w % RING);
        rec[r] = static_cast<uint32_t>(t0);
        rec[r + 1] = static_cast<uint32_t>(t1 - t0);
        rec[r + 2] = static_cast<uint32_t>(t2 - t1);
        rec[r + 3] = L | (late_cycles << 16);  // (GLM53_W1_LATE: its part's cycles / 16 in the high half)
        box[8] = behind;
        box[1] = w + 1;
    }
    const uint64_t t_end = get_timestamp();
    box[6] = static_cast<uint32_t>(t_end);
    box[7] = static_cast<uint32_t>(t_end >> 32);
}
