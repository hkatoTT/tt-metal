// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The prefill MoE that reads each expert's weights once (moe_prefill.py).
//   plan: from the route op's rows ([N, 32] fp32 tiled: columns 0-7 the chosen local experts' ids, 16-23 their
//         weights, weight 0 for an unused column), the pair table: expert e's pairs at rows off[e] .. off[e] + n[e]
//         (off a multiple of 32, the rest of the tile rows padding), per row its token (SENT for padding) and weight,
//         and per token its pair rows (SENT after them).
//   seg page (uint32): words 0 .. 72 off (off[72]: the table's rows), words 128 .. 199 n.
namespace moe_prefill {
constexpr uint32_t SENT = 0xFFFFFFFFu;
constexpr uint32_t SEG_WORDS = 256;  // the seg page: 1 KB
constexpr uint32_t SEG_COUNT = 128;  // n[e] at SEG_COUNT + e
constexpr uint32_t ROW_WORDS = 16;   // a page of the pair table's arrays and of the token rows: 16 words, 64 bytes
constexpr uint32_t K = 8;            // chosen experts a token (the route op's columns)
constexpr uint32_t F = 1024, R = 64;  // fp32 face bytes, face row bytes
}  // namespace moe_prefill

// BALANCE (moe_prefill.py): the column's tile rows split evenly over a stage's `units` (mp_gu5's groups, down's cores)
// in place of whole items round robin (unit u: items u, u + units, ..), when the round robin's busiest unit would
// have more than `pct` % of the even split's share (real routing gives one expert up to half of the column's pairs,
// and its unit all of them; uniform routing stays round robin, which reads each expert's weights once).  The items'
// tile rows in item order make a line (`per` items an expert, item e per + q: down's (e, qq)), on which each item with
// rows also weighs `a` rows before them (its weights' load); unit u takes the rows whose place falls in
// [W u / units, W (u + 1) / units).  The units' shares meet at the same places, so every row is one unit's.  A unit's
// reader, compute (through its writer) and writer walk the same segments (item, first row, rows) in order, skipping
// items without rows; each row's computation is the same either way (the values bit for bit).
#if !defined(COMPILE_FOR_TRISC)
namespace moe_prefill {
struct BalancedWalk {
    bool even;
    uint32_t lo, hi, p, item, items, step;
};
inline uint32_t item_rows(volatile tt_l1_ptr uint32_t* seg, uint32_t per, uint32_t i) {
    return (seg[SEG_COUNT + i / per] + 31) / 32;
}
inline BalancedWalk balanced_walk(
    volatile tt_l1_ptr uint32_t* seg, uint32_t local, uint32_t per, uint32_t a, uint32_t u, uint32_t units,
    uint32_t pct) {
    const uint32_t items = local * per;
    uint32_t w = 0, worst = 0;
    for (uint32_t v = 0; v < units; ++v) {  // the round robin's units
        uint32_t c = 0;
        for (uint32_t i = v; i < items; i += units) {
            const uint32_t rows = item_rows(seg, per, i);
            c += rows ? a + rows : 0;
        }
        w += c;
        worst = c > worst ? c : worst;
    }
    if (worst * 100 <= pct * (w / units + a)) {
        return BalancedWalk{false, 0, 0, 0, u, items, units};
    }
    return BalancedWalk{true, w * u / units, w * (u + 1) / units, 0, 0, items, 1};
}
// the unit's next segment -> true (item, its first row r0, nr rows), or false at the unit's end
inline bool next_segment(
    volatile tt_l1_ptr uint32_t* seg, uint32_t per, uint32_t a, BalancedWalk& s, uint32_t& item, uint32_t& r0,
    uint32_t& nr) {
    if (!s.even) {
        while (s.item < s.items) {
            const uint32_t i = s.item;
            s.item += s.step;
            const uint32_t rows = item_rows(seg, per, i);
            if (rows) {
                item = i;
                r0 = 0;
                nr = rows;
                return true;
            }
        }
        return false;
    }
    while (s.item < s.items && s.p < s.hi) {
        const uint32_t i = s.item++;
        const uint32_t rows = item_rows(seg, per, i);
        if (rows == 0) {
            continue;
        }
        const uint32_t first = s.p + a;  // the item's first row's place
        s.p += a + rows;
        const uint32_t b0 = s.lo > first ? s.lo - first : 0;
        const uint32_t b1 = s.hi > first ? (s.hi - first < rows ? s.hi - first : rows) : 0;
        if (b1 > b0) {
            item = i;
            r0 = b0;
            nr = b1 - b0;
            return true;
        }
    }
    return false;
}
}  // namespace moe_prefill
#endif

// gate | up (mp_gu5_*.cpp, below): x's and W1's shapes
namespace moe_prefill {
constexpr uint32_t KD = 128;   // K tiles (hidden 4096)
constexpr uint32_t KBLK = 16;  // K tiles a member's block (512 columns)
constexpr uint32_t COLS = 8;   // an expert's column tiles on a chip (intermediate 256)
}  // namespace moe_prefill

// down (mp_dn_*.cpp): a core's items (e, qq): W2(e, q) [256, 128] (e_w2's tile rows (q slots + e) 8 + k, 4 columns)
// for the 4 column blocks q = 4 qq .. 4 qq + 3, read once; per tile row of e's pairs, H's 8 tiles of the row (read
// once) times each (fp32), untilized with the 4 blocks side by side, each row (512 values) to its pair's row of Y
// [M, 4096] (row-major, fp32 or bf16), columns 512 qq ...
// combine (mp_cb_*.cpp): a core's items (token tile row, block of 512 columns): per slot j (a token's j-th pair) the
// 32 tokens' Y rows of the block side by side, row-major (zeros for a token with fewer pairs), summed over the slots
// in fp32 as flat 1,024-value pieces (an element-wise sum keeps the row-major order), bf16, tilized, to out [N, 4096]
// (tiled: the block's 16 tiles of the token tile row).
namespace moe_prefill {
constexpr uint32_t DN_W2 = 0;    // W2(e, q): 8 K tiles x 4 columns
constexpr uint32_t DN_HT = 1;    // a tile row's 8 h tiles
constexpr uint32_t DN_Y = 2;     // a tile row's y, untilized (32 rows x the 4 blocks' 512 values)
constexpr uint32_t DN_SCR = 3;   // the reader's scratch: the seg page
constexpr uint32_t DN_WSCR = 4;  // the writer's scratch: the seg page
constexpr uint32_t DN_K = 8;     // h's K tiles (256)
constexpr uint32_t DN_N = 4;     // an output column block's tiles (128 columns: e_w2's shard width)
constexpr uint32_t DN_Q = 4;     // column blocks an item (H read once for them)
constexpr uint32_t DN_Y8 = 5;    // Y_BFP8: a tile row's y as 16 flat bfp8 pieces (two rows of 512 values each)
constexpr uint32_t DN_E8 = 6;    // Y_BFP8: the writer's exponents twice, 64 bytes a row of the tile row
constexpr uint32_t Y8_PIECE = 1088;  // a flat bfp8 piece: 64 exponents, 1,024 mantissas
constexpr uint32_t Y8_REC = 1088;    // a Y8 record: a pair row's 1,024 columns as one flat bfp8 piece
constexpr uint32_t CB_COLS8 = 1024;  // Y_BFP8: combine's columns a block (one token row a piece)
constexpr uint32_t CB_S = 0;     // a slot's 32 rows of the block (16 flat fp32 pieces)
constexpr uint32_t CB_O = 1;     // the sums (16 flat bf16 pieces)
constexpr uint32_t CB_SCR = 2;   // the reader's scratch: the tokens' pair rows
constexpr uint32_t CB_WSCR = 3;  // the writer's scratch: the same
constexpr uint32_t CB_T = 4;     // the sums tilized (16 tiles: the block's 32 rows x 512 columns)
constexpr uint32_t CB_SL = 5;    // an item's slots, the reader to the writer
constexpr uint32_t CB_COLS = 512;  // columns a block
constexpr uint32_t CB_PIECES = 16;  // flat 1,024-value pieces a block (32 rows x 512)
}  // namespace moe_prefill

// semaphores (ids)
namespace moe_prefill {
constexpr uint32_t S_READY = 0;  // mp_gu5: the members ready for a row's partials
constexpr uint32_t S_XVALID = 1;
constexpr uint32_t S_NOC1 = 2;  // mp_gu5, down, combine: NoC 1 handed over by the writer (mp_noc1.hpp)
}  // namespace moe_prefill

// gate | up by groups split along K (mp_gu5_*.cpp): member c owns K block c (columns 512 c .. of x) and, for the
// output, column tile c (gate c, up c).  Per tile row it gathers and tilizes only its x block, multiplies it by W1's
// K block c of all 8 column tiles (16 K tiles x 16 output tiles: the partial sums, fp32), and sends each column
// tile's two partials to that tile's owner (7 x 8 KB out, 7 x 8 KB in, in place of gu4's 256 KB multicast of the x
// row to every member); the owner sums its 8 partials in fp32 (the order: member 0 .. 7) and finishes h.
namespace moe_prefill {
constexpr uint32_t G5_W1 = 0;    // W1's K block c, all 8 column tiles: [16 K][8 c' x (gate, up)], two experts
constexpr uint32_t G5_XRM = 1;   // this member's K block of the 32 x rows, row-major (16 tile pages), two rows
constexpr uint32_t G5_XT = 2;    // the same tilized (16 tiles), two rows
constexpr uint32_t G5_WCOL = 3;  // the pairs' weights in column 0 (bf16)
constexpr uint32_t G5_H = 4;     // h
constexpr uint32_t G5_SCR = 5;   // the reader's scratch
constexpr uint32_t G5_WSCR = 6;  // the writer's scratch
constexpr uint32_t G5_PO = 7;    // this member's partial sums of a row: 16 tiles fp32 (c' x 2 + g), two rows
constexpr uint32_t G5_RED = 8;   // the 8 members' partials of this member's column tile: 16 tiles fp32 (s x 2 + g), two rows
constexpr uint32_t G5_N = 2 * COLS;  // output tiles of a row's partials (16)
constexpr uint32_t Q1_X = 17, Q1_Y = 12;  // the reader's NoC 1 shares by physical x, y (2 bits each, 13 words)
constexpr uint32_t G5_CHUNK = 4;     // output tiles a Dest pass (fp32 Dest, half sync)
}  // namespace moe_prefill
