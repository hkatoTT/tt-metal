// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Heads-first Q and output (SdpaDecodeParams::heads_first, SDPA_HEADS_FIRST): Q is [1, NH, B, D] and the output
// [1, NH, B, vD], bf16 tiles, interleaved (Q in L1). Token b is row b % 32 of tile row b / 32 of each head's matrix.
// The op computes on a token's tile with its heads as rows (row h = head h, the layout of a [1, B, NH, D] Q), so the
// reader puts that tile together from 32-byte face rows (NH heads x the head's tiles x 2 faces, into the rows the
// legacy layout has them in) and the writer sends the output tile's rows back the same way. Only bytes move: the
// compute takes the same Q tile and gives the same output rows as with the legacy layouts.
//
// A NoC read under 64 bytes from DRAM needs the same offset mod 64 at both ends on Blackhole (a token's row and a
// head's row share it for half the heads only); from L1 the rule is 16 bytes, which every face row meets. Hence Q
// in L1. The writes (32 bytes) need the same offset mod 16: they meet it in L1 and DRAM alike.
namespace sdpa_heads_first {

constexpr uint32_t FACE_BYTES = 512;  // a 16 x 16 face of bf16
constexpr uint32_t ROW_BYTES = 32;    // a face row: 16 bf16

// Row r's offset (in its first face) in a bf16 tile: faces 0 / 1 hold rows 0-15, faces 2 / 3 rows 16-31.
FORCE_INLINE constexpr uint32_t row_offset(uint32_t r) { return (r / 16) * 2 * FACE_BYTES + (r % 16) * ROW_BYTES; }

// Pages page, page + 1, .. of an interleaved L1 / DRAM tensor (page p in bank p mod the banks), one after another
// without a division a page.
template <bool DRAM>
struct PageWalk {
    uint32_t bank;
    uint32_t row_addr;  // the tensor's address plus this bank row's offset
    const uint32_t page_bytes;

    FORCE_INLINE PageWalk(uint32_t addr, uint32_t page_bytes_, uint32_t page) : page_bytes(page_bytes_) {
        const uint32_t row = interleaved_addr_gen::get_bank_offset_index<DRAM>(page);
        bank = interleaved_addr_gen::get_bank_index<DRAM>(page, row);
        row_addr = addr + row * page_bytes;
    }
    FORCE_INLINE uint64_t noc_addr(uint32_t offset, uint8_t noc) const {
        return noc_address_backend::bank_address<DRAM>(
            bank, row_addr + interleaved_addr_gen::get_bank_offset<DRAM>(bank) + offset, noc);
    }
    FORCE_INLINE void next() {
        constexpr uint32_t banks = DRAM ? NUM_DRAM_BANKS : NUM_L1_BANKS;
        if (++bank == banks) {
            bank = 0;
            row_addr += page_bytes;
        }
    }
};

// The 32-byte NoC transfers, one page (two face rows: offsets o and o + FACE_BYTES) at a time. Lean: issued with the
// command buffer's state (the length and the PCIe bits set once, each page's coordinate once for its two rows, the
// rows' addresses and the command each) and counted once at the end (dedicated NoC mode's counters); else one plain
// call a row (each writes every register and counts itself).
FORCE_INLINE uint32_t coordinate(uint64_t a) { return (uint32_t)(a >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK; }

template <bool Lean>
struct RowReads {
    uint32_t n = 0;
    const uint8_t noc;
    FORCE_INLINE RowReads(uint64_t first, uint8_t noc_) : noc(noc_) {
        if constexpr (Lean) {
            static_assert(noc_mode == DM_DEDICATED_NOC, "the counters of dedicated NoC mode");
            ncrisc_noc_read_set_state<noc_mode, true /* one_packet */>(noc, read_cmd_buf, first, ROW_BYTES);
        }
    }
    FORCE_INLINE void page(uint64_t src, uint32_t dst) {
        if constexpr (Lean) {
            while (!noc_cmd_buf_ready(noc, read_cmd_buf));
            NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, coordinate(src));
            // (the first issue after the poll above: the coordinate does not make the buffer busy)
            ncrisc_noc_read_with_state<noc_mode, false, true, true>(noc, read_cmd_buf, (uint32_t)src, dst);
            ncrisc_noc_read_with_state<noc_mode, false, true, false>(
                noc, read_cmd_buf, (uint32_t)src + FACE_BYTES, dst + FACE_BYTES);
            n += 2;
        } else {
            noc_async_read_one_packet(src, dst, ROW_BYTES, noc);
            noc_async_read_one_packet(src + FACE_BYTES, dst + FACE_BYTES, ROW_BYTES, noc);
        }
    }
    FORCE_INLINE ~RowReads() {
        if constexpr (Lean) {
            noc_reads_num_issued[noc] += n;
        }
    }
};

template <bool Lean>
struct RowWrites {
    uint32_t n = 0;
    const uint8_t noc;
    FORCE_INLINE RowWrites(uint64_t first, uint8_t noc_) : noc(noc_) {
        if constexpr (Lean) {
            static_assert(noc_mode == DM_DEDICATED_NOC, "the counters of dedicated NoC mode");
            ncrisc_noc_write_set_state<false /* posted */, true /* one_packet */>(
                noc, write_cmd_buf, first, ROW_BYTES, NOC_UNICAST_WRITE_VC);
        }
    }
    FORCE_INLINE void page(uint32_t src, uint64_t dst) {
        if constexpr (Lean) {
            while (!noc_cmd_buf_ready(noc, write_cmd_buf));
            NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, coordinate(dst));
            ncrisc_noc_write_with_state<noc_mode, false, false, true, true>(noc, write_cmd_buf, src, (uint32_t)dst);
            ncrisc_noc_write_with_state<noc_mode, false, false, true, false>(
                noc, write_cmd_buf, src + FACE_BYTES, (uint32_t)dst + FACE_BYTES);
            n += 2;
        } else {
            noc_async_write_one_packet(src, dst, ROW_BYTES, noc);
            noc_async_write_one_packet(src + FACE_BYTES, dst + FACE_BYTES, ROW_BYTES, noc);
        }
    }
    FORCE_INLINE ~RowWrites() {
        if constexpr (Lean) {
            noc_nonposted_writes_num_issued[noc] += n;
            noc_nonposted_writes_acked[noc] += n;
        }
    }
};

// Token b's rows of heads [h0, h1) from Q [1, NH, B, D] (in L1; Bt tile rows a head, Dt tiles a row, pages of
// page_bytes) into the token's Q tiles at l1 (tile_bytes each: a half tile holds heads 0-15, a full one 0-31), each
// head's at the row the legacy layout has it in. Reads only (counted): the caller waits for them.
template <bool Lean, uint32_t Bt, uint32_t Dt, uint32_t tile_bytes>
FORCE_INLINE void read_rows(
    uint32_t q_addr, uint32_t page_bytes, uint32_t b, uint32_t h0, uint32_t h1, uint32_t l1, uint8_t noc) {
    if (h0 >= h1) {
        return;
    }
    const uint32_t src = row_offset(b % 32);
    RowReads<Lean> reads(PageWalk<false>(q_addr, page_bytes, (h0 * Bt + b / 32) * Dt).noc_addr(src, noc), noc);
    for (uint32_t h = h0; h < h1; ++h) {
        PageWalk<false> w(q_addr, page_bytes, (h * Bt + b / 32) * Dt);
        uint32_t dst = l1 + row_offset(h);
        for (uint32_t t = 0; t < Dt; ++t, dst += tile_bytes) {
            reads.page(w.noc_addr(src, noc), dst);
            w.next();
        }
    }
}

// The output tiles' rows of heads [h0, h1) (at l1, tile_bytes each, head h at its row as the legacy layout has it)
// to token b's rows of the output [1, NH, B, vD] (interleaved, L1 or DRAM). Writes only (counted): the caller waits
// for them.
template <bool Lean, bool DRAM, uint32_t Bt, uint32_t Dt, uint32_t tile_bytes>
FORCE_INLINE void write_rows(
    uint32_t out_addr, uint32_t page_bytes, uint32_t b, uint32_t h0, uint32_t h1, uint32_t l1, uint8_t noc) {
    if (h0 >= h1) {
        return;
    }
    const uint32_t dst = row_offset(b % 32);
    RowWrites<Lean> writes(PageWalk<DRAM>(out_addr, page_bytes, (h0 * Bt + b / 32) * Dt).noc_addr(dst, noc), noc);
    for (uint32_t h = h0; h < h1; ++h) {
        PageWalk<DRAM> w(out_addr, page_bytes, (h * Bt + b / 32) * Dt);
        uint32_t src = l1 + row_offset(h);
        for (uint32_t t = 0; t < Dt; ++t, src += tile_bytes) {
            writes.page(src, w.noc_addr(dst, noc));
            w.next();
        }
    }
}

// The output's rows past its B tokens (rows r0 .. 31 of the last tile row, 0 < r0 < 32) set to zero: of the NH x Dt
// tiles, this core (index i of n) takes tiles i, i + n, ... A zeroed run of bytes (1 KB) at zeros. Writes only.
template <bool DRAM, uint32_t NH, uint32_t Bt, uint32_t Dt, uint32_t r0>
FORCE_INLINE void zero_padding(
    uint32_t out_addr, uint32_t page_bytes, uint32_t zeros, uint32_t i, uint32_t n, uint8_t noc) {
    static_assert(r0 > 0 && r0 < 32, "rows r0 .. 31 of the last tile row");
    const InterleavedAddrGen<DRAM> out{.bank_base_address = out_addr, .page_size = page_bytes};
    for (uint32_t t = i; t < NH * Dt; t += n) {
        const uint32_t page = ((t / Dt) * Bt + Bt - 1) * Dt + t % Dt;
        if constexpr (r0 < 16) {  // rows r0 .. 15 of faces 0 and 1, then faces 2 and 3 whole
            noc_async_write(zeros, out.get_noc_addr(page, r0 * ROW_BYTES, noc), (16 - r0) * ROW_BYTES, noc);
            noc_async_write(
                zeros, out.get_noc_addr(page, FACE_BYTES + r0 * ROW_BYTES, noc), (16 - r0) * ROW_BYTES, noc);
            noc_async_write(zeros, out.get_noc_addr(page, 2 * FACE_BYTES, noc), 2 * FACE_BYTES, noc);
        } else {  // rows r0 - 16 .. 15 of faces 2 and 3
            noc_async_write(zeros, out.get_noc_addr(page, row_offset(r0), noc), (32 - r0) * ROW_BYTES, noc);
            noc_async_write(
                zeros, out.get_noc_addr(page, row_offset(r0) + FACE_BYTES, noc), (32 - r0) * ROW_BYTES, noc);
        }
    }
}

}  // namespace sdpa_heads_first
