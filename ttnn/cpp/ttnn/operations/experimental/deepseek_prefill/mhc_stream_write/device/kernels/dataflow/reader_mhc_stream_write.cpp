// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per output tile (row tile rt: tokens 8 rt .. 8 rt + 7, column tile c): the streams tile (rt, c) and the block_out
// tile (rt / 4, c), its rows past the tokens zeroed (they meet zero columns of Q, but may hold any bits).  When rt
// changes, W and Q for its 8 tokens (bf16, rounded to nearest even as HF's .to(bfloat16)):
//   W[4a + j][4a + i] = comb_t[i][j],  Q[4a + j][8 (rt % 4) + a] = post_t[j]   (t = 8 rt + a < T; zero otherwise)
// A tile is 4 faces of 16 x 16; element (r, c) is at face 2 (r / 16) + c / 16, row r % 16, column c % 16.
// The zeros come over the NoC from the core's own zero region (the RISC-V's stores take ~1 us a KB).
// The units' reads are in flight as many at a time as cb_s and cb_bo have pages (at most d_tiles: unit i in page
// i % pages, with the transaction id 1 + i % pages), waited for one by one in order.  A row tile's W and Q (their
// pages zeroed, its comb and post rows read: transaction id TRID_WQ) start with the reads of its first unit and are
// made when that unit is pushed, while the next units' reads are in flight.  One row tile's W and Q at a time: the
// first unit of the next row tile is not issued before (a range may start a few units before a row tile's end).
// MHC_CP (mhc_stream_write_read's decode forms, defined as the CB index; MHC_CP_SEM its semaphore): every write core
// would read the same comb and post page (one DRAM bank's requests in a queue, ~3 us at 8 tokens), so a mix
// core reads their pages once and multicasts them into this CB (at the same address on every write core: comb's pages
// then post's), setting MHC_CP_SEM to 1 after.
namespace {
constexpr uint32_t FACE_ROW_BYTES_F32 = 64;
constexpr uint32_t FACE_BYTES_F32 = 1024;
constexpr uint32_t TRID_WQ = 14, TRID_ZERO = 15;  // (the units' 1 .. pages, pages <= 13)

FORCE_INLINE uint32_t to_bf16(float x) {
    union {
        float f;
        uint32_t u;
    } v{.f = x};
    const uint32_t u = v.u + 0x7FFFu + ((v.u >> 16) & 1u);
    return u >> 16;
}

// the byte offset of token t's row (16 fp32) in its fp32 tile of a [T, 16] tensor: face 2 (t % 32) / 16, row t % 16
FORCE_INLINE uint32_t row_offset(uint32_t t) { return ((t % 32) / 16) * 2 * FACE_BYTES_F32 + (t % 16) * FACE_ROW_BYTES_F32; }

// `bytes` (a multiple of 32) of zeros at `dst`; not waited for
FORCE_INLINE void zero_async(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + o, bytes - o < MEM_ZEROS_SIZE ? bytes - o : MEM_ZEROS_SIZE);
    }
}

// rows r0 .. 31 of the bf16 tile at `t` -> 0 (a face row is 32 bytes; faces 0, 1 hold rows 0-15, faces 2, 3 16-31)
FORCE_INLINE void zero_rows_async(uint32_t t, uint32_t r0) {
    for (uint32_t f = 0; f < 4; ++f) {
        const uint32_t lo = f < 2 ? 0 : 16;
        const uint32_t from = r0 > lo ? r0 - lo : 0;
        if (from < 16) {
            zero_async(t + f * 512 + from * 32, (16 - from) * 32);
        }
    }
}

// W and Q (zeroed) of row tile rt from its tokens' comb and post rows (n <= 8 tokens, 64 bytes each).  Token a's
// rows 4a .. 4a + 3 and W's columns 4a .. 4a + 3 lie in one face (0 for a < 4, 3 after), the 4 columns consecutive in
// a face row: two words a row (column 4a at an even half word).  Q's column 8 (rt % 4) + a: a half word a row.
FORCE_INLINE void make_wq(uint32_t w_addr, uint32_t q_addr, uint32_t comb_addr, uint32_t post_addr, uint32_t rt, uint32_t n) {
    uint32_t* w32 = reinterpret_cast<uint32_t*>(w_addr);
    uint16_t* q16 = reinterpret_cast<uint16_t*>(q_addr);
    const float* comb = reinterpret_cast<const float*>(comb_addr);
    const float* post = reinterpret_cast<const float*>(post_addr);
    const uint32_t qc = 8 * (rt % 4);
    for (uint32_t a = 0; a < n; ++a) {
        const uint32_t r = (4 * a) % 16;  // the face row (and column) of 4a
        const uint32_t w0 = (a < 4 ? 0 : 3) * 256 + r * 16 + r;
        const uint32_t col = qc + a;
        const uint32_t q0 = ((a < 4 ? 0 : 2) + col / 16) * 256 + r * 16 + col % 16;
        const float* m = comb + a * 16;  // m[4 i + j] = comb_t[i][j]
        const float* p = post + a * 16;
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t h = w0 + j * 16;
            w32[h / 2] = to_bf16(m[j]) | (to_bf16(m[4 + j]) << 16);
            w32[h / 2 + 1] = to_bf16(m[8 + j]) | (to_bf16(m[12 + j]) << 16);
            q16[q0 + j * 16] = static_cast<uint16_t>(to_bf16(p[j]));
        }
    }
    asm volatile("" ::: "memory");  // the stores before the pages' push
}
}  // namespace

void kernel_main() {
    const uint32_t streams_addr = get_arg_val<uint32_t>(0);
    const uint32_t block_out_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb_s = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bo = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_q = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(4);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t tokens = get_compile_time_arg_val(6);
    constexpr auto streams_args = TensorAccessorArgs<7>();
    constexpr auto block_out_args = TensorAccessorArgs<streams_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<block_out_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    const uint32_t tile = get_local_cb_interface(cb_s).fifo_page_size;  // bf16
    const uint32_t pages = get_local_cb_interface(cb_s).fifo_num_pages;  // (cb_bo's the same)
    const uint32_t ahead = pages < d_tiles ? pages : d_tiles;
    const auto s_streams = TensorAccessor(streams_args, streams_addr);
    const auto s_block_out = TensorAccessor(block_out_args, block_out_addr);
    [[maybe_unused]] const auto s_post = TensorAccessor(post_args, post_addr);
    [[maybe_unused]] const auto s_comb = TensorAccessor(comb_args, comb_addr);

    CircularBuffer s_cb(cb_s), bo(cb_bo), w(cb_w), q(cb_q), scr(cb_scr);
    [[maybe_unused]] const uint32_t scr_addr = scr.get_write_ptr();  // comb rows at 0, post rows at 512 (8 tokens x 64 bytes each)
    const uint32_t s_base = s_cb.get_write_ptr(), bo_base = bo.get_write_ptr();  // page 0 (at the launch)

    uint32_t issued = 0, issued_rt = 0xFFFFFFFFu, slot_in = 0;
    bool wq_pending = false;  // issued_rt's W and Q started, not made yet
    auto may_issue = [&](uint32_t limit) {
        return issued < count && issued < limit && !(wq_pending && (start + issued) / d_tiles != issued_rt);
    };
    auto issue = [&](uint32_t pushed) {  // the next unit's reads (`pushed` units pushed so far)
        const uint32_t u = start + issued;
        const uint32_t rt = u / d_tiles;
        const uint32_t c = u % d_tiles;
        if (rt != issued_rt) {
            issued_rt = rt;
            wq_pending = true;
            w.reserve_back(1);
            q.reserve_back(1);
            noc_async_read_set_trid(TRID_WQ);
#ifndef MHC_CP
            // the 8 tokens' rows are consecutive rows of one face: one read each for comb and post
            const uint32_t t0 = 8 * rt;
            const uint32_t n = tokens - t0 < 8 ? tokens - t0 : 8;
            noc_async_read(s_comb.get_noc_addr(t0 / 32, row_offset(t0)), scr_addr, n * FACE_ROW_BYTES_F32);
            noc_async_read(s_post.get_noc_addr(t0 / 32, row_offset(t0)), scr_addr + 512, n * FACE_ROW_BYTES_F32);
#endif
            zero_async(w.get_write_ptr(), tile);
            zero_async(q.get_write_ptr(), tile);
        }
        s_cb.reserve_back(issued - pushed + 1);
        bo.reserve_back(issued - pushed + 1);
        noc_async_read_set_trid(1 + slot_in);
        noc_async_read(s_streams.get_noc_addr(rt * d_tiles + c), s_base + slot_in * tile, tile);
        noc_async_read(s_block_out.get_noc_addr((rt / 4) * d_tiles + c), bo_base + slot_in * tile, tile);
        slot_in = slot_in + 1 == pages ? 0 : slot_in + 1;
        ++issued;
    };

    while (may_issue(ahead)) {
        issue(0);
    }
    uint32_t made_rt = 0xFFFFFFFFu, slot_out = 0;
#ifdef MHC_CP
    constexpr uint32_t cp_pages = (tokens + 31) / 32, tile_f32 = 4 * FACE_BYTES_F32;
    const uint32_t cp_addr = get_local_cb_interface(MHC_CP).fifo_wr_ptr;
    bool cp_landed = false;
#endif
    for (uint32_t p = 0; p < count; ++p) {
        const uint32_t rt = (start + p) / d_tiles;
        if (rt != made_rt) {
            made_rt = rt;
            const uint32_t t0 = 8 * rt;
            noc_async_read_barrier_with_trid(TRID_WQ);
#ifdef MHC_CP
            if (!cp_landed) {
                noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MHC_CP_SEM)), 1);
                cp_landed = true;
            }
            const uint32_t comb_rows = cp_addr + (t0 / 32) * tile_f32 + row_offset(t0);
            const uint32_t post_rows = comb_rows + cp_pages * tile_f32;
#else
            const uint32_t comb_rows = scr_addr, post_rows = scr_addr + 512;
#endif
            make_wq(w.get_write_ptr(), q.get_write_ptr(), comb_rows, post_rows, rt, tokens - t0 < 8 ? tokens - t0 : 8);
            w.push_back(1);
            q.push_back(1);
            wq_pending = false;
        }
        noc_async_read_barrier_with_trid(1 + slot_out);
        // block_out rows past the tokens -> 0
        const uint32_t first = (rt / 4) * 32;
        if (first + 32 > tokens) {
            noc_async_read_set_trid(TRID_ZERO);
            zero_rows_async(bo_base + slot_out * tile, tokens > first ? tokens - first : 0);
            noc_async_read_barrier_with_trid(TRID_ZERO);
        }
        s_cb.push_back(1);
        bo.push_back(1);
        slot_out = slot_out + 1 == pages ? 0 : slot_out + 1;
        while (may_issue(p + 1 + ahead)) {
            issue(p + 1);
        }
    }
    noc_async_read_set_trid(0);
}
