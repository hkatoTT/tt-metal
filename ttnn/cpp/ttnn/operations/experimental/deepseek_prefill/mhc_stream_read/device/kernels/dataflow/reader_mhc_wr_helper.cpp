// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// mhc_stream_write_read's prefill groups with the write spread over the cores (MHC_WR_MERGED): a helper core (one the
// groups leave free) makes the new streams tiles of slice columns k_own .. K - 1 of the slices of its units (a unit:
// slice j = g S + s, column kk; 4 tiles, its 4 row tiles), handed to the slice core by the writer.  This reader: per
// group met, W and Q of its 4 row tiles (as reader_mhc_wr_slice_groups.cpp makes them); per unit, the block_out tile
// (g, s K + kk) and the 4 old streams tiles (4 g + rt, s K + kk).
//   CT: CB_SOLD, CB_BO, CB_W, CB_Q, d_tiles, k_tiles, k_own, slices, CB_SCR (4 KB: a group's comb and post rows),
//       then the accessors of the old streams, block_out, post, comb.  W and Q have 2 groups' pages (the next group's
//       made while the compute still holds the last's)
//   RT: those 4 addresses, the first unit, the units, the call's first group in the tensors (offset)
namespace {
constexpr uint32_t ROW_TILES = 4;
constexpr uint32_t FACE_ROW_BYTES_F32 = 64;
constexpr uint32_t FACE_BYTES_F32 = 1024;
constexpr uint32_t TRID_WQ = 10, TRID_ZERO = 11;  // (the units' reads 1 .. 8)

FORCE_INLINE uint32_t to_bf16(float x) {
    union {
        float f;
        uint32_t u;
    } v{.f = x};
    const uint32_t u = v.u + 0x7FFFu + ((v.u >> 16) & 1u);
    return u >> 16;
}

FORCE_INLINE uint32_t row_offset(uint32_t t) { return ((t % 32) / 16) * 2 * FACE_BYTES_F32 + (t % 16) * FACE_ROW_BYTES_F32; }

FORCE_INLINE void zero_async(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + o, bytes - o < MEM_ZEROS_SIZE ? bytes - o : MEM_ZEROS_SIZE);
    }
}

// as reader_mhc_stream_write.cpp's make_wq
FORCE_INLINE void make_wq(uint32_t w_addr, uint32_t q_addr, uint32_t comb_addr, uint32_t post_addr, uint32_t rt, uint32_t n) {
    uint32_t* w32 = reinterpret_cast<uint32_t*>(w_addr);
    uint16_t* q16 = reinterpret_cast<uint16_t*>(q_addr);
    const float* comb = reinterpret_cast<const float*>(comb_addr);
    const float* post = reinterpret_cast<const float*>(post_addr);
    const uint32_t qc = 8 * (rt % 4);
    for (uint32_t a = 0; a < n; ++a) {
        const uint32_t r = (4 * a) % 16;
        const uint32_t w0 = (a < 4 ? 0 : 3) * 256 + r * 16 + r;
        const uint32_t col = qc + a;
        const uint32_t q0 = ((a < 4 ? 0 : 2) + col / 16) * 256 + r * 16 + col % 16;
        const float* m = comb + a * 16;
        const float* p = post + a * 16;
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t h = w0 + j * 16;
            w32[h / 2] = to_bf16(m[j]) | (to_bf16(m[4 + j]) << 16);
            w32[h / 2 + 1] = to_bf16(m[8 + j]) | (to_bf16(m[12 + j]) << 16);
            q16[q0 + j * 16] = static_cast<uint16_t>(to_bf16(p[j]));
        }
    }
    asm volatile("" ::: "memory");
}
}  // namespace

void kernel_main() {
    const uint32_t streams_addr = get_arg_val<uint32_t>(0);
    const uint32_t block_out_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    const uint32_t off_g = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_sold = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bo = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_q = get_compile_time_arg_val(3);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t k_own = get_compile_time_arg_val(6);
    constexpr uint32_t slices = get_compile_time_arg_val(7);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(8);
    constexpr auto streams_args = TensorAccessorArgs<9>();
    constexpr auto block_out_args = TensorAccessorArgs<streams_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<block_out_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr uint32_t helped = k_tiles - k_own;  // the helped columns of a slice

    const uint32_t tile = get_local_cb_interface(cb_sold).fifo_page_size;
    const uint32_t sold_pages = get_local_cb_interface(cb_sold).fifo_num_pages;
    const uint32_t ahead = sold_pages / ROW_TILES;  // units in flight (<= 8)
    const auto s_streams = TensorAccessor(streams_args, streams_addr);
    const auto s_block_out = TensorAccessor(block_out_args, block_out_addr);
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);
    CircularBuffer sold(cb_sold), bo(cb_bo), w(cb_w), q(cb_q);
    const uint32_t sold_base = sold.get_write_ptr(), bo_base = bo.get_write_ptr();
    const uint32_t scr = get_local_cb_interface(cb_scr).fifo_wr_ptr;  // a group's comb and post rows

    auto group_of = [&](uint32_t u) { return (u / helped) / slices; };
    auto issue = [&](uint32_t i) {  // unit start + i's reads into slot i % ahead
        const uint32_t u = start + i;
        const uint32_t j = u / helped, kk = k_own + u % helped;
        const uint32_t g = off_g + j / slices, s = j % slices;
        const uint32_t c = s * k_tiles + kk, slot = i % ahead;
        noc_async_read_set_trid(1 + i % 8);
#ifdef MHC_WR_PROBE_NOREAD
        return;  // (probe: no reads, the stale tiles computed)
#endif
        noc_async_read(s_block_out.get_noc_addr(g * d_tiles + c), bo_base + slot * tile, tile);
        for (uint32_t r = 0; r < ROW_TILES; ++r) {
            noc_async_read(
                s_streams.get_noc_addr((4 * g + r) * d_tiles + c), sold_base + (slot * ROW_TILES + r) * tile, tile);
        }
    };
    uint32_t issued = 0, wq_g = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t gl = group_of(start + i);
        if (gl != wq_g) {  // W and Q of this group (the compute pops the last group's when the group changes)
            wq_g = gl;
            const uint32_t g = off_g + gl;
            w.reserve_back(ROW_TILES);
            q.reserve_back(ROW_TILES);
            noc_async_read_set_trid(TRID_WQ);
            for (uint32_t r = 0; r < ROW_TILES; ++r) {
                const uint32_t t0 = 32 * g + 8 * r;
                noc_async_read(s_comb.get_noc_addr(g, row_offset(t0)), scr + r * 1024, 8 * FACE_ROW_BYTES_F32);
                noc_async_read(s_post.get_noc_addr(g, row_offset(t0)), scr + r * 1024 + 512, 8 * FACE_ROW_BYTES_F32);
            }
            noc_async_read_set_trid(TRID_ZERO);
            zero_async(w.get_write_ptr(), ROW_TILES * tile);
            zero_async(q.get_write_ptr(), ROW_TILES * tile);
            noc_async_read_barrier_with_trid(TRID_ZERO);
            noc_async_read_barrier_with_trid(TRID_WQ);
            for (uint32_t r = 0; r < ROW_TILES; ++r) {
                make_wq(w.get_write_ptr() + r * tile, q.get_write_ptr() + r * tile, scr + r * 1024, scr + r * 1024 + 512,
                        4 * g + r, 8);  // fmt: skip
            }
            w.push_back(ROW_TILES);
            q.push_back(ROW_TILES);
        }
        while (issued < count && issued < i + ahead) {
            sold.reserve_back((issued - i + 1) * ROW_TILES);
            bo.reserve_back(issued - i + 1);
            issue(issued);
            ++issued;
        }
        noc_async_read_barrier_with_trid(1 + i % 8);
        sold.push_back(ROW_TILES);
        bo.push_back(1);
    }
    noc_async_read_set_trid(0);
}
