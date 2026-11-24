// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// mhc_stream_write_read's prefill groups with the write on the slice cores (MHC_WR_MERGED): slice s of group g (32
// tokens, 4 row tiles) makes its own tiles of the new streams, so no write cores, hand-off or semaphore increments.
// This reader: (as reader_mhc_read_slice_groups.cpp) its part of fn and gamma, multicast by the writer to the column's
// other groups (`parts`); then the write's operands for the compute: the group's W and Q of its 4 row tiles (from its
// tokens' comb and post rows, as reader_mhc_stream_write.cpp makes them: bf16, nearest even), then per column k of the
// slice the block_out tile (g, s K + k) (once: the 4 row tiles' Q products read it) and the 4 old streams tiles
// (rt, s K + k), for its own columns k < k_own (the helper cores make the others' tiles and write them into CB_S: once
// every sender's flag is set, CB_READY's page tells the compute); then P and the norm's scale from the mix core (their
// semaphores), as the read's reader.
//   CT: CB_SOLD, CB_BO, CB_W, CB_Q, CB_FN, CB_G, CB_PIN, CB_NS, d_tiles, k_tiles, SEM_P, SEM_NS, parts, SEM_OWN, SEM_FN0,
//       k_own, SEM_H0, CB_READY, CB_S, then the accessors of the old streams, block_out, post, comb, fn, gamma, new
//       streams
//   RT: those 6 addresses, s, g (the group in the tensors: the call's offset included), part, its senders, the new
//       streams' address
// Once CB_S is whole (its own columns made by the compute, the helpers' landed), it writes the slice's new streams to
// DRAM (page (4 g + rt) d_tiles + s K + k; this NoC is idle until P), while the compute projects them.
namespace {
constexpr uint32_t ROW_TILES = 4;  // a group's 32 tokens
constexpr uint32_t PROJ_TILES = 3;
constexpr uint32_t FACE_ROW_BYTES_F32 = 64;
constexpr uint32_t FACE_BYTES_F32 = 1024;
constexpr uint32_t TRID_FN = 9, TRID_WQ = 10, TRID_ZERO = 11;  // (the columns' reads 1 .. 8)

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

FORCE_INLINE void zero_async(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + o, bytes - o < MEM_ZEROS_SIZE ? bytes - o : MEM_ZEROS_SIZE);
    }
}

// W and Q (zeroed) of row tile rt from its 8 tokens' comb and post rows (64 bytes each), as reader_mhc_stream_write.cpp
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
    const uint32_t fn_addr = get_arg_val<uint32_t>(4);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(5);
    const uint32_t s = get_arg_val<uint32_t>(6);
    const uint32_t g = get_arg_val<uint32_t>(7);
    const uint32_t part = get_arg_val<uint32_t>(8);
    const uint32_t senders = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_sold = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bo = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_q = get_compile_time_arg_val(3);
    constexpr uint32_t cb_fn = get_compile_time_arg_val(4);
    constexpr uint32_t cb_g = get_compile_time_arg_val(5);
    constexpr uint32_t cb_pin = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ns = get_compile_time_arg_val(7);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t sem_p = get_compile_time_arg_val(10);
    constexpr uint32_t sem_ns = get_compile_time_arg_val(11);
    constexpr uint32_t parts = get_compile_time_arg_val(12);  // 1: fn and gamma read whole here
    constexpr uint32_t sem_own = get_compile_time_arg_val(13);
    constexpr uint32_t sem_fn0 = get_compile_time_arg_val(14);
    constexpr uint32_t k_own = get_compile_time_arg_val(15);
    constexpr uint32_t sem_h0 = get_compile_time_arg_val(16);
    constexpr uint32_t cb_ready = get_compile_time_arg_val(17);
    constexpr uint32_t cb_s = get_compile_time_arg_val(18);
    constexpr auto streams_args = TensorAccessorArgs<19>();
    constexpr auto block_out_args = TensorAccessorArgs<streams_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<block_out_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto fn_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<fn_args.next_compile_time_args_offset()>();
    constexpr auto new_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t tile = get_local_cb_interface(cb_sold).fifo_page_size;  // bf16
    const uint32_t sold_pages = get_local_cb_interface(cb_sold).fifo_num_pages;
    const auto s_streams = TensorAccessor(streams_args, streams_addr);
    const auto s_block_out = TensorAccessor(block_out_args, block_out_addr);
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);
    const auto s_fn = TensorAccessor(fn_args, fn_addr);
    const auto s_gamma = TensorAccessor(gamma_args, gamma_addr);
    const auto s_new = TensorAccessor(new_args, get_arg_val<uint32_t>(10));

    CircularBuffer sold(cb_sold), bo(cb_bo), w(cb_w), q(cb_q), fn(cb_fn), gm(cb_g), pin(cb_pin), ns(cb_ns);

    // fn and gamma (as the read's reader): its part, the others' by the column's multicasts
    fn.reserve_back(k_tiles * PROJ_TILES);
    gm.reserve_back(k_tiles);
    const uint32_t fn_base = fn.get_write_ptr(), g_base = gm.get_write_ptr();
    constexpr uint32_t kp = k_tiles / parts;
    const uint32_t k0 = parts > 1 ? part * kp : 0, k1 = parts > 1 ? (part + 1) * kp : k_tiles;
    noc_async_read_set_trid(TRID_FN);
    for (uint32_t k = k0; k < k1; ++k) {
        const uint32_t kt = s * k_tiles + k;
        for (uint32_t n = 0; n < PROJ_TILES; ++n) {
            noc_async_read(s_fn.get_noc_addr(kt * PROJ_TILES + n), fn_base + (k * PROJ_TILES + n) * tile, tile);
        }
        noc_async_read(s_gamma.get_noc_addr(kt), g_base + k * tile, tile);
    }

    // W and Q of the group's 4 row tiles: the 32 tokens' comb and post rows (8 tokens a read: consecutive face rows)
    w.reserve_back(ROW_TILES);
    q.reserve_back(ROW_TILES);
    const uint32_t w_base = w.get_write_ptr(), q_base = q.get_write_ptr();
    // (the scratch: the old streams' CB before its first use, ROW_TILES pages of comb rows then post rows)
    const uint32_t scr = get_local_cb_interface(cb_sold).fifo_wr_ptr;
    noc_async_read_set_trid(TRID_WQ);
    for (uint32_t r = 0; r < ROW_TILES; ++r) {
        const uint32_t t0 = 32 * g + 8 * r;
        noc_async_read(s_comb.get_noc_addr(g, row_offset(t0)), scr + r * 1024, 8 * FACE_ROW_BYTES_F32);
        noc_async_read(s_post.get_noc_addr(g, row_offset(t0)), scr + r * 1024 + 512, 8 * FACE_ROW_BYTES_F32);
    }
    noc_async_read_set_trid(TRID_ZERO);
    zero_async(w_base, ROW_TILES * tile);
    zero_async(q_base, ROW_TILES * tile);
    noc_async_read_barrier_with_trid(TRID_ZERO);
    noc_async_read_barrier_with_trid(TRID_WQ);
    for (uint32_t r = 0; r < ROW_TILES; ++r) {
        make_wq(w_base + r * tile, q_base + r * tile, scr + r * 1024, scr + r * 1024 + 512, 4 * g + r, 8);
    }
    w.push_back(ROW_TILES);
    q.push_back(ROW_TILES);
    noc_async_read_barrier_with_trid(TRID_FN);
    noc_async_read_set_trid(0);
    if constexpr (parts > 1) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own)), 1);
    }

    // per column k: block_out (g, s K + k) and the old streams (4 g + rt, s K + k), up to the CBs' pages in flight
    const uint32_t ahead = sold_pages / ROW_TILES;  // columns in flight (cb_bo has as many pages)
    const uint32_t sold_base = sold.get_write_ptr(), bo_base = bo.get_write_ptr();
    uint32_t issued = 0;
    auto issue = [&](uint32_t k) {
        const uint32_t slot = k % ahead;
        const uint32_t c = s * k_tiles + k;
        noc_async_read_set_trid(1 + k % 8);
#ifdef MHC_WR_PROBE_NOREAD
        return;  // (probe)
#endif
        noc_async_read(s_block_out.get_noc_addr(g * d_tiles + c), bo_base + slot * tile, tile);
        for (uint32_t r = 0; r < ROW_TILES; ++r) {
            noc_async_read(
                s_streams.get_noc_addr((4 * g + r) * d_tiles + c), sold_base + (slot * ROW_TILES + r) * tile, tile);
        }
    };
    // fn's columns as the compute reads them: each other part's multicast waited for at its first column
    auto push_fn = [&](uint32_t k) {
        if constexpr (parts > 1) {
            const uint32_t qq = k / kp;
            if (qq != part && k % kp == 0) {
                noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_fn0 + qq)), 1);
            }
        }
        fn.push_back(PROJ_TILES);
    };
    for (uint32_t k = 0; k < k_own; ++k) {
        while (issued < k_own && issued < k + ahead) {
            sold.reserve_back((issued - k + 1) * ROW_TILES);
            bo.reserve_back(issued - k + 1);
            issue(issued);
            ++issued;
        }
        noc_async_read_barrier_with_trid(1 + k % 8);
        sold.push_back(ROW_TILES);
        bo.push_back(1);
        push_fn(k);
    }
    for (uint32_t k = k_own; k < k_tiles; ++k) {
        push_fn(k);
    }
    // the helpers' columns landed in CB_S: every sender's flag
    for (uint32_t i = 0; i < senders; ++i) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_h0 + i)), 1);
    }
    CircularBuffer ready(cb_ready);
    ready.reserve_back(1);
    ready.push_back(1);
#ifndef MHC_WR_WRITER_DRAM
    {  // the slice's new streams to DRAM once CB_S is whole (column major: tile k R + rt)
        CircularBuffer sn(cb_s);
        const uint32_t s_base = get_local_cb_interface(cb_s).fifo_rd_ptr;  // (nothing popped yet: the base)
        sn.wait_front(ROW_TILES * k_tiles);
#ifndef MHC_WR_PROBE_NOWRITE
        for (uint32_t k = 0; k < k_tiles; ++k) {
            for (uint32_t rt = 0; rt < ROW_TILES; ++rt) {
                noc_async_write(
                    s_base + (k * ROW_TILES + rt) * tile,
                    s_new.get_noc_addr((4 * g + rt) * d_tiles + s * k_tiles + k),
                    tile);
            }
        }
#endif
    }
#endif
    noc_async_read_set_trid(0);
    gm.push_back(k_tiles);  // (every part's gamma came with its fn)

    pin.reserve_back(ROW_TILES);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_p)), 1);
    pin.push_back(ROW_TILES);
    ns.reserve_back(1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ns)), 1);
    ns.push_back(1);
    noc_async_write_barrier();  // (the new streams)
}
