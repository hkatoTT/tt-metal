// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per work unit (pair p, value block j): the state's column of tiles S_ij (i = 0..3, bf16) and v's block j; per pair
// (its first unit on this core), the q / k / decay rows (4 tiles wide) and beta, held by the compute for the pair's
// units.
// The rows come from the pair's row r of a 32-row tile row.  A vector
// goes into row 0 of its "row" tiles or column 0 of its "col" tiles (the compute broadcasts it across columns); the
// other rows and columns are never used (the compute's rows are independent), so they are not cleared.  An fp32 tile
// is 4 faces of 16 x 16 (1 KiB each): row r is 64 bytes in face 2 (r / 16) and 64 in the next face; column 0 is the
// first element of each face row of faces 0 and 2.  The writer writes the new state back in place.
namespace {
constexpr uint32_t FACE_BYTES = 1024;
constexpr uint32_t FACE_ROW_BYTES = 64;
constexpr uint32_t N = 4;  // tiles per 128

// Fill column 0 of tile `dst` from row 0 of tile `src`: element (t, 0) <- (0, t).  In place (src == dst) too: (0, 0)
// is both.
FORCE_INLINE void row_to_column(uint32_t src_addr, uint32_t dst_addr) {
    volatile tt_l1_ptr float* s = reinterpret_cast<volatile tt_l1_ptr float*>(src_addr);
    volatile tt_l1_ptr float* d = reinterpret_cast<volatile tt_l1_ptr float*>(dst_addr);
    for (uint32_t i = 0; i < 16; ++i) {
        d[i * 16] = s[i];  // face 0: (i, 0) <- (0, i)
    }
    for (uint32_t i = 0; i < 16; ++i) {
        d[512 + i * 16] = s[256 + i];  // face 2: (16 + i, 0) <- face 1: (0, 16 + i)
    }
}

// Byte offset from the CB's write pointer of the tile `t` tiles past it, wrapping at the end of the CB's memory: the
// batches vary in size, so one can straddle the end (the CB holds whole units, so a unit's tiles never do)
FORCE_INLINE uint32_t wrapped(uint32_t cb, uint32_t t, uint32_t tile) {
    const auto& f = get_local_cb_interface(cb);
    return f.fifo_wr_ptr + t * tile < f.fifo_limit ? t * tile : t * tile - f.fifo_size;
}

// cb_push_back for those batches: cb_push_back takes the write pointer back to the CB's start only when it lands on
// the end exactly, so after a batch that straddled it the pointer stayed past the end, and two such batches later
// `wrapped` wrote past the CB (from the 5th batch on a core: 32 users a row and more, 17+ units a core; KDA decode
// pcc 0.93 against tt/).  Here the pointer is taken back by the CB's size whenever it passes the end, as the
// compute's unpacker does with its read pointer (llk_pop_tiles).
FORCE_INLINE void push_wrapped(uint32_t cb, uint32_t n) {
    volatile tt_reg_ptr uint32_t* received = get_cb_tiles_received_ptr(cb);
    received[0] += n;
    auto& f = get_local_cb_interface(cb);
    f.fifo_wr_ptr += n * f.fifo_page_size;
    if (f.fifo_wr_ptr >= f.fifo_limit) {
        f.fifo_wr_ptr -= f.fifo_size;
    }
}
}  // namespace

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t decay_addr = get_arg_val<uint32_t>(4);
    const uint32_t beta_addr = get_arg_val<uint32_t>(5);
    const uint32_t start = get_arg_val<uint32_t>(6);
    const uint32_t count = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_s = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ecol = get_compile_time_arg_val(1);
    constexpr uint32_t cb_kcol = get_compile_time_arg_val(2);
    constexpr uint32_t cb_krow = get_compile_time_arg_val(3);
    constexpr uint32_t cb_qrow = get_compile_time_arg_val(4);
    constexpr uint32_t cb_vrow = get_compile_time_arg_val(5);
    constexpr uint32_t cb_brow = get_compile_time_arg_val(6);
    constexpr auto state_args = TensorAccessorArgs<7>();
    constexpr auto q_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto decay_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<decay_args.next_compile_time_args_offset()>();
    // the op's form (bits)
    constexpr uint32_t form = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());

    const uint32_t tile = get_local_cb_interface(cb_ecol).fifo_page_size;  // an fp32 tile (the rows, v)
    const uint32_t s_tile = get_local_cb_interface(cb_s).fifo_page_size;  // a state tile (bf16)
    const auto s_state = TensorAccessor(state_args, state_addr);
    const auto s_q = TensorAccessor(q_args, q_addr);
    const auto s_k = TensorAccessor(k_args, k_addr);
    const auto s_v = TensorAccessor(v_args, v_addr);
    const auto s_decay = TensorAccessor(decay_args, decay_addr);
    const auto s_beta = TensorAccessor(beta_args, beta_addr);

    Noc noc;
    CircularBuffer s_cb(cb_s), ecol(cb_ecol), kcol(cb_kcol), krow(cb_krow), qrow(cb_qrow), vrow(cb_vrow),
        brow(cb_brow);
    // form bit 128: a pair's value blocks on this core taken in an order rotated by the core (the writer the same), so
    // the cores' concurrent state reads and writes spread over the DRAM banks (a unit's tiles 16 p + 4 i + j sit in
    // banks j and j + 4 of 8: cores in step, as those with equal unit counts are, all hit the same two)
    constexpr bool rotate = (form & 128) != 0;
    const uint32_t rot = rotate && count > 0 ? (start / count) % N : 0;

    // One batch per pair: its units on this core (consecutive value blocks), every read issued before one barrier, so a
    // core has up to N state columns in flight rather than one.  The compute takes them a unit at a time.
    for (uint32_t u = start; u < start + count;) {
        const uint32_t p = u / N;   // pair
        const uint32_t j0 = u % N;  // its first value block here
        const uint32_t nb = N - j0 < start + count - u ? N - j0 : start + count - u;
        const uint32_t row_tile = p / 32;
        const uint32_t r = p % 32;
        const uint32_t lo = (r / 16) * 2 * FACE_BYTES + (r % 16) * FACE_ROW_BYTES;  // row r, columns 0-15
        const uint32_t hi = lo + FACE_BYTES;                                         // columns 16-31

        {
            DeviceZoneScopedN("R-SWAIT");
            s_cb.reserve_back(nb * N);
        }
        // unit b's state column into its slot b
        auto read_state = [&](uint32_t b) {
            const uint32_t jb = j0 + (b + rot) % nb;
            for (uint32_t i = 0; i < N; ++i) {
                noc.async_read(
                    s_state,
                    s_cb,
                    s_tile,
                    {.page_id = p * N * N + i * N + jb},
                    {.offset_bytes = wrapped(cb_s, b * N + i, s_tile)});
            }
        };

        // row r of tile c of the row tile -> row 0 of the destination tile at byte offset `at`
        auto read_row = [&](const auto& src, CircularBuffer& dst, uint32_t c, uint32_t at) {
            const uint32_t page = row_tile * N + c;
            noc.async_read(src, dst, FACE_ROW_BYTES, {.page_id = page, .offset_bytes = lo}, {.offset_bytes = at});
            noc.async_read(
                src, dst, FACE_ROW_BYTES, {.page_id = page, .offset_bytes = hi}, {.offset_bytes = at + FACE_BYTES});
        };
        // unit b's v row into slot b
        auto read_v = [&](uint32_t b) {
            const uint32_t jb = j0 + (b + rot) % nb;
            read_row(s_v, vrow, jb, wrapped(cb_vrow, b, tile));
        };
        vrow.reserve_back(nb);
        for (uint32_t b = 0; b < nb; ++b) {
            read_state(b);
        }
        {
            DeviceZoneScopedN("R-PAIR");
            ecol.reserve_back(N);
            kcol.reserve_back(N);
            krow.reserve_back(N);
            qrow.reserve_back(N);
            brow.reserve_back(1);
            for (uint32_t c = 0; c < N; ++c) {
                read_row(s_decay, ecol, c, c * tile);
                read_row(s_k, krow, c, c * tile);
                read_row(s_q, qrow, c, c * tile);
            }
            noc.async_read(
                s_beta, brow, FACE_ROW_BYTES, {.page_id = row_tile, .offset_bytes = lo}, {.offset_bytes = 0});
        }
        for (uint32_t b = 0; b < nb; ++b) {
            read_v(b);
        }
        {
            DeviceZoneScopedN("R-BAR");
            noc.async_read_barrier();
        }

        {
            DeviceZoneScopedN("R-T");
            for (uint32_t c = 0; c < N; ++c) {
                row_to_column(ecol.get_write_ptr() + c * tile, ecol.get_write_ptr() + c * tile);
                row_to_column(krow.get_write_ptr() + c * tile, kcol.get_write_ptr() + c * tile);
            }
            // beta (element 0) across row 0
            volatile tt_l1_ptr float* b = reinterpret_cast<volatile tt_l1_ptr float*>(brow.get_write_ptr());
            const float beta = b[0];
            for (uint32_t t = 0; t < 16; ++t) {
                b[t] = beta;
                b[256 + t] = beta;
            }
            ecol.push_back(N);
            krow.push_back(N);
            brow.push_back(1);
            kcol.push_back(N);
            qrow.push_back(N);
        }
        push_wrapped(cb_s, nb * N);
        push_wrapped(cb_vrow, nb);
        u += nb;
    }
}
