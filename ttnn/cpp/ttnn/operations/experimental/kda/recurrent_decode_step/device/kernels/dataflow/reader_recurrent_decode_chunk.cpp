// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"

// recurrent_decode_step's chunked verify form (compute/recurrent_decode_chunk.cpp): a core's units a pair at a time.
// The pair's chunk has fixed rows: this step's T tokens at rows T .. 2T - 1, the last step's tokens redone (r =
// keep[user] + 1 of them, 1 .. T) at rows T - r .. T - 1, and rows 0 .. T - r - 1 doing nothing (decay 1, beta 0).
// The state holds the start of the last step (the op writes the start of this one over it).  Per pair (the batch of
// its units on this core):
//   - k at row p and q at row 16 + p (CB_KQ), the decay at row p and again at 16 + p (CB_AL), v at row p of each block's
//     tile (CB_V); beta in column 0 and row 0 of CB_BT (0 on the rows doing nothing); CB_MZ's row 0 1 on rows < T;
//   - the last step's saved slot (the batch's first unit's slot of `saved`, its own whole tiles: k, decay, beta and the
//     blocks' v rows of its T tokens) in a read a tile whatever r, this step's gathered in L1 and written back;
//   - S0: the batch's state tiles.
// The reads do not depend on r, so a batch's reads go out before its keep code is read (the first batch's with the
// keep codes).  The rows no batch writes (k, v rows 0 .. T - 1 at first and from 2T, q rows 16 .. 16 + T - 1 and
// 16 + 2T .., v rows 16 .., the decays' rows from 2T) are set at a CB slot's first use, behind its batch's DRAM reads
// (zeros, the decays ones), and the rows doing nothing (k and v zeros at first, later an earlier batch's: finite, beta 0
// drops them) get their decays ones each batch.  The constant tiles the compute takes (-strictly lower 16, lower 16 at
// rows 16 .., I, I at rows / columns 16 ..) are made here: zeros behind the first batch's reads, then their ones.  The
// rows of a token's vector come from row_of(): today's prepare puts token k of pair p at row k P + p (layout 0);
// layout 1 (decode_prepare's pair_major) is a pair's T tokens on consecutive rows p T + k, read at once a face.
namespace {
constexpr uint32_t FACE = 1024;  // bytes of a 16 x 16 fp32 face
constexpr uint32_t ROWB = 64;    // bytes of a face row
constexpr uint32_t N = 4;
constexpr uint32_t CB_KQ = 0, CB_AL = 1, CB_BT = 2, CB_MZ = 3, CB_V = 4, CB_S = 5, CB_CONST = 6, CB_CONSTM = 7;
constexpr uint32_t CB_KEEP = 26, CB_SCR = 27;
constexpr uint32_t N_CONST = 4;

// Byte offset from the CB's write pointer of the tile `t` tiles past it, wrapping at the end of the CB's memory
FORCE_INLINE uint32_t wrapped(uint32_t cb, uint32_t t, uint32_t tile) {
    const auto& f = get_local_cb_interface(cb);
    return f.fifo_wr_ptr + t * tile < f.fifo_limit ? t * tile : t * tile - f.fifo_size;
}

FORCE_INLINE void push_wrapped(uint32_t cb, uint32_t n) {
    volatile tt_reg_ptr uint32_t* received = get_cb_tiles_received_ptr(cb);
    received[0] += n;
    auto& f = get_local_cb_interface(cb);
    f.fifo_wr_ptr += n * f.fifo_page_size;
    if (f.fifo_wr_ptr >= f.fifo_limit) {
        f.fifo_wr_ptr -= f.fifo_size;
    }
}

FORCE_INLINE uint64_t local(uint32_t addr) { return get_noc_addr(my_x[noc_index], my_y[noc_index], addr); }
}  // namespace

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t decay_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t state_addr = get_arg_val<uint32_t>(5);
    const uint32_t saved_addr = get_arg_val<uint32_t>(6);
    const uint32_t keep_addr = get_arg_val<uint32_t>(7);
    const uint32_t start = get_arg_val<uint32_t>(8);
    const uint32_t count = get_arg_val<uint32_t>(9);
    // the last step's start state and rows read from here (the same as state / saved in place; out of place, a
    // ping-pong's other set); this step's rows still written to `saved`
    const uint32_t state_src_addr = get_arg_val<uint32_t>(10);
    const uint32_t saved_src_addr = get_arg_val<uint32_t>(11);
    (void)state_addr;

    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t pairs = get_compile_time_arg_val(1);
    constexpr uint32_t heads = get_compile_time_arg_val(2);
    constexpr uint32_t keep_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t layout = get_compile_time_arg_val(4);
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto decay_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<decay_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();
    constexpr auto saved_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto keep_args = TensorAccessorArgs<saved_args.next_compile_time_args_offset()>();
    constexpr uint32_t LF = 2 * T;  // the chunk's rows (this step's tokens from row T)
    // the slot of `saved` of a batch's first unit (bytes): the pair's k rows (4 tiles x 2 faces x T rows), the decay
    // rows (as k), beta (T floats, 64 bytes), then the batch's blocks' v rows (2 faces x T rows each, up to 4)
    constexpr uint32_t K_OFF = 0, A_OFF = 8 * T * ROWB, B_OFF = 16 * T * ROWB, V_OFF = 16 * T * ROWB + ROWB;
    constexpr uint32_t V_BLOCK = 2 * T * ROWB, SLOT = V_OFF + N * V_BLOCK;
    constexpr uint32_t PAGE = 4096, SLOT_PAGES = (SLOT + PAGE - 1) / PAGE;  // (a unit's slot: whole fp32 tiles)

    const auto s_q = TensorAccessor(q_args, q_addr);
    const auto s_k = TensorAccessor(k_args, k_addr);
    const auto s_v = TensorAccessor(v_args, v_addr);
    const auto s_decay = TensorAccessor(decay_args, decay_addr);
    const auto s_beta = TensorAccessor(beta_args, beta_addr);
    const auto s_state = TensorAccessor(state_args, state_src_addr);  // (read only here: the writer writes the state)
    const auto s_saved = TensorAccessor(saved_args, saved_addr);
    const auto s_saved_src = TensorAccessor(saved_args, saved_src_addr);
    const auto s_keep = TensorAccessor(keep_args, keep_addr);

    Noc noc;
    CircularBuffer kq(CB_KQ), al(CB_AL), bt(CB_BT), mz(CB_MZ), vv(CB_V), sc(CB_S), cs(CB_CONST), csm(CB_CONSTM);
    CircularBuffer kb(CB_KEEP), scr(CB_SCR);
    const uint32_t tile = get_local_cb_interface(CB_KQ).fifo_page_size;  // fp32
    const uint32_t s_tile = get_local_cb_interface(CB_S).fifo_page_size;  // the state's dtype

    const uint32_t st_old = scr.get_write_ptr();  // the last step's saved slot as read
    // a batch's DRAM reads: the last step's saved slot (whatever r) and S0 of its nb blocks (into CB_S from sa)
    auto read_dram = [&](uint32_t u, uint32_t nb, uint32_t sa) {
        const uint32_t p = u / N, j0 = u % N, slot_bytes = V_OFF + nb * V_BLOCK;
        for (uint32_t i = 0; i * PAGE < slot_bytes; ++i) {
            const uint32_t n_bytes = slot_bytes - i * PAGE < PAGE ? slot_bytes - i * PAGE : PAGE;
            noc_async_read(s_saved_src.get_noc_addr(u * SLOT_PAGES + i), st_old + i * PAGE, n_bytes);
        }
        for (uint32_t b = 0; b < nb; ++b) {
            for (uint32_t i = 0; i < N; ++i) {
                const uint32_t page = p * N * N + i * N + j0 + b;
                noc_async_read(s_state.get_noc_addr(page), sa + wrapped(CB_S, b * N + i, s_tile), s_tile);
            }
        }
    };
    // the first batch's issued first, its DRAM latency under the setup below
    if (count > 0) {
        const uint32_t nb0 = N - start % N < count ? N - start % N : count;
        sc.reserve_back(nb0 * N);
        read_dram(start, nb0, sc.get_write_ptr());
    }
    const uint32_t st_new = st_old + SLOT;        // this step's, gathered for its write
    const uint32_t s_new = st_new + SLOT;         // this step's beta rows (64 bytes a token)
    const uint32_t f_zero = s_new + 8 * ROWB;     // a face of zeros and one of ones (the rows no token fills)
    const uint32_t f_one = f_zero + FACE;
    volatile tt_l1_ptr float* b_old = reinterpret_cast<volatile tt_l1_ptr float*>(st_old + B_OFF);
    volatile tt_l1_ptr float* b_new = reinterpret_cast<volatile tt_l1_ptr float*>(s_new);
    volatile tt_l1_ptr float* b_stage = reinterpret_cast<volatile tt_l1_ptr float*>(st_new + B_OFF);
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(f_zero);
        for (uint32_t i = 0; i < FACE / 4; ++i) {
            z[i] = 0;
            z[FACE / 4 + i] = 0x3f800000u;  // 1.0f
        }
    }
    // rows a .. b - 1 of the face at `face` from a source face (zeros or ones), if any
    auto fill = [&](uint32_t face, uint32_t a, uint32_t b, uint32_t src) {
        if (b > a) {
            noc_async_read(local(src), face + a * ROWB, (b - a) * ROWB);
        }
    };
    auto base = [](uint32_t cb) {
        const auto& f = get_local_cb_interface(cb);
        return f.fifo_limit - f.fifo_size;
    };
    // beta's column 0 and row 0 from LF (zeros) and CB_MZ's row 0, in both slots
    for (uint32_t t = 0; t < 2; ++t) {
        volatile tt_l1_ptr float* bp = reinterpret_cast<volatile tt_l1_ptr float*>(base(CB_BT) + t * tile);
        volatile tt_l1_ptr float* mp = reinterpret_cast<volatile tt_l1_ptr float*>(base(CB_MZ) + t * tile);
        for (uint32_t x = LF; x < 32; ++x) {
            bp[(x / 16) * 512 + (x % 16) * 16] = 0.0f;  // (x, 0)
            bp[(x / 16) * 256 + (x % 16)] = 0.0f;       // (0, x)
        }
        for (uint32_t x = 0; x < 32; ++x) {
            mp[(x / 16) * 256 + (x % 16)] = x < T ? 1.0f : 0.0f;
        }
    }

    // the keep codes; the constant tiles' zeros (they land with the first batch's rows, their ones after them)
    noc.async_read(s_keep, kb, keep_bytes, {.page_id = 0}, {.offset_bytes = 0});
    volatile tt_l1_ptr uint32_t* keep = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kb.get_write_ptr());
    cs.reserve_back(N_CONST);
    csm.reserve_back(N_CONST);
    const uint32_t ca = cs.get_write_ptr();
    for (uint32_t f = 0; f < 4 * N_CONST; ++f) {
        noc_async_read(local(f_zero), ca + f * FACE, FACE);
    }

    // the row of token k of pair p in q .. beta
    auto row_of = [](uint32_t p, uint32_t k) { return layout == 0 ? p + k * pairs : p * T + k; };

    uint32_t bi = 0, v_used = 0;  // the batches and v tiles so far (the CB slots' first uses)
    for (uint32_t u = start; u < start + count;) {
        const uint32_t p = u / N;
        const uint32_t j0 = u % N;
        const uint32_t nb = N - j0 < start + count - u ? N - j0 : start + count - u;

        kq.reserve_back(N);
        al.reserve_back(N);
        bt.reserve_back(1);
        mz.reserve_back(1);
        vv.reserve_back(nb);
        noc_async_writes_flushed();  // (the saved slot written from the staging and these slots has left L1)
        const uint32_t kqa = kq.get_write_ptr(), ala = al.get_write_ptr();
        const uint32_t va = vv.get_write_ptr();
        auto vtile = [&](uint32_t b) { return va + wrapped(CB_V, b, tile); };

        sc.reserve_back(nb * N);
        const uint32_t sa = sc.get_write_ptr();
        const uint32_t slot_bytes = V_OFF + nb * V_BLOCK;  // (the batch's part of its slot)
        {
            DeviceZoneScopedN("CR-ISSUE");
            // the last step's saved slot (whatever r) and S0 (the first batch's issued at the start), this step's
            // tokens at rows T .. LF - 1
            if (u != start) {
                read_dram(u, nb, sa);
            }
            if constexpr (layout == 1 && 16 % T == 0) {  // a pair's T rows (consecutive in a face) in one read a face
                const uint32_t row = p * T, rt = row / 32, rr = row % 32;
                const uint32_t lo = (rr / 16) * 2 * FACE + (rr % 16) * ROWB;
                for (uint32_t c = 0; c < N; ++c) {
                    const uint32_t page = rt * N + c;
                    const uint32_t kt = kqa + c * tile, at = ala + c * tile;
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_read(s_k.get_noc_addr(page, lo + f * FACE), kt + f * FACE + T * ROWB, T * ROWB);
                        noc_async_read(s_q.get_noc_addr(page, lo + f * FACE), kt + (2 + f) * FACE + T * ROWB, T * ROWB);
                        noc_async_read(s_decay.get_noc_addr(page, lo + f * FACE), at + f * FACE + T * ROWB, T * ROWB);
                    }
                }
                noc_async_read(s_beta.get_noc_addr(rt, lo), s_new, T * ROWB);
                for (uint32_t b = 0; b < nb; ++b) {
                    const uint32_t page = rt * N + j0 + b;
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_read(s_v.get_noc_addr(page, lo + f * FACE), vtile(b) + f * FACE + T * ROWB, T * ROWB);
                    }
                }
            } else {
                for (uint32_t k = 0; k < T; ++k) {
                    const uint32_t row = row_of(p, k), rt = row / 32, rr = row % 32;
                    const uint32_t lo = (rr / 16) * 2 * FACE + (rr % 16) * ROWB;  // its face row (columns 0-15)
                    const uint32_t pos = T + k;
                    for (uint32_t c = 0; c < N; ++c) {
                        const uint32_t page = rt * N + c;
                        const uint32_t kt = kqa + c * tile, at = ala + c * tile;
                        for (uint32_t f = 0; f < 2; ++f) {
                            noc_async_read(s_k.get_noc_addr(page, lo + f * FACE), kt + f * FACE + pos * ROWB, ROWB);
                            noc_async_read(s_q.get_noc_addr(page, lo + f * FACE), kt + (2 + f) * FACE + pos * ROWB, ROWB);
                            noc_async_read(s_decay.get_noc_addr(page, lo + f * FACE), at + f * FACE + pos * ROWB, ROWB);
                        }
                    }
                    noc_async_read(s_beta.get_noc_addr(rt, lo), s_new + k * ROWB, ROWB);
                    for (uint32_t b = 0; b < nb; ++b) {
                        const uint32_t page = rt * N + j0 + b;
                        for (uint32_t f = 0; f < 2; ++f) {
                            noc_async_read(s_v.get_noc_addr(page, lo + f * FACE), vtile(b) + f * FACE + pos * ROWB, ROWB);
                        }
                    }
                }
            }
            // a CB slot's first use: the rows no batch writes (behind the reads)
            if (bi < 2) {
                for (uint32_t c = 0; c < N; ++c) {
                    const uint32_t kt = kqa + c * tile, at = ala + c * tile;
                    noc_async_read(local(f_zero), kt, T * ROWB);
                    for (uint32_t f = 0; f < 3; ++f) {  // (rows LF .. 15 of face f and 0 .. T - 1 of face f + 1)
                        noc_async_read(local(f_zero), kt + f * FACE + LF * ROWB, (16 - T) * ROWB);
                    }
                    fill(kt + 3 * FACE, LF, 16, f_zero);
                    fill(at, LF, 16, f_one);
                    fill(at + FACE, LF, 16, f_one);
                }
            }
            for (uint32_t b = 0; b < nb; ++b) {
                if (v_used + b < 2 * N) {
                    const uint32_t vt = vtile(b);
                    noc_async_read(local(f_zero), vt, T * ROWB);
                    noc_async_read(local(f_zero), vt + LF * ROWB, (16 - T) * ROWB);
                    fill(vt + FACE, LF, 16, f_zero);
                    noc_async_read(local(f_zero), vt + 2 * FACE, FACE);  // (rows 16 .. 31: times Tb's zero columns)
                    noc_async_read(local(f_zero), vt + 3 * FACE, FACE);
                }
            }
        }
        {
            DeviceZoneScopedN("CR-BAR");
            noc_async_read_barrier();
        }
        if (u == start) {  // the constants' ones
            volatile tt_l1_ptr float* c0 = reinterpret_cast<volatile tt_l1_ptr float*>(ca);
            volatile tt_l1_ptr float* c1 = c0 + 1024;
            volatile tt_l1_ptr float* c2 = c1 + 1024;
            volatile tt_l1_ptr float* c3 = c2 + 1024;
            for (uint32_t i = 0; i < 16; ++i) {
                for (uint32_t j = 0; j < i; ++j) {
                    c0[i * 16 + j] = -1.0f;  // face 0: -strictly lower
                }
                for (uint32_t j = 0; j <= i; ++j) {
                    c1[512 + i * 16 + j] = 1.0f;  // face 2 (rows 16 ..): lower
                }
                c2[i * 17] = 1.0f;
                c2[768 + i * 17] = 1.0f;
                c3[768 + i * 17] = 1.0f;
            }
            cs.push_back(N_CONST);
            csm.push_back(N_CONST);
        }
        const uint32_t code = keep[p / heads];
        const uint32_t r = code < T - 1 ? code + 1 : T;
        const uint32_t z0 = T - r;  // the first redone row (rows before it do nothing)

        // the redone tokens' rows (positions z0 .. T - 1) from the staging; the rows before them: decays ones
        for (uint32_t c = 0; c < N; ++c) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t off = (c * 2 + f) * T * ROWB;
                noc_async_read(local(st_old + K_OFF + off), kqa + c * tile + f * FACE + z0 * ROWB, r * ROWB);
                noc_async_read(local(st_old + A_OFF + off), ala + c * tile + f * FACE + z0 * ROWB, r * ROWB);
                fill(ala + c * tile + f * FACE, 0, z0, f_one);
            }
        }
        for (uint32_t b = 0; b < nb; ++b) {
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_read(
                    local(st_old + V_OFF + b * V_BLOCK + f * T * ROWB), vtile(b) + f * FACE + z0 * ROWB, r * ROWB);
            }
        }
        // beta: column 0 (row x) and row 0 (column x) of rows 0 .. LF - 1 (0 on the rows doing nothing)
        volatile tt_l1_ptr float* bp = reinterpret_cast<volatile tt_l1_ptr float*>(bt.get_write_ptr());
        for (uint32_t x = 0; x < LF; ++x) {
            const float be = x < z0 ? 0.0f : x < T ? b_old[x - z0] : b_new[(x - T) * 16];
            bp[(x / 16) * 512 + (x % 16) * 16] = be;  // (x, 0)
            bp[(x / 16) * 256 + (x % 16)] = be;       // (0, x)
        }
        for (uint32_t k = 0; k < T; ++k) {
            b_stage[k] = b_new[k * 16];
        }
        noc_async_read_barrier();
        // the decays again at rows 16 .. 31 (faces 2, 3; rows LF .. 15 the ones)
        for (uint32_t c = 0; c < N; ++c) {
            noc_async_read(local(ala + c * tile), ala + c * tile + 2 * FACE, 2 * FACE);
        }
        noc_async_read_barrier();
        kq.push_back(N);
        al.push_back(N);
        bt.push_back(1);
        mz.push_back(1);
        push_wrapped(CB_V, nb);
        push_wrapped(CB_S, nb * N);

        // this step's tokens' rows (rows T .. LF - 1) gathered and saved for the next step's redo
        {
            DeviceZoneScopedN("CR-SAVE");
            for (uint32_t c = 0; c < N; ++c) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t off = (c * 2 + f) * T * ROWB;
                    noc_async_read(local(kqa + c * tile + f * FACE + T * ROWB), st_new + K_OFF + off, T * ROWB);
                    noc_async_read(local(ala + c * tile + f * FACE + T * ROWB), st_new + A_OFF + off, T * ROWB);
                }
            }
            for (uint32_t b = 0; b < nb; ++b) {
                // (the v tiles' addresses as reserved: va was the write pointer before the push)
                const auto& f = get_local_cb_interface(CB_V);
                const uint32_t a = va + b * tile < f.fifo_limit ? va + b * tile : va + b * tile - f.fifo_size;
                for (uint32_t ff = 0; ff < 2; ++ff) {
                    noc_async_read(local(a + ff * FACE + T * ROWB), st_new + V_OFF + b * V_BLOCK + ff * T * ROWB, T * ROWB);
                }
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i * PAGE < slot_bytes; ++i) {
                const uint32_t n_bytes = slot_bytes - i * PAGE < PAGE ? slot_bytes - i * PAGE : PAGE;
                noc_async_write(st_new + i * PAGE, s_saved.get_noc_addr(u * SLOT_PAGES + i), n_bytes);
            }
        }
        u += nb;
        ++bi;
        v_used += nb;
    }
    noc_async_write_barrier();
}
