// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per row of this core: the top k of the choice keys (sigmoid(logits) + bias; ties to the lower expert), the chosen
// scores' sum s (in expert order), and for each chosen expert e of the chip's base .. base + local - 1 the weight
// bf16((score_e * (1 / (s + 1e-20))) * scale) (route()'s order of operations, rounded to nearest even) in column
// e - base of the output row, the other columns zero.  With local 0, every chosen expert instead: its id (uint32) in
// columns 0 .. k - 1 and its weight in columns 16 .. 16 + k - 1, in increasing id order (the experts op's sliced
// layout, where each chip computes every chosen expert); with local above 16, likewise the chosen experts of base ..
// base + local - 1 only, as e - base (the column-sliced layout: a mesh column's chips share its experts), and when the
// call asks for it (`shared`) and the chip's local base page has 1 in element 1, then the shared expert as id local
// with weight 1.0 for every token (the experts op's slot after the column's experts).  The keys are compared as
// integers (the float's bits made monotonic); the few float operations are the RISC-V's software ones.
// A fp32 tile is 4 faces of 16 x 16 (1 KB each): row y's columns 0-15 are face (y / 16) 2, 16-31 the next face.  When
// this core's rows of a tile row are all in one half, the scores and keys of that half are in faces 0-1 (row y at
// y % 16).
namespace {
union Bits {
    uint32_t u;
    float f;
};

FORCE_INLINE float as_float(uint32_t u) {
    Bits b;
    b.u = u;
    return b.f;
}

FORCE_INLINE uint32_t as_bits(float f) {
    Bits b;
    b.f = f;
    return b.u;
}

FORCE_INLINE uint32_t round_bf16(uint32_t u) { return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u; }
}  // namespace

void kernel_main() {
    uint32_t a = 0;
    const uint32_t base_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t count = get_arg_val<uint32_t>(a++);
    const uint32_t tokens = get_arg_val<uint32_t>(a++);
    const uint32_t zeros_end = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_sc = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ch = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t et = get_compile_time_arg_val(3);
    // k
    constexpr uint32_t k_arg = get_compile_time_arg_val(4);
    constexpr uint32_t k = k_arg & 0xffu;
    // | 1 << 16: the k picks' global ids (+ 1) two to a word in id columns 12 .. 15, weights 0 (the progressive form's
    // budget: the experts op's builders count every mesh column's chosen experts; every reader stops at the row's first
    // zero weight, at most column 9)
    constexpr bool globals = ((k_arg >> 16) & 1u) != 0;
    constexpr uint32_t local = get_compile_time_arg_val(5);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(6);
    constexpr uint32_t base_page = get_compile_time_arg_val(7);
    constexpr bool with_shared = get_compile_time_arg_val(8) != 0;
    constexpr bool pack = get_compile_time_arg_val(9) != 0;  // the reader's packed rows: row j at word (j mod rpt) E
    constexpr uint32_t rpt = get_compile_time_arg_val(10);   // of tile j / rpt
    constexpr auto base_args = TensorAccessorArgs<11>();
    constexpr auto out_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();

    const auto s_base = TensorAccessor(base_args, base_addr);
    const auto s_out = TensorAccessor(out_args, out_addr);
    const uint32_t tile = get_local_cb_interface(cb_sc).fifo_page_size;
    constexpr uint32_t F = 1024, R = 64;  // fp32 face, face row

    Noc noc;
    CircularBuffer sc(cb_sc), ch(cb_ch), wcb(cb_w);
    // the scratch: 32 output rows (64 B each) at 0, a zero row at 2048, the local base page at 3072; with local 0,
    // the weights' 32 rows at 4096
    const uint32_t scr = wcb.get_write_ptr();
    const uint32_t zero_row = scr + 2048;
    const uint32_t base_l1 = scr + 3072;
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_row);
    for (uint32_t i = 0; i < 16; ++i) {
        z[i] = 0;
    }
    noc_async_read(s_base.get_noc_addr(0), base_l1, base_page);
    noc_async_read_barrier();
    const uint32_t base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_l1)[0];
    const bool shared = with_shared && local > 16 && reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_l1)[1] != 0;
    const float scale = as_float(scale_bits);

    auto row_addr = [&](uint32_t r, uint32_t half) {  // row r's columns 16 half .. 16 half + 15 in the output
        const uint32_t y = r % 32;
        return s_out.get_noc_addr(r / 32, (y / 16) * 2 * F + half * F + (y % 16) * R);
    };

    // row r's top k and outputs; its value of expert t 32 + 16 half + i at word i of key_at(t, half) / score_at
    auto do_row = [&](uint32_t r, auto&& key_at, auto&& score_at) {
        const uint32_t y = r % 32;
        uint32_t keys[k];
        uint32_t ids[k];
        for (uint32_t i = 0; i < k; ++i) {
            keys[i] = 0;
            ids[i] = 0;
        }
        uint32_t kmin = 0;
        for (uint32_t t = 0; t < et; ++t) {
            for (uint32_t half = 0; half < 2; ++half) {
                volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(key_at(t, half));
                for (uint32_t i = 0; i < 16; ++i) {
                    const uint32_t b = p[i];
                    const uint32_t key = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
                    if (key > kmin) {
                        uint32_t pos = k - 1;
                        while (pos > 0 && keys[pos - 1] < key) {
                            keys[pos] = keys[pos - 1];
                            ids[pos] = ids[pos - 1];
                            --pos;
                        }
                        keys[pos] = key;
                        ids[pos] = t * 32 + half * 16 + i;
                        kmin = keys[k - 1];
                    }
                }
            }
        }
        // the chosen experts in increasing order, so the sum runs as a reduction along the row would
        for (uint32_t i = 1; i < k; ++i) {
            const uint32_t v = ids[i];
            uint32_t j = i;
            while (j > 0 && ids[j - 1] > v) {
                ids[j] = ids[j - 1];
                --j;
            }
            ids[j] = v;
        }
        auto score = [&](uint32_t e) {
            const uint32_t ty = e / 32, x = e % 32;
            return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(score_at(ty, x / 16))[x % 16];
        };
        float sum = 0.0f;
        for (uint32_t i = 0; i < k; ++i) {
            sum += as_float(score(ids[i]));
        }
        const float inv = 1.0f / (sum + 1e-20f);
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + y * R);
        for (uint32_t j = 0; j < 16; ++j) {
            o[j] = 0;
        }
        if constexpr (local == 0 || local > 16) {
            volatile tt_l1_ptr uint32_t* ow =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 4096 + y * R);
            for (uint32_t j = 0; j < 16; ++j) {
                ow[j] = 0;
            }
            uint32_t j = 0;
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t e = ids[i];
                if (local == 0 || (e >= base && e < base + local)) {
                    o[j] = local == 0 ? e : e - base;
                    ow[j++] = round_bf16(as_bits((as_float(score(e)) * inv) * scale));
                }
            }
            if (shared) {
                o[j] = local;
                ow[j] = 0x3f800000u;  // 1.0
            }
            if constexpr (globals) {
                static_assert(k <= 8, "the global picks: 8 at most in columns 12 .. 15");
                for (uint32_t i = 0; i < k; ++i) {
                    o[12 + i / 2] = o[12 + i / 2] | ((ids[i] + 1) << (16 * (i % 2)));
                }
            }
            noc_async_write(scr + y * R, row_addr(r, 0), R);
            noc_async_write(scr + 4096 + y * R, row_addr(r, 1), R);
        } else {
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t e = ids[i];
                if (e >= base && e < base + local) {
                    o[e - base] = round_bf16(as_bits((as_float(score(e)) * inv) * scale));
                }
            }
            noc_async_write(scr + y * R, row_addr(r, 0), R);
            noc_async_write(zero_row, row_addr(r, 1), R);
        }
    };

    if (count > 0 && pack) {
        constexpr uint32_t E = et * 32;
        const uint32_t np = (count + rpt - 1) / rpt;
        sc.wait_front(np);
        ch.wait_front(np);
        const uint32_t sc_base = sc.get_read_ptr();
        const uint32_t ch_base = ch.get_read_ptr();
        for (uint32_t j = 0; j < count; ++j) {
            const uint32_t row = (j / rpt) * tile + (j % rpt) * E * 4;
            do_row(
                start + j,
                [&](uint32_t t, uint32_t half) { return ch_base + row + (32 * t + 16 * half) * 4; },
                [&](uint32_t t, uint32_t half) { return sc_base + row + (32 * t + 16 * half) * 4; });
            if ((start + j) % 32 == 31) {
                noc_async_write_barrier();  // the scratch rows (by row mod 32) are reused
            }
        }
        noc_async_write_barrier();
        sc.pop_front(np);
        ch.pop_front(np);
    } else if (count > 0) {
        for (uint32_t tr = start / 32; tr <= (start + count - 1) / 32; ++tr) {
            sc.wait_front(et);
            ch.wait_front(et);
            const uint32_t sc_base = sc.get_read_ptr();
            const uint32_t ch_base = ch.get_read_ptr();
            const uint32_t r0 = tr * 32 > start ? tr * 32 : start;
            const uint32_t r1 = (tr + 1) * 32 < start + count ? (tr + 1) * 32 : start + count;
            const bool half = (r0 % 32) / 16 == ((r1 - 1) % 32) / 16;
            for (uint32_t r = r0; r < r1; ++r) {
                const uint32_t off = (half ? 0 : ((r % 32) / 16) * 2 * F) + (r % 16) * R;
                do_row(
                    r,
                    [&](uint32_t t, uint32_t h) { return ch_base + t * tile + h * F + off; },
                    [&](uint32_t t, uint32_t h) { return sc_base + t * tile + h * F + off; });
            }
            noc_async_write_barrier();  // the scratch rows are reused by the next tile row
            sc.pop_front(et);
            ch.pop_front(et);
        }
    }
    for (uint32_t r = tokens; r < zeros_end; ++r) {
        noc_async_write(zero_row, row_addr(r, 0), R);
        noc_async_write(zero_row, row_addr(r, 1), R);
    }
    noc_async_write_barrier();
}
