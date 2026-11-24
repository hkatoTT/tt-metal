// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"
#include "moe_decode_experts_common.hpp"

// A builder core (b: runtime arg 1), on BRISC / NoC 1, for the packed units from many tile rows of tokens: id tile b
// of moe_decode_route's output (the column's picks of tokens 32 b .. 32 b + 31) read and scanned once into record b
// (moe_decode_experts_common.hpp's rec_*), written to builder 0 at the same place (its count sem_go raised), and
// builder 0, once all the records are there and every gate | up and down core has said it is in the program (their
// BRISCs raise its sem_rdy first thing), multicasts the records to all of them at the same place of their scratch and
// sets their sem_rec to `rows`.  (Each builder multicasting its own record, 4 multicasts each into the same two
// rectangles, queued up: the records were all in only 24 / 37 us in at 256 / 512 tokens.)
//
// The progressive form (MOE_PROG defined by the program factory, a quality-changing option): each builder also sums its
// tile's weights by expert into its sums slot (after the id tile in its scratch), in fixed point (fixed_weight:
// integers, exact in any order: the RISC-V's soft float made builder 0's sums and choice ~40 us of the op at 512
// tokens), with its tile's chosen experts of every mesh column, and writes them to builder 0's slot b with its
// record; builder 0, once all have come, adds them up in record order and marks the experts read with their correction
// (below).
//
// Row weights (MOE_PROG_ROWS, GLM53_MOE_PROG_ROWS): a verify step's token j U + u of mesh row
// r (column token r MOE_PROG_ROWS_TROW + j U + u, U = MOE_PROG_ROWS_U users a row) counts toward the sums with its
// row j's weight q_j / 256 (MOE_PROG_ROWS_Q: how often row j's output is used, P(a >= j)), so that the corrections go
// to the rows whose outputs are kept rather than to drafts past the first miss.  Integers still: (fixed_weight(w) q_j)
// >> 8, the product in 64 bits.
// (MOE_PROG_FAST2: the scan's parts as zones of their own under the device profiler; nothing without it)
#ifdef MOE_PROG_FAST2
#define RB_PART(name) DeviceZoneScopedN(name)
#else
#define RB_PART(name)
#endif

void kernel_main() {
    const uint32_t routing_addr = get_arg_val<uint32_t>(0);
    const uint32_t b = get_arg_val<uint32_t>(1);
    constexpr uint32_t local = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);
    constexpr uint32_t sem_rdy = get_compile_time_arg_val(3);
    constexpr uint32_t sem_go = get_compile_time_arg_val(4);
    constexpr uint32_t sem_rec = get_compile_time_arg_val(5);
    constexpr uint32_t consumers = get_compile_time_arg_val(6);
    constexpr uint32_t gu_x0 = get_compile_time_arg_val(7);  // the gate | up cores' rectangle (NoC 1: far corner first)
    constexpr uint32_t gu_y0 = get_compile_time_arg_val(8);
    constexpr uint32_t gu_x1 = get_compile_time_arg_val(9);
    constexpr uint32_t gu_y1 = get_compile_time_arg_val(10);
    constexpr uint32_t dn_x0 = get_compile_time_arg_val(11);  // the down cores'
    constexpr uint32_t dn_y0 = get_compile_time_arg_val(12);
    constexpr uint32_t dn_x1 = get_compile_time_arg_val(13);
    constexpr uint32_t dn_y1 = get_compile_time_arg_val(14);
    constexpr uint32_t num_gu = get_compile_time_arg_val(15);
    constexpr uint32_t num_dn = get_compile_time_arg_val(16);
    // the op's coordinator for NOC1 weight reads (gu_ctrl.cpp): once its multicasts are done and each grid column's top
    // core has raised sem_xdone (its x slice's multicast done: `tops` of them), builder 0 raises every gate | up and
    // down core's sem_go1 (runtime args 4 ..: their NoC coordinates)
    constexpr uint32_t sem_xdone = get_compile_time_arg_val(17);
    constexpr uint32_t tops = get_compile_time_arg_val(18);
    constexpr uint32_t sem_go1 = get_compile_time_arg_val(19);
    // the in-op gather (h1_send.cpp): the routing tile comes over the fabric, piece by piece, each raising the global
    // semaphore at L1 address h1_sem here (0: off) once it has landed; h1_count of them, then the semaphore reset to 0
    constexpr uint32_t h1_sem = get_compile_time_arg_val(20);
    constexpr uint32_t h1_count = get_compile_time_arg_val(21);
    constexpr auto r_args = TensorAccessorArgs<22>();
    const auto s_r = TensorAccessor(r_args, routing_addr);

    CircularBuffer scr(cb_scr);
    const uint32_t rec = scr.get_write_ptr() + b * rec_bytes(local);
    const uint32_t tile = scr.get_write_ptr() + rec_brisc(rows, local);
#ifdef MOE_PROG
    const uint32_t sums_l1 = tile + 4096;  // builder b's sums at slot b (4 local bytes each)
    // ... and after them the tile's chosen experts of every mesh column (the routing's global picks in id columns 12 ..
    // 15, moe_route / moe_decode_route with the progressive form): 3 words a column (72 bits)
    constexpr uint32_t SLOT_WORDS = local + 12;
#endif
    uint32_t n = 0;
    {
        DeviceZoneScopedN("RB-SCAN");
        if constexpr (h1_sem != 0) {
            DeviceZoneScopedN("RB-H1");
            volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h1_sem);
            noc_semaphore_wait_min(landed, h1_count);
            noc_semaphore_set(landed, 0);
        }
        noc_async_read(s_r.get_noc_addr(b), tile, 4096);
        noc_async_read_barrier();
#ifdef MOE_PROG_FAST2
        // (MOE_PROG_FAST2: the tile read as plain words once it is in; the records below still written as volatile)
        invalidate_l1_cache();
        asm volatile("" ::: "memory");
        const uint32_t* p = reinterpret_cast<const uint32_t*>(tile);
#else
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
#endif
        volatile tt_l1_ptr uint8_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(rec);
        volatile tt_l1_ptr uint16_t* off = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rec + rec_off(local));
        volatile tt_l1_ptr uint32_t* picks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec + rec_hdr(local));
        uint16_t pos[local];  // the counts, then the next place of each expert's picks (on the stack, not in L1)
        {
        RB_PART("RB-CNT");
        for (uint32_t e = 0; e < local; ++e) {
            pos[e] = 0;
        }
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t row = (r / 16) * 2 * 256 + (r % 16) * 16;  // columns 0-15; 16-31 a face (256) further
            for (uint32_t c = 0; c < 16; ++c) {
                if (static_cast<int32_t>(p[row + 256 + c]) <= 0) {  // the row's last pick (they start at column 0)
                    break;
                }
                ++pos[p[row + c]];
            }
        }
        for (uint32_t e = 0; e < local; ++e) {
            const uint32_t c = pos[e];
            cnt[e] = c;
            off[e] = n;
            pos[e] = n;
            n += c;
        }
        }
        for (uint32_t e = local; e < rec_off(local); ++e) {  // (the last word's other lanes: summed four at a time)
            cnt[e] = 0;
        }
#ifdef MOE_PROG
        uint32_t sums[local];
        for (uint32_t e = 0; e < local; ++e) {
            sums[e] = 0;
        }
#endif
        {
        RB_PART("RB-PICK");
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t row = (r / 16) * 2 * 256 + (r % 16) * 16;
            for (uint32_t c = 0; c < 16; ++c) {
                const uint32_t w = p[row + 256 + c];
                if (static_cast<int32_t>(w) <= 0) {
                    break;
                }
                const uint32_t e = p[row + c];
                picks[pos[e]++] = (w & 0xffff0000u) | (32 * b + r);
#ifdef MOE_PROG
#ifdef MOE_PROG_ROWS
                {
                    constexpr uint32_t q[] = {MOE_PROG_ROWS_Q};
                    const uint32_t j = ((32 * b + r) % MOE_PROG_ROWS_TROW) / MOE_PROG_ROWS_U;
                    sums[e] += static_cast<uint32_t>((static_cast<uint64_t>(fixed_weight(w)) * q[j]) >> 8);
                }
#else
                sums[e] += fixed_weight(w);
#endif
#endif
            }
        }
        }
#ifdef MOE_PROG
        {
        RB_PART("RB-SLOT");  // (to the scan's end: the sums' slot and the columns' picks; a scope of its own, as
                             // RB-SCAN's zone is in the one around it)
        volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sums_l1 + b * 4 * SLOT_WORDS);
        for (uint32_t e = 0; e < local; ++e) {
            slot[e] = sums[e];
        }
#ifdef MOE_PROG_FAST2
        {
            // (MOE_PROG_FAST2: the 4 columns' picks as one 288-bit set, global id - 1 its bit: column c's bits 72 c ..
            // 72 c + 71, counted by builder 0 as the 3-word sets below; 9 of the 12 words sent)
            uint32_t bm[9];
            for (uint32_t w = 0; w < 9; ++w) {
                bm[w] = 0;
            }
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t row = (r / 16) * 2 * 256 + (r % 16) * 16;
                for (uint32_t c = 12; c < 16; ++c) {
                    const uint32_t v = p[row + c];
                    const uint32_t g0 = (v & 0xffffu) - 1u, g1 = (v >> 16) - 1u;  // (0: none, wraps past 287)
                    if (g0 < 4 * 72) {
                        bm[g0 >> 5] |= 1u << (g0 & 31u);
                    }
                    if (g1 < 4 * 72) {
                        bm[g1 >> 5] |= 1u << (g1 & 31u);
                    }
                }
            }
            for (uint32_t w = 0; w < 9; ++w) {
                slot[local + w] = bm[w];
            }
        }
#else
        {
            uint32_t bm[12];
            for (uint32_t w = 0; w < 12; ++w) {
                bm[w] = 0;
            }
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t row = (r / 16) * 2 * 256 + (r % 16) * 16;
                for (uint32_t c = 12; c < 16; ++c) {
                    const uint32_t v = p[row + c];
                    for (uint32_t h = 0; h < 2; ++h) {
                        const uint32_t g = (v >> (16 * h)) & 0xffffu;  // a global id + 1 (0: none)
                        if (g != 0 && g <= 4 * 72) {
                            const uint32_t col = (g - 1) / 72, bit = (g - 1) % 72;
                            bm[3 * col + bit / 32] |= 1u << (bit % 32);
                        }
                    }
                }
            }
            for (uint32_t w = 0; w < 12; ++w) {
                slot[local + w] = bm[w];
            }
        }
#endif
        }
#endif
    }
    const uint32_t bytes = (rec_hdr(local) + 4 * n + 15) / 16 * 16;
    if (b != 0) {  // to builder 0 (runtime args 2, 3)
        const uint32_t x0 = get_arg_val<uint32_t>(2), y0 = get_arg_val<uint32_t>(3);
        noc_async_write(rec, get_noc_addr(x0, y0, rec), bytes);
#ifdef MOE_PROG
        const uint32_t slot = sums_l1 + b * 4 * SLOT_WORDS;
        noc_async_write(slot, get_noc_addr(x0, y0, slot), 4 * SLOT_WORDS);
#endif
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(x0, y0, get_semaphore(sem_go)), 1);
        noc_async_atomic_barrier();
        return;
    }
    {
        DeviceZoneScopedN("RB-GO");  // the other records, and every gate | up and down core in the program
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go)), rows - 1);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rdy)), consumers);
    }
#ifdef MOE_PROG
    {
        // the progressive form: the column's chosen experts (D) read with their correction, the largest weight sums
        // first (equal sums: the lower expert), as many as the budget takes: the column's reads at most
        // MOE_PROG_F_MILLI / 1000 of D's BFP8 bytes, every chosen expert's base (BFP4) first: m = floor(|D| (F 1088 -
        // 576) / 576) (BFP8 / BFP4 tiles 1088 / 576 bytes), all of D from F = 1.06 on.  Two more forms (value-changing,
        // used only when chosen): MOE_PROG_F_MILLI >= 10000 (GLM53_MOE_PROG_F=10 + F): F's budget as above, then the
        // in-op shared expert's correction marked as well (slot 72 of 73, its one past the budget; at F = 0.5 the
        // busiest column's budget is m = 0 and the shared expert, in every column, went without it); == 9000
        // (GLM53_MOE_PROG_F=9, a comparison): the shared expert's correction only.  Below 9000 nothing here changes
        DeviceZoneScopedN("RB-PROG");
        uint32_t total[local];
#ifdef MOE_PROG_FAST
        // (MOE_PROG_FAST, the program factory's GLM53_MOE_PROG_FAST, on by default: the same sums and chosen set,
        // the rows' words read as plain words once the records are in, and the chosen experts by OR of the records'
        // counts, four a word (only count != 0 is used; the lanes past local are zero, rec_off))
        invalidate_l1_cache();
        asm volatile("" ::: "memory");
        constexpr uint32_t CW = rec_off(local) / 4;
        uint32_t any[CW];
#ifdef MOE_PROG_FAST2
        // (MOE_PROG_FAST2: the counts' OR and the columns' 288-bit sets now, the rows' sums only for a budget that
        // marks some of D and not all, 0 < m < |D|, and only the chosen experts' sums: below)
        uint32_t picks9[9];
        {
            const uint32_t* s0 = reinterpret_cast<const uint32_t*>(sums_l1);
            const uint32_t* c0 = reinterpret_cast<const uint32_t*>(scr.get_write_ptr());
            for (uint32_t w = 0; w < CW; ++w) {
                any[w] = c0[w];
            }
            for (uint32_t w = 0; w < 9; ++w) {
                picks9[w] = s0[local + w];
            }
        }
        for (uint32_t r = 1; r < rows; ++r) {
            const uint32_t* s = reinterpret_cast<const uint32_t*>(sums_l1 + r * 4 * SLOT_WORDS);
            const uint32_t* c = reinterpret_cast<const uint32_t*>(scr.get_write_ptr() + r * rec_bytes(local));
            for (uint32_t w = 0; w < CW; ++w) {
                any[w] |= c[w];
            }
            for (uint32_t w = 0; w < 9; ++w) {
                picks9[w] |= s[local + w];
            }
        }
#else
        {
            const uint32_t* s0 = reinterpret_cast<const uint32_t*>(sums_l1);
            const uint32_t* c0 = reinterpret_cast<const uint32_t*>(scr.get_write_ptr());
            for (uint32_t e = 0; e < local; ++e) {
                total[e] = s0[e];
            }
            for (uint32_t w = 0; w < CW; ++w) {
                any[w] = c0[w];
            }
        }
        for (uint32_t r = 1; r < rows; ++r) {
            const uint32_t* s = reinterpret_cast<const uint32_t*>(sums_l1 + r * 4 * SLOT_WORDS);
            const uint32_t* c = reinterpret_cast<const uint32_t*>(scr.get_write_ptr() + r * rec_bytes(local));
            for (uint32_t e = 0; e < local; ++e) {
                total[e] += s[e];
            }
            for (uint32_t w = 0; w < CW; ++w) {
                any[w] |= c[w];
            }
        }
#endif
        auto is_chosen = [&](uint32_t e) { return ((any[e / 4] >> (8 * (e % 4))) & 0xffu) != 0; };
        uint32_t chosen = 0;
        for (uint32_t e = 0; e < local; ++e) {
            chosen += is_chosen(e) ? 1 : 0;
        }
#define MOE_PROG_CHOSEN(e) is_chosen(e)
#else
        uint16_t count[local];
        for (uint32_t e = 0; e < local; ++e) {
            total[e] = 0;
            count[e] = 0;
        }
        for (uint32_t r = 0; r < rows; ++r) {
            volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sums_l1 + r * 4 * SLOT_WORDS);
            volatile tt_l1_ptr uint8_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(
                scr.get_write_ptr() + r * rec_bytes(local));
            for (uint32_t e = 0; e < local; ++e) {
                total[e] += slot[e];
                count[e] += cnt[e];
            }
        }
        uint32_t chosen = 0;
        for (uint32_t e = 0; e < local; ++e) {
            chosen += count[e] != 0 ? 1 : 0;
        }
#define MOE_PROG_CHOSEN(e) (count[e] != 0)
#endif
        // the busiest mesh column's chosen experts (the routing's global picks, OR'ed over the records; none: this
        // column's own): the budget F of its BFP8 bytes, read by every column alike (the step waits for the busiest
        // one), m = floor((F 1088 busiest - 576 |D|) / 576)
        uint32_t busiest = 0;
#ifdef MOE_PROG_FAST2
        for (uint32_t col = 0; col < 4; ++col) {  // bits 72 col .. 72 col + 71 of the set
            uint32_t d = 0;
            for (uint32_t i = 72 * col; i < 72 * col + 72;) {
                const uint32_t sh = i & 31u, take = (32u - sh) < (72 * col + 72 - i) ? (32u - sh) : (72 * col + 72 - i);
                uint32_t v = picks9[i >> 5] >> sh;
                if (take < 32u) {
                    v &= (1u << take) - 1u;
                }
                d += static_cast<uint32_t>(__builtin_popcount(v));
                i += take;
            }
            busiest = d > busiest ? d : busiest;
        }
#else
        {
            uint32_t bm[12];
            for (uint32_t w = 0; w < 12; ++w) {
                bm[w] = 0;
            }
            for (uint32_t r = 0; r < rows; ++r) {
                volatile tt_l1_ptr uint32_t* slot =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sums_l1 + r * 4 * SLOT_WORDS);
                for (uint32_t w = 0; w < 12; ++w) {
                    bm[w] |= slot[local + w];
                }
            }
            for (uint32_t col = 0; col < 4; ++col) {
                const uint32_t d = static_cast<uint32_t>(__builtin_popcount(bm[3 * col]) +
                                                         __builtin_popcount(bm[3 * col + 1]) +
                                                         __builtin_popcount(bm[3 * col + 2]));
                busiest = d > busiest ? d : busiest;
            }
        }
#endif
        busiest = busiest > chosen ? busiest : chosen;
#if MOE_PROG_F_MILLI >= 10000
        constexpr uint32_t f_milli = MOE_PROG_F_MILLI - 10000;  // (F, plus the shared expert's correction: below)
#else
        constexpr uint32_t f_milli = MOE_PROG_F_MILLI;
#endif
        const uint32_t num = f_milli * 1088u * busiest, den = 576000u * chosen;
        uint32_t m = num <= den ? 0u : (num - den) / 576000u;
        m = m < chosen ? m : chosen;
        uint32_t mask[mask_words(local)];
#ifdef MOE_PROG_FAST
        // (MOE_PROG_FAST: the same m marks, the order of the scans below: the larger sum first, of equal sums the lower
        // expert, a strict order, found by a quickselect of the chosen experts' ids in place of m scans of the 73)
        for (uint32_t w = 0; w < mask_words(local); ++w) {
            mask[w] = 0;
        }
        if (m >= chosen) {
            for (uint32_t e = 0; e < local; ++e) {
                if (is_chosen(e)) {
                    mask[e / 32] |= 1u << (e % 32);
                }
            }
        } else if (m > 0) {
            uint8_t ids[local];
            int32_t n = 0;
            for (uint32_t e = 0; e < local; ++e) {
                if (is_chosen(e)) {
                    ids[n++] = static_cast<uint8_t>(e);
                }
            }
#ifdef MOE_PROG_FAST2
            for (int32_t i = 0; i < n; ++i) {  // (the same totals: the rows' sums, in any order, mod 2^32)
                const uint32_t* s = reinterpret_cast<const uint32_t*>(sums_l1) + ids[i];
                uint32_t t = 0;
                for (uint32_t r = 0; r < rows; ++r) {
                    t += s[r * SLOT_WORDS];
                }
                total[ids[i]] = t;
            }
#endif
            auto before = [&](uint32_t a, uint32_t c) {  // a ahead of c
                return total[a] > total[c] || (total[a] == total[c] && a < c);
            };
            const int32_t k = static_cast<int32_t>(m) - 1;
            int32_t lo = 0, hi = n - 1;
            while (lo < hi) {  // ids[0 .. k] the m first once done
                const int32_t mid = lo + (hi - lo) / 2;
                uint8_t t = ids[mid];
                ids[mid] = ids[hi];
                ids[hi] = t;
                const uint32_t piv = ids[hi];
                int32_t st = lo;
                for (int32_t i = lo; i < hi; ++i) {
                    if (before(ids[i], piv)) {
                        t = ids[i];
                        ids[i] = ids[st];
                        ids[st] = t;
                        ++st;
                    }
                }
                ids[hi] = ids[st];
                ids[st] = static_cast<uint8_t>(piv);
                if (st == k) {
                    break;
                }
                if (st < k) {
                    lo = st + 1;
                } else {
                    hi = st - 1;
                }
            }
            for (int32_t d = 0; d <= k; ++d) {
                mask[ids[d] / 32] |= 1u << (ids[d] % 32);
            }
        }
#else
        if (2 * m <= chosen) {  // the m largest marked
            for (uint32_t w = 0; w < mask_words(local); ++w) {
                mask[w] = 0;
            }
            for (uint32_t d = 0; d < m; ++d) {
                uint32_t best = local;
                for (uint32_t e = 0; e < local; ++e) {
                    if (count[e] != 0 && !((mask[e / 32] >> (e % 32)) & 1u) &&
                        (best == local || total[e] > total[best])) {
                        best = e;
                    }
                }
                mask[best / 32] |= 1u << (best % 32);
            }
        } else {  // every chosen one marked, then the chosen - m smallest unmarked (equal: the higher expert)
            for (uint32_t w = 0; w < mask_words(local); ++w) {
                mask[w] = 0;
            }
            for (uint32_t e = 0; e < local; ++e) {
                if (count[e] != 0) {
                    mask[e / 32] |= 1u << (e % 32);
                }
            }
            for (uint32_t d = m; d < chosen; ++d) {
                uint32_t best = local;
                for (uint32_t e = 0; e < local; ++e) {
                    if (((mask[e / 32] >> (e % 32)) & 1u) && (best == local || total[e] <= total[best])) {
                        best = e;
                    }
                }
                mask[best / 32] &= ~(1u << (best % 32));
            }
        }
#endif
#if MOE_PROG_F_MILLI == 9000
        // (GLM53_MOE_PROG_F=9: the in-op shared expert's correction only, slot 72 of the 73: no routed expert's)
        for (uint32_t w = 0; w < mask_words(local); ++w) {
            mask[w] = 0;
        }
#endif
#if MOE_PROG_F_MILLI == 9000 || MOE_PROG_F_MILLI >= 10000
        // (GLM53_MOE_PROG_F=9 and 10 + F: the shared expert's correction always, past the budget by its one; without
        // the shared expert in the op, local 72, nothing more)
        if (local == 73 && MOE_PROG_CHOSEN(local - 1)) {
            mask[(local - 1) / 32] |= 1u << ((local - 1) % 32);
        }
#endif
        volatile tt_l1_ptr uint32_t* out =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr.get_write_ptr() + prog_mask_off(local));
        for (uint32_t w = 0; w < mask_words(local); ++w) {
            out[w] = mask[w];
        }
    }
#endif
    const uint32_t base = scr.get_write_ptr(), all = rows * rec_bytes(local);
    noc_async_write_multicast(base, get_noc_multicast_addr(gu_x0, gu_y0, gu_x1, gu_y1, base), all, num_gu);
    noc_async_write_multicast(base, get_noc_multicast_addr(dn_x0, dn_y0, dn_x1, dn_y1, base), all, num_dn);
    noc_async_write_barrier();
    volatile tt_l1_ptr uint32_t* value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);  // (the tile is read)
    *value = rows;
    const uint32_t sem = get_semaphore(sem_rec);
    noc_semaphore_set_multicast(tile, get_noc_multicast_addr(gu_x0, gu_y0, gu_x1, gu_y1, sem), num_gu);
    noc_semaphore_set_multicast(tile, get_noc_multicast_addr(dn_x0, dn_y0, dn_x1, dn_y1, sem), num_dn);
    noc_async_write_barrier();
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xdone)), tops);
    const uint32_t go = get_semaphore(sem_go1);
    for (uint32_t c = 0; c < consumers; ++c) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(4 + 2 * c), get_arg_val<uint32_t>(5 + 2 * c), go), 1);
    }
    noc_async_atomic_barrier();
}
