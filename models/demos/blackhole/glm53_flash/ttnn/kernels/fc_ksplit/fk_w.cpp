// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

#ifndef AHEAD
#define AHEAD 3  // weight blocks in flight (CB_W holds AHEAD + 1)
#endif

// The K-split decode matmul's weights, partials and output (ttnn/fc_ksplit.py; RISCV_0, NoC 0): this core's blocks of
// its column tile n of w (block b: K tiles KB b .. KB b + KB - 1, page k nt + n) into CB_W (AHEAD in flight; issued
// only when CB_W has room, never blocking, so that the core can serve the partials meanwhile), and
//   - a helper h (slice h > 0): each B_b its compute packed into CB_SEND written into the owner's CB_RECV0 + h at the
//     partial's own place (the owner's ring holds all of helper h's partials: no credits), then, after the write's
//     barrier, its count of partials into the owner's SEM_RDY0 + h (a plain 4-byte write);
//   - the owner (slice 0): in the sums' order, each other slice's partial pushed to the compute from its ring once the
//     helper's count shows it, and at the end the MT output tiles written to DRAM.
// The CBs alike on every core of the program (the owner's ring addresses are a helper's own).  Counts by plain writes
// of absolute values from 16-byte words of CB_WORDS (a word a count); each core's own semaphores back to 0 at its end.
void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t col = get_arg_val<uint32_t>(2);    // n
    const uint32_t slice = get_arg_val<uint32_t>(3);  // 0: the owner
    const uint32_t own_x = get_arg_val<uint32_t>(4);  // a helper: the owner's NoC 0 x, y
    const uint32_t own_y = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_send = get_compile_time_arg_val(2);
    constexpr uint32_t cb_recv0 = get_compile_time_arg_val(3);  // helper h's ring at the owner: cb_recv0 + h
    constexpr uint32_t cb_words = get_compile_time_arg_val(4);
    constexpr uint32_t sem_rdy0 = get_compile_time_arg_val(5);  // the owner's: sem_rdy0 + h, helper h's count
    constexpr uint32_t KB = get_compile_time_arg_val(6);
    constexpr uint32_t NB = get_compile_time_arg_val(7);
    constexpr uint32_t MT = get_compile_time_arg_val(8);
    constexpr uint32_t S = get_compile_time_arg_val(9);
    constexpr uint32_t nt = get_compile_time_arg_val(10);  // w's (and the output's) column tiles
    constexpr auto w_args = TensorAccessorArgs<11>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const auto s_w = TensorAccessor(w_args, w_addr);
    const uint32_t wt = get_local_cb_interface(cb_w).fifo_page_size;
    noc_async_read_set_trid(0, noc_index);  // (transaction ids from a clean state, as rm_w.cpp)
    noc_async_write_set_trid(0, noc_index);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
    DeviceZoneScopedN("FK-W");
    // this core's blocks: b < NB - 1 with PAT[b % L] = slice (FK_PAT: L in bits 0 .. 3, then 2 bits an entry), and the
    // owner NB - 1 too
    auto slice_of = [](uint32_t b) { return (FK_PAT >> (4 + 2 * (b % (FK_PAT & 15)))) & 3u; };
    uint8_t blocks[NB];
    uint32_t nown = 0;
    for (uint32_t b = 0; b + 1 < NB; ++b) {
        if (slice_of(b) == slice) {
            blocks[nown++] = b;
        }
    }
    if (slice == 0) {
        blocks[nown++] = NB - 1;
    }
    auto own_block = [&](uint32_t j) { return static_cast<uint32_t>(blocks[j]); };
    constexpr uint32_t depth = AHEAD + 1;
    const uint32_t wbase = get_write_ptr(cb_w);
    uint32_t issued = 0, pushed = 0;
    auto issue = [&]() {  // the next block's reads, when CB_W has room for it (true: issued)
        if (issued >= nown || issued - pushed >= AHEAD ||
            !cb_pages_reservable_at_back(cb_w, (issued - pushed + 1) * KB)) {
            return false;
        }
        const uint32_t b = own_block(issued);
        const uint32_t dst = wbase + (issued % depth) * KB * wt;
        noc_async_read_set_trid(issued % depth + 1);
        for (uint32_t i = 0; i < KB; ++i) {
            noc_async_read(s_w.get_noc_addr((b * KB + i) * nt + col), dst + i * wt, wt);
        }
        ++issued;
        return true;
    };
    auto push = [&]() {  // the next issued block to the compute
        noc_async_read_barrier_with_trid(pushed % depth + 1);
        cb_reserve_back(cb_w, KB);
        cb_push_back(cb_w, KB);
        ++pushed;
    };
    if (slice != 0) {
        const uint32_t words = get_write_ptr(cb_words);
        const uint32_t pbytes = MT * get_local_cb_interface(cb_send).fifo_page_size;  // a partial
        const uint32_t ring = get_write_ptr(cb_recv0 + slice);  // (the owner's ring for this helper: the same address)
        const uint64_t owner_rdy = get_noc_addr(own_x, own_y, get_semaphore(sem_rdy0 + slice));
        uint32_t sent = 0;
        while (pushed < nown || sent < nown) {
            while (issue()) {
            }
            if (pushed < issued) {
                push();
            }
            if (sent < nown && cb_pages_available_at_front(cb_send, MT)) {
                noc_async_write(get_read_ptr(cb_send), get_noc_addr(own_x, own_y, ring + sent * pbytes), pbytes);
                noc_async_write_barrier();
                volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16 * sent);
                *w = sent + 1;
                noc_semaphore_set_remote(words + 16 * sent, owner_rdy);
                cb_pop_front(cb_send, MT);
                ++sent;
            }
        }
        noc_async_write_barrier();
    } else {
        const auto s_out = TensorAccessor(out_args, out_addr);
        uint32_t taken[S] = {0};
        for (uint32_t b = 0; b < NB; ++b) {
            while (issue()) {
            }
            const bool own = b == NB - 1 || slice_of(b) == 0;
            if (own) {
                while (pushed == issued) {  // (its reads not issued yet: CB_W full until the compute takes a block)
                    issue();
                }
                push();
                continue;
            }
            const uint32_t h = slice_of(b);
            volatile tt_l1_ptr uint32_t* rdy =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rdy0 + h));
            while (true) {  // (the weights topped up while the helper's partial is not there)
                invalidate_l1_cache();
                if (*rdy >= taken[h] + 1) {
                    break;
                }
                issue();
            }
            cb_reserve_back(cb_recv0 + h, MT);  // (the helper wrote the ring's next place: the rings in step)
            cb_push_back(cb_recv0 + h, MT);
            ++taken[h];
        }
        cb_wait_front(cb_out, MT);
        const uint32_t ot = get_local_cb_interface(cb_out).fifo_page_size;
        for (uint32_t r = 0; r < MT; ++r) {
            noc_async_write(get_read_ptr(cb_out) + r * ot, s_out.get_noc_addr(r * nt + col), ot);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, MT);
        for (uint32_t h = 1; h < S; ++h) {  // (every helper's last count came before the waits above)
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rdy0 + h)), 0);
        }
    }
    noc_async_read_set_trid(0);
}
