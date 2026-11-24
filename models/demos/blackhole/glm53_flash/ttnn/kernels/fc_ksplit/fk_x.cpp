// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

// The K-split decode matmul's x (ttnn/fc_ksplit.py; RISCV_1, NoC 1, the sender's reads and multicast as router_mm's
// rm_x.cpp): this slice's blocks of x (block j = the slice's j-th block b: x's MT tile rows, K tiles KB b .. KB b +
// KB - 1, row-major in the slot) into CB_X, a ring of XS slots, at the same address on every core (CB_X on the sender
// too, its staging slots).
//   - The slice's sender (a core of its own, outside the slice's rectangle) reads each block from DRAM into its
//     staging slot (up to XS in flight), waits until every core of the slice has reserved that slot (their counts of
//     slots reserved, written into the sender's CB_XF word of theirs), multicasts it to the rectangle and then sets
//     their SEM_X to the blocks sent (a word a block in CB_WORDS, never rewritten).
//   - The slice's cores reserve their slots as they free up, tell the sender their count (a 4-byte write from a word
//     of their own, a word a count), and push each block to the compute once SEM_X shows it.
// Every count a plain write (no atomic: an atomic's response may come after the kernel's end);
// each core sets its own words and semaphore back to 0 at its end, after the last write into them.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t index = get_arg_val<uint32_t>(1);  // 0: the slice's sender, else 1 + the core's index in the slice
    const uint32_t cores = get_arg_val<uint32_t>(2);  // the slice's cores (the sender not counted)
    const uint32_t mc0_x = get_arg_val<uint32_t>(3);  // the slice's rectangle in NoC 1's corner order
    const uint32_t mc0_y = get_arg_val<uint32_t>(4);
    const uint32_t mc1_x = get_arg_val<uint32_t>(5);
    const uint32_t mc1_y = get_arg_val<uint32_t>(6);
    const uint32_t snd_x = get_arg_val<uint32_t>(7);  // the sender's NoC 1 coordinates
    const uint32_t snd_y = get_arg_val<uint32_t>(8);
    const uint32_t slice = get_arg_val<uint32_t>(9);  // the slice: its blocks b < NB - 1 with PAT[b % L] = slice (FK_PAT)
    const uint32_t last = get_arg_val<uint32_t>(10);  // 1: also the last block NB - 1 (the owners' slice)

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xf = get_compile_time_arg_val(1);     // the sender's words: a slice core's reserved count
    constexpr uint32_t cb_words = get_compile_time_arg_val(2);  // this core's source words
    constexpr uint32_t sem_x = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);  // x's K tiles
    constexpr uint32_t KB = get_compile_time_arg_val(5);
    constexpr uint32_t MT = get_compile_time_arg_val(6);
    constexpr uint32_t NB = get_compile_time_arg_val(7);
    constexpr uint32_t XS = get_compile_time_arg_val(8);  // CB_X's slots
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr uint32_t SLOT = MT * KB;  // tiles a slot

    noc_async_read_set_trid(0, noc_index);  // (transaction ids from a clean state, as rm_x.cpp)
    noc_async_write_set_trid(0, noc_index);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
    DeviceZoneScopedN("FK-X");
    uint8_t list[NB];
    uint32_t blocks = 0;
    for (uint32_t b = 0; b + 1 < NB; ++b) {
        if (((FK_PAT >> (4 + 2 * (b % (FK_PAT & 15)))) & 3u) == slice) {
            list[blocks++] = b;
        }
    }
    if (last) {
        list[blocks++] = NB - 1;
    }
    auto block_of = [&](uint32_t j) { return static_cast<uint32_t>(list[j]); };
    const uint32_t xt = get_local_cb_interface(cb_x).fifo_page_size;
    const uint32_t base = get_write_ptr(cb_x);
    const uint32_t words = get_write_ptr(cb_words);
    auto word = [&](uint32_t i) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16 * i); };
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x));
    if (index == 0) {
        const auto s_x = TensorAccessor(x_args, x_addr);
        const uint32_t xf = get_write_ptr(cb_xf);
        const uint32_t sem_addr = get_semaphore(sem_x);
        constexpr uint32_t TRID0 = 9;  // ids 9 .. 9 + XS - 1
        auto issue = [&](uint32_t j) {
            const uint32_t b = block_of(j);
            const uint32_t dst = base + (j % XS) * SLOT * xt;
            noc_async_read_set_trid(TRID0 + j % XS);
            for (uint32_t r = 0; r < MT; ++r) {
                for (uint32_t i = 0; i < KB; ++i) {
                    noc_async_read(s_x.get_noc_addr(r * kt + b * KB + i), dst + (r * KB + i) * xt, xt);
                }
            }
        };
        const uint32_t ahead = blocks < XS ? blocks : XS;
        for (uint32_t j = 0; j < ahead; ++j) {
            issue(j);
        }
        for (uint32_t j = 0; j < blocks; ++j) {
            noc_async_read_barrier_with_trid(TRID0 + j % XS);
            for (uint32_t i = 1; i <= cores; ++i) {  // every slice core reserved the slot
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xf + 16 * i), j + 1);
            }
            const uint32_t src = base + (j % XS) * SLOT * xt;
            noc_async_write_multicast(
                src, get_noc_multicast_addr(mc0_x, mc0_y, mc1_x, mc1_y, src), SLOT * xt, cores, true);
            *word(j) = j + 1;
            noc_semaphore_set_multicast(
                words + 16 * j, get_noc_multicast_addr(mc0_x, mc0_y, mc1_x, mc1_y, sem_addr), cores);
            if (j + XS < blocks) {  // the slot's next block, once this one has left it
                noc_async_writes_flushed();
                issue(j + XS);
            }
        }
        noc_async_read_set_trid(0);
        noc_async_write_barrier();
        for (uint32_t i = 1; i <= cores; ++i) {  // (every slice core's last count came before the waits above)
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xf + 16 * i) = 0;
        }
    } else {
        const uint64_t my_word_at_sender = get_noc_addr(snd_x, snd_y, get_write_ptr(cb_xf) + 16 * index);
        uint32_t reserved = 0, pushed = 0;
        while (pushed < blocks) {
            while (reserved < blocks && reserved - pushed < XS &&
                   cb_pages_reservable_at_back(cb_x, (reserved - pushed + 1) * SLOT)) {
                ++reserved;
                *word(reserved - 1) = reserved;
                noc_semaphore_set_remote(words + 16 * (reserved - 1), my_word_at_sender);
            }
            invalidate_l1_cache();
            if (*sem < pushed + 1) {
                continue;
            }
            cb_reserve_back(cb_x, SLOT);
            cb_push_back(cb_x, SLOT);
            ++pushed;
        }
        noc_async_write_barrier();
        noc_semaphore_set(sem, 0);  // (the sender's last flag came before the last wait)
    }
}
