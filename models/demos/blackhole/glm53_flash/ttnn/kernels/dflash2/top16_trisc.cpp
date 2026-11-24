// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// top16's TRISC split (GLM53_DFLASH2_TOP16_TRISC, dflash2_ops.Top16): the core's
// unpack thread scans the last rows of the core's top16 items (top16.cpp is the data-movement RISCs' part).  For each of
// the core's items (risc 0 / 1) with rows t0 .. g - 1 here: wait for the item's sync CB (its RISC pushes a page once the
// item's tiles are in L1 and its done word is zeroed), scan the rows into the item's lists (`scan_rows`, the same code
// and order as the RISC's), write 1 into the done word and pop the page.  The math and pack threads do nothing.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/debug/kernel_profiler.hpp"

#include "top16_common.hpp"

void kernel_main() {
    constexpr uint32_t runs = get_compile_time_arg_val(0);  // runs a group
    constexpr uint32_t per = get_compile_time_arg_val(1);   // column tiles a run, at most
    constexpr uint32_t G = get_compile_time_arg_val(2);     // rows a group, at most
    constexpr uint32_t cb = get_compile_time_arg_val(3);    // the scratch
    constexpr uint32_t cb_sync0 = get_compile_time_arg_val(4);
    constexpr uint32_t FROW = 64;
    constexpr uint32_t LIST = TOPN * 8;
    constexpr uint32_t TILE_B = 2 * G * FROW;
    constexpr uint32_t DATA = per * TILE_B;
    constexpr uint32_t SL = runs * G * LIST;
#ifdef TRISC_UNPACK
    // (the scratch's start: one page, its read pointer there.  Not get_tile_address: that posts the address into the
    // math and pack threads' mailboxes, which these threads would never read: 4 values unread in a writer's FIFOs stall
    // it: the second or third call hung)
    const uint32_t l1 = get_tile_l1_byte_address(get_operand_id(cb), 0);
    for (uint32_t i = 0; i < 2; ++i) {
        // per item: present, risc, rows g, t0, ct0, ct1, run, valid columns
        const uint32_t a = 8 * i;
        if (get_arg_val<uint32_t>(a) == 0) {
            continue;
        }
        const uint32_t risc = get_arg_val<uint32_t>(a + 1);
        const uint32_t g = get_arg_val<uint32_t>(a + 2);
        const uint32_t t0 = get_arg_val<uint32_t>(a + 3);
        const uint32_t ct0 = get_arg_val<uint32_t>(a + 4);
        const uint32_t ct1 = get_arg_val<uint32_t>(a + 5);
        const uint32_t run = get_arg_val<uint32_t>(a + 6);
        const uint32_t valid_cols = get_arg_val<uint32_t>(a + 7);
        const uint32_t data = l1 + risc * DATA;
        const uint32_t mine = l1 + 2 * DATA + risc * SL + run * G * LIST;
        volatile uint32_t* done = reinterpret_cast<volatile uint32_t*>(l1 + 2 * DATA + 2 * SL + 2 * G * 2 * FROW + risc * 16);
        cb_wait_front(cb_sync0 + risc, 1);
        asm volatile("fence" ::: "memory");  // (the RISC's small L1 cache: lines the NoC wrote)
        {
            DeviceZoneScopedN("T16_TSCAN");
            scan_rows(data, t0, g, ct0, ct1, valid_cols, mine, G, TILE_B);
        }
        asm volatile("fence" ::: "memory");
        *done = 1;
        cb_pop_front(cb_sync0 + risc, 1);
    }
#endif
}
