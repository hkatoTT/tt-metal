// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The column's way back inside the experts op: a reducer core's adds, as
// reduce_scatter_minimal_async's ring makes them (ring_reduction.cpp: bf16 sources, a 16-bit Dst, fp32_dest_acc off),
// so that the sums are the same bits as ttnn.reduce_scatter's over the mesh column (its
// ring reduce-scatter):
//   add:   out = ELWADD(a, b)              (a: the partial sum come from the previous chip, b: this chip's partial)
//   final: out = ELWADD(a + b, Dst: d)     (a: the chain from chip d - 1, d: the one from chip d + 1)
// on blocks of B tiles (a block: a slice's tile rows by the core's tile columns), in chunks of 8 tiles (Dst's half).
// Runtime args: the op count, the ops (4 words each), then a pop count (blocks) for each of the n_cbs cbs from
// first_cb:
//   [kind (0 add, 1 final) | out cb << 8, a cb << 8 | a's block, b cb << 8 | b's block, d cb << 8 | d's block]
// A source block k is waited for as its cb's first (k + 1) B tiles (the landing cbs are pushed block by block in
// order and popped only at the end, by the counts given); an add's output is pushed as its cb's next block (later ops
// may take it), and then one tile is pushed to `sig_cb` (if not 0: the reader-writer's word that it is there); a
// final's output to its cb, which the writer pops.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);         // tiles a block
    constexpr uint32_t n_cbs = get_compile_time_arg_val(1);     // the cbs popped at the end
    constexpr uint32_t first_cb = get_compile_time_arg_val(2);  // (their ids: first_cb .. first_cb + n_cbs - 1)
    constexpr uint32_t sig_cb = get_compile_time_arg_val(3);    // 0: none
    constexpr uint32_t CHUNK = 8;

    size_t a = 0;
    const uint32_t n_ops = get_arg_val<uint32_t>(a++);
    const size_t pops = a + 4 * n_ops;
    bool started = false;
    for (uint32_t op = 0; op < n_ops; ++op) {
        const uint32_t w0 = get_arg_val<uint32_t>(a++), w1 = get_arg_val<uint32_t>(a++);
        const uint32_t w2 = get_arg_val<uint32_t>(a++), w3 = get_arg_val<uint32_t>(a++);
        const bool final = (w0 & 0xff) == 1;
        const uint32_t out = w0 >> 8;
        const uint32_t cba = w1 >> 8, ka = w1 & 0xff, cbb = w2 >> 8, kb = w2 & 0xff, cbd = w3 >> 8, kd = w3 & 0xff;
        CircularBuffer(cba).wait_front((ka + 1) * B);
        CircularBuffer(cbb).wait_front((kb + 1) * B);
        if (final) {
            CircularBuffer(cbd).wait_front((kd + 1) * B);
        }
        if (!started) {
            compute_kernel_hw_startup(cba, cbb, out);
            started = true;
        }
        CircularBuffer o(out);
        o.reserve_back(B);
        for (uint32_t c0 = 0; c0 < B; c0 += CHUNK) {
            const uint32_t n = B - c0 < CHUNK ? B - c0 : CHUNK;
            tile_regs_acquire();
            if (final) {
                copy_init(cbd);
                for (uint32_t i = 0; i < n; ++i) {
                    copy_tile(cbd, kd * B + c0 + i, i);
                }
                add_init(cba, cbb, true);
            } else {
                add_init(cba, cbb, false);
            }
            for (uint32_t i = 0; i < n; ++i) {
                add_tiles(cba, cbb, ka * B + c0 + i, kb * B + c0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile<true>(i, out, c0 + i);
            }
            tile_regs_release();
        }
        o.push_back(B);
        if (!final && sig_cb != 0) {
            CircularBuffer s(sig_cb);
            s.reserve_back(1);
            s.push_back(1);
        }
    }
    for (uint32_t i = 0; i < n_cbs; ++i) {
        const uint32_t blocks = get_arg_val<uint32_t>(pops + i);
        if (blocks != 0) {
            CircularBuffer c(first_cb + i);
            c.wait_front(blocks * B);
            c.pop_front(blocks * B);
        }
    }
}
