// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2 (dflash2_ops.heads_split): the block's q and k heads [1, 1, U B, W] bf16 (tiled; rows (u, pos), the q heads
// at 128-column blocks 0 .., the k heads at k0 ..) into the virtual users' tiles of the paged sdpa decode:
//   q [1, 2 U, 32, 128]: virtual user (u, j), row h B + pos = q head G j + h of row u B + pos (G q heads a KV head);
//   k [1, 2 U, 32, 128]: virtual user (u, j), row pos = k head j of row u B + pos;
// the other rows zero.  A core takes items (virtual user, column tile t): it reads the source tiles of its user's tile
// row (B divides 32: a user's rows lie in one tile row) and copies the rows' face rows (32 bytes) on the RISC-V.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t i0 = get_arg_val<uint32_t>(3);  // this core's items (v, t): v * 4 + t
    const uint32_t i1 = get_arg_val<uint32_t>(4);
    const uint32_t k_tile0 = get_arg_val<uint32_t>(5);  // k's first tile (q and k one tensor: k after q's 4 2 U tiles)

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);       // q heads a KV head
    constexpr uint32_t kvh = get_compile_time_arg_val(2);     // KV heads a chip
    constexpr uint32_t src_wt = get_compile_time_arg_val(3);  // tiles across the source
    constexpr uint32_t k0 = get_compile_time_arg_val(4);      // the k heads' first 128-column block
    constexpr uint32_t cb = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048, FACE = 512, FROW = 32, WT = 4;

    const auto s_src = TensorAccessor(src_args, src_addr, TILE);
    const auto s_q = TensorAccessor(out_args, q_addr, TILE);
    const auto s_k = TensorAccessor(out_args, k_addr, TILE);
    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t srcs = l1;               // G source tiles
    const uint32_t outq = srcs + G * TILE;  // the q output tile
    const uint32_t outk = outq + TILE;      // the k output tile

    for (uint32_t item = i0; item < i1; ++item) {
        const uint32_t v = item / WT, t = item % WT;
        const uint32_t u = v / kvh, j = v % kvh;
        const uint32_t tr = (u * B) / 32, rb0 = (u * B) % 32;
        {
            for (uint32_t h = 0; h < G; ++h) {
                noc_async_read(s_src.get_noc_addr(tr * src_wt + (j * G + h) * WT + t), srcs + h * TILE, TILE);
            }
            for (uint32_t off = 0; off < 2 * TILE; off += MEM_ZEROS_SIZE) {
                noc_async_read(get_noc_addr(MEM_ZEROS_BASE), outq + off, MEM_ZEROS_SIZE);
            }
            noc_async_read_barrier();
            // q: rows h B + pos
            for (uint32_t h = 0; h < G; ++h) {
                for (uint32_t pos = 0; pos < B; ++pos) {
                    const uint32_t rs = rb0 + pos, rd = h * B + pos;
                    for (uint32_t f = 0; f < 2; ++f) {
                        const uint32_t* from = reinterpret_cast<const uint32_t*>(
                            srcs + h * TILE + ((rs / 16) * 2 + f) * FACE + (rs % 16) * FROW);
                        uint32_t* to =
                            reinterpret_cast<uint32_t*>(outq + ((rd / 16) * 2 + f) * FACE + (rd % 16) * FROW);
                        for (uint32_t w = 0; w < FROW / 4; ++w) {
                            to[w] = from[w];
                        }
                    }
                }
            }
            noc_async_write(outq, s_q.get_noc_addr(v * WT + t), TILE);
            // k: the source tile of its head, rows pos
            noc_async_read(s_src.get_noc_addr(tr * src_wt + (k0 + j) * WT + t), srcs, TILE);
            noc_async_read_barrier();
            for (uint32_t pos = 0; pos < B; ++pos) {
                const uint32_t rs = rb0 + pos, rd = pos;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t* from =
                        reinterpret_cast<const uint32_t*>(srcs + ((rs / 16) * 2 + f) * FACE + (rs % 16) * FROW);
                    uint32_t* to = reinterpret_cast<uint32_t*>(outk + ((rd / 16) * 2 + f) * FACE + (rd % 16) * FROW);
                    for (uint32_t w = 0; w < FROW / 4; ++w) {
                        to[w] = from[w];
                    }
                }
            }
            noc_async_write(outk, s_k.get_noc_addr(k_tile0 + v * WT + t), TILE);
            noc_async_write_barrier();
        }
    }
}
