// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sequence-parallel KDA's state maps to the rows that read them (kda_maps_x.py, maps_x_send.cpp), a copy core's
// RISC: this chip's own maps' tiles t0 .. t0 + n - 1 of `packed` (bf16 tiles, a page a tile) into the landing's slot r
// (pages r TILES + t), on this RISC's NoC, through NB slots of this RISC's half of the core's CB, read NB ahead by
// transaction ids (as x2d_copy.cpp).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);  // (its half of the CB)
    constexpr uint32_t TILES = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t cb_copy = get_compile_time_arg_val(3);
    constexpr auto p_args = TensorAccessorArgs<4>();
    constexpr auto l_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 2048;
    static_assert(NB >= 1 && NB <= 15, "kda maps copy: transaction ids 1 .. 15");

    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t l_addr = get_arg_val<uint32_t>(1);
    const uint32_t r = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t n = get_arg_val<uint32_t>(4);
    const auto s_p = TensorAccessor(p_args, p_addr, TILE);
    const auto s_l = TensorAccessor(l_args, l_addr, TILE);
    const uint32_t buf = get_write_ptr(cb_copy) + role * NB * TILE;

    auto read = [&](uint32_t i) {
        noc_async_read_set_trid(i % NB + 1);
        noc_async_read(s_p.get_noc_addr(t0 + i), buf + (i % NB) * TILE, TILE);
    };
    for (uint32_t i = 0; i < NB && i < n; ++i) {
        read(i);
    }
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read_barrier_with_trid(i % NB + 1);
        noc_async_write(buf + (i % NB) * TILE, s_l.get_noc_addr(r * TILES + t0 + i), TILE);
        if (i + NB < n) {
            noc_async_writes_flushed();  // (the slot again)
            read(i + NB);
        }
    }
    noc_async_read_set_trid(0);
    noc_async_write_barrier();
}
