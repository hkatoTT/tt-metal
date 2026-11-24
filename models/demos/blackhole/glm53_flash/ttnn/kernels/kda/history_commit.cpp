// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// KDA prefill commit (kda_prefill_ops.history_commit): a mesh row's request's conv history (rows 0 .. K - 1 of the
// tiles [1, K, C] bf16) into row `user` of the K history tensors [1, 1, B, C] fp32 (tiled), `user` from the commit
// table (row-major int32, one value; -1: the row has no request, nothing is written), in place of each row's slice,
// typecast to fp32 and the one-hot multiply and addcmul over all B users.  bf16 -> fp32 is exact (the bits moved up
// 16).  Per tile across C a row is two face rows: each 32-byte bf16 face row is read into a 64-byte slot of L1 at the
// source's offset mod 64 (a NoC transfer under 64 bytes needs the same offset mod 64 at both ends), widened into a
// 64-byte fp32 slot and written to the destination's face row (64-byte aligned).  This core: tiles t0 .. t1 - 1.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t t1 = get_arg_val<uint32_t>(3);
    // the K destinations' addresses: runtime args 4 .. 4 + K - 1

    constexpr uint32_t tiles = get_compile_time_arg_val(0);       // tiles across C
    constexpr uint32_t rows = get_compile_time_arg_val(1);        // K (< 16: face rows of the first two faces)
    constexpr uint32_t table_page = get_compile_time_arg_val(2);  // the table's page (bytes)
    constexpr uint32_t per = get_compile_time_arg_val(3);         // a core's tiles at most
    constexpr uint32_t cb = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto table_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t STILE = 2048, SFACE = 512, SFROW = 32;  // bf16 tiles
    constexpr uint32_t DTILE = 4096, DFACE = 1024;             // fp32 tiles
    constexpr uint32_t SLOT = 64;                              // a face row's slot (an fp32 face row: 64 bytes)
    constexpr uint32_t slots = rows * per * 2;

    const uint32_t l1 = get_write_ptr(cb);
    const auto s_table = TensorAccessor(table_args, table_addr, table_page);
    noc_async_read(s_table.get_noc_addr(0), l1, table_page);
    noc_async_read_barrier();
    const int32_t user = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1)[0];
    if (user < 0) {
        return;
    }
    const uint32_t src_slots = l1 + ((table_page + SLOT - 1) & ~(SLOT - 1));
    const uint32_t dst_slots = src_slots + slots * SLOT;
    const auto s_src = TensorAccessor(src_args, src_addr, STILE);

    uint32_t n = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t t = t0; t < t1; ++t) {
            for (uint32_t h = 0; h < 2; ++h, ++n) {
                const uint32_t off = h * SFACE + r * SFROW;
                noc_async_read(s_src.get_noc_addr(t, off), src_slots + n * SLOT + off % SLOT, SFROW);
            }
        }
    }
    noc_async_read_barrier();

    const uint32_t ur = static_cast<uint32_t>(user) % 32, tile_row = static_cast<uint32_t>(user) / 32;
    n = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const auto s_dst = TensorAccessor(dst_args, get_arg_val<uint32_t>(4 + r), DTILE);
        for (uint32_t t = t0; t < t1; ++t) {
            for (uint32_t h = 0; h < 2; ++h, ++n) {
                const uint32_t off = h * SFACE + r * SFROW;
                volatile tt_l1_ptr uint16_t* from =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_slots + n * SLOT + off % SLOT);
                volatile tt_l1_ptr uint32_t* to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_slots + n * SLOT);
                for (uint32_t w = 0; w < 16; ++w) {
                    to[w] = static_cast<uint32_t>(from[w]) << 16;
                }
                const uint32_t dst_off = ((ur / 16) * 2 + h) * DFACE + (ur % 16) * SLOT;
                noc_async_write(dst_slots + n * SLOT, s_dst.get_noc_addr(tile_row * tiles + t, dst_off), SLOT);
            }
        }
    }
    noc_async_write_barrier();
}
