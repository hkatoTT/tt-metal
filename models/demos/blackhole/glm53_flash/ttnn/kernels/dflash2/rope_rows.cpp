// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// DFlash2 (dflash2_ops.rope_rows): the RoPE cos and sin of each output row, gathered from fp32 tables (row-major, a
// table row a page) into tiles [N, 128] fp32.  Row o takes position idx[o / div] + (o % mod if mod).  The tables:
//   fmt 0: cos_tab and sin_tab [T, 128] (512 bytes a row);
//   fmt 1: one table [T, 128], cos's first half (columns 0-63) then sin's (64-127): the RoPE's two halves are the same
//          values (cat(freqs, freqs)), so column c of cos is column c % 64 of the row, of sin column 64 + c % 64.
// The table row of output row o (i = o / div):
//   wmode 0: the position (clamped into the table: an index past the index tensor or a negative position: row 0);
//   wmode 1: i (a table of the call's own rows, one per index entry, made by the host);
//   wmode 2: a window per user: row u * wrows + (position - base[u] + woff), u = (i / udiv) % umod, out of the window:
//            its user's row 0 (each user's rows of positions base[u] - woff .., made by the host; base [n_base] int32).
//   (wmode 1 and 2, the drafter's former host-made rows, are unused: dflash2_ops.rope_rows passes 0.)
// Each face row (16 fp32, 64 bytes) is read straight from its table row into the tile in L1 (both 64-byte aligned) and
// the tiles are written whole.  The core with `first` also writes cur_pos[v] = idx[v / cur_div] + cur_add for v <
// n_cur (the paged sdpa decode's position per virtual user), when n_cur > 0.
void kernel_main() {
    const uint32_t idx_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_tab = get_arg_val<uint32_t>(1);
    const uint32_t sin_tab = get_arg_val<uint32_t>(2);
    const uint32_t cos_out = get_arg_val<uint32_t>(3);
    const uint32_t sin_out = get_arg_val<uint32_t>(4);
    const uint32_t cur_addr = get_arg_val<uint32_t>(5);
    const uint32_t i0 = get_arg_val<uint32_t>(6);  // this core's items (tile row, column tile): tr * 4 + t
    const uint32_t i1 = get_arg_val<uint32_t>(7);
    const uint32_t first = get_arg_val<uint32_t>(8);
    const uint32_t base_addr = get_arg_val<uint32_t>(9);

    constexpr uint32_t n_idx = get_compile_time_arg_val(0);  // entries of the index
    constexpr uint32_t div = get_compile_time_arg_val(1);
    constexpr uint32_t mod = get_compile_time_arg_val(2);
    constexpr uint32_t n_cur = get_compile_time_arg_val(3);
    constexpr uint32_t cur_div = get_compile_time_arg_val(4);
    constexpr uint32_t cur_add = get_compile_time_arg_val(5);
    constexpr uint32_t max_pos = get_compile_time_arg_val(6);  // table rows
    constexpr uint32_t cb = get_compile_time_arg_val(7);
    constexpr uint32_t fmt = get_compile_time_arg_val(8);
    constexpr uint32_t wmode = get_compile_time_arg_val(9);
    constexpr uint32_t wrows = get_compile_time_arg_val(10);
    constexpr uint32_t udiv = get_compile_time_arg_val(11);
    constexpr uint32_t umod = get_compile_time_arg_val(12);
    constexpr uint32_t woff = get_compile_time_arg_val(13);
    constexpr uint32_t n_base = get_compile_time_arg_val(14);  // entries of the base (wmode 2)
    constexpr auto idx_args = TensorAccessorArgs<15>();
    constexpr auto tab_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();
    constexpr auto cur_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto base_args = TensorAccessorArgs<cur_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096, FACE = 1024, FROW = 64, TAB_ROW = 512, WT = 4;  // fp32 tiles, 128 wide
    constexpr uint32_t idx_bytes = n_idx * 4;
    constexpr uint32_t idx_slot = (idx_bytes + 63) & ~63u;
    constexpr uint32_t cur_slot = ((n_cur * 4 > 64 ? n_cur * 4 : 64) + 63) & ~63u;
    constexpr uint32_t base_bytes = n_base * 4;

    const auto s_idx = TensorAccessor(idx_args, idx_addr, idx_bytes);
    const auto s_cos = TensorAccessor(tab_args, cos_tab, TAB_ROW);
    const auto s_sin = TensorAccessor(tab_args, sin_tab, TAB_ROW);
    const auto s_cos_out = TensorAccessor(out_args, cos_out, TILE);
    const auto s_sin_out = TensorAccessor(out_args, sin_out, TILE);

    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t tiles = l1 + idx_slot;  // the item's cos tile, then its sin tile
    const uint32_t cur_buf = tiles + 2 * TILE;
    const uint32_t base_buf = cur_buf + cur_slot;
    noc_async_read(s_idx.get_noc_addr(0), l1, idx_bytes);
    if constexpr (n_base > 0) {
        const auto s_base = TensorAccessor(base_args, base_addr, base_bytes);
        noc_async_read(s_base.get_noc_addr(0), base_buf, base_bytes);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* idx = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1);
    volatile tt_l1_ptr int32_t* base = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base_buf);

    for (uint32_t item = i0; item < i1; ++item) {
        const uint32_t tr = item / WT, t = item % WT;
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t o = tr * 32 + r;
            const uint32_t i = o / div;
            uint32_t row;
            if constexpr (wmode == 1) {
                row = i < max_pos ? i : 0;
            } else {
                int32_t pos = i < n_idx ? idx[i] : 0;
                if constexpr (mod != 0) {
                    pos += static_cast<int32_t>(o % mod);
                }
                if constexpr (wmode == 0) {
                    row = (pos < 0 || pos >= static_cast<int32_t>(max_pos)) ? 0 : static_cast<uint32_t>(pos);
                } else {
                    const uint32_t u = (i / udiv) % umod;
                    const int32_t w = pos - (u < n_base ? base[u] : 0) + static_cast<int32_t>(woff);
                    row = u * wrows + ((w < 0 || w >= static_cast<int32_t>(wrows)) ? 0u : static_cast<uint32_t>(w));
                    if (row >= max_pos) {
                        row = 0;
                    }
                }
            }
            const uint32_t at = (r / 16) * 2 * FACE + (r % 16) * FROW;
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t dst = at + h * FACE;
                if constexpr (fmt == 0) {
                    const uint32_t off = t * 128 + h * FROW;  // columns 32 t + 16 h .. of the table row
                    noc_async_read(s_cos.get_noc_addr(row, off), tiles + dst, FROW);
                    noc_async_read(s_sin.get_noc_addr(row, off), tiles + TILE + dst, FROW);
                } else {
                    const uint32_t off = ((t * 32 + h * 16) % 64) * 4;  // cos's column (32 t + 16 h) % 64
                    noc_async_read(s_cos.get_noc_addr(row, off), tiles + dst, FROW);
                    noc_async_read(s_cos.get_noc_addr(row, off + 256), tiles + TILE + dst, FROW);  // sin's: + 64
                }
            }
        }
        noc_async_read_barrier();
        noc_async_write(tiles, s_cos_out.get_noc_addr(item), TILE);
        noc_async_write(tiles + TILE, s_sin_out.get_noc_addr(item), TILE);
        noc_async_write_barrier();
    }

    if constexpr (n_cur > 0) {
        if (first == 0) {
            return;
        }
        constexpr uint32_t cur_bytes = n_cur * 4;
        volatile tt_l1_ptr int32_t* cur = reinterpret_cast<volatile tt_l1_ptr int32_t*>(cur_buf);
        for (uint32_t v = 0; v < n_cur; ++v) {
            const uint32_t i = v / cur_div;
            cur[v] = (i < n_idx ? idx[i] : 0) + static_cast<int32_t>(cur_add);
        }
        const auto s_cur = TensorAccessor(cur_args, cur_addr, cur_bytes);
        noc_async_write(cur_buf, s_cur.get_noc_addr(0), cur_bytes);
        noc_async_write_barrier();
    }
}
