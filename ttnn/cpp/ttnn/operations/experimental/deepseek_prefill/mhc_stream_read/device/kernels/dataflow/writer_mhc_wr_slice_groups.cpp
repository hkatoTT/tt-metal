// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// mhc_stream_write_read's prefill groups with the write on the slice cores (MHC_WR_MERGED): as
// writer_mhc_read_slice_groups.cpp, after first writing the slice's tiles of the new streams to DRAM as the compute
// makes them (CB_S: the 4 row tiles of column k at k R + rt, pushed 4 a column; page (4 g + rt) d_tiles + s K + k of
// the new streams; compile-time arg 15 CB_S, then the new streams' accessor after the output's; runtime arg 9 its
// address).  The groups form: as writer_mhc_read_slice.cpp, its group's mix core (runtime args 2, 3) and the output's
// tile row g (runtime arg 4).
// Slice s: its partials into the mix core's CB_PART (tile (rt, n) at (rt 4 + n) S + s, the order mhc_stream_mix sums
// them in), followed by a semaphore increment there once written; then its columns of the normed block input to the
// output.  With `parts` > 1 (fn read once a column, reader_mhc_read_slice_groups.cpp): first, once the reader has the
// group's own part q of fn and gamma (SEM_OWN; runtime arg 8), that part multicast to the other groups' slice s (the
// grid column, runtime args 5 .. 7) at the same addresses (the CBs' base: this program's first push), then SEM_FN0 + q
// set there.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(1);
    const uint32_t mix_x = get_arg_val<uint32_t>(2);
    const uint32_t mix_y = get_arg_val<uint32_t>(3);
    const uint32_t g = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_part = get_compile_time_arg_val(2);
    constexpr uint32_t slices = get_compile_time_arg_val(3);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t sem_part = get_compile_time_arg_val(6);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(7);
    // 0: the output's tile row g whole; 8 (groups of 8 tokens): the group's rows 8 (g % 4) .. of tile row g / 4, the
    // tiles' rows 0 .. 7 (faces 0 and 1) into those rows of the faces (0 and 1, or 2 and 3)
    constexpr uint32_t sub_rows = get_compile_time_arg_val(8);
    constexpr uint32_t parts = get_compile_time_arg_val(9);  // 1: no multicast of fn
    constexpr uint32_t sem_own = get_compile_time_arg_val(10);
    constexpr uint32_t sem_fn0 = get_compile_time_arg_val(11);
    constexpr uint32_t cb_fn = get_compile_time_arg_val(12);
    constexpr uint32_t cb_g = get_compile_time_arg_val(13);
    constexpr bool fn_unicast = get_compile_time_arg_val(14) != 0;  // the reader sends the part (no multicast here)
    constexpr uint32_t cb_s = get_compile_time_arg_val(15);
    constexpr uint32_t k_own = get_compile_time_arg_val(16);  // its own columns (the helpers write the others')
    constexpr auto out_args = TensorAccessorArgs<17>();
    constexpr auto new_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t t32 = get_local_cb_interface(cb_out).fifo_page_size;
    const uint32_t t16 = get_local_cb_interface(cb_y).fifo_page_size;
#ifdef MHC_PART_AFTER_NS
    // (GLM53_MHC_PART_MIX: no CB_PART on this core; the mix core's follows its CB_NS, at this core's CB_NS address)
    const uint32_t part_base = get_local_cb_interface(MHC_PART_AFTER_NS).fifo_limit;
#else
    const uint32_t part_base = get_local_cb_interface(cb_part).fifo_wr_ptr;  // the mix core's too
#endif
    const auto s_out = TensorAccessor(out_args, out_addr);
    const auto s_new = TensorAccessor(new_args, get_arg_val<uint32_t>(9));

    Noc noc;
    CircularBuffer out(cb_out), y(cb_y);
    if constexpr (parts > 1 && !fn_unicast) {
        constexpr uint32_t kp = k_tiles / parts, proj_tiles = 3;
        const uint32_t col_x = get_arg_val<uint32_t>(5), top_y = get_arg_val<uint32_t>(6);
        const uint32_t bottom_y = get_arg_val<uint32_t>(7);
        const uint32_t q = get_arg_val<uint32_t>(8);  // its part (g in the groups of 8, g / 2 in the prefill's)
        // the column's rectangle, its corners in the order this NoC's multicast takes them
        auto mcast = [&](uint32_t addr) {
            return noc_index == 0 ? get_noc_multicast_addr(col_x, top_y, col_x, bottom_y, addr)
                                  : get_noc_multicast_addr(col_x, bottom_y, col_x, top_y, addr);
        };
        const uint32_t fn_base = get_local_cb_interface(cb_fn).fifo_wr_ptr;  // (every core's: the first push)
        const uint32_t g_base = get_local_cb_interface(cb_g).fifo_wr_ptr;
        const uint32_t fn_part = fn_base + q * kp * proj_tiles * t16, g_part = g_base + q * kp * t16;
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own)), 1);
#ifdef MHC_WR_FN_UNICAST_W
        // (MHC_WR_UNICAST: no multicast in the program) the part to the column's other groups' slice s by unicasts on
        // this NoC (the writer's: not the one the reader's DRAM reads take; runtime arg 10 their count, then their NoC
        // x, y), then their flag SEM_FN0 + q set by a write (no atomics)
        const uint32_t dests = get_arg_val<uint32_t>(10);
        for (uint32_t i = 0; i < dests; ++i) {
            const uint32_t x = get_arg_val<uint32_t>(11 + 2 * i), y = get_arg_val<uint32_t>(12 + 2 * i);
            noc_async_write(fn_part, get_noc_addr(x, y, fn_part), kp * proj_tiles * t16);
            noc_async_write(g_part, get_noc_addr(x, y, g_part), kp * t16);
        }
        noc_async_write_barrier();  // (the parts landed before their flags)
        const uint32_t sem = get_semaphore(sem_fn0 + q);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) = 1;
        for (uint32_t i = 0; i < dests; ++i) {
            const uint32_t x = get_arg_val<uint32_t>(11 + 2 * i), y = get_arg_val<uint32_t>(12 + 2 * i);
            noc_semaphore_set_remote(sem, get_noc_addr(x, y, sem));
        }
        (void)mcast;
#else
        noc_async_write_multicast(fn_part, mcast(fn_part), kp * proj_tiles * t16, parts - 1);
        noc_async_write_multicast(g_part, mcast(g_part), kp * t16, parts - 1);
        noc_async_write_barrier();
        const uint32_t sem = get_semaphore(sem_fn0 + q);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) = 1;
        noc_semaphore_set_multicast(sem, mcast(sem), parts - 1);
#endif
    }
#ifdef MHC_WR_WRITER_DRAM
    {  // the new streams, a column at a time as the compute packs them
        CircularBuffer sn(cb_s);
        const uint32_t s_base = get_local_cb_interface(cb_s).fifo_wr_ptr;  // (nothing pushed yet: the base)
        const uint32_t t_s = get_local_cb_interface(cb_s).fifo_page_size;
        for (uint32_t k = 0; k < k_own; ++k) {
            sn.wait_front((k + 1) * row_tiles);
            for (uint32_t rt = 0; rt < row_tiles; ++rt) {
#ifdef MHC_WR_PROBE_NOWRITE
                continue;
#endif
                noc_async_write(
                    s_base + (k * row_tiles + rt) * t_s,
                    s_new.get_noc_addr((row_tiles * g + rt) * d_tiles + s * k_tiles + k),
                    t_s);
            }
        }
        noc_async_writes_flushed();
    }
#endif
    out.wait_front(4 * row_tiles);
    for (uint32_t i = 0; i < 4 * row_tiles; ++i) {  // i = rt 4 + n
        noc_async_write(
            out.get_read_ptr() + i * t32, get_noc_addr(mix_x, mix_y, part_base + (i * slices + s) * t32), t32);
    }
    noc_async_write_barrier();
#ifdef MHC_WR_PART_FLAGS
    {  // (MHC_WR_UNICAST) the mix core's flag MHC_WR_PART_FLAGS + s set by a write, in place of an atomic increment
        const uint32_t flag = get_semaphore(sem_part);  // (unused on a slice core: the flag's source word)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag) = 1;
        noc_semaphore_set_remote(flag, get_noc_addr(mix_x, mix_y, get_semaphore(MHC_WR_PART_FLAGS + s)));
    }
#else
    noc_semaphore_inc(get_noc_addr(mix_x, mix_y, get_semaphore(sem_part)), 1);
#endif
    out.pop_front(4 * row_tiles);

    y.wait_front(k_tiles);
    if constexpr (sub_rows == 0) {
        for (uint32_t c = 0; c < k_tiles; ++c) {
            noc.async_write(y, s_out, t16, {.offset_bytes = c * t16}, {.page_id = g * d_tiles + s * k_tiles + c});
        }
        noc.async_write_barrier();
    } else {
        constexpr uint32_t per = 32 / (sub_rows ? sub_rows : 32), face = 512, face_row = 32;  // bf16
        const uint32_t r0 = (g % per) * sub_rows;
        const uint32_t fo = (r0 / 16) * 2 * face + (r0 % 16) * face_row;
        const uint32_t src = y.get_read_ptr();
        for (uint32_t c = 0; c < k_tiles; ++c) {
            const uint64_t dst = s_out.get_noc_addr((g / per) * d_tiles + s * k_tiles + c);
            noc_async_write(src + c * t16, dst + fo, sub_rows * face_row);
            noc_async_write(src + c * t16 + face, dst + fo + face, sub_rows * face_row);
        }
        noc_async_write_barrier();
    }
    y.pop_front(k_tiles);
    noc_async_atomic_barrier();  // (the mix core's increment's ack in before the kernel ends)
}
