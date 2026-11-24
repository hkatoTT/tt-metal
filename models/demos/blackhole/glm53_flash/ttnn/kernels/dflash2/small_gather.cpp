// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A small row-major gather over the mesh column (dflash2_ops.small_gather), one
// core a chip, one-shot over the 2D fabric (head_pick's column stage, head_pick_common.hpp): this chip's payload (its
// `rows` pages of the input, `bytes` in all, at most a landing page) to the column's seven other chips (a 4-hop
// multicast one way, a 3-hop one the other: their landing page (parity, this chip's row) of head_pick's "head_col"
// family, each packet a write fused with an increment of the receive semaphore for this chip), and into this chip's
// output at pages me rows ..; the seven others' payloads read from the landing pages as they land and written to their
// rows of the output.  A copy: the same bytes as ttnn.all_gather(dim=2, cluster_axis=0).
#include "../head_pick/head_pick_common.hpp"

using namespace head_pick;

void kernel_main() {
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);    // the input's pages (row-major rows)
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);  // a page's bytes
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);  // a landing page
    constexpr uint32_t send_bytes = get_compile_time_arg_val(4);  // the payload sent (rows x row_bytes, padded to 64)
    constexpr auto in_args = TensorAccessorArgs<5>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto area_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t ring = 8;

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t area_addr = get_arg_val<uint32_t>(a++);
    const uint32_t page_base = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);  // this chip's row
    const uint32_t dst_sem = get_arg_val<uint32_t>(a++);
    const uint32_t wx = get_arg_val<uint32_t>(a++);
    const uint32_t wy = get_arg_val<uint32_t>(a++);
    const uint32_t sem_args = a;
    a += ring;
    auto fwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);
    auto bwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    DeviceZoneScopedN("SMALL-GATHER");
    // the payload, then one landed payload a source row (64-byte aligned: reads under 64 bytes keep their offset
    // mod 64)
    const uint32_t scr = (get_write_ptr(cb_scr) + 63) & ~63u;
    const auto s_in = TensorAccessor(in_args, in_addr, row_bytes);
    const auto s_out = TensorAccessor(out_args, out_addr, row_bytes);
    const auto s_area = TensorAccessor(area_args, area_addr, page_bytes);
    for (uint32_t j = 0; j < rows; ++j) {
        noc_async_read(s_in.get_noc_addr(j), scr + j * row_bytes, row_bytes);
    }
    fwd.open_start();
    bwd.open_start();
    PacketHeaderPool::reset();
    Header h_f = PacketHeaderPool::allocate_header();  // 4 hops the forward way
    Header h_b = PacketHeaderPool::allocate_header();  // 3 hops the backward way
    uint16_t hf[4] = {0, 0, 0, 0}, hb[4] = {0, 0, 0, 0};
    hf[get_next_hop_router_direction(mesh_id, fwd_chip)] = ring / 2;
    hb[get_next_hop_router_direction(mesh_id, bwd_chip)] = ring - 1 - ring / 2;
    fabric_set_mcast_route(h_f, fwd_chip, mesh_id, hf[0], hf[1], hf[2], hf[3]);
    fabric_set_mcast_route(h_b, bwd_chip, mesh_id, hb[0], hb[1], hb[2], hb[3]);
    noc_async_read_barrier();
    fwd.open_finish();
    bwd.open_finish();

    const uint64_t dst = s_area.get_noc_addr(page_base + me);
    const uint64_t sem = safe_get_noc_addr(wx, wy, dst_sem, 0);
    send(fwd, h_f, scr, dst, sem, send_bytes);
    send(bwd, h_b, scr, dst, sem, send_bytes);
    for (uint32_t j = 0; j < rows; ++j) {  // this chip's own rows
        noc_async_write(scr + j * row_bytes, s_out.get_noc_addr(me * rows + j), row_bytes);
    }
    noc_async_writes_flushed();

    uint32_t pending = ((1u << ring) - 1) & ~(1u << me);
    while (pending != 0) {
        invalidate_l1_cache();
        for (uint32_t left = pending; left != 0; left &= left - 1) {
            const uint32_t k = __builtin_ctz(left);
            auto* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(sem_args + k));
            if (*s == 0) {
                continue;
            }
            noc_semaphore_set(s, 0);  // (this parity's next writes come two calls of the family later)
            pending &= ~(1u << k);
            const uint32_t slot = scr + (k + 1) * page_bytes;
            noc_async_read(s_area.get_noc_addr(page_base + k), slot, send_bytes);
            noc_async_read_barrier();
            for (uint32_t j = 0; j < rows; ++j) {
                noc_async_write(slot + j * row_bytes, s_out.get_noc_addr(k * rows + j), row_bytes);
            }
        }
    }
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
}
