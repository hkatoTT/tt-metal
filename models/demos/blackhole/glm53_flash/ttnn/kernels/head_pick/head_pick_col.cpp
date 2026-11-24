// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The greedy head's pick over the mesh (head_pick.py), NCRISC: this chip's top logit and its id per row of tokens
// (column 0 of the value and id tiles) into the payload P, handed to the row kernel (SEM_P); then, once it has the
// row's best R (SEM_R), R to the mesh column's seven other chips (a 4-hop multicast one way and a 3-hop one the other:
// their landing page (parity, this chip's row)), the seven others' R folded into the mesh's best, and its ids into
// column 0 of the output's tiles (the rest zero).
#include "head_pick_common.hpp"

using namespace head_pick;

void kernel_main() {
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t sem_p = get_compile_time_arg_val(3);
    constexpr uint32_t sem_r = get_compile_time_arg_val(4);
    constexpr uint32_t off_r = get_compile_time_arg_val(5);
    constexpr uint32_t off_in = get_compile_time_arg_val(6);
    constexpr uint32_t off_tiles = get_compile_time_arg_val(7);  // the input's and output's tiles
    constexpr uint32_t trows = (rows + 31) / 32;
    constexpr auto value_args = TensorAccessorArgs<8>();
    constexpr auto id_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<id_args.next_compile_time_args_offset()>();
    constexpr auto area_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t ring = 8, tile = 4096, face = 1024, face_row = 64;

    size_t a = 0;
    const uint32_t value_addr = get_arg_val<uint32_t>(a++);
    const uint32_t id_addr = get_arg_val<uint32_t>(a++);
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

    DeviceZoneScopedN("HP-COL");
    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t tiles = scr + off_tiles;
    const auto s_value = TensorAccessor(value_args, value_addr, tile);
    const auto s_id = TensorAccessor(id_args, id_addr, tile);
    const auto s_out = TensorAccessor(out_args, out_addr, tile);
    for (uint32_t t = 0; t < trows; ++t) {
        noc_async_read(s_value.get_noc_addr(t), tiles + t * tile, tile);
        noc_async_read(s_id.get_noc_addr(t), tiles + (trows + t) * tile, tile);
    }
    fwd.open_start();
    bwd.open_start();
    noc_async_read_barrier();
    auto col0 = [&](uint32_t r) { return (r % 32 / 16) * 2 * face + (r % 16) * face_row + (r / 32) * tile; };
    {
        volatile tt_l1_ptr float* p = reinterpret_cast<volatile tt_l1_ptr float*>(scr);
        for (uint32_t r = 0; r < rows; ++r) {
            p[2 * r] = *reinterpret_cast<volatile tt_l1_ptr float*>(tiles + col0(r));
            p[2 * r + 1] = *reinterpret_cast<volatile tt_l1_ptr float*>(tiles + trows * tile + col0(r));
        }
    }
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_p)), 1);

    PacketHeaderPool::reset();
    Header h_f = PacketHeaderPool::allocate_header();  // 4 hops the forward way
    Header h_b = PacketHeaderPool::allocate_header();  // 3 hops the backward way
    uint16_t hf[4] = {0, 0, 0, 0}, hb[4] = {0, 0, 0, 0};
    hf[get_next_hop_router_direction(mesh_id, fwd_chip)] = ring / 2;
    hb[get_next_hop_router_direction(mesh_id, bwd_chip)] = ring - 1 - ring / 2;
    fabric_set_mcast_route(h_f, fwd_chip, mesh_id, hf[0], hf[1], hf[2], hf[3]);
    fabric_set_mcast_route(h_b, bwd_chip, mesh_id, hb[0], hb[1], hb[2], hb[3]);
    fwd.open_finish();
    bwd.open_finish();

    const auto s_area = TensorAccessor(area_args, area_addr, page_bytes);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_r)), 1);
    const uint32_t r_l1 = scr + off_r;
    const uint64_t dst = s_area.get_noc_addr(page_base + me);
    const uint64_t sem = safe_get_noc_addr(wx, wy, dst_sem, 0);
    send(fwd, h_f, r_l1, dst, sem, rows * 8);
    send(bwd, h_b, r_l1, dst, sem, rows * 8);
    noc_async_writes_flushed();
    // the mesh's best into P's place (P is sent and folded by now)
    copy_words(scr, r_l1, rows * 2);
    gather_ring(s_area, page_base, ring, me, sem_args, scr, scr + off_in, rows);

    // the ids into column 0 of the output's tiles, the rest zero
    for (uint32_t t = 0; t < trows; ++t) {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tiles + t * tile);
        for (uint32_t i = 0; i < tile / 4; ++i) {
            z[i] = 0;
        }
    }
    volatile tt_l1_ptr float* best = reinterpret_cast<volatile tt_l1_ptr float*>(scr);
    for (uint32_t r = 0; r < rows; ++r) {
        *reinterpret_cast<volatile tt_l1_ptr float*>(tiles + col0(r)) = best[2 * r + 1];
    }
    for (uint32_t t = 0; t < trows; ++t) {
        noc_async_write(tiles + t * tile, s_out.get_noc_addr(t), tile);
    }
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
}
