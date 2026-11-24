// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The greedy head's pick for the own mesh row's tokens, v3 (head_pick.py `own`; head_pick3_common.hpp), BRISC: once
// the column kernel has the column's seven slices in their slots (SEM_CS), rows [t / 2, t) of Rc; with the column
// kernel's half too (SEM_RC), Rc to the row's three other chips (a 2-hop multicast one way, a unicast the other: their
// DRAM landing page (parity, this chip's column), bumping their receive semaphore for this chip); the three others' Rc
// read into the row slots (SEM_RS to the column kernel), then rows [t / 2, t) of the mesh's best into column 0 of the
// id tiles (SEM_DB).
#include "head_pick3_common.hpp"

using namespace head_pick;
using namespace head_pick2;
using namespace head_pick3;

void kernel_main() {
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t sem_cs = get_compile_time_arg_val(3);
    constexpr uint32_t sem_rc = get_compile_time_arg_val(4);
    constexpr uint32_t sem_rs = get_compile_time_arg_val(5);
    constexpr uint32_t sem_db = get_compile_time_arg_val(6);
    constexpr uint32_t off_r = get_compile_time_arg_val(7);
    constexpr uint32_t off_rs = get_compile_time_arg_val(8);
    constexpr uint32_t off_cs = get_compile_time_arg_val(9);
    constexpr uint32_t off_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(11);
    constexpr auto area_args = TensorAccessorArgs<12>();
    constexpr uint32_t ring = 4, col_ring = 8;
    constexpr uint32_t trows = (rows + 31) / 32, t = rows / col_ring, half = t / 2;
    static_assert(rows % 64 == 0 && t * 8 <= slot_bytes, "head pick v3: 8 rows a mesh row at least");

    size_t a = 0;
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t area_addr = get_arg_val<uint32_t>(a++);
    const uint32_t page_base = get_arg_val<uint32_t>(a++);  // (parity, chip 0)'s page: + the chip's place
    const uint32_t me = get_arg_val<uint32_t>(a++);          // this chip's column
    const uint32_t dst_sem = get_arg_val<uint32_t>(a++);     // the peers' receive semaphore for this chip (parity)
    const uint32_t wx = get_arg_val<uint32_t>(a++);          // this op's core (the same on every chip)
    const uint32_t wy = get_arg_val<uint32_t>(a++);
    const uint32_t my_row = get_arg_val<uint32_t>(a++);      // this chip's mesh row (the column stage's slot skipped)
    const uint32_t sem_args = a;  // this parity's receive semaphores, one for each chip of the row
    a += ring;
    auto fwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);
    auto bwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    DeviceZoneScopedN("HP3-ROW");
    fwd.open_start();
    bwd.open_start();
    PacketHeaderPool::reset();
    Header h_far = PacketHeaderPool::allocate_header();   // 2 hops the forward way
    Header h_near = PacketHeaderPool::allocate_header();  // the backward neighbour
    uint16_t hops[4] = {0, 0, 0, 0};
    hops[get_next_hop_router_direction(mesh_id, fwd_chip)] = 2;
    fabric_set_mcast_route(h_far, fwd_chip, mesh_id, hops[0], hops[1], hops[2], hops[3]);
    fabric_set_unicast_route(h_near, bwd_chip, mesh_id);
    fwd.open_finish();
    bwd.open_finish();

    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t id_tiles = scr + off_tiles + trows * 4096;
    const auto s_area = TensorAccessor(area_args, area_addr, page_bytes);
    local_wait(sem_cs);
    fold_slice(scr + off_r, scr + my_row * t * 8, scr + off_cs, slot_bytes, col_ring, my_row, half, t);
    local_wait(sem_rc);
    const uint64_t dst = s_area.get_noc_addr(page_base + me);
    const uint64_t sem = safe_get_noc_addr(wx, wy, dst_sem, 0);
    send(fwd, h_far, scr + off_r, dst, sem, t * 8);
    send(bwd, h_near, scr + off_r, dst, sem, t * 8);
    noc_async_writes_flushed();
    land_slices(s_area, page_base, ring, me, sem_args, scr + off_rs, slot_bytes, 0, t * 8);
    local_set(sem_rs);
    fold_slice_ids(id_tiles, my_row * t, scr + off_r, scr + off_rs, slot_bytes, ring, me, half, t);
    local_set(sem_db);
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
