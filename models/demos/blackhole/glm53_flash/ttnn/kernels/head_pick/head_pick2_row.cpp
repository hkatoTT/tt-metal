// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The greedy head's pick over the mesh, v2 (head_pick.py, the default; head_pick2_common.hpp), BRISC: the mesh row's
// stage and half the folds.  Once the column kernel has put this chip's payload P (a 64-bit key a row of tokens) in
// the scratch (SEM_P), P to the row's three other chips (a 2-hop multicast one way, a unicast the other: their DRAM
// landing page (parity, this chip's column), bumping their receive semaphore for this chip); the three others' P read
// into their row slots (SEM_RS to the column kernel), then rows [half, rows) of the row's best R (SEM_RB).  After the
// column kernel has the mesh column's seven R in their slots (SEM_CS): rows [half, rows) of the mesh's best into
// column 0 of the id tiles (SEM_DB).
#include "head_pick2_common.hpp"

using namespace head_pick;
using namespace head_pick2;

void kernel_main() {
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);  // of tokens: the payloads' entries
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t sem_p = get_compile_time_arg_val(3);
    constexpr uint32_t sem_rs = get_compile_time_arg_val(4);
    constexpr uint32_t sem_rb = get_compile_time_arg_val(5);
    constexpr uint32_t sem_cs = get_compile_time_arg_val(6);
    constexpr uint32_t sem_db = get_compile_time_arg_val(7);
    constexpr uint32_t off_r = get_compile_time_arg_val(8);   // the scratch: P at 0, R, the row slots, the column
    constexpr uint32_t off_rs = get_compile_time_arg_val(9);  // slots, the input's tiles (value, then id)
    constexpr uint32_t off_cs = get_compile_time_arg_val(10);
    constexpr uint32_t off_tiles = get_compile_time_arg_val(11);
    constexpr auto area_args = TensorAccessorArgs<12>();
    constexpr uint32_t ring = 4, col_ring = 8;
    constexpr uint32_t trows = (rows + 31) / 32, half = rows / 2;

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

    DeviceZoneScopedN("HP2-ROW");
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
    const auto s_area = TensorAccessor(area_args, area_addr, page_bytes);
    local_wait(sem_p);
    const uint64_t dst = s_area.get_noc_addr(page_base + me);
    const uint64_t sem = safe_get_noc_addr(wx, wy, dst_sem, 0);
    send(fwd, h_far, scr, dst, sem, rows * 8);
    send(bwd, h_near, scr, dst, sem, rows * 8);
    noc_async_writes_flushed();
    land_ring(s_area, page_base, ring, me, sem_args, scr + off_rs, page_bytes, rows * 8);
    local_set(sem_rs);
    fold_into(scr + off_r, scr, scr + off_rs, page_bytes, ring, me, half, rows);
    local_set(sem_rb);
    fwd.close();
    bwd.close();
    noc_async_write_barrier();

    local_wait(sem_cs);
    fold_ids(scr + off_tiles + trows * 4096, scr + off_r, scr + off_cs, page_bytes, col_ring, my_row, half, rows);
    local_set(sem_db);
}
