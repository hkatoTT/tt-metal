// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The greedy head's pick over the mesh (head_pick.py), BRISC: the mesh row's stage.  Once the column kernel has put
// this chip's payload P (per row of tokens its top logit and the global id, as two floats) in the scratch, P to the
// row's three other chips (a 2-hop multicast one way, a unicast the other: their DRAM landing page (parity, this chip's
// column), bumping their receive semaphore for this chip), then the three others' P folded into R (the row's best),
// and R handed back to the column kernel (SEM_R).
#include "head_pick_common.hpp"

using namespace head_pick;

void kernel_main() {
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);  // of tokens: the payloads' entries
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t sem_p = get_compile_time_arg_val(3);
    constexpr uint32_t sem_r = get_compile_time_arg_val(4);
    constexpr uint32_t off_r = get_compile_time_arg_val(5);   // the scratch: P at 0, R at off_r, one landed at off_in
    constexpr uint32_t off_in = get_compile_time_arg_val(6);
    constexpr auto area_args = TensorAccessorArgs<7>();
    constexpr uint32_t ring = 4;

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
    const uint32_t sem_args = a;  // this parity's receive semaphores, one for each chip of the row
    a += ring;
    auto fwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);
    auto bwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    DeviceZoneScopedN("HP-ROW");
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
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_p)), 1);
    const uint64_t dst = s_area.get_noc_addr(page_base + me);
    const uint64_t sem = safe_get_noc_addr(wx, wy, dst_sem, 0);
    send(fwd, h_far, scr, dst, sem, rows * 8);
    send(bwd, h_near, scr, dst, sem, rows * 8);
    noc_async_writes_flushed();
    copy_words(scr + off_r, scr, rows * 2);
    gather_ring(s_area, page_base, ring, me, sem_args, scr + off_r, scr + off_in, rows);
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_r)), 1);
}
