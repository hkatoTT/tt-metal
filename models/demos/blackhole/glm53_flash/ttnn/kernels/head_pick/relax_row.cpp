// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The relaxed greedy head's pick over the mesh (head_relax.py), BRISC: the mesh row's stage, as head_pick_row.cpp with
// the payloads of three words a row (value, id, the next token's logit).
#include "relax_common.hpp"

using namespace head_pick;

void kernel_main() {
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t sem_p = get_compile_time_arg_val(3);
    constexpr uint32_t sem_r = get_compile_time_arg_val(4);
    constexpr uint32_t off_r = get_compile_time_arg_val(5);
    constexpr uint32_t off_in = get_compile_time_arg_val(6);
    constexpr auto area_args = TensorAccessorArgs<7>();
    constexpr uint32_t ring = 4;
    constexpr uint32_t bytes = rows * head_relax::WORDS * 4;

    size_t a = 0;
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t area_addr = get_arg_val<uint32_t>(a++);
    const uint32_t page_base = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t dst_sem = get_arg_val<uint32_t>(a++);
    const uint32_t wx = get_arg_val<uint32_t>(a++);
    const uint32_t wy = get_arg_val<uint32_t>(a++);
    const uint32_t sem_args = a;
    a += ring;
    auto fwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);
    auto bwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    DeviceZoneScopedN("HR-ROW");
    fwd.open_start();
    bwd.open_start();
    PacketHeaderPool::reset();
    Header h_far = PacketHeaderPool::allocate_header();
    Header h_near = PacketHeaderPool::allocate_header();
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
    send(fwd, h_far, scr, dst, sem, bytes);
    send(bwd, h_near, scr, dst, sem, bytes);
    noc_async_writes_flushed();
    copy_words(scr + off_r, scr, rows * head_relax::WORDS);
    head_relax::gather_ring(s_area, page_base, ring, me, sem_args, scr + off_r, scr + off_in, rows);
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_r)), 1);
}
