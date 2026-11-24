// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The greedy head's pick for the own mesh row's tokens, v3 (head_pick.py `own`; head_pick3_common.hpp), NCRISC: this
// chip's top logit and its id per row of tokens (column 0 of the value and id tiles) into the payload P as keys (all M
// rows), P to the mesh column's seven other chips (a 4-hop multicast one way and a 3-hop one the other: their landing
// page (parity, this chip's row)); the seven others' slices of this chip's mesh row (t rows at 8 t r) read into the
// column slots (SEM_CS to the row kernel), rows [0, t / 2) of Rc (SEM_RC); once the row kernel has the row's three Rc
// in their slots (SEM_RS), rows [0, t / 2) of the mesh's best into column 0 of the id tiles, and once the row kernel
// has its half there (SEM_DB), the id tiles holding the own rows to the output.
#include "head_pick3_common.hpp"

using namespace head_pick;
using namespace head_pick2;
using namespace head_pick3;

void kernel_main() {
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);  // M: the payloads' entries
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t sem_cs = get_compile_time_arg_val(3);
    constexpr uint32_t sem_rc = get_compile_time_arg_val(4);
    constexpr uint32_t sem_rs = get_compile_time_arg_val(5);
    constexpr uint32_t sem_db = get_compile_time_arg_val(6);
    constexpr uint32_t off_r = get_compile_time_arg_val(7);  // the scratch: P at 0, Rc, the row slots, the column
    constexpr uint32_t off_rs = get_compile_time_arg_val(8);  // slots, the input's tiles (value, then id)
    constexpr uint32_t off_cs = get_compile_time_arg_val(9);
    constexpr uint32_t off_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(11);
    constexpr auto value_args = TensorAccessorArgs<12>();
    constexpr auto id_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<id_args.next_compile_time_args_offset()>();
    constexpr auto area_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t ring = 8, row_ring = 4, tile = 4096;
    constexpr uint32_t trows = (rows + 31) / 32, t = rows / ring, half = t / 2;
    static_assert(rows % 64 == 0 && t * 8 <= slot_bytes, "head pick v3: 8 rows a mesh row at least");

    size_t a = 0;
    const uint32_t value_addr = get_arg_val<uint32_t>(a++);
    const uint32_t id_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t mesh_id = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_chip = get_arg_val<uint32_t>(a++);
    const uint32_t area_addr = get_arg_val<uint32_t>(a++);
    const uint32_t page_base = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);  // this chip's mesh row
    const uint32_t dst_sem = get_arg_val<uint32_t>(a++);
    const uint32_t wx = get_arg_val<uint32_t>(a++);
    const uint32_t wy = get_arg_val<uint32_t>(a++);
    const uint32_t my_col = get_arg_val<uint32_t>(a++);  // this chip's column (the row stage's slot skipped)
    const uint32_t sem_args = a;
    a += ring;
    auto fwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);
    auto bwd = Conn::build_from_args<ProgrammableCoreType::TENSIX>(a);

    DeviceZoneScopedN("HP3-COL");
    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t tiles = scr + off_tiles;
    const uint32_t id_tiles = tiles + trows * tile;
    const auto s_value = TensorAccessor(value_args, value_addr, tile);
    const auto s_id = TensorAccessor(id_args, id_addr, tile);
    const auto s_out = TensorAccessor(out_args, out_addr, tile);
    for (uint32_t i = 0; i < trows; ++i) {
        noc_async_read(s_value.get_noc_addr(i), tiles + i * tile, tile);
        noc_async_read(s_id.get_noc_addr(i), id_tiles + i * tile, tile);
    }
    fwd.open_start();
    bwd.open_start();
    noc_async_read_barrier();
    compiler_fence();
    {
        uint32_t* p = reinterpret_cast<uint32_t*>(scr);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t at = col0(r);
            p[2 * r] = order_key(*reinterpret_cast<const uint32_t*>(tiles + at));
            p[2 * r + 1] = ~*reinterpret_cast<const uint32_t*>(id_tiles + at);
        }
    }

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
    const uint64_t dst = s_area.get_noc_addr(page_base + me);
    const uint64_t sem = safe_get_noc_addr(wx, wy, dst_sem, 0);
    send(fwd, h_f, scr, dst, sem, rows * 8);
    send(bwd, h_b, scr, dst, sem, rows * 8);
    noc_async_writes_flushed();
    land_slices(s_area, page_base, ring, me, sem_args, scr + off_cs, slot_bytes, me * t * 8, t * 8);
    local_set(sem_cs);
    // Rc: rows [0, half) here, the rest by the row kernel
    fold_slice(scr + off_r, scr + me * t * 8, scr + off_cs, slot_bytes, ring, me, 0, half);
    local_set(sem_rc);

    // the mesh's best of the own rows: [0, half) here, the rest by the row kernel
    local_wait(sem_rs);
    fold_slice_ids(id_tiles, me * t, scr + off_r, scr + off_rs, slot_bytes, row_ring, my_col, 0, half);
    local_wait(sem_db);

    for (uint32_t i = (me * t) / 32; i <= (me * t + t - 1) / 32; ++i) {
        noc_async_write(id_tiles + i * tile, s_out.get_noc_addr(i), tile);
    }
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
