// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The relaxed greedy head's pick over the mesh (head_relax.py), NCRISC: head_pick_col.cpp's stage with the looser
// acceptance in it (ACCEPT_RELAX).  Each row of tokens' next token comes from the step's tokens (every mesh row's,
// gathered: row r's virtual user v's next is its v + U, none for the last level); where this chip holds it, its logit
// is read from the head's product (one 64-byte read a row) into the payload's third word (-BIG elsewhere).  Once the
// mesh's best (value, id) and the next token's logit are folded, the next token is the row's id where the top's logit
// less the next one's is within the gap (a device scalar; the same float subtraction and compare as the ttnn ops it
// replaces), else the argmax; the argmax ids too (`raw`).
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
    constexpr uint32_t off_tiles = get_compile_time_arg_val(7);  // the value and id tiles (then the outputs')
    constexpr uint32_t off_tok = get_compile_time_arg_val(8);    // the tokens, a 64-aligned slot a mesh row
    constexpr uint32_t off_part = get_compile_time_arg_val(9);   // a 64-byte block of the product a row of tokens
    constexpr uint32_t off_aux = get_compile_time_arg_val(10);   // the gap's 64 bytes, then the next ids, the offsets
    constexpr uint32_t users = get_compile_time_arg_val(11);     // U: a level's tokens a mesh row
    constexpr uint32_t per_row = get_compile_time_arg_val(12);   // b = (K + 1) U: the tokens a mesh row
    constexpr uint32_t wt = get_compile_time_arg_val(13);        // the product's column tiles
    constexpr uint32_t tok_stride = get_compile_time_arg_val(14);
    constexpr uint32_t trows = (rows + 31) / 32;
    constexpr auto value_args = TensorAccessorArgs<15>();
    constexpr auto id_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<id_args.next_compile_time_args_offset()>();
    constexpr auto raw_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto part_args = TensorAccessorArgs<raw_args.next_compile_time_args_offset()>();
    constexpr auto tok_args = TensorAccessorArgs<part_args.next_compile_time_args_offset()>();
    constexpr auto delta_args = TensorAccessorArgs<tok_args.next_compile_time_args_offset()>();
    constexpr auto area_args = TensorAccessorArgs<delta_args.next_compile_time_args_offset()>();
    constexpr uint32_t ring = 8, tile = 4096, face = 1024, face_row = 64;
    constexpr uint32_t NONE = 0xFFFFFFFFu;
    constexpr uint32_t NEG_BIG = 0xF149F2CAu;  // -1e30 (the ttnn form's masked logit)
    constexpr uint32_t bytes = rows * head_relax::WORDS * 4;

    size_t a = 0;
    const uint32_t value_addr = get_arg_val<uint32_t>(a++);
    const uint32_t id_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t raw_addr = get_arg_val<uint32_t>(a++);
    const uint32_t part_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tok_addr = get_arg_val<uint32_t>(a++);
    const uint32_t delta_addr = get_arg_val<uint32_t>(a++);
    const uint32_t id_base = get_arg_val<uint32_t>(a++);     // this chip's first global id
    const uint32_t valid_cols = get_arg_val<uint32_t>(a++);  // its real ids
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

    DeviceZoneScopedN("HR-COL");
    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t tiles = scr + off_tiles;
    const uint32_t tok = scr + off_tok;
    const uint32_t part = scr + off_part;
    const uint32_t aux = scr + off_aux;
    volatile tt_l1_ptr uint32_t* nx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aux + 64);
    volatile tt_l1_ptr uint32_t* at = nx + rows;  // the next logit's word in its block, or NONE
    const auto s_value = TensorAccessor(value_args, value_addr, tile);
    const auto s_id = TensorAccessor(id_args, id_addr, tile);
    const auto s_out = TensorAccessor(out_args, out_addr, tile);
    const auto s_raw = TensorAccessor(raw_args, raw_addr, tile);
    const auto s_part = TensorAccessor(part_args, part_addr, tile);
    const auto s_tok = TensorAccessor(tok_args, tok_addr, per_row * 4);
    const auto s_delta = TensorAccessor(delta_args, delta_addr, tile);
    for (uint32_t t = 0; t < trows; ++t) {
        noc_async_read(s_value.get_noc_addr(t), tiles + t * tile, tile);
        noc_async_read(s_id.get_noc_addr(t), tiles + (trows + t) * tile, tile);
    }
    for (uint32_t r = 0; r < ring; ++r) {
        noc_async_read(s_tok.get_noc_addr(r), tok + r * tok_stride, per_row * 4);
    }
    noc_async_read(s_delta.get_noc_addr(0), aux, 64);  // (the gap: the tile's first word)
    fwd.open_start();
    bwd.open_start();
    noc_async_read_barrier();
    // each row's next token, and its logit's 64-byte block where this chip holds it
    for (uint32_t m = 0; m < rows; ++m) {
        const uint32_t v = m % per_row;
        const uint32_t t = v + users < per_row
                               ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok + (m / per_row) * tok_stride)[v + users]
                               : NONE;
        nx[m] = t;
        at[m] = NONE;
        if (t != NONE && t >= id_base && t - id_base < valid_cols) {
            const uint32_t c = t - id_base, rr = m % 32, cc = c % 32;
            const uint32_t byte = (((rr / 16) * 2 + cc / 16) * 256 + (rr % 16) * 16 + cc % 16) * 4;
            noc_async_read(s_part.get_noc_addr((m / 32) * wt + c / 32, byte & ~63u), part + m * 64, 64);
            at[m] = (byte & 63) / 4;
        }
    }
    noc_async_read_barrier();
    auto col0 = [&](uint32_t r) { return (r % 32 / 16) * 2 * face + (r % 16) * face_row + (r / 32) * tile; };
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
        for (uint32_t r = 0; r < rows; ++r) {
            p[3 * r] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tiles + col0(r));
            p[3 * r + 1] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tiles + trows * tile + col0(r));
            p[3 * r + 2] = at[r] == NONE ? NEG_BIG
                                         : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(part + r * 64)[at[r]];
        }
    }
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_p)), 1);
    // the output tiles zeroed from the core's zeros region while the payloads travel (the value and id tiles are in
    // P by now; the ring's read barriers wait for these too)
    for (uint32_t off = 0; off < 2 * trows * tile; off += MEM_ZEROS_SIZE) {
        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), tiles + off, MEM_ZEROS_SIZE);
    }

    PacketHeaderPool::reset();
    Header h_f = PacketHeaderPool::allocate_header();
    Header h_b = PacketHeaderPool::allocate_header();
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
    send(fwd, h_f, r_l1, dst, sem, bytes);
    send(bwd, h_b, r_l1, dst, sem, bytes);
    noc_async_writes_flushed();
    copy_words(scr, r_l1, rows * head_relax::WORDS);
    head_relax::gather_ring(s_area, page_base, ring, me, sem_args, scr, scr + off_in, rows);

    // the relaxed ids into column 0 of the output's tiles and the argmax ids into the raw's (the rest zero)
    noc_async_read_barrier();
    const uint32_t delta_bits = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aux);
    float delta;
    __builtin_memcpy(&delta, &delta_bits, 4);
    // a gap below 0 takes nothing (the top's logit is the largest: top - ln >= 0); an equal logit's gap is +0
    const bool none = (delta_bits >> 31) != 0 && delta_bits != 0x80000000u;
    const bool zero_ok = !none;  // (0 <= delta)
    volatile tt_l1_ptr uint32_t* best = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t id = best[3 * r + 1];
        uint32_t out = id;
        if (!none && nx[r] != NONE) {
            const uint32_t tb = best[3 * r], lb = best[3 * r + 2];
            bool near;
            if (order_key(tb) == order_key(lb)) {
                near = zero_ok;
            } else {
                float top, ln;
                __builtin_memcpy(&top, &tb, 4);
                __builtin_memcpy(&ln, &lb, 4);
                near = top - ln <= delta;
            }
            if (near) {
                const float f = static_cast<float>(nx[r]);
                __builtin_memcpy(&out, &f, 4);
            }
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tiles + col0(r)) = out;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tiles + trows * tile + col0(r)) = id;
    }
    for (uint32_t t = 0; t < trows; ++t) {
        noc_async_write(tiles + t * tile, s_out.get_noc_addr(t), tile);
        noc_async_write(tiles + (trows + t) * tile, s_raw.get_noc_addr(t), tile);
    }
    fwd.close();
    bwd.close();
    noc_async_write_barrier();
}
