// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "mp_common.hpp"
#include "mp_noc1.hpp"

// The prefill MoE's combine, the writer (mp_common.hpp): per item, the most pairs any of its 32 tokens has (the
// slots, from the reader through CB_SL) to the compute (mailbox), then the summed block's 16 tiles to out's tile row,
// tile columns 16 block ...
void kernel_main() {
    using namespace moe_prefill;
    uint32_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tpair_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_items = get_arg_val<uint32_t>(a++);

    constexpr uint32_t out_tile = get_compile_time_arg_val(0);  // bytes of an out tile (bf16)
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr auto p_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr uint32_t nocs = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 1);  // DNCB_NOCS
    // CB_AHEAD: the next item's slots to the compute before this item's tiles are written (with CB_S holding two
    // items: the reader reads item i + 1's Y rows while the compute sums item i); 0: after
    constexpr bool ahead = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 2) != 0;
    // the tiles an item (a block of 32 x PIECES columns): 16, or 32 (Y_BFP8's blocks of 1,024)
    constexpr uint32_t PIECES = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 3);
    // out's tile rows in batches (col_row_rs): N = ROWS mesh rows' TT tile rows
    // each, written batch-major ([B, 1, ROWS BT, D], batch b holding each mesh row's tile rows b BT ..); BT = TT: as
    // they come
    constexpr uint32_t ROWS = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t TT = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 5);
    constexpr uint32_t BT = get_compile_time_arg_val(p_args.next_compile_time_args_offset() + 6);
    static_assert(BT >= 1 && TT % BT == 0, "combine: batches");
    auto out_row = [](uint32_t tr) -> uint32_t {
        if constexpr (BT == TT) {
            return tr;
        }
        const uint32_t d = tr / TT, r = tr % TT;
        return (r / BT) * (ROWS * BT) + d * BT + r % BT;
    };
    const auto s_out = TensorAccessor(o_args, out_addr, out_tile);
    (void)tpair_addr;  // (the reader reads the pair rows)
    if constexpr (nocs == 2) {  // reads nothing: NoC 1 to the reader from the start
        hand_over_noc1();
    }

    auto mail = [&]() {  // the next item's slots, from the reader (which reads the tokens' pair rows), to the compute
        cb_wait_front(CB_SL, 1);
        const uint32_t slots = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_SL));
        cb_pop_front(CB_SL, 1);
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, slots);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, slots);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, slots);
    };
    if (ahead && n_items > 0) {
        mail();
    }
    for (uint32_t i = 0; i < n_items; ++i) {
        const uint32_t item = get_arg_val<uint32_t>(a + i);
        const uint32_t tr = out_row(item / (128 / PIECES)), blk = item % (128 / PIECES);
        if (!ahead || i + 1 < n_items) {  // (ahead: item i + 1's, at most two in the mailboxes)
            mail();
        }
        cb_wait_front(CB_T, PIECES);
        const uint32_t src = get_read_ptr(CB_T);
        for (uint32_t i = 0; i < PIECES; ++i) {  // tiles (tr, PIECES blk + i) of out's 128 a tile row
            noc_async_write(src + i * out_tile, s_out.get_noc_addr(tr * 128 + blk * PIECES + i), out_tile);
        }
        noc_async_writes_flushed();
        cb_pop_front(CB_T, PIECES);
    }
    noc_async_write_barrier();
}
