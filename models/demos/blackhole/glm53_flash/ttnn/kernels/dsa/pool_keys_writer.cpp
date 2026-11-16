// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A prefill chunk's pools' keys in one op (dsa_indexer.pool_keys), the writer: item idx (tile idx = 4 r + j of the
// keys / gates: pools 8 r .. 8 r + 7, feature tile j) gives rows 8 (r mod 4) .. + 7 of tile 4 (r / 4) + j of the
// output [1, 1, T / 4, 128] (bf16 tiles): row 0 of each of compute's 8 key tiles (32 bytes of face 0 and of face 1)
// copied into a 512-byte stage (the 8 rows of the upper face, then of the lower face of the output's face pair), then
// written as two 256-byte runs of the output tile (the item's rows of its 2 faces).
//   CT: CB_KEY, CB_STAGE, then the accessor of the output, then the shape key.  RT: the address, the items [i0, i1).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TILE = 2048;
constexpr uint32_t FACE = 512;
constexpr uint32_t COLS = 4;
constexpr uint32_t POOLS = 8;
}  // namespace

void kernel_main() {
    constexpr uint32_t CB_KEY = get_compile_time_arg_val(0);
    constexpr uint32_t CB_STAGE = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    static_assert(out_args.AlignedPageSize == TILE);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t i0 = get_arg_val<uint32_t>(1);
    const uint32_t i1 = get_arg_val<uint32_t>(2);
    if (i1 <= i0) {
        return;
    }
    cb_reserve_back(CB_STAGE, 1);
    const uint32_t stage = get_write_ptr(CB_STAGE);
    for (uint32_t idx = i0; idx < i1; ++idx) {
        const uint32_t r = idx / COLS, j = idx % COLS, h = r % 4;
        cb_wait_front(CB_KEY, POOLS);
        const uint32_t src = get_read_ptr(CB_KEY);
        for (uint32_t k = 0; k < POOLS; ++k) {
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + k * TILE + f * FACE);
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + f * 256 + k * 32);
                for (uint32_t w = 0; w < 8; ++w) {
                    d[w] = s[w];
                }
            }
        }
        const uint32_t page = (r / 4) * COLS + j;
        const uint32_t face = (h / 2) * 2 * FACE + (h % 2) * 256;
        noc_async_write(stage, out.get_noc_addr(page, face), 256);
        noc_async_write(stage + 256, out.get_noc_addr(page, face + FACE), 256);
        noc_async_write_barrier();
        cb_pop_front(CB_KEY, POOLS);
    }
}
