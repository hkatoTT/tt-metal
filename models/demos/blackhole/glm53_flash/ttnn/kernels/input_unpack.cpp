// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// A decode step's packed inputs (input_pack.py): the packed row (one DRAM page) read into L1 at a 64-byte aligned
// address, then each field's pages written from their 64-byte slots to the field's own interleaved DRAM tensor (page p
// at page p; the slot and the page start at the same offset mod 64).  No multicast.
//   CT: cb, the fields' count.  RT: packed address, packed bytes, the count, then per field: address, page bytes, pages,
//   its first slot's byte offset in the packed row.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t n_fields = get_compile_time_arg_val(1);
    constexpr uint32_t SLOT = 64;
    const uint32_t packed_addr = get_arg_val<uint32_t>(0);
    const uint32_t packed_bytes = get_arg_val<uint32_t>(1);

    const uint32_t l1 = (get_write_ptr(cb) + SLOT - 1) & ~(SLOT - 1);
    const InterleavedAddrGen<true> src{.bank_base_address = packed_addr, .page_size = packed_bytes};
    noc_async_read(src.get_noc_addr(0), l1, packed_bytes);
    noc_async_read_barrier();
    for (uint32_t f = 0; f < n_fields; ++f) {
        const uint32_t base = 3 + 4 * f;
        const uint32_t dst_addr = get_arg_val<uint32_t>(base);
        const uint32_t page = get_arg_val<uint32_t>(base + 1);
        const uint32_t pages = get_arg_val<uint32_t>(base + 2);
        const uint32_t offset = get_arg_val<uint32_t>(base + 3);
        const uint32_t slot = (page + SLOT - 1) & ~(SLOT - 1);
        const InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = page};
        for (uint32_t p = 0; p < pages; ++p) {
            noc_async_write(l1 + offset + p * slot, dst.get_noc_addr(p), page);
        }
    }
    noc_async_write_barrier();
}
