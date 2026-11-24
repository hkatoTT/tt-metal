// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// A traced prefill's packed inputs (prefill_pack.py): this core's runs of the fields' pages.  A run's pages sit in
// consecutive 64-byte aligned slots of the packed row (one DRAM page a chip): as many slots as fit a buffer are read
// in one read into L1 (two buffers in turn, 64-byte aligned: a slot, its L1 copy and the field's page start at the
// same offset mod 64), then each page is written to the field's interleaved DRAM tensor (page p at page p), the
// writes not waited on until the buffer comes round again.  No multicast.
//   CT: cb, the most runs a core (the program cache's key), a buffer's bytes, the layout's tag (the cache's key),
//   batch (0: one slot a read).
//   RT: packed address, packed bytes, the core's runs, then per run: address, page bytes, first page, pages, the first
//   page's slot's byte offset in the packed row, slot bytes.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t buf_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr uint32_t ALIGN = 64;
    const uint32_t packed_addr = get_arg_val<uint32_t>(0);
    const uint32_t packed_bytes = get_arg_val<uint32_t>(1);
    const uint32_t runs = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<true> src{.bank_base_address = packed_addr, .page_size = packed_bytes};
    const uint64_t row = src.get_noc_addr(0);
    const uint32_t base = (get_write_ptr(cb) + ALIGN - 1) & ~(ALIGN - 1);
    uint32_t turn = 0;
    for (uint32_t r = 0; r < runs; ++r) {
        const uint32_t a = 3 + 6 * r;
        const uint32_t dst_addr = get_arg_val<uint32_t>(a);
        const uint32_t page = get_arg_val<uint32_t>(a + 1);
        const uint32_t first = get_arg_val<uint32_t>(a + 2);
        const uint32_t pages = get_arg_val<uint32_t>(a + 3);
        const uint32_t offset = get_arg_val<uint32_t>(a + 4);
        const uint32_t slot = get_arg_val<uint32_t>(a + 5);
        const InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = page};
        const uint32_t per = batch ? buf_bytes / slot : 1;  // (slots a read: the host keeps slot <= buf_bytes)
        for (uint32_t p = 0; p < pages; p += per, turn ^= 1) {
            const uint32_t n = pages - p < per ? pages - p : per;
            const uint32_t l1 = base + turn * buf_bytes;
            noc_async_writes_flushed();  // (the buffer's writes two reads ago have left it)
            noc_async_read(row + offset + p * slot, l1, batch ? n * slot : page);
            noc_async_read_barrier();
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_write(l1 + i * slot, dst.get_noc_addr(first + p + i), page);
            }
        }
    }
    noc_async_write_barrier();
}
