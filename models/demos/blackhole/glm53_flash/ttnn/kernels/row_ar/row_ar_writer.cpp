// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row all-reduce (row_all_reduce.py), a worker core's writer (BRISC): zeros into the first `split` of its staging
// tiles (this program's circular buffer) while the slices travel, then those output tiles (the reader zeroes and puts
// the rest, row_ar_reader.cpp).
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "row_ar_common.hpp"
#include "../oneshot_util.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t seg = get_compile_time_arg_val(1);   // bytes of a face's rows in the compacted slice
    constexpr uint32_t rows = get_compile_time_arg_val(2);  // t: the rows kept
    constexpr uint32_t pages = get_compile_time_arg_val(3);  // the compute's output pages
    constexpr uint32_t cb_staging = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile0 = get_arg_val<uint32_t>(1);
    const uint32_t split = get_arg_val<uint32_t>(2);

    const uint32_t staging = get_write_ptr(cb_staging);
    oneshot_zero(staging, split * 2048);
    const auto s_out = TensorAccessor(out_args, out_addr, 2048);
    cb_wait_front(cb_out, pages);
    DeviceZoneScopedN("AR_WRITE");
    row_ar_put_tiles<seg, rows>(s_out, get_read_ptr(cb_out), staging, tile0, 0, split);
}
