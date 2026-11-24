// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);
// AllReduceAsyncParams::compact_rows: each input core's tiles' first compact_rows rows only (faces 0 and 1, 32 bytes a
// face row), packed into whole pages of cb0 (a core's num_tiles_per_core tiles: num_tiles_per_core compact_rows / 32
// pages), core by core from the first; 0: whole tiles
constexpr uint32_t compact_rows = get_compile_time_arg_val(3);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_common_arg_val<address_t>(ttnn::ccl::AllReduceReaderCommonArgs::input);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    Noc noc_obj;
    CircularBuffer cb0(cb0_id);
    UnicastEndpoint src_ep;

    if constexpr (compact_rows != 0) {
        DeviceZoneScopedN("AR-R-READ");
        constexpr uint32_t part = compact_rows * 32;  // a face's token rows, bytes
        const uint32_t pages = num_tiles_per_core * compact_rows / 32;
        for (uint32_t c = 0; c < num_cores; ++c) {
            cb0.reserve_back(pages);
            uint32_t dst = cb0.get_write_ptr();
            for (uint32_t k = 0; k < num_tiles_per_core; ++k) {
                const uint64_t src =
                    get_noc_addr(core_noc_x[c], core_noc_y[c], tensor_address0 + k * tensor0_page_size);
                noc_async_read(src, dst, part);               // face 0
                noc_async_read(src + 512, dst + part, part);  // face 1
                dst += 2 * part;
            }
            noc_async_read_barrier();
            cb0.push_back(pages);
        }
        return;
    }
    DeviceZoneScopedN("AR-R-READ");

    // interleaved addrgen
    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        cb0.reserve_back(num_tiles_to_read_this_core);

        noc_obj.async_read(
            src_ep,
            cb0,
            num_tiles_to_read_this_core * tensor0_page_size,
            {.noc_x = core_noc_x[core_id],
             .noc_y = core_noc_y[core_id],
             .addr = tensor_address0 + shard_tile_id * tensor0_page_size},
            {});
        noc_obj.async_read_barrier();

        cb0.push_back(num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }
}
