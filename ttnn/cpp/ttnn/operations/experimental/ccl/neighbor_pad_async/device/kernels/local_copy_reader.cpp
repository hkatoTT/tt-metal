// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;

constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr uint32_t stick_size = get_compile_time_arg_val(1);

void kernel_main() {
    // Common runtime args (multicast once per kernel, not unicast per core)
    const address_t input_tensor_address = get_common_arg_val<address_t>(0);
    // CRTA[1] = output_addr (unused by reader, reserved for consistency with writer)
    const uint32_t stick_start_id = get_common_arg_val<uint32_t>(2);
    const uint32_t input_halo_dim_size = get_common_arg_val<uint32_t>(3);
    const uint32_t num_sticks_to_read = get_common_arg_val<uint32_t>(4);
    const uint32_t num_sticks_per_halo_dim = get_common_arg_val<uint32_t>(5);

    // Per-core runtime args (only work distribution — truly unique per core)
    uint32_t arg_idx = 0;
    const uint32_t total_rows_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rows_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t front_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t front_count = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto src_args = TensorAccessorArgs<2>();
    uint32_t read_size = stick_size;
    const auto src_accessor = TensorAccessor(src_args, input_tensor_address);

    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);

#ifdef NP_FRONT_TENSOR
    // Phase A: this core's slice of the already padded front frames, page for page (the writer puts them at the
    // same output pages, which the zero-fill covers otherwise).
    {
        constexpr auto front_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
        const auto front_accessor = TensorAccessor(front_args, get_common_arg_val<address_t>(6));
        constexpr uint32_t batch = (HK_LC_BATCH > 1) ? HK_LC_BATCH : 1;
        for (uint32_t base = 0; base < front_count; base += batch) {
            const uint32_t n = (front_count - base) < batch ? (front_count - base) : batch;
            cb_output.reserve_back(batch);
            uint32_t l1_addr = cb_output.get_write_ptr();
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_read(front_accessor.get_noc_addr(front_start + base + i), l1_addr, read_size);
                l1_addr += stick_size;
            }
            noc_async_read_barrier();
            cb_output.push_back(batch);
        }
    }
#else
    (void)front_start;
    (void)front_count;
#endif

    for (uint32_t s = 0; s < rows_count; s++) {
        const uint32_t linear_row = total_rows_start + s;  // [0 .. outer_dim_size*input_halo_dim_size)
        const uint32_t outer_idx = linear_row / input_halo_dim_size;
        const uint32_t t = linear_row % input_halo_dim_size;
        const uint32_t outer_dim_offset = outer_idx * (num_sticks_per_halo_dim * input_halo_dim_size);

        uint32_t src_stick_id = t * num_sticks_per_halo_dim + stick_start_id + outer_dim_offset;
#if defined(HK_LC_BATCH) && HK_LC_BATCH > 1
        // EXPERIMENT (hkato): HK_LC_BATCH sticks per CB transaction, one barrier per batch
        for (uint32_t base = 0; base < num_sticks_to_read; base += HK_LC_BATCH) {
            const uint32_t n = (num_sticks_to_read - base) < HK_LC_BATCH ? (num_sticks_to_read - base) : HK_LC_BATCH;
            cb_output.reserve_back(HK_LC_BATCH);
            uint32_t l1_addr = cb_output.get_write_ptr();
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_read(src_accessor.get_noc_addr(src_stick_id), l1_addr, read_size);
                src_stick_id++;
                l1_addr += stick_size;
            }
            noc_async_read_barrier();
            cb_output.push_back(HK_LC_BATCH);
        }
#else
        for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
            cb_output.reserve_back(1);
            noc_obj.async_read(src_accessor, cb_output, read_size, {.page_id = src_stick_id}, {});
            src_stick_id++;
            noc_obj.async_read_barrier();
            cb_output.push_back(1);
        }
#endif
    }
}
