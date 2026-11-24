// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DSA decode's sparse attention split over cores (dsa_indexer.sparse_split): the writer (from ttnn's
// sparse_sdpa_writer.cpp).  Builds compute's persistent tiles once (the reduce scaler, the column identity for the
// row sums, the all -inf row), gathers its half of each K chunk on its NoC (cb_kreq / cb_kack from the reader), then
// writes the unit's partial (split_common.hpp: the unnormalized out tiles, the row sums, the row maxima) as tiles
// to the partials tensor.  An EMPTY request: acknowledged, nothing written (the combine skips that split).
//   CT: V_DHT, K_DIM, G, S, CB_PART, CB_SCALE, CB_COL_IDENTITY, CB_NEGINF, CB_IDX, CB_KREQ, CB_KACK, KV_BYTES, then
//       the accessors of the partials (tiles) and kv [1, 1, T, K_DIM] (row-major), WRITE_EMPTY, CB_FILL, SQT
//   RT: the partials' and kv's addresses, the first unit and the count of this core's units (each in turn), the
//       partials' unit of this call's unit 0

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_gather.hpp"
#include "split_common.hpp"

constexpr uint32_t one_bf16_packed = 0x3F803F80u;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t vDHt = get_compile_time_arg_val(0);
    constexpr uint32_t k_dim = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t cb_part = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(5);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(6);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(7);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(8);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(9);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(10);
    constexpr uint32_t kv_bytes = get_compile_time_arg_val(11);  // the latent cache's datum: 2 (bf16) or 1 (fp8 e4m3)
    constexpr auto part_args = TensorAccessorArgs<12>();
    constexpr auto kv_args = TensorAccessorArgs<part_args.next_compile_time_args_offset()>();
    // WRITE_EMPTY (split latents: the row's chips' partials combined without their index rows): an empty share writes a
    // neutral partial (out and row sums 0, row maxima -inf) from CB_FILL (two tiles: zeros, -inf)
    constexpr uint32_t write_empty = get_compile_time_arg_val(kv_args.next_compile_time_args_offset());
    constexpr uint32_t cb_fill = get_compile_time_arg_val(kv_args.next_compile_time_args_offset() + 1);
    // query tile rows a unit (the split latents' 64-head unit: 2, a partial each): unit u = (query s, split j) writes
    // partial r to the partials' unit (SQT s + r) G + j (SQT 1: u)
    constexpr uint32_t SQT = get_compile_time_arg_val(kv_args.next_compile_time_args_offset() + 2);

    const uint32_t part_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t u_first = get_arg_val<uint32_t>(2);  // this core's units u_first .. u_first + n_units - 1
    const uint32_t n_units = get_arg_val<uint32_t>(3);
    const uint32_t base_unit = get_arg_val<uint32_t>(4);  // the partials' unit of this call's unit 0 (split latents)
    if (n_units == 0 || u_first >= S * G) {
        return;
    }
    constexpr uint32_t row_bytes = k_dim * kv_bytes;  // K (latent) rows
    constexpr uint32_t part_tiles = dsa_split::part_tiles(vDHt);
    constexpr uint32_t tile_bytes = tt::constants::TILE_HW * 2;
    static_assert(part_args.AlignedPageSize == tile_bytes && kv_args.AlignedPageSize == row_bytes);

    Noc noc;
    experimental::CB part_cb(cb_part);
    const auto part = TensorAccessor(part_args, part_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    experimental::CB idx_cb(cb_idx), kreq_cb(cb_kreq), kack_cb(cb_kack);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    // the reduce scaler (1.0; compute's exp applies the softmax scale), the column identity, the -inf row
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);
    {
        experimental::CB neginf_cb(cb_neginf);
        neginf_cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neginf_cb.get_write_ptr());
        for (uint32_t c = 0; c < tt::constants::FACE_WIDTH; ++c) {
            p[c] = neg_inf_bf16;
            p[tt::constants::FACE_HW + c] = neg_inf_bf16;
        }
        neginf_cb.push_back(1);
    }

    for (uint32_t k = 0; k < n_units; ++k) {
    const uint32_t unit = u_first + k;
    if (unit >= S * G) {
        continue;
    }
    bool last = false, empty = false;
    while (!last) {
        kreq_cb.wait_front(1);
        uint32_t base, half, dst_l1;
        {
            volatile tt_l1_ptr uint32_t* request = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
            base = request[sparse_sdpa::gather_request::BASE];
            half = request[sparse_sdpa::gather_request::SPLIT];
            last = request[sparse_sdpa::gather_request::IS_LAST] != 0;
            dst_l1 = request[sparse_sdpa::gather_request::DST_L1];
        }
        kreq_cb.pop_front(1);
        if (half == dsa_split::EMPTY) {
            empty = true;
        } else {
            sparse_sdpa::trid_ring_gather<false, 1, 1, 0, 0, sparse_sdpa::K_TRID_RING>(
                noc, kv, dst_l1, idx_ptr, base, 0, half, row_bytes, 0);
        }
        kack_cb.reserve_back(1);
        kack_cb.push_back(1);
    }
    const uint32_t first = base_unit + (unit / G) * SQT * G + unit % G;  // partial r: first + r G
    if (empty) {
        if constexpr (write_empty) {
            experimental::CB fill_cb(cb_fill);
            fill_cb.reserve_back(2);
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fill_cb.get_write_ptr());
            for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
                f[i] = 0;
                f[tile_bytes / 4 + i] = 0xFF80FF80u;  // two bf16 -inf
            }
            for (uint32_t r = 0; r < SQT; ++r) {
                for (uint32_t t = 0; t < part_tiles; ++t) {  // out tiles and row sums 0, row maxima -inf
                    const uint32_t src = t == part_tiles - 1 ? tile_bytes : 0;
                    noc.async_write(
                        fill_cb, part, tile_bytes, {.offset_bytes = src},
                        {.page_id = (first + r * G) * part_tiles + t});
                }
            }
            noc.async_write_barrier();
        }
        continue;
    }
    static_assert(S * G * part_tiles > 0);
    for (uint32_t r = 0; r < SQT; ++r) {  // (unit < S G: checked at the top; base_unit: the caller's)
        part_cb.wait_front(part_tiles);
        for (uint32_t t = 0; t < part_tiles; ++t) {
            noc.async_write(
                part_cb, part, tile_bytes, {.offset_bytes = t * tile_bytes},
                {.page_id = (first + r * G) * part_tiles + t});
        }
        noc.async_write_barrier();
        part_cb.pop_front(part_tiles);
    }
    }  // units
}
