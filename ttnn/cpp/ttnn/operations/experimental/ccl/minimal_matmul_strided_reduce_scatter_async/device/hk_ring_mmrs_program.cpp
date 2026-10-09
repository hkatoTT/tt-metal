// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// EXPERIMENT (hkato, optional, default off): TT_MMRS_RING=1 builds the fused matmul + reduce-scatter + addcmul
// entirely on the matmul cores, so the matmul gets the whole grid instead of sharing it with separate
// reduce-scatter worker and mux cores. Each core runs the ring schedule itself: at step k it computes its share of
// slice s(k), adds the partial the neighbour device sent for the same tiles, and the in1 sink row relays every
// column's partials to the mirror cores of the neighbour over its own fabric connections. See
// kernels/hk_ring_mm_in1.cpp for the protocol.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_program.hpp"

namespace ttnn::experimental::prim {

bool hk_mmrs_ring_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TT_MMRS_RING");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

namespace {

constexpr const char* kHkKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/kernels/";

// Forwarding order along one axis, as minimal_matmul: the injector at 0, then increasing for NOC_0 and
// decreasing for NOC_1.
std::vector<uint32_t> chain_order(uint32_t n, tt::tt_metal::NOC noc) {
    std::vector<uint32_t> order = {0};
    for (uint32_t i = 1; i < n; i++) {
        order.push_back(noc == tt::tt_metal::NOC::NOC_0 ? i : n - i);
    }
    return order;
}

// Index of the in1 relay-kernel ready-semaphore arg (see the in1 runtime-arg layout below)
constexpr uint32_t kIn1BaseArgs = 14;

}  // namespace

ttnn::device_operation::CachedProgram<MinimalMatmulStridedReduceScatterAsyncProgramFactory::shared_variables_t>
hk_ring_mmrs_program(
    const MinimalMatmulStridedReduceScatterAsyncParams& attrs,
    const ttnn::MeshCoordinate& coord,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& outputs) {
    using namespace tt::tt_metal;
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    Program program{};
    const Tensor& in0 = tensor_args.input_tensor;
    const Tensor& w = tensor_args.weight_tensor;
    TT_FATAL(
        tensor_args.addcmul_input_tensor1.has_value() && tensor_args.addcmul_input_tensor2.has_value() &&
            attrs.fused_ternary_scalar.has_value(),
        "TT_MMRS_RING needs the fused addcmul inputs");
    TT_FATAL(
        !tensor_args.mm_optional_input_tensor.has_value() && !attrs.matmul_struct.fused_activation.has_value(),
        "TT_MMRS_RING supports no fused concat or activation");
    // An optional bias must be the full [1, N] row on every device (the caller replicates RowParallelLinear's
    // rank-0 bias); it is added once, at the last step, for this device's own slice.
    const bool has_bias = tensor_args.bias.has_value();
    const Tensor& ta = *tensor_args.addcmul_input_tensor1;
    const Tensor& tb = *tensor_args.addcmul_input_tensor2;
    const Tensor& out = outputs.at(2);
    auto* mesh_device = in0.device();
    TT_FATAL(attrs.matmul_struct.config.has_value(), "TT_MMRS_RING needs an explicit matmul config");
    const auto cfg = attrs.matmul_struct.config.value();
    const CoreCoord grid = cfg.compute_with_storage_grid_size;
    const uint32_t R = attrs.ring_size;
    const auto topology = attrs.topology;
    TT_FATAL(topology == ttnn::ccl::Topology::Ring && R >= 2, "TT_MMRS_RING needs a ring");

    const uint32_t d = ::ttnn::ccl::get_linearized_index_from_physical_coord(in0, coord, attrs.cluster_axis);
    const auto fwd =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(in0, coord, 1, topology, attrs.cluster_axis);
    const auto bwd =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(in0, coord, -1, topology, attrs.cluster_axis);
    TT_FATAL(fwd.has_value() && bwd.has_value(), "ring neighbours must exist");
    TT_FATAL(
        ::ttnn::ccl::get_linearized_index_from_physical_coord(in0, *fwd, attrs.cluster_axis) == (d + 1) % R &&
            ::ttnn::ccl::get_linearized_index_from_physical_coord(in0, *bwd, attrs.cluster_axis) == (d + R - 1) % R,
        "TT_MMRS_RING: unexpected ring neighbour order");

    const uint32_t K = w.padded_shape()[-2];
    const uint32_t N = w.padded_shape()[-1];
    TT_FATAL(in0.padded_shape()[-1] == K, "in0 K {} != weight K {}", in0.padded_shape()[-1], K);
    const uint32_t M = in0.physical_volume() / K;
    const uint32_t M_tiles = M / TILE_HEIGHT;
    const uint32_t K_tiles = K / TILE_WIDTH;
    const uint32_t N_tiles = N / TILE_WIDTH;
    TT_FATAL(N_tiles % R == 0, "N tiles {} must split into {} slices", N_tiles, R);
    const uint32_t out_Wt = N_tiles / R;
    // G M-groups per core row: each row splits into G groups of grid.x / G cores that share one in0 chain, so a
    // group covers one slice's columns with Nb = out_Wt / (grid.x / G) tiles per core (TT_MMRS_RING_G, default 2)
    const uint32_t G = [] {
        const char* e = std::getenv("TT_MMRS_RING_G");
        return e != nullptr ? static_cast<uint32_t>(std::atoi(e)) : 2u;
    }();
    TT_FATAL(G >= 1 && grid.x % G == 0, "TT_MMRS_RING_G {} must divide the grid width {}", G, grid.x);
    const uint32_t cols_per_group = grid.x / G;
    TT_FATAL(out_Wt % cols_per_group == 0, "slice width {} must split over {} cores", out_Wt, cols_per_group);
    const uint32_t Nb = out_Wt / cols_per_group;
    const uint32_t Mb = cfg.M_block_size;
    const uint32_t Kb = cfg.K_block_size;
    const uint32_t sub_h = cfg.subblock_h;
    const uint32_t sub_w = cfg.subblock_w;
    TT_FATAL(K_tiles % Kb == 0, "K tiles {} must be a multiple of K_block {}", K_tiles, Kb);
    TT_FATAL(Nb % sub_w == 0 && Mb % sub_h == 0, "subblock {}x{} must divide block {}x{}", sub_h, sub_w, Mb, Nb);
    const uint32_t M_per_core = tt::div_up(M_tiles, grid.y * G);
    const uint32_t MB = tt::div_up(M_per_core, Mb);
    const uint32_t KB = K_tiles / Kb;
    const uint32_t block = Mb * Nb;
    TT_FATAL(grid.x % 2 == 0, "grid width must be even (forward / backward relay columns)");
    const uint32_t num_relay_links = grid.x / 2;
    // Relay direction per column: the first half of every row group's columns runs the forward ring, the second
    // half the backward ring, so the columns that share in1 across groups also share the slice order. The k-th
    // forward (backward) column relays on link k; its start-barrier peer is the backward (forward) column on the
    // same link.
    TT_FATAL(cols_per_group % 2 == 0, "row groups need an even number of columns");
    std::vector<uint32_t> col_link(grid.x);
    std::vector<uint32_t> fwd_cols;
    std::vector<uint32_t> bwd_cols;
    auto col_fwd = [&](uint32_t x) { return (x % cols_per_group) < cols_per_group / 2; };
    for (uint32_t x = 0; x < grid.x; x++) {
        auto& v = col_fwd(x) ? fwd_cols : bwd_cols;
        col_link[x] = static_cast<uint32_t>(v.size());
        v.push_back(x);
    }

    // a / b / out are [M, N/R] (b one broadcast tile row), tiled, interleaved
    TT_FATAL(
        ta.padded_shape()[-1] / TILE_WIDTH == out_Wt && ta.physical_volume() / ta.padded_shape()[-1] == M,
        "TT_MMRS_RING: residual must be [M, N/R]");
    TT_FATAL(
        tb.padded_shape()[-1] / TILE_WIDTH == out_Wt && tb.physical_volume() / tb.padded_shape()[-1] == TILE_HEIGHT,
        "TT_MMRS_RING: gate must be a single broadcast row");
    TT_FATAL(
        out.padded_shape()[-1] / TILE_WIDTH == out_Wt && out.physical_volume() / out.padded_shape()[-1] == M,
        "TT_MMRS_RING: output must be [M, N/R]");
    for (const Tensor* t : {&in0, &w, &ta, &tb, &out}) {
        TT_FATAL(
            t->layout() == Layout::TILE && t->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "TT_MMRS_RING: tensors must be tiled and interleaved");
    }
    TT_FATAL(tb.dtype() != DataType::FLOAT32, "TT_MMRS_RING: fp32 gate not supported");
    if (has_bias) {
        const Tensor& bias = *tensor_args.bias;
        TT_FATAL(
            bias.padded_shape()[-1] / TILE_WIDTH == N_tiles &&
                bias.physical_volume() / bias.padded_shape()[-1] == TILE_HEIGHT && bias.layout() == Layout::TILE &&
                bias.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "TT_MMRS_RING: bias must be a replicated, tiled [1, N] row");
    }

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), attrs.matmul_struct.compute_kernel_config);
    (void)packer_l1_acc;

    const auto in0_df = datatype_to_dataformat_converter(in0.dtype());
    const auto in1_df = datatype_to_dataformat_converter(w.dtype());
    const auto send_df = datatype_to_dataformat_converter(outputs.at(0).dtype());
    const auto interm_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const auto final_df = datatype_to_dataformat_converter(out.dtype());
    const auto a_df = datatype_to_dataformat_converter(ta.dtype());
    const auto b_df = datatype_to_dataformat_converter(tb.dtype());
    const uint32_t in0_ts = tt::tile_size(in0_df);
    const uint32_t in1_ts = tt::tile_size(in1_df);
    const uint32_t send_ts = tt::tile_size(send_df);
    const uint32_t interm_ts = tt::tile_size(interm_df);
    const uint32_t final_ts = tt::tile_size(final_df);
    const uint32_t a_ts = tt::tile_size(a_df);
    const uint32_t b_ts = tt::tile_size(b_df);

    const uint32_t tiles_per_packet =
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()) / send_ts;
    TT_FATAL(tiles_per_packet >= 1, "a tile must fit in a fabric packet");
    const uint32_t group_tiles = tiles_per_packet * 4;

    const CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1});
    std::map<std::string, std::string> kernel_defines;  // defines shared by every kernel
    // in0 / in1 buffering depth in K blocks (TT_MMRS_RING_BUF, default 2)
    const uint32_t nbuf = [] {
        const char* e = std::getenv("TT_MMRS_RING_BUF");
        return e != nullptr ? static_cast<uint32_t>(std::atoi(e)) : 2u;
    }();
    tt::tt_metal::create_cb(tt::CBIndex::c_0, program, all_cores, in0_ts, nbuf * Mb * Kb, in0_df);
    tt::tt_metal::create_cb(tt::CBIndex::c_1, program, all_cores, in1_ts, nbuf * Kb * Nb, in1_df);
    // partials waiting for the relay; one block is enough (the relay drains it within the next block's K loop)
    const uint32_t send_slots = [] {
        const char* e = std::getenv("TT_MMRS_RING_SEND_SLOTS");
        return e != nullptr ? static_cast<uint32_t>(std::atoi(e)) : 1u;
    }();
    tt::tt_metal::create_cb(tt::CBIndex::c_2, program, all_cores, send_ts, send_slots * block, send_df);
    kernel_defines["HK_SEND_SLOTS"] = std::to_string(send_slots);
    tt::tt_metal::create_cb(tt::CBIndex::c_4, program, all_cores, final_ts, 2 * Nb, final_df);  // output rows
    tt::tt_metal::create_cb(tt::CBIndex::c_3, program, all_cores, interm_ts, block, interm_df);
    tt::tt_metal::create_cb(tt::CBIndex::c_5, program, all_cores, a_ts, 2 * Nb, a_df);
    tt::tt_metal::create_cb(tt::CBIndex::c_6, program, all_cores, b_ts, Nb, b_df);
    tt::tt_metal::create_cb(tt::CBIndex::c_7, program, all_cores, send_ts, (R - 1) * MB * block, send_df);
    tt::tt_metal::create_cb(tt::CBIndex::c_8, program, all_cores, 16, 16, tt::DataFormat::Float32);  // relay scratch
    tt::tt_metal::create_cb(tt::CBIndex::c_9, program, all_cores, send_ts, group_tiles, send_df);    // relay staging
    if (has_bias) {
        const auto bias_df = datatype_to_dataformat_converter(tensor_args.bias->dtype());
        tt::tt_metal::create_cb(tt::CBIndex::c_10, program, all_cores, tt::tile_size(bias_df), Nb, bias_df);
        kernel_defines["HK_RING_BIAS"] = "1";
    }

    const uint32_t in0_sender_sem = CreateSemaphore(program, all_cores, 0);
    const uint32_t in0_receiver_sem = CreateSemaphore(program, all_cores, 0);
    const uint32_t in0_valid_sem = CreateSemaphore(program, all_cores, 1);
    const uint32_t in1_sender_sem = CreateSemaphore(program, all_cores, 0);
    const uint32_t in1_receiver_sem = CreateSemaphore(program, all_cores, 0);
    const uint32_t in1_valid_sem = CreateSemaphore(program, all_cores, 1);
    const uint32_t relay_done_sem = CreateSemaphore(program, all_cores, 0);
    const uint32_t recv_sem = CreateSemaphore(program, all_cores, 0);

    // in0 with the output writes on RISCV_1 / the DRAM-write NOC, in1 and the relay on RISCV_0 / the DRAM-read NOC (the
    // other way round measured 2.6-3.5 ms instead of 2.3)
    const auto in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch());
    const auto in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch());
    const auto in0_risc = DataMovementProcessor::RISCV_1;
    const auto in1_risc = DataMovementProcessor::RISCV_0;
    const auto in0_base_order = chain_order(cols_per_group, in0_noc);  // along a row group (offset by the group start)
    // Row y's in0 chain starts at relative column y % cols_per_group, spreading the in0 DRAM readers over all columns
    // instead of one column per group (2.35 -> 2.30 ms)
    auto in0_order_of = [&](uint32_t y) {
        std::vector<uint32_t> o = in0_base_order;
        for (auto& v : o) {
            v = (v + y) % cols_per_group;
        }
        return o;
    };
    const auto in1_order = chain_order(grid.y, in1_noc);  // down a column
    const uint32_t relay_row = in1_order.back();
    TT_FATAL(relay_row != 0, "TT_MMRS_RING needs at least 2 core rows");

    // --- kernels
    auto in0_ct = [&](bool injector) {
        std::vector<uint32_t> ct = {
            M_tiles,
            K_tiles,
            Mb,
            Kb,
            Nb,
            MB,
            R,
            in0_ts,
            final_ts,
            in0_sender_sem,
            in0_receiver_sem,
            in0_valid_sem,
            injector ? 1u : 0u,
            relay_done_sem,
            out_Wt};
        TensorAccessorArgs(*in0.buffer()).append_to(ct);
        TensorAccessorArgs(*out.buffer()).append_to(ct);
        return ct;
    };
    const std::string in0_src = std::string(kHkKernelDir) + "hk_ring_mm_in0.cpp";
    std::vector<CoreRange> in0_inj_ranges;
    std::vector<CoreRange> in0_rest_ranges;
    for (uint32_t y = 0; y < grid.y; y++) {
        const uint32_t head = in0_order_of(y)[0];
        for (uint32_t x = 0; x < grid.x; x++) {
            (x % cols_per_group == head ? in0_inj_ranges : in0_rest_ranges)
                .emplace_back(CoreCoord{x, y}, CoreCoord{x, y});
        }
    }
    const KernelHandle in0_inj = CreateKernel(
        program,
        in0_src,
        CoreRangeSet(in0_inj_ranges),
        DataMovementConfig{
            .processor = in0_risc, .noc = in0_noc, .compile_args = in0_ct(true), .defines = kernel_defines});
    const CoreRangeSet in0_rest(in0_rest_ranges);
    const KernelHandle in0_oth = CreateKernel(
        program,
        in0_src,
        in0_rest,
        DataMovementConfig{
            .processor = in0_risc, .noc = in0_noc, .compile_args = in0_ct(false), .defines = kernel_defines});

    auto in1_ct = [&](bool injector) {
        std::vector<uint32_t> ct = {
            K_tiles,
            N_tiles,
            Kb,
            Nb,
            Mb,
            MB,
            R,
            in1_ts,
            a_ts,
            b_ts,
            in1_sender_sem,
            in1_receiver_sem,
            in1_valid_sem,
            injector ? 1u : 0u,
            recv_sem,
            out_Wt,
            send_ts,
            relay_done_sem,
            static_cast<uint32_t>(grid.y),
            tiles_per_packet,
            group_tiles};
        TensorAccessorArgs(*w.buffer()).append_to(ct);
        TensorAccessorArgs(*ta.buffer()).append_to(ct);
        TensorAccessorArgs(*tb.buffer()).append_to(ct);
        if (has_bias) {
            TensorAccessorArgs(*tensor_args.bias->buffer()).append_to(ct);
        }
        return ct;
    };
    const std::string in1_src = std::string(kHkKernelDir) + "hk_ring_mm_in1.cpp";
    const KernelHandle in1_inj = CreateKernel(
        program,
        in1_src,
        CoreRange(CoreCoord{0, in1_order[0]}, CoreCoord{grid.x - 1, in1_order[0]}),
        DataMovementConfig{
            .processor = in1_risc, .noc = in1_noc, .compile_args = in1_ct(true), .defines = kernel_defines});
    std::vector<CoreRange> mid_ranges;
    for (uint32_t y = 0; y < grid.y; y++) {
        if (y != in1_order[0] && y != relay_row) {
            mid_ranges.emplace_back(CoreCoord{0, y}, CoreCoord{grid.x - 1, y});
        }
    }
    std::optional<KernelHandle> in1_mid;
    if (!mid_ranges.empty()) {
        in1_mid = CreateKernel(
            program,
            in1_src,
            CoreRangeSet(mid_ranges),
            DataMovementConfig{
                .processor = in1_risc, .noc = in1_noc, .compile_args = in1_ct(false), .defines = kernel_defines});
    }
    KernelHandle in1_relay = CreateKernel(
        program,
        in1_src,
        CoreRange(CoreCoord{0, relay_row}, CoreCoord{grid.x - 1, relay_row}),
        DataMovementConfig{.processor = in1_risc, .noc = in1_noc, .compile_args = in1_ct(false), .defines = [&] {
                               auto d = kernel_defines;
                               d["HK_RELAY"] = "1";
                               return d;
                           }()});

    const float scalar = attrs.fused_ternary_scalar.value();
    const KernelHandle compute = CreateKernel(
        program,
        std::string(kHkKernelDir) + "hk_ring_mm_compute.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = {KB, Mb, Kb, Nb, MB, sub_h, sub_w, R, scalar == 1.0f ? 1u : 0u, fp32_dest_acc_en ? 4u : 8u},
            .defines = kernel_defines});

    // --- runtime args
    const auto src_node = mesh_device->get_fabric_node_id(coord);
    const auto fwd_node = mesh_device->get_fabric_node_id(*fwd);
    const auto bwd_node = mesh_device->get_fabric_node_id(*bwd);
    const auto fwd_links = tt::tt_fabric::get_forwarding_link_indices(src_node, fwd_node);
    const auto bwd_links = tt::tt_fabric::get_forwarding_link_indices(src_node, bwd_node);
    for (uint32_t l = 0; l < num_relay_links; l++) {
        TT_FATAL(
            std::find(fwd_links.begin(), fwd_links.end(), l) != fwd_links.end() &&
                std::find(bwd_links.begin(), bwd_links.end(), l) != bwd_links.end(),
            "TT_MMRS_RING needs fabric link {} in both ring directions",
            l);
    }
    const uint32_t ready_sem_addr = attrs.semaphore.at(0).address();
    uint32_t scalar_bits;
    std::memcpy(&scalar_bits, &scalar, sizeof(scalar_bits));

    MinimalMatmulStridedReduceScatterAsyncProgramFactory::shared_variables_t sv{};
    sv.hk_ring = true;
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            const CoreCoord core{x, y};
            const uint32_t g = x / cols_per_group;
            const uint32_t gx0 = g * cols_per_group;
            const uint32_t M_start = (y * G + g) * M_per_core;
            const uint32_t M_end = M_start + M_per_core;
            const uint32_t M_valid_end = std::max(std::min(M_end, M_tiles), M_start);  // may hold no rows
            const uint32_t col_in_slice = (x - gx0) * Nb;
            const bool fwd_col = col_fwd(x);

            // in0: forwarded along the row
            const auto in0_order = in0_order_of(y);
            const auto it0 = std::find(in0_order.begin(), in0_order.end(), x - gx0);
            const uint32_t i0 = static_cast<uint32_t>(it0 - in0_order.begin());
            const CoreCoord prev0 =
                mesh_device->worker_core_from_logical_core({gx0 + in0_order[i0 == 0 ? 0 : i0 - 1], y});
            const CoreCoord next0 = mesh_device->worker_core_from_logical_core(
                {gx0 + in0_order[std::min<uint32_t>(i0 + 1, cols_per_group - 1)], y});
            const CoreCoord relay_phys = mesh_device->worker_core_from_logical_core({x, relay_row});
            const std::vector<uint32_t> in0_rt = {
                in0.buffer()->address(),
                out.buffer()->address(),
                i0 == cols_per_group - 1 ? 1u : 0u,
                static_cast<uint32_t>(next0.x),
                static_cast<uint32_t>(next0.y),
                static_cast<uint32_t>(prev0.x),
                static_cast<uint32_t>(prev0.y),
                M_start,
                M_end,
                M_valid_end,
                col_in_slice,
                static_cast<uint32_t>(relay_phys.x),
                static_cast<uint32_t>(relay_phys.y),
                y};
            const KernelHandle k0 = x - gx0 == in0_order[0] ? in0_inj : in0_oth;
            SetRuntimeArgs(program, k0, core, in0_rt);
            sv.hk_in0.emplace_back(core, k0);

            // in1: forwarded down the column
            const auto it1 = std::find(in1_order.begin(), in1_order.end(), y);
            const uint32_t i1 = static_cast<uint32_t>(it1 - in1_order.begin());
            const CoreCoord prev1 = mesh_device->worker_core_from_logical_core({x, in1_order[i1 == 0 ? 0 : i1 - 1]});
            const CoreCoord next1 =
                mesh_device->worker_core_from_logical_core({x, in1_order[std::min<uint32_t>(i1 + 1, grid.y - 1)]});
            std::vector<uint32_t> in1_rt = {
                w.buffer()->address(),
                ta.buffer()->address(),
                tb.buffer()->address(),
                has_bias ? tensor_args.bias->buffer()->address() : 0u,
                i1 == grid.y - 1 ? 0u : 1u,
                static_cast<uint32_t>(next1.x),
                static_cast<uint32_t>(next1.y),
                static_cast<uint32_t>(prev1.x),
                static_cast<uint32_t>(prev1.y),
                M_start,
                M_valid_end,
                col_in_slice,
                0u,  // second in1 destination (unused)
                0u};
            TT_FATAL(in1_rt.size() == kIn1BaseArgs, "in1 runtime-arg layout changed");
            for (uint32_t k = 0; k < R; k++) {
                // forward ring: d receives slice d-k-1 ... ends with its own; backward: d+k+1 ...
                in1_rt.push_back(fwd_col ? (d + 2 * R - k - 1) % R : (d + k + 1) % R);
            }
            KernelHandle k1;
            if (y == relay_row) {
                k1 = in1_relay;
                const uint32_t peer_col = (fwd_col ? bwd_cols : fwd_cols).at(col_link[x]);
                const CoreCoord peer = mesh_device->worker_core_from_logical_core({peer_col, relay_row});
                in1_rt.push_back(ready_sem_addr);
                in1_rt.push_back(static_cast<uint32_t>(peer.x));
                in1_rt.push_back(static_cast<uint32_t>(peer.y));
                for (uint32_t s = 0; s < grid.y; s++) {
                    const CoreCoord src = mesh_device->worker_core_from_logical_core({x, s});
                    const uint32_t s_start = (s * G + g) * M_per_core;
                    in1_rt.push_back(static_cast<uint32_t>(src.x));
                    in1_rt.push_back(static_cast<uint32_t>(src.y));
                    in1_rt.push_back(s_start);
                    in1_rt.push_back(std::max(std::min(s_start + M_per_core, M_tiles), s_start));
                }
                const uint32_t link = col_link[x];
                tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                    src_node,
                    std::vector<tt::tt_fabric::FabricNodeId>{fwd_col ? fwd_node : bwd_node},
                    std::vector<uint32_t>{link},
                    program,
                    in1_relay,
                    core,
                    in1_rt,
                    tt::tt_fabric::FabricApiType::Linear);
            } else {
                k1 = y == in1_order[0] ? in1_inj : *in1_mid;
            }
            SetRuntimeArgs(program, k1, core, in1_rt);
            sv.hk_in1.emplace_back(core, k1);
            if (y == relay_row) {
                sv.hk_relay_cores.push_back(core);
            }

            SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{M_start, M_end, M_valid_end, scalar_bits});
        }
    }
    sv.hk_relay_kernel = in1_relay;
    sv.hk_ready_sem_arg = kIn1BaseArgs + R;
    return {std::move(program), std::move(sv)};
}

void hk_ring_mmrs_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MinimalMatmulStridedReduceScatterAsyncProgramFactory::shared_variables_t& sv,
    const MinimalMatmulStridedReduceScatterAsyncParams& attrs,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& outputs) {
    using namespace tt::tt_metal;
    const uint32_t in0_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t w_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t a_addr = tensor_args.addcmul_input_tensor1->buffer()->address();
    const uint32_t b_addr = tensor_args.addcmul_input_tensor2->buffer()->address();
    const uint32_t out_addr = outputs.at(2).buffer()->address();
    for (const auto& [core, k] : sv.hk_in0) {
        auto& args = GetRuntimeArgs(program, k, core);
        args[0] = in0_addr;
        args[1] = out_addr;
    }
    for (const auto& [core, k] : sv.hk_in1) {
        auto& args = GetRuntimeArgs(program, k, core);
        args[0] = w_addr;
        args[1] = a_addr;
        args[2] = b_addr;
        if (tensor_args.bias.has_value()) {
            args[3] = tensor_args.bias->buffer()->address();
        }
    }
    for (const auto& core : sv.hk_relay_cores) {
        auto& args = GetRuntimeArgs(program, sv.hk_relay_kernel, core);
        args[sv.hk_ready_sem_arg] = attrs.semaphore.at(0).address();
    }
}

}  // namespace ttnn::experimental::prim
