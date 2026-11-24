// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/col_row_reduce_scatter/device/col_row_reduce_scatter_program_factory.hpp"

#include <array>
#include <cstdlib>
#include <set>
#include <utility>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <variant>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t kColAxis = 0;  // the mesh column's chips (the mesh rows)
constexpr uint32_t kRowAxis = 1;  // the mesh row's chips
constexpr uint32_t kAddChunk = 8;  // tiles a chunk of the add stage (Dst's half with a 16-bit Dst)
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/ccl/col_row_reduce_scatter/device/kernels/";

// `n` semaphores of consecutive ids on `cores` (a batch each: semaphore first + b), clear of every id the program has on
// any of them -> first
uint32_t add_semaphore_block(tt::tt_metal::ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t n) {
    constexpr uint32_t kSemaphoresPerCore = 16;
    std::array<bool, kSemaphoresPerCore> used{};
    for (const auto& sem : desc.semaphores) {
        if (sem.core_type == tt::CoreType::WORKER && sem.id < kSemaphoresPerCore && sem.core_ranges.intersects(cores)) {
            used[sem.id] = true;
        }
    }
    for (uint32_t first = 0; first + n <= kSemaphoresPerCore; ++first) {
        bool free = true;
        for (uint32_t i = 0; i < n; ++i) {
            free &= !used[first + i];
        }
        if (free) {
            for (uint32_t i = 0; i < n; ++i) {
                desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                    .id = first + i, .core_type = tt::CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
            }
            return first;
        }
    }
    TT_THROW("col_row_reduce_scatter: no {} consecutive semaphore ids free on {}", n, cores.str());
}

// a relay kernel: waits for `producers` increments of batch b on its own semaphore local_sem + b (one a batch),
// then raises `recv_sem` on each of `receivers` by one
tt::tt_metal::KernelDescriptor relay_kernel(
    const CoreRangeSet& set,
    const CoreCoord& core,
    uint32_t local_sem,
    uint32_t batches,
    uint32_t producers,
    uint32_t recv_sem,
    const std::vector<CoreCoord>& receivers_noc,
    bool on_brisc) {
    tt::tt_metal::KernelDescriptor k;
    k.kernel_source = std::string(kKernelDir) + "col_row_rs_relay.cpp";
    k.core_ranges = set;
    k.compile_time_args = {local_sem, batches, producers};
    std::vector<uint32_t> rt = {recv_sem, static_cast<uint32_t>(receivers_noc.size())};
    for (const auto& c : receivers_noc) {
        rt.push_back(static_cast<uint32_t>(c.x));
        rt.push_back(static_cast<uint32_t>(c.y));
    }
    k.runtime_args.emplace_back(core, rt);
    if (on_brisc) {
        k.config = tt::tt_metal::WriterConfigDescriptor{};
    } else {
        k.config = tt::tt_metal::ReaderConfigDescriptor{};
    }
    return k;
}

tt::tt_metal::ProgramDescriptor build_program_descriptor(
    const ColRowReduceScatterParams& args,
    const ttnn::MeshCoordinate& coord,
    const ColRowReduceScatterInputs& tensor_args,
    ColRowReduceScatterResult& outputs) {
    tt::tt_metal::ProgramDescriptor desc;
    // the carried program: its semaphores first, so that the op's own take ids clear of them on any core
    // they share (none: its cores are checked against the op's below)
    const bool carry = args.extra_program.has_value();
    if (carry) {
        desc.semaphores = args.extra_program->semaphores;
    }
    auto* mesh_device = tensor_args.input.device();
    const auto& mesh_shape = mesh_device->shape();
    const uint32_t col_ring = mesh_shape[kColAxis];
    const uint32_t row_ring = mesh_shape[kRowAxis];
    const uint32_t batches = tensor_args.input.logical_shape()[0];
    const auto compute_config = ttnn::ccl::resolve_fp32_acc_compute_kernel_config(std::nullopt, tensor_args.input.dtype());
    // GLM53_CRR_BFP8 (the column's ring in BFLOAT8_B; value-changing, default off): its adds in a fp32 Dest, so that
    // each running sum is exact before the packer rounds it to BFP8 once (the BH packer rounds ties away from zero from
    // the Dest value, which ties rarely in fp32); the row's ring takes the add stage's bf16 total with `compute_config`
    const bool col8 = tensor_args.input.dtype() == tt::tt_metal::DataType::BFLOAT8_B;
    const auto col_compute_config = col8 ? std::optional<ttnn::DeviceComputeKernelConfig>(ttnn::DeviceComputeKernelConfig{
                                               .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                                               .fp32_dest_acc_en = true,
                                           })
                                         : compute_config;
    const bool add = tensor_args.shared.has_value();
    const Tensor& row_input = add ? *tensor_args.total : outputs.col_output;
    const bool ag = tensor_args.ag_output.has_value();
    // the all-gather's core: told by every writer of the row reduce-scatter when a batch is written
    const CoreRangeSet ag_set(CoreRange(args.ag_core, args.ag_core));
    std::optional<std::array<uint32_t, 3>> row_consumer;
    uint32_t ag_sem = 0;
    CoreCoord ag_noc;
    if (ag) {
        ag_sem = add_semaphore_block(desc, ag_set, batches);  // (the row writers' signals: one a batch)
        ag_noc = mesh_device->worker_core_from_logical_core(args.ag_core);
        row_consumer = std::array<uint32_t, 3>{static_cast<uint32_t>(ag_noc.x), static_cast<uint32_t>(ag_noc.y), ag_sem};
    }

    // The row's reduce-scatter first (its reader and writer are kernels 0 and 1), fused: it waits for a signal per
    // batch on a semaphore of its workers (ReduceScatterOpReceiver), which the relay core below raises.
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> row_signal =
        ttnn::experimental::ccl::ReduceScatterFusedOpSignaler();
    row_signal->init_fused_op();
    const auto row_fwd =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(outputs.col_output, coord, 1, args.topology, kRowAxis);
    const auto row_bwd =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(outputs.col_output, coord, -1, args.topology, kRowAxis);
    const uint32_t row_index = ttnn::ccl::get_linearized_index_from_physical_coord(outputs.col_output, coord, kRowAxis);
    const auto row_kernels = build_ring_reduce_scatter_minimal_async_program_descriptor_ext(
        desc,
        row_input,
        tensor_args.row_intermediate,
        coord,
        row_fwd,
        row_bwd,
        outputs.row_output,
        3,
        ag ? 1u : args.row_num_links,
        row_ring,
        row_index,
        args.topology,
        args.row_semaphores,
        args.row_barrier,
        row_signal,
        args.row_chunks_per_sync,
        args.row_workers_per_direction,
        args.row_core_grid_offset,
        compute_config,
        /*force_per_batch=*/true,
        row_consumer);
    TT_FATAL(
        !row_signal->fused_op_receiver_cores_noc.empty() && !row_signal->fused_op_receiver_signal_semaphores.empty(),
        "col_row_reduce_scatter: the row reduce-scatter's receivers are not set");

    // The relay core: counts the column writers' signals and raises the row workers' semaphore once a batch.
    const CoreRangeSet relay_set(CoreRange(args.relay_core, args.relay_core));
    const uint32_t relay_sem = add_semaphore_block(desc, relay_set, batches);  // (one a batch)
    const CoreCoord relay_noc = mesh_device->worker_core_from_logical_core(args.relay_core);

    // The column's reduce-scatter, one traversal per batch, each writer telling the relay core when its batch is
    // written; with a carried producer (extra_producers), fused: it waits for a signal per batch from the carried
    // program's relay kernel (below) as the row's does.
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> col_signal;
    if (carry && args.extra_producers > 0) {
        col_signal = ttnn::experimental::ccl::ReduceScatterFusedOpSignaler();
        col_signal->init_fused_op();
    }
    const auto col_fwd =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(tensor_args.input, coord, 1, args.topology, kColAxis);
    const auto col_bwd =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(tensor_args.input, coord, -1, args.topology, kColAxis);
    const uint32_t col_index = ttnn::ccl::get_linearized_index_from_physical_coord(tensor_args.input, coord, kColAxis);
    const auto col_kernels = build_ring_reduce_scatter_minimal_async_program_descriptor_ext(
        desc,
        tensor_args.input,
        tensor_args.col_intermediate,
        coord,
        col_fwd,
        col_bwd,
        outputs.col_output,
        2,
        args.col_num_links,
        col_ring,
        col_index,
        args.topology,
        args.col_semaphores,
        args.col_barrier,
        col_signal,
        args.col_chunks_per_sync,
        args.col_workers_per_direction,
        args.col_core_grid_offset,
        col_compute_config,
        /*force_per_batch=*/true,
        std::array<uint32_t, 3>{
            static_cast<uint32_t>(relay_noc.x), static_cast<uint32_t>(relay_noc.y), relay_sem});
    const uint32_t producers = desc.kernels[col_kernels.writer_kernel_index].core_ranges.num_cores();
    if (col_signal.has_value()) {
        TT_FATAL(
            !col_signal->fused_op_receiver_cores_noc.empty() && !col_signal->fused_op_receiver_signal_semaphores.empty(),
            "col_row_reduce_scatter: the column reduce-scatter's receivers are not set");
    }

    // The carried program, once the op's kernels are all in (each return below): its producers' relay kernel, then
    // its kernels (the address slots bound to the tensors) and CBs, on cores none of the op's take.
    auto carry_extra = [&](tt::tt_metal::ProgramDescriptor& d) {
        if (!carry) {
            return;
        }
        const auto& extra = *args.extra_program;
        std::set<std::pair<uint32_t, uint32_t>> taken;
        auto take = [&](const CoreRangeSet& set, const char* what) {
            for (const auto& r : set.ranges()) {
                for (auto x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                    for (auto y = r.start_coord.y; y <= r.end_coord.y; ++y) {
                        TT_FATAL(
                            taken.emplace(static_cast<uint32_t>(x), static_cast<uint32_t>(y)).second,
                            "col_row_reduce_scatter: {} on core ({}, {}), which the op takes",
                            what,
                            x,
                            y);
                    }
                }
            }
        };
        {
            std::set<std::pair<uint32_t, uint32_t>> own;  // (the op's kernels share cores among themselves)
            for (const auto& k : d.kernels) {
                for (const auto& r : k.core_ranges.ranges()) {
                    for (auto x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                        for (auto y = r.start_coord.y; y <= r.end_coord.y; ++y) {
                            own.emplace(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
                        }
                    }
                }
            }
            taken = std::move(own);
        }
        if (args.extra_producers > 0) {
            const CoreRangeSet set(CoreRange(args.extra_relay_core, args.extra_relay_core));
            // (one semaphore a batch, extra_relay_sem + b: the relay counts every producer per batch)
            for (uint32_t b = 0; b < batches; ++b) {
                bool has_sem = false;
                for (const auto& sem : extra.semaphores) {
                    has_sem |= sem.id == args.extra_relay_sem + b && sem.core_ranges.contains(args.extra_relay_core);
                }
                TT_FATAL(
                    has_sem,
                    "col_row_reduce_scatter: extra_program has no semaphore extra_relay_sem + {} on extra_relay_core",
                    b);
            }
            take(set, "the carried program's relay");
            d.kernels.push_back(relay_kernel(
                set,
                args.extra_relay_core,
                args.extra_relay_sem,
                batches,
                args.extra_producers,
                col_signal->fused_op_receiver_signal_semaphores[0],
                col_signal->fused_op_receiver_cores_noc,
                true));
        }
        std::set<std::pair<uint32_t, uint32_t>> carried;  // (the carried kernels share cores among themselves)
        for (const auto& k : extra.kernels) {
            for (const auto& r : k.core_ranges.ranges()) {
                for (auto x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                    for (auto y = r.start_coord.y; y <= r.end_coord.y; ++y) {
                        carried.emplace(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
                    }
                }
            }
        }
        for (const auto& c : carried) {
            TT_FATAL(
                taken.emplace(c).second,
                "col_row_reduce_scatter: a carried kernel on core ({}, {}), which the op takes",
                c.first,
                c.second);
        }
        for (size_t ki = 0; ki < extra.kernels.size(); ++ki) {
            tt::tt_metal::KernelDescriptor k = extra.kernels[ki];
            std::vector<std::pair<uint32_t, uint32_t>> slots;  // (arg, tensor)
            for (size_t i = 0; i < args.extra_slots.size(); i += 3) {
                if (args.extra_slots[i] == ki) {
                    slots.emplace_back(args.extra_slots[i + 1], args.extra_slots[i + 2]);
                }
            }
            if (!slots.empty()) {
                auto rts = std::move(k.runtime_args);
                k.runtime_args.clear();
                k.buffer_bindings.clear();
                for (const auto& [core, vals] : rts) {
                    std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> v(vals.begin(), vals.end());
                    for (const auto& [arg, ti] : slots) {
                        TT_FATAL(arg < v.size(), "col_row_reduce_scatter: carried kernel {}: no runtime arg {}", ki, arg);
                        v[arg] = ti < tensor_args.extra_inputs.size() ? tensor_args.extra_inputs[ti].buffer()
                                                                      : tensor_args.input.buffer();
                    }
                    k.emplace_runtime_args(core, v);
                }
            }
            d.kernels.push_back(std::move(k));
        }
        for (const auto& cb : extra.cbs) {
            d.cbs.push_back(cb);
        }
    };

    if (ag) {
        const uint32_t row_writers = desc.kernels[row_kernels.writer_kernel_index].core_ranges.num_cores();
        const auto& cs = outputs.col_output.padded_shape();
        const uint32_t R = cs[2] / tt::constants::TILE_HEIGHT;  // tile rows a batch
        constexpr uint32_t kAgNb = 8, kAgSlot = 4 * 2048;
        constexpr uint8_t cb_stage = 0;
        const uint32_t ag_done = ttnn::experimental::ccl::add_semaphore_descriptor(desc, ag_set);
        tt::tt_metal::CBDescriptor stage;
        stage.total_size = 2 * kAgNb * kAgSlot;
        stage.core_ranges = ag_set;
        stage.format_descriptors.push_back(tt::tt_metal::CBFormatDescriptor{
            .buffer_index = cb_stage, .data_format = tt::DataFormat::Float16_b, .page_size = kAgSlot});
        desc.cbs.push_back(std::move(stage));
        const auto node = mesh_device->get_fabric_node_id(coord);
        const uint32_t counter = static_cast<uint32_t>(args.ag_counter->address());
        // (the odd packets the other way round the row's ring of 4, so both directions' links carry the same:
        // col_row_ag.cpp)
        for (uint32_t role = 0; role < 2; ++role) {
            const auto& peer = role == 0 ? row_fwd : row_bwd;
            TT_FATAL(peer.has_value(), "col_row_reduce_scatter: the row's neighbours (a ring)");
            const auto peer_node = mesh_device->get_fabric_node_id(*peer);
            tt::tt_metal::KernelDescriptor k;
            k.kernel_source = std::string(kKernelDir) + "col_row_ag.cpp";
            k.core_ranges = ag_set;
            k.compile_time_args = {
                role, batches, R, ag_sem, row_writers, kAgNb, cb_stage, ag_done, 4 * batches};
            tt::tt_metal::TensorAccessorArgs(outputs.row_output.buffer()).append_to(k.compile_time_args);
            tt::tt_metal::TensorAccessorArgs(tensor_args.ag_output->buffer()).append_to(k.compile_time_args);
            std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> rt = {
                outputs.row_output.buffer(),
                tensor_args.ag_output->buffer(),
                row_index,
                static_cast<uint32_t>(*node.mesh_id),
                static_cast<uint32_t>(peer_node.chip_id),
                role == 0 ? 2u : 1u,
                counter,
                static_cast<uint32_t>(ag_noc.x),
                static_cast<uint32_t>(ag_noc.y)};
            std::vector<uint32_t> fabric_args;
            tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                node, peer_node, args.ag_link, desc, args.ag_core, fabric_args, tt::CoreType::WORKER);
            for (const uint32_t v : fabric_args) {
                rt.emplace_back(v);
            }
            k.emplace_runtime_args(args.ag_core, rt);
            if (role == 0) {
                k.config = tt::tt_metal::WriterConfigDescriptor{};
            } else {
                k.config = tt::tt_metal::ReaderConfigDescriptor{};
            }
            desc.kernels.push_back(std::move(k));
        }
    }

    const auto& row_receivers = row_signal->fused_op_receiver_cores_noc;
    const uint32_t row_sem = row_signal->fused_op_receiver_signal_semaphores[0];
    if (!add) {  // the column's writers -> the row's workers
        desc.kernels.push_back(
            relay_kernel(relay_set, args.relay_core, relay_sem, batches, producers, row_sem, row_receivers, true));
        carry_extra(desc);
        return desc;
    }

    // The add stage: total = col_output + shared (ELWADD, a 16-bit Dst: ttnn.add's bits), add_cores cores in a row,
    // each a contiguous range of every batch's tiles.  The relay core's BRISC passes the column's batches on to the add
    // cores; its NCRISC the add cores' batches on to the row's workers.
    std::vector<CoreCoord> add_cores;
    std::vector<CoreCoord> add_noc;
    std::vector<CoreRange> add_ranges;
    for (uint32_t k = 0; k < args.add_cores; ++k) {
        const CoreCoord c(args.add_core_grid_offset.x + k, args.add_core_grid_offset.y);
        add_cores.push_back(c);
        add_noc.push_back(mesh_device->worker_core_from_logical_core(c));
        add_ranges.emplace_back(c, c);
    }
    // (one rectangle: a program goes out range by range)
    const CoreRangeSet add_set = CoreRangeSet(add_ranges).merge_ranges();
    const uint32_t add_sem = ttnn::experimental::ccl::add_semaphore_descriptor(desc, add_set);
    const uint32_t relay_sem_b = add_semaphore_block(desc, relay_set, batches);  // (one a batch)
    desc.kernels.push_back(
        relay_kernel(relay_set, args.relay_core, relay_sem, batches, producers, add_sem, add_noc, true));
    desc.kernels.push_back(relay_kernel(
        relay_set, args.relay_core, relay_sem_b, batches, args.add_cores, row_sem, row_receivers, false));

    const auto& cs = outputs.col_output.padded_shape();
    const uint32_t tiles_per_batch = (cs[2] / tt::constants::TILE_HEIGHT) * (cs[3] / tt::constants::TILE_WIDTH);
    constexpr uint8_t cb_a = 0, cb_b = 1, cb_out = 2;
    // (cb_a the column's output: bf16, or bfp8_b with GLM53_CRR_BFP8; the shared partial and the total bf16)
    const tt::DataFormat col_df = tt::tt_metal::datatype_to_dataformat_converter(outputs.col_output.dtype());
    for (const uint8_t cb : {cb_a, cb_b, cb_out}) {
        const tt::DataFormat df = cb == cb_a ? col_df : tt::DataFormat::Float16_b;
        const uint32_t tile_bytes = tt::tile_size(df);
        tt::tt_metal::CBDescriptor d;
        d.total_size = 2 * kAddChunk * tile_bytes;
        d.core_ranges = add_set;
        d.format_descriptors.push_back(
            tt::tt_metal::CBFormatDescriptor{.buffer_index = cb, .data_format = df, .page_size = tile_bytes});
        desc.cbs.push_back(std::move(d));
    }
    tt::tt_metal::KernelDescriptor reader, compute, writer;
    reader.kernel_source = std::string(kKernelDir) + "col_row_add_reader.cpp";
    reader.core_ranges = add_set;
    reader.compile_time_args = {cb_a, cb_b, add_sem, batches, tiles_per_batch, kAddChunk};
    tt::tt_metal::TensorAccessorArgs(outputs.col_output.buffer()).append_to(reader.compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.shared->buffer()).append_to(reader.compile_time_args);
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};
    compute.kernel_source = std::string(kKernelDir) + "col_row_add_compute.cpp";
    compute.core_ranges = add_set;
    compute.compile_time_args = {cb_a, cb_b, cb_out, batches, kAddChunk};
    compute.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .fp32_dest_acc_en = false, .dst_full_sync_en = false};
    writer.kernel_source = std::string(kKernelDir) + "col_row_add_writer.cpp";
    writer.core_ranges = add_set;
    writer.compile_time_args = {cb_out, batches, tiles_per_batch, kAddChunk, relay_sem_b};
    tt::tt_metal::TensorAccessorArgs(tensor_args.total->buffer()).append_to(writer.compile_time_args);
    writer.config = tt::tt_metal::WriterConfigDescriptor{};
    for (uint32_t k = 0; k < args.add_cores; ++k) {
        const uint32_t start = k * tiles_per_batch / args.add_cores, end = (k + 1) * tiles_per_batch / args.add_cores;
        reader.emplace_runtime_args(
            add_cores[k],
            {std::variant<uint32_t, tt::tt_metal::Buffer*>(outputs.col_output.buffer()),
             std::variant<uint32_t, tt::tt_metal::Buffer*>(tensor_args.shared->buffer()),
             std::variant<uint32_t, tt::tt_metal::Buffer*>(start),
             std::variant<uint32_t, tt::tt_metal::Buffer*>(end)});
        compute.runtime_args.emplace_back(add_cores[k], std::vector<uint32_t>{start, end});
        writer.emplace_runtime_args(
            add_cores[k],
            {std::variant<uint32_t, tt::tt_metal::Buffer*>(tensor_args.total->buffer()),
             std::variant<uint32_t, tt::tt_metal::Buffer*>(start),
             std::variant<uint32_t, tt::tt_metal::Buffer*>(end),
             std::variant<uint32_t, tt::tt_metal::Buffer*>(static_cast<uint32_t>(relay_noc.x)),
             std::variant<uint32_t, tt::tt_metal::Buffer*>(static_cast<uint32_t>(relay_noc.y))});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    carry_extra(desc);
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor ColRowReduceScatterProgramFactory::create_workload_descriptor(
    const ColRowReduceScatterParams& args,
    const ColRowReduceScatterInputs& tensor_args,
    ColRowReduceScatterResult& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    const auto coords = tensor_coords.coords();
    workload.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord), build_program_descriptor(args, coord, tensor_args, output_tensors)});
    }
    return workload;
}

ColRowReduceScatterMeshWorkloadFactory::cached_mesh_workload_t ColRowReduceScatterMeshWorkloadFactory::create_mesh_workload(
    const ColRowReduceScatterParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ColRowReduceScatterInputs& tensor_args,
    ColRowReduceScatterResult& output_tensors) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output_tensors);
}

void ColRowReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ColRowReduceScatterParams& args,
    const ColRowReduceScatterInputs& tensor_args,
    ColRowReduceScatterResult& output_tensors) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output_tensors);
}

}  // namespace ttnn::experimental::prim
