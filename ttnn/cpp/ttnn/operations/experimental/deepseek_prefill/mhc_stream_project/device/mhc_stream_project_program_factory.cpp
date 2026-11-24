// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_project_program_factory.hpp"
#include "mhc_stream_project_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
// Work units (K slice s, row tile rt): the slice's streams tiles S and fn tiles, -> the partial [32 x 128] fp32: the
// 96 projection columns (S @ fn) and the Gram tile S @ S^T (its diagonal: each row's sum of squares).  bf16 products
// are exact and the matmuls sum in fp32.
constexpr uint32_t CB_S = tt::CBIndex::c_0;
constexpr uint32_t CB_FN = tt::CBIndex::c_1;
constexpr uint32_t CB_OUT = tt::CBIndex::c_16;
constexpr uint32_t PROJ_TILES = 3;
constexpr uint32_t OUT_TILES = PROJ_TILES + 1;

void add_cb(ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t index, uint32_t n_tiles, tt::DataFormat df) {
    const uint32_t tile_size = tt::tile_size(df);
    desc.cbs.push_back(CBDescriptor{
        .total_size = n_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index), .data_format = df, .page_size = tile_size}}},
    });
}
}  // namespace

ProgramDescriptor MhcStreamProjectProgramFactory::create_descriptor(
    const MhcStreamProjectParams& operation_attributes,
    const MhcStreamProjectTensorArgs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    auto* streams_buffer = tensor_args.streams.buffer();
    auto* fn_buffer = tensor_args.fn.buffer();
    auto* out_buffer = tensor_return_value.buffer();

    const uint32_t d_tiles = tensor_args.streams.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t row_tiles = tensor_args.streams.logical_shape()[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t slices = operation_attributes.slices;
    const uint32_t k_tiles = d_tiles / slices;

    auto grid = tensor_args.streams.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(grid, slices * row_tiles);

    add_cb(desc, all_cores, CB_S, k_tiles, tt::DataFormat::Float16_b);
    add_cb(desc, all_cores, CB_FN, k_tiles * PROJ_TILES, tt::DataFormat::Float16_b);
    add_cb(desc, all_cores, CB_OUT, OUT_TILES, tt::DataFormat::Float32);

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_project/device/kernels/";

    std::vector<uint32_t> reader_ct = {CB_S, CB_FN, d_tiles, row_tiles, k_tiles};
    TensorAccessorArgs(streams_buffer).append_to(reader_ct);
    TensorAccessorArgs(fn_buffer).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_mhc_stream_project.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_OUT, row_tiles};
    TensorAccessorArgs(out_buffer).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_mhc_stream_project.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/mhc_stream_project.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {k_tiles};
    compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};

    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t count = core_group_1.contains(core) ? units_per_core_1 : units_per_core_2;
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(streams_buffer);
            rt.push_back(fn_buffer);
            rt.push_back(start);
            rt.push_back(count);
            reader.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(out_buffer);
            rt.push_back(start);
            rt.push_back(count);
            writer.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(count);
            compute.emplace_runtime_args(core, rt);
        }
        start += count;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
