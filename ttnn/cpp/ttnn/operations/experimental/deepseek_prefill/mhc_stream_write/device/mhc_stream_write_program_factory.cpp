// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_write_program_factory.hpp"
#include "mhc_stream_write_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
// Work units are the output tiles (row tile rt: 8 tokens' 4 streams; column tile c).  Per row tile the reader builds W
// (the 8 tokens' comb^T blocks on the diagonal) and Q (each token's post against its block_out row), bf16 as HF rounds
// them; per unit the compute takes bf16(W @ S) + bf16(Q @ BO).
constexpr uint32_t CB_S = tt::CBIndex::c_0;      // streams tile
constexpr uint32_t CB_BO = tt::CBIndex::c_1;     // block_out tile (rows past the tokens zeroed)
constexpr uint32_t CB_W = tt::CBIndex::c_2;      // comb^T blocks
constexpr uint32_t CB_Q = tt::CBIndex::c_3;      // post
constexpr uint32_t CB_SCR = tt::CBIndex::c_4;    // the reader's comb / post rows (fp32)
constexpr uint32_t CB_T1 = tt::CBIndex::c_16;    // W @ S
constexpr uint32_t CB_T2 = tt::CBIndex::c_17;    // Q @ BO
constexpr uint32_t CB_OUT = tt::CBIndex::c_18;   // the sum

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

ProgramDescriptor MhcStreamWriteProgramFactory::create_descriptor(
    const MhcStreamWriteParams& operation_attributes,
    const MhcStreamWriteTensorArgs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;

    auto* streams_buffer = tensor_args.streams.buffer();
    auto* block_out_buffer = tensor_args.block_out.buffer();
    auto* post_buffer = tensor_args.post.buffer();
    auto* comb_buffer = tensor_args.comb.buffer();
    auto* out_buffer = tensor_return_value.buffer();

    const uint32_t d_tiles = tensor_args.streams.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t row_tiles = tensor_args.streams.logical_shape()[-2] / tt::constants::TILE_HEIGHT;
    auto grid = tensor_args.streams.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(grid, row_tiles * d_tiles);

    add_cb(desc, all_cores, CB_S, 2, bf16);
    add_cb(desc, all_cores, CB_BO, 2, bf16);
    add_cb(desc, all_cores, CB_W, 2, bf16);
    add_cb(desc, all_cores, CB_Q, 2, bf16);
    add_cb(desc, all_cores, CB_SCR, 1, f32);
    add_cb(desc, all_cores, CB_T1, 1, bf16);
    add_cb(desc, all_cores, CB_T2, 1, bf16);
    add_cb(desc, all_cores, CB_OUT, 2, bf16);

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_write/device/kernels/";

    std::vector<uint32_t> reader_ct = {
        CB_S, CB_BO, CB_W, CB_Q, CB_SCR, d_tiles, operation_attributes.tokens};
    for (auto* b : {streams_buffer, block_out_buffer, post_buffer, comb_buffer}) {
        TensorAccessorArgs(b).append_to(reader_ct);
    }
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_mhc_stream_write.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_OUT, 0, 0, 0, 0, 0, 0};  // (no hand-off to a read's slices)
    TensorAccessorArgs(out_buffer).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_mhc_stream_write.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/mhc_stream_write.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {d_tiles, CB_S, CB_BO, CB_W, CB_Q, CB_T1, CB_T2, CB_OUT};
    compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};

    // Buffer* first so cache hits patch the addresses.
    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t count = core_group_1.contains(core) ? units_per_core_1 : units_per_core_2;
        {
            KernelDescriptor::RTArgList rt;
            for (auto* b : {streams_buffer, block_out_buffer, post_buffer, comb_buffer}) {
                rt.push_back(b);
            }
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
            rt.push_back(start);
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
