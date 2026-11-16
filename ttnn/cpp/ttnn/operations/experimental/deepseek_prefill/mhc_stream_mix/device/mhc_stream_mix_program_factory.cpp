// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_mix_program_factory.hpp"

#include <algorithm>
#include "mhc_stream_mix_device_operation_types.hpp"

#include <bit>
#include <cstdlib>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
// One core per group of 32 tokens (one token tile, 128 stream rows; R above 128 in whole groups: group g on core
// (0, g) reads its row tiles 4 g .. 4 g + 3 of the partials and writes token tile row g of post, comb and P, its P
// [32, 128] the weights of its own 128 rows).  See the compute kernel for the steps.
constexpr uint32_t CB_PART = tt::CBIndex::c_0;
constexpr uint32_t CB_MC = tt::CBIndex::c_1;
constexpr uint32_t CB_SC = tt::CBIndex::c_2;
constexpr uint32_t CB_POST = tt::CBIndex::c_3;
constexpr uint32_t CB_COMB = tt::CBIndex::c_4;
constexpr uint32_t CB_P = tt::CBIndex::c_5;
constexpr uint32_t MAX_ROW_TILES = 4;
constexpr uint32_t PART_CHUNK = 16;  // partial tiles per read batch

void add_cb(ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t index, uint32_t n_tiles) {
    const uint32_t tile_size = tt::tile_size(tt::DataFormat::Float32);
    desc.cbs.push_back(CBDescriptor{
        .total_size = n_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = tt::DataFormat::Float32,
            .page_size = tile_size}}},
    });
}
}  // namespace

ProgramDescriptor MhcStreamMixProgramFactory::create_descriptor(
    const MhcStreamMixParams& attrs,
    const MhcStreamMixTensorArgs& tensor_args,
    tensor_return_value_t& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    auto* partial_buffer = tensor_args.partial.buffer();
    auto* mix_consts_buffer = tensor_args.mix_consts.buffer();
    auto* consts_buffer = tensor_args.consts.buffer();
    auto* post_buffer = outputs[0].buffer();
    auto* comb_buffer = outputs[1].buffer();
    auto* p_buffer = outputs[2].buffer();

    const uint32_t slices = tensor_args.partial.logical_shape()[0];
    const uint32_t total_rt = tensor_args.partial.logical_shape()[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t groups = total_rt > MAX_ROW_TILES ? total_rt / MAX_ROW_TILES : 1;
    const uint32_t row_tiles = total_rt / groups;  // a group's
    const uint32_t tokens = groups > 1 ? tt::constants::TILE_HEIGHT : attrs.tokens;  // a group's
    const CoreRangeSet core{CoreRange(CoreCoord{0, 0}, CoreCoord{0, groups - 1})};

    // the partials are read CHUNK tiles at a time with one barrier (double-buffered)
    const uint32_t chunk = std::min(row_tiles * 4 * slices, PART_CHUNK);
    add_cb(desc, core, CB_PART, 2 * chunk);
    add_cb(desc, core, CB_MC, 13);
    add_cb(desc, core, CB_SC, 8);
    add_cb(desc, core, CB_POST, 1);
    add_cb(desc, core, CB_COMB, 1);
    add_cb(desc, core, CB_P, MAX_ROW_TILES);
    add_cb(desc, core, tt::CBIndex::c_16, 4 * MAX_ROW_TILES);  // the summed partials (proj 0-2, Gram) per row tile
    add_cb(desc, core, tt::CBIndex::c_17, 3 * MAX_ROW_TILES);  // the own-block masked projections
    add_cb(desc, core, tt::CBIndex::c_18, MAX_ROW_TILES);      // the Gram diagonals
    add_cb(desc, core, tt::CBIndex::c_19, 3);                  // own
    add_cb(desc, core, tt::CBIndex::c_20, 1);                  // the squares per token, spread
    add_cb(desc, core, tt::CBIndex::c_21, 1);                  // mixes before the scale
    add_cb(desc, core, tt::CBIndex::c_22, 1);                  // the scale
    add_cb(desc, core, tt::CBIndex::c_23, 1);                  // mixes
    add_cb(desc, core, tt::CBIndex::c_24, 2);                  // Sinkhorn ping
    add_cb(desc, core, tt::CBIndex::c_25, 2);                  // Sinkhorn pong
    add_cb(desc, core, tt::CBIndex::c_26, 1);                  // pre
    add_cb(desc, core, tt::CBIndex::c_27, 1);                  // pre spread over the rows' lanes

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_mix/device/kernels/";

    std::vector<uint32_t> reader_ct = {CB_PART, CB_MC, CB_SC, slices, row_tiles, chunk, total_rt};
    TensorAccessorArgs(partial_buffer).append_to(reader_ct);
    TensorAccessorArgs(mix_consts_buffer).append_to(reader_ct);
    TensorAccessorArgs(consts_buffer).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_mhc_stream_mix.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = core;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_POST, CB_COMB, CB_P, row_tiles};
    TensorAccessorArgs(post_buffer).append_to(writer_ct);
    TensorAccessorArgs(comb_buffer).append_to(writer_ct);
    TensorAccessorArgs(p_buffer).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_mhc_stream_mix.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = core;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/mhc_stream_mix.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = core;
    compute.compile_time_args = {
        slices,
        row_tiles,
        attrs.iters,
        std::bit_cast<uint32_t>(attrs.eps),
        std::bit_cast<uint32_t>(attrs.norm_eps),
        std::bit_cast<uint32_t>(attrs.inv_count),
        tokens,
        // the Sinkhorn of comb on the SFPU in Dest (mhc_stream_mix.cpp `sinkhorn_sfpu`: 22 us faster a call, comb
        // 8x closer to float64) unless MHC_SINKHORN_SFPU=0, which keeps the matmul form's pack / unpack per normalization
        static_cast<uint32_t>(std::getenv("MHC_SINKHORN_SFPU") == nullptr || std::getenv("MHC_SINKHORN_SFPU")[0] != '0')};
    compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};

    for (uint32_t g = 0; g < groups; ++g) {
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(partial_buffer);
            rt.push_back(mix_consts_buffer);
            rt.push_back(consts_buffer);
            rt.push_back(g * row_tiles);  // the group's first row tile
            reader.emplace_runtime_args(CoreCoord{0, g}, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(post_buffer);
            rt.push_back(comb_buffer);
            rt.push_back(p_buffer);
            rt.push_back(g);  // the group's token tile row
            writer.emplace_runtime_args(CoreCoord{0, g}, rt);
        }
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
