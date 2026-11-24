// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode_route_program_factory.hpp"
#include "moe_decode_route_device_operation_types.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
const std::string KDIR = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_decode_route/device/kernels/";

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

// Route: the real tokens' rows split over the cores; a core takes each tile row its rows touch (the E / 32 logits
// tiles), the compute makes sigmoid(logits) and that plus the bias for the whole tile row, and the writer picks each
// of its rows' top k (RISC-V, integer keys), normalizes the chosen scores and writes the chip's local columns.  Core 0
// also writes the rows after the real tokens (zeros).  Packed (E up to 1,024, up to 8 rows a core; MOE_ROUTE_PACK=0:
// off): the reader takes only the core's rows, 1024 / E of them a tile (each row's E values 2 x 64 B from each of its
// E / 32 tiles), so that the compute's tiles are its rows' and not the 9 of each tile row (bit for bit the same):
// 8 / 32 / 128 / 512 tokens 29.1 / 33.0 / 54.0 / 79.7 -> 23.5 / 24.1 / 35.0 / 66.6 us, 1,024 (9 rows a core) 109.6 ->
// 112.3.
ProgramDescriptor MoeDecodeRouteProgramFactory::create_descriptor(
    const MoeDecodeRouteParams& attrs,
    const MoeDecodeRouteTensorArgs& t,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    constexpr uint32_t CB_LOG = tt::CBIndex::c_0;   // a tile row of logits
    constexpr uint32_t CB_BIAS = tt::CBIndex::c_1;  // the bias (row 0)
    constexpr uint32_t CB_SC = tt::CBIndex::c_16;   // sigmoid(logits)
    constexpr uint32_t CB_CH = tt::CBIndex::c_17;   // sigmoid(logits) + bias, the choice keys
    constexpr uint32_t CB_W = tt::CBIndex::c_18;    // the writer's output rows, its local base, the weights' rows

    ProgramDescriptor desc;
    const auto f32 = tt::DataFormat::Float32;
    const uint32_t n = t.logits.logical_shape()[-2];
    const uint32_t experts = t.logits.logical_shape()[-1];
    const uint32_t et = experts / tt::constants::TILE_WIDTH;

    auto grid = t.logits.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        split_work_to_cores(grid, attrs.tokens);

    const char* pack_env = std::getenv("MOE_ROUTE_PACK");
    const uint32_t rpt = 1024 / experts;  // packed rows a tile
    const bool pack = (pack_env == nullptr || pack_env[0] != '0') && rpt >= 1 &&
                      std::max(rows_per_core_1, rows_per_core_2) <= 8;
    const uint32_t np = pack ? (std::max(rows_per_core_1, rows_per_core_2) + rpt - 1) / rpt : et;  // tiles a core
    add_cb(desc, all_cores, CB_LOG, np, f32);
    add_cb(desc, all_cores, CB_BIAS, pack ? 1 : et, f32);
    add_cb(desc, all_cores, CB_SC, np, f32);
    add_cb(desc, all_cores, CB_CH, np, f32);
    add_cb(desc, all_cores, CB_W, 2, f32);

    Buffer* logits = t.logits.buffer();
    Buffer* bias = t.bias.buffer();
    Buffer* base = t.local_base.buffer();
    Buffer* out = output.buffer();

    std::vector<uint32_t> reader_ct = {CB_LOG, CB_BIAS, et, pack ? 1u : 0u, rpt};
    TensorAccessorArgs(logits).append_to(reader_ct);
    TensorAccessorArgs(bias).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = KDIR + "dataflow/reader_moe_decode_route.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    // the progressive form (GLM53_MOE_PROG_F, a quality-changing option): every row's k picks' global ids too (+ 1, two
    // to a word) in id columns 12 .. 15 with weights 0, for the experts op's builders to count every mesh column's
    // chosen experts (the budget of the busiest column, moe_decode_experts' route_build.cpp); read at the program's
    // creation (not in the program cache's key)
    const uint32_t globals = std::getenv("GLM53_MOE_PROG_F") != nullptr && attrs.k <= 8 ? 1u : 0u;
    std::vector<uint32_t> writer_ct = {
        CB_SC,
        CB_CH,
        CB_W,
        et,
        attrs.k | (globals << 16),
        attrs.local,
        std::bit_cast<uint32_t>(attrs.scale),
        static_cast<uint32_t>(base->aligned_page_size()),
        attrs.shared ? 1u : 0u,
        pack ? 1u : 0u,
        rpt};
    TensorAccessorArgs(base).append_to(writer_ct);
    TensorAccessorArgs(out).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = KDIR + "dataflow/writer_moe_decode_route.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[CB_LOG] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[CB_BIAS] = UnpackToDestMode::UnpackToDestFp32;
    KernelDescriptor compute;
    compute.kernel_source = KDIR + "compute/moe_decode_route.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {et, pack ? 1u : 0u, rpt};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t count = core_group_1.contains(core) ? rows_per_core_1 : rows_per_core_2;
        const uint32_t zeros_end = i == 0 ? n : attrs.tokens;  // core 0: rows tokens .. n - 1 are zeros
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(logits);
            rt.push_back(bias);
            rt.push_back(start);
            rt.push_back(count);
            reader.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(base);
            rt.push_back(out);
            rt.push_back(start);
            rt.push_back(count);
            rt.push_back(attrs.tokens);
            rt.push_back(zeros_end);
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
