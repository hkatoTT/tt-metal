// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_finish_program_factory.hpp"
#include "decode_finish_device_operation_types.hpp"

#include <algorithm>
#include <bit>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
// Work units: head h, block of 16 users (the rows of a face pair of the output's tile row).  The block's o rows (one
// per (user, head) pair) gathered into rows 0-15 of 4 tiles; the gate's rows read as whole bf16 faces (widened exactly
// by the unpacker).  The output is rounded to bf16 in Dest and packed as bf16; the writer writes whole faces.
constexpr uint32_t CB_O = tt::CBIndex::c_0;     // o rows
constexpr uint32_t CB_G = tt::CBIndex::c_1;     // the gate (bf16)
constexpr uint32_t CB_W = tt::CBIndex::c_2;     // o_norm in row 0 (resident)
constexpr uint32_t CB_ONES = tt::CBIndex::c_3;  // all ones
constexpr uint32_t CB_R = tt::CBIndex::c_16;    // o rounded to bf16 (held in fp32)
constexpr uint32_t CB_SQ = tt::CBIndex::c_17;   // the squares' partial sums
constexpr uint32_t CB_INV = tt::CBIndex::c_18;  // the norm's inverse
constexpr uint32_t CB_OUT = tt::CBIndex::c_19;  // the gated norm (bf16)
constexpr uint32_t N = 4;

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

ProgramDescriptor DecodeFinishProgramFactory::create_descriptor(
    const DecodeFinishParams& attrs,
    const DecodeFinishTensorArgs& t,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const uint32_t heads = attrs.heads;
    const uint32_t users = t.fg.logical_shape()[-2];
    const uint32_t blocks = (users + 15) / 16;
    const uint32_t fg_tiles = t.fg.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t out_tiles = heads * N;

    auto* o_buffer = t.o.buffer();
    auto* fg_buffer = t.fg.buffer();
    auto* w_buffer = t.o_norm.buffer();
    auto* out_buffer = output.buffer();

    auto grid = t.o.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    const uint32_t units = heads * blocks;
    // The pieces (unit, first row, rows) of each core: when the units fit the cores, each unit in s pieces of whole
    // 4-row groups (rows independent: the same values; the reads and writes of a piece whole 4-row groups of a face,
    // 128-byte aligned), a piece a core (the verify step -0.01 / -0.14 / -0.05 ms at 8 / 32 /
    // 128 users); else whole units (split_work_to_cores).
    struct Piece {
        uint32_t unit, row0, rows;
    };
    std::vector<std::vector<Piece>> per_core;
    auto unit_rows = [&](uint32_t u) { return std::min(16u, users - (u % blocks) * 16); };
    const uint32_t n_cores = grid.x * grid.y;
    if (units <= n_cores) {
        const uint32_t s = std::clamp(n_cores / units, 1u, 4u);
        for (uint32_t u = 0; u < units; ++u) {
            const uint32_t rows = unit_rows(u), groups = (rows + 3) / 4, per = 4 * ((groups + s - 1) / s);
            for (uint32_t r0 = 0; r0 < rows; r0 += per) {
                per_core.push_back({Piece{u, r0, std::min(per, rows - r0)}});
            }
        }
    } else {
        auto [n, all, g1, g2, c1, c2] = split_work_to_cores(grid, units);
        for (uint32_t i = 0, start = 0; i < n; ++i) {
            const CoreCoord core = {i / num_cores_y, i % num_cores_y};
            const uint32_t count = g1.contains(core) ? c1 : c2;
            per_core.emplace_back();
            for (uint32_t u = start; u < start + count; ++u) {
                per_core.back().push_back(Piece{u, 0, unit_rows(u)});
            }
            start += count;
        }
    }
    const uint32_t num_cores = per_core.size();
    uint32_t rows_max = 1;  // the compute's rows (a piece's, at most 16)
    for (const auto& pieces : per_core) {
        for (const auto& pc : pieces) {
            rows_max = std::max(rows_max, pc.rows);
        }
    }
    // (cores 0 .. num_cores - 1 column by column, as split_work_to_cores: rectangles, so that the program's kernels and
    // configuration go out by multicast; a set of single cores made the launch ~10 us slower at 64 cores)
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, false);

    add_cb(desc, all_cores, CB_O, 2 * N, f32);
    add_cb(desc, all_cores, CB_G, 2 * N, bf16);
    add_cb(desc, all_cores, CB_W, N, f32);
    add_cb(desc, all_cores, CB_ONES, 1, f32);
    add_cb(desc, all_cores, CB_R, N, f32);
    add_cb(desc, all_cores, CB_SQ, 1, f32);
    add_cb(desc, all_cores, CB_INV, 1, f32);
    add_cb(desc, all_cores, CB_OUT, 2 * N, bf16);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/kda/decode_finish/device/kernels/";

    std::vector<uint32_t> reader_ct = {CB_O, CB_G, CB_W, CB_ONES, heads, users, fg_tiles};
    TensorAccessorArgs(o_buffer).append_to(reader_ct);
    TensorAccessorArgs(fg_buffer).append_to(reader_ct);
    TensorAccessorArgs(w_buffer).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_decode_finish.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_OUT, users, out_tiles};
    TensorAccessorArgs(out_buffer).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_decode_finish.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb : {CB_O, CB_W, CB_R, CB_INV}) {
        unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/decode_finish.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {std::bit_cast<uint32_t>(attrs.eps), rows_max};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto& pieces = per_core[i];
        KernelDescriptor::RTArgList rr, wr, cr;
        rr.push_back(o_buffer);
        rr.push_back(fg_buffer);
        rr.push_back(w_buffer);
        rr.push_back(static_cast<uint32_t>(pieces.size()));
        wr.push_back(out_buffer);
        wr.push_back(static_cast<uint32_t>(pieces.size()));
        for (const auto& pc : pieces) {
            for (auto* l : {&rr, &wr}) {
                l->push_back(pc.unit);
                l->push_back(pc.row0);
                l->push_back(pc.rows);
            }
        }
        cr.push_back(static_cast<uint32_t>(pieces.size()));
        reader.emplace_runtime_args(core, rr);
        writer.emplace_runtime_args(core, wr);
        compute.emplace_runtime_args(core, cr);
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
