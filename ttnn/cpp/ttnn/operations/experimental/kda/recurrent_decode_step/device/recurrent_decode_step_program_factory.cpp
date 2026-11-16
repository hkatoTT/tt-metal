// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_decode_step_program_factory.hpp"
#include "recurrent_decode_step_device_operation_types.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

// Work units of a (user, head) pair's value block (a pair's 4 value blocks are independent): the state's column of 4
// bf16 tiles, the q / k / decay rows (4 tiles wide), v's block (1 tile).  "row" tiles carry a vector in row 0, "col"
// tiles in column 0 (broadcast across columns by the compute); exact copies go through unpack-to-dest (fp32), matmul
// operands through SrcA / SrcB.  (The plain step's; the chunked form's are its own, below.)
constexpr uint32_t CB_S = tt::CBIndex::c_0;       // state in (bf16)
constexpr uint32_t CB_ECOL = tt::CBIndex::c_1;    // decay columns (exact, broadcast)
constexpr uint32_t CB_KCOL = tt::CBIndex::c_2;    // k columns (exact, broadcast)
constexpr uint32_t CB_KROW = tt::CBIndex::c_3;    // k rows (matmul)
constexpr uint32_t CB_QROW = tt::CBIndex::c_4;    // q rows (matmul)
constexpr uint32_t CB_VROW = tt::CBIndex::c_5;    // v rows (exact)
constexpr uint32_t CB_BROW = tt::CBIndex::c_6;    // beta in row 0 (exact)
constexpr uint32_t CB_DS_X = tt::CBIndex::c_16;   // decayed state (exact)
constexpr uint32_t CB_DS_M = tt::CBIndex::c_17;   // decayed state (matmul), CB_DS_X's memory
constexpr uint32_t CB_DELTA = tt::CBIndex::c_19;  // delta rows (exact, broadcast)
constexpr uint32_t CB_SNEW = tt::CBIndex::c_20;   // new state (fp32: o's matmul input)
constexpr uint32_t CB_SOUT = tt::CBIndex::c_21;   // new state out (bf16)
constexpr uint32_t CB_O = tt::CBIndex::c_22;      // q S' out

constexpr uint32_t N = 4;  // tiles per 128

void add_cb(
    ProgramDescriptor& desc,
    const CoreRangeSet& cores,
    uint32_t index,
    uint32_t n_tiles,
    uint32_t tile_size,
    tt::DataFormat format = tt::DataFormat::Float32) {
    desc.cbs.push_back(CBDescriptor{
        .total_size = n_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = tile_size,
        }}},
    });
}

// Two CB indices over the same memory (pushed and popped alike: one pack serves both readers)
void add_shared_cb(
    ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t a, uint32_t b, uint32_t n_tiles, uint32_t tile_size) {
    auto format = [&](uint32_t index) {
        return CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index), .data_format = tt::DataFormat::Float32, .page_size = tile_size};
    };
    desc.cbs.push_back(CBDescriptor{
        .total_size = n_tiles * tile_size, .core_ranges = cores, .format_descriptors = {format(a), format(b)}});
}

// The chunked form's split of the units (pair p, value block j) over the cores: consecutive ranges with the most
// costly core's cost the least, a range's cost its pairs' shared rounds (~3.75 value blocks each) and its value
// blocks, a core's budget scaled by `pct(k)` (the k-th core's speed in percent).  -> (start, count) a core, in core
// order (fewer than the cores when that is cheaper: 64 pairs take one a core rather than 2 - 3 units over two pairs)
template <typename Pct>
std::vector<std::pair<uint32_t, uint32_t>> chunk_split(uint32_t units, uint32_t cores, Pct pct) {
    constexpr uint32_t CP = 15, CB = 4;
    auto ranges = [&](uint32_t budget) {
        std::vector<std::pair<uint32_t, uint32_t>> out;
        for (uint32_t a = 0; a < units;) {
            if (out.size() == cores) {
                return std::vector<std::pair<uint32_t, uint32_t>>{};
            }
            const uint32_t cap = budget * pct(out.size()) / 100;
            uint32_t b = a, cost = 0;
            while (b < units) {
                const uint32_t add = CB + ((b == a || b % 4 == 0) ? CP : 0);
                if (cost + add > cap) {
                    break;
                }
                cost += add;
                ++b;
            }
            if (b == a) {
                return std::vector<std::pair<uint32_t, uint32_t>>{};
            }
            out.emplace_back(a, b - a);
            a = b;
        }
        return out;
    };
    uint32_t lo = CP + CB, hi = 2 * (CP + CB) * units;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (!ranges(mid).empty()) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return ranges(lo);
}

// The chunk's speed by logical grid row in percent: from its measured time a unit at
// 128 users (about 10.2 us on row 1 and 6.0 us on rows 6 - 9; the lower rows' NoC 0 reads from DRAM are
// slower).  A measurement of this grid's placement and this chip set: measure again when either changes.  Used when
// the pairs are more than the cores (128 users: 256 pairs a chip): the units split over the whole grid with each
// core's budget scaled by its row's speed.  With fewer pairs (8 / 32 users) the split is as before.
// The saved rows (`saved`) of a batch go to the slot of the batch's first unit, so where they land follows the split:
// the next step must read them with the same split.  The split depends only on the pairs and the grid (not on the
// tokens, nor on eager or traced runs), and a state's verify steps all have the same pairs.
constexpr uint32_t CHUNK_ROW_PCT[] = {70, 60, 70, 70, 80, 90, 100, 100, 100, 100};

// The plain step's speed by logical grid row in percent, by its form (used with form bit
// 1024): from the cores' ends at 64 and 32 users a row (34 / 17 units a core), the upper rows
// bound by the compute (~5.8 us a unit), the lower ones waiting on their reads of the state from DRAM.
//   form 0 (every row reads on NoC 0; 64 users: rows 0 - 1 end at 271 / 264 us, rows 7 - 9 at 198 - 201);
//   form bit 256 (rows 0 - 2 read on NoC 1, the model's bf16 form 384: the slowest rows 0, 1 and 3, the ones just
//   below the NoC 1 rows), by the units a core: from 30 (1,024 pairs, 64 users a row) and fewer (512 pairs, 32 users).
//   These are used too where bit 1024 puts the top 5 rows on NoC 1 (below; the rows then nearly even on chip 0, the
//   tables still the fastest of the forms measured on the trace).
// Their own tables: the chunked verify form's (CHUNK_ROW_PCT) is another kernel and shape.  Measurements of this grid's
// placement and this chip set: measure again when either changes.
constexpr uint32_t STEP_ROW_PCT[] = {73, 75, 86, 92, 92, 96, 98, 99, 100, 100};
constexpr uint32_t STEP_ROW_PCT_NOC1[] = {90, 90, 94, 91, 92, 96, 99, 99, 100, 100};
constexpr uint32_t STEP_ROW_PCT_NOC1_FEW[] = {78, 81, 91, 77, 86, 92, 96, 96, 99, 100};

// The plain step's units over the cores in proportion to their speed: core k (column-major order) `pct(k)` percent,
// the least T (units at 100 %) with sum_k floor(T pct(k) / 100) >= units, then the excess taken off the cores the
// most loaded for their speed.  -> each core's count (consecutive ranges in core order, as the even split's).
template <typename Pct>
std::vector<uint32_t> row_weighted_counts(uint32_t units, uint32_t cores, Pct pct) {
    auto total = [&](uint32_t t) {
        uint64_t sum = 0;
        for (uint32_t k = 0; k < cores; ++k) {
            sum += static_cast<uint64_t>(t) * pct(k) / 100;
        }
        return sum;
    };
    uint32_t lo = 1, hi = units;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (total(mid) >= units) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    std::vector<uint32_t> counts(cores);
    uint64_t sum = 0;
    for (uint32_t k = 0; k < cores; ++k) {
        counts[k] = static_cast<uint32_t>(static_cast<uint64_t>(lo) * pct(k) / 100);
        sum += counts[k];
    }
    for (; sum > units; --sum) {  // (the core with the largest count / speed loses one)
        uint32_t worst = 0;
        for (uint32_t k = 1; k < cores; ++k) {
            if (static_cast<uint64_t>(counts[k]) * pct(worst) >= static_cast<uint64_t>(counts[worst]) * pct(k)) {
                worst = k;
            }
        }
        --counts[worst];
    }
    return counts;
}

// The chunked verify form (`saved`, kernels *_chunk): a core's units a pair at a time, the
// pair's chunk (the last step's tokens to redo and this step's T) through the shared rounds once, then each of its
// value blocks.  The reader reads the keep codes and the rows (and makes the constant tiles); the writer writes o and
// the state at the start of the step (after the redone tokens) over the state it was read from.
ProgramDescriptor create_chunk_descriptor(
    const RecurrentDecodeStepParams& attrs, const RecurrentDecodeStepTensorArgs& t, Tensor& out_tensor) {
    ProgramDescriptor desc;
    const uint32_t tile = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t pairs = attrs.pairs, tokens = attrs.tokens;
    auto* device = t.state.device();
    auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    const bool by_rows = grid.y == std::size(CHUNK_ROW_PCT);
    const auto split =
        by_rows && pairs > grid.x * grid.y
            ? chunk_split(pairs * N, grid.x * grid.y, [&](uint32_t k) { return CHUNK_ROW_PCT[k % num_cores_y]; })
            : chunk_split(pairs * N, grid.x * grid.y, [](uint32_t) { return 100u; });
    const uint32_t num_cores = split.size();
    // cores 0 .. num_cores - 1 column by column as rectangles (the kernels and configuration by
    // multicast)
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, false);
    const uint32_t s_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const tt::DataFormat s_fmt = tt::DataFormat::Float16_b;
    enum : uint32_t {
        C_KQ = 0, C_AL = 1, C_BT = 2, C_MZ = 3, C_V = 4, C_S = 5, C_CONST = 6, C_CONSTM = 7, C_G = 8, C_KQP = 9,
        C_KM = 10, C_M = 11, C_N = 12, C_P = 13, C_Y = 14, C_YM = 15, C_F = 16, C_TB = 17, C_E = 18, C_C = 19,
        C_KZ = 21, C_GC = 22, C_X1 = 23, C_O = 24, C_SO = 25, C_KEEP = 26, C_SCR = 27
    };
    add_cb(desc, all_cores, C_KQ, 2 * N, tile);
    add_cb(desc, all_cores, C_AL, 2 * N, tile);
    add_cb(desc, all_cores, C_BT, 2, tile);
    add_cb(desc, all_cores, C_MZ, 2, tile);
    add_cb(desc, all_cores, C_V, 2 * N, tile);
    add_cb(desc, all_cores, C_S, 2 * N * N, s_tile, s_fmt);
    add_shared_cb(desc, all_cores, C_CONST, C_CONSTM, 4, tile);  // (the constant tiles, made by the reader)
    add_cb(desc, all_cores, C_G, N, tile);
    add_cb(desc, all_cores, C_KQP, N, tile);
    add_cb(desc, all_cores, C_KM, N, tile);
    add_cb(desc, all_cores, C_M, 1, tile);
    add_cb(desc, all_cores, C_N, 1, tile);
    // (N^2 .. : as many tiles as a pair takes, so that a pair's never straddle the CB's end)
    add_cb(desc, all_cores, C_P, tokens <= 2 ? 1 : tokens <= 4 ? 2 : 3, tile);
    add_shared_cb(desc, all_cores, C_Y, C_YM, 2, tile);
    add_cb(desc, all_cores, C_F, 1, tile);
    add_cb(desc, all_cores, C_TB, 1, tile);
    add_cb(desc, all_cores, C_E, 1, tile);
    add_cb(desc, all_cores, C_C, N, tile);
    add_cb(desc, all_cores, C_KZ, N, tile);
    add_cb(desc, all_cores, C_GC, N, tile);
    add_cb(desc, all_cores, C_X1, 1, tile);
    add_cb(desc, all_cores, C_O, 2, tile);
    add_cb(desc, all_cores, C_SO, 4 * N, s_tile, s_fmt);  // (four blocks: the writer's DRAM writes drain behind)
    // the reader's staging: the last step's saved slot as read, this step's gathered, this step's beta rows, a face of
    // zeros and one of ones
    const uint32_t slot_bytes = 1536 * tokens + 64;
    add_cb(desc, all_cores, C_SCR, 1, 2 * slot_bytes + 512 + 2048);
    Buffer* keep_buffer = t.keep->buffer();
    const uint32_t keep_bytes = tt::round_up(keep_buffer->page_size(), 32u);
    desc.cbs.push_back(CBDescriptor{
        .total_size = keep_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(C_KEEP),
            .data_format = tt::DataFormat::UInt32,
            .page_size = keep_bytes,
        }}},
    });
    Buffer* state_buffer = t.state.buffer();
    Buffer* saved_buffer = t.saved->buffer();
    // (out of place: the last step's start state and rows read from the sources; the reader's runtime args 10 / 11)
    Buffer* state_src = t.source.has_value() ? t.source->buffer() : state_buffer;
    Buffer* saved_src = t.saved_source.has_value() ? t.saved_source->buffer() : saved_buffer;
    std::array<Buffer*, 5> rows = {t.q.buffer(), t.k.buffer(), t.v.buffer(), t.decay.buffer(), t.beta.buffer()};
    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_decode_step/device/kernels/";
    const uint32_t heads = std::max(attrs.heads, 1u);

    // layout: token k of pair p at row k P + p (0) or p T + k (1, decode_prepare's pair_major)
    std::vector<uint32_t> reader_ct = {tokens, pairs, heads, keep_bytes, attrs.pair_major ? 1u : 0u};
    for (auto* b : rows) {
        TensorAccessorArgs(b).append_to(reader_ct);
    }
    for (auto* b : {state_buffer, saved_buffer, keep_buffer}) {
        TensorAccessorArgs(b).append_to(reader_ct);
    }
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_recurrent_decode_chunk.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {tokens, pairs};
    for (auto* b : {state_buffer, out_tensor.buffer()}) {
        TensorAccessorArgs(b).append_to(writer_ct);
    }
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_recurrent_decode_chunk.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    // exact copies into Dest (the rest are matmul operands, transposes through SrcA or bf16 copies through SrcA)
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb : {C_KQ, C_AL, C_BT, C_MZ, C_CONST, C_G, C_M, C_Y, C_GC}) {
        unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/recurrent_decode_chunk.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {};
    compute.defines = {{"KDA_CHUNK_TOKENS", std::to_string(tokens)}};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    // The top n grid rows' cores read on NoC 1 and write on NoC 0 (their reader on BRISC with NoC 1, their writer on
    // NCRISC with NoC 0, as the plain step's form bit 256; the same values).  With T = 8 the top rows' reads on NoC 0
    // set the op's end (the op alone from a trace): 3 rows, 32 users
    // 46.3 -> 40.4 us, 128 users 112.8 -> 108.4 us, 8 users the same.  With T = 4 (128 users) the rows below the
    // swapped ones take their place (100.3 -> 102.3 us at 3 rows): none.
    std::set<CoreCoord> upper;
    const uint32_t up_rows = tokens >= 8 ? 3u : 0u;
    if (up_rows > 0) {
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            const auto xy = device->worker_core_from_logical_core(core);
            if (xy.y >= 2 && xy.y < 2 + up_rows) {
                upper.insert(core);
            }
        }
    }
    KernelDescriptor reader_up = reader, writer_up = writer;
    reader_up.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1};
    writer_up.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0};
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto [start, count] = split[i];
        const bool up = upper.count(core) != 0;
        KernelDescriptor::RTArgList rr;
        for (auto* b : rows) {
            rr.push_back(b);
        }
        for (auto* b : {state_buffer, saved_buffer, keep_buffer}) {
            rr.push_back(b);
        }
        rr.push_back(start);
        rr.push_back(count);
        rr.push_back(state_src);
        rr.push_back(saved_src);
        (up ? reader_up : reader).emplace_runtime_args(core, rr);
        KernelDescriptor::RTArgList wr;
        for (auto* b : {state_buffer, out_tensor.buffer()}) {
            wr.push_back(b);
        }
        wr.push_back(start);
        wr.push_back(count);
        (up ? writer_up : writer).emplace_runtime_args(core, wr);
        KernelDescriptor::RTArgList cr;
        cr.push_back(start);
        cr.push_back(count);
        compute.emplace_runtime_args(core, cr);
    }
    if (upper.empty()) {
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
    } else {
        std::vector<CoreRange> lower_ranges, upper_ranges;
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            (upper.count(core) != 0 ? upper_ranges : lower_ranges).push_back(CoreRange(core, core));
        }
        // (as few rectangles: a program goes out range by range, so a range a core lengthens its dispatch)
        reader.core_ranges = writer.core_ranges = CoreRangeSet(lower_ranges).merge_ranges();
        reader_up.core_ranges = writer_up.core_ranges = CoreRangeSet(upper_ranges).merge_ranges();
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
        desc.kernels.push_back(std::move(reader_up));
        desc.kernels.push_back(std::move(writer_up));
    }
    desc.kernels.push_back(std::move(compute));
    return desc;
}
}  // namespace

ProgramDescriptor RecurrentDecodeStepProgramFactory::create_descriptor(
    const RecurrentDecodeStepParams& operation_attributes,
    const RecurrentDecodeStepTensorArgs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    if (tensor_args.saved.has_value()) {
        return create_chunk_descriptor(operation_attributes, tensor_args, tensor_return_value);
    }
    // The plain step: one token a pair
    ProgramDescriptor desc;
    const uint32_t tile_size = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t pairs = operation_attributes.pairs;
    const uint32_t form = operation_attributes.form;

    auto* state_buffer = tensor_args.state.buffer();
    auto* q_buffer = tensor_args.q.buffer();
    auto* k_buffer = tensor_args.k.buffer();
    auto* v_buffer = tensor_args.v.buffer();
    auto* decay_buffer = tensor_args.decay.buffer();
    auto* beta_buffer = tensor_args.beta.buffer();
    auto* out_buffer = tensor_return_value.buffer();

    auto grid = tensor_args.state.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(grid, pairs * N);

    // The reader reads a pair's units on a core at once (up to N state columns, N v blocks), double-buffered.  The
    // state kept in bf16: CB_S and CB_SOUT bf16 tiles; CB_SNEW (o's matmul input) a CB of its own, fp32
    const uint32_t s_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const tt::DataFormat s_fmt = tt::DataFormat::Float16_b;
    add_cb(desc, all_cores, CB_S, 2 * N * N, s_tile, s_fmt);
    add_cb(desc, all_cores, CB_ECOL, 2 * N, tile_size);
    add_cb(desc, all_cores, CB_KCOL, 2 * N, tile_size);
    add_cb(desc, all_cores, CB_KROW, 2 * N, tile_size);
    add_cb(desc, all_cores, CB_QROW, 2 * N, tile_size);
    add_cb(desc, all_cores, CB_VROW, 2 * N, tile_size);
    add_cb(desc, all_cores, CB_BROW, 2, tile_size);
    add_shared_cb(desc, all_cores, CB_DS_X, CB_DS_M, N, tile_size);
    add_cb(desc, all_cores, CB_DELTA, 1, tile_size);
    add_cb(desc, all_cores, CB_SNEW, 2 * N, tile_size);
    add_cb(desc, all_cores, CB_SOUT, 2 * N, s_tile, s_fmt);
    add_cb(desc, all_cores, CB_O, 2, tile_size);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_decode_step/device/kernels/";

    std::vector<uint32_t> reader_ct = {CB_S, CB_ECOL, CB_KCOL, CB_KROW, CB_QROW, CB_VROW, CB_BROW};
    for (auto* b : {state_buffer, q_buffer, k_buffer, v_buffer, decay_buffer, beta_buffer}) {
        TensorAccessorArgs(b).append_to(reader_ct);
    }
    reader_ct.push_back(form);  // (bit 128: the reader's)
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_recurrent_decode_step.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_SOUT, CB_O};
    TensorAccessorArgs(state_buffer).append_to(writer_ct);
    TensorAccessorArgs(out_buffer).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_recurrent_decode_step.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};
    if ((form & 128) != 0) {
        writer.defines.emplace_back("KDA_STEP_ROTATE", "1");
    }

    // exact copies into Dest (the bf16 state tiles go through SrcA into the fp32 Dest: exact)
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb : {CB_ECOL, CB_KCOL, CB_VROW, CB_BROW, CB_DS_X, CB_DELTA}) {
        unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/recurrent_decode_step.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {};

    // form bit 256: the top 3 rows' cores (virtual y 2-4) read on NoC 1: their reader on BRISC with NoC 1, their
    // writer on NCRISC with NoC 0 (each RISC still its own NoC); the same kernels, two descriptors each.
    // With form bit 1024 (the units by grid row speed, below) the top 5 rows (virtual y 2-6) in place of 3: the reads
    // more even over the two NoCs (1,024 / 512 pairs 246.4 / 157.4 -> 230.4 / 137.0 us with the same tables; 4 rows
    // 240.3 / 143.0, 6 rows 232.9 / 147.3)
    std::set<CoreCoord> upper;
    const uint32_t upper_last = (form & 1024) != 0 ? 6 : 4;
    if ((form & 256) != 0) {
        auto* device = tensor_args.state.device();
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            const auto xy = device->worker_core_from_logical_core(core);
            if (xy.y >= 2 && xy.y <= upper_last) {
                upper.insert(core);
            }
        }
    }
    KernelDescriptor reader_up = reader, writer_up = writer;
    reader_up.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1};
    writer_up.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    // each core's units: the even split, or with form bit 1024 and more units than cores in proportion to its grid
    // row's speed (STEP_ROW_PCT, or STEP_ROW_PCT_NOC1(_FEW) with bit 256).  The same units with the same computation on
    // other cores: the same values.
    std::vector<uint32_t> counts(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        counts[i] = core_group_1.contains(core) ? units_per_core_1 : units_per_core_2;
    }
    if ((form & 1024) != 0 && num_cores_y == std::size(STEP_ROW_PCT) && pairs * N > num_cores) {
        const uint32_t* pct = (form & 256) == 0           ? STEP_ROW_PCT
                              : pairs * N >= 30 * num_cores ? STEP_ROW_PCT_NOC1
                                                            : STEP_ROW_PCT_NOC1_FEW;
        counts = row_weighted_counts(pairs * N, num_cores, [&](uint32_t k) { return pct[k % num_cores_y]; });
    }

    // Buffer* first so cache hits patch the addresses.
    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t count = counts[i];
        {
            KernelDescriptor::RTArgList rt;
            for (auto* b : {state_buffer, q_buffer, k_buffer, v_buffer, decay_buffer, beta_buffer}) {
                rt.push_back(b);
            }
            rt.push_back(start);
            rt.push_back(count);
            (upper.count(core) != 0 ? reader_up : reader).emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(state_buffer);
            rt.push_back(out_buffer);
            rt.push_back(start);
            rt.push_back(count);
            (upper.count(core) != 0 ? writer_up : writer).emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(start);
            rt.push_back(count);
            compute.emplace_runtime_args(core, rt);
        }
        start += count;
    }

    if (upper.empty()) {
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
    } else {
        std::vector<CoreRange> lower_ranges, upper_ranges;
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            (upper.count(core) != 0 ? upper_ranges : lower_ranges).push_back(CoreRange(core, core));
        }
        reader.core_ranges = writer.core_ranges = CoreRangeSet(lower_ranges).merge_ranges();
        reader_up.core_ranges = writer_up.core_ranges = CoreRangeSet(upper_ranges).merge_ranges();
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
        desc.kernels.push_back(std::move(reader_up));
        desc.kernels.push_back(std::move(writer_up));
    }
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
