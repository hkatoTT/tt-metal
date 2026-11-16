// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_prepare_program_factory.hpp"
#include "decode_prepare_device_operation_types.hpp"

#include <algorithm>
#include <array>
#include <bit>
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

// Work units: part (0 q, 1 k, 2 v, 3 the gates: decay and beta), head h, block of 16 users (the rows of a face pair of
// a [B, ..] tile row, moved to rows 0-15 by the reader).  A part's 128 channels are 4 tiles.
constexpr uint32_t CB_CUR = tt::CBIndex::c_0;    // the conv input (proj's bf16 rows)
constexpr uint32_t CB_H0 = tt::CBIndex::c_1;     // the conv history, oldest first (fp32)
constexpr uint32_t CB_H1 = tt::CBIndex::c_2;
constexpr uint32_t CB_H2 = tt::CBIndex::c_3;
constexpr uint32_t CB_TAP = tt::CBIndex::c_4;    // the 4 taps' rows (fp32)
constexpr uint32_t CB_F = tt::CBIndex::c_5;      // f (bf16)
constexpr uint32_t CB_DT = tt::CBIndex::c_6;     // dt_bias (fp32)
constexpr uint32_t CB_RATE = tt::CBIndex::c_7;   // exp(A_log) (fp32)
constexpr uint32_t CB_BETA = tt::CBIndex::c_8;   // the beta logit at (0, 0) (in fp32)
constexpr uint32_t CB_ONES = tt::CBIndex::c_9;   // all ones (fp32)
constexpr uint32_t CB_STG = tt::CBIndex::c_10;   // the reader's beta logits (bf16) before their move to (i, 0)
constexpr uint32_t CB_QKV = tt::CBIndex::c_16;   // SiLU(conv) rounded to bf16 (held in fp32)
constexpr uint32_t CB_SQ = tt::CBIndex::c_17;    // the squares' partial sums
constexpr uint32_t CB_INV = tt::CBIndex::c_18;   // the norm's inverse
constexpr uint32_t CB_OUT = tt::CBIndex::c_19;   // the part's output rows (fp32)
constexpr uint32_t CB_BOUT = tt::CBIndex::c_20;  // sigmoid(beta) rounded to bf16 (held in fp32)
constexpr uint32_t CB_WSCR = tt::CBIndex::c_21;  // the writer's beta row (fp32)
constexpr uint32_t CB_CURH = tt::CBIndex::c_22;  // the conv input in fp32, for history 2 (the writer writes it)
constexpr uint32_t CB_KEEP = tt::CBIndex::c_23;  // the keep flags, one a user (MTP's verify step)
constexpr uint32_t CB_ROWS = tt::CBIndex::c_24;  // scratch: proj / fg faces read whole for a later token's rows
constexpr uint32_t CB_SAVED = tt::CBIndex::c_25;  // scratch: the history checkpoint's rows (with keep)
constexpr uint32_t CB_XT = tt::CBIndex::c_12;     // more tokens: the earlier tokens' inputs in fp32 (the later history)
// the stacked form (`stack_tokens`): every token's conv input rows (bf16, the unit's whole tile row) and its users'
// history then inputs as one column of fp32 rows a channel tile (the h0 h1 h2 of a pass its runs)
constexpr uint32_t CB_SEQ = tt::CBIndex::c_28, CB_HST = tt::CBIndex::c_29;
// split_channels: q / k's partial sum of squares over the unit-half's 2 tiles (sent to the other half's core) and the
// other half's, received (2 slots: a pass apart at most)
constexpr uint32_t CB_SQP = tt::CBIndex::c_30, CB_SQX = tt::CBIndex::c_31;
constexpr uint32_t N = 4;

// The stacked form: with few users a row (a unit's users times its tokens over 16 rows), a pass takes several tokens
// at once, token k's users in rows (k - first) users .. of the pass (as more users of one token: the conv's history
// rows of each made by the reader from those users' history and the earlier tokens' inputs), so that the compute runs
// ceil(tokens / G) passes in place of `tokens` (G = the tokens a pass, 16 / users).  The same values (the same terms in
// the same order per row).  Prepare + delta rule (op alone): 1 / 2 / 4 users a row x T = 8 157.5 / 237.8 / 296.0
// -> 110.7 / 204.4 / 248.5 us, T = 4 93.0 / 130.5 / 146.6 -> 74.0 / 114.3 / 144.0, T = 2 62.5 / 77.8 / 92.4 -> 57.7 / 73.3
// / 91.8.  0: off (one token, more than 4 users, or more than 32 rows).
uint32_t stack_tokens(uint32_t users, uint32_t tokens) {
    // (users 1, 2 or 4: a pass's first row at a face's start for the reader's whole-face reads, and at least 4 tokens a
    // pass: with 8 users a row, 2 tokens a pass, it was no faster)
    if (tokens < 2 || users > 4 || 16 % users != 0 || users * tokens > 32) {
        return 0;
    }
    const uint32_t g = std::min(16u / users, tokens);
    return g >= 2 ? g : 0;
}

// row_split: the plain call's speed by logical grid row in percent (chip 0, a unit's time (core end - 5 us) / units by
// row), by the units a core: 2 and more (64 users a row: 256 units, 120 cores) and fewer (32 users: 128 units; either
// table gives the 8 units over one a core to the fastest row), with the top rows reading on NoC 1 (below; measured with
// the top 5 rows on NoC 1, the even split's 2-unit cores; with every row on NoC 0 the lower rows starve)
constexpr uint32_t UPPER_ROWS = 6;
constexpr uint32_t ROW_PCT[] = {65, 65, 69, 72, 72, 66, 74, 82, 92, 100};
constexpr uint32_t ROW_PCT_FEW[] = {48, 50, 53, 57, 61, 67, 73, 81, 86, 100};

// `units` over `cores` (logical core k at grid row k % rows) in proportion to `pct` of each core's row, a unit a core
// at least (units >= cores): the smallest even share t with the sum of max(1, t pct / 100) reaching `units`, then the
// surplus taken a unit at a time from the core with the most units for its speed
std::vector<uint32_t> row_weighted_counts(uint32_t units, uint32_t cores, uint32_t rows, const uint32_t* pct) {
    auto at = [&](uint32_t t, uint32_t k) { return std::max(1u, t * pct[k % rows] / 100); };
    auto total = [&](uint32_t t) {
        uint64_t sum = 0;
        for (uint32_t k = 0; k < cores; ++k) {
            sum += at(t, k);
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
        counts[k] = at(lo, k);
        sum += counts[k];
    }
    for (; sum > units; --sum) {
        uint32_t worst = cores;
        for (uint32_t k = 0; k < cores; ++k) {
            if (counts[k] > 1 && (worst == cores || static_cast<uint64_t>(counts[k]) * pct[worst % rows] >=
                                                         static_cast<uint64_t>(counts[worst]) * pct[k % rows])) {
                worst = k;
            }
        }
        --counts[worst];
    }
    return counts;
}

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

ProgramDescriptor DecodePrepareProgramFactory::create_descriptor(
    const DecodePrepareParams& attrs,
    const DecodePrepareTensorArgs& t,
    tensor_return_value_t& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const uint32_t heads = attrs.heads;
    const uint32_t users = prepare_users(attrs, t);

    std::array<Buffer*, 7> in = {
        t.proj.buffer(),
        t.fg.buffer(),
        t.history0.buffer(),
        t.history1.buffer(),
        t.history2.buffer(),
        t.taps.buffer(),
        t.gate_consts.buffer()};
    std::array<Buffer*, 5> out = {
        outputs[0].buffer(), outputs[1].buffer(), outputs[2].buffer(), outputs[3].buffer(), outputs[4].buffer()};

    auto grid = t.proj.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    const uint32_t blocks = (users + 15) / 16;  // users per unit: the 16 rows of a face pair
    const uint32_t units = 4 * heads * blocks;
    // split_channels: q / k / v's units as 2 items (channel tiles 0-1 and 2-3, on neighbouring cores: items i and i + 1),
    // the gates' whole; an item a core (the halves of a unit wait for each other: never two on one core)
    struct Item {
        uint32_t unit, c_lo, c_n;
    };
    std::vector<Item> items;
    for (uint32_t u = 0; u < units && attrs.split_channels; ++u) {
        if ((u / blocks) % 4 < 3) {
            items.push_back({u, 0, 2});
            items.push_back({u, 2, 2});
        } else {
            items.push_back({u, 0, N});
        }
    }
    const bool split = !items.empty() && items.size() <= grid.x * grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(grid, split ? static_cast<uint32_t>(items.size()) : units);

    // a unit's tokens' conv inputs, f and beta logits together (two units deep up to 4 tokens, one above)
    const uint32_t tokens = attrs.tokens;
    const uint32_t depth = tokens <= 4 ? 2 : 1;
    const uint32_t stack_g = stack_tokens(users, tokens);  // tokens a pass (0: one)
    const uint32_t stack_p = stack_g != 0 ? (tokens + stack_g - 1) / stack_g : 0;  // the passes
    if (stack_g != 0) {
        add_cb(desc, all_cores, CB_SEQ, N, bf16);   // a tile a channel tile: rows token-major, users * tokens <= 32
        add_cb(desc, all_cores, CB_HST, 2 * N, f32);  // a column of (3 + tokens) users <= 56 fp32 rows a channel tile
    }
    add_cb(desc, all_cores, CB_CUR, depth * tokens * N, bf16);
    add_cb(desc, all_cores, CB_H0, 2 * N, f32);
    add_cb(desc, all_cores, CB_H1, 2 * N, f32);
    add_cb(desc, all_cores, CB_H2, 2 * N, f32);
    add_cb(desc, all_cores, CB_TAP, 2 * 4 * N, f32);
    add_cb(desc, all_cores, CB_F, depth * tokens * N, bf16);
    add_cb(desc, all_cores, CB_DT, 2 * N, f32);
    add_cb(desc, all_cores, CB_RATE, 2 * N, f32);
    add_cb(desc, all_cores, CB_BETA, depth * tokens, f32);
    add_cb(desc, all_cores, CB_ONES, 1, f32);
    add_cb(desc, all_cores, CB_STG, 1, bf16);
    add_cb(desc, all_cores, CB_QKV, N, f32);
    add_cb(desc, all_cores, CB_SQ, 1, f32);
    add_cb(desc, all_cores, CB_INV, 1, f32);
    add_cb(desc, all_cores, CB_OUT, 2 * N, f32);
    add_cb(desc, all_cores, CB_BOUT, 2, f32);
    add_cb(desc, all_cores, CB_WSCR, 1, f32);
    add_cb(desc, all_cores, CB_CURH, 2 * N, f32);
    if (split) {
        add_cb(desc, all_cores, CB_SQP, 1, f32);
        add_cb(desc, all_cores, CB_SQX, 2, f32);
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = 0, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }
    if (attrs.tokens > 1 && users % 16 != 0) {
        add_cb(desc, all_cores, CB_ROWS, 1, bf16);  // 2 faces of 512 bytes
    }
    if (tokens > 1) {
        add_cb(desc, all_cores, CB_XT, std::max(tokens - 1, 2u) * N, f32);  // the earlier tokens' inputs of a unit
    }
    Buffer* keep = t.keep.has_value() ? t.keep->buffer() : nullptr;
    Buffer* saved = t.saved.has_value() ? t.saved->buffer() : nullptr;
    // out of place (history0_out .. saved_out): the new history and checkpoint rows written there (the reader's and the
    // writer's writes), the inputs only read; in place, the inputs themselves
    const bool oop = t.saved_out.has_value();
    Buffer* h0_out = oop ? t.history0_out->buffer() : in[2];
    Buffer* h1_out = oop ? t.history1_out->buffer() : in[3];
    Buffer* h2_out = oop ? t.history2_out->buffer() : in[4];
    Buffer* saved_out = oop ? t.saved_out->buffer() : saved;
    const uint32_t keep_bytes = keep != nullptr ? tt::round_up(keep->page_size(), 32u) : 0;  // whole 32-byte units
    if (keep != nullptr) {
        add_cb(desc, all_cores, CB_SAVED, (tokens > 1 ? tokens - 1 : 1) * N, f32);
        desc.cbs.push_back(CBDescriptor{
            .total_size = keep_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(CB_KEEP), .data_format = tt::DataFormat::UInt32,
                .page_size = keep_bytes}}},
        });
    }

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/kda/decode_prepare/device/kernels/";
    const uint32_t proj_tiles = t.proj.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t fg_tiles = t.fg.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t hist_tiles = t.history0.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    std::vector<uint32_t> reader_ct = {
        CB_CUR, CB_H0, CB_H1, CB_H2, CB_TAP, CB_F, CB_DT, CB_RATE, CB_BETA, CB_ONES, CB_STG,
        heads, proj_tiles, fg_tiles, hist_tiles, users};
    for (auto* b : in) {
        TensorAccessorArgs(b).append_to(reader_ct);
    }
    // the history's choice: (always some accessors' args, see the kernel)
    const uint32_t saved_pages = in[2]->num_pages();  // a history's (S_1 after S_0)
    for (uint32_t v : {keep != nullptr ? 1u : 0u, CB_KEEP, keep_bytes, CB_ROWS, CB_SAVED, attrs.tokens, saved_pages}) {
        reader_ct.push_back(v);
    }
    TensorAccessorArgs(keep != nullptr ? keep : in[2]).append_to(reader_ct);
    TensorAccessorArgs(saved != nullptr ? saved : in[2]).append_to(reader_ct);
    for (uint32_t v : {stack_g, stack_p, CB_SEQ, CB_HST}) {
        reader_ct.push_back(v);
    }
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_decode_prepare.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_OUT, CB_BOUT, CB_WSCR, heads, users, CB_CURH, hist_tiles};
    for (auto* b : out) {
        TensorAccessorArgs(b).append_to(writer_ct);
    }
    TensorAccessorArgs(in[4]).append_to(writer_ct);  // history 2
    writer_ct.push_back(attrs.tokens);
    writer_ct.push_back(keep != nullptr ? 1u : 0u);  // more than 3 tokens: the earlier inputs into the checkpoint
    writer_ct.push_back(saved_pages);
    TensorAccessorArgs(saved != nullptr ? saved : in[2]).append_to(writer_ct);
    writer_ct.push_back(stack_g);
    writer_ct.push_back(stack_p);
    writer_ct.push_back(attrs.pair_major ? 1u : 0u);
    writer_ct.push_back(CB_SQP);
    writer_ct.push_back(CB_SQX);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_decode_prepare.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb : {CB_H0, CB_H1, CB_H2, CB_TAP, CB_DT, CB_RATE, CB_BETA, CB_QKV, CB_INV, CB_SQX}) {
        unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    if (attrs.tokens > 1) {
        unpack_to_dest_mode[CB_XT] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/decode_prepare.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    // (stacked: one token a pass for the compute, over the pass's stack_g x users rows)
    compute.compile_time_args = {
        std::bit_cast<uint32_t>(attrs.lower), std::bit_cast<uint32_t>(attrs.q_scale), stack_g != 0 ? stack_g * users : users};
    if (stack_g != 0) {
        compute.defines = {{"KDA_PREP_STACK", std::to_string(stack_p)}};
    } else if (attrs.tokens > 1) {
        compute.defines = {{"KDA_PREP_TOKENS", std::to_string(attrs.tokens)}};
    }
    if (split) {
        compute.defines.emplace_back("KDA_PREP_SPLIT", "1");
        reader.defines.emplace_back("KDA_PREP_SPLIT", "1");
        writer.defines.emplace_back("KDA_PREP_SPLIT", "1");
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    // the units a core: split_work_to_cores' even share, or (row_split, the plain call) by grid row speed
    std::vector<uint32_t> counts(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        counts[i] = split ? 1 : (core_group_1.contains(core) ? units_per_core_1 : units_per_core_2);
    }
    const bool rows_form = attrs.row_split && !split && num_cores_y == std::size(ROW_PCT) && units > num_cores;
    // row_split: the top 6 rows' cores (virtual y 2 - 7) read on NoC 1, their reader on BRISC with NoC 1 and their writer
    // on NCRISC with NoC 0 (each RISC still its own NoC, as recurrent_decode_step's form bit 256): the reads over both
    // NoCs, the lower rows no longer starved on NoC 0 alone.  With the table, 64 / 32 users a row 99.7 / 63.5 -> 67.0 /
    // 43.7 us (the top 5 rows 69.9 / 43.1, 4 72.9 / 44.2, 7 73.7 / 43.4, 8 76.8 / 45.3, none 87.1 / 51.8).
    // The split form (the verify step's) the same with its top 5 rows (the exchange's semaphore is set by a write, the
    // writer on either NoC): the op alone from a trace, 8 / 32 / 128 users T = 8 and 128 users T = 4 34.9 / 54.8 /
    // 108.5 / 66.5 -> 29.4 / 49.0 / 100.0 / 62.3 us (4 rows 31.2 / 50.0 / 99.8 / 62.2, 6 29.4 / 49.0 / 100.8 / 62.2,
    // 8 31.8 / 51.3 / 107.0 / 68.5)
    auto* device = t.proj.device();
    std::set<CoreCoord> upper;
    const uint32_t upper_rows = split ? 5 : rows_form ? UPPER_ROWS : 0;
    for (uint32_t i = 0; i < num_cores && upper_rows > 0; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto xy = device->worker_core_from_logical_core(core);
        if (xy.y >= 2 && xy.y < 2 + upper_rows) {
            upper.insert(core);
        }
    }
    if (rows_form) {
        counts = row_weighted_counts(units, num_cores, num_cores_y, units < 2 * num_cores ? ROW_PCT_FEW : ROW_PCT);
    }
    KernelDescriptor reader_up = reader, writer_up = writer;
    reader_up.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1};
    writer_up.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0};

    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t count = counts[i];
        if (split) {
            start = items[i].unit;
        }
        // (split: the item's channel tiles, and the other half's core: the item next to it)
        const uint32_t c_lo = split ? items[i].c_lo : 0, c_n = split ? items[i].c_n : N;
        CoreCoord partner = {0, 0};
        if (split && c_n == 2) {
            const uint32_t j = c_lo == 0 ? i + 1 : i - 1;
            partner = device->worker_core_from_logical_core(CoreCoord{j / num_cores_y, j % num_cores_y});
        }
        {
            KernelDescriptor::RTArgList rt;
            for (auto* b : in) {
                rt.push_back(b);
            }
            rt.push_back(start);
            rt.push_back(count);
            if (keep != nullptr) {
                rt.push_back(keep);
                rt.push_back(saved);
            } else {
                rt.push_back(0u);
                rt.push_back(0u);
            }
            rt.push_back(c_lo);
            rt.push_back(c_n);
            // (the reader's writes: history 0 and 1, the checkpoint's first rows)
            rt.push_back(h0_out);
            rt.push_back(h1_out);
            if (saved_out != nullptr) {
                rt.push_back(saved_out);
            } else {
                rt.push_back(0u);
            }
            (upper.count(core) != 0 ? reader_up : reader).emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            for (auto* b : out) {
                rt.push_back(b);
            }
            rt.push_back(h2_out);  // history 2 (written only)
            rt.push_back(start);
            rt.push_back(count);
            rt.push_back(h1_out);          // history 1 (more tokens)
            rt.push_back(users * heads);  // token k's pairs k B H on
            rt.push_back(h0_out);          // history 0 (three tokens)
            if (saved_out != nullptr) {
                rt.push_back(saved_out);  // the checkpoint (more than 3 tokens)
            } else {
                rt.push_back(0u);
            }
            rt.push_back(c_lo);
            rt.push_back(c_n);
            rt.push_back(static_cast<uint32_t>(partner.x));
            rt.push_back(static_cast<uint32_t>(partner.y));
            (upper.count(core) != 0 ? writer_up : writer).emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(start);
            rt.push_back(count);
            rt.push_back(c_lo);
            rt.push_back(c_n);
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

}  // namespace ttnn::experimental::prim
