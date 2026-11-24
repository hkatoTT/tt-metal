// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_read_program_factory.hpp"
#include "mhc_stream_read_device_operation_types.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <cstdlib>
#include <string>
#include <variant>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

// A list of single-core ranges as few rectangles (merge_ranges): a program goes out range by range, so a range a core
// lengthens its dispatch.
CoreRangeSet runs_set(const std::vector<CoreRange>& ranges) { return CoreRangeSet(ranges).merge_ranges(); }
// The mix core (0, 0) and the slice cores: slice s at (1 + s % 8, s / 8).  The CBs that one side writes into over
// the NoC are on every core, so their address is the same everywhere.  The mix core sends P as soon as it has it
// (before the Sinkhorn of comb, which P does not need), then the block input's norm scale, which it makes from P and
// the streams' Gram tiles (mhc_stream_mix.cpp's make_ns): the slices exchange nothing.
constexpr uint32_t SLICE_X0 = 1, SLICE_W = 8;
// every core
constexpr uint32_t CB_PART = tt::CBIndex::c_0;  // mix core: the slices' partials (rt, n, s), as mhc_stream_mix reads
constexpr uint32_t CB_PIN = tt::CBIndex::c_9;   // slice cores: P, one tile per row tile
constexpr uint32_t CB_NS = tt::CBIndex::c_10;   // slice cores: the norm's scale per token (from the mix core)
// the mix core (mhc_stream_mix.cpp's indices; c_16 .. c_27 its intermediates)
constexpr uint32_t CB_MC = tt::CBIndex::c_1;
constexpr uint32_t CB_SC = tt::CBIndex::c_2;
constexpr uint32_t CB_POST = tt::CBIndex::c_3;
constexpr uint32_t CB_COMB = tt::CBIndex::c_4;
constexpr uint32_t CB_P = tt::CBIndex::c_5;
constexpr uint32_t CB_NSO = tt::CBIndex::c_7;   // the norm's scale it makes
constexpr uint32_t CB_P2 = tt::CBIndex::c_11;   // P for it
constexpr uint32_t CB_PG = tt::CBIndex::c_15;   // P_rt @ G_rt
constexpr uint32_t CB_Q = tt::CBIndex::c_30;    // (P_rt @ G_rt) * P_rt
constexpr uint32_t MAX_ROW_TILES = 4;  // mhc_stream_mix.cpp's CB sizes
// the slice cores
constexpr uint32_t CB_Y = tt::CBIndex::c_6;     // the normed block input (bf16)
constexpr uint32_t CB_S = tt::CBIndex::c_12;    // the streams' tiles of this slice, every row tile (bf16)
constexpr uint32_t CB_FN = tt::CBIndex::c_13;   // fn's rows of this slice (bf16)
constexpr uint32_t CB_G = tt::CBIndex::c_14;    // gamma's tiles of this slice (bf16)
constexpr uint32_t CB_OUT = tt::CBIndex::c_28;  // the partials, 4 per row tile (fp32)
constexpr uint32_t CB_X = tt::CBIndex::c_29;    // the block input's tiles of this slice (bf16)
constexpr uint32_t SEM_PART = 0, SEM_NS = 1, SEM_P = 2;
// the groups of 8 with fn read once a column (`mcast`): the slice reader's own part landed (for its writer), and
// part q's multicast landed (set by group q's slice writer)
constexpr uint32_t SEM_OWN = 3, SEM_FN0 = 4;

void add_cb(ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t index, uint32_t n_tiles, tt::DataFormat df) {
    const uint32_t tile_size = tt::tile_size(df);
    desc.cbs.push_back(CBDescriptor{
        .total_size = n_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index), .data_format = df, .page_size = tile_size}}},
    });
}

KernelDescriptor kernel(
    const std::string& source,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    std::variant<ReaderConfigDescriptor, WriterConfigDescriptor, ComputeConfigDescriptor> config) {
    KernelDescriptor k;
    k.kernel_source = source;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    std::visit([&](auto&& c) { k.config = c; }, config);
    return k;
}
// mhc_stream_write_read (the write before this read in the same program): mhc_stream_write's kernels on the cores
// the read leaves free, each new streams tile written to DRAM and into the slice core that projects it (CB_S, at the same
// address on every core of the program: CB_S is the first CB of every core here), followed by an increment of that
// core's SEM_S; the slice readers wait for their tiles' count in place of reading them from DRAM.
struct FusedWrite {
    Buffer* streams;    // the write's streams (the residual)
    Buffer* block_out;  // the sub-layer's output
    Buffer* post;
    Buffer* comb;
    Buffer* out;  // the new streams
    uint32_t tokens;
};
constexpr uint32_t SEM_S = 15;
constexpr uint32_t CBW_S = tt::CBIndex::c_20, CBW_BO = tt::CBIndex::c_21, CBW_W = tt::CBIndex::c_22;
constexpr uint32_t CBW_Q = tt::CBIndex::c_23, CBW_SCR = tt::CBIndex::c_24, CBW_T1 = tt::CBIndex::c_25;
constexpr uint32_t CBW_T2 = tt::CBIndex::c_26, CBW_OUT = tt::CBIndex::c_27;
// CBW_S's and CBW_BO's pages: the write reader's units in flight (<= 13).  Decode: 1.  In the model its block_out is
// the all-reduce's output kept sharded in L1 (`ar_keep_l1`): more reads in flight queue at the shard cores
// (5 layers traced, the op at 8 / 32 / 64 tokens a row: 1 page 26.4 / 35.7 / 45.0 us, 2: 26.4 / 37.1-38.8 /
// 46.8-47.2, 6: 27.8 / 41.1 / 45.4 (no comb / post multicast); the write before 28.9 / 37.0-38.2 / 45.8-46.0).  The
// prefill's groups (block_out in DRAM): 2 (the op alone at 256 tokens a row: 1 page 135.3 us, 2: 133.2, 6: 136.9)
constexpr uint32_t WRITE_PAGES_DECODE = 1, WRITE_PAGES_PREFILL = 2;
// MHC_WR_MERGED's CBs on the slice cores: the old streams (4 a column), block_out (1 a column), MERGED_AHEAD columns in
// flight; W and Q of the group's row tiles; the two bf16 products
constexpr uint32_t CBM_SOLD = tt::CBIndex::c_20, CBM_BO = tt::CBIndex::c_21, CBM_W = tt::CBIndex::c_22;
constexpr uint32_t CBM_Q = tt::CBIndex::c_23, CBM_T1 = tt::CBIndex::c_24, CBM_T2 = tt::CBIndex::c_25;
constexpr uint32_t MERGED_AHEAD = 8;  // (<= 8: the reader's transaction ids 1 .. 8)
// ... and the helper cores' (the cores the groups leave): their units' outputs, a group's comb and post rows; the slice
// cores' CB_READY (the helpers' columns landed); the slice cores' flags of up to MERGED_SENDERS senders
constexpr uint32_t CBM_HOUT = tt::CBIndex::c_27, CBM_SCR = tt::CBIndex::c_26, CBM_READY = tt::CBIndex::c_26;
constexpr uint32_t SEM_H0 = 12, MERGED_SENDERS = 4;
// The decode forms (single, groups of 8): the write side's comb and post read once, by the mix core (of group 0), and
// multicast into CB_CP of the write cores (one CB over the mix cores and the write cores: one address), SEM_CP set
// after (every write core reading the same page queued at one DRAM bank, ~3 us at 8 tokens)
constexpr uint32_t CB_CP = tt::CBIndex::c_8, SEM_CP = 14;
// GLM53_MHC_FINISH=1 or 2 (the decode forms with the fused write; a switch, off here unless set, read at the program's
// creation, not in the program cache's key; the model sets 2 by default, ttnn/common.py): 5's comb (Sinkhorn) and post
// off the mix core's chain (without it made after P and the norm scale, the op's last ~1 us), on a finisher core a
// group taken from the write cores: the mix writer writes the mixes (CB_MIXES, one fp32 tile, a CB over the mix and
// finisher cores: one address) into it and increments its SEM_FIN; the finisher runs the mix kernels' MHC_FINISHER
// forms (the reader the Sinkhorn constants, the compute make_comb and post, the writer comb and post to the outputs).
// The same calls on the same values.  Taken only where it leaves the most units a write core makes unchanged (32
// tokens a row: 512 units on 84 -> 80 cores, 7 at most either way; 8 tokens: 128 on 111 -> 110, 2; not 64: 1,024 on 48
// -> 40, 22 -> 26).  =1: the mix writer writes the mixes and signals the finisher before P; =2 (MHC_FIN_SPLIT): writes
// them before P, signals after P's send (whose barrier covers the mixes' write), P not held back by that write's
// barrier.
constexpr uint32_t SEM_FIN = 13;
constexpr uint32_t CB_MIXES = tt::CBIndex::c_23;     // mhc_stream_mix.cpp's CB_MIX
constexpr uint32_t CB_FIN_SENT = tt::CBIndex::c_31;  // the mix core: its writer sent the mixes (the compute may pop)

// group g's finisher: want[g] if a write core, else the write cores' last; out of `write_cores` (they keep CB_S,
// CB_CP and the semaphores there: the comb / post multicast's rectangles unchanged).  None where the switch is off or
// where it would raise the most units a write core makes
std::vector<CoreCoord> pick_finishers(
    std::vector<CoreCoord>& write_side, const std::vector<CoreCoord>& want, uint32_t units) {
    const char* env = std::getenv("GLM53_MHC_FINISH");
    const uint32_t n = write_side.size(), k = want.size();
    if (env == nullptr || (env[0] != '1' && env[0] != '2') || n < k + 8 ||
        (units + n - 1) / n != (units + n - k - 1) / (n - k)) {
        return {};
    }
    std::vector<CoreCoord> out;
    for (const auto& w : want) {
        auto it = std::find(write_side.begin(), write_side.end(), w);
        if (it == write_side.end()) {
            it = write_side.end() - 1;
        }
        out.push_back(*it);
        write_side.erase(it);
    }
    return out;
}

// GLM53_MHC_FINISH: the finishers' own CBs (added after the program's others; CB_MIXES, over the mix cores and them,
// is placed with the mix core's CBs), the mix cores' CB_FIN_SENT, the finishers' SEM_FIN
void add_finisher_cbs(
    ProgramDescriptor& desc,
    const CoreRangeSet& mix_cores,
    const CoreRangeSet& fin_set,
    const std::vector<uint32_t>& sems,
    bool sfpu_sinkhorn) {
    const auto f32 = tt::DataFormat::Float32;
    add_cb(desc, fin_set, CB_SC, 8, f32);
    add_cb(desc, fin_set, CB_POST, 1, f32);
    add_cb(desc, fin_set, CB_COMB, 1, f32);
    if (!sfpu_sinkhorn) {  // (the FPU Sinkhorn's ping and pong)
        add_cb(desc, fin_set, tt::CBIndex::c_24, 2, f32);
        add_cb(desc, fin_set, tt::CBIndex::c_25, 2, f32);
    }
    add_cb(desc, mix_cores, CB_FIN_SENT, 1, f32);
    TT_FATAL(
        std::find(sems.begin(), sems.end(), SEM_FIN) == sems.end(), "mhc_stream_write_read: SEM_FIN {} taken", SEM_FIN);
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = SEM_FIN, .core_type = tt::CoreType::WORKER, .core_ranges = fin_set, .initial_value = 0});
}

// The grid's cores outside `taken` (column-major, as mhc_stream_write deals its units out)
std::vector<CoreCoord> free_cores(const CoreCoord& grid, const std::vector<CoreCoord>& taken) {
    std::vector<CoreCoord> out;
    for (uint32_t x = 0; x < grid.x; ++x) {
        for (uint32_t y = 0; y < grid.y; ++y) {
            bool used = false;
            for (const auto& c : taken) {
                used = used || (c.x == x && c.y == y);
            }
            if (!used) {
                out.push_back(CoreCoord{x, y});
            }
        }
    }
    return out;
}

// The cores as few rectangles as their layout allows: each grid row's runs of consecutive x, then the runs that repeat
// on consecutive rows merged (one range a core made the program's dispatch, its kernel binaries, CBs and semaphores
// sent range by range, ~25 us slower)
CoreRangeSet as_set(const std::vector<CoreCoord>& cores) {
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> rows;  // (x0, x1) -> the rows with that run
    std::map<uint32_t, std::vector<uint32_t>> by_row;
    for (const auto& c : cores) {
        by_row[c.y].push_back(c.x);
    }
    for (auto& [y, xs] : by_row) {
        std::sort(xs.begin(), xs.end());
        for (size_t i = 0; i < xs.size();) {
            size_t j = i;
            while (j + 1 < xs.size() && xs[j + 1] == xs[j] + 1) {
                ++j;
            }
            rows[{xs[i], xs[j]}].push_back(y);
            i = j + 1;
        }
    }
    std::vector<CoreRange> ranges;
    for (auto& [run, ys] : rows) {
        std::sort(ys.begin(), ys.end());
        for (size_t i = 0; i < ys.size();) {
            size_t j = i;
            while (j + 1 < ys.size() && ys[j + 1] == ys[j] + 1) {
                ++j;
            }
            ranges.emplace_back(CoreCoord{run.first, ys[i]}, CoreCoord{run.second, ys[j]});
            i = j + 1;
        }
    }
    return CoreRangeSet(ranges);
}

// the write cores' rectangles for the multicast of comb and post (mhc_cp.hpp's runtime args)
template <typename Device>
void append_cp_rects(KernelDescriptor::RTArgList& args, Device* device, const std::vector<CoreCoord>& write_cores) {
    const auto ranges = as_set(write_cores).ranges();
    args.push_back(static_cast<uint32_t>(ranges.size()));
    for (const auto& r : ranges) {
        const CoreCoord lo = device->worker_core_from_logical_core(r.start_coord);
        const CoreCoord hi = device->worker_core_from_logical_core(r.end_coord);
        for (const auto v : {lo.x, lo.y, hi.x, hi.y}) {
            args.push_back(static_cast<uint32_t>(v));
        }
        args.push_back(static_cast<uint32_t>(r.size()));
    }
}

// The write side's CBs and kernels on `cores` (CB_S already on them): the units (output tiles, row tile major) dealt
// out in order, each core a run; the slice cores' logical coordinates (group major: `slice_at[g slices + s]`).
template <typename Device>
void add_write_side(
    ProgramDescriptor& desc,
    const FusedWrite& fw,
    const std::vector<CoreCoord>& cores,
    Device* device,
    uint32_t d_tiles,
    uint32_t row_tiles,
    uint32_t k_tiles,
    uint32_t slices,
    uint32_t group_rows,
    const std::vector<CoreCoord>& slice_at,
    bool cp,
    uint32_t unit_off = 0) {
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const CoreRangeSet set = as_set(cores);
    const uint32_t pages = group_rows > 1 ? WRITE_PAGES_PREFILL : WRITE_PAGES_DECODE;
    add_cb(desc, set, CBW_S, pages, bf16);  // the write reader's units in flight
    add_cb(desc, set, CBW_BO, pages, bf16);
    add_cb(desc, set, CBW_W, 2, bf16);
    add_cb(desc, set, CBW_Q, 2, bf16);
    add_cb(desc, set, CBW_SCR, 1, f32);
    add_cb(desc, set, CBW_T1, 1, bf16);
    add_cb(desc, set, CBW_T2, 1, bf16);
    add_cb(desc, set, CBW_OUT, 2, bf16);
    const std::string wdir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_write/device/kernels/";
    std::vector<uint32_t> ct = {CBW_S, CBW_BO, CBW_W, CBW_Q, CBW_SCR, d_tiles, fw.tokens};
    for (auto* b : {fw.streams, fw.block_out, fw.post, fw.comb}) {
        TensorAccessorArgs(b).append_to(ct);
    }
    auto reader = kernel(wdir + "dataflow/reader_mhc_stream_write.cpp", set, ct, ReaderConfigDescriptor{});
    if (cp) {  // comb and post from the mix core's multicast (CB_CP, SEM_CP)
        reader.defines = {{"MHC_CP", std::to_string(CB_CP)}, {"MHC_CP_SEM", std::to_string(SEM_CP)}};
    }
    ct = {CBW_OUT, 1, CB_S, SEM_S, k_tiles, slices, group_rows};
    TensorAccessorArgs(fw.out).append_to(ct);
    auto writer = kernel(wdir + "dataflow/writer_mhc_stream_write.cpp", set, ct, WriterConfigDescriptor{});
    auto compute = kernel(
        wdir + "compute/mhc_stream_write.cpp",
        set,
        {d_tiles, CBW_S, CBW_BO, CBW_W, CBW_Q, CBW_T1, CBW_T2, CBW_OUT},
        ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
    std::vector<uint32_t> coords;
    for (const auto& c : slice_at) {
        const CoreCoord noc = device->worker_core_from_logical_core(c);
        coords.push_back(static_cast<uint32_t>(noc.x));
        coords.push_back(static_cast<uint32_t>(noc.y));
    }
    {  // the slice cores' NoC x, y: the same for every write core (common runtime args)
        KernelDescriptor::RTArgList common;
        for (uint32_t v : coords) {
            common.push_back(v);
        }
        writer.emplace_common_runtime_args(common);
    }
    const uint32_t units = row_tiles * d_tiles, n = cores.size();
    for (uint32_t i = 0, start = 0; i < n; ++i) {
        const uint32_t count = units / n + (i < units % n ? 1 : 0);
        KernelDescriptor::RTArgList r;
        for (auto* b : {fw.streams, fw.block_out, fw.post, fw.comb}) {
            r.push_back(b);
        }
        r.push_back(start + unit_off);  // (the tensors' units: a long wave's later call's start past the earlier's)
        r.push_back(count);
        reader.emplace_runtime_args(cores[i], r);
        KernelDescriptor::RTArgList w;
        w.push_back(fw.out);
        w.push_back(start);
        w.push_back(count);
        w.push_back(unit_off);  // (the new streams' pages: unit + unit_off; the hand-off by the call's unit)
        writer.emplace_runtime_args(cores[i], w);
        KernelDescriptor::RTArgList c;
        c.push_back(start);
        c.push_back(count);
        compute.emplace_runtime_args(cores[i], c);
        start += count;
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
}

// The groups form (more than 64 rows: prefill): a group per 32 tokens (4 row tiles), its mix core at (x0, y) and its
// S slice cores at (x0 + 1 + s, y), two groups a grid row (x0 = 0, 6), each group doing what the single form does for
// its own row tiles (the same compute kernels with 4 row tiles and 32 tokens), the groups independent.  With more than
// 5 slices (decode's 32 or 64 tokens a row): a group per 8 tokens (one row tile: the single form's 8-token work, its
// mix core's share a quarter of a 32-token one's), one a grid row (its mix core at (0, g), the slices beside it), each
// writing its 8 rows of the outputs' tile row g / 4 (faces' rows: the 4 groups of a tile row write apart).
ProgramDescriptor create_groups_descriptor(
    const MhcStreamReadParams& attrs,
    const MhcStreamReadTensorArgs& t,
    Buffer* post,
    Buffer* comb,
    Buffer* out,
    const FusedWrite* fw = nullptr) {
    ProgramDescriptor desc;
    Buffer* streams = t.streams.buffer();
    Buffer* fn = t.fn.buffer();
    Buffer* mix_consts = t.mix_consts.buffer();
    Buffer* consts = t.consts.buffer();
    Buffer* gamma = t.gamma.buffer();
    const uint32_t d_tiles = t.streams.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const bool g8 = attrs.slices > 5;
    const uint32_t row_tiles = g8 ? 1 : 4, group_tokens = g8 ? 8 : 32, group_x = 6;
    const uint32_t sub_rows = g8 ? 8 : 0;  // the writers: 0 whole tiles (tile row g), 8 the group's rows of g / 4
    const uint32_t groups = attrs.tokens / group_tokens;
    // the call's first group in the tensors (a long wave's later call: `offset`), for the addresses only (the cores and
    // the hand-off by the call's own groups)
    const uint32_t off_g = attrs.offset / group_tokens;
    const uint32_t slices = attrs.slices;
    const uint32_t k_tiles = d_tiles / slices;
    std::vector<CoreRange> mix_ranges, slice_ranges;
    auto mix_at = [&](uint32_t g) { return g8 ? CoreCoord{0, g} : CoreCoord{(g % 2) * group_x, g / 2}; };
    for (uint32_t g = 0; g < groups; ++g) {
        const CoreCoord m = mix_at(g);
        mix_ranges.emplace_back(m, m);
        slice_ranges.emplace_back(CoreCoord{m.x + 1, m.y}, CoreCoord{m.x + slices, m.y});
    }
    const CoreRangeSet mix_cores(mix_ranges), slice_cores(slice_ranges);
    const CoreRangeSet all = mix_cores.merge(slice_cores);
    const auto f32 = tt::DataFormat::Float32;
    const auto bf16 = tt::DataFormat::Float16_b;
    std::vector<CoreCoord> write_cores, slice_at;
    const bool cp = fw != nullptr && g8;  // (the decode forms)
    const uint32_t cp_pages = cp ? (attrs.tokens + 31) / 32 : 0;
    // merged: the prefill's groups with the write on the slice cores (each slice core makes its own tiles of the new
    // streams: reader_mhc_wr_slice_groups.cpp, MHC_WR_MERGED in mhc_read_slice.cpp,
    // writer_mhc_wr_slice_groups.cpp), no write cores, hand-off or semaphore increments; block_out read once a column
    // (the write cores read it once a row tile); the same values.  The slice cores' write takes ~0.44 us a new streams
    // tile computing (two HiFi4 matmuls, three packs and an add; 128 tiles a slice core at 512 tokens: 57 us), the rest
    // waiting on its DRAM reads, the top grid rows the longest (below, the top rows' reads on NoC 1); it gains
    // only where the write cores are few: 512 tokens a call 194.9 -> 178.5 us, 128 / 256 104.6 / 133.3 -> 120.8 /
    // 126.3.  Fewer groups leave more cores: those help (below), and the call takes 128 / 256 tokens in
    // 104.6 / 133.3 -> 97.8 / 119.9 us.  Every prefill groups call (the write cores: the decode's groups of 8)
    const bool merged = fw != nullptr && !g8;
    if (fw != nullptr) {  // CB_S first on every core of the program (the write / helper cores' hand-off address)
        std::vector<CoreCoord> taken;
        for (uint32_t g = 0; g < groups; ++g) {
            const CoreCoord m = mix_at(g);
            taken.push_back(m);
            for (uint32_t s = 0; s < slices; ++s) {
                taken.push_back(CoreCoord{m.x + 1 + s, m.y});
                slice_at.push_back(CoreCoord{m.x + 1 + s, m.y});
            }
        }
        write_cores = free_cores(t.streams.device()->compute_with_storage_grid_size(), taken);
        TT_FATAL(write_cores.size() >= 8, "mhc_stream_write_read: {} cores left for the write", write_cores.size());
        add_cb(desc, all.merge(as_set(write_cores)), CB_S, row_tiles * k_tiles, bf16);
        if (cp) {
            add_cb(desc, mix_cores.merge(as_set(write_cores)), CB_CP, 2 * cp_pages, f32);
        }
    }
    // GLM53_MHC_PART_MIX=1 (the prefill's groups only; a switch, off unless set: ttnn/w1_landing.py sets it with its
    // landing at 8 / 32 users): CB_PART (the slices' partials, which only the mix core reads) on the mix cores only,
    // right after CB_PIN and CB_NS (both on every core of the program, at one address), not on every core: the slice
    // cores' CBs end 256 KB lower at 4 slices x 4 row tiles (the prefill's mHC write-read 1,248,640 -> 986,496 B, its
    // read 1,140,096 -> 877,952), room for a larger weight landing (GLM53_W1, ttnn/w1_landing.py).  The slice writers
    // take the mix core's CB_PART address as their own CB_NS's end (MHC_PART_AFTER_NS).  The same values and traffic;
    // read at the program's creation (not in the program cache's key)
    const bool part_mix = [&] {
        const char* env = std::getenv("GLM53_MHC_PART_MIX");
        return !g8 && env != nullptr && env[0] == '1';
    }();
    // GLM53_MHC_FINISH (above): group g's finisher right of its slices where a write core
    std::vector<CoreCoord> write_side = write_cores, fin_at;
    if (cp) {
        std::vector<CoreCoord> want;
        for (uint32_t g = 0; g < groups; ++g) {
            want.push_back(CoreCoord{mix_at(g).x + slices + 1, mix_at(g).y});
        }
        fin_at = pick_finishers(write_side, want, groups * row_tiles * d_tiles);
    }
    const bool fin = !fin_at.empty();
    const CoreRangeSet fin_set = fin ? as_set(fin_at) : mix_cores;
    if (!part_mix) {
        add_cb(desc, all, CB_PART, slices * row_tiles * 4, f32);
    }
    add_cb(desc, all, CB_PIN, row_tiles, f32);
    add_cb(desc, all, CB_NS, 1, f32);
    if (part_mix) {
        add_cb(desc, mix_cores, CB_PART, slices * row_tiles * 4, f32);
    }
    add_cb(desc, mix_cores, CB_MC, 13, f32);
    add_cb(desc, mix_cores, CB_SC, 8, f32);
    add_cb(desc, mix_cores, CB_POST, 1, f32);
    add_cb(desc, mix_cores, CB_COMB, 1, f32);
    add_cb(desc, mix_cores, CB_P, MAX_ROW_TILES, f32);
    add_cb(desc, mix_cores, CB_NSO, 1, f32);
    add_cb(desc, mix_cores, CB_P2, MAX_ROW_TILES, f32);
    add_cb(desc, mix_cores, CB_PG, MAX_ROW_TILES, f32);
    add_cb(desc, mix_cores, CB_Q, MAX_ROW_TILES, f32);
    add_cb(desc, mix_cores, tt::CBIndex::c_16, 4 * MAX_ROW_TILES, f32);
    add_cb(desc, mix_cores, tt::CBIndex::c_17, 3 * MAX_ROW_TILES, f32);
    add_cb(desc, mix_cores, tt::CBIndex::c_18, MAX_ROW_TILES, f32);
    add_cb(desc, mix_cores, tt::CBIndex::c_19, 3, f32);
    for (uint32_t cb = tt::CBIndex::c_20; cb <= tt::CBIndex::c_23; ++cb) {  // (the mixes on the finishers too)
        add_cb(desc, fin && cb == CB_MIXES ? mix_cores.merge(fin_set) : mix_cores, cb, 1, f32);
    }
    add_cb(desc, mix_cores, tt::CBIndex::c_24, 2, f32);
    add_cb(desc, mix_cores, tt::CBIndex::c_25, 2, f32);
    add_cb(desc, mix_cores, tt::CBIndex::c_26, 1, f32);
    add_cb(desc, mix_cores, tt::CBIndex::c_27, 1, f32);
    if (fw == nullptr) {
        add_cb(desc, slice_cores, CB_S, row_tiles * k_tiles, bf16);
    }
    // merged: the slice cores make the new streams of their columns k < k_own, the helper cores (the write cores of the
    // other form) those of the rest, in units of a slice's column (4 tiles) dealt out in order, written into the slice
    // core's CB_S; each slice waits for its senders' flags (SEM_H0 + slot, set by a write).  k_own: all of them from 16
    // groups on (512 tokens: the helpers' hand-off and the DRAM traffic cost more than they take off, k_own 8 / 12 / 16
    // / 32: 198.8 / 194.9 / 183.1 / 178.5 us), a quarter below.  defer (with helpers): the slice reader
    // writes its whole slice to DRAM once CB_S is whole, in place of the slice writer and the helpers writing each tile
    // as made (without helpers, at 512 tokens, it costs more, 220 - 231 us: the writes then meet the partials'
    // and P's traffic).
    const uint32_t k_own = !merged ? k_tiles : (groups >= 16 ? k_tiles : k_tiles / 4);
    const bool defer = merged && k_own < k_tiles;
    const uint32_t helped = k_tiles - k_own;
    const uint32_t units = merged ? groups * slices * helped : 0;
    std::vector<std::pair<uint32_t, uint32_t>> helper_run;  // (first unit, units) a helper core
    std::vector<uint32_t> senders(groups * slices, 0);      // a slice's senders
    std::vector<std::vector<uint32_t>> helper_slots;        // a helper's slots, a slot a slice it sends to
    if (merged && units > 0) {
        // (at most MERGED_SENDERS senders a slice: a helper's run at least helped / (MERGED_SENDERS - 1) units)
        const uint32_t n = std::min<uint32_t>(
            {static_cast<uint32_t>(write_cores.size()), units, groups * slices * (MERGED_SENDERS - 1)});
        for (uint32_t i = 0, start = 0; i < n; ++i) {
            const uint32_t count = units / n + (i < units % n ? 1 : 0);
            helper_run.emplace_back(start, count);
            std::vector<uint32_t> slots;
            for (uint32_t j = start / helped; j <= (start + count - 1) / helped; ++j) {
                slots.push_back(senders[j]++);
            }
            helper_slots.push_back(std::move(slots));
            start += count;
        }
        for (uint32_t c : senders) {
            TT_FATAL(c <= MERGED_SENDERS, "mhc_stream_write_read: {} senders to a slice core", c);
        }
    }
    if (merged) {  // the write's operands on the slice cores (the mix cores' c_20 .. c_25 are their own CBs)
        add_cb(desc, slice_cores, CBM_READY, 1, bf16);
        add_cb(desc, slice_cores, CBM_SOLD, MERGED_AHEAD * row_tiles, bf16);
        add_cb(desc, slice_cores, CBM_BO, MERGED_AHEAD, bf16);
        add_cb(desc, slice_cores, CBM_W, row_tiles, bf16);
        add_cb(desc, slice_cores, CBM_Q, row_tiles, bf16);
        add_cb(desc, slice_cores, CBM_T1, 2, bf16);
        add_cb(desc, slice_cores, CBM_T2, 2, bf16);
    }
    add_cb(desc, slice_cores, CB_FN, k_tiles * 3, bf16);
    add_cb(desc, slice_cores, CB_G, k_tiles, bf16);
    add_cb(desc, slice_cores, CB_OUT, 4 * row_tiles, f32);
    add_cb(desc, slice_cores, CB_X, k_tiles, bf16);
    add_cb(desc, slice_cores, CB_Y, k_tiles, bf16);
    // fn's and gamma's slice read once a column of slice cores: group g's slice s reads K tiles q kp .. (q + 1) kp - 1
    // of its slice (q its part: g in the groups of 8, one a grid row) and multicasts them to the other groups' slice s
    // (the same grid column), on the writer's NoC (the reads on the reader's), in place of every group reading the
    // slice whole (the same bits; the op at 32 / 64 tokens a row 37.7 / 53.5 -> 32.2 / 37.5 us).  The prefill's
    // groups: two a grid row, so the slice cores s of the even groups are one grid column and the odd groups' another,
    // each column its groups' parts (q = g / 2) (16 groups at 512 tokens: fn and gamma 16 of the 32 MB a chip's slice
    // cores read, 2 MB read once a column; 59a4357173).  Where the parts do not divide the slice, every group reads it
    const uint32_t col_groups = g8 ? groups : groups / 2;
    const bool mcast_g8 = g8 && groups >= 2 && k_tiles % groups == 0;
    const bool mcast_groups = !g8 && groups >= 4 && groups % 2 == 0 && k_tiles % col_groups == 0;
    const bool mcast = (mcast_g8 || mcast_groups) && SEM_FN0 + col_groups <= 16;
    const uint32_t parts = mcast ? col_groups : 1;
    auto part_of = [&](uint32_t g) { return g8 ? g : g / 2; };
    // FN_UNICAST_MAX (4): with at most this many parts, the groups of 8's parts
    // go to the other groups' slice s as unicasts from the slice reader, on its NoC (NoC0: no multicast in this
    // program), not the writer's multicast on NoC1, where the write side's outputs (to DRAM and to the slice cores)
    // held it back by up to ~16 us on some chips, a different chip each call.
    // 4 groups (32 tokens a row) gain (32 users K = 7 -0.21 ms, 128 users MTP -0.25 ms), 8 lose (128 users K =
    // 3 +0.25 ms: 7 unicasts a slice); 0: never.  The prefill's groups keep the multicast.
    constexpr uint32_t FN_UNICAST_MAX = 4;
    const bool fn_unicast = mcast && mcast_g8 && parts <= FN_UNICAST_MAX;
    // The prefill's merged groups without helpers (512 tokens a call and more): the slice cores of the top 3 grid rows
    // read on NoC 1 (their reader on BRISC, their writer on NCRISC with NoC 0; each RISC still its own NoC): the
    // write's DRAM reads starve the top rows on NoC 0 (the write phase 61 us on the lower rows, 114 on the
    // top).  For that, no multicast in the program (no NoC carrying both a multicast and DRAM reads): fn's parts from
    // the slice writers to the column's other groups by unicasts on the writer's NoC (from the reader, on the NoC of
    // the write's reads: 512 tokens 181 -> 196 us), P and the norm scale from the mix writer to its slices by
    // unicasts, and the slices' partials signalled to the mix core by flags (SEM_H0 + s, written: no atomics).  The
    // same values; 512 / 1,024 tokens 182 / 359 -> 168 / 330 us (the unicasts alone 181 / 357; the top 2 / 4 rows
    // 187 / 175 at 512, rows 2 - 4 + 6 178, the rows' mix cores too 170).  WR_NOC1_ROWS: the grid rows
    // y = 2 + i of its bits i
    constexpr uint32_t WR_NOC1_ROWS = 0x7u;
    const bool wr_unicast = merged && helped == 0 && mcast && mcast_groups;
    const uint32_t wr_mask = wr_unicast ? WR_NOC1_ROWS : 0;
    // (the unicast form: the mix core waits for flag SEM_H0 + s of each slice s, and the program has MERGED_SENDERS of
    // them; slices 1 / 2 / 4 at D 4096, but a hidden of 160 k would take 5 and wait for a semaphore it does not have)
    TT_FATAL(
        !wr_unicast || slices <= MERGED_SENDERS,
        "mhc_stream_write_read: {} slices' partial flags, {} semaphores (SEM_H0 ..)",
        slices,
        MERGED_SENDERS);
    std::vector<uint32_t> sems = {SEM_PART, SEM_NS, SEM_P};
    if (mcast) {
        sems.push_back(SEM_OWN);
        for (uint32_t q = 0; q < parts; ++q) {
            sems.push_back(SEM_FN0 + q);
        }
    }
    const CoreRangeSet sem_cores = fw != nullptr ? all.merge(as_set(write_cores)) : all;
    if (fw != nullptr && !merged) {
        sems.push_back(SEM_S);
    }
    if (merged) {
        TT_FATAL(SEM_FN0 + parts <= SEM_H0, "mhc_stream_write_read: {} parts", parts);
        for (uint32_t i = 0; i < MERGED_SENDERS; ++i) {
            sems.push_back(SEM_H0 + i);
        }
    }
    if (cp) {
        TT_FATAL(SEM_FN0 + parts <= SEM_CP, "mhc_stream_write_read: {} parts", parts);
        sems.push_back(SEM_CP);
    }
    for (uint32_t id : sems) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = sem_cores, .initial_value = 0});
    }

    auto* device = t.streams.device();
    auto noc1_row = [&](const CoreCoord& logical) {
        const uint32_t y = device->worker_core_from_logical_core(logical).y;
        return wr_mask != 0 && y >= 2 && y < 34 && ((wr_mask >> (y - 2)) & 1u) != 0;
    };
    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_read/device/kernels/";
    const std::string mix_dir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_mix/device/kernels/";
    std::vector<uint32_t> ct = {CB_S, CB_FN, CB_G, CB_PIN, CB_NS, d_tiles, row_tiles, k_tiles, SEM_P, SEM_NS};
    ct.insert(ct.end(), {parts, SEM_OWN, SEM_FN0, static_cast<uint32_t>(fw != nullptr), SEM_S});
    ct.push_back(static_cast<uint32_t>(fn_unicast));
    TensorAccessorArgs(streams).append_to(ct);
    TensorAccessorArgs(fn).append_to(ct);
    TensorAccessorArgs(gamma).append_to(ct);
    if (merged) {
        ct = {CBM_SOLD, CBM_BO, CBM_W, CBM_Q, CB_FN, CB_G, CB_PIN, CB_NS, d_tiles, k_tiles, SEM_P, SEM_NS};
        ct.insert(ct.end(), {parts, SEM_OWN, SEM_FN0, k_own, SEM_H0, CBM_READY, CB_S});
        for (auto* b : {fw->streams, fw->block_out, fw->post, fw->comb, fn, gamma, fw->out}) {
            TensorAccessorArgs(b).append_to(ct);
        }
    }
    auto slice_reader = kernel(
        kdir + (merged ? "dataflow/reader_mhc_wr_slice_groups.cpp" : "dataflow/reader_mhc_read_slice_groups.cpp"),
        slice_cores,
        ct,
        ReaderConfigDescriptor{});
    ct = {CB_OUT, CB_Y, CB_PART, slices, row_tiles, k_tiles, SEM_PART, d_tiles, sub_rows};
    ct.insert(ct.end(), {parts, SEM_OWN, SEM_FN0, CB_FN, CB_G, static_cast<uint32_t>(fn_unicast)});
    if (merged) {
        ct.push_back(CB_S);
        ct.push_back(k_own);
    }
    TensorAccessorArgs(out).append_to(ct);
    if (merged) {
        TensorAccessorArgs(fw->out).append_to(ct);
    }
    auto slice_writer = kernel(
        kdir + (merged ? "dataflow/writer_mhc_wr_slice_groups.cpp" : "dataflow/writer_mhc_read_slice_groups.cpp"),
        slice_cores,
        ct,
        WriterConfigDescriptor{});
    auto slice_compute = kernel(
        kdir + "compute/mhc_read_slice.cpp",
        slice_cores,
        {k_tiles, row_tiles, group_tokens},
        ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
    // MHC_WR_PROBE (investigation only, wrong values): bit 1 no DRAM reads of the write's tiles, 2 no products or sums,
    // 4 no DRAM writes of the new streams
    const char* probe_env = std::getenv("MHC_WR_PROBE");
    const uint32_t probe = probe_env == nullptr ? 0u : static_cast<uint32_t>(std::strtoul(probe_env, nullptr, 10));
    auto probe_defines = [&](KernelDescriptor& k, uint32_t bit, const char* name) {
        if (merged && (probe & bit) != 0) {
            k.defines.emplace_back(name, "1");
        }
    };
    if (merged) {
        slice_compute.defines = {{"MHC_WR_MERGED", "1"}, {"MHC_WR_KOWN", std::to_string(k_own)}};
    }
    if (merged && !defer) {  // the slice writer writes each new tile as made (and the reader none)
        slice_reader.defines.emplace_back("MHC_WR_WRITER_DRAM", "1");
        slice_writer.defines.emplace_back("MHC_WR_WRITER_DRAM", "1");
    }
    probe_defines(slice_reader, 1, "MHC_WR_PROBE_NOREAD");
    probe_defines(slice_compute, 2, "MHC_WR_PROBE_NOMATH");
    probe_defines(slice_writer, 4, "MHC_WR_PROBE_NOWRITE");
    if (wr_unicast) {
        slice_writer.defines.emplace_back("MHC_WR_FN_UNICAST_W", "1");
        slice_writer.defines.emplace_back("MHC_WR_PART_FLAGS", std::to_string(SEM_H0));
    }
    if (part_mix) {
        slice_writer.defines.emplace_back("MHC_PART_AFTER_NS", std::to_string(CB_NS));
    }
    ct = {CB_PART, CB_MC, CB_SC, slices, row_tiles, SEM_PART};
    TensorAccessorArgs(mix_consts).append_to(ct);
    TensorAccessorArgs(consts).append_to(ct);
    if (cp) {
        ct.push_back(cp_pages);
        TensorAccessorArgs(fw->comb).append_to(ct);
        TensorAccessorArgs(fw->post).append_to(ct);
    }
    auto mix_reader = kernel(kdir + "dataflow/reader_mhc_read_mix.cpp", mix_cores, ct, ReaderConfigDescriptor{});
    ct = {CB_POST, CB_COMB, CB_P, CB_PIN, row_tiles, slices, SEM_P, CB_NSO, CB_NS, SEM_NS, sub_rows};
    TensorAccessorArgs(post).append_to(ct);
    TensorAccessorArgs(comb).append_to(ct);
    auto mix_writer =
        kernel(kdir + "dataflow/writer_mhc_read_mix_groups.cpp", mix_cores, ct, WriterConfigDescriptor{});
    if (wr_unicast) {
        mix_reader.defines.emplace_back("MHC_WR_PART_FLAGS", std::to_string(SEM_H0));
        mix_writer.defines.emplace_back("MHC_WR_UNICAST", "1");
    }
    if (cp) {
        mix_reader.defines = {{"MHC_CP", std::to_string(CB_CP)}};
        mix_writer.defines = {
            {"MHC_CP", std::to_string(CB_CP)},
            {"MHC_CP_PAGES", std::to_string(cp_pages)},
            {"MHC_CP_SEM", std::to_string(SEM_CP)}};
    }
    const char* sfpu_env = std::getenv("MHC_SINKHORN_SFPU");
    auto mix_compute = kernel(
        mix_dir + "compute/mhc_stream_mix.cpp",
        mix_cores,
        {slices,
         row_tiles,
         attrs.iters,
         std::bit_cast<uint32_t>(attrs.eps),
         std::bit_cast<uint32_t>(attrs.norm_eps),
         std::bit_cast<uint32_t>(attrs.inv_count),
         group_tokens,
         static_cast<uint32_t>(sfpu_env == nullptr || sfpu_env[0] != '0'),
         std::bit_cast<uint32_t>(attrs.inv_hidden),
         std::bit_cast<uint32_t>(attrs.rms_eps)},
        ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
    mix_compute.defines = {{"MHC_READ_NORM", "1"}};
    // GLM53_MHC_FINISH: the finishers' kernels (the mix kernels' MHC_FINISHER forms, the mix core's compile-time args)
    KernelDescriptor fin_reader, fin_writer, fin_compute;
    if (fin) {
        mix_compute.defines.emplace_back("MHC_FIN_AWAY", std::to_string(CB_FIN_SENT));
        mix_writer.defines.emplace_back("MHC_FIN_AWAY", std::to_string(CB_FIN_SENT));
        mix_writer.defines.emplace_back("MHC_FIN_MIX", std::to_string(CB_MIXES));
        mix_writer.defines.emplace_back("MHC_FIN_SEM", std::to_string(SEM_FIN));
        if (const char* fe = std::getenv("GLM53_MHC_FINISH"); fe != nullptr && fe[0] == '2') {  // (the mixes
            // written before P, signalled after P's send: MHC_FIN_SPLIT)
            mix_writer.defines.emplace_back("MHC_FIN_SPLIT", "1");
        }
        fin_reader = kernel(
            kdir + "dataflow/reader_mhc_read_mix.cpp", fin_set, mix_reader.compile_time_args, ReaderConfigDescriptor{});
        fin_reader.defines = {{"MHC_FINISHER", std::to_string(CB_MIXES)}, {"MHC_FINISHER_SEM", std::to_string(SEM_FIN)}};
        fin_writer = kernel(
            kdir + "dataflow/writer_mhc_read_mix_groups.cpp",
            fin_set,
            mix_writer.compile_time_args,
            WriterConfigDescriptor{});
        fin_writer.defines = {{"MHC_FINISHER", "1"}};
        fin_compute = kernel(
            mix_dir + "compute/mhc_stream_mix.cpp",
            fin_set,
            mix_compute.compile_time_args,
            ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
        fin_compute.defines = {{"MHC_READ_NORM", "1"}, {"MHC_FINISHER", "1"}};
    }

    // the NoC 1 rows' slice cores (above): their reader on BRISC with NoC 1 and writer on NCRISC with NoC 0; the same
    // kernels, two descriptors each
    KernelDescriptor slice_reader_up = slice_reader, slice_writer_up = slice_writer;
    slice_reader_up.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1};
    slice_writer_up.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0};
    std::vector<CoreRange> slice_lower, slice_upper;
    for (uint32_t g = 0; g < groups; ++g) {
        const CoreCoord m = mix_at(g);
        const CoreCoord m_noc = device->worker_core_from_logical_core(m);
        const CoreCoord lo = device->worker_core_from_logical_core(CoreCoord{m.x + 1, m.y});
        const CoreCoord hi = device->worker_core_from_logical_core(CoreCoord{m.x + slices, m.y});
        for (uint32_t s = 0; s < slices; ++s) {
            const CoreCoord core{m.x + 1 + s, m.y};
            KernelDescriptor::RTArgList r;
            if (merged) {
                for (auto* b : {fw->streams, fw->block_out, fw->post, fw->comb, fn, gamma}) {
                    r.push_back(b);
                }
                r.push_back(s);
                r.push_back(off_g + g);
                r.push_back(part_of(g));
                r.push_back(senders[g * slices + s]);
                r.push_back(fw->out);
            } else {
                r.push_back(streams);
                r.push_back(fn);
                r.push_back(gamma);
                r.push_back(s);
                r.push_back((off_g + g) * row_tiles);
                r.push_back(part_of(g));
            }
            if (fn_unicast) {  // the other groups' slice s (one a grid row, the same grid column)
                r.push_back(groups - 1);
                for (uint32_t h = 0; h < groups; ++h) {
                    if (h != g) {
                        const CoreCoord d = device->worker_core_from_logical_core(CoreCoord{m.x + 1 + s, mix_at(h).y});
                        r.push_back(static_cast<uint32_t>(d.x));
                        r.push_back(static_cast<uint32_t>(d.y));
                    }
                }
            }
            const bool up = noc1_row(core);
            (up ? slice_upper : slice_lower).push_back(CoreRange(core, core));
            (up ? slice_reader_up : slice_reader).emplace_runtime_args(core, r);
            KernelDescriptor::RTArgList w;
            w.push_back(out);
            w.push_back(s);
            w.push_back(static_cast<uint32_t>(m_noc.x));
            w.push_back(static_cast<uint32_t>(m_noc.y));
            w.push_back(off_g + g);
            // the column of slice s over the groups (the multicast's rectangle: logical (x0 + 1 + s, its groups' rows))
            const CoreCoord top = device->worker_core_from_logical_core(CoreCoord{m.x + 1 + s, mix_at(0).y});
            const CoreCoord bottom = device->worker_core_from_logical_core(CoreCoord{m.x + 1 + s, mix_at(groups - 1).y});
            w.push_back(static_cast<uint32_t>(top.x));
            w.push_back(static_cast<uint32_t>(top.y));
            w.push_back(static_cast<uint32_t>(bottom.y));
            w.push_back(part_of(g));
            if (merged) {
                w.push_back(fw->out);
            }
            if (wr_unicast) {  // the column's other groups' slice s (the groups of g's parity, one a grid row)
                w.push_back(groups / 2 - 1);
                for (uint32_t h = g % 2; h < groups; h += 2) {
                    if (h != g) {
                        const CoreCoord d = device->worker_core_from_logical_core(CoreCoord{m.x + 1 + s, mix_at(h).y});
                        w.push_back(static_cast<uint32_t>(d.x));
                        w.push_back(static_cast<uint32_t>(d.y));
                    }
                }
            }
            (up ? slice_writer_up : slice_writer).emplace_runtime_args(core, w);
        }
        KernelDescriptor::RTArgList r;
        r.push_back(mix_consts);
        r.push_back(consts);
        if (cp) {  // group 0's mix core reads and sends the write side's comb and post
            r.push_back(fw->comb);
            r.push_back(fw->post);
            r.push_back(static_cast<uint32_t>(g == 0));
        }
        mix_reader.emplace_runtime_args(m, r);
        KernelDescriptor::RTArgList w;
        w.push_back(post);
        w.push_back(comb);
        w.push_back(static_cast<uint32_t>(lo.x));
        w.push_back(static_cast<uint32_t>(lo.y));
        w.push_back(static_cast<uint32_t>(hi.x));
        w.push_back(static_cast<uint32_t>(hi.y));
        w.push_back(off_g + g);
        if (cp) {
            if (g == 0) {
                append_cp_rects(w, device, write_cores);
            } else {
                w.push_back(0u);
            }
        }
        if (fin) {  // its finisher's NoC x, y; the finisher's args
            const CoreCoord f = device->worker_core_from_logical_core(fin_at[g]);
            w.push_back(static_cast<uint32_t>(f.x));
            w.push_back(static_cast<uint32_t>(f.y));
            KernelDescriptor::RTArgList fr;
            fr.push_back(mix_consts);
            fr.push_back(consts);
            fin_reader.emplace_runtime_args(fin_at[g], fr);
            KernelDescriptor::RTArgList fw_args;
            fw_args.push_back(post);
            fw_args.push_back(comb);
            for (const auto v : {lo.x, lo.y, hi.x, hi.y}) {
                fw_args.push_back(static_cast<uint32_t>(v));
            }
            fw_args.push_back(off_g + g);
            fin_writer.emplace_runtime_args(fin_at[g], fw_args);
        }
        if (wr_unicast) {  // (not with cp: the groups of 8's) its slices' NoC x, y for P's unicasts
            for (uint32_t s = 0; s < slices; ++s) {
                const CoreCoord d = device->worker_core_from_logical_core(CoreCoord{m.x + 1 + s, m.y});
                w.push_back(static_cast<uint32_t>(d.x));
                w.push_back(static_cast<uint32_t>(d.y));
            }
        }
        mix_writer.emplace_runtime_args(m, w);
    }
    if (!slice_upper.empty()) {
        slice_reader.core_ranges = slice_writer.core_ranges = runs_set(slice_lower);
        slice_reader_up.core_ranges = slice_writer_up.core_ranges = runs_set(slice_upper);
        desc.kernels.push_back(std::move(slice_reader_up));
        desc.kernels.push_back(std::move(slice_writer_up));
    }
    for (auto* k : {&slice_reader, &slice_writer, &slice_compute, &mix_reader, &mix_writer, &mix_compute}) {
        desc.kernels.push_back(std::move(*k));
    }
    if (fin) {
        for (auto* k : {&fin_reader, &fin_writer, &fin_compute}) {
            desc.kernels.push_back(std::move(*k));
        }
    }
    if (merged && !helper_run.empty()) {  // the helper cores
        std::vector<CoreCoord> helpers(write_cores.begin(), write_cores.begin() + helper_run.size());
        const CoreRangeSet hset = as_set(helpers);
        add_cb(desc, hset, CBM_SOLD, MERGED_AHEAD * row_tiles, bf16);
        add_cb(desc, hset, CBM_BO, MERGED_AHEAD, bf16);
        add_cb(desc, hset, CBM_W, 2 * row_tiles, bf16);
        add_cb(desc, hset, CBM_Q, 2 * row_tiles, bf16);
        add_cb(desc, hset, CBM_T1, 2, bf16);
        add_cb(desc, hset, CBM_T2, 2, bf16);
        add_cb(desc, hset, CBM_HOUT, 2 * row_tiles, bf16);
        add_cb(desc, hset, CBM_SCR, 1, f32);  // (4 KB: a group's comb and post rows)
        std::vector<uint32_t> hct = {CBM_SOLD, CBM_BO, CBM_W, CBM_Q, d_tiles, k_tiles, k_own, slices, CBM_SCR};
        for (auto* b : {fw->streams, fw->block_out, fw->post, fw->comb}) {
            TensorAccessorArgs(b).append_to(hct);
        }
        auto h_reader = kernel(kdir + "dataflow/reader_mhc_wr_helper.cpp", hset, hct, ReaderConfigDescriptor{});
        auto h_compute = kernel(
            kdir + "compute/mhc_wr_helper.cpp",
            hset,
            {CBM_SOLD, CBM_BO, CBM_W, CBM_Q, CBM_T1, CBM_T2, CBM_HOUT, k_tiles, k_own, slices},
            ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
        hct = {CBM_HOUT, CB_S, SEM_H0, d_tiles, k_tiles, k_own, slices};
        TensorAccessorArgs(fw->out).append_to(hct);
        auto h_writer = kernel(kdir + "dataflow/writer_mhc_wr_helper.cpp", hset, hct, WriterConfigDescriptor{});
        {  // the slice cores' NoC x, y (slice j = g S + s of the call)
            KernelDescriptor::RTArgList common;
            for (uint32_t g = 0; g < groups; ++g) {
                const CoreCoord m = mix_at(g);
                for (uint32_t s = 0; s < slices; ++s) {
                    const CoreCoord noc = device->worker_core_from_logical_core(CoreCoord{m.x + 1 + s, m.y});
                    common.push_back(static_cast<uint32_t>(noc.x));
                    common.push_back(static_cast<uint32_t>(noc.y));
                }
            }
            h_writer.emplace_common_runtime_args(common);
        }
        for (uint32_t i = 0; i < helper_run.size(); ++i) {
            const auto [start, count] = helper_run[i];
            KernelDescriptor::RTArgList r;
            for (auto* b : {fw->streams, fw->block_out, fw->post, fw->comb}) {
                r.push_back(b);
            }
            r.push_back(start);
            r.push_back(count);
            r.push_back(off_g);
            h_reader.emplace_runtime_args(helpers[i], r);
            KernelDescriptor::RTArgList c;
            c.push_back(start);
            c.push_back(count);
            h_compute.emplace_runtime_args(helpers[i], c);
            KernelDescriptor::RTArgList w;
            w.push_back(fw->out);
            w.push_back(start);
            w.push_back(count);
            w.push_back(off_g);
            for (uint32_t slot : helper_slots[i]) {
                w.push_back(slot);
            }
            h_writer.emplace_runtime_args(helpers[i], w);
        }
        if (!defer) {  // the helpers write their tiles to DRAM too
            h_writer.defines.emplace_back("MHC_WR_HELPER_DRAM", "1");
        }
        probe_defines(h_reader, 1, "MHC_WR_PROBE_NOREAD");
        probe_defines(h_compute, 2, "MHC_WR_PROBE_NOMATH");
        probe_defines(h_writer, 4, "MHC_WR_PROBE_NOWRITE");
        for (auto* k : {&h_reader, &h_compute, &h_writer}) {
            desc.kernels.push_back(std::move(*k));
        }
    }
    if (fw != nullptr && !merged) {
        // the write's tiles of row tile rt to group rt / row_tiles (one row tile a group of 8, four a group of 32)
        add_write_side(
            desc,
            *fw,
            write_side,
            device,
            d_tiles,
            groups * row_tiles,
            k_tiles,
            slices,
            row_tiles,
            slice_at,
            cp,
            off_g * row_tiles * d_tiles);
    }
    if (fin) {
        add_finisher_cbs(desc, mix_cores, fin_set, sems, sfpu_env == nullptr || sfpu_env[0] != '0');
    }
    return desc;
}


// The single form (at most 64 rows: 16 tokens): the mix core (0, 0) and the slice cores beside it
ProgramDescriptor create_single_descriptor(
    const MhcStreamReadParams& attrs,
    const MhcStreamReadTensorArgs& t,
    Buffer* post,
    Buffer* comb,
    Buffer* out,
    const FusedWrite* fw = nullptr) {
    ProgramDescriptor desc;
    Buffer* streams = t.streams.buffer();
    Buffer* fn = t.fn.buffer();
    Buffer* mix_consts = t.mix_consts.buffer();
    Buffer* consts = t.consts.buffer();
    Buffer* gamma = t.gamma.buffer();

    const uint32_t d_tiles = t.streams.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t row_tiles = t.streams.logical_shape()[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t slices = attrs.slices;
    const uint32_t k_tiles = d_tiles / slices;
    const uint32_t slice_rows = (slices + SLICE_W - 1) / SLICE_W;
    const uint32_t slice_cols = slices < SLICE_W ? slices : SLICE_W;

    const CoreRangeSet mix_core{CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0})};
    const CoreRangeSet slice_cores{
        CoreRange(CoreCoord{SLICE_X0, 0}, CoreCoord{SLICE_X0 + slice_cols - 1, slice_rows - 1})};
    const CoreRangeSet all = mix_core.merge(slice_cores);
    const auto f32 = tt::DataFormat::Float32;
    const auto bf16 = tt::DataFormat::Float16_b;
    std::vector<CoreCoord> write_cores, slice_at;
    const bool cp = fw != nullptr;  // comb and post multicast (CB_CP)
    const uint32_t cp_pages = cp ? (attrs.tokens + 31) / 32 : 0;
    if (fw != nullptr) {  // CB_S first on every core of the program (the write cores' hand-off address)
        std::vector<CoreCoord> taken = {CoreCoord{0, 0}};
        for (uint32_t s = 0; s < slices; ++s) {
            taken.push_back(CoreCoord{SLICE_X0 + s % SLICE_W, s / SLICE_W});
            slice_at.push_back(CoreCoord{SLICE_X0 + s % SLICE_W, s / SLICE_W});
        }
        write_cores = free_cores(t.streams.device()->compute_with_storage_grid_size(), taken);
        add_cb(desc, all.merge(as_set(write_cores)), CB_S, row_tiles * k_tiles, bf16);
        if (cp) {
            add_cb(desc, mix_core.merge(as_set(write_cores)), CB_CP, 2 * cp_pages, f32);
        }
    }

    // GLM53_MHC_FINISH (above): the finisher right of the slices where a write core
    std::vector<CoreCoord> write_side = write_cores, fin_at;
    if (cp && slice_rows == 1) {
        fin_at = pick_finishers(write_side, {CoreCoord{SLICE_X0 + slice_cols, 0}}, row_tiles * d_tiles);
    }
    const bool fin = !fin_at.empty();
    const CoreRangeSet fin_set = fin ? as_set(fin_at) : mix_core;
    add_cb(desc, all, CB_PART, slices * row_tiles * 4, f32);
    add_cb(desc, all, CB_PIN, row_tiles, f32);
    add_cb(desc, all, CB_NS, 1, f32);

    add_cb(desc, mix_core, CB_MC, 13, f32);
    add_cb(desc, mix_core, CB_SC, 8, f32);
    add_cb(desc, mix_core, CB_POST, 1, f32);
    add_cb(desc, mix_core, CB_COMB, 1, f32);
    add_cb(desc, mix_core, CB_P, MAX_ROW_TILES, f32);
    add_cb(desc, mix_core, CB_NSO, 1, f32);
    add_cb(desc, mix_core, CB_P2, MAX_ROW_TILES, f32);
    add_cb(desc, mix_core, CB_PG, MAX_ROW_TILES, f32);
    add_cb(desc, mix_core, CB_Q, MAX_ROW_TILES, f32);
    add_cb(desc, mix_core, tt::CBIndex::c_16, 4 * MAX_ROW_TILES, f32);  // the summed partials per row tile
    add_cb(desc, mix_core, tt::CBIndex::c_17, 3 * MAX_ROW_TILES, f32);  // the own-block masked projections
    add_cb(desc, mix_core, tt::CBIndex::c_18, MAX_ROW_TILES, f32);      // the Gram diagonals
    add_cb(desc, mix_core, tt::CBIndex::c_19, 3, f32);                  // own
    for (uint32_t cb = tt::CBIndex::c_20; cb <= tt::CBIndex::c_23; ++cb) {  // squares, mixes, scale, mixes
        add_cb(desc, fin && cb == CB_MIXES ? mix_core.merge(fin_set) : mix_core, cb, 1, f32);
    }
    add_cb(desc, mix_core, tt::CBIndex::c_24, 2, f32);  // Sinkhorn ping
    add_cb(desc, mix_core, tt::CBIndex::c_25, 2, f32);  // Sinkhorn pong
    add_cb(desc, mix_core, tt::CBIndex::c_26, 1, f32);  // pre
    add_cb(desc, mix_core, tt::CBIndex::c_27, 1, f32);  // pre spread over the rows' lanes

    if (fw == nullptr) {
        add_cb(desc, slice_cores, CB_S, row_tiles * k_tiles, bf16);
    }
    add_cb(desc, slice_cores, CB_FN, k_tiles * 3, bf16);
    add_cb(desc, slice_cores, CB_G, k_tiles, bf16);
    add_cb(desc, slice_cores, CB_OUT, 4 * row_tiles, f32);
    add_cb(desc, slice_cores, CB_X, k_tiles, bf16);
    add_cb(desc, slice_cores, CB_Y, k_tiles, bf16);

    const CoreRangeSet sem_cores = fw != nullptr ? all.merge(as_set(write_cores)) : all;
    std::vector<uint32_t> sems = {SEM_PART, SEM_NS, SEM_P};
    if (fw != nullptr) {
        sems.push_back(SEM_S);
    }
    if (cp) {
        sems.push_back(SEM_CP);
    }
    for (uint32_t id : sems) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = sem_cores, .initial_value = 0});
    }

    auto* device = t.streams.device();
    const CoreCoord mix_noc = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const CoreCoord lo = device->worker_core_from_logical_core(CoreCoord{SLICE_X0, 0});
    const CoreCoord hi = device->worker_core_from_logical_core(CoreCoord{SLICE_X0 + slice_cols - 1, slice_rows - 1});

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_read/device/kernels/";
    const std::string mix_dir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_stream_mix/device/kernels/";

    // the slice cores
    std::vector<uint32_t> ct = {CB_S, CB_FN, CB_G, CB_PIN, CB_NS, d_tiles, row_tiles, k_tiles, SEM_P, SEM_NS};
    ct.insert(ct.end(), {static_cast<uint32_t>(fw != nullptr), SEM_S});
    TensorAccessorArgs(streams).append_to(ct);
    TensorAccessorArgs(fn).append_to(ct);
    TensorAccessorArgs(gamma).append_to(ct);
    auto slice_reader = kernel(kdir + "dataflow/reader_mhc_read_slice.cpp", slice_cores, ct, ReaderConfigDescriptor{});
    ct = {CB_OUT, CB_Y, CB_PART, slices, row_tiles, k_tiles, SEM_PART, mix_noc.x, mix_noc.y};
    TensorAccessorArgs(out).append_to(ct);
    auto slice_writer = kernel(kdir + "dataflow/writer_mhc_read_slice.cpp", slice_cores, ct, WriterConfigDescriptor{});
    auto slice_compute = kernel(
        kdir + "compute/mhc_read_slice.cpp",
        slice_cores,
        {k_tiles, row_tiles, attrs.tokens},
        ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});

    // the mix core
    ct = {CB_PART, CB_MC, CB_SC, slices, row_tiles, SEM_PART};
    TensorAccessorArgs(mix_consts).append_to(ct);
    TensorAccessorArgs(consts).append_to(ct);
    if (cp) {
        ct.push_back(cp_pages);
        TensorAccessorArgs(fw->comb).append_to(ct);
        TensorAccessorArgs(fw->post).append_to(ct);
    }
    auto mix_reader = kernel(kdir + "dataflow/reader_mhc_read_mix.cpp", mix_core, ct, ReaderConfigDescriptor{});
    ct = {CB_POST, CB_COMB, CB_P, CB_PIN, row_tiles, slices, SEM_P, lo.x, lo.y, hi.x, hi.y, CB_NSO, CB_NS, SEM_NS};
    TensorAccessorArgs(post).append_to(ct);
    TensorAccessorArgs(comb).append_to(ct);
    auto mix_writer = kernel(kdir + "dataflow/writer_mhc_read_mix.cpp", mix_core, ct, WriterConfigDescriptor{});
    if (cp) {
        mix_reader.defines = {{"MHC_CP", std::to_string(CB_CP)}};
        mix_writer.defines = {
            {"MHC_CP", std::to_string(CB_CP)},
            {"MHC_CP_PAGES", std::to_string(cp_pages)},
            {"MHC_CP_SEM", std::to_string(SEM_CP)}};
    }
    const char* sfpu_env = std::getenv("MHC_SINKHORN_SFPU");
    auto mix_compute = kernel(
        mix_dir + "compute/mhc_stream_mix.cpp",
        mix_core,
        {slices,
         row_tiles,
         attrs.iters,
         std::bit_cast<uint32_t>(attrs.eps),
         std::bit_cast<uint32_t>(attrs.norm_eps),
         std::bit_cast<uint32_t>(attrs.inv_count),
         attrs.tokens,
         static_cast<uint32_t>(sfpu_env == nullptr || sfpu_env[0] != '0'),
         std::bit_cast<uint32_t>(attrs.inv_hidden),
         std::bit_cast<uint32_t>(attrs.rms_eps)},
        ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
    mix_compute.defines = {{"MHC_READ_NORM", "1"}};
    // GLM53_MHC_FINISH: the finisher's kernels (the mix kernels' MHC_FINISHER forms, the mix core's compile-time args)
    KernelDescriptor fin_reader, fin_writer, fin_compute;
    if (fin) {
        mix_compute.defines.emplace_back("MHC_FIN_AWAY", std::to_string(CB_FIN_SENT));
        mix_writer.defines.emplace_back("MHC_FIN_AWAY", std::to_string(CB_FIN_SENT));
        mix_writer.defines.emplace_back("MHC_FIN_MIX", std::to_string(CB_MIXES));
        mix_writer.defines.emplace_back("MHC_FIN_SEM", std::to_string(SEM_FIN));
        if (const char* fe = std::getenv("GLM53_MHC_FINISH"); fe != nullptr && fe[0] == '2') {  // (the mixes
            // written before P, signalled after P's send: MHC_FIN_SPLIT)
            mix_writer.defines.emplace_back("MHC_FIN_SPLIT", "1");
        }
        fin_reader = kernel(
            kdir + "dataflow/reader_mhc_read_mix.cpp", fin_set, mix_reader.compile_time_args, ReaderConfigDescriptor{});
        fin_reader.defines = {{"MHC_FINISHER", std::to_string(CB_MIXES)}, {"MHC_FINISHER_SEM", std::to_string(SEM_FIN)}};
        fin_writer = kernel(
            kdir + "dataflow/writer_mhc_read_mix.cpp", fin_set, mix_writer.compile_time_args, WriterConfigDescriptor{});
        fin_writer.defines = {{"MHC_FINISHER", "1"}};
        fin_compute = kernel(
            mix_dir + "compute/mhc_stream_mix.cpp",
            fin_set,
            mix_compute.compile_time_args,
            ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
        fin_compute.defines = {{"MHC_READ_NORM", "1"}, {"MHC_FINISHER", "1"}};
    }

    for (uint32_t s = 0; s < slices; ++s) {
        const CoreCoord core{SLICE_X0 + s % SLICE_W, s / SLICE_W};
        KernelDescriptor::RTArgList r;
        r.push_back(streams);
        r.push_back(fn);
        r.push_back(gamma);
        r.push_back(s);
        slice_reader.emplace_runtime_args(core, r);
        KernelDescriptor::RTArgList w;
        w.push_back(out);
        w.push_back(s);
        slice_writer.emplace_runtime_args(core, w);
    }
    {
        KernelDescriptor::RTArgList r;
        r.push_back(mix_consts);
        r.push_back(consts);
        if (cp) {
            r.push_back(fw->comb);
            r.push_back(fw->post);
            r.push_back(1u);
        }
        mix_reader.emplace_runtime_args(CoreCoord{0, 0}, r);
        KernelDescriptor::RTArgList w;
        w.push_back(post);
        w.push_back(comb);
        if (cp) {
            append_cp_rects(w, device, write_cores);
        }
        if (fin) {  // the finisher's NoC x, y; its args
            const CoreCoord f = device->worker_core_from_logical_core(fin_at[0]);
            w.push_back(static_cast<uint32_t>(f.x));
            w.push_back(static_cast<uint32_t>(f.y));
            KernelDescriptor::RTArgList fr;
            fr.push_back(mix_consts);
            fr.push_back(consts);
            fin_reader.emplace_runtime_args(fin_at[0], fr);
            KernelDescriptor::RTArgList fw_args;
            fw_args.push_back(post);
            fw_args.push_back(comb);
            fin_writer.emplace_runtime_args(fin_at[0], fw_args);
        }
        mix_writer.emplace_runtime_args(CoreCoord{0, 0}, w);
    }
    for (auto* k : {&slice_reader, &slice_writer, &slice_compute, &mix_reader, &mix_writer, &mix_compute}) {
        desc.kernels.push_back(std::move(*k));
    }
    if (fin) {
        for (auto* k : {&fin_reader, &fin_writer, &fin_compute}) {
            desc.kernels.push_back(std::move(*k));
        }
    }
    if (fw != nullptr) {
        add_write_side(desc, *fw, write_side, device, d_tiles, row_tiles, k_tiles, slices, 0, slice_at, cp);
    }
    if (fin) {
        add_finisher_cbs(desc, mix_core, fin_set, sems, sfpu_env == nullptr || sfpu_env[0] != '0');
    }
    return desc;
}
}  // namespace

ProgramDescriptor MhcStreamReadProgramFactory::create_descriptor(
    const MhcStreamReadParams& attrs,
    const MhcStreamReadTensorArgs& t,
    tensor_return_value_t& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    Buffer* post = outputs[0].buffer();
    Buffer* comb = outputs[1].buffer();
    Buffer* out = outputs[2].buffer();
    if (t.streams.logical_shape()[-2] > 64) {
        return create_groups_descriptor(attrs, t, post, comb, out);
    }
    return create_single_descriptor(attrs, t, post, comb, out);
}

ProgramDescriptor MhcStreamWriteReadProgramFactory::create_descriptor(
    const MhcStreamReadParams& attrs,
    const MhcStreamWriteReadTensorArgs& t,
    tensor_return_value_t& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const MhcStreamReadTensorArgs read{t.streams, t.fn, t.mix_consts, t.consts, t.gamma};
    const FusedWrite fw{
        t.streams.buffer(),
        t.block_out.buffer(),
        t.post.buffer(),
        t.comb.buffer(),
        outputs[0].buffer(),
        t.block_out.logical_shape()[-2]};
    Buffer* post = outputs[1].buffer();
    Buffer* comb = outputs[2].buffer();
    Buffer* out = outputs[3].buffer();
    if (t.streams.logical_shape()[-2] > 64) {
        return create_groups_descriptor(attrs, read, post, comb, out, &fw);
    }
    return create_single_descriptor(attrs, read, post, comb, out, &fw);
}

}  // namespace ttnn::experimental::prim
