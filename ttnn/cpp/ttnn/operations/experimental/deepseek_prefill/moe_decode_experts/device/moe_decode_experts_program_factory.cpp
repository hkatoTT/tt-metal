// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode_experts_program_factory.hpp"
#include "moe_decode_experts_device_operation_types.hpp"

#include <bit>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <functional>
#include <optional>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
// whether a weight tensor is ND-sharded a block or more a shard (a core's slice contiguous in one bank): the weight
// streams read a block in one request
bool block_sharded(const Tensor& t) {
    return t.memory_config().is_sharded() && t.memory_config().nd_shard_spec().has_value() &&
           t.memory_config().nd_shard_spec()->shard_shape[-2] != tt::constants::TILE_HEIGHT;
}

const std::string KDIR = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_decode_experts/device/kernels/";
// the down cores west of the gate | up cores, so the h tiles go west to them on NoC 1
constexpr uint32_t DN_X = 4, DN_Y = 8, DN_X0 = 0;     // the down cores: x 0-3, y 0-7 (core q = 4 y + x)
constexpr uint32_t GU_X = 8, GU_Y = 8, GU_X0 = DN_X;  // the gate | up cores: x 4-11, y 0-7 (core p = 8 y + x - 4)
// packed units (builder cores): the records and the two RISCs' unit lists (moe_decode_experts_common.hpp's rec_zeros)
constexpr uint32_t scr_recs(uint32_t rows, uint32_t local) {
    const uint32_t rec = ((local + 3) / 4 * 4 + 2 * local + 63) / 64 * 64 + 4 * 32 * 16;
    const uint32_t lists = (6 * (local + rows * 16) + 2 * local + 63) / 64 * 64;
    return rows * rec + 2 * lists;
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

KernelDescriptor dm_kernel(
    const std::string& name,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    DataMovementProcessor processor) {
    KernelDescriptor k;
    k.kernel_source = KDIR + "dataflow/" + name;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    // the weight streams (RISCV_1) read DRAM on NoC 0, as ttnn's readers do (on NoC 1 they ran at about half the
    // rate); the control kernels (RISCV_0: the multicasts, the semaphores, y) are on NoC 1
    k.config = DataMovementConfigDescriptor{
        .processor = processor, .noc = processor == DataMovementProcessor::RISCV_1 ? NOC::NOC_0 : NOC::NOC_1};
    return k;
}

KernelDescriptor compute_kernel(
    const std::string& name,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    bool full_sync = false,
    std::optional<uint32_t> fp32_to_dest = std::nullopt,  // a CB unpacked to Dest as fp32
    MathFidelity fidelity = MathFidelity::HiFi4) {
    KernelDescriptor k;
    k.kernel_source = KDIR + "compute/" + name;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    std::vector<UnpackToDestMode> to_dest(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_to_dest) {
        to_dest[*fp32_to_dest] = UnpackToDestMode::UnpackToDestFp32;
    }
    k.config = ComputeConfigDescriptor{
        .math_fidelity = fidelity,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = full_sync,
        .unpack_to_dest_mode = std::move(to_dest),
    };
    return k;
}

KernelDescriptor::RTArgList args(Buffer* a, Buffer* b) {
    KernelDescriptor::RTArgList rt;
    rt.push_back(a);
    rt.push_back(b);
    return rt;
}

void for_each_core(const CoreRangeSet& cores, const std::function<void(const CoreCoord&)>& fn) {
    for (const auto& range : cores.ranges()) {
        for (auto core : range) {
            fn(core);
        }
    }
}

// The experts.  Gate | up cores: BRISC gu_ctrl (x, the routing columns, h to the down cores), NCRISC the W1 stream,
// the compute.  Down cores: BRISC dn_ctrl (the h slots, y), NCRISC the W2 stream, the compute.  Each chip holds I = 256
// (T = 8 column tiles) of every expert of its mesh column: an item is a group of up to 64 / T chosen experts (64
// gate | up column tiles, 64 down K tiles).
ProgramDescriptor experts_descriptor(
    const MoeDecodeExpertsParams& attrs,
    const MoeDecodeExpertsTensorArgs& t,
    Tensor& output,
    const CoreRangeSet& gu,
    const CoreRangeSet& dn,
    uint32_t kd,
    uint32_t ki,
    uint32_t cols,
    tt::DataFormat wdf,
    tt::DataFormat wdf2,  // W1's and W2's formats (the low-precision patterns may differ)
    const std::optional<ttnn::MeshCoordinate>& coord) {
    constexpr uint32_t CB_X = tt::CBIndex::c_0;      // gate | up: x (bf16, kd tiles)
    constexpr uint32_t CB_W = tt::CBIndex::c_1;      // the weight blocks (W1 on gate | up cores, W2 on down cores)
    constexpr uint32_t CB_WCOL = tt::CBIndex::c_2;   // gate | up: the routing column (bf16)
    constexpr uint32_t CB_SCR = tt::CBIndex::c_3;    // the routing tile, BRISC's at 0 and NCRISC's at 4 KB
    constexpr uint32_t CB_H = tt::CBIndex::c_4;      // down: the 3 h slots, right after CB_SCR (below)
    constexpr uint32_t CB_HOUT = tt::CBIndex::c_16;  // gate | up: this core's h tile
    constexpr uint32_t CB_Y = tt::CBIndex::c_17;     // down: the output columns (bf16)
    constexpr uint32_t CB_ACC = tt::CBIndex::c_18;   // down, packed units: the rows' sums (fp32)
    // packed units (below): gate | up: the tokens' K slice, the unit's 0/1 tiles, its gathered K slice, the partial
    // gate and up of the T columns, and the row's partials of this core's column (2 slots); down: the slots' 0/1
    // tiles transposed, a slot's y (fp32)
    constexpr uint32_t CB_XALL = tt::CBIndex::c_5, CB_P = tt::CBIndex::c_6, CB_XS = tt::CBIndex::c_7;
    constexpr uint32_t CB_PART = tt::CBIndex::c_19, CB_PIN = tt::CBIndex::c_20;
    // K tiles per weight block, and blocks per CB (DEPTH - 1 of them in flight)
    auto env = [](const char* name, uint32_t fallback) {
        const char* v = std::getenv(name);
        return v ? static_cast<uint32_t>(std::stoul(v)) : fallback;
    };
    constexpr uint32_t W1_BLOCK = 16;
    constexpr uint32_t SEM_READY = 0, SEM_XVALID = 1, SEM_FREE = 2, SEM_ARRIVE0 = 3, SEM_HVALID0 = 6, SEM_DFREE = 9;
    constexpr uint32_t SEM_PARRIVE0 = 10, SEM_PFREE0 = 12;  // packed units: the partials' arrivals, frees (2 slots)

    ProgramDescriptor desc;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const CoreRangeSet all = gu.merge(dn);
    const uint32_t num_gu = GU_X * GU_Y, num_dn = DN_X * DN_Y;
    // x is 1-16 tile rows of tokens and `weights` the chosen ids and weights per row (moe_decode_route's, the column's
    // range): one tile row an item per group of up to 64 / T chosen experts; more, packed units (below)
    const uint32_t T = attrs.inter / tt::constants::TILE_WIDTH;  // an expert's column tiles on the chip (8)
    const uint32_t rows = t.x.logical_shape()[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t d_tiles = t.x.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    // (the progressive form, GLM53_MOE_PROG_F, below: packed units from one tile row of tokens too, its paths being
    // the units' only)
    const bool prog_env = std::getenv("GLM53_MOE_PROG_F") != nullptr;
    const bool units = rows > 1 || prog_env;  // (below)
    // one tile row: W2 blocks of one slot (8 / 32 tokens 162 / 373 -> 154 / 367 us); packed units: a ring of 3 blocks
    // (128 / 256 tokens 672 / 842 -> 666 / 816 us)
    const uint32_t W2_BLOCK = !units ? 8 : 16;
    const uint32_t DEPTH = units ? 3 : 4;
    // the weight streams' requests alternating between NOC0 and NOC1 (8 tiles each): one NOC's reads cap at ~47 GB/s a
    // DRAM bank, both reach its ~64, NOC1 from when the control kernel on
    // the core is done reading (SEM_NOC1, stream_active.cpp).  One tile row with both: 8 / 32 tokens 155.1 / 369.6 ->
    // 136.5 / 314.9 us (decode 8 / 32 users -1.1 / -2.8 ms).  The packed units, 128 tokens: 662.6 us, both
    // groups 741.8 (NOC1 crowded), down only 637.0, gate | up only 624.2 (decode 128 users -1.3 ms).  They hung
    // now and then (the units with gate | up in ~1 min of decode; one tile row once in ~6,000 steps at 32
    // users), every time down core 0's BRISC stuck issuing the h unit's NOC1 multicast (the NoC jammed by that
    // multicast and the NOC1 weight reads, it seems): so the safe forms (below: no multicast while NOC1 reads).
    // Measured (8 / 32 / 128 / 256 / 512 tokens, us; decode users ms): NOC0 only 149.6 / 363.0 / 661.5 / 611.7 /
    // 635.1; down only 140.0 / 331.4 / 564.8 / 565.7 / 596.6; gate | up only 128.2 / 302.6 / 581.3 / 597.4 / 630.8
    // (decode 23.0 / 35.2 / 55.2 / 62.0 / 82.9 -> 21.0 / 32.0 / 52.0 / 62.0 / 83.3); both 123.1 / 308.4 / 696.7 / 688.2
    // / 708.0.  Gate | up with one row of tokens, down with packed units; traced 2,000-step stress runs without a hang:
    // 8 / 32 / 128 / 256 / 512 users 4 / 4 / 3 / 3 / 3 runs.
    constexpr uint32_t SEM_NOC1 = 15;
    // one row of tokens: the routing read once by the x sender and multicast to every core (gu_ctrl.cpp), in place of
    // ~290 RISCs reading the one 4 KB page from DRAM (8-15 us); semaphore
    // SEM_ROUTE.
    // The routing on every core 14 -> 4 us, but x reached the gate | up cores ~5 us later (the sender multicast the
    // routing first): decode the same.  Since the sender reads x before it waits for the other cores: 1 / 8 /
    // 32 tokens 62.1 / 153.0 / 367.7 -> 54.9 / 148.9 / 363.0 us; 2 x 2,000 traced steps at 32 and at 8 users
    // without a hang
    const uint32_t BOTH_GU = !units ? 1 : 0, BOTH_DN = units ? 1 : 0;
    constexpr uint32_t SEM_ROUTE = 14;
    // each W1 block one shard (its 8 blocks of an expert's column in the 8 banks); W2 a core's slice one
    TT_FATAL(
        block_sharded(t.w1) && t.w1.memory_config().nd_shard_spec()->shard_shape[-2] % (W1_BLOCK * 32) == 0,
        "w1 must be ND-sharded a whole number of blocks ({} tile rows) a shard", W1_BLOCK);
    TT_FATAL(block_sharded(t.w2), "w2 must be ND-sharded a down core's slice a shard");
    TT_FATAL(W2_BLOCK % T == 0, "the W2 blocks ({} K tiles) must hold whole slots ({} tiles)", W2_BLOCK, T);
    // With 2-16 tile rows (T = 8, a row of gate | up cores a slot): packed units
    // (moe_decode_experts_common.hpp's packed_units), each slot one expert's tokens packed in a tile row.  The slot's
    // T gate | up cores each gather their K slice of the unit's x (its 0/1 tiles times the tokens' K slice, kept in
    // L1: exact copies), make the partial gate and up of all T columns over that slice (the slice's W1 block of each
    // column, all in one bank), and send each column's pair to its core, which adds the T pairs in fp32 in a fixed
    // order (the matmul's single accumulation, in another order) and makes h; the down cores make each slot's y
    // (fp32), times its 0/1 tiles transposed into the tokens' rows, added up in fp32 in L1 over the slots and items.
    // The tokens' K slice j is the same for the 8 cores of grid column j: the top one reads it and multicasts it down
    // the column (rather than each reading its own, 8 times the reads).
    // Packed units: the gather by matmul_block (exact copies), and from SCATTER_ROWS tile rows of tokens on (9:
    // 512-token calls, 16 rows) the down cores' scatter in passes of SCATTER_SLOTS slots (the pass's y kept, each token
    // row's sum over them in Dest, one packer accumulation a row and pass; fp32 in another order than slot by slot)
    constexpr uint32_t SCATTER_ROWS = 9, SCATTER_SLOTS = 4, GATHER_MASK_ROWS = 9;
    const uint32_t scatter_slots = units && rows >= SCATTER_ROWS ? SCATTER_SLOTS : 1;
    TT_FATAL(scatter_slots == 1 || (num_gu / T) % scatter_slots == 0, "SCATTER_SLOTS must divide the slots of an item");
    // ... each token row's pass over only the slots with a token in it (the item's incidence after its 0/1 tiles)
    const uint32_t scatter_mask = scatter_slots > 1 ? 1 : 0;
    // the gather (by matmul_block) over only the token rows the unit has tokens in (the same sums), from
    // GATHER_MASK_ROWS tile rows of tokens on
    const uint32_t gather_mask = units && rows >= GATHER_MASK_ROWS ? 1 : 0;
    // Packed units: the routing's id tiles scanned once by builder cores, tile row b by core (b mod 8, 8 + b / 8)
    // (route_build.cpp), their records gathered on builder 0 and multicast from there to every gate | up and down core,
    // in place of each of their RISCs reading and scanning all the tiles twice (moe_decode_experts_common.hpp's rec_*:
    // the same units, the same y bit for bit).  256 / 512 tokens 765 / 873 -> 612 / 635 us with the rest below
    constexpr uint32_t build_rows = 8;
    // The weights read on both NOCs safely: no NoC multicast while any core reads DRAM on NOC1 (in one op the two
    // jammed now and then): h to the down cores by unicasts in two steps (gate | up core -> its grid row's
    // down cores -> their grid columns), and NOC1's reads only once the op's other multicasts (the routing and x with
    // one row of tokens; the builders' records and x's slices down the grid columns with packed units) are all done: a
    // coordinator raises every core's go semaphore
    // The progressive form (a quality-changing option, off by default): each slot's
    // weights a base and a correction (BFP4(W), BFP4(W - base)), the corrections after the bases (w1: slots local ..
    // 2 local - 1; w2: each down core's slice 2 local slots, the corrections from local on), and builder 0 marking the
    // column's experts read with their correction this call, the largest weight sums first, within a budget of the
    // column's reads: GLM53_MOE_PROG_F (the fraction of its chosen experts' BFP8 bytes; 1.06 or more: every one).
    // Packed units from builders only, from one tile row of tokens too (above; dedupe as without it, by item:
    // stream_active.cpp); read at the program's creation (not in the program cache's key)
    uint32_t prog_milli = 0;
    bool prog = false;
    if (const char* v = std::getenv("GLM53_MOE_PROG_F")) {
        prog = true;
        prog_milli = static_cast<uint32_t>(std::lround(std::stod(v) * 1000.0));
        TT_FATAL(units, "GLM53_MOE_PROG_F: packed units only ({} tile rows of tokens)", rows);
    }
    // Row weights (GLM53_MOE_PROG_ROWS, with the progressive form): a comma list of a verify
    // step's row weights P(a >= j), j = 0 .. T - 1 (T = K + 1 tokens a user), the builders' sums weighted by them
    // (route_build.cpp) when the column's N tokens are its 8 mesh rows of T U each, U <= 2 (8 or 16 users: at 32 it
    // did not help, at 128 users K = 3 it hurt); read at the program's creation (not in the program cache's key)
    std::string prog_rows_q;
    uint32_t prog_rows_trow = 0, prog_rows_u = 0;
    if (const char* v = std::getenv("GLM53_MOE_PROG_ROWS"); prog && v != nullptr && *v != '\0') {
        std::vector<uint32_t> q;
        for (std::string s(v); !s.empty();) {
            const size_t c = s.find(',');
            q.push_back(static_cast<uint32_t>(std::lround(std::stod(s.substr(0, c)) * 256.0)));
            s = c == std::string::npos ? std::string() : s.substr(c + 1);
        }
        const uint32_t n = t.x.logical_shape()[-2], trow = n / 8, tq = q.size();
        if (n % 8 == 0 && trow % tq == 0 && trow / tq <= 2) {
            prog_rows_trow = trow;
            prog_rows_u = trow / tq;
            for (uint32_t j = 0; j < tq; ++j) {
                TT_FATAL(q[j] <= 256, "GLM53_MOE_PROG_ROWS: the weights in [0, 1]");
                prog_rows_q += (j != 0 ? "," : "") + std::to_string(q[j]);
            }
        }
    }
    // the go semaphore: packed units the x sender's (no one's with builders), one row a packed unit's partials' (none
    // then); builder 0 counts the grid columns' x multicasts done on its own id 2
    constexpr uint32_t SEM_XDONE = 2;
    const uint32_t SEM_GO1 = units ? SEM_READY : SEM_PARRIVE0;
    constexpr uint32_t SEM_REC = 14, SEM_RDY = 0, SEM_GO = 1;  // SEM_RDY and SEM_GO on the builders only
    // ... and the down cores' 0/1 tiles in bfp8_b, two items' (dn_ctrl.cpp; exact 0 / 1), and the grid column's top
    // core's x slice read by its weight stream on NOC0 while the records come (gu_ctrl.cpp)
    CoreRangeSet bld;
    if (units) {
        TT_FATAL(
            t.x.device()->compute_with_storage_grid_size().y >= GU_Y + (rows + 7) / 8,
            "the builder cores need {} grid rows below the op's", (rows + 7) / 8);
        for (uint32_t b = 0; b < rows; ++b) {
            bld = bld.merge(CoreRangeSet(CoreRange({b % 8, GU_Y + b / 8}, {b % 8, GU_Y + b / 8})));
        }
    }
    // the packed units' 0/1 matmuls (the gather's and the scatter's: in0, SrcB, a 0/1 tile) at fidelity ONEHOT_FIDELITY
    // (HiFi2) from onehot_rows (as the builders') tile rows of tokens on: the phases on SrcB's low bits add zeros, the
    // same sums (compute/onehot_mm.hpp; compared bit for bit).  256 / 512 tokens 730.0 / 809.1 -> 674.3 / 759.1 us
    constexpr uint32_t ONEHOT_FIDELITY = 2;
    const uint32_t onehot_rows = build_rows;
    // the weights' matmuls at MOE_EXPERTS_FIDELITY (0 LoFi, 2 HiFi2, 3 HiFi3, 4 HiFi4, the default; 1: by the
    // weights' format, LoFi on BFP4 and HiFi4 else, for layers of mixed formats): a low-precision pattern.
    // Below HiFi4 the 0/1 matmuls always take onehot_mm.hpp's own fidelity (2: the same
    // sums), as the kernel's would then drop the copied values' low bits.
    auto by_format = [](uint32_t f, tt::DataFormat df) {
        return f != 1 ? f : (df == tt::DataFormat::Bfp4_b ? 0u : 4u);
    };
    const uint32_t fid_env = env("MOE_EXPERTS_FIDELITY", 4);
    const uint32_t fid = by_format(fid_env, wdf);
    const uint32_t fid_dn = by_format(env("MOE_EXPERTS_FIDELITY_DN", fid_env), wdf2);  // down's own
    TT_FATAL(fid == 0 || (fid >= 2 && fid <= 4), "MOE_EXPERTS_FIDELITY must be 0, 2, 3 or 4");
    TT_FATAL(fid_dn == 0 || (fid_dn >= 2 && fid_dn <= 4), "MOE_EXPERTS_FIDELITY_DN must be 0, 2, 3 or 4");
    uint32_t onehot_fid = units && onehot_rows != 0 && rows >= onehot_rows ? ONEHOT_FIDELITY : 4;
    if ((fid != 4 || fid_dn != 4) && onehot_fid == 4) {
        onehot_fid = 2;
    }
    // the gate | up and down cores' Dest sections cleared by MATH at acquire (ZEROACC) in place of the SFPU's zero
    // fills, with packed units from mclear_rows (as the builders') tile rows of tokens on: bit for bit; 256 /
    // 512 tokens 730.0 / 809.1 -> 657.1 / 762.3 us, with HiFi2 above 632.0 / 714.8
    const uint32_t mclear_rows = build_rows;
    const uint32_t mclear = units && mclear_rows != 0 && rows >= mclear_rows ? 1 : 0;
    // (x's valid semaphore, which the top core's x read sets locally)
    constexpr uint32_t SEM_XREAD = SEM_XVALID;
    // The in-op gather (attrs.h1_sem_x): the routing comes over the fabric into the column's landing
    // buffer `weights`, x straight into the grid columns' top gate | up cores' L1 (from 32 tokens a mesh row their
    // CB_XALL, below a compact landing CB they read into it), from each mesh row's t tokens (the sender core,
    // h1_send.cpp; this mesh row's own x the top cores' streams read) while the op starts.  Builder b waits for
    // h1_rcount increments of the routing semaphore (one from each mesh row with rows in its tile), each top core's
    // control kernel for h1_xcount of the x semaphore (each mesh row's K slice groups, one a tile row of x, end with an
    // increment on each link they use), then sets it back to 0.  Values bit for bit as without (the same x and
    // routing); MoE layer alone 64 / 128 / 256 / 512 tokens a column -27 / -37 / -39 / -47 us
    const bool h1 = attrs.h1_sem_x != 0;
    // The column's way back in the op (attrs.rs_sem, with the in-op gather at 32 / 64 tokens a mesh row):
    // in place of y [8 t, D] and ttnn.reduce_scatter over the mesh column after the op, the ring of
    // reduce_scatter_minimal_async run here, with its sums (the same bits): the down cores round
    // their rows slice by slice in the order the ring needs them (dn_compute.cpp) and write them into 16 reducer cores
    // (rs_move.cpp, the builder rows' cores: reducer j the tile columns 8 j .. 8 j + 7, chunk parity j % 2), which add
    // with a 16-bit Dst as the ring does (rs_compute.cpp: the down cores' Dst is fp32) and hand their blocks to 4 return
    // senders (rs_send.cpp, cores (8 .. 11, GU_Y): link 0 / 1 by way up / down), which send them over the fabric to the
    // same reducer of the ring neighbour once the in-op gather's sender has closed its connections (h1_send.cpp).  The
    // op returns this mesh row's [t, D].
    const bool rs = attrs.rs_sem != 0;
    // ... the down cores' rounded rows written into the reducers by their NCRISC (the W2 stream, done by then: NoC 0,
    // the y-increasing way, straight to the reducer rows below; the form before, dn_ctrl on BRISC and NoC 1, the
    // y-decreasing way around the grid column from the lower rows, whose rows came up to 24 us late at 64 tokens a row)
    const uint32_t h1_t = h1 ? t.x_row->logical_shape()[-2] : 0;
    const uint32_t h1_tr = h1_t >= 32 ? h1_t / 32 : 1;
    const uint32_t h1_rcount = !h1 ? 0 : h1_t >= 32 ? 1 : 32 / h1_t;
    // (a K slice group of a mesh row's x: 16 tiles' rows of its t tokens, in packets of up to h1_maxp bytes)
    const uint32_t h1_maxp = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() >= 8192 ? 8192 : 4096;
    const uint32_t h1_gbytes = 16 * (h1_t >= 32 ? 2048 : 64 * h1_t);
    const uint32_t h1_gpk = h1 ? h1_gbytes / std::min(h1_gbytes, h1_maxp) : 0;
    const uint32_t h1_xcount = !h1 ? 0 : 8 * h1_tr * std::min(h1_gpk, 2u);
    constexpr uint32_t CB_H1X = tt::CBIndex::c_8;  // in-op gather below 32 tokens a row: the top cores' compact landing
    if (h1) {
        TT_FATAL(units, "in-op gather: packed units");
        TT_FATAL(coord.has_value(), "in-op gather: the op's mesh coordinate");
        TT_FATAL(
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() >= 4096, "gather: 4 KB packets (a routing tile)");
    }
    // (The weight streams' first 2 blocks read before the records came, as if item 0 were experts 0 .. 7, and kept
    // when the units agreed: the same y, but slower: 256 / 512 tokens 611.6 / 634.2 -> 616.9 / 714.5 us.)
    // (A group's slots on a diagonal of the grid, so that one with few live slots reads on every grid row, made no
    // difference.)  A test switch for the times (the results are wrong): MOE_EXPERTS_STREAM_SKIP, bits 1 / 2 /
    // 4: the W1 stream / the W2 stream / the x sender read nothing from DRAM.
    const char* skip_env = std::getenv("MOE_EXPERTS_STREAM_SKIP");
    const uint32_t skip = skip_env ? static_cast<uint32_t>(std::stoul(skip_env)) : 0;
    // (By mesh column with one tile row, a last group of m <= 4 slots split over 8 / m rows of cores, each a part of
    // the K blocks, was slower: the time goes with the chosen experts already, ~9.5 us each, not the groups.)
    // The CBs on all the cores first, the two groups' own above them: tt-metal places a CB on all the cores above
    // the end of the group that last raised their common end, so after the gate | up CBs a later, longer down CB
    // was not seen and CB_SCR landed on CB_ACC (BF16 weights, packed units: the down cores' unit lists overwritten).
    // One row: the routing tile (room for two), the chosen experts' lists, the down stream's zero tiles, a slot's T
    // rows (moe_decode_experts_common.hpp's scr_*); packed units: the records and the two RISCs' unit lists, then the
    // zero tiles.
    const uint32_t scr_bytes = (units ? scr_recs(rows, attrs.local) : 2 * rows * 4096 + 2048) +
                               T * cols * std::max(tt::tile_size(wdf), tt::tile_size(wdf2));
    add_cb(desc, units ? all.merge(bld) : all, CB_SCR, (scr_bytes + 4095) / 4096, f32);
    // CB_H only on the down cores, placed right after CB_SCR (tt-metal places a CB at the end of its own cores' CBs):
    // the gate | up cores write into it at their CB_SCR's end (gu_ctrl.cpp), and keep its 384 KB (16 tile rows of
    // packed units did not fit in L1 with it)
    add_cb(desc, dn, CB_H, 3 * num_gu, bf16);
    // The weight rings: DEPTH blocks (DEPTH - 1 in flight); packed units' W1 blocks a K slice
    TT_FATAL(!units || W1_BLOCK == kd / T, "packed units: the W1 blocks ({} K tiles) must be a K slice", W1_BLOCK);
    // (a down ring 9 deep, a whole item's W2 read ahead with one row of tokens, was slower: 8 / 32 tokens 153.0 /
    // 367.7 -> 157.5 / 378.6 us)
    // (with builders, the down cores' W2 ring 4 / 6 / 8 blocks deep, to read further ahead of their compute, was
    // slower: 256 tokens 610.7 -> 616.4 / 626.7 / 636.4 us; and the streams' runs in requests of 4 / 8 / 16 tiles,
    // MOE_EXPERTS_READ_CHUNK: 256 / 512 tokens 610.7 / 634.5 -> 672.5 / 709.7, 614.5 / 650.5, 615.0 / 644.2)
    const uint32_t gu_depth = DEPTH, dn_depth = DEPTH;
    add_cb(desc, gu, CB_W, gu_depth * W1_BLOCK * 2, wdf);
    add_cb(desc, dn, CB_W, dn_depth * W2_BLOCK * cols, wdf2);
    if (units) {
        add_cb(desc, gu, CB_XALL, rows * W1_BLOCK, bf16);
        add_cb(desc, gu, CB_P, 2 * (rows + (prog ? 1u : gather_mask)), bf16);
        add_cb(desc, gu, CB_XS, W1_BLOCK, bf16);
        add_cb(desc, gu, CB_PART, 2 * T, f32);
        add_cb(desc, gu, CB_PIN, 2 * 2 * T, f32);
        add_cb(desc, dn, CB_P, 2 * ((num_gu / T) * rows + (prog ? 1u : scatter_mask)), tt::DataFormat::Bfp8_b);
        add_cb(desc, dn, CB_XS, scatter_slots * cols, f32);
    } else {
        add_cb(desc, gu, CB_X, kd, bf16);
    }
    add_cb(desc, gu, CB_WCOL, 2, bf16);  // (one h tile an item and gate | up core)
    add_cb(desc, gu, CB_HOUT, 2, bf16);
    add_cb(desc, dn, CB_Y, 2 * cols, bf16);
    if (units) {
        add_cb(desc, dn, CB_ACC, rows * cols, f32);
    }
    if (h1 && h1_t < 32) {  // (last: above every other CB of the top cores)
        const CoreRangeSet tops(CoreRange({GU_X0, 0}, {GU_X0 + GU_X - 1, 0}));
        add_cb(desc, tops, CB_H1X, 8 * h1_gbytes / 2048, bf16);
    }
    for (uint32_t id : {SEM_READY, SEM_XVALID, SEM_FREE, SEM_ARRIVE0, SEM_ARRIVE0 + 1, SEM_ARRIVE0 + 2, SEM_HVALID0,
                        SEM_HVALID0 + 1, SEM_HVALID0 + 2, SEM_DFREE, SEM_PARRIVE0, SEM_PARRIVE0 + 1, SEM_PFREE0,
                        SEM_PFREE0 + 1}) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = all, .initial_value = 0});
    }
    if (!units) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = SEM_ROUTE, .core_type = tt::CoreType::WORKER, .core_ranges = all, .initial_value = 0});
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = SEM_NOC1, .core_type = tt::CoreType::WORKER, .core_ranges = all, .initial_value = 0});
    if (units) {  // (the builders' semaphore id is SEM_ROUTE's)
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = SEM_REC, .core_type = tt::CoreType::WORKER, .core_ranges = all, .initial_value = 0});
        for (uint32_t id : {SEM_RDY, SEM_GO}) {
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = bld, .initial_value = 0});
        }
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = SEM_XDONE, .core_type = tt::CoreType::WORKER, .core_ranges = bld, .initial_value = 0});
    }

    auto* device = t.x.device();
    auto noc = [&](uint32_t x, uint32_t y) { return device->worker_core_from_logical_core(CoreCoord(x, y)); };
    // the in-op gather: the sender core, the grid's last (below the op's cores and the builders), and its semaphores:
    // the x groups read (BRISC -> NCRISC), the top cores in the program, their landing address (the mailbox)
    constexpr uint32_t SEM_H1 = 0, SEM_H1_TOPS = 1, SEM_H1_MAIL = 2;
    const CoreCoord h1_core(
        device->compute_with_storage_grid_size().x - 1, device->compute_with_storage_grid_size().y - 1);
    const CoreCoord h1_sender = noc(h1_core.x, h1_core.y);
    // multicast rectangles for the control kernels on NoC 1: from the bottom-right corner to the top-left
    const CoreCoord sender = noc(GU_X0, 0), gu0 = noc(GU_X0 + GU_X - 1, GU_Y - 1), gu1 = noc(GU_X0, 0);
    const CoreCoord dn0 = noc(DN_X0 + DN_X - 1, DN_Y - 1), dn1 = noc(DN_X0, 0);  // dn1: down core 0
    Buffer* routing = t.weights.buffer();
    const CoreCoord b0 = noc(0, GU_Y);  // builder 0
    const uint32_t limit_bits = std::bit_cast<uint32_t>(attrs.limit);
    const uint32_t neg_limit_bits = std::bit_cast<uint32_t>(-attrs.limit);

    // gate | up cores
    std::vector<uint32_t> ct = {CB_X, CB_WCOL, CB_HOUT, CB_H, CB_SCR, attrs.local, kd, sender.x, sender.y,
                                gu0.x, gu0.y, gu1.x, gu1.y, num_gu, num_dn, num_gu,
                                SEM_READY, SEM_XVALID, SEM_FREE, SEM_ARRIVE0, GU_X0,
                                T, rows, units ? 1u : 0u, CB_XALL, CB_P, CB_PART, CB_PIN, SEM_PARRIVE0, SEM_PFREE0,
                                (skip & 4) != 0, SEM_NOC1, BOTH_GU, SEM_ROUTE, dn0.x, dn0.y, dn1.x, dn1.y, gather_mask,
                                b0.x, b0.y, SEM_RDY, SEM_REC, SEM_XREAD, SEM_GO1,
                                SEM_XDONE, attrs.h1_sem_x, h1_xcount, SEM_H1_TOPS, SEM_H1_MAIL, h1_t, CB_H1X};
    TensorAccessorArgs(routing).append_to(ct);
    TensorAccessorArgs(t.x.buffer()).append_to(ct);
    auto gu_ctrl = dm_kernel("gu_ctrl.cpp", gu, std::move(ct), DataMovementProcessor::RISCV_0);
    // packed units: a unit of the same expert as the one before takes its weights from L1 (stream_active.cpp's
    // dedupe: 1 the down cores' W2 from their own ring, 2 the gate | up cores' W1 from the core above, at the same ring
    // address; the same bytes, so the same values).  The gate | up cores' two semaphores for it: SEM_HVALID0 and the
    // next, which only the down cores use otherwise.  Real routing, one MoE layer, the experts op alone: 512
    // tokens a column (128 users' T = 4) 744-813 -> 697-739 us with 3 (-27 to -88), with 1 (down only) -19 to -41;
    // 256 (MTP K = 1) -5 to -15 us; the random router's (no repeated expert) the same; the partial sums bit for bit
    // (stress-tested).  3: an expert's later units take its weights from the earlier unit's, down and gate | up
    const uint32_t dedupe = units ? 3 : 0;
    constexpr uint32_t SEM_DD = SEM_HVALID0;
    ct = {CB_W, CB_SCR, 2, kd, attrs.local, W1_BLOCK, gu_depth, 1u, rows, DEPTH - 1, T, units ? 1u : 0u,
          (skip & 1) != 0, BOTH_GU, SEM_NOC1, SEM_ROUTE, SEM_REC, units ? 1u : 0u, CB_XALL, SEM_XREAD,
          (skip & 4) != 0, SEM_GO1, attrs.h1_sem_x, h1_t, h1_t >= 32 ? CB_XALL : CB_H1X, h1_tr * std::min(h1_gpk, 2u)};
    TensorAccessorArgs(t.w1.buffer()).append_to(ct);
    TensorAccessorArgs(t.x.buffer()).append_to(ct);
    TensorAccessorArgs(h1 ? t.x_row->buffer() : t.x.buffer()).append_to(ct);
    ct.push_back(dedupe);  // (after the accessors' args)
    ct.push_back(SEM_DD);
    for (uint32_t v : {0u, 0u, 0u, 0u}) {  // (the down streams' rounded rows: not here)
        ct.push_back(v);
    }
    auto gu_stream = dm_kernel("stream_active.cpp", gu, std::move(ct), DataMovementProcessor::RISCV_1);
    auto gu_compute = compute_kernel(
        "gu_compute.cpp",
        gu,
        {kd, W1_BLOCK, limit_bits, neg_limit_bits, rows, units ? 1u : 0u, T, gather_mask, onehot_fid, mclear},
        units,
        units ? std::optional<uint32_t>(CB_PIN) : std::nullopt,
        static_cast<MathFidelity>(fid));
    std::vector<uint32_t> dn_noc;  // the down cores' NoC coordinates, q = 0 .. 31, for the h tiles' unicasts
    for (uint32_t q = 0; q < num_dn; ++q) {
        const CoreCoord c = noc(DN_X0 + q % DN_X, q / DN_X);
        dn_noc.push_back(c.x);
        dn_noc.push_back(c.y);
    }
    std::vector<uint32_t> gu_noc;  // the gate | up cores' (p = 0 .. 63), for the go semaphore's unicasts
    for (uint32_t p = 0; p < num_gu; ++p) {
        const CoreCoord c = noc(GU_X0 + p % GU_X, p / GU_X);
        gu_noc.push_back(c.x);
        gu_noc.push_back(c.y);
    }
    for_each_core(gu, [&](const CoreCoord& core) {
        const uint32_t p = core.y * GU_X + core.x - GU_X0;  // the core's column tile (sliced: slot p / T, column p % T)
        auto rt = args(routing, t.x.buffer());
        for (uint32_t v : dn_noc) {
            rt.push_back(v);
        }
        rt.push_back(p);
        // packed units: the NoC coordinates of the row's cores (the grid column's 8 cores all read bank x: on a
        // diagonal, 8 banks each, 128 tokens took 775 against 702 us)
        for (uint32_t c = 0; c < GU_X; ++c) {
            const CoreCoord peer = noc(GU_X0 + c, core.y);
            rt.push_back(peer.x);
            rt.push_back(peer.y);
        }
        // packed units: the grid column's multicast rectangle on NoC 1 (from its bottom core to its top one)
        const CoreCoord bottom = noc(core.x, GU_Y - 1), top = noc(core.x, 0);
        for (uint32_t v : {bottom.x, bottom.y, top.x, top.y}) {
            rt.push_back(v);
        }
        if (!units) {  // the x sender raises every core's go semaphore (runtime args 87 ..)
            for (uint32_t v : gu_noc) {
                rt.push_back(v);
            }
        }
        if (h1) {  // the in-op gather, packed units: the sender core (runtime args 87, 88)
            rt.push_back(static_cast<uint32_t>(h1_sender.x));
            rt.push_back(static_cast<uint32_t>(h1_sender.y));
        }
        gu_ctrl.emplace_runtime_args(core, rt);
        // the first W1 block (one tile row: the K blocks of a column in the 8 banks, from the grid column's)
        const uint32_t rot = !units ? p % (kd / W1_BLOCK) : 0;
        KernelDescriptor::RTArgList stream_rt;
        stream_rt.push_back(t.w1.buffer());
        stream_rt.push_back(p);
        stream_rt.push_back(rot);
        if (units) {  // packed units: x (runtime arg 3)
            stream_rt.push_back(t.x.buffer());
        }
        if (h1) {  // the in-op gather: this mesh row's x and its index (runtime args 4, 5)
            stream_rt.push_back(t.x_row->buffer());
            stream_rt.push_back(coord.value()[0]);
        }
        for (uint32_t i = 3 + (units ? 1 : 0) + (h1 ? 2 : 0); i < 6; ++i) {  // (dedupe's runtime args at 6, 7)
            stream_rt.push_back(0u);
        }
        // dedupe: the cores above and below in the grid column (the slots before and after), NoC x | y << 16
        const auto packed_xy = [&](uint32_t y) {
            if (y >= GU_Y) {
                return ~0u;
            }
            const CoreCoord c = noc(core.x, y);
            return static_cast<uint32_t>(c.x) | static_cast<uint32_t>(c.y) << 16;
        };
        stream_rt.push_back(core.y > 0 ? packed_xy(core.y - 1) : ~0u);
        stream_rt.push_back(packed_xy(core.y + 1));
        gu_stream.emplace_runtime_args(core, stream_rt);
        KernelDescriptor::RTArgList compute_rt;
        compute_rt.push_back(rot);
        gu_compute.emplace_runtime_args(core, compute_rt);
    });

    // down cores.  y by mesh row (row_tokens) with packed units: the zeros of its padded layout written first, while
    // the records come, and at the end only the tokens' rows (dn_ctrl.cpp; the same bytes)
    // ... and with packed units (slots per scatter pass 1), the down cores' compute leaves out the last item's slots
    // past the last unit (zero W2, no tokens: the same sums; the last item's down work is the op's tail)
    const uint32_t dn_live = units && scatter_slots == 1 ? 1 : 0;
    ct = {CB_H, CB_Y, CB_SCR, attrs.local, cols, sender.x, sender.y, num_gu,
          SEM_READY, SEM_FREE, SEM_ARRIVE0, d_tiles, T, rows,
          SEM_HVALID0, SEM_DFREE, num_dn, units ? 1u : 0u, CB_P, SEM_NOC1, BOTH_DN,
          SEM_ROUTE, attrs.row_tokens, scatter_slots, scatter_mask, b0.x, b0.y, SEM_RDY, SEM_REC, dn_live};
    TensorAccessorArgs(output.buffer()).append_to(ct);
    ct.push_back(rs ? 1u : 0u);  // (after the accessor)
    auto dn_ctrl = dm_kernel("dn_ctrl.cpp", dn, std::move(ct), DataMovementProcessor::RISCV_0);
    ct = {CB_W, CB_SCR, cols, ki, attrs.local, W2_BLOCK, dn_depth, 2u, rows, DEPTH - 1, T, units ? 1u : 0u,
          (skip & 2) != 0, BOTH_DN, SEM_NOC1, SEM_ROUTE, SEM_REC, 0u, CB_XALL, SEM_XREAD, 0u, SEM_GO1, 0u, 0u, 0u, 0u};
    TensorAccessorArgs(t.w2.buffer()).append_to(ct);
    TensorAccessorArgs(t.x.buffer()).append_to(ct);
    TensorAccessorArgs(t.x.buffer()).append_to(ct);
    ct.push_back(dedupe);
    ct.push_back(SEM_DD);
    for (uint32_t v : {rs ? 1u : 0u, rs ? h1_t / 32 : 0u, static_cast<uint32_t>(CB_Y), cols}) {
        ct.push_back(v);
    }
    auto dn_stream = dm_kernel("stream_active.cpp", dn, std::move(ct), DataMovementProcessor::RISCV_1);
    auto dn_compute =
        compute_kernel(
            "dn_compute.cpp", dn,
            {ki, W2_BLOCK, cols, rows, units ? 1u : 0u, T, scatter_slots, scatter_mask, onehot_fid, mclear, dn_live,
             units ? env("MOE_EXPERTS_FIN_PACK", 0) : 0u, rs ? 1u : 0u, rs ? h1_t / 32 : 0u},
            false, CB_ACC, static_cast<MathFidelity>(fid_dn));
    for_each_core(dn, [&](const CoreCoord& core) {
        const uint32_t q = core.y * DN_X + core.x - DN_X0;  // the core's column block
        auto ctrl_rt = args(routing, output.buffer());
        ctrl_rt.push_back(q);
        // the grid row's gate | up cores, then the grid column's down cores (rows 0 .. 7): runtime args 3 .. 34
        for (uint32_t c = 0; c < GU_X; ++c) {
            const CoreCoord g = noc(GU_X0 + c, core.y);
            ctrl_rt.push_back(g.x);
            ctrl_rt.push_back(g.y);
        }
        for (uint32_t r = 0; r < DN_Y; ++r) {
            const CoreCoord d = noc(core.x, r);
            ctrl_rt.push_back(d.x);
            ctrl_rt.push_back(d.y);
        }
        if (rs) {  // the ring index
            dn_compute.emplace_runtime_args(core, {static_cast<uint32_t>(coord.value()[0])});
        }
        dn_ctrl.emplace_runtime_args(core, ctrl_rt);
        KernelDescriptor::RTArgList stream_rt;
        stream_rt.push_back(t.w2.buffer());
        stream_rt.push_back(q);
        stream_rt.push_back(0u);
        for (uint32_t i = 3; i < 8; ++i) {  // (dedupe's runtime args 6, 7: none here)
            stream_rt.push_back(~0u);
        }
        if (rs) {  // runtime args 8 .. 10: reducer q / 2's NoC x, y and the mailbox (rs_mail)
            const CoreCoord rc = noc((q / 2) % 8, GU_Y + q / 16);
            stream_rt.push_back(static_cast<uint32_t>(rc.x));
            stream_rt.push_back(static_cast<uint32_t>(rc.y));
            stream_rt.push_back(attrs.rs_mail);
        }
        dn_stream.emplace_runtime_args(core, stream_rt);
    });

    auto prog_defines = [&](KernelDescriptor& k) {
        k.defines.emplace_back("MOE_PROG", "1");
        k.defines.emplace_back("MOE_PROG_F_MILLI", std::to_string(prog_milli));
    };
    for (auto* k : {&gu_ctrl, &gu_stream, &gu_compute, &dn_ctrl, &dn_stream, &dn_compute}) {
        if (prog) {
            prog_defines(*k);
        }
        desc.kernels.push_back(std::move(*k));
    }
    if (units) {
        ct = {attrs.local, rows, CB_SCR, SEM_RDY, SEM_GO, SEM_REC, num_gu + num_dn, gu0.x, gu0.y, gu1.x, gu1.y,
              dn0.x, dn0.y, dn1.x, dn1.y, num_gu, num_dn, SEM_XDONE, GU_X, SEM_GO1, attrs.h1_sem_r, h1_rcount};
        TensorAccessorArgs(routing).append_to(ct);
        auto builder = dm_kernel("route_build.cpp", bld, std::move(ct), DataMovementProcessor::RISCV_0);
        if (prog) {
            prog_defines(builder);
            if (prog_rows_trow != 0) {
                builder.defines.emplace_back("MOE_PROG_ROWS", "1");
                builder.defines.emplace_back("MOE_PROG_ROWS_TROW", std::to_string(prog_rows_trow));
                builder.defines.emplace_back("MOE_PROG_ROWS_U", std::to_string(prog_rows_u));
                builder.defines.emplace_back("MOE_PROG_ROWS_Q", prog_rows_q);
            }
        }
        for (uint32_t b = 0; b < rows; ++b) {
            KernelDescriptor::RTArgList rt;
            rt.push_back(routing);
            rt.push_back(b);
            rt.push_back(b0.x);
            rt.push_back(b0.y);
            if (b == 0) {  // every gate | up and down core, for the go semaphore
                for (uint32_t v : gu_noc) {
                    rt.push_back(v);
                }
                for (uint32_t v : dn_noc) {
                    rt.push_back(v);
                }
            }
            builder.emplace_runtime_args(CoreCoord(b % 8, GU_Y + b / 8), rt);
        }
        desc.kernels.push_back(std::move(builder));
    }
    if (h1) {  // the sender core
        const CoreCoord sc = h1_core;
        const CoreRangeSet scs(CoreRange(sc, sc));
        constexpr uint32_t CB_H1 = tt::CBIndex::c_8;  // the stage: 2 routing tiles, then this row's x pieces
        const uint32_t pb = h1_t >= 32 ? 2048 : h1_t * 64;
        const uint32_t stage = (2 * 4096 + h1_tr * kd * pb + 4095) / 4096 * 4096;
        desc.cbs.push_back(CBDescriptor{
            .total_size = stage,
            .core_ranges = scs,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(CB_H1), .data_format = bf16, .page_size = 4096}}},
        });
        for (uint32_t id : {SEM_H1, SEM_H1_TOPS, SEM_H1_MAIL}) {
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = scs, .initial_value = 0});
        }
        const ttnn::MeshCoordinate& c = coord.value();
        const uint32_t mr = device->shape()[0], r = c[0];
        const auto node = device->get_fabric_node_id(c);
        const auto fnode = device->get_fabric_node_id(ttnn::MeshCoordinate((r + 1) % mr, c[1]));
        const auto bnode = device->get_fabric_node_id(ttnn::MeshCoordinate((r + mr - 1) % mr, c[1]));
        TT_FATAL(mr == 8, "in-op gather: a mesh column of 8 chips");
        for (uint32_t role = 0; role < 2; ++role) {  // BRISC on NoC 0 and link 0, NCRISC on NoC 1 and link 1
            std::vector<uint32_t> sct = {role, h1_t, kd, kd / T, CB_H1, SEM_H1, SEM_H1_TOPS, SEM_H1_MAIL, h1_maxp};
            TensorAccessorArgs(t.x_row->buffer()).append_to(sct);
            TensorAccessorArgs(routing).append_to(sct);
            sct.push_back(rs ? 1u : 0u);  // (h1_send.cpp: the return senders' go)
            KernelDescriptor k;
            k.kernel_source = KDIR + "dataflow/h1_send.cpp";
            k.source_type = KernelDescriptor::SourceType::FILE_PATH;
            k.core_ranges = scs;
            k.compile_time_args = std::move(sct);
            k.config = DataMovementConfigDescriptor{
                .processor = role == 0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1,
                .noc = role == 0 ? NOC::NOC_0 : NOC::NOC_1};
            KernelDescriptor::RTArgList rt;
            rt.push_back(t.x_row->buffer());
            rt.push_back(routing);
            for (uint32_t v : {r, static_cast<uint32_t>(*node.mesh_id), static_cast<uint32_t>(fnode.chip_id),
                               static_cast<uint32_t>(bnode.chip_id), attrs.h1_sem_x, attrs.h1_sem_r,
                               static_cast<uint32_t>(h1_sender.x), static_cast<uint32_t>(h1_sender.y)}) {
                rt.push_back(v);
            }
            for (uint32_t j = 0; j < GU_X; ++j) {  // the top gate | up cores
                const CoreCoord top = noc(GU_X0 + j, 0);
                rt.push_back(static_cast<uint32_t>(top.x));
                rt.push_back(static_cast<uint32_t>(top.y));
            }
            for (uint32_t b = 0; b < 16; ++b) {  // the builders (tile row b)
                const CoreCoord bc = b < rows ? noc(b % 8, GU_Y + b / 8) : CoreCoord(0, 0);
                rt.push_back(static_cast<uint32_t>(bc.x));
                rt.push_back(static_cast<uint32_t>(bc.y));
            }
            std::vector<uint32_t> fabric;
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(node, fnode, role, desc, sc, fabric);
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(node, bnode, role, desc, sc, fabric);
            rt.append(fabric);
            if (rs) {  // the return senders of link `role` (up, down) and their mailbox
                for (uint32_t w = 0; w < 2; ++w) {
                    const CoreCoord rsc = noc(8 + 2 * role + w, GU_Y);
                    rt.push_back(static_cast<uint32_t>(rsc.x));
                    rt.push_back(static_cast<uint32_t>(rsc.y));
                }
                rt.push_back(attrs.rs_mail);
            }
            k.emplace_runtime_args(sc, rt);
            desc.kernels.push_back(std::move(k));
        }
    }
    if (rs) {
        TT_FATAL(h1 && units && cols == 4 && (h1_t == 32 || h1_t == 64), "the way back: in-op gather, 32 / 64 tokens");
        const uint32_t R = h1_t / 32, B = 8 * R;  // a block: a slice's tile rows by a reducer's 8 tile columns
        constexpr uint32_t REDUCERS = 16, CB_OWN = 9, CB_LO = 10, CB_HI = 11, CB_SUM = 12, CB_OUT = 21, CB_SIG = 22,
                           CB_FLAGS = 23, CB_STAGE = 24;
        auto red = [&](uint32_t j) { return CoreCoord(j % 8, GU_Y + j / 8); };
        auto snd = [&](uint32_t l, uint32_t down) { return CoreCoord(8 + 2 * l + down, GU_Y); };
        std::vector<CoreRange> rr, sr;
        for (uint32_t j = 0; j < REDUCERS; ++j) {
            rr.emplace_back(red(j), red(j));
        }
        for (uint32_t s4 = 0; s4 < 4; ++s4) {
            sr.emplace_back(snd(s4 / 2, s4 % 2), snd(s4 / 2, s4 % 2));
        }
        const CoreRangeSet reds(rr), snds(sr), both = reds.merge(snds);
        auto cb = [&](const CoreRangeSet& cores, uint32_t index, uint32_t bytes, uint32_t page) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = bytes,
                .core_ranges = cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(index), .data_format = bf16, .page_size = page}}},
            });
        };
        // (on the reducers and the senders alike, so that a sender finds a reducer's blocks at its own addresses)
        cb(both, CB_OWN, 8 * B * 2048, 2048);
        cb(both, CB_LO, 4 * B * 2048, 2048);
        cb(both, CB_HI, 4 * B * 2048, 2048);
        cb(both, CB_SUM, 5 * B * 2048, 2048);
        cb(both, CB_FLAGS, 16 * 8, 16 * 8);
        cb(reds, CB_OUT, B * 2048, 2048);
        cb(reds, CB_SIG, 8 * 32, 32);
        cb(snds, CB_STAGE, 2 * B * 2048, 2048);
        const ttnn::MeshCoordinate& c = coord.value();
        const uint32_t mr = device->shape()[0], r = c[0];
        const auto node = device->get_fabric_node_id(c);
        const auto fnode = device->get_fabric_node_id(ttnn::MeshCoordinate((r + 1) % mr, c[1]));
        const auto bnode = device->get_fabric_node_id(ttnn::MeshCoordinate((r + mr - 1) % mr, c[1]));
        // the reducers: data movement (NCRISC, NoC 0) and the adds (a 16-bit Dst)
        std::vector<uint32_t> mct = {CB_OWN, CB_LO, CB_HI, CB_SUM, CB_OUT, CB_SIG, CB_FLAGS, R, d_tiles};
        TensorAccessorArgs(output.buffer()).append_to(mct);
        KernelDescriptor mv;
        mv.kernel_source = KDIR + "dataflow/rs_move.cpp";
        mv.source_type = KernelDescriptor::SourceType::FILE_PATH;
        mv.core_ranges = reds;
        mv.compile_time_args = std::move(mct);
        mv.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0};
        KernelDescriptor cp;
        cp.kernel_source = KDIR + "compute/rs_compute.cpp";
        cp.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cp.core_ranges = reds;
        cp.compile_time_args = {B, 4, CB_OWN, CB_SIG};
        cp.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,  // (the ring's adds: a 16-bit Dst)
            .dst_full_sync_en = false,
            .unpack_to_dest_mode = std::vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default),
        };
        // a link's reducers (j / 2 even: link 0) in order: their slot in the senders' flags
        std::vector<uint32_t> link_of(REDUCERS), slot(REDUCERS), count(2, 0);
        for (uint32_t j = 0; j < REDUCERS; ++j) {
            link_of[j] = (j / 2) % 2;
            slot[j] = count[link_of[j]]++;
        }
        auto op = [](uint32_t kind, uint32_t out, uint32_t ca, uint32_t ka, uint32_t cb2, uint32_t kb, uint32_t cd = 0,
                     uint32_t kd2 = 0) {
            return std::vector<uint32_t>{kind | out << 8, ca << 8 | ka, cb2 << 8 | kb, cd << 8 | kd2};
        };
        for (uint32_t j = 0; j < REDUCERS; ++j) {
            const bool even = j % 2 == 0;
            const uint32_t l = link_of[j];
            const CoreCoord d0 = noc((2 * j) % DN_X, (2 * j) / DN_X), d1 = noc((2 * j + 1) % DN_X, (2 * j + 1) / DN_X);
            const CoreCoord up = noc(snd(l, 0).x, snd(l, 0).y), dw = noc(snd(l, 1).x, snd(l, 1).y);
            KernelDescriptor::RTArgList mrt;
            mrt.push_back(output.buffer());
            for (uint32_t v : {j, even ? 0u : 1u, attrs.rs_sem, attrs.rs_mail, static_cast<uint32_t>(d0.x),
                               static_cast<uint32_t>(d0.y), static_cast<uint32_t>(d1.x), static_cast<uint32_t>(d1.y),
                               static_cast<uint32_t>(up.x), static_cast<uint32_t>(up.y), static_cast<uint32_t>(dw.x),
                               static_cast<uint32_t>(dw.y), slot[j]}) {
                mrt.push_back(v);
            }
            mv.emplace_runtime_args(red(j), mrt);
            // own blocks: slices c + 4, c + 3, c + 5, c + 2, c + 6, c + 1, c + 7, c (dn_compute.cpp); the ops as the
            // ring's chains, with the chains' links from the neighbours (rs_move.cpp)
            std::vector<uint32_t> ops;
            auto add = [&](std::vector<uint32_t> v) { ops.insert(ops.end(), v.begin(), v.end()); };
            if (even) {
                add(op(0, CB_SUM, CB_LO, 0, CB_OWN, 1));
                add(op(0, CB_SUM, CB_LO, 1, CB_OWN, 3));
                add(op(0, CB_SUM, CB_HI, 0, CB_OWN, 4));
                add(op(0, CB_SUM, CB_LO, 2, CB_OWN, 5));
                add(op(0, CB_SUM, CB_HI, 1, CB_OWN, 6));
                add(op(1, CB_OUT, CB_LO, 3, CB_OWN, 7, CB_HI, 2));
            } else {
                add(op(0, CB_SUM, CB_HI, 0, CB_OWN, 2));
                add(op(0, CB_SUM, CB_HI, 1, CB_OWN, 4));
                add(op(0, CB_SUM, CB_LO, 0, CB_OWN, 3));
                add(op(0, CB_SUM, CB_HI, 2, CB_OWN, 6));
                add(op(0, CB_SUM, CB_LO, 1, CB_OWN, 5));
                add(op(1, CB_OUT, CB_LO, 2, CB_OWN, 7, CB_HI, 3));
            }
            KernelDescriptor::RTArgList crt;
            crt.push_back(6u);
            for (uint32_t v : ops) {
                crt.push_back(v);
            }
            for (uint32_t v : {8u, even ? 4u : 3u, even ? 3u : 4u, 5u}) {  // pops: own, lo, hi, sums
                crt.push_back(v);
            }
            cp.emplace_runtime_args(red(j), crt);
        }
        desc.kernels.push_back(std::move(mv));
        desc.kernels.push_back(std::move(cp));
        // the return senders (BRISC; NoC 0 for link 0, NoC 1 for link 1)
        const uint32_t maxp = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() >= 8192 ? 8192 : 4096;
        for (uint32_t l = 0; l < 2; ++l) {
            for (uint32_t down = 0; down < 2; ++down) {
                const CoreCoord sc = snd(l, down);
                KernelDescriptor k;
                k.kernel_source = KDIR + "dataflow/rs_send.cpp";
                k.source_type = KernelDescriptor::SourceType::FILE_PATH;
                k.core_ranges = CoreRangeSet(CoreRange(sc, sc));
                k.compile_time_args = {CB_FLAGS, CB_STAGE, CB_OWN, CB_SUM, CB_LO, CB_HI, R, maxp};
                k.config = DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_0, .noc = l == 0 ? NOC::NOC_0 : NOC::NOC_1};
                const auto dst = down ? bnode : fnode;
                KernelDescriptor::RTArgList rt;
                for (uint32_t v : {down, static_cast<uint32_t>(*node.mesh_id), static_cast<uint32_t>(dst.chip_id),
                                   attrs.rs_sem, attrs.rs_mail}) {
                    rt.push_back(v);
                }
                for (uint32_t j = 0; j < REDUCERS; ++j) {
                    if (link_of[j] == l) {
                        const CoreCoord rc = noc(red(j).x, red(j).y);
                        rt.push_back(static_cast<uint32_t>(rc.x));
                        rt.push_back(static_cast<uint32_t>(rc.y));
                        rt.push_back(j % 2);
                    }
                }
                std::vector<uint32_t> fabric;
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(node, dst, l, desc, sc, fabric);
                rt.append(fabric);
                k.emplace_runtime_args(sc, rt);
                desc.kernels.push_back(std::move(k));
            }
        }
    }
    return desc;
}
}  // namespace

ProgramDescriptor MoeDecodeExpertsProgramFactory::create_descriptor(
    const MoeDecodeExpertsParams& attrs,
    const MoeDecodeExpertsTensorArgs& t,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const uint32_t d = t.x.logical_shape()[-1];
    const uint32_t kd = d / tt::constants::TILE_WIDTH;           // gate | up K tiles
    // down K tiles per item: T for each of a group's 64 / T experts
    const uint32_t ki = GU_X * GU_Y;
    const uint32_t cols = d / tt::constants::TILE_WIDTH / (DN_X * DN_Y);
    const auto wdf = datatype_to_dataformat_converter(t.w1.dtype());
    const auto wdf2 = datatype_to_dataformat_converter(t.w2.dtype());
    const CoreRangeSet gu(CoreRange({GU_X0, 0}, {GU_X0 + GU_X - 1, GU_Y - 1}));
    const CoreRangeSet dn(CoreRange({DN_X0, 0}, {DN_X0 + DN_X - 1, DN_Y - 1}));

    return experts_descriptor(attrs, t, output, gu, dn, kd, ki, cols, wdf, wdf2, mesh_dispatch_coordinate);
}

}  // namespace ttnn::experimental::prim
