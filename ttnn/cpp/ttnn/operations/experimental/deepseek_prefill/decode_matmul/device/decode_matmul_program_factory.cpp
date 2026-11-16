// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_matmul_program_factory.hpp"
#include "decode_matmul_device_operation_types.hpp"
#include "decode_matmul_blocks.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/allocator.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
// Per core (slice s, block b): NCRISC streams the weight block (K tiles k0 .. k0 + kc - 1 of slice s, kc = K / 32 / ks
// or one more, N tiles n0 .. n0 + ncb - 1 of block b) into CB_W in chunks of cr K rows x sb columns, several in flight;
// BRISC brings x's slice (read from DRAM and multicast to the rectangle by the rectangle's core 0, the others wait on
// SEM_X), sends this core's fp32 partials to the block's owners (output tile t of the block belongs to the core of
// slice t mod ks) and counts the arrivals of its own (SEM_RED), then writes its output tiles; the compute makes the
// partials (matmul_block over the K slice, fp32 in Dest) and, on an owner, adds the ks slices' partials of each owned
// tile in fp32 in the order of K (unpacked to Dest, SFPU adds).  With one slice the compute packs the output directly.
constexpr uint32_t CB_X = tt::CBIndex::c_0;      // x's K slice (bf16, kc tiles; the same address on every core)
constexpr uint32_t CB_W = tt::CBIndex::c_1;      // the weight chunks (a ring of `depth` chunks)
constexpr uint32_t CB_RECV = tt::CBIndex::c_2;   // an owner's received partials (fp32; slot i, slice s at i ks + s)
constexpr uint32_t CB_SCR = tt::CBIndex::c_3;    // the x sender's semaphore words (one per chunk)
constexpr uint32_t CB_PART = tt::CBIndex::c_16;  // this core's partials (fp32, ncb tiles)
constexpr uint32_t CB_OUT = tt::CBIndex::c_17;   // output tiles (the output dtype)
constexpr uint32_t SEM_X = 0, SEM_RED = 1, SEM_NOC1 = 3;
// with the weight stream on both NoCs: the chunks in flight until NoC 1 is handed over (1 against `ahead`
// throughout: decode steps a further -0.05 to -0.15 ms)
constexpr uint32_t AHEAD0 = 1;
const std::string KDIR = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/decode_matmul/device/kernels/";

uint32_t dm_div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// the matmuls' fidelity: DECODE_MATMUL_FIDELITY (0 LoFi, 2 HiFi2, 3 HiFi3, 4 HiFi4, the default; 1: LoFi on BFP4
// weights, HiFi4 else), a low-precision pattern; the partials' sums are SFPU adds of fp32
// Dest tiles, not affected
MathFidelity matmul_fidelity(tt::DataFormat wdf) {
    const char* v = std::getenv("DECODE_MATMUL_FIDELITY");
    uint32_t f = v ? static_cast<uint32_t>(std::stoul(v)) : 4;
    if (f == 1) {
        f = wdf == tt::DataFormat::Bfp4_b ? 0 : 4;
    }
    TT_FATAL(f == 0 || (f >= 2 && f <= 4), "DECODE_MATMUL_FIDELITY must be 0, 2, 3 or 4");
    return static_cast<MathFidelity>(f);
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
    const std::string& name, const CoreRangeSet& cores, std::vector<uint32_t> ct, DataMovementProcessor processor) {
    KernelDescriptor k;
    k.kernel_source = KDIR + "dataflow/" + name;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    // the weight stream (RISCV_1) reads DRAM on NoC 0 (on NoC 1 the experts op's streams ran at about half the rate);
    // the control kernel (RISCV_0: x, the multicasts, the partials, the output) is on NoC 1
    k.config = DataMovementConfigDescriptor{
        .processor = processor, .noc = processor == DataMovementProcessor::RISCV_1 ? NOC::NOC_0 : NOC::NOC_1};
    return k;
}

}  // namespace

ProgramDescriptor DecodeMatmulProgramFactory::create_descriptor(
    const DecodeMatmulParams& attrs,
    const DecodeMatmulTensorArgs& t,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    Buffer* x_buf = t.x.buffer();
    Buffer* w_buf = t.w.buffer();
    Buffer* out_buf = output.buffer();
    auto* device = t.x.device();
    const auto grid = device->compute_with_storage_grid_size();

    const uint32_t kt = t.x.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t nt = t.w.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t ks = attrs.k_split, nc = attrs.n_per_core, cw = attrs.rect_w, rh = attrs.rect_h;
    const uint32_t banks = device->allocator()->get_num_banks(BufferType::DRAM);
    const auto blocks = decode_matmul_blocks(nt, banks, nc, attrs.strided != 0);
    const uint32_t stride = attrs.strided ? banks : 1;
    TT_FATAL(banks >= 1 && banks <= 16, "decode_matmul: {} DRAM banks (at most 16)", banks);
    TT_FATAL(
        x_buf->aligned_page_size() == x_buf->page_size() && w_buf->aligned_page_size() == w_buf->page_size(),
        "decode_matmul: the tiles must be DRAM-aligned (a bank's pages adjacent)");
    // slice s: K tiles s kt / ks .. (s + 1) kt / ks - 1 (ks need not divide kt: kc_max or kc_max - 1 tiles)
    // x's tile rows (two for 33-64 rows of tokens: each core's x slice k-major, tile (r, k) at k mt + r, so that a
    // chunk of K is one run for the multicast and matmul_block takes the mt rows with an in0 row stride of 1)
    const uint32_t mt = t.x.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    TT_FATAL(mt == 1 || (mt == 2 && ks == 1), "decode_matmul: two tile rows of x need K whole a core");
    const uint32_t kc_max = dm_div_up(kt, ks), nb = blocks.size(), rect = cw * rh;
    auto k_first = [&](uint32_t s) { return s * kt / ks; };
    const uint32_t rects_x = grid.x / cw;
    TT_FATAL(nb <= rect && rects_x * (grid.y / rh) >= ks, "decode_matmul: the plan does not fit the grid");
    auto origin = [&](uint32_t s) { return CoreCoord{(s % rects_x) * cw, (s / rects_x) * rh}; };
    auto core_of = [&](uint32_t s, uint32_t p) {
        const CoreCoord o = origin(s);
        return CoreCoord{o.x + p % cw, o.y + p / cw};
    };
    std::vector<CoreRange> ranges;
    for (uint32_t s = 0; s < ks; ++s) {
        const CoreCoord o = origin(s);
        ranges.emplace_back(o, CoreCoord{o.x + cw - 1, o.y + rh - 1});
    }
    const CoreRangeSet all(ranges);

    // N sub-blocks of at most 8 tiles (Dest holds 8 fp32 tiles in full sync); chunks of cr K rows x sb tiles (a slice's
    // last chunk may hold fewer rows), about 8 tiles each, `ahead` of them in flight (env overrides for tests)
    const uint32_t nsub = dm_div_up(mt * nc, 8), sb = dm_div_up(nc, nsub);
    const uint32_t chunk_target = attrs.chunk ? attrs.chunk : 8;
    const uint32_t cr = std::min(kc_max, std::max(1u, chunk_target / sb));
    const uint32_t chunks = nsub * dm_div_up(kc_max, cr);
    // 2 chunks in flight a core (1 until NoC 1 is handed over, AHEAD0): with 6 the banks' and
    // NoCs' queues were deep and the cores of a bank came out uneven (DSA o_proj, the weight stream's slowest / mean
    // core 43.6 / 29.5 us), the op paced by the slowest; with 2, 41.1 / 33.1 us: the 5 decode shapes 2.5-4.6 us
    // faster at x of 32 rows, about as fast as NoC 1 from the start (unsafe), the decode step -0.3 to -0.4 ms
    const uint32_t ahead = std::min(attrs.ahead ? attrs.ahead : 2u, chunks);
    TT_FATAL(ahead >= 1 && ahead + 1 <= 15, "DECODE_MATMUL_AHEAD must be 1..14");
    const uint32_t depth = ahead + 1;
    const uint32_t owned_max = dm_div_up(nc, ks);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const auto wdf = datatype_to_dataformat_converter(t.w.dtype());
    const auto odf = datatype_to_dataformat_converter(attrs.output_dtype);
    // every CB on every core of the rectangles (no CB on only some of them: tt-metal could then overlap them)
    add_cb(desc, all, CB_X, mt * kc_max, bf16);
    add_cb(desc, all, CB_W, depth * cr * sb, wdf);
    add_cb(desc, all, CB_RECV, ks > 1 ? owned_max * ks : 1, f32);
    const uint32_t x_chunk = 8;  // x's tiles a multicast
    add_cb(desc, all, CB_SCR, dm_div_up(dm_div_up(kc_max, x_chunk) * 16, tt::tile_size(f32)), f32);
    add_cb(desc, all, CB_PART, ks > 1 ? nc : 1, f32);
    add_cb(desc, all, CB_OUT, std::max(2u, mt * nc), odf);
    for (uint32_t id : {SEM_X, SEM_RED, SEM_NOC1}) {
        desc.semaphores.push_back(
            SemaphoreDescriptor{.id = id, .core_type = tt::CoreType::WORKER, .core_ranges = all, .initial_value = 0});
    }

    TT_FATAL(owned_max <= 4, "decode_matmul: {} output tiles an owner (at most 4)", owned_max);
    // the weight stream on both NoCs (attrs.nocs 2) in the dedicated NoC mode: the reader (NCRISC) also reads on NoC
    // 1 on every core but x's senders, the only cores where the control kernel (BRISC, NoC 1) reads, once that kernel
    // has cleared NoC 1's transaction ids and handed it over (SEM_NOC1); in the dynamic mode the control kernel's
    // linked x multicast hung
    TT_FATAL(ahead + 1 <= 8, "decode_matmul: the weight stream's transaction ids 1 .. {} (at most 8)", ahead + 1);
    std::vector<uint32_t> ct = {CB_W, attrs.debug == 1 ? 1u : 0u, SEM_NOC1};
    TensorAccessorArgs(w_buf).append_to(ct);
    auto weights = dm_kernel("decode_matmul_weights.cpp", all, std::move(ct), DataMovementProcessor::RISCV_1);
    ct = {CB_X, CB_PART, CB_RECV, CB_OUT, CB_SCR, SEM_X, SEM_RED, SEM_NOC1, attrs.nocs == 2 ? 1u : 0u, mt, kt, nt, sb};
    TensorAccessorArgs(x_buf).append_to(ct);
    TensorAccessorArgs(out_buf).append_to(ct);
    auto ctrl = dm_kernel("decode_matmul_ctrl.cpp", all, std::move(ct), DataMovementProcessor::RISCV_0);
    KernelDescriptor compute;
    compute.kernel_source = KDIR + "compute/decode_matmul_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all;
    compute.compile_time_args = {attrs.debug == 2 ? 1u : 0u, mt};
    std::vector<UnpackToDestMode> to_dest(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    to_dest[CB_RECV] = UnpackToDestMode::UnpackToDestFp32;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = matmul_fidelity(wdf),
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .unpack_to_dest_mode = std::move(to_dest),
    };

    for (uint32_t s = 0; s < ks; ++s) {
        const CoreCoord o = origin(s);
        // the rectangle's multicast corners on NoC 1: from the bottom-right core to the top-left one
        const CoreCoord mc0 = device->worker_core_from_logical_core(CoreCoord{o.x + cw - 1, o.y + rh - 1});
        const CoreCoord mc1 = device->worker_core_from_logical_core(o);
        // block i on position floor(i rect / nb): the busy cores spread over the rectangle
        std::vector<int32_t> block_at(rect, -1);
        for (uint32_t i = 0; i < nb; ++i) {
            block_at[i * rect / nb] = static_cast<int32_t>(i);
        }
        std::vector<uint32_t> pos_of(nb);
        for (uint32_t p = 0; p < rect; ++p) {
            if (block_at[p] >= 0) {
                pos_of[block_at[p]] = p;
            }
        }
        for (uint32_t p = 0; p < rect; ++p) {
            const CoreCoord core = core_of(s, p);
            const int32_t b = block_at[p];
            const uint32_t n0 = b >= 0 ? blocks[b].n0 : 0;
            const uint32_t ncb = b >= 0 ? blocks[b].ncb : 0;  // 0: idle (only receives x)
            const uint32_t owned = ks > 1 ? (s < ncb ? dm_div_up(ncb - s, ks) : 0) : ncb;
            const uint32_t k0 = k_first(s), kc = k_first(s + 1) - k0;
            {
                KernelDescriptor::RTArgList rt;
                rt.push_back(w_buf);
                for (uint32_t v : {k0, n0, ncb, nt, kc, cr, sb, ahead, stride}) {
                    rt.push_back(v);
                }
                rt.push_back(attrs.nocs == 2 && p != 0 ? 2u : 1u);  // (position 0: x's sender, its BRISC reads)
                rt.push_back(attrs.nocs == 2 && p != 0 ? std::min(AHEAD0, ahead) : 0u);
                weights.emplace_runtime_args(core, rt);
            }
            {
                KernelDescriptor::RTArgList rt;
                rt.push_back(x_buf);
                rt.push_back(out_buf);
                const uint32_t sender = p == 0 ? 1u : 0u;
                for (uint32_t v :
                     {k0,
                      kc,
                      n0,
                      ncb,
                      s,
                      ks,
                      rect,
                      sender,
                      x_chunk,
                      owned,
                      static_cast<uint32_t>(mc0.x),
                      static_cast<uint32_t>(mc0.y),
                      static_cast<uint32_t>(mc1.x),
                      static_cast<uint32_t>(mc1.y),
                      stride}) {
                    rt.push_back(v);
                }
                // the block's owners (slices 0 .. min(ncb, ks) - 1): their NoC coordinates
                for (uint32_t q = 0; q < std::min(ncb, ks) && ks > 1; ++q) {
                    const CoreCoord peer = device->worker_core_from_logical_core(core_of(q, pos_of[b]));
                    rt.push_back(peer.x);
                    rt.push_back(peer.y);
                }
                ctrl.emplace_runtime_args(core, rt);
            }
            {
                KernelDescriptor::RTArgList rt;
                for (uint32_t v : {ncb, kc, cr, sb, ks, owned}) {
                    rt.push_back(v);
                }
                compute.emplace_runtime_args(core, rt);
            }
        }
    }

    desc.kernels.push_back(std::move(weights));
    desc.kernels.push_back(std::move(ctrl));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
