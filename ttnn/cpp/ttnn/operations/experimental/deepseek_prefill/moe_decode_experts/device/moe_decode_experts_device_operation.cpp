// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode_experts_device_operation.hpp"

#include <cstdlib>
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {
constexpr uint32_t DN_CORES = 32;

void check(const Tensor& t, tt::tt_metal::DataType dtype, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE && t.is_allocated(), "{} must be an allocated device tensor", name);
    TT_FATAL(t.dtype() == dtype, "{} has the wrong dtype", name);
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
    TT_FATAL(t.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM, "{} must be in DRAM", name);
}
}  // namespace

void MoeDecodeExpertsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    using tt::tt_metal::DataType;
    check(t.x, DataType::BFLOAT16, "x");
    // I = 256 (T = 8 column tiles of every expert of the mesh column on each chip: a gate | up core per (expert,
    // column tile), the chosen experts in groups of 8), on 1-16 tile rows of tokens (more than one: packed units;
    // their L1 grows with the rows: the gate | up cores' x K slices 32 KB a row).
    TT_FATAL(args.inter == 256, "I must be 256 (8 column tiles of each expert on the chip)");
    check(t.w1, t.w1.dtype(), "w1");
    check(t.w2, t.w2.dtype(), "w2");
    const uint32_t n = t.x.logical_shape()[-2];
    const uint32_t d = t.x.logical_shape()[-1];
    check(t.weights, DataType::FLOAT32, "weights");
    TT_FATAL(n % 32 == 0 && n >= 32 && n <= 512, "x must be 1-16 tile rows of tokens (N = 32 .. 512)");
    TT_FATAL(t.weights.logical_shape()[-2] == n && t.weights.logical_shape()[-1] == 32, "weights must be [N, 32]");
    TT_FATAL(args.local >= 1 && args.local <= 512, "local must be 1..512");
    TT_FATAL(d % (32 * DN_CORES) == 0, "D must split over the 32 down cores");
    TT_FATAL(
        args.row_tokens == 0 || ((args.row_tokens == 8 || args.row_tokens == 16) && n % args.row_tokens == 0),
        "row_tokens must be 0, or 8 / 16 dividing N (y by mesh row)");
    const uint32_t p = args.inter / 32;
    // the progressive form (GLM53_MOE_PROG_F, the program factory): each slot a base and a correction, 2 L slots
    const uint32_t slots = std::getenv("GLM53_MOE_PROG_F") != nullptr ? 2 * args.local : args.local;
    TT_FATAL(
        t.w1.logical_shape()[-2] == p * slots * d && t.w1.logical_shape()[-1] == 64,
        "w1 must be [P L D, 64] (each core's gate and up column tiles, expert by expert; 2 L with GLM53_MOE_PROG_F)");
    TT_FATAL(
        t.w2.logical_shape()[-2] == DN_CORES * slots * args.inter && t.w2.logical_shape()[-1] == d / DN_CORES,
        "w2 must be [Q L I, D / Q] (each down core's columns, expert by expert; 2 L with GLM53_MOE_PROG_F)");
    auto grid = t.x.device()->compute_with_storage_grid_size();
    TT_FATAL(grid.x >= 12 && grid.y >= 8, "needs a 12 x 8 core grid");
    if (args.h1_sem_x != 0 || args.h1_sem_r != 0) {  // the in-op gather: the column's x and routing over the fabric
        TT_FATAL(args.h1_sem_x != 0 && args.h1_sem_r != 0, "in-op gather: needs both semaphores");
        TT_FATAL(t.x_row.has_value(), "in-op gather: needs x_row (this mesh row's tokens)");
        const auto& xr = t.x_row.value();
        check(xr, DataType::BFLOAT16, "x_row");
        TT_FATAL(!xr.memory_config().is_sharded(), "x_row must be DRAM interleaved");
        TT_FATAL(!t.x.memory_config().is_sharded() && !t.weights.memory_config().is_sharded(),
                 "in-op gather: x and weights (the landing buffers) must be DRAM interleaved");
        const uint32_t rt = xr.logical_shape()[-2];
        TT_FATAL(rt == 8 || rt == 16 || rt == 32 || rt == 64, "in-op gather: x_row must have 8, 16, 32 or 64 tokens");
        TT_FATAL(xr.logical_shape()[-1] == d, "in-op gather: x_row must be as wide as x");
        TT_FATAL(
            n == 8 * rt && args.inter == 256 && n >= 64,
            "in-op gather: x must be the column's 8 t tokens (I = 256, packed units)");
        TT_FATAL(grid.y >= 10, "in-op gather: the sender core (the grid's last) must be below the others");
    }
    if (args.rs_sem != 0 || args.rs_mail != 0) {  // the column's way back in the op
        TT_FATAL(args.rs_sem != 0 && args.rs_mail != 0, "the way back in the op needs both semaphores");
        TT_FATAL(args.h1_sem_x != 0, "the way back in the op: with the in-op gather only");
        const uint32_t rt = t.x_row.value().logical_shape()[-2];
        TT_FATAL(rt == 32 || rt == 64, "the way back in the op: 32 or 64 tokens a mesh row");
        TT_FATAL(args.row_tokens == 0, "the way back in the op: not with row_tokens");
        TT_FATAL(grid.x >= 12 && grid.y >= 10, "the way back in the op: a 12 x 10 grid");
    }
    TT_FATAL(!args.rs_memory_config.has_value() || args.rs_sem != 0, "rs_memory_config: with the way back in the op");
}

MoeDecodeExpertsDeviceOperation::spec_return_value_t MoeDecodeExpertsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& t) {
    const uint32_t n = t.x.logical_shape()[-2];
    const uint32_t rows = args.rs_sem != 0 ? t.x_row.value().logical_shape()[-2]  // (the way back: this mesh row's)
                          : args.row_tokens ? n / args.row_tokens * 32
                                            : n;
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, rows, t.x.logical_shape()[-1]}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            args.rs_memory_config.value_or(tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM})));
}

MoeDecodeExpertsDeviceOperation::tensor_return_value_t MoeDecodeExpertsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(args, t), t.x.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor moe_decode_experts(
    const Tensor& x,
    const Tensor& weights,
    const Tensor& w1,
    const Tensor& w2,
    uint32_t local,
    uint32_t inter,
    float limit,
    uint32_t row_tokens,
    const std::optional<Tensor>& x_row,
    uint32_t h1_sem_x,
    uint32_t h1_sem_r,
    uint32_t rs_sem,
    uint32_t rs_mail,
    const std::optional<tt::tt_metal::MemoryConfig>& rs_memory_config) {
    using OperationType = ttnn::experimental::prim::MoeDecodeExpertsDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        local, inter, limit, row_tokens, h1_sem_x, h1_sem_r, rs_sem, rs_mail, rs_memory_config};
    auto tensor_args = OperationType::tensor_args_t{x, weights, w1, w2, x_row};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
