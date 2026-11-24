// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_prepare_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::experimental::prim {

namespace {
constexpr uint32_t DIM = 128;

void check(const Tensor& t, tt::tt_metal::DataType dtype, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE && t.is_allocated(), "{} must be an allocated device tensor", name);
    TT_FATAL(t.dtype() == dtype, "{} has the wrong dtype", name);
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
    TT_FATAL(
        !t.memory_config().is_sharded() && t.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "{} must be DRAM interleaved",
        name);
}
}  // namespace

uint32_t prepare_users(const DecodePrepareParams& args, const DecodePrepareTensorArgs& t) {
    return args.users != 0 ? args.users : t.proj.logical_shape()[-2];
}

void DecodePrepareDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    using tt::tt_metal::DataType;
    check(t.proj, DataType::BFLOAT16, "proj");
    check(t.fg, DataType::BFLOAT16, "fg");
    check(t.history0, DataType::FLOAT32, "history0");
    check(t.history1, DataType::FLOAT32, "history1");
    check(t.history2, DataType::FLOAT32, "history2");
    check(t.taps, DataType::FLOAT32, "taps");
    check(t.gate_consts, DataType::FLOAT32, "gate_consts");
    const uint32_t channels = 3 * args.heads * DIM;
    const uint32_t b = t.proj.logical_shape()[-2];
    const uint32_t users = prepare_users(args, t);
    TT_FATAL(users <= b, "proj has no rows 0 .. {}", users);
    TT_FATAL(args.tokens >= 1 && args.tokens <= 8, "tokens must be 1 .. 8");
    TT_FATAL(args.tokens == 1 || args.tokens * users <= b, "tokens: rows 0 .. tokens B of proj and fg");
    TT_FATAL(!t.keep.has_value() || args.tokens >= 2, "keep and saved: a verify step of 2 .. 8 tokens");
    TT_FATAL(!args.row_split || (args.tokens == 1 && !t.keep.has_value()), "row_split: the plain step's call");
    TT_FATAL(args.heads <= 16, "the beta logits must fit one face row (at most 16 heads)");
    TT_FATAL(t.proj.logical_shape()[-1] >= channels + args.heads, "proj must hold q | k | v and the beta logits");
    TT_FATAL(t.fg.logical_shape()[-1] >= args.heads * DIM && t.fg.logical_shape()[-2] == b, "fg must be [B, >= H 128]");
    for (const auto* h : {&t.history0, &t.history1, &t.history2}) {
        TT_FATAL(h->logical_shape()[-1] == channels && h->logical_shape()[-2] == users, "history must be [B, 3 H 128]");
    }
    TT_FATAL(t.keep.has_value() == t.saved.has_value(), "keep and saved go together");
    if (t.keep.has_value()) {
        const auto& k = t.keep.value();
        TT_FATAL(k.storage_type() == StorageType::DEVICE && k.is_allocated(), "keep must be an allocated device tensor");
        TT_FATAL(k.dtype() == DataType::UINT32 && k.layout() == tt::tt_metal::Layout::ROW_MAJOR, "keep: UINT32 ROW_MAJOR");
        TT_FATAL(
            !k.memory_config().is_sharded() && k.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "keep must be DRAM interleaved");
        TT_FATAL(k.logical_volume() == users && k.logical_shape()[-1] == users, "keep must be [1, B]: a flag a user");
        check(t.saved.value(), DataType::FLOAT32, "saved");
        const auto& sp = t.saved->padded_shape();
        const auto& hp = t.history0.padded_shape();
        const uint32_t rows = args.tokens - 1;  // the checkpoint rows: [rows, B, 3 H 128]
        TT_FATAL(
            sp[-1] == hp[-1] && sp[-2] == hp[-2] && t.saved->physical_volume() == rows * t.history0.physical_volume(),
            "saved must be {} histories", rows);
    }
    {  // out of place: all four outputs or none, each like its input (the history's, saved's)
        const int outs = int(t.history0_out.has_value()) + int(t.history1_out.has_value()) +
                         int(t.history2_out.has_value()) + int(t.saved_out.has_value());
        TT_FATAL(outs == 0 || outs == 4, "history0_out .. saved_out: all four or none");
        TT_FATAL(outs == 0 || t.saved.has_value(), "history0_out .. saved_out: with keep and saved");
        auto like = [&](const std::optional<Tensor>& o, const Tensor& in, const char* name) {
            if (o.has_value()) {
                TT_FATAL(o->storage_type() == StorageType::DEVICE && o->is_allocated() && o->dtype() == in.dtype() &&
                             o->layout() == in.layout() && o->padded_shape() == in.padded_shape() &&
                             o->memory_config() == in.memory_config() && o->buffer() != in.buffer(),
                         "{} must be like its input (another buffer)", name);
            }
        };
        like(t.history0_out, t.history0, "history0_out");
        like(t.history1_out, t.history1, "history1_out");
        like(t.history2_out, t.history2, "history2_out");
        if (t.saved.has_value()) {
            like(t.saved_out, t.saved.value(), "saved_out");
        }
    }
    TT_FATAL(t.taps.logical_shape()[-1] == channels && t.taps.logical_shape()[-2] == 4, "taps must be [4, 3 H 128]");
    TT_FATAL(
        t.gate_consts.logical_shape()[-1] == args.heads * DIM && t.gate_consts.logical_shape()[-2] == 2,
        "gate_consts must be [2, H 128]");
}

DecodePrepareDeviceOperation::spec_return_value_t DecodePrepareDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& t) {
    const uint32_t pairs = prepare_users(args, t) * args.heads * args.tokens;
    const auto where = args.l1_outputs ? tt::tt_metal::BufferType::L1 : tt::tt_metal::BufferType::DRAM;
    auto spec = [where](uint32_t rows, uint32_t cols) {
        return tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, rows, cols}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, where}));
    };
    return {spec(pairs, DIM), spec(pairs, DIM), spec(pairs, DIM), spec(pairs, DIM), spec(pairs, 1)};
}

DecodePrepareDeviceOperation::tensor_return_value_t DecodePrepareDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    auto specs = compute_output_specs(args, t);
    auto* device = t.proj.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device),
        create_device_tensor(specs[3], device),
        create_device_tensor(specs[4], device)};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 5> kda_decode_prepare(
    const Tensor& proj,
    const Tensor& fg,
    const Tensor& history0,
    const Tensor& history1,
    const Tensor& history2,
    const Tensor& taps,
    const Tensor& gate_consts,
    uint32_t heads,
    float lower,
    float q_scale,
    uint32_t users,
    const std::optional<Tensor>& keep,
    const std::optional<Tensor>& saved,
    uint32_t tokens,
    bool pair_major,
    bool l1_outputs,
    bool split_channels,
    bool row_split,
    const std::optional<Tensor>& history0_out,
    const std::optional<Tensor>& history1_out,
    const std::optional<Tensor>& history2_out,
    const std::optional<Tensor>& saved_out) {
    using OperationType = ttnn::experimental::prim::DecodePrepareDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        heads, lower, q_scale, users, tokens, pair_major, l1_outputs, split_channels, row_split};
    auto tensor_args =
        OperationType::tensor_args_t{proj, fg, history0, history1, history2, taps, gate_consts, keep, saved,
                                     history0_out, history1_out, history2_out, saved_out};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
