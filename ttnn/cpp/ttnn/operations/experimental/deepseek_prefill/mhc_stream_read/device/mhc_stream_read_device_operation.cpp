// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_read_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

#include <cstdlib>

namespace ttnn::experimental::prim {

// the outputs' tokens: the streams' rows / n in the prefill's groups form (a long wave's calls each take a part:
// `offset`), else the call's
static uint32_t total_tokens(const MhcStreamReadParams& args, const Tensor& streams) {
    const uint32_t rows = streams.logical_shape()[-2];
    return rows > 64 && args.slices <= 5 ? rows / args.n : args.tokens;
}

void MhcStreamReadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using tt::constants::TILE_HW;
    using tt::constants::TILE_WIDTH;
    const auto& t = tensor_args;
    for (const auto* x : {&t.streams, &t.fn, &t.mix_consts, &t.consts, &t.gamma}) {
        TT_FATAL(x->storage_type() == StorageType::DEVICE && x->is_allocated(), "inputs must be allocated device tensors");
        TT_FATAL(x->layout() == tt::tt_metal::Layout::TILE, "inputs must be TILE layout");
        TT_FATAL(
            !x->memory_config().is_sharded() && x->memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "inputs must be DRAM interleaved");
    }
    for (const auto* x : {&t.streams, &t.fn, &t.gamma}) {
        TT_FATAL(x->dtype() == tt::tt_metal::DataType::BFLOAT16, "streams, fn and gamma must be BFLOAT16");
    }
    for (const auto* x : {&t.mix_consts, &t.consts}) {
        TT_FATAL(x->dtype() == tt::tt_metal::DataType::FLOAT32, "the constants must be FLOAT32");
    }
    const auto& s = t.streams.logical_shape();
    const uint32_t rows = s[-2], d = s[-1];
    TT_FATAL(args.n == 4, "n must be 4");
    TT_FATAL(args.offset == 0 || (rows > 64 && args.slices <= 5), "offset: the prefill's groups form only");
    if (rows > 64 && args.slices > 5) {  // the groups form of 8 tokens a group, one a grid row (decode)
        const auto grid = t.streams.device()->compute_with_storage_grid_size();
        TT_FATAL(args.tokens % 32 == 0 && rows == args.tokens * args.n, "groups of 8: whole tile rows of tokens");
        TT_FATAL(args.slices + 1 <= grid.x && d % (args.slices * TILE_WIDTH) == 0, "groups of 8: bad slices");
        TT_FATAL(args.tokens / 8 <= grid.y, "groups of 8: {} tokens do not fit", args.tokens);
    } else if (rows > 64) {  // the groups form (mhc_stream_read_program_factory.cpp): 32 tokens a group, two a grid row
        const auto grid = t.streams.device()->compute_with_storage_grid_size();
        TT_FATAL(
            args.tokens % 32 == 0 && args.offset % 32 == 0 && rows % args.n == 0 &&
                args.offset + args.tokens <= rows / args.n,
            "groups: 32 tokens a group from a multiple of 32, within the streams' rows / n tokens");
        TT_FATAL(
            args.offset == 0 || t.post_out.has_value(), "groups: a later call (offset) writes an earlier's outputs");
        TT_FATAL(args.slices >= 1 && args.slices <= 5 && d % (args.slices * TILE_WIDTH) == 0, "groups: bad slices");
        TT_FATAL(grid.x >= 12 && (args.tokens / 32 + 1) / 2 <= grid.y, "groups: {} tokens do not fit", args.tokens);
    } else {
        TT_FATAL(rows % 32 == 0, "streams must have 32 or 64 rows (at most 16 tokens), or 4 x tokens rows");
        TT_FATAL(args.tokens >= 1 && args.tokens * args.n <= rows, "the tokens' rows must fit the streams");
        TT_FATAL(args.slices >= 1 && args.slices <= 16 && d % (args.slices * TILE_WIDTH) == 0, "bad slice count");
        TT_FATAL(args.slices % 8 == 0 || args.slices <= 8, "slices must fit an 8-core-wide rectangle");
    }
    TT_FATAL(t.fn.logical_shape()[-2] == d && t.fn.logical_shape()[-1] == args.n * 24, "fn must be [D, n 24]");
    TT_FATAL(t.gamma.logical_shape()[-1] == d, "gamma must be [.., D]");
    TT_FATAL(t.mix_consts.physical_volume() / TILE_HW == 13, "mix_consts must be 13 tiles");
    TT_FATAL(t.consts.physical_volume() / TILE_HW == 8, "consts must be 8 tiles");
    TT_FATAL(
        t.post_out.has_value() == t.comb_out.has_value() && t.post_out.has_value() == t.x_out.has_value(),
        "the outputs: all three or none");
    if (t.post_out.has_value()) {
        const auto specs = compute_output_specs(args, t);
        const std::array<const Tensor*, 3> outs = {&*t.post_out, &*t.comb_out, &*t.x_out};
        for (uint32_t i = 0; i < 3; ++i) {
            TT_FATAL(outs[i]->tensor_spec() == specs[i], "output {}: {} expected", i, specs[i]);
            TT_FATAL(outs[i]->is_allocated(), "output {} must be allocated", i);
        }
    }
}

MhcStreamReadDeviceOperation::spec_return_value_t MhcStreamReadDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto spec = [](ttnn::Shape shape, tt::tt_metal::DataType dtype) {
        return tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                dtype,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig{
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    };
    const uint32_t d = tensor_args.streams.logical_shape()[-1];
    const uint32_t tokens = total_tokens(args, tensor_args.streams);
    return {
        spec(ttnn::Shape({tokens, args.n}), tt::tt_metal::DataType::FLOAT32),
        spec(ttnn::Shape({tokens, args.n * args.n}), tt::tt_metal::DataType::FLOAT32),
        spec(ttnn::Shape({1, 1, tokens, d}), tt::tt_metal::DataType::BFLOAT16)};
}

MhcStreamReadDeviceOperation::tensor_return_value_t MhcStreamReadDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.post_out.has_value()) {
        return {*tensor_args.post_out, *tensor_args.comb_out, *tensor_args.x_out};
    }
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.streams.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device)};
}

void MhcStreamWriteReadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& t = tensor_args;
    MhcStreamReadDeviceOperation::validate_on_program_cache_miss(
        args,
        MhcStreamReadTensorArgs{t.streams, t.fn, t.mix_consts, t.consts, t.gamma, t.post_out, t.comb_out, t.x_out});
    const uint32_t rows = t.streams.logical_shape()[-2];
    const uint32_t tokens = total_tokens(args, t.streams);
    TT_FATAL(args.offset == 0 || t.streams_out.has_value(), "a later call (offset) writes an earlier's new streams");
    if (t.streams_out.has_value()) {
        TT_FATAL(
            t.streams_out->tensor_spec() == compute_output_specs(args, t)[0] && t.streams_out->is_allocated(),
            "streams_out: the streams' spec, allocated");
    }
    // the single form (at most 16 tokens), the groups of 8 at 32 or 64 tokens, or the prefill's groups of 32 (the write
    // on the cores the read leaves)
    TT_FATAL(
        rows <= 64 || (args.slices > 5 && (args.tokens == 32 || args.tokens == 64)) ||
            (args.slices <= 5 && args.tokens <= 512),
        "mhc_stream_write_read: up to 16 tokens, 32 or 64 in groups of 8, or up to 512 in groups of 32");
    TT_FATAL(t.block_out.logical_shape()[-2] == tokens, "block_out must be [.., T, D] for the streams' T tokens");
    TT_FATAL(t.block_out.dtype() == tt::tt_metal::DataType::BFLOAT16, "block_out must be BFLOAT16");
    TT_FATAL(t.block_out.layout() == tt::tt_metal::Layout::TILE, "block_out must be TILE layout");
    for (const auto* x : {&t.post, &t.comb}) {
        TT_FATAL(x->dtype() == tt::tt_metal::DataType::FLOAT32, "post and comb must be FLOAT32");
        TT_FATAL(
            x->layout() == tt::tt_metal::Layout::TILE && !x->memory_config().is_sharded() &&
                x->memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "post and comb must be DRAM interleaved TILE");
        TT_FATAL(x->logical_shape()[-2] == tokens, "post and comb must be [T, ..]");
    }
}

MhcStreamWriteReadDeviceOperation::spec_return_value_t MhcStreamWriteReadDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& t = tensor_args;
    const auto read = MhcStreamReadDeviceOperation::compute_output_specs(
        args, MhcStreamReadTensorArgs{t.streams, t.fn, t.mix_consts, t.consts, t.gamma});
    const auto streams = tt::tt_metal::TensorSpec(
        t.streams.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    return {streams, read[0], read[1], read[2]};
}

MhcStreamWriteReadDeviceOperation::tensor_return_value_t MhcStreamWriteReadDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.streams_out.has_value()) {
        return {*tensor_args.streams_out, *tensor_args.post_out, *tensor_args.comb_out, *tensor_args.x_out};
    }
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.streams.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device),
        create_device_tensor(specs[3], device)};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 3> mhc_stream_read(
    const Tensor& streams,
    const Tensor& fn,
    const Tensor& mix_consts,
    const Tensor& consts,
    const Tensor& gamma,
    uint32_t tokens,
    uint32_t n,
    uint32_t iters,
    uint32_t slices,
    float eps,
    float norm_eps,
    float rms_eps,
    uint32_t offset,
    const std::optional<std::array<Tensor, 3>>& outputs) {
    using OperationType = ttnn::experimental::prim::MhcStreamReadDeviceOperation;
    const uint32_t hidden = streams.logical_shape()[-1];
    const float inv_count = 1.0f / static_cast<float>(n * hidden);
    const float inv_hidden = 1.0f / static_cast<float>(hidden);
    auto operation_attributes = OperationType::operation_attributes_t{
        tokens, n, iters, slices, eps, norm_eps, inv_count, rms_eps, inv_hidden, offset};
    auto tensor_args = OperationType::tensor_args_t{streams, fn, mix_consts, consts, gamma};
    if (outputs.has_value()) {
        tensor_args.post_out = (*outputs)[0];
        tensor_args.comb_out = (*outputs)[1];
        tensor_args.x_out = (*outputs)[2];
    }
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

std::array<Tensor, 4> mhc_stream_write_read(
    const Tensor& streams,
    const Tensor& block_out,
    const Tensor& post,
    const Tensor& comb,
    const Tensor& fn,
    const Tensor& mix_consts,
    const Tensor& consts,
    const Tensor& gamma,
    uint32_t tokens,
    uint32_t n,
    uint32_t iters,
    uint32_t slices,
    float eps,
    float norm_eps,
    float rms_eps,
    uint32_t offset,
    const std::optional<std::array<Tensor, 4>>& outputs) {
    using OperationType = ttnn::experimental::prim::MhcStreamWriteReadDeviceOperation;
    const uint32_t hidden = streams.logical_shape()[-1];
    const float inv_count = 1.0f / static_cast<float>(n * hidden);
    const float inv_hidden = 1.0f / static_cast<float>(hidden);
    auto operation_attributes = OperationType::operation_attributes_t{
        tokens, n, iters, slices, eps, norm_eps, inv_count, rms_eps, inv_hidden, offset};
    auto tensor_args = OperationType::tensor_args_t{streams, block_out, post, comb, fn, mix_consts, consts, gamma};
    if (outputs.has_value()) {
        tensor_args.streams_out = (*outputs)[0];
        tensor_args.post_out = (*outputs)[1];
        tensor_args.comb_out = (*outputs)[2];
        tensor_args.x_out = (*outputs)[3];
    }
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
