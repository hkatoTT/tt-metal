// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_write_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {
void check_dram_tile(const Tensor& t, tt::tt_metal::DataType dtype, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE && t.is_allocated(), "{} must be an allocated device tensor", name);
    TT_FATAL(t.dtype() == dtype, "{} has the wrong dtype", name);
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
    TT_FATAL(
        !t.memory_config().is_sharded() && t.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "{} must be DRAM interleaved",
        name);
}

// block_out may also stay where the row's all-reduce left it, sharded in L1 (the reader takes its pages through a
// TensorAccessor either way): one move back to DRAM less per sub-layer
void check_block_out(const Tensor& t) {
    if (t.memory_config().is_sharded() && t.memory_config().buffer_type() == tt::tt_metal::BufferType::L1) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE && t.is_allocated(), "block_out must be allocated");
        TT_FATAL(t.dtype() == tt::tt_metal::DataType::BFLOAT16, "block_out has the wrong dtype");
        TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "block_out must be TILE layout");
        TT_FATAL(t.buffer()->buffer_distribution_spec().has_value(), "block_out's shards need a distribution spec");
        return;
    }
    check_dram_tile(t, tt::tt_metal::DataType::BFLOAT16, "block_out");
}
}  // namespace

void MhcStreamWriteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    const auto& streams = tensor_args.streams;
    const auto& block_out = tensor_args.block_out;
    check_dram_tile(streams, tt::tt_metal::DataType::BFLOAT16, "streams");
    check_block_out(block_out);
    check_dram_tile(tensor_args.post, tt::tt_metal::DataType::FLOAT32, "post");
    check_dram_tile(tensor_args.comb, tt::tt_metal::DataType::FLOAT32, "comb");
    TT_FATAL(args.n == 4, "n must be 4 (8 tokens' streams per 32-row tile)");
    const uint32_t d = streams.logical_shape()[-1];
    const uint32_t rows = streams.logical_shape()[-2];
    TT_FATAL(d % TILE_WIDTH == 0 && block_out.logical_shape()[-1] == d, "streams and block_out must be [.., D]");
    TT_FATAL(rows % TILE_HEIGHT == 0, "streams' rows must fill whole tiles (n Tp, Tp a multiple of 8)");
    TT_FATAL(args.tokens * args.n <= rows, "streams must hold n rows per token");
    TT_FATAL(streams.logical_volume() == static_cast<uint64_t>(rows) * d, "streams must be [n Tp, D] (leading dims 1)");
    TT_FATAL(
        block_out.logical_volume() == static_cast<uint64_t>(args.tokens) * d,
        "block_out must be [T, D] (leading dims 1)");
    TT_FATAL(
        tensor_args.post.logical_shape()[-2] == args.tokens && tensor_args.post.logical_shape()[-1] == args.n,
        "post must be [T, n]");
    TT_FATAL(
        tensor_args.comb.logical_shape()[-2] == args.tokens &&
            tensor_args.comb.logical_shape()[-1] == args.n * args.n,
        "comb must be [T, n n]");
    TT_FATAL(
        tensor_args.post.padded_shape()[-1] == TILE_WIDTH && tensor_args.comb.padded_shape()[-1] == TILE_WIDTH,
        "post and comb must be one tile wide");
}

MhcStreamWriteDeviceOperation::spec_return_value_t MhcStreamWriteDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return tt::tt_metal::TensorSpec(
        tensor_args.streams.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

MhcStreamWriteDeviceOperation::tensor_return_value_t MhcStreamWriteDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.streams.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor mhc_stream_write(
    const Tensor& streams, const Tensor& block_out, const Tensor& post, const Tensor& comb, uint32_t n) {
    using OperationType = ttnn::experimental::prim::MhcStreamWriteDeviceOperation;
    const uint32_t tokens = block_out.logical_shape()[-2];
    auto operation_attributes = OperationType::operation_attributes_t{n, tokens};
    auto tensor_args = OperationType::tensor_args_t{streams, block_out, post, comb};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
