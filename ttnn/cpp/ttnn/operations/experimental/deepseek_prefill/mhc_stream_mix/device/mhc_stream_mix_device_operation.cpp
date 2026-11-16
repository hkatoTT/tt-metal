// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_mix_device_operation.hpp"

#include <algorithm>
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

void MhcStreamMixDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using tt::constants::TILE_HW;
    for (const auto* t : {&tensor_args.partial, &tensor_args.mix_consts, &tensor_args.consts}) {
        TT_FATAL(t->storage_type() == StorageType::DEVICE && t->is_allocated(), "inputs must be allocated device tensors");
        TT_FATAL(t->dtype() == tt::tt_metal::DataType::FLOAT32, "inputs must be FLOAT32");
        TT_FATAL(t->layout() == tt::tt_metal::Layout::TILE, "inputs must be TILE layout");
        TT_FATAL(
            !t->memory_config().is_sharded() && t->memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "inputs must be DRAM interleaved");
    }
    const auto& p = tensor_args.partial.logical_shape();
    TT_FATAL(p.rank() == 3 && p[-1] == 128, "partial must be [S, R, 128]");
    TT_FATAL(
        p[-2] % 32 == 0 && (p[-2] <= 128 || p[-2] % 128 == 0),
        "partial's R must be 32, 64, 96 or 128 (at most 32 tokens), or a multiple of 128 (groups of 32 tokens)");
    TT_FATAL(args.n == 4, "n must be 4");
    TT_FATAL(args.tokens >= 1 && args.tokens * args.n <= p[-2], "the tokens' rows must fit R");
    TT_FATAL(p[-2] <= 128 || args.tokens * args.n == p[-2], "above 32 tokens, whole groups of 32 tokens");
    TT_FATAL(p[-2] / 128 <= 10, "at most 10 groups of 32 tokens (one core each)");
    TT_FATAL(tensor_args.mix_consts.physical_volume() / TILE_HW == 13, "mix_consts must be 13 tiles");
    TT_FATAL(tensor_args.consts.physical_volume() / TILE_HW == 8, "consts must be 8 tiles");
}

MhcStreamMixDeviceOperation::spec_return_value_t MhcStreamMixDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto spec = [](uint32_t rows, uint32_t cols) {
        return tt::tt_metal::TensorSpec(
            ttnn::Shape({rows, cols}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig{
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    };
    // P: [T, R] (one group); above 32 tokens each group's [32, 128] (its own rows only), as [T, 128]
    const uint32_t rows = std::min<uint32_t>(tensor_args.partial.logical_shape()[-2], 128);
    return {spec(args.tokens, args.n), spec(args.tokens, args.n * args.n), spec(args.tokens, rows)};
}

MhcStreamMixDeviceOperation::tensor_return_value_t MhcStreamMixDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.partial.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device)};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 3> mhc_stream_mix(
    const Tensor& partial,
    const Tensor& mix_consts,
    const Tensor& consts,
    uint32_t tokens,
    uint32_t n,
    uint32_t iters,
    float eps,
    float norm_eps,
    uint32_t hidden) {
    using OperationType = ttnn::experimental::prim::MhcStreamMixDeviceOperation;
    const float inv_count = 1.0f / static_cast<float>(n * hidden);
    auto operation_attributes = OperationType::operation_attributes_t{tokens, n, iters, eps, norm_eps, inv_count};
    auto tensor_args = OperationType::tensor_args_t{partial, mix_consts, consts};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
