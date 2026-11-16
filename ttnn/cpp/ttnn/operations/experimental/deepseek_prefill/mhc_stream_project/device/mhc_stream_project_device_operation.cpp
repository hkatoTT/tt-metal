// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_project_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {
constexpr uint32_t PROJ = 96;  // 4 streams x 24 projection outputs
}

void MhcStreamProjectDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using tt::constants::TILE_WIDTH;
    for (const auto* t : {&tensor_args.streams, &tensor_args.fn}) {
        TT_FATAL(t->storage_type() == StorageType::DEVICE && t->is_allocated(), "inputs must be allocated device tensors");
        TT_FATAL(t->dtype() == tt::tt_metal::DataType::BFLOAT16, "streams and fn must be BFLOAT16");
        TT_FATAL(t->layout() == tt::tt_metal::Layout::TILE, "streams and fn must be TILE layout");
        TT_FATAL(
            !t->memory_config().is_sharded() && t->memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "streams and fn must be DRAM interleaved");
    }
    const auto& s = tensor_args.streams.logical_shape();
    const auto& f = tensor_args.fn.logical_shape();
    TT_FATAL(s[-2] % TILE_WIDTH == 0, "streams' rows must fill whole tiles");
    TT_FATAL(f[-2] == s[-1] && f[-1] == PROJ, "fn must be [D, {}]", PROJ);
    TT_FATAL(tensor_args.streams.logical_volume() == static_cast<uint64_t>(s[-2]) * s[-1], "streams must be [R, D]");
    TT_FATAL(tensor_args.fn.logical_volume() == static_cast<uint64_t>(f[-2]) * f[-1], "fn must be [D, 96]");
    const uint32_t d_tiles = s[-1] / TILE_WIDTH;
    TT_FATAL(args.slices > 0 && d_tiles % args.slices == 0, "the slices must divide D's {} tiles", d_tiles);
}

MhcStreamProjectDeviceOperation::spec_return_value_t MhcStreamProjectDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const uint32_t rows = tensor_args.streams.logical_shape()[-2];
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({args.slices, rows, PROJ + tt::constants::TILE_WIDTH}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

MhcStreamProjectDeviceOperation::tensor_return_value_t MhcStreamProjectDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.streams.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor mhc_stream_project(const Tensor& streams, const Tensor& fn, uint32_t slices) {
    using OperationType = ttnn::experimental::prim::MhcStreamProjectDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{slices};
    auto tensor_args = OperationType::tensor_args_t{streams, fn};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
