// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_finish_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {
constexpr uint32_t DIM = 128;

void check(const Tensor& t, tt::tt_metal::DataType dtype, const char* name, bool l1_ok = false) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE && t.is_allocated(), "{} must be an allocated device tensor", name);
    TT_FATAL(t.dtype() == dtype, "{} has the wrong dtype", name);
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
    TT_FATAL(
        !t.memory_config().is_sharded() &&
            (t.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM ||
             (l1_ok && t.memory_config().buffer_type() == tt::tt_metal::BufferType::L1)),
        "{} must be DRAM{} interleaved",
        name,
        l1_ok ? " or L1" : "");
}
}  // namespace

void DecodeFinishDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    using tt::tt_metal::DataType;
    check(t.o, DataType::FLOAT32, "o", true);  // (the recurrence's o may be L1 interleaved)
    check(t.fg, DataType::BFLOAT16, "fg");
    check(t.o_norm, DataType::FLOAT32, "o_norm");
    const uint32_t b = t.fg.logical_shape()[-2];
    TT_FATAL(t.o.logical_shape()[-1] == DIM && t.o.logical_shape()[-2] == b * args.heads, "o must be [B H, 128]");
    TT_FATAL(t.fg.logical_shape()[-1] >= 2 * args.heads * DIM, "fg must hold f | gate");
    TT_FATAL(t.o_norm.logical_shape()[-1] == DIM, "o_norm must be [.., 128]");
}

DecodeFinishDeviceOperation::spec_return_value_t DecodeFinishDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, t.fg.logical_shape()[-2], args.heads * DIM}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

DecodeFinishDeviceOperation::tensor_return_value_t DecodeFinishDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(args, t), t.o.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor kda_decode_finish(
    const Tensor& o, const Tensor& fg, const Tensor& o_norm, uint32_t heads, float eps) {
    using OperationType = ttnn::experimental::prim::DecodeFinishDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{heads, eps};
    auto tensor_args = OperationType::tensor_args_t{o, fg, o_norm};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
