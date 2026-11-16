// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode_route_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {
void check(const Tensor& t, tt::tt_metal::DataType dtype, tt::tt_metal::Layout layout, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE && t.is_allocated(), "{} must be an allocated device tensor", name);
    TT_FATAL(t.dtype() == dtype, "{} has the wrong dtype", name);
    TT_FATAL(t.layout() == layout, "{} has the wrong layout", name);
    TT_FATAL(
        !t.memory_config().is_sharded() && t.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "{} must be DRAM interleaved",
        name);
}

tt::tt_metal::TensorSpec spec(uint32_t rows, uint32_t cols, tt::tt_metal::DataType dtype, tt::tt_metal::Layout layout) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, rows, cols}),
        tt::tt_metal::TensorLayout(
            dtype,
            tt::tt_metal::PageConfig(layout),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}
}  // namespace

void MoeDecodeRouteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    check(t.logits, DataType::FLOAT32, Layout::TILE, "logits");
    check(t.bias, DataType::FLOAT32, Layout::TILE, "bias");
    check(t.local_base, DataType::UINT32, Layout::ROW_MAJOR, "local_base");
    const uint32_t n = t.logits.logical_shape()[-2];
    const uint32_t e = t.logits.logical_shape()[-1];
    TT_FATAL(n % 32 == 0 && e % 32 == 0, "logits must be whole tiles, [N, E] with N and E multiples of 32");
    TT_FATAL(t.bias.logical_shape()[-1] == e, "bias must be [1, E]");
    TT_FATAL(args.tokens >= 1 && args.tokens <= n, "tokens must be 1..N");
    TT_FATAL(args.k >= 1 && args.k <= 16 && args.k <= e, "k must be 1..16");
    // 1..16: the chip's weights in columns 0 .. local - 1 (one face row); 0: every chosen id and weight; above 16: the
    // chosen ids of base .. base + local - 1 (as e - base) and their weights
    TT_FATAL(args.local <= 16 || args.local < e, "local must be 0..16, or a range of experts below E");
}

MoeDecodeRouteDeviceOperation::spec_return_value_t MoeDecodeRouteDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& t) {
    return spec(t.logits.logical_shape()[-2], 32, tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE);
}

MoeDecodeRouteDeviceOperation::tensor_return_value_t MoeDecodeRouteDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(args, t), t.logits.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor moe_decode_route(
    const Tensor& logits,
    const Tensor& bias,
    const Tensor& local_base,
    uint32_t tokens,
    uint32_t k,
    uint32_t local,
    float scale,
    bool shared) {
    using OperationType = ttnn::experimental::prim::MoeDecodeRouteDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{tokens, k, local, scale, shared};
    auto tensor_args = OperationType::tensor_args_t{logits, bias, local_base};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
