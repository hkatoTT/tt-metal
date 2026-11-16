// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode_route.hpp"
#include "device/moe_decode_route_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_route {

ttnn::Tensor moe_decode_route(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& bias,
    const ttnn::Tensor& local_base,
    uint32_t tokens,
    uint32_t k,
    uint32_t local,
    float scale,
    bool shared) {
    return ttnn::prim::moe_decode_route(logits, bias, local_base, tokens, k, local, scale, shared);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_route
