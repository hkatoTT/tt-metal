// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_decode_experts.hpp"
#include "device/moe_decode_experts_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_experts {

ttnn::Tensor moe_decode_experts(
    const ttnn::Tensor& x,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& w1,
    const ttnn::Tensor& w2,
    uint32_t local,
    uint32_t inter,
    float limit,
    uint32_t row_tokens,
    const std::optional<ttnn::Tensor>& x_row,
    uint32_t h1_sem_x,
    uint32_t h1_sem_r,
    uint32_t rs_sem,
    uint32_t rs_mail,
    const std::optional<tt::tt_metal::MemoryConfig>& rs_memory_config,
    const std::vector<uint32_t>& w1_spec,
    const std::optional<tt::tt_metal::DataType>& rs_dtype) {
    return ttnn::prim::moe_decode_experts(
        x, weights, w1, w2, local, inter, limit, row_tokens, x_row, h1_sem_x, h1_sem_r, rs_sem, rs_mail,
        rs_memory_config, w1_spec, rs_dtype);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_decode_experts
