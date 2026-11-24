// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_decode_step.hpp"
#include "device/recurrent_decode_step_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor recurrent_decode_step(
    const ttnn::Tensor& state,
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& decay,
    const ttnn::Tensor& beta,
    const std::optional<ttnn::Tensor>& source,
    const std::optional<ttnn::Tensor>& keep,
    uint32_t heads,
    const std::optional<ttnn::Tensor>& output,
    uint32_t form,
    uint32_t tokens,
    const std::optional<ttnn::Tensor>& saved,
    bool pair_major,
    const std::optional<ttnn::Tensor>& saved_source) {
    return ttnn::prim::kda_recurrent_decode_step(
        state, q, k, v, decay, beta, source, keep, heads, output, form, tokens, saved, pair_major, saved_source);
}

}  // namespace ttnn::experimental::kda
