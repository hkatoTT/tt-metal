// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_mix.hpp"
#include "device/mhc_stream_mix_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_mix {

std::array<ttnn::Tensor, 3> mhc_stream_mix(
    const ttnn::Tensor& partial,
    const ttnn::Tensor& mix_consts,
    const ttnn::Tensor& consts,
    uint32_t tokens,
    uint32_t n,
    uint32_t sinkhorn_iters,
    float eps,
    float norm_eps,
    uint32_t hidden) {
    return ttnn::prim::mhc_stream_mix(partial, mix_consts, consts, tokens, n, sinkhorn_iters, eps, norm_eps, hidden);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_mix
