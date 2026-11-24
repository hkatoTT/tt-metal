// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_read.hpp"
#include "device/mhc_stream_read_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_read {

std::array<ttnn::Tensor, 3> mhc_stream_read(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& fn,
    const ttnn::Tensor& mix_consts,
    const ttnn::Tensor& consts,
    const ttnn::Tensor& gamma,
    uint32_t tokens,
    uint32_t n,
    uint32_t sinkhorn_iters,
    uint32_t slices,
    float eps,
    float norm_eps,
    float rms_eps,
    uint32_t offset,
    const std::optional<std::array<ttnn::Tensor, 3>>& outputs) {
    return ttnn::prim::mhc_stream_read(
        streams,
        fn,
        mix_consts,
        consts,
        gamma,
        tokens,
        n,
        sinkhorn_iters,
        slices,
        eps,
        norm_eps,
        rms_eps,
        offset,
        outputs);
}

std::array<ttnn::Tensor, 4> mhc_stream_write_read(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& block_out,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    const ttnn::Tensor& fn,
    const ttnn::Tensor& mix_consts,
    const ttnn::Tensor& consts,
    const ttnn::Tensor& gamma,
    uint32_t tokens,
    uint32_t n,
    uint32_t sinkhorn_iters,
    uint32_t slices,
    float eps,
    float norm_eps,
    float rms_eps,
    uint32_t offset,
    const std::optional<std::array<ttnn::Tensor, 4>>& outputs) {
    return ttnn::prim::mhc_stream_write_read(
        streams,
        block_out,
        post,
        comb,
        fn,
        mix_consts,
        consts,
        gamma,
        tokens,
        n,
        sinkhorn_iters,
        slices,
        eps,
        norm_eps,
        rms_eps,
        offset,
        outputs);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_read
