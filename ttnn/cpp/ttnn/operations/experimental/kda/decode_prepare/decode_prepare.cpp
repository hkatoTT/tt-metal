// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_prepare.hpp"
#include "device/decode_prepare_device_operation.hpp"

namespace ttnn::experimental::kda {

std::array<ttnn::Tensor, 5> decode_prepare(
    const ttnn::Tensor& proj,
    const ttnn::Tensor& fg,
    const ttnn::Tensor& history0,
    const ttnn::Tensor& history1,
    const ttnn::Tensor& history2,
    const ttnn::Tensor& taps,
    const ttnn::Tensor& gate_consts,
    uint32_t heads,
    float lower,
    float q_scale,
    uint32_t users,
    const std::optional<ttnn::Tensor>& keep,
    const std::optional<ttnn::Tensor>& saved,
    uint32_t tokens,
    bool pair_major,
    bool l1_outputs,
    bool split_channels,
    bool row_split,
    const std::optional<ttnn::Tensor>& history0_out,
    const std::optional<ttnn::Tensor>& history1_out,
    const std::optional<ttnn::Tensor>& history2_out,
    const std::optional<ttnn::Tensor>& saved_out) {
    return ttnn::prim::kda_decode_prepare(
        proj, fg, history0, history1, history2, taps, gate_consts, heads, lower, q_scale, users, keep, saved, tokens,
        pair_major, l1_outputs, split_channels, row_split, history0_out, history1_out, history2_out, saved_out);
}

}  // namespace ttnn::experimental::kda
