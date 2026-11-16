// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_finish.hpp"
#include "device/decode_finish_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor decode_finish(
    const ttnn::Tensor& o,
    const ttnn::Tensor& fg,
    const ttnn::Tensor& o_norm,
    uint32_t heads,
    float eps) {
    return ttnn::prim::kda_decode_finish(o, fg, o_norm, heads, eps);
}

}  // namespace ttnn::experimental::kda
