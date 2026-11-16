// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_write.hpp"
#include "device/mhc_stream_write_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_write {

ttnn::Tensor mhc_stream_write(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& block_out,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    uint32_t n) {
    return ttnn::prim::mhc_stream_write(streams, block_out, post, comb, n);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_write
