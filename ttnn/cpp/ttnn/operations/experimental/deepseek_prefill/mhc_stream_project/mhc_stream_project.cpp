// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_stream_project.hpp"
#include "device/mhc_stream_project_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_project {

ttnn::Tensor mhc_stream_project(const ttnn::Tensor& streams, const ttnn::Tensor& fn, uint32_t slices) {
    return ttnn::prim::mhc_stream_project(streams, fn, slices);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_stream_project
