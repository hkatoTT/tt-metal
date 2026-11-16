// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "mhc_stream_read_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct MhcStreamReadProgramFactory {
    using tensor_return_value_t = std::array<Tensor, 3>;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MhcStreamReadParams& operation_attributes,
        const MhcStreamReadTensorArgs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

struct MhcStreamWriteReadProgramFactory {
    using tensor_return_value_t = std::array<Tensor, 4>;  // the new streams, post, comb, the normed block input

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MhcStreamReadParams& operation_attributes,
        const MhcStreamWriteReadTensorArgs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
