// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "decode_finish_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct DecodeFinishProgramFactory {
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DecodeFinishParams& operation_attributes,
        const DecodeFinishTensorArgs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
