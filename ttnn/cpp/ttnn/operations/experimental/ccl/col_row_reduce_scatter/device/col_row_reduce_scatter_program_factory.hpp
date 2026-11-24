// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/experimental/ccl/col_row_reduce_scatter/device/col_row_reduce_scatter_device_operation_types.hpp"

namespace ttnn::experimental::prim {

namespace detail {

struct ColRowReduceScatterDescriptorAdapterOperation {
    using operation_attributes_t = ColRowReduceScatterParams;
    using tensor_args_t = ColRowReduceScatterInputs;
    using spec_return_value_t = ColRowReduceScatterResultSpec;
    using tensor_return_value_t = ColRowReduceScatterResult;
};

}  // namespace detail

struct ColRowReduceScatterProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ColRowReduceScatterParams& args,
        const ColRowReduceScatterInputs& tensor_args,
        ColRowReduceScatterResult& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// Tensor addresses are buffer bindings, patched by the framework on a cache hit.  The GlobalSemaphores are fixed for
// a program (their addresses are in the program hash), so nothing else is re-applied.
struct ColRowReduceScatterMeshWorkloadFactory {
    using descriptor_adapter_t =
        ttnn::device_operation::MeshDeviceOperationAdapter<detail::ColRowReduceScatterDescriptorAdapterOperation>::
            DescriptorMeshWorkloadAdapter<ColRowReduceScatterProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const ColRowReduceScatterParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ColRowReduceScatterInputs& tensor_args,
        ColRowReduceScatterResult& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ColRowReduceScatterParams& args,
        const ColRowReduceScatterInputs& tensor_args,
        ColRowReduceScatterResult& output_tensors);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<ColRowReduceScatterMeshWorkloadFactory>);

}  // namespace ttnn::experimental::prim
