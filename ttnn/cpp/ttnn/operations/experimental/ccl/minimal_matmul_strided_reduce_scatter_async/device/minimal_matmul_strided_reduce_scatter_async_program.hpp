// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minimal_matmul_strided_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MinimalMatmulStridedReduceScatterAsyncProgramFactory {
    struct shared_variables_t {
        operations::experimental::ccl::strided_reduce_scatter_async::detail::StridedReduceScatterProgramArtifacts
            rs_shared_variables;
        MinimalMatmulProgramFactory::shared_variables_t mm_shared_variables;
        // EXPERIMENT (hkato): TT_MMRS_RING=1 program (hk_ring_mmrs_program.cpp)
        bool hk_ring = false;
        std::vector<std::pair<CoreCoord, tt::tt_metal::KernelHandle>> hk_in0;
        std::vector<std::pair<CoreCoord, tt::tt_metal::KernelHandle>> hk_in1;
        std::vector<CoreCoord> hk_relay_cores;
        tt::tt_metal::KernelHandle hk_relay_kernel = 0;
        uint32_t hk_ready_sem_arg = 0;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensor);
};

// EXPERIMENT (hkato, optional, default off): TT_MMRS_RING=1, matmul + ring reduce-scatter + addcmul on the matmul cores
bool hk_mmrs_ring_enabled();
ttnn::device_operation::CachedProgram<MinimalMatmulStridedReduceScatterAsyncProgramFactory::shared_variables_t>
hk_ring_mmrs_program(
    const MinimalMatmulStridedReduceScatterAsyncParams& attrs,
    const ttnn::MeshCoordinate& coord,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& outputs);
void hk_ring_mmrs_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MinimalMatmulStridedReduceScatterAsyncProgramFactory::shared_variables_t& sv,
    const MinimalMatmulStridedReduceScatterAsyncParams& attrs,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& outputs);

}  // namespace ttnn::experimental::prim
