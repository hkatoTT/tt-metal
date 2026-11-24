// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A long-lived custom kernel on DRAM cores' RISCs (DRISC), one program per chip of a mesh, launched and waited for
// through slow dispatch as the tensor prefetcher's (impl/buffers/tensor_prefetcher_manager.cpp): the GLM-5.3-Flash
// expert weight prefetch's reader (models/demos/blackhole/glm53_flash).  The kernel's L1 sits in the DRISC kernel
// working region (`GlmDriscRegion`); host and kernel talk through words there.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {
namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

// The DRISC kernel working region on every bank: (base address, size in bytes).
std::pair<uint32_t, uint32_t> GlmDriscRegion(distributed::MeshDevice& mesh_device);

// The DRAM-logical cores a bank's DRISC kernels may run on (the tensor prefetcher's two sender roles: the free
// subchannel, then the NOC1-endpoint one), bank by bank, as the mesh's first chip names them (the prefetcher checks
// that every chip names the same).
std::vector<std::vector<CoreCoord>> GlmDriscSenderCores(distributed::MeshDevice& mesh_device);

// Launch `kernel_path` on the logical DRAM cores `cores` of every chip of the mesh (NOC 0), with `compile_args` and
// per (chip, core) runtime args (`runtime_args[chip * cores.size() + i]`, chips in the mesh view's order).  Returns
// at once; the kernels run until they return.  One launch a mesh at a time.
void GlmDriscStart(
    distributed::MeshDevice& mesh_device,
    const std::string& kernel_path,
    const std::vector<CoreCoord>& cores,
    const std::vector<uint32_t>& compile_args,
    const std::vector<std::vector<uint32_t>>& runtime_args);

// The GDDR controller's MPFE port of a logical DRAM core (as the tensor prefetcher's get_mpfe_port: its subchannel
// + 1), on the mesh's first chip.
uint32_t GlmDriscMpfePort(distributed::MeshDevice& mesh_device, const CoreCoord& dram_core);

// The NoC (virtual) coordinates of logical DRAM cores, for worker kernels writing into their DRISCs' L1 (NoC 0).
std::vector<CoreCoord> GlmDriscNocCoords(distributed::MeshDevice& mesh_device, const std::vector<CoreCoord>& cores);

// Wait until every kernel of the mesh's launch has returned, and forget the launch.
void GlmDriscWait(distributed::MeshDevice& mesh_device);

// Words at `address` in a DRAM core's L1 (its DRISC's), on one chip (mesh view order) or, write only, every chip
// (chip < 0).
void GlmDriscWriteL1(
    distributed::MeshDevice& mesh_device,
    int chip,
    const CoreCoord& core,
    uint32_t address,
    const std::vector<uint32_t>& words);
std::vector<uint32_t> GlmDriscReadL1(
    distributed::MeshDevice& mesh_device, int chip, const CoreCoord& core, uint32_t address, uint32_t words);

}  // namespace experimental
}  // namespace tt::tt_metal
