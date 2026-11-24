// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/glm_drisc.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <span>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_device_view_impl.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::experimental {

namespace glm_drisc_detail {

struct Launch {
    std::vector<IDevice*> devices;
    std::vector<std::unique_ptr<Program>> programs;
};

std::mutex& launches_mutex() {
    static std::mutex m;
    return m;
}
std::map<const distributed::MeshDevice*, Launch>& launches() {
    static std::map<const distributed::MeshDevice*, Launch> l;
    return l;
}

std::vector<IDevice*> mesh_devices(distributed::MeshDevice& mesh_device) {
    std::vector<IDevice*> out;
    for (auto* device : mesh_device.get_view().impl().get_devices()) {
        out.push_back(device);
    }
    return out;
}

}  // namespace glm_drisc_detail
using namespace glm_drisc_detail;

std::pair<uint32_t, uint32_t> GlmDriscRegion(distributed::MeshDevice& mesh_device) {
    auto& arena = mesh_device.impl().drisc_l1_arena();
    return {static_cast<uint32_t>(arena.kernel_working_region_base()), arena.kernel_working_region_size()};
}

std::vector<std::vector<CoreCoord>> GlmDriscSenderCores(distributed::MeshDevice& mesh_device) {
    auto devices = mesh_devices(mesh_device);
    const uint32_t banks = devices.front()->num_dram_channels();
    std::vector<std::vector<CoreCoord>> out;
    for (uint32_t b = 0; b < banks; ++b) {
        out.push_back(mesh_device.impl().dram_sender_logical_cores(devices.front(), b));
    }
    return out;
}

void GlmDriscStart(
    distributed::MeshDevice& mesh_device,
    const std::string& kernel_path,
    const std::vector<CoreCoord>& cores,
    const std::vector<uint32_t>& compile_args,
    const std::vector<std::vector<uint32_t>>& runtime_args) {
    std::lock_guard<std::mutex> lock(launches_mutex());
    TT_FATAL(!launches().contains(&mesh_device), "GlmDriscStart: a launch is still running on this mesh");
    Launch launch;
    launch.devices = mesh_devices(mesh_device);
    TT_FATAL(
        runtime_args.size() == launch.devices.size() * cores.size(),
        "GlmDriscStart: {} runtime arg lists for {} chips x {} cores",
        runtime_args.size(),
        launch.devices.size(),
        cores.size());
    for (uint32_t d = 0; d < launch.devices.size(); ++d) {
        auto program = std::make_unique<Program>();
        for (uint32_t i = 0; i < cores.size(); ++i) {
            KernelHandle kernel = CreateKernel(
                *program,
                kernel_path,
                cores[i],
                DramConfig{.noc = NOC::NOC_0, .compile_args = compile_args, .defines = {{"WATCHER_NOINLINE", "1"}}});
            SetRuntimeArgs(*program, kernel, cores[i], runtime_args[d * cores.size() + i]);
        }
        launch.programs.push_back(std::move(program));
    }
    for (uint32_t d = 0; d < launch.devices.size(); ++d) {
        launch.programs[d]->impl().compile(launch.devices[d], /*force_slow_dispatch=*/true);
        slow_dispatch::WriteRuntimeArgsToDevice(*launch.devices[d], *launch.programs[d], /*force_slow_dispatch=*/true);
        slow_dispatch::LaunchProgramAsync(*launch.devices[d], *launch.programs[d], /*force_slow_dispatch=*/true);
    }
    launches().emplace(&mesh_device, std::move(launch));
}

uint32_t GlmDriscMpfePort(distributed::MeshDevice& mesh_device, const CoreCoord& dram_core) {
    auto devices = mesh_devices(mesh_device);
    const auto& soc_desc =
        MetalContext::instance(mesh_device.impl().get_context_id()).get_cluster().get_soc_desc(devices.front()->id());
    const CoreCoord physical = soc_desc.get_physical_dram_core_from_logical(dram_core);
    const tt::umd::CoreCoord subchannel = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(physical.x, physical.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED),
        tt::CoordSystem::LOGICAL);
    return subchannel.y + 1;
}

std::vector<CoreCoord> GlmDriscNocCoords(distributed::MeshDevice& mesh_device, const std::vector<CoreCoord>& cores) {
    auto devices = mesh_devices(mesh_device);
    std::vector<CoreCoord> out;
    for (const auto& c : cores) {
        out.push_back(devices.front()->virtual_core_from_logical_core(c, CoreType::DRAM));
    }
    return out;
}

void GlmDriscWait(distributed::MeshDevice& mesh_device) {
    std::lock_guard<std::mutex> lock(launches_mutex());
    auto it = launches().find(&mesh_device);
    if (it == launches().end()) {
        return;
    }
    for (uint32_t d = 0; d < it->second.devices.size(); ++d) {
        slow_dispatch::WaitProgramDone(*it->second.devices[d], *it->second.programs[d]);
    }
    launches().erase(it);
}

void GlmDriscWriteL1(
    distributed::MeshDevice& mesh_device,
    int chip,
    const CoreCoord& core,
    uint32_t address,
    const std::vector<uint32_t>& words) {
    // (the DRAM core's L1 through its NoC window, address + the HAL's L1 NoC offset, as slow dispatch writes a DRAM
    // kernel's config: the host's NoC sanitizer takes that range; the bare L1 address it calls a bad DRAM hw core
    // address, so a run with the watcher on would stop there)
    auto devices = mesh_devices(mesh_device);
    const auto& ctx = MetalContext::instance(mesh_device.impl().get_context_id());
    const uint64_t at = ctx.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM) + static_cast<uint64_t>(address);
    for (int d = 0; d < static_cast<int>(devices.size()); ++d) {
        if (chip < 0 || chip == d) {
            const CoreCoord v = devices[d]->virtual_core_from_logical_core(core, CoreType::DRAM);
            ctx.get_cluster().write_core(
                words.data(), static_cast<uint32_t>(words.size() * sizeof(uint32_t)), tt_cxy_pair(devices[d]->id(), v), at);
        }
    }
}

std::vector<uint32_t> GlmDriscReadL1(
    distributed::MeshDevice& mesh_device, int chip, const CoreCoord& core, uint32_t address, uint32_t words) {
    auto devices = mesh_devices(mesh_device);
    TT_FATAL(chip >= 0 && chip < static_cast<int>(devices.size()), "GlmDriscReadL1: no chip {}", chip);
    const auto& ctx = MetalContext::instance(mesh_device.impl().get_context_id());
    const uint64_t at = ctx.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM) + static_cast<uint64_t>(address);
    const CoreCoord v = devices[chip]->virtual_core_from_logical_core(core, CoreType::DRAM);
    ctx.get_cluster().l1_barrier(devices[chip]->id());
    std::vector<uint32_t> out;
    ctx.get_cluster().read_core(out, words * sizeof(uint32_t), tt_cxy_pair(devices[chip]->id(), v), at);
    return out;
}

}  // namespace tt::tt_metal::experimental
