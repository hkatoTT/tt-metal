// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <api/tt-metalium/device.hpp>

#include "trace/trace_node.hpp"

namespace tt::tt_metal {

class Hal;

class SimpleTraceAllocator {
    // The fixture intentionally has a protected virtual destructor; clang-tidy diagnoses this declaration.
    // NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
    friend class SimpleTraceAllocatorFixture;
    friend class SimpleTraceAllocatorDeviceFixture;

public:
    struct RingbufferConfig {
        uint32_t start;
        uint32_t size;
    };

    explicit SimpleTraceAllocator(const std::vector<RingbufferConfig>& ringbuffer_configs) {
        region_allocators_.reserve(ringbuffer_configs.size());
        for (const auto& config : ringbuffer_configs) {
            region_allocators_.emplace_back(config.size, extra_data_);
            ringbuffer_starts_.push_back(config.start);
        }
    }

    void allocate_trace_programs(const Hal& hal, std::vector<TraceNode*>& trace_nodes);

private:
    struct ExtraData {
        enum { kNonBinary, kBinary, kNumTypes };
        // The index of the trace node when each type of data from this trace node is next used.
        std::array<std::optional<uint32_t>, kNumTypes> next_use_idx;
        // The sync value reached when this trace node finishes executing.
        uint32_t finished_sync_count;
    };

    struct MemoryUsage {
        // The last trace_idx that used this region. May be updated when the region is reused.
        uint32_t trace_idx;
        // The type of data in this region.
        uint32_t data_type;
        uint32_t size;
        uint64_t program_id;
    };
    static bool intersects(uint32_t begin_1, uint32_t size_1, uint32_t begin_2, uint32_t size_2) {
        return (begin_1 < begin_2 + size_2) && (begin_2 < begin_1 + size_1);
    }

    static std::optional<uint32_t> merge_syncs(std::optional<uint32_t> sync_1, std::optional<uint32_t> sync_2) {
        if (sync_1.has_value() && sync_2.has_value()) {
            return std::max(sync_1.value(), sync_2.value());
        }
        if (sync_1.has_value()) {
            return sync_1;
        }
        return sync_2;
    }

    class RegionAllocator {
        friend class SimpleTraceAllocatorFixture;
        friend class SimpleTraceAllocatorDeviceFixture;

    public:
        RegionAllocator(uint32_t ringbuffer_size, std::vector<ExtraData>& extra_data) :
            ringbuffer_size_(ringbuffer_size), extra_data_(extra_data) {}

        // Returns sync_idx and address. When `top_down` is true the candidate placements are
        // walked from the top of the ringbuffer downward (one slot ending at the top, plus one
        // slot ending just before each existing region). This is intended for short-lived data
        // (non-binary entries and binaries that won't be reused later in the trace) so that
        // long-lived cached binaries can stay packed at the bottom of the buffer.
        std::pair<std::optional<uint32_t>, std::optional<uint32_t>> allocate_region(
            uint32_t size, uint32_t trace_idx, uint32_t data_type, uint64_t program_id, bool top_down = false);

        void add_region(uint32_t data_type, uint64_t program_id, uint32_t addr) {
            program_ids_memory_map_[data_type][program_id] = addr;
        }

        void update_region_trace_idx(uint64_t region_addr, uint32_t trace_idx) {
            auto it = regions_.find(region_addr);
            if (it != regions_.end()) {
                it->second.trace_idx = trace_idx;
            }
        }

        std::optional<uint32_t> get_region(uint32_t data_type, uint64_t program_id) {
            auto it = program_ids_memory_map_[data_type].find(program_id);
            if (it != program_ids_memory_map_[data_type].end()) {
                return it->second;
            }
            return std::nullopt;
        }

        void reset_allocator() {
            for (auto& map : program_ids_memory_map_) {
                map.clear();
            }
            regions_.clear();
        }

        // TT_METAL_TRACE_ALLOC_BEAM (the planned placement, allocate_trace_programs_on_subdevice): every placement
        // `allocate_region`'s walk evaluates for a region of `size` (the same slots, the same Belady cost without the
        // stall term, the sync it would need), none when the slot holds this trace index's own data
        struct Candidate {
            uint32_t addr;
            float cost;
            std::optional<uint32_t> sync;
        };
        std::vector<Candidate> candidates(uint32_t size, uint32_t trace_idx, bool top_down) const;
        // ... and a region placed at `addr` (the regions it overlaps evicted, `sync` the sync it needs); false (nothing
        // changed) when that would overwrite this trace index's own data or not fit the ringbuffer
        bool place_at(
            uint32_t addr,
            uint32_t size,
            uint32_t trace_idx,
            uint32_t data_type,
            uint64_t program_id,
            std::optional<uint32_t>& sync);

    private:
        uint32_t ringbuffer_size_;
        std::array<std::map<uint64_t, uint32_t>, ExtraData::kNumTypes> program_ids_memory_map_;
        std::map<uint32_t, MemoryUsage> regions_;
        std::vector<ExtraData>& extra_data_;
    };

    void allocate_trace_programs_on_subdevice(
        const Hal& hal, std::vector<TraceNode*>& trace_nodes, SubDeviceId sub_device_id);

    // TT_METAL_TRACE_ALLOC_BEAM=B (B > 1): the Tensix binaries' placements planned over the whole trace by a beam
    // search of width B that keeps the states with the fewest bytes written right after a stall on a recent program
    // (weights 1 / 0.25 / 0.05 for the previous / 2 / 3 programs back), each node's non-binary entry placed by the
    // default walk.  One entry a node: kPlanFree (allocate by the default walk: cached, no binary, another sub-device),
    // kPlanReset (the ring reset, as the default walk does it), else the binary's address.
    static constexpr int64_t kPlanFree = -1;
    static constexpr int64_t kPlanReset = -2;
    std::vector<int64_t> plan_binaries(
        const Hal& hal, std::vector<TraceNode*>& trace_nodes, SubDeviceId sub_device_id, uint32_t beam);

    std::vector<ExtraData> extra_data_;
    std::vector<RegionAllocator> region_allocators_;
    std::vector<uint32_t> ringbuffer_starts_;
};

}  // namespace tt::tt_metal
