// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "simple_trace_allocator.hpp"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <set>
#include <string>
#include <tuple>

#include "hal/generated/dev_msgs.hpp"
#include "kernels/kernel.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal {

namespace {
// TT_METAL_TRACE_ALLOC_LOG=<path>: one line a traced program (appended to the file) with this allocator's choices for
// the Tensix kernel config ring - the sizes, whether the binary was cached, how many programs back it syncs and the
// stall kind - for studying the gaps between programs; nothing changes in what is dispatched
std::FILE* trace_alloc_log() {
    static const char* path = std::getenv("TT_METAL_TRACE_ALLOC_LOG");
    if (path == nullptr || *path == '\0') {
        return nullptr;
    }
    return std::fopen(path, "a");
}
}  // namespace

std::pair<std::optional<uint32_t>, std::optional<uint32_t>> SimpleTraceAllocator::RegionAllocator::allocate_region(
    uint32_t size, uint32_t trace_idx, uint32_t data_type, uint64_t program_id, bool top_down) {
    std::optional<uint32_t> best_addr;
    float best_cost = std::numeric_limits<float>::infinity();
    std::optional<uint32_t> best_region_sync_idx;
    if (size == 0) {
        return {std::nullopt, 0};
    }
    if (top_down && size > ringbuffer_size_) {
        return {std::nullopt, std::nullopt};
    }

    // Set of regions that have no future uses, so they can be evicted to avoid cluttering up regions_.
    std::set<uint32_t> marked_for_deletion;

    // Once we've filled up the entire launch message buffer, we'll sync on that rather than on any older regions.
    constexpr uint32_t max_stall_history_size = dev_msgs::launch_msg_buffer_num_entries;

    // Evaluates a single candidate placement at `addr`, optionally pruning the inner region scan
    // by a `scan_begin` cursor (used in the bottom-up walk to avoid re-scanning regions that
    // ended at or before previous placements). Updates best_addr/best_cost/best_region_sync_idx
    // and marked_for_deletion in place. Returns true when the cost reached zero so the caller
    // can stop searching.
    auto evaluate = [&](uint32_t addr, std::map<uint32_t, MemoryUsage>::const_iterator scan_begin) {
        float cost = 0;
        std::optional<uint32_t> region_sync_idx;
        bool now_in_use = false;
        for (auto it = scan_begin; it != regions_.end(); ++it) {
            auto region = *it;
            if (region.first >= addr + size) {
                break;
            }
            if (intersects(addr, size, region.first, region.second.size)) {
                if (region.second.trace_idx == trace_idx) {
                    now_in_use = true;
                    break;
                }
                auto& next_use_idx = extra_data_[region.second.trace_idx].next_use_idx[region.second.data_type];
                if (next_use_idx.has_value()) {
                    if (*next_use_idx == trace_idx) {
                        // Really try to avoid evicting a buffer that will be used by this program.
                        constexpr uint32_t current_node_eviction_penalty = 1000000000;
                        cost += current_node_eviction_penalty;
                    } else {
                        // Similar to Belady's algorithm, we want to evict the region that will be used the farthest in
                        // the future, so cost is inversely proportional to distance to next use. Also take into account
                        // the size, since that's roughly proportional to the cost of adding the region back in. We
                        // could instead divide by the size, similar to the Belady-size algorithm, but has worked worse
                        // in simulation.
                        cost += region.second.size * 1.0f / (*next_use_idx - trace_idx);
                    }
                } else if (trace_idx - region.second.trace_idx > max_stall_history_size) {
                    // Region has no future uses and is too old to worry about stalls.
                    marked_for_deletion.insert(region.first);
                }
                region_sync_idx = merge_syncs(region_sync_idx, region.second.trace_idx);
            }
        }
        if (now_in_use) {
            return false;
        }
        if (region_sync_idx.has_value()) {
            // Avoid evicting something that was last used recently, as that can cause a stall that is very bad for
            // performance. This is critical for avoiding gaps between ops, so it's given a very high cost (the
            // highest cost for a program is normally around 10,000).
            constexpr uint32_t desired_write_ahead = std::min(dev_msgs::launch_msg_buffer_num_entries, 7u);
            constexpr float stall_badness = 100000000;
            static_assert(
                max_stall_history_size > desired_write_ahead,
                "max_history_size must be greater than desired_write_ahead");
            int region_idx_diff = trace_idx - *region_sync_idx;
            if (region_idx_diff < desired_write_ahead) {
                // Stall badness is exponential.
                cost += stall_badness * (1 << (desired_write_ahead - region_idx_diff));
            }
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_addr = addr;
            best_region_sync_idx = region_sync_idx;
        }
        return cost == 0;
    };

    // Iterate over possible placements. One of these placements must be the best, since any other
    // placement would overlap the same or a larger number of allocations by moving it to one of
    // these slots. TODO: sweepline algorithm, so the best position can be calculated in linear
    // time relative to the number of regions.
    if (top_down) {
        auto first_top_down_scan_begin = [&](const std::map<uint32_t, MemoryUsage>::const_reverse_iterator& reverse_begin,
                                             std::map<uint32_t, MemoryUsage>::const_iterator default_begin,
                                             uint32_t addr) {
            auto scan_begin = default_begin;
            for (auto scan = reverse_begin; scan != regions_.crend(); ++scan) {
                if (scan->first + scan->second.size <= addr) {
                    break;
                }
                scan_begin = std::prev(scan.base());
            }
            return scan_begin;
        };

        // Top-down walk: try the slot ending exactly at ringbuffer_size_, then the slots ending
        // just before each existing region, in descending order of address.
        uint32_t top_addr = ringbuffer_size_ - size;
        bool stop = evaluate(top_addr, first_top_down_scan_begin(regions_.crbegin(), regions_.cend(), top_addr));
        if (!stop) {
            for (auto rit = regions_.crbegin(); rit != regions_.crend(); ++rit) {
                if (rit->first < size) {
                    // Slot would extend below the start of the buffer.
                    break;
                }
                uint32_t addr = rit->first - size;
                if (addr == top_addr) {
                    continue;
                }
                auto after_candidate = std::prev(rit.base());
                auto scan_begin = first_top_down_scan_begin(std::next(rit), after_candidate, addr);
                if (evaluate(addr, scan_begin)) {
                    break;
                }
            }
        }
    } else {
        // Bottom-up walk: try the very beginning of the ringbuffer and the slots starting
        // immediately after each existing region.
        uint32_t addr = 0;
        auto outer_it = regions_.begin();
        while (true) {
            if (addr + size > ringbuffer_size_) {
                break;
            }
            // outer_it must be the first region that could possibly overlap [addr, addr + size),
            // since in the last iteration we selected addr as the first address after the old
            // version of outer_it.
            if (evaluate(addr, outer_it)) {
                break;
            }
            if (outer_it == regions_.end()) {
                break;
            }
            addr = outer_it->first + outer_it->second.size;
            outer_it++;
        }
    }

    for (const auto& addr : marked_for_deletion) {
        auto it = regions_.find(addr);
        program_ids_memory_map_[it->second.data_type].erase(it->second.program_id);
        regions_.erase(it);
    }

    if (!best_addr.has_value()) {
        return {std::nullopt, best_addr};
    }

    // Evict overlapped regions.
    auto it = regions_.begin();
    while (it != regions_.end()) {
        if (intersects(*best_addr, size, it->first, it->second.size)) {
            program_ids_memory_map_[it->second.data_type].erase(it->second.program_id);
            it = regions_.erase(it);
        } else {
            ++it;
        }
    }
    regions_[*best_addr] = {trace_idx, data_type, size, program_id};
    return {best_region_sync_idx, best_addr};
}

std::vector<SimpleTraceAllocator::RegionAllocator::Candidate> SimpleTraceAllocator::RegionAllocator::candidates(
    uint32_t size, uint32_t trace_idx, bool top_down) const {
    std::vector<Candidate> out;
    if (size == 0 || size > ringbuffer_size_) {
        return out;
    }
    auto evaluate = [&](uint32_t addr) {
        float cost = 0;
        std::optional<uint32_t> sync;
        for (const auto& [raddr, region] : regions_) {
            if (raddr >= addr + size) {
                break;
            }
            if (intersects(addr, size, raddr, region.size)) {
                if (region.trace_idx == trace_idx) {
                    return;  // (this node's own data)
                }
                const auto& next_use_idx = extra_data_[region.trace_idx].next_use_idx[region.data_type];
                if (next_use_idx.has_value()) {
                    cost +=
                        *next_use_idx == trace_idx ? 1000000000.0f : region.size * 1.0f / (*next_use_idx - trace_idx);
                }
                sync = merge_syncs(sync, region.trace_idx);
            }
        }
        out.push_back({addr, cost, sync});
    };
    if (top_down) {  // (allocate_region's walks)
        uint32_t top_addr = ringbuffer_size_ - size;
        evaluate(top_addr);
        for (auto rit = regions_.crbegin(); rit != regions_.crend(); ++rit) {
            if (rit->first < size) {
                break;
            }
            uint32_t addr = rit->first - size;
            if (addr != top_addr) {
                evaluate(addr);
            }
        }
    } else {
        uint32_t addr = 0;
        auto it = regions_.begin();
        while (addr + size <= ringbuffer_size_) {
            evaluate(addr);
            if (it == regions_.end()) {
                break;
            }
            addr = it->first + it->second.size;
            ++it;
        }
    }
    return out;
}

bool SimpleTraceAllocator::RegionAllocator::place_at(
    uint32_t addr,
    uint32_t size,
    uint32_t trace_idx,
    uint32_t data_type,
    uint64_t program_id,
    std::optional<uint32_t>& sync) {
    if (addr + size > ringbuffer_size_) {
        return false;
    }
    for (const auto& [raddr, region] : regions_) {
        if (raddr >= addr + size) {
            break;
        }
        if (intersects(addr, size, raddr, region.size) && region.trace_idx == trace_idx) {
            return false;
        }
    }
    sync = std::nullopt;
    auto it = regions_.begin();
    while (it != regions_.end()) {
        if (intersects(addr, size, it->first, it->second.size)) {
            sync = merge_syncs(sync, it->second.trace_idx);
            program_ids_memory_map_[it->second.data_type].erase(it->second.program_id);
            it = regions_.erase(it);
        } else {
            ++it;
        }
    }
    regions_[addr] = {trace_idx, data_type, size, program_id};
    return true;
}

std::vector<int64_t> SimpleTraceAllocator::plan_binaries(
    const Hal& hal, std::vector<TraceNode*>& trace_nodes, SubDeviceId sub_device_id, uint32_t beam) {
    // The Tensix ring only, and only when no other core type has kernels in the trace (their syncs are not modelled)
    std::optional<uint32_t> tensix;
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        if (hal.get_programmable_core_type(index) == HalProgrammableCoreType::TENSIX) {
            tensix = index;
        }
    }
    if (!tensix.has_value() || !hal.get_core_kernel_stored_in_config_buffer(HalProgrammableCoreType::TENSIX)) {
        return {};
    }
    for (auto* node : trace_nodes) {
        if (node->sub_device_id != sub_device_id) {
            continue;
        }
        for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
            if (index != *tensix && !node->program->get_kernel_groups(index).empty()) {
                return {};
            }
        }
    }
    constexpr uint32_t max_queued_programs = dev_msgs::launch_msg_buffer_num_entries - 1;
    constexpr uint32_t desired_write_ahead = std::min(dev_msgs::launch_msg_buffer_num_entries, 7u);
    const uint32_t children = std::min<uint32_t>(4, beam);
    struct State {
        RegionAllocator alloc;
        std::deque<size_t> window;
        std::optional<uint32_t> last_stall;
        bool first_dispatched;
        double cost;
    };
    std::vector<State> states;
    states.push_back({region_allocators_[*tensix], {}, std::nullopt, false, 0.0});
    // per node, per kept state: (its parent among the states kept at the node before, its choice)
    std::vector<std::vector<std::pair<uint32_t, int64_t>>> history(trace_nodes.size());
    for (size_t i = 0; i < trace_nodes.size(); i++) {
        auto& node = *trace_nodes[i];
        if (node.sub_device_id != sub_device_id) {
            continue;
        }
        ProgramConfig& program_config = node.program->get_program_config(*tensix);
        const uint32_t nb_size = program_config.kernel_text_offset;
        const uint32_t bin_size = program_config.kernel_text_size;
        const uint64_t pid = node.program->get_id();
        const bool bin_top_down = !extra_data_[i].next_use_idx[ExtraData::kBinary].has_value();
        std::vector<State> next;
        std::vector<std::pair<uint32_t, int64_t>> origin;
        for (uint32_t s = 0; s < states.size(); s++) {
            const State& st = states[s];
            RegionAllocator base = st.alloc;
            auto [nb_sync, nb_addr] = base.allocate_region(nb_size, i, ExtraData::kNonBinary, pid, /*top_down=*/true);
            struct Choice {
                RegionAllocator alloc;
                std::optional<uint32_t> nb_sync, bin_sync;
                int64_t plan;
                bool sent;
            };
            std::vector<Choice> choices;
            if (bin_size == 0) {
                choices.push_back({base, nb_sync, std::nullopt, kPlanFree, false});
            } else if (auto cached = base.get_region(ExtraData::kBinary, pid)) {
                base.update_region_trace_idx(*cached, i);
                choices.push_back({base, nb_sync, std::nullopt, kPlanFree, false});
            } else {
                auto cands = base.candidates(bin_size, i, bin_top_down);
                if (cands.empty()) {  // (the main loop's reset)
                    RegionAllocator reset = base;
                    reset.reset_allocator();
                    reset.allocate_region(nb_size, i, ExtraData::kNonBinary, pid, /*top_down=*/true);
                    auto res = reset.allocate_region(bin_size, i, ExtraData::kBinary, pid, bin_top_down);
                    if (!res.second.has_value() || st.window.empty()) {
                        return {};
                    }
                    reset.add_region(ExtraData::kBinary, pid, *res.second);
                    std::optional<uint32_t> last = static_cast<uint32_t>(st.window.back());
                    choices.push_back(
                        {reset, merge_syncs(nb_sync, last), merge_syncs(res.first, last), kPlanReset, true});
                } else {
                    // the oldest sync first (none, or 7 or more programs back, ties), then the Belady cost
                    auto key = [&](const RegionAllocator::Candidate& c) {
                        uint32_t recency = c.sync.has_value() ? std::min<uint32_t>(i - *c.sync, desired_write_ahead)
                                                              : desired_write_ahead;
                        return std::make_pair(-static_cast<int>(recency), c.cost);
                    };
                    std::stable_sort(
                        cands.begin(), cands.end(), [&](const auto& a, const auto& b) { return key(a) < key(b); });
                    for (uint32_t c = 0; c < std::min<size_t>(children, cands.size()); c++) {
                        RegionAllocator placed = base;
                        std::optional<uint32_t> sync;
                        if (!placed.place_at(cands[c].addr, bin_size, i, ExtraData::kBinary, pid, sync)) {
                            continue;
                        }
                        placed.add_region(ExtraData::kBinary, pid, cands[c].addr);
                        choices.push_back({placed, nb_sync, sync, static_cast<int64_t>(cands[c].addr), true});
                    }
                    if (choices.empty()) {
                        return {};
                    }
                }
            }
            for (auto& ch : choices) {
                std::deque<size_t> window = st.window;
                std::optional<uint32_t> bin_sync = ch.bin_sync;
                if (window.size() >= max_queued_programs) {
                    bin_sync = merge_syncs(bin_sync, static_cast<uint32_t>(window.front()));
                }
                std::optional<uint32_t> last = st.last_stall;
                double cost = st.cost;
                const bool need_nb = ch.nb_sync.has_value() && (!last.has_value() || *ch.nb_sync > *last);
                const bool need_bin = bin_sync.has_value() && (!last.has_value() || *bin_sync > *last);
                if (need_nb || need_bin) {
                    uint32_t combined = *merge_syncs(ch.nb_sync, bin_sync);
                    const size_t ago = i - combined;
                    const double weight = ago == 1 ? 1.0 : ago == 2 ? 0.25 : ago == 3 ? 0.05 : 0.0;
                    cost += weight * ((ch.sent ? bin_size : 0) + (need_nb ? nb_size : 0) + 512);
                    last = combined;
                }
                window.push_back(i);
                if (window.size() > max_queued_programs) {
                    window.pop_front();
                }
                next.push_back({std::move(ch.alloc), std::move(window), last, true, cost});
                origin.emplace_back(s, ch.plan);
            }
        }
        std::vector<uint32_t> order(next.size());
        for (uint32_t k = 0; k < order.size(); k++) {
            order[k] = k;
        }
        std::stable_sort(
            order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return next[a].cost < next[b].cost; });
        order.resize(std::min<size_t>(order.size(), beam));
        states.clear();
        for (uint32_t k : order) {
            states.push_back(std::move(next[k]));
            history[i].push_back(origin[k]);
        }
    }
    // the best final state's choices, back to the first node
    std::vector<int64_t> plan(trace_nodes.size(), kPlanFree);
    uint32_t s = 0;
    for (size_t i = trace_nodes.size(); i-- > 0;) {
        if (history[i].empty()) {
            continue;
        }
        plan[i] = history[i][s].second;
        s = history[i][s].first;
    }
    return plan;
}

void SimpleTraceAllocator::allocate_trace_programs(const Hal& hal, std::vector<TraceNode*>& trace_nodes) {
    std::map<uint64_t, uint32_t> program_ids_use_map;
    extra_data_.resize(trace_nodes.size());

    std::set<SubDeviceId> sub_device_ids;
    for (size_t i = trace_nodes.size(); i-- > 0;) {
        auto& node = *trace_nodes[i];
        auto it = program_ids_use_map.find(node.program->get_id());
        if (it != program_ids_use_map.end()) {
            // Binary is reused, but the nonbinary is not.
            extra_data_[i].next_use_idx[ExtraData::kBinary] = it->second;
        }
        program_ids_use_map[node.program->get_id()] = static_cast<uint32_t>(i);
        sub_device_ids.insert(node.sub_device_id);
    }
    for (const auto& sub_device_id : sub_device_ids) {
        for (auto& allocator : region_allocators_) {
            allocator.reset_allocator();
        }
        allocate_trace_programs_on_subdevice(hal, trace_nodes, sub_device_id);
    }
}

void SimpleTraceAllocator::allocate_trace_programs_on_subdevice(
    const Hal& hal, std::vector<TraceNode*>& trace_nodes, SubDeviceId sub_device_id) {
    uint32_t expected_workers_completed = 0;
    uint32_t programmable_core_count = hal.get_programmable_core_type_count();
    // For core types where the binary goes to a fixed L1 address (not stored in the config buffer),
    // track the last trace index that used each core type so we can sync before overwriting.
    std::vector<std::optional<uint32_t>> last_fixed_addr_sync_idx(programmable_core_count);
    // Last trace index at which each CrossNodeDFB-using program was launched. Every launch
    // rewrites that program's dedicated config Buffer (resetting credits), so a re-launch
    // within the trace must wait until the previous launch of the same program completes.
    std::map<uint64_t, uint32_t> last_cross_node_launch_idx;
    bool first_program_dispatched = false;
    std::optional<uint32_t> last_stall_idx;
    std::deque<size_t> subdevice_launch_window;
    std::FILE* log = trace_alloc_log();
    // TT_METAL_TRACE_ALLOC_BEAM=B (B > 1, read at each trace): the Tensix binaries placed by `plan_binaries`
    std::vector<int64_t> plan;
    if (const char* beam_env = std::getenv("TT_METAL_TRACE_ALLOC_BEAM"); beam_env != nullptr) {
        const long beam = std::strtol(beam_env, nullptr, 10);
        if (beam > 1) {
            plan = plan_binaries(hal, trace_nodes, sub_device_id, static_cast<uint32_t>(std::min(beam, 64L)));
        }
    }
    if (log != nullptr) {
        std::fprintf(
            log,
            "# trace alloc: sub-device %u, %zu nodes\n",
            static_cast<unsigned>(*sub_device_id),
            trace_nodes.size());
    }

    for (size_t i = 0; i < trace_nodes.size(); i++) {
        auto& node = *trace_nodes[i];
        if (node.sub_device_id != sub_device_id) {
            continue;
        }

        std::optional<uint32_t> nonbinary_sync_idx;
        std::optional<uint32_t> binary_sync_idx;
        // Reinitialize dispatch_metadata. TraceNodes may be shared across device ranges in a mesh
        // trace, so the caller (record_end) relies on this reinitialization to clear stale values
        // from a previous device range's processing.
        node.dispatch_metadata = TraceDispatchMetadata{};
        node.dispatch_metadata.binary_kernel_config_addrs.resize(programmable_core_count);
        node.dispatch_metadata.nonbinary_kernel_config_addrs.resize(programmable_core_count);

        bool all_binaries_cached = true;
        // (TT_METAL_TRACE_ALLOC_LOG: the Tensix ring's choices)
        struct {
            uint32_t nb = 0, bin = 0, nb_addr = 0, bin_addr = 0;
            int cached = -1, reset = 0;
            std::optional<uint32_t> nb_sync, bin_sync;
        } tlog;

        for (uint32_t index = 0; index < programmable_core_count; index++) {
            auto core_type = hal.get_programmable_core_type(index);
            if (!hal.has_programmable_core_type(core_type) || core_type == HalProgrammableCoreType::IDLE_ETH) {
                continue;
            }
            ProgramConfig& program_config = node.program->get_program_config(index);
            bool binary_in_config = hal.get_core_kernel_stored_in_config_buffer(core_type);

            // Only Tensix has a dedicated binary write offset in the dispatcher, so only Tensix
            // can place its binary at a separately-allocated address. For other core types the
            // dispatcher writes the binary at a fixed offset from the non-binary base, so the
            // entire config (non-binary + binary) must be allocated as one contiguous region.
            bool has_separate_binary_offset = (core_type == HalProgrammableCoreType::TENSIX);

            uint32_t non_binary_size;
            if (has_separate_binary_offset && binary_in_config) {
                non_binary_size = program_config.kernel_text_offset;
            } else {
                non_binary_size = node.program->get_program_config_sizes()[index];
            }
            uint32_t binary_size = program_config.kernel_text_size;
            auto& allocator = region_allocators_[index];

            uint64_t pid = node.program->get_id();
            // Non-binary entries are never reused, so place them top-down to leave the bottom of
            // the ringbuffer free for cached binaries.
            auto [rta_sync_idx, rta_addr] =
                allocator.allocate_region(non_binary_size, i, ExtraData::kNonBinary, pid, /*top_down=*/true);

            nonbinary_sync_idx = merge_syncs(nonbinary_sync_idx, rta_sync_idx);

            uint32_t binary_addr = 0;

            if (has_separate_binary_offset && binary_in_config && binary_size > 0) {
                // Binary is stored in the config buffer and the dispatcher has a dedicated write
                // offset for this core type; allocate separately so it can be cached across
                // invocations of the same program.
                if (auto mem_addr = allocator.get_region(ExtraData::kBinary, pid)) {
                    binary_addr = *mem_addr;
                    allocator.update_region_trace_idx(*mem_addr, i);
                    if (core_type == HalProgrammableCoreType::TENSIX) {
                        tlog.cached = 1;
                    }
                } else {
                    if (core_type == HalProgrammableCoreType::TENSIX) {
                        tlog.cached = 0;
                    }
                    all_binaries_cached = false;
                    // If this binary won't be used again later in the trace, treat it as
                    // short-lived and pack it top-down. Otherwise allocate bottom-up so it can
                    // remain cached for future invocations of the same program.
                    bool binary_top_down = !extra_data_[i].next_use_idx[ExtraData::kBinary].has_value();
                    std::pair<std::optional<uint32_t>, std::optional<uint32_t>> res;
                    std::optional<uint32_t> planned_sync;
                    if (core_type == HalProgrammableCoreType::TENSIX && i < plan.size() && plan[i] >= 0 &&
                        allocator.place_at(
                            static_cast<uint32_t>(plan[i]), binary_size, i, ExtraData::kBinary, pid, planned_sync)) {
                        res = {planned_sync, static_cast<uint32_t>(plan[i])};  // (the plan's place, its sync)
                    } else {
                        res = allocator.allocate_region(binary_size, i, ExtraData::kBinary, pid, binary_top_down);
                    }
                    if (!res.second.has_value()) {
                        // Clear the allocator and try again. Should succeed unless the total size
                        // of the program is larger than the config buffer.
                        allocator.reset_allocator();
                        tlog.reset = 1;
                        std::tie(rta_sync_idx, rta_addr) = allocator.allocate_region(
                            non_binary_size, i, ExtraData::kNonBinary, pid, /*top_down=*/true);
                        res = allocator.allocate_region(binary_size, i, ExtraData::kBinary, pid, binary_top_down);
                        TT_ASSERT(res.second.has_value(), "Failed to allocate binary region");
                        TT_ASSERT(
                            !subdevice_launch_window.empty(),
                            "Failed to allocate binary region on first program on sub-device");
                        auto last_subdevice_idx = static_cast<uint32_t>(subdevice_launch_window.back());
                        binary_sync_idx = merge_syncs(binary_sync_idx, last_subdevice_idx);
                        nonbinary_sync_idx = merge_syncs(nonbinary_sync_idx, last_subdevice_idx);
                    } else {
                        binary_sync_idx = merge_syncs(res.first, binary_sync_idx);
                    }
                    binary_addr = *res.second;
                    allocator.add_region(ExtraData::kBinary, pid, binary_addr);
                }
            } else if (!binary_in_config && !node.program->get_kernel_groups(index).empty()) {
                // Binary goes to a fixed L1 address (not in the config buffer). Must sync with the
                // previous program that used this fixed address before overwriting it.
                all_binaries_cached = false;
                if (last_fixed_addr_sync_idx[index].has_value()) {
                    binary_sync_idx = merge_syncs(binary_sync_idx, last_fixed_addr_sync_idx[index]);
                }
                last_fixed_addr_sync_idx[index] = i;
            } else if (!has_separate_binary_offset && !node.program->get_kernel_groups(index).empty()) {
                // Binary is included in the non-binary allocation (no separate binary write offset
                // for this core type). The binary is always re-sent as part of the full config.
                all_binaries_cached = false;
            }

            if (log != nullptr && core_type == HalProgrammableCoreType::TENSIX) {
                tlog.nb = non_binary_size;
                tlog.bin = binary_size;
                tlog.nb_addr = rta_addr.value_or(0);
                tlog.bin_addr = binary_addr;
                tlog.nb_sync = rta_sync_idx;
                tlog.bin_sync = binary_sync_idx;
            }
            TT_ASSERT(rta_addr.has_value(), "Failed to allocate non-binary region");
            node.dispatch_metadata.nonbinary_kernel_config_addrs[index] = {
                .addr = *rta_addr + ringbuffer_starts_[index]};
            node.dispatch_metadata.binary_kernel_config_addrs[index] = {
                .addr = binary_addr + ringbuffer_starts_[index]};
        }

        // Mirror the non-trace CrossNode ordering (see FDMeshCommandQueue::enqueue_program):
        // the config Buffer rewrite goes out with the non-binary config data, so it must
        // stall_first until the previous launch of the same program completes.
        if (!node.program->get_per_core_cross_node_dfbs().empty()) {
            auto [it, inserted] =
                last_cross_node_launch_idx.try_emplace(node.program->get_id(), static_cast<uint32_t>(i));
            if (!inserted) {
                nonbinary_sync_idx = merge_syncs(nonbinary_sync_idx, it->second);
                it->second = static_cast<uint32_t>(i);
            }
        }

        node.dispatch_metadata.send_binary = !all_binaries_cached;
        extra_data_[i].finished_sync_count = expected_workers_completed + node.num_workers;

        // Subtract 1 because we don't want to overwrite watcher data for the last program to complete executing.
        constexpr uint32_t max_queued_programs = dev_msgs::launch_msg_buffer_num_entries - 1;

        // Do adjustments to the sync index to ensure we don't overflow the worker launch message buffer. The launch
        // message buffer is written to after the binary.
        if (subdevice_launch_window.size() >= max_queued_programs) {
            binary_sync_idx = merge_syncs(binary_sync_idx, static_cast<uint32_t>(subdevice_launch_window.front()));
        }

        if (!first_program_dispatched) {
            // The first program to be dispatched should stall on 0, since there may be undetermined commands in the
            // ringbuffer before this we want to wait for. In particular in the mesh device case we can add go messages
            // for unused nodes before replaying the trace.
            node.dispatch_metadata.sync_count = 0;
            node.dispatch_metadata.stall_first = true;
            first_program_dispatched = true;
        }

        // Only one sync count can currently be specified, so pick the latest one.
        // nonbinary sync requires stall_first (before writing config data); binary-only sync uses
        // stall_before_program (after config, before binary/launch).
        bool needs_nonbinary_sync =
            nonbinary_sync_idx.has_value() && (!last_stall_idx.has_value() || *nonbinary_sync_idx > *last_stall_idx);
        bool needs_binary_sync =
            binary_sync_idx.has_value() && (!last_stall_idx.has_value() || *binary_sync_idx > *last_stall_idx);
        if (needs_nonbinary_sync || needs_binary_sync) {
            uint32_t combined_sync_idx = *merge_syncs(nonbinary_sync_idx, binary_sync_idx);
            node.dispatch_metadata.sync_count = extra_data_[combined_sync_idx].finished_sync_count;
            if (needs_nonbinary_sync) {
                node.dispatch_metadata.stall_first = true;
            } else {
                node.dispatch_metadata.stall_before_program = true;
            }
            last_stall_idx = combined_sync_idx;
        }
        if (log != nullptr) {
            std::string names;
            for (uint32_t index = 0; index < programmable_core_count; index++) {
                if (hal.get_programmable_core_type(index) != HalProgrammableCoreType::TENSIX) {
                    continue;
                }
                for (const auto& [handle, kernel] : node.program->get_kernels(index)) {
                    if (!names.empty()) {
                        names += "+";
                    }
                    names += kernel->name();
                }
            }
            size_t rta_bytes = 0;
            for (const auto& v : node.rta_data) {
                rta_bytes += v.size();
            }
            auto ago = [&](std::optional<uint32_t> idx) { return idx.has_value() ? static_cast<long>(i) - *idx : -1L; };
            const bool stalled = node.dispatch_metadata.stall_first || node.dispatch_metadata.stall_before_program;
            std::fprintf(
                log,
                "%zu rid %u pid %llu workers %u nb %u bin %u cached %d reset %d nb_addr %u bin_addr %u "
                "nb_sync_ago %ld bin_sync_ago %ld stall_first %u stall_before %u sync_ago %ld send_binary %d "
                "rta_bytes %zu kernels %s\n",
                i,
                node.program_runtime_id,
                static_cast<unsigned long long>(node.program->get_id()),
                node.num_workers,
                tlog.nb,
                tlog.bin,
                tlog.cached,
                tlog.reset,
                tlog.nb_addr,
                tlog.bin_addr,
                ago(tlog.nb_sync),
                ago(tlog.bin_sync),
                static_cast<unsigned>(node.dispatch_metadata.stall_first),
                static_cast<unsigned>(node.dispatch_metadata.stall_before_program),
                stalled ? ago(last_stall_idx) : -1L,
                node.dispatch_metadata.send_binary ? 1 : 0,
                rta_bytes,
                names.c_str());
        }
        expected_workers_completed += node.num_workers;
        subdevice_launch_window.push_back(i);
        if (subdevice_launch_window.size() > max_queued_programs) {
            subdevice_launch_window.pop_front();
        }
    }
    if (log != nullptr) {
        std::fclose(log);
    }
}

}  // namespace tt::tt_metal
