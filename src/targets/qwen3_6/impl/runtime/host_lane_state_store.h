#pragma once

// Pinned host storage a Program parks lane state into.
//
// Every byte is allocated in the constructor, so a park is a pure transfer: no CUDA allocation
// happens on the eviction path. The store is deliberately variant-agnostic -- it mirrors pools and
// geometries, knows nothing about Qwen3.6's state roles, and is shared by rank 0 and rank 1 because
// at tp 2 both ranks' KV pools carry the same plane geometry.
//
// The KV half frees device memory when a lane is parked. The linear-attention half does not: that
// pool is statically partitioned by lane, so its slot bytes are never returned. It is mirrored
// because a restored prefix is only legal when the frontier's GDN state is coherent with it, and
// that state cannot be rebuilt from KV.

#include "core/arena.h"
#include "core/host_kv_arena.h"
#include "core/host_kv_offload.h"
#include "core/host_linear_state.h"
#include "core/kv_page_geometry.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class HostLaneStateStore {
public:
    // `kv_layouts` is the set of distinct host page layouts this Program can park; duplicates are
    // collapsed, because both ranks share one geometry at tp 2. `linear_image_bytes` is the size of
    // ONE lane's linear-attention image (its two slot roles across every layer).
    HostLaneStateStore(std::vector<HostKVPageLayout> kv_layouts, std::size_t kv_bytes,
                       std::size_t linear_image_bytes, std::uint32_t lane_count,
                       std::size_t hidden_bytes);

    HostLaneStateStore(const HostLaneStateStore&)            = delete;
    HostLaneStateStore& operator=(const HostLaneStateStore&) = delete;
    HostLaneStateStore(HostLaneStateStore&&)                 = delete;
    HostLaneStateStore& operator=(HostLaneStateStore&&)      = delete;

    // The offload that mirrors `geometry`, or nullptr when this store has no layout for it.
    [[nodiscard]] HostKVOffload* offload_for(const KVPageGeometry& geometry) noexcept;

    [[nodiscard]] HostLinearStateArena& linear_arena() noexcept { return *linear_arena_; }
    [[nodiscard]] const HostLinearStateArena& linear_arena() const noexcept { return *linear_arena_; }

    // Rank 0's hidden mirror for `lane`: two rows, tail then rewrite checkpoint, each `hidden_bytes`.
    [[nodiscard]] std::byte* hidden_row(std::uint32_t lane, bool rewrite_checkpoint) noexcept;

    [[nodiscard]] std::size_t kv_capacity_bytes() const noexcept { return kv_capacity_bytes_; }
    [[nodiscard]] std::size_t kv_occupied_bytes() const noexcept;
    [[nodiscard]] std::size_t linear_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t linear_occupied_bytes() const noexcept;

    [[nodiscard]] std::uint32_t lane_count() const noexcept { return lane_count_; }
    [[nodiscard]] std::size_t hidden_bytes() const noexcept { return hidden_bytes_; }

private:
    std::unique_ptr<HostKVArena> kv_arena_;
    std::vector<std::unique_ptr<HostKVOffload>> offloads_;
    std::unique_ptr<HostLinearStateArena> linear_arena_;
    std::unique_ptr<PinnedHostBuffer> hidden_storage_;
    std::size_t kv_capacity_bytes_ = 0;
    std::uint32_t lane_count_      = 0;
    std::size_t hidden_bytes_      = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
