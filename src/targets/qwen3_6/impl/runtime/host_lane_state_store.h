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
    // collapsed, because both ranks share one geometry at tp 2. `linear_layout` describes ONE
    // lane's linear-attention image (its two slot roles across every layer), one image per lane per
    // rank; `resume_layout` is the narrower image a disk resume stores and needs -- the one state
    // role a restored prefix resumes from, and nothing a resume never reads. `rank_count` is 1 at
    // tp 1 and 2 at tp 2.
    HostLaneStateStore(std::vector<HostKVPageLayout> kv_layouts, std::size_t kv_bytes,
                       HostLinearStateLayout linear_layout, HostLinearStateLayout resume_layout,
                       std::uint32_t lane_count, std::uint32_t rank_count,
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

    // Staging the disk tier moves bytes through. It is deliberately OUTSIDE the lane budget: a
    // restore must never compete with a parked lane for room, and its size (one 32-page image per
    // KV layout plus one state image) is held for the Program's lifetime.
    [[nodiscard]] static constexpr std::uint32_t staging_pages() noexcept { return kStagingPages; }
    [[nodiscard]] HostKVOffload* staging_offload_for(const KVPageGeometry& geometry) noexcept;
    [[nodiscard]] HostKVImage& staging_kv(const KVPageGeometry& geometry);
    // One state image: rank 0's resumable linear-attention image, then rank 1's, then rank 0's two
    // hidden rows. It is exactly the blob the disk tier stores and reads back, so neither path
    // needs a second copy.
    [[nodiscard]] const HostLinearStateLayout& tier_state_layout() const noexcept {
        return tier_state_layout_;
    }
    [[nodiscard]] std::byte* tier_state_staging() noexcept;
    [[nodiscard]] std::size_t tier_state_staging_bytes() const noexcept {
        return staging_state_bytes_;
    }
    [[nodiscard]] std::uint32_t ranks() const noexcept { return rank_count_; }

    [[nodiscard]] std::size_t kv_capacity_bytes() const noexcept { return kv_capacity_bytes_; }
    [[nodiscard]] std::size_t kv_occupied_bytes() const noexcept;
    [[nodiscard]] std::size_t linear_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t linear_occupied_bytes() const noexcept;

    [[nodiscard]] std::uint32_t lane_count() const noexcept { return lane_count_; }
    [[nodiscard]] std::size_t hidden_bytes() const noexcept { return hidden_bytes_; }

private:
    // One batch is one disk read: 32 records is enough to keep NVMe busy without a lane-sized
    // buffer, and it bounds the staging image every KV family has to hold.
    static constexpr std::uint32_t kStagingPages = 32;

    std::unique_ptr<HostKVArena> kv_arena_;
    std::vector<std::unique_ptr<HostKVOffload>> offloads_;
    std::unique_ptr<HostLinearStateArena> linear_arena_;
    std::unique_ptr<PinnedHostBuffer> hidden_storage_;
    std::unique_ptr<HostKVArena> staging_arena_;
    std::vector<std::unique_ptr<HostKVOffload>> staging_offloads_;
    std::vector<std::optional<HostKVImage>> staging_images_;
    std::unique_ptr<PinnedHostBuffer> staging_state_;
    HostLinearStateLayout linear_layout_;
    HostLinearStateLayout tier_state_layout_;
    std::size_t staging_state_bytes_ = 0;
    std::size_t kv_capacity_bytes_ = 0;
    std::uint32_t lane_count_      = 0;
    std::uint32_t rank_count_      = 0;
    std::size_t hidden_bytes_      = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
