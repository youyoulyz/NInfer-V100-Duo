#include "targets/qwen3_6/impl/runtime/host_lane_state_store.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {

HostLaneStateStore::HostLaneStateStore(std::vector<HostKVPageLayout> kv_layouts,
                                       std::size_t kv_bytes, HostLinearStateLayout linear_layout,
                                       HostLinearStateLayout resume_layout,
                                       std::uint32_t lane_count, std::uint32_t rank_count,
                                       std::size_t hidden_bytes)
    : kv_capacity_bytes_(kv_bytes), lane_count_(lane_count), rank_count_(rank_count),
      hidden_bytes_(hidden_bytes) {
    if (lane_count == 0) {
        throw std::invalid_argument("Host lane state store needs at least one lane");
    }
    if (rank_count == 0) {
        throw std::invalid_argument("Host lane state store needs at least one rank");
    }
    if (hidden_bytes == 0) {
        throw std::invalid_argument("Host lane state store needs a positive hidden width");
    }
    if (kv_bytes == 0) {
        throw std::invalid_argument("Host lane state store needs a positive KV budget");
    }
    const std::size_t linear_image_bytes = linear_layout.total_bytes;
    if (linear_image_bytes == 0 || resume_layout.total_bytes == 0) {
        throw std::invalid_argument("Host lane state store needs a positive linear state image");
    }

    // Both ranks share one page geometry at tp 2, and a store that registered the same page layout
    // twice could never tell two images apart, so collapse duplicates before the arena sees them.
    std::vector<HostKVPageLayout> distinct;
    for (HostKVPageLayout& layout : kv_layouts) {
        if (std::find(distinct.begin(), distinct.end(), layout) == distinct.end()) {
            distinct.push_back(std::move(layout));
        }
    }
    if (distinct.empty()) {
        throw std::invalid_argument("Host lane state store needs at least one KV page layout");
    }

    kv_arena_ =
        std::make_unique<HostKVArena>(kv_bytes, std::span<const HostKVPageLayout>(distinct));
    offloads_.reserve(distinct.size());
    for (const HostKVPageLayout& layout : distinct) {
        offloads_.push_back(std::make_unique<HostKVOffload>(*kv_arena_, layout.geometry));
    }
    // One image per lane per rank: a lane that is already parked cannot park again, so this is the
    // ceiling, and both ranks' pools are distinct physical pools.
    linear_arena_ = std::make_unique<HostLinearStateArena>(
        linear_image_bytes * static_cast<std::size_t>(lane_count_) * rank_count_);
    hidden_storage_ = std::make_unique<PinnedHostBuffer>(
        hidden_bytes * 2ULL * static_cast<std::size_t>(lane_count_));

    // Staging. Its arenas are sized to exactly what is handed out here, so every allocation
    // succeeds once and the park/restore paths never touch the CUDA allocator.
    const std::size_t staging_kv_bytes =
        std::accumulate(distinct.begin(), distinct.end(), std::size_t{0},
                        [](std::size_t total, const HostKVPageLayout& layout) {
                            return total + kStagingPages * layout.page_stride;
                        });
    staging_arena_ = std::make_unique<HostKVArena>(
        staging_kv_bytes, std::span<const HostKVPageLayout>(distinct));
    staging_offloads_.reserve(distinct.size());
    staging_images_.reserve(distinct.size());
    for (const HostKVPageLayout& layout : distinct) {
        staging_offloads_.push_back(
            std::make_unique<HostKVOffload>(*staging_arena_, layout.geometry));
        staging_images_.push_back(staging_offloads_.back()->allocate(kStagingPages));
        if (!staging_images_.back()) {
            throw std::runtime_error("Host lane state store cannot reserve its KV staging image");
        }
    }
    linear_layout_       = std::move(linear_layout);
    tier_state_layout_   = std::move(resume_layout);
    staging_state_bytes_ = tier_state_layout_.total_bytes * static_cast<std::size_t>(rank_count_) +
                           2ULL * hidden_bytes;
    staging_state_ = std::make_unique<PinnedHostBuffer>(staging_state_bytes_);
}

HostKVOffload* HostLaneStateStore::offload_for(const KVPageGeometry& geometry) noexcept {
    for (const std::unique_ptr<HostKVOffload>& offload : offloads_) {
        if (offload->layout().geometry == geometry) { return offload.get(); }
    }
    return nullptr;
}

HostKVOffload* HostLaneStateStore::staging_offload_for(const KVPageGeometry& geometry) noexcept {
    for (const std::unique_ptr<HostKVOffload>& offload : staging_offloads_) {
        if (offload->layout().geometry == geometry) { return offload.get(); }
    }
    return nullptr;
}

HostKVImage& HostLaneStateStore::staging_kv(const KVPageGeometry& geometry) {
    for (std::size_t index = 0; index < staging_offloads_.size(); ++index) {
        if (staging_offloads_[index]->layout().geometry == geometry) {
            return *staging_images_[index];
        }
    }
    throw std::invalid_argument("Host lane state store has no staging image for this geometry");
}

std::byte* HostLaneStateStore::tier_state_staging() noexcept {
    return static_cast<std::byte*>(staging_state_->data());
}

std::byte* HostLaneStateStore::hidden_row(std::uint32_t lane, bool rewrite_checkpoint) noexcept {
    if (lane >= lane_count_) { return nullptr; }
    const std::size_t row =
        static_cast<std::size_t>(lane) * 2ULL + (rewrite_checkpoint ? 1ULL : 0ULL);
    return static_cast<std::byte*>(hidden_storage_->data()) + row * hidden_bytes_;
}

std::size_t HostLaneStateStore::kv_occupied_bytes() const noexcept {
    return kv_arena_->occupied_bytes();
}

std::size_t HostLaneStateStore::linear_capacity_bytes() const noexcept {
    return linear_arena_->capacity_bytes();
}

std::size_t HostLaneStateStore::linear_occupied_bytes() const noexcept {
    return linear_arena_->occupied_bytes();
}

} // namespace ninfer::targets::qwen3_6::detail
