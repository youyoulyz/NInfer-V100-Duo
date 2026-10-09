#include "targets/qwen3_6/impl/runtime/host_lane_state_store.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {

HostLaneStateStore::HostLaneStateStore(std::vector<HostKVPageLayout> kv_layouts,
                                       std::size_t kv_bytes, std::size_t linear_image_bytes,
                                       std::uint32_t lane_count, std::size_t hidden_bytes)
    : kv_capacity_bytes_(kv_bytes), lane_count_(lane_count), hidden_bytes_(hidden_bytes) {
    if (lane_count == 0) {
        throw std::invalid_argument("Host lane state store needs at least one lane");
    }
    if (hidden_bytes == 0) {
        throw std::invalid_argument("Host lane state store needs a positive hidden width");
    }
    if (kv_bytes == 0) {
        throw std::invalid_argument("Host lane state store needs a positive KV budget");
    }
    if (linear_image_bytes == 0) {
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
    // One image per lane; a lane that is already parked cannot park again, so this is the ceiling.
    linear_arena_ = std::make_unique<HostLinearStateArena>(
        linear_image_bytes * static_cast<std::size_t>(lane_count_));
    hidden_storage_ = std::make_unique<PinnedHostBuffer>(
        hidden_bytes * 2ULL * static_cast<std::size_t>(lane_count_));
}

HostKVOffload* HostLaneStateStore::offload_for(const KVPageGeometry& geometry) noexcept {
    for (const std::unique_ptr<HostKVOffload>& offload : offloads_) {
        if (offload->layout().geometry == geometry) { return offload.get(); }
    }
    return nullptr;
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
