#include "targets/qwen3_6/impl/runtime/lane_disk_tier.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {
namespace {

namespace pc = runtime::prefix_cache;

// A page the queue will not take means the prefix would have a hole, and a restore stops at the
// first hole, so the spill fails rather than publishing an incomplete chain.
constexpr auto kSpillTimeout = std::chrono::milliseconds(5000);
constexpr auto kPollInterval = std::chrono::microseconds(50);

} // namespace

LaneDiskTier::LaneDiskTier(Options options) : options_(std::move(options)) {
    if (options_.path.empty() || options_.capacity_bytes == 0 || options_.text_stride == 0 ||
        options_.state_bytes == 0) {
        throw std::invalid_argument(
            "disk lane tier needs a path, a budget, a page stride and a state stride");
    }
    if (options_.ranks == 0 || options_.ranks >= kMaximumRanks) {
        throw std::invalid_argument("disk lane tier has an invalid tensor-parallel width");
    }
    if (options_.backend_stride == 0 && options_.backend_lag != 0) {
        throw std::invalid_argument("disk lane tier has a backend lag without a backend family");
    }
    DiskKVBridge::Options bridge;
    bridge.base_path           = options_.path;
    bridge.main_page_stride    = options_.text_stride;
    bridge.backend_page_stride = options_.backend_stride;
    bridge.state_page_stride   = options_.state_bytes;
    bridge.capacity_bytes      = options_.capacity_bytes;
    bridge.verify_crc          = options_.verify_crc;
    // The pages here are the only copy of a parked conversation: a spill that does not fit fails
    // rather than evicting a chain another lane still resumes from.
    bridge.allow_eviction      = false;
    bridge_ = std::make_unique<DiskKVBridge>(std::move(bridge));
}

std::uint32_t LaneDiskTier::prefix_pages(std::uint32_t frontier) noexcept {
    return (frontier + pc::kBlockTokens - 1U) / pc::kBlockTokens;
}

std::uint32_t LaneDiskTier::backend_frontier(std::uint32_t frontier) const noexcept {
    return frontier > options_.backend_lag ? frontier - options_.backend_lag : 0U;
}

std::uint32_t LaneDiskTier::block_frontier(std::uint32_t page,
                                           std::uint32_t frontier) const noexcept {
    return std::min((page + 1U) * pc::kBlockTokens, frontier);
}

std::uint32_t LaneDiskTier::chain_frontier(const DiskResumePoint& point,
                                           DiskKVKind kind) const noexcept {
    return kind == DiskKVKind::MainKV ? point.frontier : point.backend_frontier;
}

std::uint32_t LaneDiskTier::page_count(const DiskResumePoint& point,
                                       DiskKVKind kind) const noexcept {
    return kind == DiskKVKind::MainKV ? point.text_pages : point.backend_pages;
}

std::size_t LaneDiskTier::stride_for(DiskKVKind kind) const noexcept {
    return kind == DiskKVKind::MainKV ? options_.text_stride : options_.backend_stride;
}

// Queues one block and waits for the bridge's own verdict, which is the only thing that
// distinguishes "written" from "the family is full": the queueing entry point reports success for
// a page whose write the writer thread later refuses, and a chain with such a hole must not be
// published. The borrowed bytes stay alive for as long as the ticket is pending, which is why a
// timeout drains the queue before the caller may release them.
bool LaneDiskTier::write_block(const DiskKVIdentity& id, DiskKVKind kind,
                               std::span<const std::byte> bytes) {
    const auto deadline = std::chrono::steady_clock::now() + kSpillTimeout;
    for (;;) {
        SpillTicket ticket = bridge_->try_submit(id, kind, bytes);
        if (ticket) {
            while (DiskKVBridge::poll(ticket) == SpillStatus::Pending) {
                if (std::chrono::steady_clock::now() > deadline) {
                    bridge_->wait_idle();
                    return false;
                }
                std::this_thread::sleep_for(kPollInterval);
            }
            return DiskKVBridge::poll(ticket) == SpillStatus::Written;
        }
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(kPollInterval);
    }
}

bool LaneDiskTier::spill_pages(const PrefixDigests& digests, DiskKVKind kind, std::uint32_t rank,
                               std::uint32_t chain_frontier, std::uint32_t first,
                               std::uint32_t count, std::span<const std::byte> pages) {
    if (kind == DiskKVKind::StateImage || rank >= options_.ranks) { return false; }
    if (count == 0) { return true; }
    const std::size_t stride = stride_for(kind);
    if (stride == 0 || pages.size() != static_cast<std::size_t>(count) * stride) { return false; }
    if (chain_frontier == 0 || chain_frontier > digests.size() ||
        first + count > prefix_pages(chain_frontier)) {
        return false;
    }
    const std::uint32_t tag = rank_tag(rank);
    for (std::uint32_t index = 0; index < count; ++index) {
        const std::uint32_t page = first + index;
        const DiskKVIdentity id =
            make_identity(digests, tag, block_frontier(page, chain_frontier));
        if (bridge_->contains(id, kind)) { continue; }
        if (!write_block(id, kind, pages.subspan(static_cast<std::size_t>(index) * stride,
                                                 stride))) {
            return false;
        }
    }
    return true;
}

bool LaneDiskTier::spill_state(const PrefixDigests& digests, std::uint32_t frontier,
                               std::span<const std::byte> state) {
    if (frontier == 0 || frontier > digests.size() || state.size() != options_.state_bytes) {
        return false;
    }
    // The image carries every rank's bytes, but it is addressed by the identity tag itself: a
    // rank's KV shard is what needs a per-rank key, and the state is not sharded.
    return bridge_->spill_page_sync(make_identity(digests, options_.tag, frontier),
                                    DiskKVKind::StateImage, state);
}

std::optional<DiskResumePoint> LaneDiskTier::plan_resume(const PrefixDigests& digests,
                                                          std::uint32_t prompt_tokens) const {
    std::vector<std::uint32_t> frontiers = bridge_->live_state_frontiers();
    std::sort(frontiers.begin(), frontiers.end(), std::greater<>());
    frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());
    const auto chain_present = [&](std::uint32_t rank, DiskKVKind kind, std::uint32_t chain) {
        const std::uint32_t tag = rank_tag(rank);
        for (std::uint32_t page = 0; page < prefix_pages(chain); ++page) {
            if (!bridge_->contains(make_identity(digests, tag, block_frontier(page, chain)),
                                   kind)) {
                return false;
            }
        }
        return true;
    };
    for (const std::uint32_t frontier : frontiers) {
        if (frontier == 0 || frontier > prompt_tokens || digests.size() < frontier) { continue; }
        if (!bridge_->contains(make_identity(digests, options_.tag, frontier),
                               DiskKVKind::StateImage)) {
            continue;
        }
        bool complete = true;
        for (std::uint32_t rank = 0; rank < options_.ranks && complete; ++rank) {
            complete = chain_present(rank, DiskKVKind::MainKV, frontier);
        }
        // The backend chain is only required where the engine will claim one: a configuration
        // without an MTP pool never spills it, so a text-only record stays resumable.
        const std::uint32_t backend = backend_frontier(frontier);
        for (std::uint32_t rank = 0; rank < options_.ranks && complete; ++rank) {
            if (options_.backend_stride != 0) {
                complete = chain_present(rank, DiskKVKind::BackendKV, backend);
            }
        }
        if (!complete) { continue; }
        DiskResumePoint point;
        point.frontier         = frontier;
        point.backend_frontier = backend;
        point.text_pages       = prefix_pages(frontier);
        point.backend_pages    = options_.backend_stride != 0 ? prefix_pages(backend) : 0;
        point.text_bytes       = point.text_pages * options_.text_stride;
        point.backend_bytes    = point.backend_pages * options_.backend_stride;
        return point;
    }
    return std::nullopt;
}

bool LaneDiskTier::load_kv_pages(const PrefixDigests& digests, const DiskResumePoint& point,
                                 DiskKVKind kind, std::uint32_t rank, std::uint32_t first,
                                 std::uint32_t count, std::byte* destination) const {
    if (kind == DiskKVKind::StateImage || destination == nullptr || count == 0 ||
        rank >= options_.ranks || first + count > page_count(point, kind) ||
        point.frontier == 0) {
        return false;
    }
    const std::size_t stride = stride_for(kind);
    const std::uint32_t chain = chain_frontier(point, kind);
    const std::uint32_t tag   = rank_tag(rank);
    std::vector<DiskKVIdentity> ids(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        ids[index] = make_identity(digests, tag, block_frontier(first + index, chain));
    }
    return bridge_->restore_pages(ids, kind,
                                  {destination, static_cast<std::size_t>(count) * stride});
}

bool LaneDiskTier::load_state(const PrefixDigests& digests, const DiskResumePoint& point,
                              std::span<std::byte> state) const {
    if (point.frontier == 0 || state.size() != options_.state_bytes) { return false; }
    return bridge_->restore_page(make_identity(digests, options_.tag, point.frontier),
                                 DiskKVKind::StateImage, state);
}

void LaneDiskTier::wait_idle() { bridge_->wait_idle(); }

std::uint32_t LaneDiskTier::state_slots() const {
    return bridge_->slot_count(DiskKVKind::StateImage);
}

std::size_t LaneDiskTier::used_bytes() const { return bridge_->used_bytes(); }

DiskKVBridgeStats LaneDiskTier::stats() const { return bridge_->stats(); }

} // namespace ninfer::targets::qwen3_6::detail
