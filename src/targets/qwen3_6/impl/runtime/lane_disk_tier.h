#pragma once

// The NVMe (L3) tier under the host lane tier: a content-addressed store of 64-token KV blocks plus
// one state image per resumable frontier.
//
// A block is keyed by the rolling prefix digest at its last token, so the same content in two
// conversations dedupes onto one record and an incoming prompt can address a block it never
// produced itself. The state image is keyed at the frontier it closes and is what makes a resume
// legal: the GDN recurrent state cannot be rebuilt from KV, so a resume must land exactly on a
// frontier that has an image.
//
// A record holds exactly what the host tier holds -- Main Text KV, optional MTP KV, and the state
// image (every rank's linear-attention state, then rank 0's hidden rows) -- so a miss of any kind
// is a recompute, never a wrong answer.
//
// Admission never evicts: the store holds what the device and host tiers shed, so a spill that the
// budget of its family cannot hold fails and leaves the store as it was. Replacing a parked
// conversation is a policy decision for the caller, not a side effect of another lane's fill.
//
// Tensor parallel: every rank holds a distinct shard of the same tokens, so each rank's chain is
// stored under its own tag bits and the digest image -- which is rank-independent -- decides every
// rank at once. A frontier is resumable only when EVERY rank holds its shard, because a resume that
// landed one rank's KV and not the other's would be silently wrong.

#include "core/disk_kv_bridge.h"
#include "runtime/prefix_cache/block_hash.h"
#include "targets/qwen3_6/impl/runtime/identity.h"
#include "targets/qwen3_6/impl/runtime/prefix_digests.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace ninfer::targets::qwen3_6::detail {

// A resume the disk tier can serve, and how much it has to read.
struct DiskResumePoint {
    // Largest token frontier whose whole Main Text chain is present. Equals the spilled text
    // frontier, or the containing block's end when a record covers a partially spilled block.
    std::uint32_t frontier         = 0;
    // Largest frontier whose whole backend (MTP) chain is present: `frontier - backend_lag`, and
    // the value the restored sequence may claim as `mtp_kv_valid`.
    std::uint32_t backend_frontier = 0;
    std::uint32_t text_pages       = 0;
    std::uint32_t backend_pages    = 0;
    std::size_t text_bytes         = 0;
    std::size_t backend_bytes      = 0;
};

class LaneDiskTier {
public:
    struct Options {
        std::string path;
        std::size_t capacity_bytes = 0;
        std::size_t text_stride    = 0;
        std::size_t backend_stride = 0; // zero disables the MTP family
        // One state image: every rank's linear-attention state, then rank 0's hidden rows.
        std::size_t state_bytes = 0;
        // How far behind a text frontier the backend (MTP) KV may be while still being complete:
        // the MTP bridge re-derives the last position, so its chain is spilled and restored at
        // `frontier - backend_lag`. Zero means the backend tracks the text frontier exactly.
        std::uint32_t backend_lag = 0;
        std::uint32_t tag         = 0; // identity_tag of the engine configuration
        // Tensor-parallel degree. Rank 0 keeps `tag` itself, so a single-rank tier's keys are
        // exactly the ones the identity tag names; rank r > 0 sets `tag | (r << kRankTagShift)`.
        std::uint32_t ranks = 1;
        bool verify_crc     = true;
    };

    explicit LaneDiskTier(Options options);

    LaneDiskTier(const LaneDiskTier&)            = delete;
    LaneDiskTier& operator=(const LaneDiskTier&) = delete;

    [[nodiscard]] const Options& options() const noexcept { return options_; }

    // Bits above the identity tag's three 8-bit fields, where a rank shard is named.
    static constexpr std::uint32_t kRankTagShift = 24U;
    static constexpr std::uint32_t kMaximumRanks  = 0x100U;
    [[nodiscard]] std::uint32_t rank_tag(std::uint32_t rank) const noexcept {
        return options_.tag | (rank << kRankTagShift);
    }
    [[nodiscard]] std::uint32_t ranks() const noexcept { return options_.ranks; }

    // Pages a resume at `frontier` covers: every 64-token block it spans.
    [[nodiscard]] static std::uint32_t prefix_pages(std::uint32_t frontier) noexcept;

    // The backend frontier a text frontier `frontier` implies.
    [[nodiscard]] std::uint32_t backend_frontier(std::uint32_t frontier) const noexcept;

    // Writes records [first, first + count) of one rank's chain of `kind`, whose last token is
    // `chain_frontier`. False when a record cannot be stored (a full family, an I/O error, a shape
    // the family does not accept), which leaves the chain incomplete -- and an incomplete chain
    // publishes no frontier, because `spill_state` runs last.
    bool spill_pages(const PrefixDigests& digests, DiskKVKind kind, std::uint32_t rank,
                     std::uint32_t chain_frontier, std::uint32_t first, std::uint32_t count,
                     std::span<const std::byte> pages);

    // Publishes the frontier by writing the state image that closes it. This is the commit: only a
    // stored image makes the whole prefix above it resumable.
    bool spill_state(const PrefixDigests& digests, std::uint32_t frontier,
                     std::span<const std::byte> state);

    // The deepest frontier at or below `prompt_tokens` whose state image and whole block chain are
    // present under this prompt's own digest, on every rank.
    [[nodiscard]] std::optional<DiskResumePoint> plan_resume(const PrefixDigests& digests,
                                                             std::uint32_t prompt_tokens) const;

    // Reads records [first, first + count) of the plan's chain of `kind` for `rank` into
    // `destination` (count strides). False when the range is outside the plan this point described.
    [[nodiscard]] bool load_kv_pages(const PrefixDigests& digests, const DiskResumePoint& point,
                                     DiskKVKind kind, std::uint32_t rank, std::uint32_t first,
                                     std::uint32_t count, std::byte* destination) const;

    // Reads the plan's state image into a buffer of options().state_bytes.
    [[nodiscard]] bool load_state(const PrefixDigests& digests, const DiskResumePoint& point,
                                  std::span<std::byte> state) const;

    void wait_idle();

    [[nodiscard]] std::uint32_t state_slots() const;
    [[nodiscard]] std::size_t used_bytes() const;
    [[nodiscard]] DiskKVBridgeStats stats() const;

private:
    // Queues one block and waits for the bridge's own verdict (the queue accepts a page it cannot
    // store, and a chain with a hole must not be published).
    bool write_block(const DiskKVIdentity& id, DiskKVKind kind, std::span<const std::byte> bytes);

    [[nodiscard]] std::uint32_t chain_frontier(const DiskResumePoint& point,
                                               DiskKVKind kind) const noexcept;
    [[nodiscard]] std::uint32_t page_count(const DiskResumePoint& point,
                                           DiskKVKind kind) const noexcept;
    [[nodiscard]] std::uint32_t block_frontier(std::uint32_t page,
                                               std::uint32_t frontier) const noexcept;
    [[nodiscard]] std::size_t stride_for(DiskKVKind kind) const noexcept;

    Options options_;
    std::unique_ptr<DiskKVBridge> bridge_;
};

} // namespace ninfer::targets::qwen3_6::detail
