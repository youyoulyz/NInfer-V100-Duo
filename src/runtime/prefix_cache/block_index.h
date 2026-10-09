#pragma once

// Host-side content-addressed index of 64-token KV blocks: the prefix-indexed chunk store's
// bookkeeping.
//
// A block is identified by its parent, its chained lookup hash, its exact 64 tokens and its extra
// (Vision) key. The lookup hash only selects a child bucket; equality is always decided by
// comparing parent, hash, extra and every token, so a forced collision costs a comparison and
// never produces a false hit.
//
// The index owns exact identity and topology only. Where a block's bytes live is the tier's
// business: the caller records the copies it holds through `copies` and moves the bytes itself.
// Handles carry a generation, so a handle into a removed subtree is rejected instead of aliasing
// a recycled slot.
//
// Ported from the sibling line's PrefixCacheIndex node tree (models/qwen3_5/program/prefix),
// keeping its child-key mixing, exact comparison and generation-checked handles; that line's
// snapshot, tap, GDSF and persistence machinery is deliberately not part of this.

#include "runtime/prefix_cache/block_hash.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime::prefix_cache {

inline constexpr std::uint32_t kNoBlock = std::numeric_limits<std::uint32_t>::max();

struct BlockRef {
    std::uint32_t index      = kNoBlock;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept { return index != kNoBlock; }

    [[nodiscard]] friend bool operator==(BlockRef, BlockRef) noexcept = default;
};

// Where a block's bytes live. A block may hold several copies at once, for example a device
// working copy plus a host sink.
enum class BlockCopy : std::uint8_t {
    Device = 1U << 0U,
    Host   = 1U << 1U,
    Disk   = 1U << 2U,
};

[[nodiscard]] constexpr std::uint8_t block_copy_bit(BlockCopy copy) noexcept {
    return static_cast<std::uint8_t>(copy);
}

struct BlockView {
    std::uint32_t depth       = 0;
    BlockRef parent;
    std::uint64_t lookup_hash = 0;
    std::uint64_t extra       = 0;
    std::span<const TokenId> tokens; // exactly kBlockTokens
    std::uint8_t copies       = 0;
    std::uint32_t pins        = 0;
};

struct BlockInsert {
    BlockRef node;
    bool inserted = false;
};

class BlockIndex {
public:
    explicit BlockIndex(std::uint32_t max_nodes);

    BlockIndex(const BlockIndex&)            = delete;
    BlockIndex& operator=(const BlockIndex&) = delete;

    // Exact child of `parent` (the root when `parent` is invalid) for this hash and content, or
    // nullopt. `tokens` must be exactly kBlockTokens long.
    [[nodiscard]] std::optional<BlockRef> find_child(BlockRef parent, std::uint64_t lookup_hash,
                                                    std::span<const TokenId> tokens,
                                                    std::uint64_t extra) const;

    // Interns the block, or returns the existing exact child. `copies` is the initial locality of
    // a new node; an existing node keeps its copies and gains one pin. The new node starts pinned.
    BlockInsert insert(BlockRef parent, std::uint64_t lookup_hash, std::span<const TokenId> tokens,
                       std::uint64_t extra, std::uint8_t copies = 0);

    // Walks every full block of `tokens`, interning the path, and returns the deepest block (an
    // invalid ref when the prompt has no full block). `hashes` is the chained block hash of every
    // full block and `extras` its Vision key (block_keys.h); `extras` may be empty for an all-zero
    // extra.
    [[nodiscard]] BlockRef intern_chain(std::span<const TokenId> tokens,
                                        std::span<const std::uint64_t> hashes,
                                        std::span<const std::uint64_t> extras,
                                        std::uint8_t copies = 0);

    [[nodiscard]] bool valid(BlockRef node) const noexcept;
    [[nodiscard]] BlockView view(BlockRef node) const;

    void set_copies(BlockRef node, std::uint8_t copies) noexcept;
    void pin(BlockRef node);
    void unpin(BlockRef node);

    // Drops a whole subtree, invalidating every handle in it. Returns the node count removed.
    std::uint32_t remove_subtree(BlockRef node);

    [[nodiscard]] std::uint32_t node_count() const noexcept { return node_count_; }
    [[nodiscard]] std::uint32_t root_children() const noexcept {
        return static_cast<std::uint32_t>(root_children_.size());
    }
    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(nodes_.size());
    }

private:
    struct Node {
        std::uint32_t generation = 1;
        bool occupied            = false;
        std::uint32_t parent     = kNoBlock;
        std::uint32_t depth      = 0;
        std::uint64_t hash       = 0;
        std::uint64_t extra      = 0;
        std::array<TokenId, kBlockTokens> tokens{};
        std::vector<std::uint32_t> children;
        std::uint8_t copies = 0;
        std::uint32_t pins  = 0;
    };

    [[nodiscard]] const Node& require(BlockRef node) const;
    [[nodiscard]] Node& require(BlockRef node);
    [[nodiscard]] BlockRef ref_of(std::uint32_t index) const noexcept;
    [[nodiscard]] static std::uint64_t child_key(std::uint32_t parent,
                                                 std::uint64_t lookup_hash) noexcept;

    std::vector<Node> nodes_;
    std::vector<std::uint32_t> free_nodes_;
    std::unordered_multimap<std::uint64_t, std::uint32_t> children_;
    std::vector<std::uint32_t> root_children_;
    std::uint32_t node_count_ = 0;
};

} // namespace ninfer::runtime::prefix_cache
