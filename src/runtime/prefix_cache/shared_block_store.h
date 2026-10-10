#pragma once

// Content-addressed index of resident prefix continuations: one `BlockIndex` plus, per node, the
// lanes that can currently hand that block of the chain to a claiming lane.
//
// This is the shared half of cross-lane prefix reuse. The tree answers "which resident sequence
// already covers this prefix, and how deep does its coverage reach"; whether the named lane can
// still serve that frontier is re-verified against the lane's live state by the caller, exactly as
// a resident plan always has. Keeping that split here is deliberate: the store owns content
// identity, the Program owns state, and neither has to trust the other's snapshot.
//
// Ownership of the *bytes* is not this component's business yet: a node names the lane, and the
// Program maps that lane's pages. `entries` therefore carries only lanes, frontiers and ages; the
// physical page groups a later tier moves between device, host and disk attach to the same node
// without changing the lookup.
//
// The store is deliberately host-only and CUDA-free so the tree can be exercised without a device.

#include "runtime/prefix_cache/block_index.h"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::runtime::prefix_cache {

inline constexpr std::uint32_t kNoSharedLane  = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t kNoSharedImage = std::numeric_limits<std::uint32_t>::max();
// Page-group slots of one block: one per (pool, rank). The Program names them -- Main Text on rank 0
// and rank 1, then the speculative backend on both -- and the store only moves the ids around.
inline constexpr std::uint32_t kSharedBlockSlots = 4;
inline constexpr std::int32_t kNoSharedPage     = -1;

// Which continuation a promise or an image closes: the lane's live append frontier, or the rollback
// seam a replayed turn resumes from. The restored continuation differs, so the role travels with
// the promise.
enum class SharedRole : std::uint8_t {
    None       = 0,
    Append     = 1,
    Checkpoint = 2,
};

// The page group one block occupies, one slot per pool and rank.
struct SharedBlockPages {
    std::array<std::int32_t, kSharedBlockSlots> page{kNoSharedPage, kNoSharedPage, kNoSharedPage,
                                                      kNoSharedPage};

    [[nodiscard]] bool any() const noexcept {
        for (const std::int32_t id : page) {
            if (id != kNoSharedPage) { return true; }
        }
        return false;
    }

    [[nodiscard]] friend bool operator==(const SharedBlockPages&, const SharedBlockPages&) = default;
};

// A continuation the node can hand out on its own, without the lane that produced it: the page
// group the frontier sits inside, plus a Program-side state image (the GDN state of every rank, the
// hidden rows and the prefix metadata) that restores it. `slot` is opaque here; the Program owns the
// image and returns it when the payload is dropped.
struct SharedBlockPayload {
    SharedBlockPages tail;                // the page group `frontier` sits inside (absent when the
                                          // frontier ends exactly on a block boundary)
    std::uint32_t image      = kNoSharedImage;
    std::uint32_t frontier   = 0;
    std::uint32_t kind       = 0;         // RewriteCheckpointKind, opaque here
    SharedRole role          = SharedRole::None;
    std::uint32_t last_use   = 0;
    // Where this payload's KV lives: `Device` when the store holds the chain's page groups (the
    // blocks recorded with the nodes, the frontier's page in `tail`), `Host` when the Program
    // parked a per-node copy into pinned host RAM and the device pages went back to the pool.
    std::uint8_t copies      = 0;

    [[nodiscard]] bool valid() const noexcept { return image != kNoSharedImage; }
    [[nodiscard]] bool kv_in_host() const noexcept {
        return (copies & static_cast<std::uint8_t>(BlockCopy::Host)) != 0;
    }
};

struct SharedBlockPayloadView {
    BlockRef node;
    SharedBlockPayload payload;
};
// Distinct lanes that may serve one block. A chain's block is offered by every retained lane that
// still holds it, and a burst that adopts the same prefix repeatedly keeps re-stamping the same
// node from different lanes; beyond this many the least recently used owner is displaced.
inline constexpr std::uint32_t kMaxSharedOwners = 8;

// One lane's promise for a node: it holds a continuation state at `frontier` and its KV covers the
// whole block. `last_use` is the round the promise was (re)made and orders owners and eviction.
struct SharedBlockOwner {
    std::uint32_t lane     = kNoSharedLane;
    std::uint32_t frontier = 0;
    std::uint32_t kind     = 0;   // RewriteCheckpointKind, opaque here
    SharedRole role        = SharedRole::None;
    std::uint64_t last_use = 0;

    [[nodiscard]] friend bool operator==(const SharedBlockOwner&, const SharedBlockOwner&) = default;
};

// What a lookup found: the deepest node of the queried chain that at least one live owner serves,
// and those owners. `tokens` is the node's token frontier (a multiple of kBlockTokens).
struct SharedBlockHit {
    BlockRef node;
    std::uint32_t tokens = 0;
    std::array<SharedBlockOwner, kMaxSharedOwners> owners{};
    std::uint32_t owner_count = 0;
};

// The two ways a chain can serve a request: a live lane that still holds the state (cheap, but it
// dies with the lane), or a node payload -- page groups held by the store plus a state image -- that
// outlives every lane. A lookup reports the deepest node of each kind; the Program grades them.
struct SharedBlockLookup {
    std::optional<SharedBlockHit> owner;
    std::optional<SharedBlockPayloadView> payload;
};

// Whether an owner can still hand out `tokens` of its chain. The Program decides -- it alone knows
// the lane's live frontier, its checkpoint and whether the request's own prefix matches -- so a
// stale promise costs nothing worse than a plan that falls back to the cold prefill it priced.
using SharedServesFn = std::function<bool(const SharedBlockOwner&, std::uint32_t tokens)>;

struct SharedBlockStats {
    std::uint32_t nodes        = 0;
    std::uint32_t owner_slots  = 0; // node/owner pairs currently published
    std::uint32_t ownerless    = 0; // nodes no lane serves any more
    std::uint32_t payloads     = 0; // nodes holding page groups and a state image of their own
    std::uint32_t pages        = 0; // page groups the store keeps alive
};

// The pool and image work the store cannot do itself: it names a slot and the Program maps that slot
// to the pool or arena that owns it. A hook that throws propagates to the caller, which is where a
// failed hold has to be handled.
struct SharedBlockStoreHooks {
    // Keep one page group out of its pool's free set, and give it back when the node is dropped.
    std::function<void(std::uint32_t slot, std::int32_t page)> hold_page;
    std::function<void(std::uint32_t slot, std::int32_t page)> release_page;
    // The node's state image is leaving the store; the Program returns its slot to the arena.
    std::function<void(std::uint32_t slot)> release_image;
};

class SharedBlockStore {
public:
    explicit SharedBlockStore(std::uint32_t max_nodes);

    SharedBlockStore(const SharedBlockStore&)            = delete;
    SharedBlockStore& operator=(const SharedBlockStore&) = delete;

    // Interns the chain of full `kBlockTokens` blocks of `tokens` and stamps `owner` on every node
    // whose whole block the owner's frontier covers. `hashes` is the chained lookup hash of each
    // block and `extras` its Vision key, exactly as `block_key`s produces them; `extras` may be
    // empty for an all-zero extra. Returns the chain's deepest node, invalid when the span holds no
    // full block.
    [[nodiscard]] BlockRef intern(std::span<const TokenId> tokens,
                                  std::span<const std::uint64_t> hashes,
                                  std::span<const std::uint64_t> extras,
                                  const SharedBlockOwner& owner);

    // Deepest node of `tokens`' chain that a live owner serves, and the deepest node of the same
    // chain carrying a self-contained payload. Only the tree, the owners and the payloads are read;
    // the caller re-checks a continuation against the owner's live state or the payload's metadata.
    [[nodiscard]] SharedBlockLookup
    lookup(std::span<const TokenId> tokens, std::span<const std::uint64_t> hashes,
           std::span<const std::uint64_t> extras, const SharedServesFn& serves) const;

    // The deepest interned node of `tokens`' chain, whether or not any lane serves it. Used to
    // address the node a payload belongs to when there is no promise left to name it.
    [[nodiscard]] std::optional<BlockRef> find(std::span<const TokenId> tokens,
                                               std::span<const std::uint64_t> hashes,
                                               std::span<const std::uint64_t> extras) const;

    // Records the page group of one block. The first copy wins: it is already held, so a second lane
    // publishing the same content needs no second copy of the same bytes.
    void attach_pages(BlockRef node, const SharedBlockPages& pages);

    // Attaches the continuation image a claim would restore. An existing image is released through
    // the hooks before it is replaced.
    void attach_payload(BlockRef node, const SharedBlockPayload& payload);

    // Drops a node's payload. The node then stands on its owner promises alone, so a node whose
    // lane is already gone is reclaimed by the next prune.
    void drop_payload(BlockRef node);

    // The page groups of the chain from its root to `node`, in order. Empty when any node of the
    // chain carries none, because a claim has to map the whole prefix or nothing.
    [[nodiscard]] std::vector<SharedBlockPages> chain_pages(BlockRef node) const;

    [[nodiscard]] bool chain_has_pages(BlockRef node) const;
    [[nodiscard]] SharedBlockPayload payload(BlockRef node) const;
    [[nodiscard]] SharedBlockPages pages(BlockRef node) const;

    void set_hooks(SharedBlockStoreHooks hooks);

    // Forgets a lane everywhere. Called when a lane's chain is about to be replaced or torn down,
    // so a promise it made for a prefix it no longer holds cannot outlive it.
    void forget_lane(std::uint32_t lane);

    // Drops leaf nodes that no owner serves any more, deepest branch tips first, repeating until
    // nothing droppable is left: dropping a tip can expose its parent, and an unshared branch goes
    // away whole. Returns the nodes removed.
    std::uint32_t prune();

    // Makes room for `nodes` more nodes: unserved leaves first, then the least recently used served
    // leaves. Returns false when even that cannot free enough room.
    bool ensure_capacity(std::uint32_t nodes);

    // Drops the least recently used leaf nodes until `budget` are gone, whatever serves them. Only
    // the capacity emergency uses this, because it sacrifices cached prefixes.
    std::uint32_t evict_lru(std::uint32_t budget);

    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::uint32_t node_count() const noexcept { return index_.node_count(); }
    [[nodiscard]] std::uint32_t free_nodes() const noexcept {
        return capacity_ - index_.node_count();
    }
    [[nodiscard]] std::optional<BlockView> view(BlockRef node) const;
    [[nodiscard]] std::uint32_t owner_count(BlockRef node) const noexcept;
    [[nodiscard]] bool has_payload(BlockRef node) const noexcept;
    [[nodiscard]] const std::array<SharedBlockOwner, kMaxSharedOwners>* owners(BlockRef node) const
        noexcept;
    [[nodiscard]] SharedBlockStats stats() const;

    // For every lane that is the sole owner of at least one node, the deepest token frontier it
    // alone serves. Dropping that lane is what costs this much to rebuild, so eviction orders
    // retained lanes by it: a lane whose prefix a sibling also serves still gives up nothing.
    [[nodiscard]] std::vector<std::pair<std::uint32_t, std::uint32_t>> exclusive_frontiers() const;

private:
    struct Entry {
        std::array<SharedBlockOwner, kMaxSharedOwners> owners{};
        std::uint32_t owner_count = 0;
        std::uint64_t last_use    = 0;
        SharedBlockPages pages;
        SharedBlockPayload payload;
    };

    [[nodiscard]] Entry& entry_of(std::uint32_t index) noexcept { return entries_[index]; }
    [[nodiscard]] const Entry& entry_of(std::uint32_t index) const noexcept {
        return entries_[index];
    }
    void stamp(Entry& entry, const SharedBlockOwner& owner) noexcept;
    // A leaf nothing needs any more: no lane promise left and no payload of its own.
    [[nodiscard]] bool droppable_leaf(BlockRef node) const;
    std::uint32_t drop_leaf(BlockRef node);
    void release_entry(Entry& entry) noexcept;
    [[nodiscard]] std::vector<BlockRef> ownerless_leaves() const;
    [[nodiscard]] std::optional<BlockRef> oldest_served_leaf() const;

    SharedBlockStoreHooks hooks_;
    BlockIndex index_;
    std::uint32_t capacity_ = 0;
    std::vector<Entry> entries_;
    // Every node this store interned, in insertion order. BlockIndex is the topology authority but
    // does not enumerate, and eviction has to scan candidates.
    std::vector<BlockRef> live_;
};

} // namespace ninfer::runtime::prefix_cache
