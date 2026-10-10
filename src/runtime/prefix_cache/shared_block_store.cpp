#include "runtime/prefix_cache/shared_block_store.h"

#include <algorithm>
#include <utility>
#include <stdexcept>

namespace ninfer::runtime::prefix_cache {
namespace {

void require_chain(std::span<const TokenId> tokens, std::span<const std::uint64_t> hashes,
                   std::span<const std::uint64_t> extras) {
    const std::size_t blocks = tokens.size() / kBlockTokens;
    if (hashes.size() < blocks) {
        throw std::invalid_argument("shared block store: the hash chain does not cover every block");
    }
    if (!extras.empty() && extras.size() < blocks) {
        throw std::invalid_argument("shared block store: the extra keys do not cover every block");
    }
}

} // namespace

SharedBlockStore::SharedBlockStore(std::uint32_t max_nodes)
    : index_(max_nodes), capacity_(max_nodes), entries_(max_nodes) {
    live_.reserve(max_nodes);
}

void SharedBlockStore::stamp(Entry& entry, const SharedBlockOwner& owner) noexcept {
    for (std::uint32_t index = 0; index < entry.owner_count; ++index) {
        SharedBlockOwner& existing = entry.owners[index];
        if (existing.lane != owner.lane) { continue; }
        existing.frontier = std::max(existing.frontier, owner.frontier);
        existing.last_use = std::max(existing.last_use, owner.last_use);
        return;
    }
    if (entry.owner_count < kMaxSharedOwners) {
        entry.owners[entry.owner_count++] = owner;
        return;
    }
    // Every slot is taken. The least recently used one is the promise least likely to be asked for
    // again, so it makes room for the newest.
    std::uint32_t oldest = 0;
    for (std::uint32_t index = 1; index < kMaxSharedOwners; ++index) {
        if (entry.owners[index].last_use < entry.owners[oldest].last_use) { oldest = index; }
    }
    entry.owners[oldest] = owner;
}

BlockRef SharedBlockStore::intern(std::span<const TokenId> tokens,
                                  std::span<const std::uint64_t> hashes,
                                  std::span<const std::uint64_t> extras,
                                  const SharedBlockOwner& owner) {
    require_chain(tokens, hashes, extras);
    if (owner.lane == kNoSharedLane) {
        throw std::invalid_argument("shared block store: an owner needs a lane");
    }
    const std::size_t blocks = tokens.size() / kBlockTokens;
    if (index_.node_count() + blocks > capacity_ && !ensure_capacity(static_cast<std::uint32_t>(blocks))) {
        throw std::length_error("shared block store: node capacity exhausted");
    }
    BlockRef parent;
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::uint64_t extra = extras.empty() ? 0ULL : extras[block];
        const BlockInsert inserted =
            index_.insert(parent, hashes[block],
                          tokens.subspan(block * static_cast<std::size_t>(kBlockTokens),
                                         static_cast<std::size_t>(kBlockTokens)),
                          extra);
        if (inserted.inserted) {
            entry_of(inserted.node.index) = Entry{};
            live_.push_back(inserted.node);
        }
        Entry& entry   = entry_of(inserted.node.index);
        entry.last_use = std::max(entry.last_use, owner.last_use);
        const std::uint32_t tokens_at = static_cast<std::uint32_t>(
            (block + 1U) * static_cast<std::size_t>(kBlockTokens));
        // A node the owner's continuation does not reach is left without a promise: the lane cannot
        // serve a frontier past its own checkpoint, so stamping it would only buy a wasted check.
        if (owner.frontier >= tokens_at) { stamp(entry, owner); }
        parent = inserted.node;
    }
    return parent;
}

SharedBlockLookup SharedBlockStore::lookup(std::span<const TokenId> tokens,
                                           std::span<const std::uint64_t> hashes,
                                           std::span<const std::uint64_t> extras,
                                           const SharedServesFn& serves) const {
    require_chain(tokens, hashes, extras);
    if (!serves) { throw std::invalid_argument("shared block store: lookup needs a service check"); }
    const std::size_t blocks = tokens.size() / kBlockTokens;
    SharedBlockLookup found;
    // A node payload is only usable when every block below it still holds its page group: a claim
    // maps the whole chain or none of it, so the walk carries the chain's completeness with it.
    bool chain_complete = true;
    BlockRef parent;
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::uint64_t extra = extras.empty() ? 0ULL : extras[block];
        const std::optional<BlockRef> child =
            index_.find_child(parent, hashes[block],
                              tokens.subspan(block * static_cast<std::size_t>(kBlockTokens),
                                             static_cast<std::size_t>(kBlockTokens)),
                              extra);
        if (!child.has_value()) { break; }
        const std::uint32_t tokens_at = static_cast<std::uint32_t>(
            (block + 1U) * static_cast<std::size_t>(kBlockTokens));
        const Entry& entry = entry_of(child->index);
        chain_complete = chain_complete && entry.pages.any();
        SharedBlockHit hit{.node = *child, .tokens = tokens_at};
        for (std::uint32_t index = 0; index < entry.owner_count; ++index) {
            const SharedBlockOwner& owner = entry.owners[index];
            if (!serves(owner, tokens_at)) { continue; }
            hit.owners[hit.owner_count++] = owner;
        }
        // A node no lane serves still lets the walk continue: a longer chain may name a lane whose
        // own promise on this block was displaced by newer owners.
        if (hit.owner_count != 0) { found.owner = hit; }
        // A payload whose KV the Program parked in host RAM needs no page group below it; one whose
        // KV is the chain's device pages needs every block of the chain to still hold its own.
        if (entry.payload.valid() && (entry.payload.kv_in_host() || chain_complete)) {
            found.payload = SharedBlockPayloadView{.node = *child, .payload = entry.payload};
        }
        parent = *child;
    }
    return found;
}

std::optional<BlockRef> SharedBlockStore::find(std::span<const TokenId> tokens,
                                              std::span<const std::uint64_t> hashes,
                                              std::span<const std::uint64_t> extras) const {
    require_chain(tokens, hashes, extras);
    const std::size_t blocks = tokens.size() / kBlockTokens;
    std::optional<BlockRef> deepest;
    BlockRef parent;
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::uint64_t extra = extras.empty() ? 0ULL : extras[block];
        const std::optional<BlockRef> child =
            index_.find_child(parent, hashes[block],
                              tokens.subspan(block * static_cast<std::size_t>(kBlockTokens),
                                             static_cast<std::size_t>(kBlockTokens)),
                              extra);
        if (!child.has_value()) { break; }
        deepest = child;
        parent  = *child;
    }
    return deepest;
}

void SharedBlockStore::set_hooks(SharedBlockStoreHooks hooks) { hooks_ = std::move(hooks); }

void SharedBlockStore::attach_pages(BlockRef node, const SharedBlockPages& pages) {
    if (!index_.valid(node)) { throw std::invalid_argument("shared block store: stale node"); }
    Entry& entry = entry_of(node.index);
    if (entry.pages.any()) { return; } // The first copy is already held; this one adds nothing.
    entry.pages = pages;
    if (hooks_.hold_page) {
        for (std::uint32_t slot = 0; slot < kSharedBlockSlots; ++slot) {
            if (pages.page[slot] != kNoSharedPage) { hooks_.hold_page(slot, pages.page[slot]); }
        }
    }
}

void SharedBlockStore::attach_payload(BlockRef node, const SharedBlockPayload& payload) {
    if (!index_.valid(node)) { throw std::invalid_argument("shared block store: stale node"); }
    Entry& entry = entry_of(node.index);
    if (entry.payload.image != kNoSharedImage && entry.payload.image != payload.image &&
        hooks_.release_image) {
        hooks_.release_image(entry.payload.image);
    }
    entry.payload = payload;
}

void SharedBlockStore::drop_payload(BlockRef node) {
    if (!index_.valid(node)) { return; }
    Entry& entry = entry_of(node.index);
    if (entry.payload.valid() && hooks_.release_image) { hooks_.release_image(entry.payload.image); }
    entry.payload = SharedBlockPayload{};
}

SharedBlockPayload SharedBlockStore::payload(BlockRef node) const {
    if (!index_.valid(node)) { return {}; }
    return entry_of(node.index).payload;
}

SharedBlockPages SharedBlockStore::pages(BlockRef node) const {
    if (!index_.valid(node)) { return {}; }
    return entry_of(node.index).pages;
}

std::vector<SharedBlockPages> SharedBlockStore::chain_pages(BlockRef node) const {
    std::vector<SharedBlockPages> chain;
    BlockRef cursor = node;
    while (index_.valid(cursor)) {
        chain.push_back(entry_of(cursor.index).pages);
        if (!chain.back().any()) { return {}; }
        cursor = index_.view(cursor).parent;
    }
    std::reverse(chain.begin(), chain.end());
    return chain;
}

bool SharedBlockStore::chain_has_pages(BlockRef node) const {
    BlockRef cursor = node;
    while (index_.valid(cursor)) {
        if (!entry_of(cursor.index).pages.any()) { return false; }
        cursor = index_.view(cursor).parent;
    }
    return true;
}

bool SharedBlockStore::has_payload(BlockRef node) const noexcept {
    return index_.valid(node) && entry_of(node.index).payload.valid();
}

void SharedBlockStore::forget_lane(std::uint32_t lane) {
    if (lane == kNoSharedLane) { return; }
    for (const BlockRef node : live_) {
        Entry& entry = entry_of(node.index);
        std::uint32_t kept = 0;
        for (std::uint32_t index = 0; index < entry.owner_count; ++index) {
            if (entry.owners[index].lane == lane) { continue; }
            entry.owners[kept++] = entry.owners[index];
        }
        entry.owner_count = kept;
    }
}

bool SharedBlockStore::droppable_leaf(BlockRef node) const {
    const Entry& entry = entry_of(node.index);
    if (entry.owner_count != 0 || entry.payload.valid()) { return false; }
    const std::optional<BlockView> view_here = view(node);
    return view_here.has_value() && view_here->children == 0;
}

// Gives back everything the node was holding: the page groups it owns and the state image a claim
// would have restored. Called from the only two places that can remove a node.
void SharedBlockStore::release_entry(Entry& entry) noexcept {
    if (entry.pages.any() && hooks_.release_page) {
        for (std::uint32_t slot = 0; slot < kSharedBlockSlots; ++slot) {
            if (entry.pages.page[slot] != kNoSharedPage) {
                hooks_.release_page(slot, entry.pages.page[slot]);
            }
        }
    }
    if (entry.payload.valid() && hooks_.release_image) { hooks_.release_image(entry.payload.image); }
}

std::uint32_t SharedBlockStore::drop_leaf(BlockRef node) {
    release_entry(entry_of(node.index));
    index_.remove_subtree(node);
    entry_of(node.index) = Entry{};
    std::erase(live_, node);
    return 1U;
}

std::vector<BlockRef> SharedBlockStore::ownerless_leaves() const {
    std::vector<BlockRef> leaves;
    for (const BlockRef node : live_) {
        if (droppable_leaf(node)) { leaves.push_back(node); }
    }
    return leaves;
}

std::uint32_t SharedBlockStore::prune() {
    std::uint32_t removed = 0;
    for (;;) {
        const std::vector<BlockRef> leaves = ownerless_leaves();
        if (leaves.empty()) { return removed; }
        for (const BlockRef node : leaves) { removed += drop_leaf(node); }
    }
}

std::optional<BlockRef> SharedBlockStore::oldest_served_leaf() const {
    std::optional<BlockRef> oldest;
    std::uint64_t oldest_use = 0;
    for (const BlockRef node : live_) {
        const std::optional<BlockView> entry = view(node);
        if (!entry.has_value() || entry->children != 0) { continue; }
        const std::uint64_t use = entry_of(node.index).last_use;
        if (!oldest.has_value() || use < oldest_use) {
            oldest     = node;
            oldest_use = use;
        }
    }
    return oldest;
}

std::uint32_t SharedBlockStore::evict_lru(std::uint32_t budget) {
    std::uint32_t removed = 0;
    while (removed < budget) {
        const std::optional<BlockRef> victim = oldest_served_leaf();
        if (!victim.has_value()) { break; }
        removed += drop_leaf(*victim);
    }
    return removed;
}

bool SharedBlockStore::ensure_capacity(std::uint32_t nodes) {
    if (free_nodes() >= nodes) { return true; }
    prune();
    if (free_nodes() >= nodes) { return true; }
    evict_lru(nodes - free_nodes());
    return free_nodes() >= nodes;
}

std::optional<BlockView> SharedBlockStore::view(BlockRef node) const {
    if (!index_.valid(node)) { return std::nullopt; }
    return index_.view(node);
}

std::uint32_t SharedBlockStore::owner_count(BlockRef node) const noexcept {
    if (!index_.valid(node)) { return 0; }
    return entry_of(node.index).owner_count;
}

const std::array<SharedBlockOwner, kMaxSharedOwners>*
SharedBlockStore::owners(BlockRef node) const noexcept {
    if (!index_.valid(node)) { return nullptr; }
    return &entry_of(node.index).owners;
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> SharedBlockStore::exclusive_frontiers() const {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> out;
    for (const BlockRef node : live_) {
        const Entry& entry = entry_of(node.index);
        if (entry.owner_count != 1) { continue; }
        const std::optional<BlockView> view_here = view(node);
        if (!view_here.has_value()) { continue; }
        const std::uint32_t tokens = view_here->depth * kBlockTokens;
        const std::uint32_t lane   = entry.owners[0].lane;
        const auto existing = std::find_if(out.begin(), out.end(),
                                           [lane](const auto& item) { return item.first == lane; });
        if (existing == out.end()) {
            out.emplace_back(lane, tokens);
        } else if (existing->second < tokens) {
            existing->second = tokens;
        }
    }
    return out;
}

SharedBlockStats SharedBlockStore::stats() const {
    SharedBlockStats out;
    out.nodes = index_.node_count();
    for (const BlockRef node : live_) {
        const Entry& entry          = entry_of(node.index);
        const std::uint32_t owners_here = entry.owner_count;
        out.owner_slots += owners_here;
        if (owners_here == 0) { ++out.ownerless; }
        if (entry.payload.valid()) { ++out.payloads; }
        for (const std::int32_t page : entry.pages.page) {
            if (page != kNoSharedPage) { ++out.pages; }
        }
    }
    return out;
}

} // namespace ninfer::runtime::prefix_cache
