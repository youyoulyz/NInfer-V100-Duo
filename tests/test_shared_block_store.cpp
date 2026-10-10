// Component check of the content-addressed shared block store: the tree lookup that replaces the
// per-lane prefix scan, the promise bookkeeping that keeps a stale lane out of a hit, the owned page
// groups that let a chain outlive its lane, and the leaf pruning that governs the store's capacity.
//
// What is checked:
//   1. a lookup finds the deepest interned node of the queried chain, and the owners of the nodes
//      below it;
//   2. an owner only promises blocks its frontier fully covers;
//   3. a lane the caller rejects is never returned, and forgetting a lane drops it everywhere;
//   4. pruning takes nodes nothing needs any more -- no promises, no payload -- and a whole unshared
//      branch goes away tip first while a payload keeps the chain below it;
//   5. capacity pressure prunes before it sacrifices anything a lane still serves;
//   6. the store owns page groups once they are attached: the holds are taken per slot, the chain is
//      readable in order, and every hold is given back when the node is dropped.

#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/shared_block_store.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace pc = ninfer::runtime::prefix_cache;
using ninfer::TokenId;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::vector<TokenId> ramp(std::size_t count, TokenId base) {
    std::vector<TokenId> tokens(count);
    for (std::size_t index = 0; index < count; ++index) {
        tokens[index] = base + static_cast<TokenId>(index);
    }
    return tokens;
}

std::vector<std::uint64_t> hashes_of(std::span<const TokenId> tokens) {
    return pc::block_lookup_hashes(tokens, {});
}

const pc::SharedServesFn kAnyOwner = [](const pc::SharedBlockOwner&, std::uint32_t) { return true; };
const pc::SharedServesFn kNoOwner  = [](const pc::SharedBlockOwner&, std::uint32_t) { return false; };

// The node of block `block` of an interned chain, read back through a prefix lookup.
std::optional<pc::BlockRef> node_at(pc::SharedBlockStore& store, std::span<const TokenId> tokens,
                                   std::uint32_t block) {
    const std::size_t count = (static_cast<std::size_t>(block) + 1U) * pc::kBlockTokens;
    const std::optional<pc::SharedBlockHit> hit =
        store.lookup(tokens.first(count), hashes_of(tokens.first(count)), {}, kAnyOwner).owner;
    if (!hit.has_value()) { return std::nullopt; }
    return hit->node;
}

pc::SharedBlockPages pages_of(std::uint32_t block) {
    pc::SharedBlockPages pages;
    pages.page[0] = static_cast<std::int32_t>(block);
    pages.page[2] = static_cast<std::int32_t>(1000 + block);
    return pages;
}

struct Holds {
    std::vector<std::pair<std::uint32_t, std::int32_t>> pages;
    std::vector<std::uint32_t> images;

    pc::SharedBlockStoreHooks hooks() {
        pc::SharedBlockStoreHooks out;
        out.hold_page = [this](std::uint32_t slot, std::int32_t page) {
            pages.emplace_back(slot, page);
        };
        out.release_page = [this](std::uint32_t slot, std::int32_t page) {
            const auto it = std::find(pages.begin(), pages.end(), std::make_pair(slot, page));
            if (it != pages.end()) { pages.erase(it); }
        };
        out.release_image = [this](std::uint32_t slot) { images.push_back(slot); };
        return out;
    }
};

void test_lookup_depth_and_owners() {
    pc::SharedBlockStore store(64);
    const std::vector<TokenId> short_chain = ramp(4 * pc::kBlockTokens, 100);
    const std::vector<TokenId> long_chain  = ramp(6 * pc::kBlockTokens, 100);
    (void)store.intern(short_chain, hashes_of(short_chain), {},
                       pc::SharedBlockOwner{1, 256, 0, pc::SharedRole::Append, 7});
    (void)store.intern(long_chain, hashes_of(long_chain), {},
                       pc::SharedBlockOwner{2, 384, 0, pc::SharedRole::Append, 9});

    const auto deep = store.lookup(short_chain, hashes_of(short_chain), {}, kAnyOwner).owner;
    expect(deep.has_value(), "a chain that was interned is found");
    if (deep) {
        expect(deep->tokens == 4U * pc::kBlockTokens, "the lookup stops at the queried chain's depth");
        expect(deep->owner_count == 2, "both lanes promise the shared prefix");
        const bool lanes_1_and_2 =
            std::any_of(deep->owners.begin(), deep->owners.begin() + deep->owner_count,
                        [](const pc::SharedBlockOwner& owner) { return owner.lane == 1; }) &&
            std::any_of(deep->owners.begin(), deep->owners.begin() + deep->owner_count,
                        [](const pc::SharedBlockOwner& owner) { return owner.lane == 2; });
        expect(lanes_1_and_2, "the owner set names both lanes");
    }

    const auto longest =
        store.lookup(long_chain, hashes_of(long_chain), {}, kAnyOwner).owner;
    expect(longest.has_value() && longest->tokens == 6U * pc::kBlockTokens,
           "a deeper chain is reported at its own depth");

    const std::vector<TokenId> stranger = ramp(2 * pc::kBlockTokens, 900);
    const auto none = store.lookup(stranger, hashes_of(stranger), {}, kAnyOwner).owner;
    expect(!none.has_value(), "a chain with no interned block misses");

    const auto rejected = store.lookup(short_chain, hashes_of(short_chain), {}, kNoOwner).owner;
    expect(!rejected.has_value(), "an owner the caller rejects is never a hit");
}

void test_frontier_limits_the_promise() {
    pc::SharedBlockStore store(16);
    const std::vector<TokenId> tokens = ramp(4 * pc::kBlockTokens, 200);
    // 200 tokens cover blocks 1..3 (ends 64/128/192) and stop before block 4's end (256).
    (void)store.intern(tokens, hashes_of(tokens), {},
                       pc::SharedBlockOwner{3, 200, 0, pc::SharedRole::Append, 1});
    expect(store.node_count() == 4, "the whole chain is interned, not only the promised part");
    const auto hit = store.lookup(tokens, hashes_of(tokens), {}, kAnyOwner).owner;
    expect(hit.has_value() && hit->tokens == 3U * pc::kBlockTokens,
           "a frontier that stops inside a block does not promise that block");
}

void test_forget_and_prune() {
    pc::SharedBlockStore store(32);
    const std::vector<TokenId> shared = ramp(2 * pc::kBlockTokens, 300);
    std::vector<TokenId> extended    = shared;
    const std::vector<TokenId> tail  = ramp(2 * pc::kBlockTokens, 500);
    extended.insert(extended.end(), tail.begin(), tail.end());
    (void)store.intern(shared, hashes_of(shared), {},
                       pc::SharedBlockOwner{0, 128, 0, pc::SharedRole::Append, 1});
    (void)store.intern(extended, hashes_of(extended), {},
                       pc::SharedBlockOwner{1, 256, 0, pc::SharedRole::Append, 2});
    expect(store.node_count() == 4, "the two chains share their first two nodes");

    store.forget_lane(0);
    expect(store.prune() == 0, "the shared prefix is still served by lane 1");

    store.forget_lane(1);
    expect(store.prune() == 4, "the whole branch goes away once no lane serves it");
    expect(store.node_count() == 0, "no node survives an unserved branch");
    expect(store.free_nodes() == store.capacity(), "the index is empty again");
}

void test_capacity_prefers_unserved_leaves() {
    pc::SharedBlockStore store(3);
    const std::vector<TokenId> first  = ramp(3 * pc::kBlockTokens, 400);
    const std::vector<TokenId> second = ramp(3 * pc::kBlockTokens, 700);
    (void)store.intern(first, hashes_of(first), {},
                       pc::SharedBlockOwner{4, 192, 0, pc::SharedRole::Append, 1});
    store.forget_lane(4);
    // The second chain needs three nodes; the first three are unserved, so they are pruned rather
    // than a served prefix being sacrificed.
    (void)store.intern(second, hashes_of(second), {},
                       pc::SharedBlockOwner{5, 192, 0, pc::SharedRole::Append, 2});
    expect(store.node_count() == 3, "capacity was reclaimed from the unserved branch");
    const auto hit = store.lookup(second, hashes_of(second), {}, kAnyOwner).owner;
    expect(hit.has_value() && hit->tokens == 3U * pc::kBlockTokens, "the new chain is fully resident");
}

void test_evict_lru() {
    pc::SharedBlockStore store(6);
    const std::vector<TokenId> older = ramp(3 * pc::kBlockTokens, 800);
    const std::vector<TokenId> newer = ramp(3 * pc::kBlockTokens, 1000);
    (void)store.intern(older, hashes_of(older), {},
                       pc::SharedBlockOwner{6, 192, 0, pc::SharedRole::Append, 10});
    (void)store.intern(newer, hashes_of(newer), {},
                       pc::SharedBlockOwner{7, 192, 0, pc::SharedRole::Append, 20});
    expect(store.evict_lru(3) == 3, "the oldest leaves are the sacrifice");
    expect(!store.lookup(older, hashes_of(older), {}, kAnyOwner).owner.has_value(),
           "the older chain is gone");
    expect(store.lookup(newer, hashes_of(newer), {}, kAnyOwner).owner.has_value(),
           "the newer chain survives");
}

void test_owned_pages_and_payload() {
    pc::SharedBlockStore store(16);
    Holds holds;
    store.set_hooks(holds.hooks());

    const std::vector<TokenId> tokens = ramp(4 * pc::kBlockTokens, 1300);
    const std::vector<std::uint64_t> hashes = hashes_of(tokens);
    (void)store.intern(tokens, hashes, {},
                       pc::SharedBlockOwner{2, 4 * pc::kBlockTokens, 0,
                                            pc::SharedRole::Checkpoint, 1});
    for (std::uint32_t block = 0; block < 4; ++block) {
        const auto node = node_at(store, tokens, block);
        expect(node.has_value(), "every block of an interned chain is reachable");
        if (node) { store.attach_pages(*node, pages_of(block)); }
    }
    expect(holds.pages.size() == 8, "each block holds one page group per pool it occupies");

    const auto tip = node_at(store, tokens, 3);
    const std::vector<pc::SharedBlockPages> chain = store.chain_pages(*tip);
    expect(chain.size() == 4, "the chain names every block below the tip");
    expect(chain.front().page[0] == 0 && chain.back().page[0] == 3, "the chain is in root order");
    expect(store.chain_pages(*node_at(store, tokens, 0)).size() == 1, "a chain may be one block");

    // The lane that published the chain disappears. The payload keeps the chain alive, so pruning
    // reclaims nothing and a lookup can still serve the prefix without any lane.
    store.attach_payload(*tip, pc::SharedBlockPayload{.tail = pages_of(3),
                                                      .image = 5,
                                                      .frontier = 4U * pc::kBlockTokens,
                                                      .kind = 1,
                                                      .role = pc::SharedRole::Checkpoint,
                                                      .last_use = 3});
    store.forget_lane(2);
    expect(store.prune() == 0, "a payload keeps the whole chain below it");
    const pc::SharedBlockLookup lookup = store.lookup(tokens, hashes, {}, kAnyOwner);
    expect(!lookup.owner.has_value(), "no lane serves the chain any more");
    expect(lookup.payload.has_value() && lookup.payload->payload.image == 5,
           "the payload answers instead");
    if (lookup.payload) {
        expect(lookup.payload->payload.frontier == 4U * pc::kBlockTokens,
               "the payload carries the frontier it closes");
        expect(store.chain_pages(lookup.payload->node).size() == 4,
               "the payload's chain is complete");
    }
    expect(store.stats().payloads == 1 && store.stats().pages == 8,
           "the payload and its page groups are counted");

    store.drop_payload(*tip);
    expect(holds.images.size() == 1 && holds.images.front() == 5,
           "dropping a payload returns its image slot");
    expect(store.prune() == 4, "the chain is reclaimed once the payload is gone");
    expect(holds.pages.empty(), "every page hold is given back");
}

} // namespace

int main() {
    test_lookup_depth_and_owners();
    test_frontier_limits_the_promise();
    test_forget_and_prune();
    test_capacity_prefers_unserved_leaves();
    test_evict_lru();
    test_owned_pages_and_payload();
    if (failures != 0) {
        std::cerr << failures << " shared block store check(s) failed\n";
        return 1;
    }
    std::cout << "shared block store checks passed\n";
    return 0;
}
