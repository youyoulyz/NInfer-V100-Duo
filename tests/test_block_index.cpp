// Component check of the prefix-indexed 64-token block store: the chained lookup hash and the
// exact-identity node tree.
//
// What is checked:
//   1. the chain is a pure function of the prefix, and a later token cannot disturb a hash before
//      its block;
//   2. the extras axis must cover every full block;
//   3. exact identity decides a hit: a forced constant lookup hash and a differing extra key with
//      the same tokens both miss;
//   4. handles are generation checked, so a removed subtree cannot alias a recycled slot.

#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/block_index.h"

#include <array>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace pc = ninfer::runtime::prefix_cache;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <typename Body>
void expect_throws(Body&& body, std::string_view message) {
    bool threw = false;
    try {
        body();
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, message);
}

std::vector<ninfer::TokenId> ramp(std::size_t count, ninfer::TokenId base = 0) {
    std::vector<ninfer::TokenId> tokens(count);
    for (std::size_t index = 0; index < count; ++index) {
        tokens[index] = base + static_cast<ninfer::TokenId>(index);
    }
    return tokens;
}

std::array<ninfer::TokenId, pc::kBlockTokens> block_of(ninfer::TokenId value) {
    std::array<ninfer::TokenId, pc::kBlockTokens> block{};
    block.fill(value);
    return block;
}

void test_block_hash() {
    const std::vector<ninfer::TokenId> tokens = ramp(128);
    const std::vector<std::uint64_t> hashes   = pc::block_lookup_hashes(tokens, {});
    expect(hashes.size() == 2, "two full blocks produce two hashes");
    expect(hashes[0] == pc::block_lookup_hash(pc::kRootLookupHash,
                                              std::span<const ninfer::TokenId>(tokens.data(), 64), 0),
           "the first hash chains the root seed");
    expect(hashes[1] == pc::block_lookup_hash(
                            hashes[0], std::span<const ninfer::TokenId>(tokens.data() + 64, 64), 0),
           "the second hash chains the first");
    expect(pc::block_lookup_hashes(tokens, {}) == hashes, "the chain is deterministic");

    std::vector<ninfer::TokenId> later = tokens;
    later[100] += 1;
    const std::vector<std::uint64_t> later_hashes = pc::block_lookup_hashes(later, {});
    expect(later_hashes[0] == hashes[0], "a later block does not disturb an earlier hash");
    expect(later_hashes[1] != hashes[1], "a changed token changes its block hash");

    const std::vector<std::uint64_t> with_extra =
        pc::block_lookup_hashes(tokens, std::vector<std::uint64_t>{1, 0});
    expect(with_extra[0] != hashes[0], "the extra key is part of the block hash");

    expect_throws([&] { (void)pc::block_lookup_hashes(tokens, std::vector<std::uint64_t>{0}); },
                  "extras must cover every full block");
}

void test_block_index_lookup() {
    pc::BlockIndex index(64);
    const auto first  = block_of(11);
    const auto second = block_of(22);

    const pc::BlockInsert root = index.insert({}, 0x1234U, first, 0);
    expect(root.inserted && root.node.valid(), "a new block is inserted");
    expect(index.node_count() == 1 && index.root_children() == 1, "the new block is a root child");
    const pc::BlockView root_view = index.view(root.node);
    expect(root_view.depth == 0 && !root_view.parent.valid() && root_view.lookup_hash == 0x1234U &&
               root_view.pins == 1,
           "the root node records depth, parent, hash and the inserting pin");
    expect(std::equal(root_view.tokens.begin(), root_view.tokens.end(), first.begin()),
           "the node keeps the exact tokens");

    // Re-inserting the same block is a hit: it adds a pin and does not allocate.
    const pc::BlockInsert again = index.insert({}, 0x1234U, first, 0);
    expect(!again.inserted && again.node == root.node && index.node_count() == 1, "re-insert hits");
    expect(index.view(root.node).pins == 2, "a hit adds a pin");

    // A forced constant lookup hash must not produce a false hit: the tokens differ.
    expect(!index.find_child({}, 0x1234U, second, 0).has_value(),
           "a colliding hash with different tokens must miss");

    // The same tokens under a different extra key (media identity) must miss too.
    expect(!index.find_child({}, 0x1234U, first, 1).has_value(),
           "a different extra key with the same tokens must miss");

    const pc::BlockInsert child = index.insert(root.node, 0x1234U, second, 0);
    const pc::BlockView child_view = index.view(child.node);
    expect(child_view.depth == 1 && child_view.parent == root.node && index.node_count() == 2,
           "a child records its parent and depth");
    expect(index.find_child(root.node, 0x1234U, second, 0) == child.node,
           "an exact child is found");
    expect(!index.find_child({}, 0x1234U, second, 0).has_value(),
           "the same block is not reachable from another parent");

    index.set_copies(child.node, pc::block_copy_bit(pc::BlockCopy::Host) |
                                     pc::block_copy_bit(pc::BlockCopy::Disk));
    expect(index.view(child.node).copies ==
               (pc::block_copy_bit(pc::BlockCopy::Host) |
                pc::block_copy_bit(pc::BlockCopy::Disk)),
           "copies record where the block's bytes live");

    index.unpin(root.node);
    index.unpin(root.node);
    expect(index.view(root.node).pins == 0, "unpin releases the inserting pins");
    expect_throws([&] { index.unpin(root.node); }, "unpin underflow is rejected");
}

void test_block_index_intern_and_remove() {
    pc::BlockIndex index(8);
    const std::vector<ninfer::TokenId> tokens = ramp(192);
    const std::vector<std::uint64_t> hashes   = pc::block_lookup_hashes(tokens, {});

    const pc::BlockRef deepest = index.intern_chain(tokens, hashes, {});
    expect(deepest.valid(), "intern_chain returns the deepest block");
    expect(index.node_count() == 3, "intern_chain interns every full block");
    const pc::BlockView view = index.view(deepest);
    expect(view.depth == 2 && view.pins == 1, "the deepest block is pinned once");
    expect(index.intern_chain(tokens, hashes, {}) == deepest, "interning again is a hit");
    expect(index.node_count() == 3, "interning again allocates nothing");

    const pc::BlockRef middle = index.view(deepest).parent;
    const pc::BlockRef root   = index.view(middle).parent;
    expect(root.valid() && !index.view(root).parent.valid(), "the chain bottoms out at the root");

    const std::uint32_t removed = index.remove_subtree(middle);
    expect(removed == 2 && index.node_count() == 1, "remove_subtree drops the subtree");
    expect(!index.valid(deepest) && !index.valid(middle),
           "a removed subtree invalidates its handles");
    expect(index.valid(root), "the surviving ancestor keeps its handle");
    expect(index.find_child({}, hashes[0],
                            std::span<const ninfer::TokenId>(tokens.data(), 64), 0) == root,
           "the surviving root child is still reachable");

    expect(!index.intern_chain({}, {}, {}).valid(), "a prompt with no full block interns nothing");
    expect_throws(
        [&] {
            (void)index.intern_chain(tokens, std::span<const std::uint64_t>(hashes.data(), 1), {});
        },
        "a short hash chain is rejected");

    pc::BlockIndex small(1);
    const auto block = block_of(7);
    (void)small.insert({}, 1U, block, 0);
    expect_throws([&] { (void)small.insert({}, 2U, block, 0); },
                  "capacity exhaustion is rejected");
    expect_throws([&] { (void)index.insert({}, 0U, std::span<const ninfer::TokenId>(tokens.data(), 63), 0); },
                  "a short block is rejected");
}

} // namespace

int main() {
    test_block_hash();
    test_block_index_lookup();
    test_block_index_intern_and_remove();
    if (failures != 0) {
        std::cerr << failures << " prefix cache block store checks failed\n";
        return 1;
    }
    std::cout << "prefix cache block store checks passed\n";
    return 0;
}
