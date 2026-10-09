#include "runtime/prefix_cache/block_index.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime::prefix_cache {
namespace {

[[noreturn]] void invariant(const char* message) { throw std::logic_error(message); }

void require_block_tokens(std::span<const TokenId> tokens) {
    if (tokens.size() != kBlockTokens) {
        throw std::invalid_argument("prefix cache block must contain 64 tokens");
    }
}

} // namespace

BlockIndex::BlockIndex(std::uint32_t max_nodes) {
    if (max_nodes == 0) { throw std::invalid_argument("prefix cache index requires nodes"); }
    if (max_nodes > kNoBlock / 2U) {
        throw std::invalid_argument("prefix cache index capacity is not representable");
    }
    nodes_.resize(max_nodes);
    free_nodes_.resize(max_nodes);
    for (std::uint32_t index = 0; index < max_nodes; ++index) {
        free_nodes_[index] = max_nodes - 1U - index;
    }
    children_.reserve(max_nodes);
}

bool BlockIndex::valid(BlockRef node) const noexcept {
    return node.index < nodes_.size() && nodes_[node.index].occupied &&
           nodes_[node.index].generation == node.generation;
}

const BlockIndex::Node& BlockIndex::require(BlockRef node) const {
    if (!valid(node)) { invariant("stale prefix cache node reference"); }
    return nodes_[node.index];
}

BlockIndex::Node& BlockIndex::require(BlockRef node) {
    if (!valid(node)) { invariant("stale prefix cache node reference"); }
    return nodes_[node.index];
}

BlockRef BlockIndex::ref_of(std::uint32_t index) const noexcept {
    if (index == kNoBlock) { return {}; }
    return BlockRef{index, nodes_[index].generation};
}

std::uint64_t BlockIndex::child_key(std::uint32_t parent, std::uint64_t lookup_hash) noexcept {
    // Open-addressed into one flat multimap: the parent index is mixed in so two blocks with the
    // same hash under different parents cannot compare equal by accident.
    return lookup_hash ^ ((static_cast<std::uint64_t>(parent) + 1ULL) * 0x9e3779b97f4a7c15ULL);
}

std::optional<BlockRef> BlockIndex::find_child(BlockRef parent, std::uint64_t lookup_hash,
                                               std::span<const TokenId> tokens,
                                               std::uint64_t extra) const {
    require_block_tokens(tokens);
    const std::uint32_t parent_index = parent.valid() ? parent.index : kNoBlock;
    if (parent.valid()) { (void)require(parent); }
    const auto [begin, end] = children_.equal_range(child_key(parent_index, lookup_hash));
    for (auto it = begin; it != end; ++it) {
        const Node& node = nodes_[it->second];
        if (node.parent == parent_index && node.hash == lookup_hash && node.extra == extra &&
            std::equal(tokens.begin(), tokens.end(), node.tokens.begin())) {
            return ref_of(it->second);
        }
    }
    return std::nullopt;
}

BlockInsert BlockIndex::insert(BlockRef parent, std::uint64_t lookup_hash,
                               std::span<const TokenId> tokens, std::uint64_t extra,
                               std::uint8_t copies) {
    require_block_tokens(tokens);
    if (const auto existing = find_child(parent, lookup_hash, tokens, extra)) {
        ++nodes_[existing->index].pins;
        return BlockInsert{*existing, false};
    }
    if (free_nodes_.empty()) { invariant("prefix cache node capacity exhausted"); }
    const std::uint32_t index = free_nodes_.back();
    free_nodes_.pop_back();
    Node& node     = nodes_[index];
    node.occupied  = true;
    node.parent    = parent.valid() ? parent.index : kNoBlock;
    node.depth     = parent.valid() ? nodes_[parent.index].depth + 1U : 0U;
    node.hash      = lookup_hash;
    node.extra     = extra;
    std::copy(tokens.begin(), tokens.end(), node.tokens.begin());
    node.children.clear();
    node.copies = copies;
    node.pins   = 1;
    if (parent.valid()) {
        nodes_[parent.index].children.push_back(index);
    } else {
        root_children_.push_back(index);
    }
    children_.emplace(child_key(node.parent, lookup_hash), index);
    ++node_count_;
    return BlockInsert{ref_of(index), true};
}

BlockRef BlockIndex::intern_chain(std::span<const TokenId> tokens,
                                  std::span<const std::uint64_t> hashes,
                                  std::span<const std::uint64_t> extras, std::uint8_t copies) {
    const std::size_t blocks = tokens.size() / kBlockTokens;
    if (hashes.size() < blocks) {
        throw std::invalid_argument("prefix cache block hashes do not cover every full block");
    }
    if (!extras.empty() && extras.size() < blocks) {
        throw std::invalid_argument("prefix cache block extras do not cover every full block");
    }
    BlockRef parent;
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::uint64_t extra = extras.empty() ? 0ULL : extras[block];
        parent                    = insert(parent, hashes[block],
                                           tokens.subspan(block * kBlockTokens,
                                                          static_cast<std::size_t>(kBlockTokens)),
                                           extra, copies)
                     .node;
    }
    return parent;
}

BlockView BlockIndex::view(BlockRef node) const {
    const Node& entry = require(node);
    return BlockView{
        .depth       = entry.depth,
        .parent      = ref_of(entry.parent),
        .lookup_hash = entry.hash,
        .extra       = entry.extra,
        .tokens      = entry.tokens,
        .copies      = entry.copies,
        .pins        = entry.pins,
    };
}

void BlockIndex::set_copies(BlockRef node, std::uint8_t copies) noexcept {
    nodes_[node.index].copies = copies;
}

void BlockIndex::pin(BlockRef node) { ++nodes_[node.index].pins; }

void BlockIndex::unpin(BlockRef node) {
    Node& entry = nodes_[node.index];
    if (entry.pins == 0) { invariant("prefix cache node pin underflow"); }
    --entry.pins;
}

std::uint32_t BlockIndex::remove_subtree(BlockRef node) {
    if (!valid(node)) { invariant("stale prefix cache node reference"); }
    const std::uint32_t parent = nodes_[node.index].parent;
    if (parent == kNoBlock) {
        std::erase(root_children_, node.index);
    } else {
        std::erase(nodes_[parent].children, node.index);
    }
    std::vector<std::uint32_t> pending{node.index};
    std::uint32_t removed = 0;
    while (!pending.empty()) {
        const std::uint32_t index = pending.back();
        pending.pop_back();
        Node& entry             = nodes_[index];
        const std::uint32_t up  = entry.parent;
        const auto [begin, end] = children_.equal_range(child_key(up, entry.hash));
        for (auto it = begin; it != end; ++it) {
            if (it->second == index) {
                children_.erase(it);
                break;
            }
        }
        for (const std::uint32_t child : entry.children) { pending.push_back(child); }
        entry.children.clear();
        entry.occupied = false;
        ++entry.generation;
        free_nodes_.push_back(index);
        ++removed;
    }
    node_count_ -= removed;
    return removed;
}

} // namespace ninfer::runtime::prefix_cache
