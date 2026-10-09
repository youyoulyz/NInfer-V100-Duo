#include "targets/qwen3_6/impl/runtime/prefix_digests.h"

#include <bit>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail {
namespace {

using DigestPair = std::array<std::uint64_t, 2>;

// FNV-1a-style pair: the offset and prime of the two lanes differ so a one-lane collision does not
// carry the other lane with it. Byte order and lane twist are fixed; the values are a wire format.
constexpr DigestPair kDigestOffset{1469598103934665603ULL, 7809847782465536322ULL};
constexpr DigestPair kDigestPrime{1099511628211ULL, 14029467366897019727ULL};
constexpr std::uint64_t kTokenDigestDomain   = 0x6e696e6665722d74ULL;
constexpr std::uint64_t kRewriteDigestDomain = 0x6e696e6665722d72ULL;
constexpr std::uint64_t kVisionDigestDomain  = 0x6e696e6665722d76ULL;

void mix_digest(DigestPair& digest, std::uint64_t value) noexcept {
    for (std::size_t lane = 0; lane < digest.size(); ++lane) {
        const std::uint64_t lane_value =
            lane == 0 ? value : std::rotl(value ^ 0x9e3779b97f4a7c15ULL, 29);
        for (std::uint32_t byte = 0; byte < 8; ++byte) {
            digest[lane] ^= static_cast<std::uint8_t>(lane_value >> (8U * byte));
            digest[lane] *= kDigestPrime[lane];
        }
    }
}

void mix_vision_item(DigestPair& digest, const VisionItem& item) noexcept {
    mix_digest(digest, kVisionDigestDomain);
    mix_digest(digest, static_cast<std::uint8_t>(item.modality));
    mix_digest(digest, static_cast<std::uint32_t>(item.grid.temporal));
    mix_digest(digest, static_cast<std::uint32_t>(item.grid.height));
    mix_digest(digest, static_cast<std::uint32_t>(item.grid.width));
    mix_digest(digest, item.patch_begin);
    mix_digest(digest, item.patch_count);
    for (const std::uint8_t byte : item.content_digest) { mix_digest(digest, byte); }
    mix_digest(digest, item.timestamps.size());
    for (const double timestamp : item.timestamps) {
        mix_digest(digest, std::bit_cast<std::uint64_t>(timestamp));
    }
    mix_digest(digest, item.token_spans.size());
    for (const TokenSpan& span : item.token_spans) {
        mix_digest(digest, span.begin);
        mix_digest(digest, span.count);
    }
}

void append_digest(std::vector<DigestPair>& digests, TokenId token, std::uint8_t token_type,
                   const std::array<std::int32_t, 3>& positions,
                   std::span<const std::uint32_t> rewrite_frontiers, std::size_t& next_rewrite) {
    DigestPair digest = digests.back();
    mix_digest(digest, kTokenDigestDomain);
    mix_digest(digest, static_cast<std::uint32_t>(token));
    mix_digest(digest, token_type);
    for (const std::int32_t position : positions) {
        mix_digest(digest, static_cast<std::uint32_t>(position));
    }
    // A rewrite checkpoint is folded in at the frontier it closes, so a continuation that reuses
    // the prefix sees the same key whether the checkpoint came from the prompt or a live rewrite.
    const std::size_t frontier = digests.size();
    while (next_rewrite < rewrite_frontiers.size() && rewrite_frontiers[next_rewrite] == frontier) {
        mix_digest(digest, kRewriteDigestDomain);
        mix_digest(digest, rewrite_frontiers[next_rewrite]);
        ++next_rewrite;
    }
    // Re-roll a zero lane: zero is reserved for "never written", so a freshly built entry can
    // never be mistaken for an absent one.
    for (std::uint64_t& lane : digest) {
        if (lane == 0) { lane = 1; }
    }
    digests.push_back(digest);
}

std::size_t checked_vision_end(const VisionItem& item, std::size_t prompt_tokens) {
    if (item.token_spans.empty()) {
        throw std::invalid_argument("Vision shortlist item has no token spans");
    }
    std::size_t previous_end = 0;
    for (const TokenSpan& span : item.token_spans) {
        if (span.count == 0 || span.begin > prompt_tokens ||
            span.count > prompt_tokens - span.begin || span.begin < previous_end) {
            throw std::invalid_argument("Vision shortlist item has invalid token spans");
        }
        previous_end = span.begin + span.count;
    }
    return previous_end;
}

} // namespace

void PrefixDigests::reserve(std::size_t tokens) {
    if (tokens == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("prefix digest capacity overflows size_t");
    }
    digests_.reserve(tokens + 1U);
}

void PrefixDigests::clear() noexcept { digests_.clear(); }

void PrefixDigests::assign(const PreparedPromptData& prompt) {
    assign(prompt.token_ids, prompt.token_types, prompt.positions, prompt.vision_items,
           prompt.identity.rewrite_checkpoint
               ? std::optional<std::uint32_t>(prompt.identity.rewrite_checkpoint->frontier)
               : std::nullopt);
}

void PrefixDigests::assign(std::span<const TokenId> tokens,
                           std::span<const std::uint8_t> token_types,
                           std::span<const std::int32_t> positions,
                           std::span<const VisionItem> vision_items,
                           std::optional<std::uint32_t> rewrite_frontier) {
    const std::size_t count = tokens.size();
    if (token_types.size() != count || positions.size() != 3U * count) {
        throw std::invalid_argument("prepared prompt digest metadata has an invalid shape");
    }
    std::array<std::uint32_t, 1> rewrite_holder{};
    std::span<const std::uint32_t> rewrite_frontiers;
    if (rewrite_frontier) {
        if (*rewrite_frontier == 0 || *rewrite_frontier > count) {
            throw std::invalid_argument("rewrite checkpoint must sit inside the prompt");
        }
        rewrite_holder[0] = *rewrite_frontier;
        rewrite_frontiers = rewrite_holder;
    }
    digests_.clear();
    reserve(count);
    digests_.push_back(kDigestOffset);
    std::size_t next_rewrite = 0;
    std::size_t next_vision  = 0;
    std::size_t next_vision_end =
        vision_items.empty() ? 0 : checked_vision_end(vision_items.front(), count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::array<std::int32_t, 3> axes{positions[index], positions[count + index],
                                               positions[2U * count + index]};
        append_digest(digests_, tokens[index], token_types[index], axes, rewrite_frontiers,
                      next_rewrite);
        // A Vision item's KV is only complete at its last span end, so it enters the digest there.
        const std::size_t frontier = index + 1U;
        while (next_vision < vision_items.size() && next_vision_end == frontier) {
            mix_vision_item(digests_.back(), vision_items[next_vision]);
            ++next_vision;
            if (next_vision < vision_items.size()) {
                next_vision_end = checked_vision_end(vision_items[next_vision], count);
                if (next_vision_end < frontier) {
                    throw std::invalid_argument("Vision shortlist items are not prefix ordered");
                }
            }
        }
    }
    if (next_rewrite != rewrite_frontiers.size()) {
        throw std::invalid_argument("rewrite checkpoint exceeds the prompt");
    }
    if (next_vision != vision_items.size()) {
        throw std::invalid_argument("Vision shortlist item exceeds the prompt");
    }
}

void PrefixDigests::swap(PrefixDigests& other) noexcept { digests_.swap(other.digests_); }

void PrefixDigests::append_generated(std::span<const TokenId> tokens, std::int32_t rope_delta) {
    if (digests_.empty()) {
        throw std::logic_error("generated digest append has no resident prefix");
    }
    const std::size_t begin = size();
    if (tokens.size() > std::numeric_limits<std::size_t>::max() - begin) {
        throw std::overflow_error("generated digest length overflows size_t");
    }
    for (std::size_t offset = 0; offset < tokens.size(); ++offset) {
        const std::size_t index = begin + offset;
        if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("generated digest position exceeds int32");
        }
        const std::int64_t position = static_cast<std::int64_t>(index) + rope_delta;
        if (position < std::numeric_limits<std::int32_t>::min() ||
            position > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("generated digest MRoPE position exceeds int32");
        }
        const std::int32_t value = static_cast<std::int32_t>(position);
        const std::array<std::int32_t, 3> positions{value, value, value};
        std::size_t next_rewrite = 0;
        append_digest(digests_, tokens[offset], 0, positions, {}, next_rewrite);
    }
}

void PrefixDigests::truncate(std::size_t tokens) {
    if (digests_.empty() || tokens > size()) {
        throw std::out_of_range("cannot extend prefix digest by truncation");
    }
    digests_.resize(tokens + 1U);
}

std::array<std::uint64_t, 2> PrefixDigests::at(std::size_t frontier) const {
    if (digests_.empty() || frontier > size()) {
        throw std::out_of_range("prefix digest frontier exceeds resident identity");
    }
    return digests_[frontier];
}

void PrefixDigests::restore(std::vector<std::array<std::uint64_t, 2>> image) {
    if (image.empty() || image.front() != kDigestOffset) {
        throw std::invalid_argument("restored prefix digest image has no digest seed");
    }
    for (std::size_t index = 1; index < image.size(); ++index) {
        for (const std::uint64_t lane : image[index]) {
            if (lane == 0) {
                throw std::invalid_argument("restored prefix digest lane is zero");
            }
        }
    }
    digests_ = std::move(image);
}

} // namespace ninfer::targets::qwen3_6::detail
