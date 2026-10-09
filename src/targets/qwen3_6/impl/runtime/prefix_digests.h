#pragma once

// Per-frontier content digest of a prepared prompt: one 128-bit entry per token frontier, so entry
// F is a pure function of the first F tokens and of everything else that decides their KV and
// linear-attention state.
//
// It is a shortlist, not a decision. Exact token and ResidentPrefixIdentity comparison stays
// authoritative for reuse, so a collision costs a comparison and never produces a false hit.
//
// Ported from the sibling line's PrefixShortlistDigests (models/qwen3_5/program/prefix_identity)
// with the same arithmetic, domain separators and ordering, so both implementations derive the
// same digest for the same logical input.

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class PrefixDigests {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    // Derives the whole image from a prepared prompt. Throws when the prompt's metadata axes
    // disagree or a Vision item is not prefix ordered.
    void assign(const PreparedPromptData& prompt);
    // The same derivation from the pieces a parked lane keeps once its prompt is gone; `positions`
    // is the three MRoPE axes flattened with the token-count stride.
    void assign(std::span<const TokenId> tokens, std::span<const std::uint8_t> token_types,
                std::span<const std::int32_t> positions, std::span<const VisionItem> vision_items,
                std::optional<std::uint32_t> rewrite_frontier);
    void swap(PrefixDigests& other) noexcept;
    // Extends the image by generated tokens, which carry no token type and one uniform position
    // per axis. `tokens` are the committed ids in order.
    void append_generated(std::span<const TokenId> tokens, std::int32_t rope_delta);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept {
        return digests_.empty() ? 0U : digests_.size() - 1U;
    }

    [[nodiscard]] std::array<std::uint64_t, 2> at(std::size_t frontier) const;

    // The full digest image (seed plus one entry per token), for session snapshots, so a restored
    // continuation shortlists under bit-identical keys.
    [[nodiscard]] const std::vector<std::array<std::uint64_t, 2>>& image() const noexcept {
        return digests_;
    }
    void restore(std::vector<std::array<std::uint64_t, 2>> image);

private:
    std::vector<std::array<std::uint64_t, 2>> digests_;
};

} // namespace ninfer::targets::qwen3_6::detail
