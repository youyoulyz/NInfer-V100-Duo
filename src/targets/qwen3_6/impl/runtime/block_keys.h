#pragma once

// Hybrid prefix-cache lookup keys of a prepared prompt: the chained 64-token block hashes and, for
// prompts with media, the cumulative Vision key each block carries.
//
// Ported from the sibling line's block_keys (models/qwen3_5/program/prefix) with the same key
// arithmetic and ordering, so both implementations derive the same keys for the same prompt.

#include "runtime/prefix_cache/block_hash.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

struct VisionTokenRange {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
    std::uint64_t key   = 0;
};

// Every Vision item's covered token range and identity key, ordered by first token. The key binds
// content, modality, grid, patches, timing and token placement: positions after an item depend
// on its grid, so later text blocks carry it too.
[[nodiscard]] std::vector<VisionTokenRange> vision_ranges(const PreparedPromptData& prompt);

[[nodiscard]] std::uint64_t accumulate_vision(std::uint64_t cumulative, std::uint64_t key) noexcept;

// The chained lookup hash of every full 64-token block of the prompt's tokens, and one cumulative
// Vision key per block (empty for a text-only prompt).
void prompt_block_keys(const PreparedPromptData& prompt, std::vector<std::uint64_t>& hashes,
                       std::vector<std::uint64_t>& extras);

} // namespace ninfer::targets::qwen3_6::detail
