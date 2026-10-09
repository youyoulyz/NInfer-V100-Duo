#pragma once

// Identity of the execution profile a checkpoint was captured under.
//
// A checkpoint captured under one profile must never be replayed under another: the speculative
// backend, the proposal head and the KV coding are exactly the three things that decide which
// bytes a KV page or a state image holds for the same tokens. Packing the three 8-bit fields into
// one word is the upstream convention, so a key produced here stays comparable with that lineage.

#include "ninfer/types.h"

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

[[nodiscard]] constexpr std::uint32_t identity_tag(SpeculativeBackend backend, ProposalHead head,
                                                   KvCacheStorage storage) noexcept {
    static_assert(static_cast<std::uint32_t>(SpeculativeBackend::DFlash2) < 0x100U);
    static_assert(static_cast<std::uint32_t>(ProposalHead::Optimized) < 0x100U);
    static_assert(static_cast<std::uint32_t>(KvCacheStorage::Int8Group64) < 0x100U);
    return static_cast<std::uint32_t>(backend) | (static_cast<std::uint32_t>(head) << 8U) |
           (static_cast<std::uint32_t>(storage) << 16U);
}

} // namespace ninfer::targets::qwen3_6::detail
