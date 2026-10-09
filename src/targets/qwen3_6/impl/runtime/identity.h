#pragma once

// The disk (L3) page key for one content digest frontier.
//
// The key is exactly DiskKVIdentity{lo, hi, tag, frontier}: `{lo, hi}` is the digest entry at
// `frontier`, `tag` is the execution profile (see identity_tag.h) and `frontier` keeps two
// checkpoints that share a digest prefix apart. Capture and lookup must derive the key from this
// one function; a duplicated derivation is what once made every stored checkpoint carry a tag
// that no lookup could produce.

#include "core/disk_kv_store.h"
#include "targets/qwen3_6/impl/runtime/prefix_digests.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

[[nodiscard]] inline DiskKVIdentity make_identity(const PrefixDigests& digests, std::uint32_t tag,
                                                  std::uint32_t frontier) {
    const std::array<std::uint64_t, 2> digest = digests.at(frontier);
    return DiskKVIdentity{
        .lo = digest[0], .hi = digest[1], .tag = tag, .frontier = frontier};
}

} // namespace ninfer::targets::qwen3_6::detail
