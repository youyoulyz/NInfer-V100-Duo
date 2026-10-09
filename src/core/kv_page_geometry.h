#pragma once

#include "core/paged_kv_cache.h"

#include <cstdint>
#include <vector>

namespace ninfer {

// Physical plane schema for one paged K/V layer. This is upstream's `KVPlaneGeometry`; it is the
// same record as `PagedKVPlaneSpec`, which the single-layer pool already uses, so the two share
// one definition rather than diverging.
using KVPlaneGeometry = PagedKVPlaneSpec;

// Geometry of one homogeneous paged K/V family: page width in tokens, device plane order, and the
// ordered plane list. The single-layer `PagedKVPoolSpec` carries the same planes plus pool
// bookkeeping; this record is the host tier's view of exactly the planes it mirrors.
struct KVPageGeometry {
    std::uint32_t page_tokens            = kPagedKVPageSize;
    PagedKVPlaneOrder device_plane_order = PagedKVPlaneOrder::PageMajor;
    std::vector<KVPlaneGeometry> planes;

    friend bool operator==(const KVPageGeometry&, const KVPageGeometry&) = default;
};

// The tier mirrors exactly the planes of one pool. A pool spec carries those planes plus its own
// page/logical/table bookkeeping, so the geometry is a projection rather than a second definition.
[[nodiscard]] inline KVPageGeometry kv_page_geometry(const PagedKVPoolSpec& pool) {
    return KVPageGeometry{static_cast<std::uint32_t>(kPagedKVPageSize), pool.plane_order,
                          pool.planes};
}

} // namespace ninfer
