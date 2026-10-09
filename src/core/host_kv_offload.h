#pragma once

#include "core/host_kv_arena.h"
#include "core/kv_page_geometry.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer {

// One parked device KV image living in pinned host RAM: a HostKVArena reservation of exactly
// `page_count` page records, stored in logical page order. `byte_count` is the offloaded payload.
class HostKVImage {
public:
    HostKVImage() noexcept = default;
    ~HostKVImage()                                      = default;
    HostKVImage(const HostKVImage&)                      = delete;
    HostKVImage& operator=(const HostKVImage&)           = delete;
    HostKVImage(HostKVImage&& other) noexcept            = default;
    HostKVImage& operator=(HostKVImage&& other) noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return allocation_.valid(); }
    [[nodiscard]] std::uint32_t page_count() const noexcept { return page_count_; }
    [[nodiscard]] std::size_t byte_count() const noexcept { return byte_count_; }

private:
    friend class HostKVOffload;
    HostKVImage(HostKVAllocation&& allocation, std::uint32_t page_count,
                std::size_t byte_count) noexcept
        : allocation_(std::move(allocation)), page_count_(page_count), byte_count_(byte_count) {}

    HostKVAllocation allocation_;
    std::uint32_t page_count_ = 0;
    std::size_t byte_count_   = 0;
};

// Moves a device KV allocation's pages into pinned host RAM and back. The image is keyed on
// content: `park` releases nothing itself, and `restore` writes into whatever physical pages the
// destination allocation holds, so a restored image never depends on the original page ids.
//
// Only page-major geometry is supported: one `cudaMemcpy2D` per plane per segment that is
// consecutive in both the logical and the physical page lists.
class HostKVOffload {
public:
    HostKVOffload(HostKVArena& arena, const KVPageGeometry& geometry);

    HostKVOffload(const HostKVOffload&)            = delete;
    HostKVOffload& operator=(const HostKVOffload&) = delete;

    [[nodiscard]] const HostKVPageLayout& layout() const noexcept { return layout_; }

    // True when `pool`'s planes have this offload's dtype, extents, order, and per-page bytes.
    [[nodiscard]] bool accepts(const PagedKVPool& pool) const noexcept;

    // Copies `source` to host RAM on `stream` and synchronizes before returning, so the caller may
    // release `source` immediately afterwards. Returns nullopt when the host arena cannot fit the
    // image; the source and the arena are untouched in that case.
    [[nodiscard]] std::optional<HostKVImage> park(const PagedKVPool& pool,
                                                  const PagedKVAllocation& source,
                                                  cudaStream_t stream);

    // Copies `image` into `destination`, which must be materialized to the same page count. The
    // caller republishes the block table afterwards; this method neither binds nor synchronizes.
    void restore(const HostKVImage& image, const PagedKVPool& pool, PagedKVAllocation& destination,
                 cudaStream_t stream);

    // Pointer to one image's pinned host payload, for diagnostics and verification.
    [[nodiscard]] const std::byte* data(const HostKVImage& image) const;

    // Pinned host bytes currently held by images from this offload's arena.
    [[nodiscard]] std::size_t resident_bytes() const noexcept { return arena_->occupied_bytes(); }

private:
    HostKVArena* arena_ = nullptr;
    KVPageGeometry geometry_;
    HostKVPageLayout layout_;
};

} // namespace ninfer
