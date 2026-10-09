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

    // Copies `count` page records of `source`, starting at its logical page `source_begin`, into
    // `image`'s records starting at `image_begin`. The caller owns both extents (the image is one
    // it allocated itself) and synchronizes; this is what a tier streaming a lane out in batches
    // uses, so a spill never needs an image as large as the whole lane.
    void park_range(const PagedKVPool& pool, const PagedKVAllocation& source,
                    std::uint32_t source_begin, std::uint32_t count, HostKVImage& image,
                    std::uint32_t image_begin, cudaStream_t stream);

    // Copies `image` into `destination`, which must be materialized to the same page count. The
    // caller republishes the block table afterwards; this method neither binds nor synchronizes.
    void restore(const HostKVImage& image, const PagedKVPool& pool, PagedKVAllocation& destination,
                 cudaStream_t stream);

    // Copies `count` records of `image`, starting at `image_begin`, into `destination`'s logical
    // pages starting at `destination_begin`. `destination` must be materialized past that range, and
    // `image` must be one this offload produced: an image is only addressable through the arena that
    // allocated it, so a caller that fills a staging image has to read it back through the same
    // offload rather than through the one that owns the destination.
    void restore_range(const HostKVImage& image, std::uint32_t image_begin, std::uint32_t count,
                       const PagedKVPool& pool, PagedKVAllocation& destination,
                       std::uint32_t destination_begin, cudaStream_t stream);

    // Reserves an image of `pages` page records without copying anything, for a caller that fills
    // it itself -- the disk tier reading a snapshot back. Nullopt when the arena cannot fit it.
    [[nodiscard]] std::optional<HostKVImage> allocate(std::uint32_t pages);

    // Pointer to one image's pinned host payload, for diagnostics and verification.
    [[nodiscard]] const std::byte* data(const HostKVImage& image) const;
    // Mutable base of an image's page records, `layout().page_stride` bytes apart.
    [[nodiscard]] std::byte* mutable_data(HostKVImage& image) const;

    // Pinned host bytes currently held by images from this offload's arena.
    [[nodiscard]] std::size_t resident_bytes() const noexcept { return arena_->occupied_bytes(); }

private:
    HostKVArena* arena_ = nullptr;
    KVPageGeometry geometry_;
    HostKVPageLayout layout_;
};

} // namespace ninfer
