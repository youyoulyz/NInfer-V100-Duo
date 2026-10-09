#include "core/host_kv_offload.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <span>
#include <vector>

namespace ninfer {
namespace {

[[noreturn]] void fail(const char* message) { throw std::invalid_argument(message); }

void check_cuda(cudaError_t error, const char* label) {
    if (error == cudaSuccess) { return; }
    throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
}

// A device plane is one contiguous block per plane; a host page record packs every plane with its
// own stride. A segment is consecutive in BOTH the logical list and the physical page list, so it
// is one 2D copy (host record pitch vs. device plane pitch) and nothing else.
struct Segment {
    std::uint32_t logical_begin  = 0;
    std::uint32_t physical_begin = 0;
    std::uint32_t count          = 0;
};

std::vector<Segment> segments(std::span<const std::int32_t> ids) {
    std::vector<Segment> result;
    for (std::uint32_t index = 0; index < ids.size(); ++index) {
        if (!result.empty() &&
            ids[index] == static_cast<std::int32_t>(result.back().physical_begin +
                                                    result.back().count) &&
            index == result.back().logical_begin + result.back().count) {
            ++result.back().count;
        } else {
            result.push_back(Segment{index, static_cast<std::uint32_t>(ids[index]), 1U});
        }
    }
    return result;
}

// Moves `count` page records between the host image records [host_begin, host_begin + count) and
// the allocation's logical pages [logical_begin, logical_begin + count). A segment is consecutive
// in BOTH lists, so it is one 2D copy (host record pitch vs. device plane pitch) and nothing else.
void copy_pages(const HostKVPageLayout& layout, const PagedKVPool& pool,
                std::span<const std::int32_t> ids, std::uint32_t logical_begin, std::uint32_t count,
                std::byte* host_base, std::uint32_t host_begin, cudaMemcpyKind direction,
                cudaStream_t stream, const char* label) {
    const std::vector<Segment> runs = segments(ids.subspan(logical_begin, count));
    for (const Segment& run : runs) {
        for (std::size_t plane = 0; plane < layout.planes.size(); ++plane) {
            const HostKVPlaneLayout& plane_layout = layout.planes[plane];
            const std::size_t device_pitch = static_cast<std::size_t>(pool.plane(plane).nb[3]);
            const std::size_t host_offset =
                static_cast<std::size_t>(host_begin + run.logical_begin) * layout.page_stride +
                plane_layout.offset;
            std::byte* device = static_cast<std::byte*>(pool.plane(plane).data) +
                                static_cast<std::size_t>(run.physical_begin) * device_pitch;
            const bool to_host = direction == cudaMemcpyDeviceToHost;
            check_cuda(cudaMemcpy2DAsync(to_host ? host_base + host_offset : device,
                                         to_host ? layout.page_stride : device_pitch,
                                         to_host ? device : host_base + host_offset,
                                         to_host ? device_pitch : layout.page_stride,
                                         plane_layout.page_payload_bytes, run.count, direction,
                                         stream),
                       label);
        }
    }
}

} // namespace

HostKVOffload::HostKVOffload(HostKVArena& arena, const KVPageGeometry& geometry)
    : arena_(&arena), geometry_(geometry), layout_(plan_host_kv_page_layout(geometry)) {
    if (geometry_.device_plane_order != PagedKVPlaneOrder::PageMajor) {
        fail("Host KV offload requires page-major device planes");
    }
    if (arena_->layout_for(geometry_) == nullptr) {
        fail("Host KV arena does not support this geometry");
    }
}

bool HostKVOffload::accepts(const PagedKVPool& pool) const noexcept {
    if (pool.plane_count() != layout_.planes.size()) { return false; }
    for (std::size_t index = 0; index < layout_.planes.size(); ++index) {
        const Tensor& plane                   = pool.plane(index);
        const KVPlaneGeometry& plane_geometry = geometry_.planes[index];
        if (plane.dtype != plane_geometry.dtype || plane.ne[0] != plane_geometry.leading_extent ||
            plane.ne[2] != plane_geometry.head_extent) {
            return false;
        }
        if (static_cast<std::size_t>(plane.nb[3]) != layout_.planes[index].page_payload_bytes) {
            return false;
        }
    }
    return true;
}

std::optional<HostKVImage> HostKVOffload::park(const PagedKVPool& pool,
                                               const PagedKVAllocation& source,
                                               cudaStream_t stream) {
    const std::span<const std::int32_t> ids = source.page_ids();
    if (!source.valid() || !source.belongs_to(pool) || ids.empty()) { return std::nullopt; }
    if (ids.size() > std::numeric_limits<std::uint32_t>::max() || !accepts(pool)) {
        return std::nullopt;
    }

    const std::uint32_t pages                = static_cast<std::uint32_t>(ids.size());
    std::optional<HostKVAllocation> allocation = arena_->allocate(layout_, pages);
    if (!allocation) { return std::nullopt; }

    copy_pages(layout_, pool, ids, 0, pages, arena_->writable_view(*allocation).data(), 0,
               cudaMemcpyDeviceToHost, stream, "Host KV park");
    check_cuda(cudaStreamSynchronize(stream), "Host KV park synchronize");
    return HostKVImage(std::move(*allocation), pages, pages * layout_.page_stride);
}

void HostKVOffload::park_range(const PagedKVPool& pool, const PagedKVAllocation& source,
                               std::uint32_t source_begin, std::uint32_t count, HostKVImage& image,
                               std::uint32_t image_begin, cudaStream_t stream) {
    const std::span<const std::int32_t> ids = source.page_ids();
    if (!image.valid() || count == 0 || !source.valid() || !source.belongs_to(pool) ||
        !accepts(pool)) {
        fail("Host KV range park requires a live image and source");
    }
    if (static_cast<std::size_t>(source_begin) + count > ids.size() ||
        static_cast<std::size_t>(image_begin) + count > image.page_count()) {
        fail("Host KV range park is outside the source or the image");
    }
    copy_pages(layout_, pool, ids, source_begin, count, mutable_data(image), image_begin,
               cudaMemcpyDeviceToHost, stream, "Host KV range park");
}

void HostKVOffload::restore(const HostKVImage& image, const PagedKVPool& pool,
                            PagedKVAllocation& destination, cudaStream_t stream) {
    if (!image.valid() || !destination.valid() || !image.allocation_.valid()) {
        fail("Host KV restore requires a live image and destination");
    }
    if (!destination.belongs_to(pool) || !accepts(pool)) { fail("Host KV restore pool mismatch"); }
    const std::span<const std::int32_t> ids = destination.page_ids();
    if (ids.size() != image.page_count()) { fail("Host KV restore page count mismatch"); }
    restore_range(image, 0, image.page_count(), pool, destination, 0, stream);
}

void HostKVOffload::restore_range(const HostKVImage& image, std::uint32_t image_begin,
                                  std::uint32_t count, const PagedKVPool& pool,
                                  PagedKVAllocation& destination, std::uint32_t destination_begin,
                                  cudaStream_t stream) {
    const std::span<const std::int32_t> ids = destination.page_ids();
    if (!image.valid() || count == 0 || !destination.valid() || !destination.belongs_to(pool) ||
        !accepts(pool)) {
        fail("Host KV range restore requires a live image and destination");
    }
    if (static_cast<std::size_t>(image_begin) + count > image.page_count() ||
        static_cast<std::size_t>(destination_begin) + count > ids.size()) {
        fail("Host KV range restore is outside the image or the destination");
    }
    copy_pages(layout_, pool, ids, destination_begin, count,
               const_cast<std::byte*>(arena_->view(image.allocation_).data()), image_begin,
               cudaMemcpyHostToDevice, stream, "Host KV range restore");
}

std::optional<HostKVImage> HostKVOffload::allocate(std::uint32_t pages) {
    if (pages == 0) { return std::nullopt; }
    std::optional<HostKVAllocation> allocation = arena_->allocate(layout_, pages);
    if (!allocation) { return std::nullopt; }
    return HostKVImage(std::move(*allocation), pages, pages * layout_.page_stride);
}

const std::byte* HostKVOffload::data(const HostKVImage& image) const {
    if (!image.valid()) { fail("Host KV image is empty"); }
    return arena_->view(image.allocation_).data();
}

std::byte* HostKVOffload::mutable_data(HostKVImage& image) const {
    if (!image.valid() || !image.allocation_.valid()) { fail("Host KV image is empty"); }
    return arena_->writable_view(image.allocation_).data();
}

} // namespace ninfer
