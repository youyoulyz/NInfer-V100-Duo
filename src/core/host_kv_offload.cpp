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

    std::byte* const host_base = arena_->writable_view(*allocation).data();
    const std::vector<Segment> runs = segments(ids);
    for (const Segment& run : runs) {
        for (std::size_t plane = 0; plane < layout_.planes.size(); ++plane) {
            const HostKVPlaneLayout& plane_layout = layout_.planes[plane];
            const std::size_t device_pitch =
                static_cast<std::size_t>(pool.plane(plane).nb[3]);
            std::byte* host =
                host_base + static_cast<std::size_t>(run.logical_begin) * layout_.page_stride +
                plane_layout.offset;
            const std::byte* device =
                static_cast<const std::byte*>(pool.plane(plane).data) +
                static_cast<std::size_t>(run.physical_begin) * device_pitch;
            check_cuda(cudaMemcpy2DAsync(host, layout_.page_stride, device, device_pitch,
                                         plane_layout.page_payload_bytes, run.count,
                                         cudaMemcpyDeviceToHost, stream),
                       "Host KV park");
        }
    }
    check_cuda(cudaStreamSynchronize(stream), "Host KV park synchronize");
    return HostKVImage(std::move(*allocation), pages, pages * layout_.page_stride);
}

void HostKVOffload::restore(const HostKVImage& image, const PagedKVPool& pool,
                            PagedKVAllocation& destination, cudaStream_t stream) {
    if (!image.valid() || !destination.valid() || !image.allocation_.valid()) {
        fail("Host KV restore requires a live image and destination");
    }
    if (!destination.belongs_to(pool) || !accepts(pool)) { fail("Host KV restore pool mismatch"); }
    const std::span<const std::int32_t> ids = destination.page_ids();
    if (ids.size() != image.page_count()) { fail("Host KV restore page count mismatch"); }

    const std::byte* const host_base = arena_->view(image.allocation_).data();
    const std::vector<Segment> runs  = segments(ids);
    for (const Segment& run : runs) {
        for (std::size_t plane = 0; plane < layout_.planes.size(); ++plane) {
            const HostKVPlaneLayout& plane_layout = layout_.planes[plane];
            const std::size_t device_pitch =
                static_cast<std::size_t>(pool.plane(plane).nb[3]);
            const std::byte* host =
                host_base + static_cast<std::size_t>(run.logical_begin) * layout_.page_stride +
                plane_layout.offset;
            std::byte* device = static_cast<std::byte*>(pool.plane(plane).data) +
                                static_cast<std::size_t>(run.physical_begin) * device_pitch;
            check_cuda(cudaMemcpy2DAsync(device, device_pitch, host, layout_.page_stride,
                                         plane_layout.page_payload_bytes, run.count,
                                         cudaMemcpyHostToDevice, stream),
                       "Host KV restore");
        }
    }
}

const std::byte* HostKVOffload::data(const HostKVImage& image) const {
    if (!image.valid()) { fail("Host KV image is empty"); }
    return arena_->view(image.allocation_).data();
}

} // namespace ninfer
