// P2 probe: does the upstream Host RAM tier (core/host_kv_arena) drop onto this tree's
// single-layer storage without the two-layer device pool?
//
// This file needs no device pool, no CUDA kernel, and no engine type. It exercises exactly the
// surface `HostKVExtentStore` allocates against: a geometry-planned pinned arena, page/stride
// accounting, suballocation splits, atomic recipe adoption, and transfer-work coalescing. The one
// bridge the port requires is `core/kv_page_geometry.h`, which is derived here from the tree's
// existing `PagedKVStorageLayout`.

#include "core/host_kv_arena.h"
#include "core/kv_page_geometry.h"
#include "core/paged_kv_storage.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool expect_eq(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return true; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return false;
}

// Mirrors upstream `decoder_state.cpp`: one KV family's plane list derived from the tree's
// resolved storage layout, not from a second pool spec type.
ninfer::KVPageGeometry geometry_from_storage(ninfer::KvCacheStorage storage, std::int32_t head_dim,
                                             std::uint32_t layers, std::int32_t kv_heads,
                                             ninfer::PagedKVPlaneOrder order) {
    const ninfer::PagedKVStorageLayout layer_storage =
        ninfer::paged_kv_storage_layout(storage, head_dim);
    ninfer::KVPageGeometry geometry;
    geometry.page_tokens        = ninfer::kPagedKVPageSize;
    geometry.device_plane_order = order;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * layer_storage.planes_per_layer());
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back(
            {layer_storage.key.data_dtype, layer_storage.key.data_leading_extent, kv_heads, 256});
        geometry.planes.push_back({layer_storage.value.data_dtype,
                                   layer_storage.value.data_leading_extent, kv_heads, 256});
        if (layer_storage.key.has_scale()) {
            geometry.planes.push_back({layer_storage.key.scale_dtype,
                                       layer_storage.key.scale_leading_extent, kv_heads, 256});
        }
        if (layer_storage.value.has_scale()) {
            geometry.planes.push_back({layer_storage.value.scale_dtype,
                                       layer_storage.value.scale_leading_extent, kv_heads, 256});
        }
    }
    return geometry;
}

} // namespace

int main() {
    int device_count            = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver ||
        (count_err == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;

    // 1. Geometry bridge: the tree's resolved storage layout yields the host tier's plane list.
    const ninfer::KVPageGeometry geometry = geometry_from_storage(
        ninfer::KvCacheStorage::Int8Group64, 256, 1, 2, ninfer::PagedKVPlaneOrder::PageMajor);
    failures += expect_eq(geometry.planes.size(), 4, "geometry plane count") ? 0 : 1;
    failures += expect_eq(geometry.planes[0].leading_extent, 256, "geometry key extent") ? 0 : 1;
    failures += expect_eq(geometry.planes[2].head_extent, 2, "geometry scale head extent") ? 0 : 1;

    // 1b. Tree-native bridge: the pool's own plane list projects to the same geometry.
    const ninfer::PagedKVPoolSpec pool_spec{
        .page_group_count      = 4,
        .logical_page_capacity = 4,
        .table_rows            = 1,
        .plane_order           = ninfer::PagedKVPlaneOrder::PageMajor,
        .planes                = geometry.planes,
    };
    failures += expect_eq(ninfer::kv_page_geometry(pool_spec) == geometry, 1, "pool projection")
                    ? 0
                    : 1;

    // 2. Page layout: packed host planes, 256-byte aligned, one stride per page.
    const ninfer::HostKVPageLayout layout = ninfer::plan_host_kv_page_layout(geometry);
    failures += expect_eq(layout.planes.size(), 4, "layout plane count") ? 0 : 1;
    failures += expect_eq(layout.planes[0].page_payload_bytes, 32768, "plane0 payload") ? 0 : 1;
    failures += expect_eq(layout.planes[1].offset, 32768, "plane1 offset") ? 0 : 1;
    failures += expect_eq(layout.planes[2].page_payload_bytes, 1024, "plane2 payload") ? 0 : 1;
    failures += expect_eq(layout.page_stride, 67584, "page stride") ? 0 : 1;
    failures += expect_eq(layout.page_stride % 256, 0, "page stride alignment") ? 0 : 1;

    // 3. Transfer work: payload counts every byte; operation counts coalescing runs.
    const ninfer::TransferWork d2h =
        ninfer::plan_host_kv_transfer_work(layout, /*pages=*/2, /*contiguous_runs=*/2);
    failures += expect_eq(d2h.payload_bytes, 135168, "D2H payload") ? 0 : 1;
    failures += expect_eq(d2h.copy_operations, 8, "D2H page-major operations") ? 0 : 1;
    const ninfer::TransferWork d2h_one_run =
        ninfer::plan_host_kv_transfer_work(layout, /*pages=*/2, /*contiguous_runs=*/1);
    failures += expect_eq(d2h_one_run.copy_operations, 4, "D2H single-run operations") ? 0 : 1;
    const ninfer::TransferWork d2d =
        ninfer::plan_device_kv_copy_work(layout, /*pages=*/2);
    failures += expect_eq(d2d.payload_bytes, 135168, "D2D payload") ? 0 : 1;
    failures += expect_eq(d2d.copy_operations, 8, "D2D operations") ? 0 : 1;

    const ninfer::KVPageGeometry head_major = geometry_from_storage(
        ninfer::KvCacheStorage::Int8Group64, 256, 1, 2, ninfer::PagedKVPlaneOrder::HeadMajor);
    const ninfer::HostKVPageLayout head_layout = ninfer::plan_host_kv_page_layout(head_major);
    const ninfer::TransferWork head_work =
        ninfer::plan_host_kv_transfer_work(head_layout, /*pages=*/1, /*contiguous_runs=*/1);
    failures += expect_eq(head_work.copy_operations, 8, "head-major per-page operations") ? 0 : 1;

    // 4. Arena lifecycle: allocate, write through a view, split, read back, release.
    const std::size_t capacity = layout.page_stride * 4;
    ninfer::HostKVArena arena(capacity, std::span<const ninfer::HostKVPageLayout>(&layout, 1));
    failures += expect_eq(arena.free_bytes(), capacity, "initial free bytes") ? 0 : 1;
    failures += expect_eq(arena.layout_for(head_major) == nullptr, 1, "unknown geometry rejected")
                    ? 0
                    : 1;
    failures += expect_eq(arena.layout_for(geometry) != nullptr, 1, "known geometry accepted")
                    ? 0
                    : 1;

    std::optional<ninfer::HostKVAllocation> allocation = arena.allocate(layout, 3);
    if (!allocation) {
        failures += fail("arena allocation of 3 pages failed");
        std::cout << (failures == 0 ? "PASS" : "FAIL") << '\n';
        return failures == 0 ? 0 : 1;
    }
    failures += expect_eq(allocation->page_count(), 3, "allocation page count") ? 0 : 1;
    failures += expect_eq(arena.occupied_bytes(), layout.page_stride * 3, "occupied bytes") ? 0 : 1;
    failures += expect_eq(arena.free_bytes(), layout.page_stride, "free after allocate") ? 0 : 1;
    failures += expect_eq(arena.can_allocate(layout, 1), 1, "can allocate one more") ? 0 : 1;
    failures += expect_eq(arena.can_allocate(layout, 2), 0, "cannot over-allocate") ? 0 : 1;

    {
        ninfer::HostKVAllocationView view = arena.writable_view(*allocation);
        failures += expect_eq(view.page_count(), 3, "view page count") ? 0 : 1;
        ninfer::HostKVAllocationView sub = view.subview(1, 2);
        failures += expect_eq(sub.page_count(), 2, "subview page count") ? 0 : 1;
        failures +=
            expect_eq(static_cast<std::size_t>(sub.data() - view.data()), layout.page_stride,
                      "subview stride")
                ? 0
                : 1;
        const std::byte marker = std::byte{0x5a};
        for (std::uint32_t page = 0; page < 3; ++page) {
            std::byte* page_data = view.data() + static_cast<std::size_t>(page) * layout.page_stride;
            page_data[0]                       = marker;
            page_data[layout.page_stride - 1] = marker;
        }
        const ninfer::HostKVAllocationConstView read = arena.view(*allocation);
        failures +=
            expect_eq(std::to_integer<unsigned>(read.data()[0]), 0x5a, "view round trip begin")
                ? 0
                : 1;
        failures += expect_eq(std::to_integer<unsigned>(
                                  read.data()[2 * layout.page_stride + layout.page_stride - 1]),
                              0x5a, "view round trip end")
                        ? 0
                        : 1;
    }

    std::pair<ninfer::HostKVAllocation, ninfer::HostKVAllocation> halves =
        arena.split(std::move(*allocation), 1);
    allocation.reset();
    failures += expect_eq(halves.first.page_count(), 1, "split left pages") ? 0 : 1;
    failures += expect_eq(halves.second.page_count(), 2, "split right pages") ? 0 : 1;

    // 5. Recipe adoption: atomically release the right half and allocate one page elsewhere.
    const ninfer::HostKVAllocationHandle release_handle = halves.second.handle();
    const ninfer::HostKVAllocationRequest request{&layout, 1};
    std::optional<ninfer::HostKVAllocationRecipe> recipe = arena.plan_after_releases(
        std::span<const ninfer::HostKVAllocationHandle>(&release_handle, 1),
        std::span<const ninfer::HostKVAllocationRequest>(&request, 1));
    if (!recipe) {
        failures += fail("plan_after_releases returned no recipe");
    } else {
        ninfer::HostKVAllocation targets[1];
        ninfer::HostKVAllocation* release_ptrs[] = {&halves.second};
        failures += expect_eq(arena.apply_recipe(std::move(*recipe),
                                                 std::span<ninfer::HostKVAllocation* const>(release_ptrs),
                                                 std::span<ninfer::HostKVAllocation>(targets)),
                              1, "apply recipe")
                        ? 0
                        : 1;
        failures += expect_eq(targets[0].page_count(), 1, "replacement pages") ? 0 : 1;
        failures += expect_eq(targets[0].handle().valid(), 1, "replacement handle") ? 0 : 1;
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL") << '\n';
    return failures == 0 ? 0 : 1;
}
