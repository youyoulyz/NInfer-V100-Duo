// End-to-end check of the Host RAM KV tier at the real V100-duo text-KV geometry.
//
// Geometry: Qwen3.6/3.8 27B, tp2, int8 group-64 KV. 16 full-attention layers, 2 KV heads per
// device, head_dim 256 -> 4 planes per layer (I8 K, I8 V, FP16 K-scale, FP16 V-scale), page-major.
// That is the plane list `DecoderStateSpec` produces (`layouts_impl.h` / `27b/impl/config.h`), so
// one page is 16,896 B/token * 64 tokens = 1.03 MiB per device.
//
// What is checked:
//   1. a real 8,192-token (128 page, ~132 MiB) KV allocation is copied into pinned host RAM;
//   2. the destination pointer is cudaMemoryTypeHost and process RSS grows by the KV size;
//   3. releasing the resident allocation returns every device page to the pool;
//   4. the same host image restores byte-exactly into TWO different live destination allocations,
//      so the image is keyed on content, not on the physical page ids it was parked from.

#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/host_kv_offload.h"
#include "core/kv_page_geometry.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kLayers   = 16;   // full-attention layers
constexpr std::int32_t kKvHeads   = 2;    // 4 KV heads / tp2
constexpr std::int32_t kHeadDim   = 256;
constexpr std::int32_t kQuant     = 64;
constexpr std::uint32_t kPages    = 128;  // 8,192 tokens at 64 tokens/page
constexpr std::uint32_t kCapacity = 512;
constexpr std::uint32_t kStaging  = 32;   // pages a tier holds while it streams one lane out

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool expect(bool condition, const std::string& label) {
    if (condition) { return true; }
    std::cerr << label << " failed\n";
    return false;
}

bool cuda_ok(cudaError_t error, const char* label) {
    if (error == cudaSuccess) { return true; }
    std::cerr << label << ": " << cudaGetErrorString(error) << '\n';
    return false;
}

std::size_t process_rss_bytes() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") {
            std::size_t kb = 0;
            status >> kb;
            return kb * 1024;
        }
        std::string rest;
        std::getline(status, rest);
    }
    return 0;
}

std::vector<ninfer::PagedKVPlaneSpec> text_kv_planes() {
    std::vector<ninfer::PagedKVPlaneSpec> planes;
    planes.reserve(kLayers * 4ULL);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        planes.push_back({ninfer::DType::I8, kHeadDim, kKvHeads, 256});
        planes.push_back({ninfer::DType::I8, kHeadDim, kKvHeads, 256});
        planes.push_back({ninfer::DType::FP16, kHeadDim / kQuant, kKvHeads, 256});
        planes.push_back({ninfer::DType::FP16, kHeadDim / kQuant, kKvHeads, 256});
    }
    return planes;
}

std::uint8_t pattern(std::size_t plane, std::size_t logical_page) {
    return static_cast<std::uint8_t>((plane * 37U + logical_page * 11U + 1U) & 0xFFU);
}

std::vector<std::int32_t> ids_of(const ninfer::PagedKVAllocation& allocation) {
    return std::vector<std::int32_t>(allocation.page_ids().begin(), allocation.page_ids().end());
}

} // namespace

int main() {
    int devices                 = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&devices);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver ||
        (count_err == cudaSuccess && devices == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    std::vector<ninfer::PagedKVPlaneSpec> planes = text_kv_planes();
    ninfer::LayoutBuilder builder;
    const ninfer::PagedKVPoolLayout pool_layout = ninfer::plan_paged_kv_pool(
        builder, {.page_group_count      = kCapacity,
                  .logical_page_capacity = kCapacity,
                  .table_rows            = 2,
                  .plane_order           = ninfer::PagedKVPlaneOrder::PageMajor,
                  .planes                = planes});
    ninfer::DeviceArena device_arena(builder.finish(256));
    ninfer::PagedKVPool pool({device_arena.base(), device_arena.capacity()}, pool_layout);

    const ninfer::KVPageGeometry geometry = ninfer::kv_page_geometry(
        ninfer::PagedKVPoolSpec{.page_group_count      = kCapacity,
                                .logical_page_capacity = kCapacity,
                                .table_rows            = 2,
                                .plane_order           = ninfer::PagedKVPlaneOrder::PageMajor,
                                .planes                = planes});
    const ninfer::HostKVPageLayout host_layout = ninfer::plan_host_kv_page_layout(geometry);
    const std::size_t device_page_bytes        = pool.plane(0).nb[3] + pool.plane(1).nb[3] +
                                          pool.plane(2).nb[3] + pool.plane(3).nb[3];
    failures += expect(device_page_bytes * kLayers == host_layout.page_stride,
                       "host page stride equals one device page") ? 0 : 1;
    failures += expect(host_layout.page_stride == 1081344, "production page is 1.03 MiB") ? 0 : 1;

    const std::size_t expected_bytes = host_layout.page_stride * kPages;

    // 1. Park a real device KV allocation into pinned host RAM.
    ninfer::PagedKVAllocation resident = pool.reserve(kPages);
    resident.materialize_pages(kPages, ctx.stream);
    resident.bind_row(0, ctx.stream);
    const std::vector<std::int32_t> source_ids = ids_of(resident);

    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const std::size_t pitch = pool.plane(plane).nb[3];
        const std::size_t width = host_layout.planes[plane].page_payload_bytes;
        auto* base              = static_cast<std::uint8_t*>(pool.plane(plane).data);
        for (std::uint32_t page = 0; page < kPages; ++page) {
            failures += cuda_ok(cudaMemsetAsync(
                                    base + static_cast<std::size_t>(source_ids[page]) * pitch,
                                    pattern(plane, page), width, ctx.stream),
                                "fill page")
                            ? 0
                            : 1;
        }
    }
    failures += cuda_ok(cudaStreamSynchronize(ctx.stream), "fill sync") ? 0 : 1;

    const std::size_t rss_before = process_rss_bytes();
    // Room for the whole lane image plus the bounded staging image a tier streams through.
    ninfer::HostKVArena host_arena(host_layout.page_stride * (kPages + kStaging), {&host_layout, 1});
    ninfer::HostKVOffload offload(host_arena, geometry);
    failures += expect(offload.accepts(pool), "offload accepts pool") ? 0 : 1;

    const auto park_start = std::chrono::steady_clock::now();
    std::optional<ninfer::HostKVImage> image = offload.park(pool, resident, ctx.stream);
    const auto park_end = std::chrono::steady_clock::now();
    if (!image) {
        failures += fail("park returned no image");
        std::cout << "FAIL\n";
        return failures == 0 ? 0 : 1;
    }
    const std::size_t rss_after = process_rss_bytes();

    // 2. The bytes live in pinned CPU memory and are charged to this process.
    cudaPointerAttributes attributes{};
    failures += cuda_ok(cudaPointerGetAttributes(&attributes, offload.data(*image)),
                        "pointer attributes")
                    ? 0
                    : 1;
    failures += expect(attributes.type == cudaMemoryTypeHost, "offloaded bytes are pinned host")
                    ? 0
                    : 1;
    failures += expect(offload.resident_bytes() == expected_bytes, "arena occupancy") ? 0 : 1;
    failures += expect(image->byte_count() == expected_bytes, "image byte count") ? 0 : 1;
    failures += expect(rss_after >= rss_before + expected_bytes - (4U << 20),
                       "process RSS grew by the offloaded bytes")
                    ? 0
                    : 1;

    // 3. The device pages really come back to the pool.
    resident.release();
    failures += expect(pool.free_pages() == kCapacity, "device pages returned to pool") ? 0 : 1;

    // 4. Restore the same host image into two different live allocations.
    const auto restore_start = std::chrono::steady_clock::now();
    ninfer::PagedKVAllocation first = pool.reserve(kPages);
    first.materialize_pages(kPages, ctx.stream);
    first.bind_row(0, ctx.stream);
    offload.restore(*image, pool, first, ctx.stream);
    first.publish_mapping(ctx.stream);
    ninfer::PagedKVAllocation second = pool.reserve(kPages);
    second.materialize_pages(kPages, ctx.stream);
    second.bind_row(1, ctx.stream);
    offload.restore(*image, pool, second, ctx.stream);
    second.publish_mapping(ctx.stream);
    failures += cuda_ok(cudaStreamSynchronize(ctx.stream), "restore sync") ? 0 : 1;
    const auto restore_end = std::chrono::steady_clock::now();

    const std::vector<std::int32_t> first_ids  = ids_of(first);
    const std::vector<std::int32_t> second_ids = ids_of(second);
    failures += expect(first_ids != second_ids, "the two restores used different pages") ? 0 : 1;

    std::size_t mismatches = 0;
    for (const std::vector<std::int32_t>& destination : {first_ids, second_ids}) {
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            const std::size_t pitch = pool.plane(plane).nb[3];
            const std::size_t width = host_layout.planes[plane].page_payload_bytes;
            std::vector<std::uint8_t> read_back(static_cast<std::size_t>(pool.plane(plane).bytes()));
            failures += cuda_ok(cudaMemcpy(read_back.data(), pool.plane(plane).data,
                                           read_back.size(), cudaMemcpyDeviceToHost),
                                "read plane")
                            ? 0
                            : 1;
            for (std::uint32_t page = 0; page < kPages; ++page) {
                const std::size_t begin = static_cast<std::size_t>(destination[page]) * pitch;
                for (std::size_t byte = 0; byte < width; ++byte) {
                    if (read_back[begin + byte] != pattern(plane, page)) { ++mismatches; }
                }
            }
        }
    }
    failures += expect(mismatches == 0, "both restores are byte-exact") ? 0 : 1;
    // The streamed round trip below reuses the second table row; the copies that needed it are done.
    second.unbind_row();

    const auto seconds = [](auto begin, auto end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    const double mib = static_cast<double>(expected_bytes) / (1024.0 * 1024.0);
    std::cout.setf(std::ios::fixed);
    std::cout.precision(1);
    std::cout << "offloaded " << mib << " MiB/device (" << kPages << " pages x "
              << (host_layout.page_stride / 1024) << " KiB) into pinned host RAM\n";
    std::cout.precision(2);
    std::cout << "park D2H " << seconds(park_start, park_end) * 1e3 << " ms ("
              << mib / seconds(park_start, park_end) << " MiB/s)\n";
    std::cout << "restore H2D x2 " << seconds(restore_start, restore_end) * 1e3 << " ms ("
              << 2 * mib / seconds(restore_start, restore_end) << " MiB/s)\n";
    std::cout.precision(1);
    std::cout << "host RSS " << (rss_before / (1024.0 * 1024.0)) << " -> "
              << (rss_after / (1024.0 * 1024.0)) << " MiB\n";

    // 5. The batched form a tier streams through: a bounded staging image is filled from one live
    //    allocation a range at a time and written back into another, at the same logical offsets.
    //    A run that is not the whole lane is what makes a lane-sized image unnecessary.
    std::optional<ninfer::HostKVImage> staging = offload.allocate(kStaging);
    failures += expect(staging.has_value(), "staging image allocated") ? 0 : 1;
    if (!staging) {
        std::cout << "FAIL\n";
        return 1;
    }
    ninfer::PagedKVAllocation streamed = pool.reserve(kPages);
    streamed.materialize_pages(kPages, ctx.stream);
    streamed.bind_row(1, ctx.stream);
    struct Range {
        std::uint32_t begin;
        std::uint32_t count;
    };
    // Full batches tile the whole allocation; the extra short run covers a partial batch at a
    // mid-lane offset, which is what a chain whose page count is not a multiple of the staging
    // depth looks like.
    for (const Range range : {Range{0, kStaging}, Range{32, kStaging}, Range{64, kStaging},
                              Range{96, kStaging}, Range{5, 13}}) {
        offload.park_range(pool, first, range.begin, range.count, *staging, 0, ctx.stream);
        offload.restore_range(*staging, 0, range.count, pool, streamed, range.begin, ctx.stream);
    }
    streamed.publish_mapping(ctx.stream);
    failures += cuda_ok(cudaStreamSynchronize(ctx.stream), "range sync") ? 0 : 1;
    cudaPointerAttributes staging_attributes{};
    failures += cuda_ok(cudaPointerGetAttributes(&staging_attributes, offload.data(*staging)),
                        "staging pointer attributes")
                    ? 0
                    : 1;
    failures += expect(staging_attributes.type == cudaMemoryTypeHost,
                       "staging bytes are pinned host")
                    ? 0
                    : 1;
    cudaMemsetAsync(offload.mutable_data(*staging), 0, host_layout.page_stride, ctx.stream);
    failures += cuda_ok(cudaStreamSynchronize(ctx.stream), "staging writable sync") ? 0 : 1;

    std::size_t range_mismatches = 0;
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const std::size_t pitch = pool.plane(plane).nb[3];
        const std::size_t width = host_layout.planes[plane].page_payload_bytes;
        std::vector<std::uint8_t> read_back(static_cast<std::size_t>(pool.plane(plane).bytes()));
        failures += cuda_ok(cudaMemcpy(read_back.data(), pool.plane(plane).data, read_back.size(),
                                       cudaMemcpyDeviceToHost),
                            "read plane")
                        ? 0
                        : 1;
        const std::vector<std::int32_t> streamed_ids = ids_of(streamed);
        for (std::uint32_t page = 0; page < kPages; ++page) {
            const std::size_t begin = static_cast<std::size_t>(streamed_ids[page]) * pitch;
            for (std::size_t byte = 0; byte < width; ++byte) {
                if (read_back[begin + byte] != pattern(plane, page)) { ++range_mismatches; }
            }
        }
    }
    failures += expect(range_mismatches == 0, "the streamed ranges are byte-exact") ? 0 : 1;

    image.reset();
    staging.reset();
    failures += expect(host_arena.occupied_bytes() == 0, "arena released after image drop") ? 0 : 1;

    std::cout << (failures == 0 ? "PASS" : "FAIL") << '\n';
    return failures == 0 ? 0 : 1;
}
