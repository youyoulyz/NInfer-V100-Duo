// P2 probe: can this tree's single-layer PagedKVPool carry the relocation role that upstream
// gives DeviceKVPageHandle, without a generation-bearing handle pool?
//
// PagedKVAllocation::page_ids() is the relocation address and publish_mapping() writes the
// execution table. The generation field upstream guards "this physical page still holds the
// content I parked". This probe pins down the three properties the tree must rely on instead:
//   1. a retained allocation's pages are not recycled (ownership is the pin);
//   2. a coalesced D2H park followed by an H2D restore into freshly reserved pages is byte-exact;
//   3. released physical page ids are recycled, so a tier image must key on content, not ids,
//      and must re-materialize a fresh allocation on restore rather than replay old ids.

#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kPages         = 12;
constexpr std::int32_t kPageBytes      = 64 * 64 * 2; // leading_extent * page_tokens * head_extent
constexpr std::uint32_t kPageTokens    = 64;

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool expect_true(bool condition, const char* label) {
    if (condition) { return true; }
    std::cerr << label << " failed\n";
    return false;
}

bool expect_eq(std::int64_t actual, std::int64_t expected, const char* label) {
    if (actual == expected) { return true; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return false;
}

bool cuda_ok(cudaError_t err, const char* label) {
    if (err == cudaSuccess) { return true; }
    std::cerr << label << ": " << cudaGetErrorString(err) << '\n';
    return false;
}

std::vector<std::int32_t> host_ids(const ninfer::PagedKVAllocation& allocation) {
    const std::span<const std::int32_t> ids = allocation.page_ids();
    return std::vector<std::int32_t>(ids.begin(), ids.end());
}

// Coalesces physical page ids into ascending contiguous runs.
std::vector<std::pair<std::int32_t, std::uint32_t>> coalesce(std::span<const std::int32_t> ids) {
    std::vector<std::int32_t> sorted(ids.begin(), ids.end());
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::pair<std::int32_t, std::uint32_t>> runs;
    for (const std::int32_t id : sorted) {
        if (!runs.empty() && runs.back().first + static_cast<std::int32_t>(runs.back().second) == id) {
            ++runs.back().second;
        } else {
            runs.emplace_back(id, 1U);
        }
    }
    return runs;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver ||
        (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    ninfer::LayoutBuilder builder;
    const ninfer::PagedKVPoolLayout layout = ninfer::plan_paged_kv_pool(
        builder, {.page_group_count      = kPages,
                  .logical_page_capacity = kPages,
                  .table_rows            = 2,
                  .plane_order           = ninfer::PagedKVPlaneOrder::PageMajor,
                  .planes                = {{ninfer::DType::I8, 64, 2, 256}}});
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout);
    const ninfer::Tensor& plane   = pool.plane(0);
    auto* const plane_bytes       = static_cast<std::uint8_t*>(plane.data);
    failures += expect_eq(plane.nb[3], kPageBytes, "plane page stride") ? 0 : 1;
    failures += expect_eq(pool.free_pages(), kPages, "initial free pages") ? 0 : 1;

    // A retained allocation: fill each page with a distinct byte so a restore can be verified.
    ninfer::PagedKVAllocation retained = pool.reserve(4);
    retained.materialize_pages(4, ctx.stream);
    retained.bind_row(0, ctx.stream);
    const std::vector<std::int32_t> retained_ids = host_ids(retained);
    for (std::size_t index = 0; index < retained_ids.size(); ++index) {
        const std::vector<std::uint8_t> page(kPageBytes,
                                             static_cast<std::uint8_t>(0x40 + index));
        failures += cuda_ok(cudaMemcpyAsync(plane_bytes +
                                                static_cast<std::size_t>(retained_ids[index]) *
                                                    kPageBytes,
                                            page.data(), kPageBytes, cudaMemcpyHostToDevice,
                                            ctx.stream),
                            "fill retained page")
                        ? 0
                        : 1;
    }
    failures += cuda_ok(cudaStreamSynchronize(ctx.stream), "fill sync") ? 0 : 1;

    // 1. Ownership pins: a second live allocation cannot hold any of the retained pages.
    ninfer::PagedKVAllocation concurrent = pool.reserve(2);
    concurrent.materialize_pages(2, ctx.stream);
    const std::vector<std::int32_t> concurrent_ids = host_ids(concurrent);
    const std::set<std::int32_t> retained_set(retained_ids.begin(), retained_ids.end());
    for (const std::int32_t id : concurrent_ids) {
        failures += expect_true(retained_set.count(id) == 0, "retained page recycled") ? 0 : 1;
    }
    failures += expect_eq(pool.free_pages(), kPages - 6, "free after retained + concurrent") ? 0 : 1;

    // 2. Park: coalesced D2H over the retained allocation's physical runs.
    std::vector<std::uint8_t> parked(static_cast<std::size_t>(kPageBytes) * retained_ids.size());
    const std::vector<std::pair<std::int32_t, std::uint32_t>> runs = coalesce(retained_ids);
    std::size_t run_copies = 0;
    for (const auto& [first, run_pages] : runs) {
        // `retained_ids` is in logical order; a contiguous physical run maps to consecutive
        // logical slots only when the allocation was materialized in one run. Locate the logical
        // offset of `first` to keep the parked buffer in logical order.
        const auto it =
            std::find(retained_ids.begin(), retained_ids.end(), first);
        const std::size_t logical_offset = static_cast<std::size_t>(it - retained_ids.begin());
        failures += cuda_ok(cudaMemcpyAsync(parked.data() + logical_offset * kPageBytes,
                                            plane_bytes +
                                                static_cast<std::size_t>(first) * kPageBytes,
                                            static_cast<std::size_t>(run_pages) * kPageBytes,
                                            cudaMemcpyDeviceToHost, ctx.stream),
                            "park run")
                        ? 0
                        : 1;
        ++run_copies;
    }
    failures += cuda_ok(cudaStreamSynchronize(ctx.stream), "park sync") ? 0 : 1;

    // Restore into a fresh allocation: new pages, content copied back, execution table rewritten.
    ninfer::PagedKVAllocation restored = pool.reserve(4);
    restored.materialize_pages(4, ctx.stream);
    restored.bind_row(1, ctx.stream);
    const std::vector<std::int32_t> restored_ids = host_ids(restored);
    for (std::size_t index = 0; index < restored_ids.size(); ++index) {
        failures += cuda_ok(cudaMemcpyAsync(plane_bytes +
                                                static_cast<std::size_t>(restored_ids[index]) *
                                                    kPageBytes,
                                            parked.data() + index * kPageBytes, kPageBytes,
                                            cudaMemcpyHostToDevice, ctx.stream),
                            "restore page")
                        ? 0
                        : 1;
    }
    restored.publish_mapping(ctx.stream);
    failures += cuda_ok(cudaStreamSynchronize(ctx.stream), "restore sync") ? 0 : 1;

    // The restored pages carry the parked bytes exactly, on a different physical set.
    failures += expect_true(restored_ids != retained_ids, "restore reused the parked pages") ? 0 : 1;
    for (std::size_t index = 0; index < restored_ids.size(); ++index) {
        std::vector<std::uint8_t> read_back(kPageBytes);
        failures += cuda_ok(cudaMemcpy(read_back.data(),
                                       plane_bytes +
                                           static_cast<std::size_t>(restored_ids[index]) * kPageBytes,
                                       kPageBytes, cudaMemcpyDeviceToHost),
                            "read restored page")
                        ? 0
                        : 1;
        failures += expect_true(read_back ==
                                    std::vector<std::uint8_t>(
                                        parked.begin() + static_cast<std::ptrdiff_t>(index * kPageBytes),
                                        parked.begin() +
                                            static_cast<std::ptrdiff_t>((index + 1) * kPageBytes)),
                                "restored bytes differ")
                        ? 0
                        : 1;
    }

    // publish_mapping() is what makes the restored content addressable by the execution table.
    const ninfer::Tensor row = pool.block_table_row(1);
    std::vector<std::int32_t> published(restored_ids.size());
    failures += cuda_ok(cudaMemcpy(published.data(), row.data,
                                   published.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "read execution table")
                    ? 0
                    : 1;
    failures += expect_true(published == restored_ids, "execution table mismatch") ? 0 : 1;

    // 3. Released page ids are recycled: an allocation can only get them back once freed.
    retained.release();
    concurrent.release();
    ninfer::PagedKVAllocation reused = pool.reserve(6);
    reused.materialize_pages(6, ctx.stream);
    const std::vector<std::int32_t> reused_ids = host_ids(reused);
    std::set<std::int32_t> released_union(retained_ids.begin(), retained_ids.end());
    released_union.insert(concurrent_ids.begin(), concurrent_ids.end());
    std::set<std::int32_t> reused_set(reused_ids.begin(), reused_ids.end());
    failures += expect_true(reused_set == released_union, "freed ids were not recycled") ? 0 : 1;

    std::cout << "coalesced park runs: " << run_copies << '\n';
    std::cout << (failures == 0 ? "PASS" : "FAIL") << '\n';
    return failures == 0 ? 0 : 1;
}
