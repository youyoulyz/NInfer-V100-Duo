#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace {

struct PlannedPagedCache {
    ninfer::PagedKVPoolLayout layout;
    std::size_t bytes = 0;
};

PlannedPagedCache
plan_paged_cache(std::uint32_t pages, std::uint32_t logical_pages, std::int32_t rows,
                 std::vector<ninfer::PagedKVPlaneSpec> planes,
                 ninfer::PagedKVPlaneOrder order = ninfer::PagedKVPlaneOrder::PageMajor) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = pages,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .plane_order           = order,
                                                       .planes                = std::move(planes)});
    return PlannedPagedCache{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& tensor, const std::int32_t (&expected)[4],
                const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got "
                      << tensor.ne[i] << '\n';
        }
    }
    return failures;
}

int expect_page_ids(std::span<const std::int32_t> actual,
                    std::initializer_list<std::int32_t> expected, const char* label) {
    if (actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) {
        return 0;
    }
    std::cerr << label << " page IDs differ\n";
    return 1;
}

int expect_device_page_ids(const ninfer::Tensor& row, std::initializer_list<std::int32_t> expected,
                           const char* label) {
    std::vector<std::int32_t> actual(expected.size());
    const cudaError_t err = cudaMemcpy(
        actual.data(), row.data, actual.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    return expect_page_ids(actual, expected, label);
}

int expect_zeroed_pages(ninfer::PagedKVPool& pool, ninfer::PagedKVPlaneOrder order,
                        std::span<const std::int32_t> pages, cudaStream_t stream,
                        const char* label) {
    const ninfer::Tensor& plane = pool.plane(0);
    cudaError_t err             = cudaMemsetAsync(plane.data, 0x5a, plane.bytes(), stream);
    if (err != cudaSuccess) {
        std::cerr << label << " setup failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    pool.zero_pages(pages, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " synchronization failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::vector<unsigned char> actual(plane.bytes());
    err = cudaMemcpy(actual.data(), plane.data, actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    std::vector<unsigned char> expected(actual.size(), 0x5a);
    for (const std::int32_t page : pages) {
        if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
            const std::size_t begin = static_cast<std::size_t>(page * plane.nb[3]);
            std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                      expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[3]), 0);
        } else {
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                const std::size_t begin =
                    static_cast<std::size_t>(head * plane.nb[3] + page * plane.nb[2]);
                std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                          expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[2]), 0);
            }
        }
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " cleared bytes outside the selected physical pages\n";
    return 1;
}


// Fills plane 0 with an offset-derived pattern, copies `source` onto `destination`, and then
// compares every byte against the same pattern with only the destination runs overwritten. The
// pattern makes an untouched byte, a stale byte, or a shifted copy all distinguishable.
int expect_copied_pages(ninfer::PagedKVPool& pool, ninfer::PagedKVPlaneOrder order,
                        std::initializer_list<std::int32_t> destination,
                        std::initializer_list<std::int32_t> source, cudaStream_t stream,
                        const char* label) {
    const ninfer::Tensor& plane = pool.plane(0);
    // A hash of the whole byte offset, not a small linear ramp: every page-aligned run of the pool
    // has to be distinguishable from every other one, and a pattern whose period divides the page
    // stride would make a misdirected copy compare equal.
    std::vector<unsigned char> pattern(plane.bytes());
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = static_cast<unsigned char>(((i ^ (i >> 11)) * 2654435761ULL) >> 24);
    }
    cudaError_t err = cudaMemcpyAsync(plane.data, pattern.data(), pattern.size(),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << label << " setup failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    pool.copy_pages(destination, source, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " synchronization failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    const auto runs = [&](std::int32_t page) {
        std::vector<std::pair<std::size_t, std::size_t>> out;
        if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
            out.emplace_back(static_cast<std::size_t>(page * plane.nb[3]),
                             static_cast<std::size_t>(plane.nb[3]));
        } else {
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                out.emplace_back(
                    static_cast<std::size_t>(head * plane.nb[3] + page * plane.nb[2]),
                    static_cast<std::size_t>(plane.nb[2]));
            }
        }
        return out;
    };

    std::vector<unsigned char> expected = pattern;
    const std::vector<std::int32_t> to_pages(destination);
    const std::vector<std::int32_t> from_pages(source);
    for (std::size_t index = 0; index < to_pages.size(); ++index) {
        const auto to   = runs(to_pages[index]);
        const auto from = runs(from_pages[index]);
        if (to.size() != from.size()) {
            std::cerr << label << " run geometry differs between planes\n";
            return 1;
        }
        for (std::size_t run = 0; run < to.size(); ++run) {
            std::copy_n(pattern.begin() + static_cast<std::ptrdiff_t>(from[run].first),
                        static_cast<std::ptrdiff_t>(to[run].second),
                        expected.begin() + static_cast<std::ptrdiff_t>(to[run].first));
        }
    }

    std::vector<unsigned char> actual(pattern.size());
    err = cudaMemcpy(actual.data(), plane.data, actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " page copy did not land exactly on the destination pages\n";
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    auto paged_plan = plan_paged_cache(10, 10, 2,
                                       {{ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena paged_arena(paged_plan.bytes);
    ninfer::PagedKVPool paged_pool({paged_arena.base(), paged_arena.capacity()}, paged_plan.layout);
    failures += expect_size(paged_pool.plane_count(), 4, "paged plane count");
    failures += check_shape(paged_pool.plane(0), {64, 64, 2, 10}, "paged code plane");
    failures += check_shape(paged_pool.plane(2), {1, 64, 2, 10}, "paged scale plane");
    failures += check_shape(paged_pool.block_table_row(0), {10, 1, 1, 1}, "paged block-table row");
    if (!paged_pool.plane(0).is_contiguous() || !paged_pool.plane(2).is_contiguous() ||
        !paged_pool.block_table_row(0).is_contiguous()) {
        ++failures;
        std::cerr << "Paged planes or block-table row are not contiguous\n";
    }
    const std::size_t expected_payload =
        2 * ninfer::Tensor(nullptr, ninfer::DType::I8, {64, 64, 2, 10}).bytes() +
        2 * ninfer::Tensor(nullptr, ninfer::DType::FP16, {1, 64, 2, 10}).bytes();
    failures +=
        expect_size(paged_plan.layout.payload_bytes(), expected_payload, "paged payload bytes");
    failures += expect_size(paged_plan.layout.metadata_bytes(), 10 * 2 * sizeof(std::int32_t),
                            "paged metadata bytes");
    const std::int32_t selected_pages[] = {1, 2, 6};
    failures += expect_zeroed_pages(paged_pool, ninfer::PagedKVPlaneOrder::PageMajor,
                                    selected_pages, ctx.stream, "page-major selective zero");

    auto head_major_plan = plan_paged_cache(10, 10, 1, {{ninfer::DType::BF16, 128, 8}},
                                            ninfer::PagedKVPlaneOrder::HeadMajor);
    ninfer::DeviceArena head_major_arena(head_major_plan.bytes);
    ninfer::PagedKVPool head_major_pool({head_major_arena.base(), head_major_arena.capacity()},
                                        head_major_plan.layout);
    failures += check_shape(head_major_pool.plane(0), {128, 64, 10, 8}, "head-major paged plane");
    failures += expect_zeroed_pages(head_major_pool, ninfer::PagedKVPlaneOrder::HeadMajor,
                                    selected_pages, ctx.stream, "head-major selective zero");

    // A page group can be relocated byte-for-byte inside the pool, in either plane order. This is
    // what privatizes the one page an adopted prefix shares with its owner.
    failures += expect_copied_pages(paged_pool, ninfer::PagedKVPlaneOrder::PageMajor, {0, 4}, {1, 5},
                                    ctx.stream, "page-major page copy");
    failures += expect_copied_pages(head_major_pool, ninfer::PagedKVPlaneOrder::HeadMajor, {0, 4},
                                    {1, 5}, ctx.stream, "head-major page copy");

    auto allocation_a = paged_pool.reserve(3);
    auto allocation_b = paged_pool.reserve(3);
    allocation_a.materialize_pages(3);
    allocation_b.materialize_pages(3);
    allocation_a.bind_row(0);
    allocation_b.bind_row(1);
    failures += expect_device_page_ids(allocation_a.block_table(), {0, 1, 2}, "allocation A");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "allocation B");
    allocation_a.release();

    auto allocation_c = paged_pool.reserve(6);
    allocation_c.materialize_pages(6);
    allocation_c.bind_row(0);
    failures +=
        expect_page_ids(allocation_c.page_ids(), {0, 1, 2, 6, 7, 8}, "fragmented allocation C");
    failures +=
        expect_device_page_ids(allocation_c.block_table(), {0, 1, 2, 6, 7, 8}, "fragmented row C");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "isolated row B");

    allocation_c.trim_pages(2);
    if (paged_pool.can_reserve(2)) {
        ++failures;
        std::cerr << "Unmapped entitlement was exposed as reservable capacity\n";
    }
    allocation_c.cancel_unmapped_entitlement();
    if (!paged_pool.can_reserve(5)) {
        ++failures;
        std::cerr << "Cancelled entitlement did not return reservable capacity\n";
    }

    allocation_c.unbind_row();
    const std::int32_t retained_prefix[] = {allocation_c.page_ids()[0], allocation_c.page_ids()[1]};
    const ninfer::PagedKVResize claim[]  = {
        {.allocation = &allocation_c, .mapped_pages = 2, .page_entitlement = 5}};
    ninfer::resize_paged_kv_bundle(claim);
    allocation_c.bind_row(0);
    allocation_c.materialize_pages(5);
    if (allocation_c.page_ids()[0] != retained_prefix[0] ||
        allocation_c.page_ids()[1] != retained_prefix[1]) {
        ++failures;
        std::cerr << "Retained allocation moved its prefix pages\n";
    }
    allocation_c.trim_tokens(65);
    failures += expect_size(allocation_c.mapped_page_count(), 2, "partial-tail mapped pages");
    allocation_c.trim_tokens(64);
    failures += expect_size(allocation_c.mapped_page_count(), 1, "page-aligned mapped pages");

    auto main_plan    = plan_paged_cache(4, 4, 1, {{ninfer::DType::BF16, 16, 1}});
    auto backend_plan = plan_paged_cache(2, 2, 1, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena main_arena(main_plan.bytes);
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool main_pool({main_arena.base(), main_arena.capacity()}, main_plan.layout);
    ninfer::PagedKVPool backend_pool({backend_arena.base(), backend_arena.capacity()},
                                     backend_plan.layout);
    const ninfer::PagedKVReservation impossible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 3},
        {.pool = &backend_pool, .page_entitlement = 3},
    };
    try {
        auto unused = ninfer::reserve_paged_kv_bundle(impossible_bundle);
        (void)unused;
        ++failures;
        std::cerr << "Impossible multi-pool reservation succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(main_pool.entitled_pages(), 0, "failed bundle main entitlement");
    failures += expect_size(backend_pool.entitled_pages(), 0, "failed bundle backend entitlement");

    const ninfer::PagedKVReservation possible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 2},
        {.pool = &backend_pool, .page_entitlement = 2},
    };
    auto bundle = ninfer::reserve_paged_kv_bundle(possible_bundle);
    bundle[0].materialize_pages(2);
    bundle[1].materialize_pages(2);
    const ninfer::PagedKVResize impossible_resize[] = {
        {.allocation = &bundle[0], .mapped_pages = 1, .page_entitlement = 1},
        {.allocation = &bundle[1], .mapped_pages = 1, .page_entitlement = 3},
    };
    try {
        ninfer::resize_paged_kv_bundle(impossible_resize);
        ++failures;
        std::cerr << "Impossible multi-pool resize succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(bundle[0].mapped_page_count(), 2, "failed resize main mapping");
    failures += expect_size(bundle[0].page_entitlement(), 2, "failed resize main entitlement");
    failures += expect_size(bundle[1].mapped_page_count(), 2, "failed resize backend mapping");
    failures += expect_size(bundle[1].page_entitlement(), 2, "failed resize backend entitlement");

    // A retained prefix can be mapped read-only by several lanes at once. The owner keeps
    // ownership, every borrower publishes the same physical pages into its own block-table row, and
    // the pages only rejoin the free set after the owner and the last borrower have both released.
    auto shared_plan = plan_paged_cache(8, 8, 4, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena shared_arena(shared_plan.bytes);
    ninfer::PagedKVPool shared_pool({shared_arena.base(), shared_arena.capacity()},
                                    shared_plan.layout);
    auto retained = shared_pool.reserve(2);
    retained.materialize_pages(2);
    retained.bind_row(0);
    failures += expect_size(shared_pool.free_pages(), 6, "free pages with a retained prefix");
    const std::vector<std::int32_t> prefix(retained.page_ids().begin(), retained.page_ids().end());

    auto borrower_a = shared_pool.adopt_shared(prefix);
    auto borrower_b = shared_pool.adopt_shared(prefix);
    borrower_a.bind_row(1);
    borrower_b.bind_row(2);
    failures += expect_device_page_ids(borrower_a.block_table(), {0, 1}, "borrower A row");
    failures += expect_device_page_ids(borrower_b.block_table(), {0, 1}, "borrower B row");
    failures += expect_size(borrower_a.page_entitlement(), 0, "a pure borrow owns no pages");
    failures += expect_size(shared_pool.borrowed_pages(), 4, "outstanding borrows");
    failures += expect_size(shared_pool.borrow_count(0), 2, "page 0 borrow count");
    failures += expect_size(shared_pool.free_pages(), 6, "borrowing takes no page from the owner");
    try {
        borrower_a.trim_pages(1);
        ++failures;
        std::cerr << "A borrowed mapping was trimmed\n";
    } catch (const std::logic_error&) {}

    retained.release();
    failures += expect_size(shared_pool.free_pages(), 6, "borrowed pages stay out of the free set");
    failures += expect_device_page_ids(borrower_a.block_table(), {0, 1},
                                       "borrower A survives the owner's release");
    borrower_a.release();
    failures += expect_size(shared_pool.free_pages(), 6, "pages wait for the last borrower");
    failures += expect_device_page_ids(borrower_b.block_table(), {0, 1}, "borrower B row is intact");
    borrower_b.release();
    failures += expect_size(shared_pool.free_pages(), 8, "pages return after the last release");
    failures += expect_size(shared_pool.borrowed_pages(), 0, "no outstanding borrows");
    failures += expect_size(shared_pool.entitled_pages(), 0, "borrowing owns no entitlement");

    // A lane that continues an adopted prefix owns only its suffix: its mapping is the adopted
    // pages followed by its own, trimming cannot cut into the adopted part, and releasing returns
    // only what it owns while dropping its borrows.
    auto hybrid_plan = plan_paged_cache(8, 8, 2, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena hybrid_arena(hybrid_plan.bytes);
    ninfer::PagedKVPool hybrid_pool({hybrid_arena.base(), hybrid_arena.capacity()},
                                    hybrid_plan.layout);
    auto prefix_owner = hybrid_pool.reserve(2);
    prefix_owner.materialize_pages(2);
    prefix_owner.bind_row(0);
    const std::vector<std::int32_t> owned_prefix(prefix_owner.page_ids().begin(),
                                                 prefix_owner.page_ids().end());

    auto hybrid = hybrid_pool.reserve(4);
    hybrid.adopt_prefix(owned_prefix);
    failures += expect_size(hybrid.owned_page_count(), 0, "hybrid owns nothing before it grows");
    failures += expect_size(hybrid.mapped_page_count(), 2, "hybrid maps the adopted prefix");
    failures += expect_size(hybrid.page_entitlement(), 4, "hybrid entitlement is unchanged");
    hybrid.materialize_pages(4);
    hybrid.bind_row(1);
    failures += expect_size(hybrid.owned_page_count(), 2, "hybrid owned suffix pages");
    failures += expect_page_ids(hybrid.page_ids(), {0, 1, 2, 3}, "hybrid mapping order");
    failures += expect_device_page_ids(hybrid.block_table(), {0, 1, 2, 3}, "hybrid block table");
    try {
        hybrid.trim_pages(1);
        ++failures;
        std::cerr << "An adopted prefix was trimmed\n";
    } catch (const std::logic_error&) {}
    hybrid.trim_pages(3);
    failures += expect_size(hybrid.owned_page_count(), 1, "hybrid after trimming its suffix");
    failures += expect_size(hybrid_pool.free_pages(), 5, "free pages with the hybrid mapped");

    // The owner's prefix is longer than the borrower's own entitlement: a resize may name a
    // mapping that outruns the pages this allocation owns, as long as the owned region fits.
    auto wide_plan = plan_paged_cache(8, 8, 2, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena wide_arena(wide_plan.bytes);
    ninfer::PagedKVPool wide_pool({wide_arena.base(), wide_arena.capacity()}, wide_plan.layout);
    auto wide_owner = wide_pool.reserve(4);
    wide_owner.materialize_pages(4);
    wide_owner.bind_row(0);
    const std::vector<std::int32_t> borrowed(wide_owner.page_ids().begin(),
                                             wide_owner.page_ids().end());
    auto wide_borrower = wide_pool.reserve(1);
    wide_borrower.adopt_prefix(borrowed);
    wide_borrower.materialize_pages(5);
    wide_borrower.bind_row(1);
    failures += expect_size(wide_borrower.owned_page_count(), 1, "wide borrower owned pages");
    failures += expect_size(wide_borrower.mapped_page_count(), 5, "wide borrower mapping");
    failures += expect_device_page_ids(wide_borrower.block_table(), {0, 1, 2, 3, 4},
                                       "wide borrower block table");
    try {
        wide_borrower.trim_pages(3);
        ++failures;
        std::cerr << "A wide adopted prefix was trimmed\n";
    } catch (const std::logic_error&) {}
    const ninfer::PagedKVResize wide_claim[] = {
        {.allocation = &wide_borrower, .mapped_pages = 5, .page_entitlement = 1}};
    ninfer::resize_paged_kv_bundle(wide_claim);
    failures += expect_size(wide_borrower.page_entitlement(), 1, "wide borrower entitlement");
    failures += expect_size(wide_borrower.mapped_page_count(), 5, "wide borrower resize");
    failures += expect_size(wide_pool.borrowed_pages(), 4, "wide borrower borrows");
    wide_borrower.release();
    failures += expect_size(wide_pool.borrowed_pages(), 0, "wide borrower dropped its borrows");
    wide_owner.release();
    failures += expect_size(wide_pool.free_pages(), 8, "wide pool fully returned");

    prefix_owner.release();
    failures += expect_size(hybrid_pool.free_pages(), 5, "the adopted prefix stays mapped");
    hybrid.release();
    failures += expect_size(hybrid_pool.free_pages(), 8, "hybrid release frees prefix and suffix");
    failures += expect_size(hybrid_pool.borrowed_pages(), 0, "hybrid dropped its borrows");
    failures += expect_size(hybrid_pool.entitled_pages(), 0, "hybrid owns no entitlement");

    return failures == 0 ? 0 : fail("kv cache test failed");
}
