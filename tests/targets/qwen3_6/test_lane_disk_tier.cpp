// Component check of the NVMe lane tier: content-addressed 64-token blocks plus the state image
// that makes a block-level resume legal.
//
// What is checked:
//   1. a spilled prefix reads back byte-exactly through a fresh set of buffers;
//   2. a resume is planned only where the state image AND the whole block chain are present;
//   3. a frontier above the prompt is never chosen, and a deeper frontier wins;
//   4. another conversation's digests never match, and an unfinished chain plans nothing;
//   5. the block that contains a partial frontier is keyed at the frontier itself.

#include "targets/qwen3_6/impl/runtime/lane_disk_tier.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace {

namespace q36 = ninfer::targets::qwen3_6;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::detail::PrefixDigests digests_for(std::size_t tokens, ninfer::TokenId base) {
    std::vector<ninfer::TokenId> ids(tokens);
    std::vector<std::uint8_t> types(tokens, 0);
    std::vector<std::int32_t> positions(3 * tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        ids[index] = base + static_cast<ninfer::TokenId>(index);
        positions[index]                 = static_cast<std::int32_t>(index);
        positions[tokens + index]        = static_cast<std::int32_t>(index);
        positions[2 * tokens + index]    = static_cast<std::int32_t>(index);
    }
    q36::detail::PrefixDigests digests;
    digests.assign(ids, types, positions, {}, std::nullopt);
    return digests;
}

// Spills a whole chain the way the Program does, as one batch of `prefix_pages` records.
bool spill_chain(q36::detail::LaneDiskTier& tier, const q36::detail::PrefixDigests& digests,
                 ninfer::DiskKVKind kind, std::uint32_t rank, std::uint32_t frontier,
                 const std::byte* pages, std::size_t stride) {
    const std::uint32_t count = q36::detail::LaneDiskTier::prefix_pages(frontier);
    return tier.spill_pages(digests, kind, rank, frontier, 0, count,
                            {pages, static_cast<std::size_t>(count) * stride});
}

std::vector<std::byte> pattern(std::size_t bytes, std::uint8_t seed) {
    std::vector<std::byte> out(bytes);
    for (std::size_t index = 0; index < bytes; ++index) {
        out[index] = static_cast<std::byte>((seed + index) & 0xFFU);
    }
    return out;
}

class TempDir {
public:
    explicit TempDir(const char* name) {
        path_ = std::filesystem::temp_directory_path() /
                (std::string(name) + "_" + std::to_string(static_cast<long>(::getpid())));
        std::filesystem::remove_all(path_);
    }
    ~TempDir() { std::filesystem::remove_all(path_); }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

} // namespace

int main() {
    constexpr std::size_t kStride      = 256;
    constexpr std::size_t kStateBytes  = 512;
    constexpr std::uint32_t kTag       = 7;

    TempDir dir("ninfer_lane_disk_tier_test");
    q36::detail::LaneDiskTier::Options options;
    options.path           = dir.path().string();
    options.capacity_bytes = 64ULL * 1024ULL * 1024ULL;
    options.text_stride    = kStride;
    options.state_bytes    = kStateBytes;
    options.tag            = kTag;
    q36::detail::LaneDiskTier tier(std::move(options));

    expect(q36::detail::LaneDiskTier::prefix_pages(0) == 0, "no frontier covers no block");
    expect(q36::detail::LaneDiskTier::prefix_pages(64) == 1, "a full block is one page");
    expect(q36::detail::LaneDiskTier::prefix_pages(65) == 2, "a partial block spans one more page");
    expect(q36::detail::LaneDiskTier::prefix_pages(200) == 4, "four pages cover 200 tokens");

    const q36::detail::PrefixDigests digests        = digests_for(320, 1000);
    const q36::detail::PrefixDigests other_digests  = digests_for(320, 5000);

    std::vector<std::byte> text(5 * kStride);
    for (std::uint32_t page = 0; page < 5; ++page) {
        const std::vector<std::byte> page_bytes = pattern(kStride, static_cast<std::uint8_t>(page));
        std::copy(page_bytes.begin(), page_bytes.end(), text.begin() + page * kStride);
    }
    const std::vector<std::byte> state = pattern(kStateBytes, 0xA0);

    expect(!tier.plan_resume(digests, 320).has_value(), "an empty store plans nothing");
    expect(spill_chain(tier, digests, ninfer::DiskKVKind::MainKV, 0, 200, text.data(), kStride) &&
               tier.spill_state(digests, 200, state),
           "spilling a prefix works");
    tier.wait_idle();

    const std::optional<q36::detail::DiskResumePoint> at_200 = tier.plan_resume(digests, 320);
    expect(at_200.has_value(), "the spilled prefix plans a resume");
    expect(at_200->frontier == 200 && at_200->text_pages == 4 && at_200->text_bytes == 4 * kStride,
           "the plan covers exactly the spilled blocks");
    expect(!tier.plan_resume(digests, 199).has_value(),
           "a frontier above the prompt is never chosen");
    expect(!tier.plan_resume(other_digests, 320).has_value(),
           "another conversation's digest never matches");

    // The block that contains frontier 200 is keyed at 200, not at its (absent) full end.
    std::vector<std::byte> loaded_text(at_200->text_pages * kStride);
    std::vector<std::byte> loaded_state(kStateBytes);
    expect(tier.load_kv_pages(digests, *at_200, ninfer::DiskKVKind::MainKV, 0, 0,
                              at_200->text_pages, loaded_text.data()),
           "the plan's blocks load");
    expect(tier.load_state(digests, *at_200, loaded_state), "the plan's state image loads");
    expect(loaded_text == std::vector<std::byte>(text.begin(), text.begin() + 4 * kStride),
           "the blocks read back byte-exactly");
    expect(loaded_state == state, "the state image reads back byte-exactly");

    // A deeper frontier wins once it has a complete chain.
    expect(spill_chain(tier, digests, ninfer::DiskKVKind::MainKV, 0, 300, text.data(), kStride) &&
               tier.spill_state(digests, 300, state),
           "spilling a deeper prefix");
    tier.wait_idle();
    const std::optional<q36::detail::DiskResumePoint> at_300 = tier.plan_resume(digests, 320);
    expect(at_300.has_value() && at_300->frontier == 300 && at_300->text_pages == 5,
           "the deepest complete frontier wins");
    const std::optional<q36::detail::DiskResumePoint> below = tier.plan_resume(digests, 250);
    expect(below.has_value() && below->frontier == 200,
           "a frontier above the prompt falls back to a shallower one");

    // A prefix that does not fit must fail instead of leaving a hole, and the failed frontier must
    // not become resumable: the store keeps only what it already held.
    TempDir small_dir("ninfer_lane_disk_tier_small_test");
    q36::detail::LaneDiskTier::Options small_options;
    small_options.path           = small_dir.path().string();
    small_options.capacity_bytes = 60000; // six 4 KiB blocks and exactly one state image
    small_options.text_stride    = 4096;
    small_options.state_bytes    = 4096;
    small_options.tag            = kTag;
    q36::detail::LaneDiskTier small(std::move(small_options));
    expect(small.state_slots() == 1, "the small tier holds exactly one state image");
    const q36::detail::PrefixDigests small_digests = digests_for(640, 10);
    std::vector<std::byte> big_text(8 * 4096);
    std::vector<std::byte> big_state(4096);
    expect(spill_chain(small, small_digests, ninfer::DiskKVKind::MainKV, 0, 384, big_text.data(),
                       4096) &&
               small.spill_state(small_digests, 384, big_state),
           "a prefix that fits spills");
    small.wait_idle();
    const std::optional<q36::detail::DiskResumePoint> at_384 = small.plan_resume(small_digests, 512);
    expect(at_384.has_value() && at_384->frontier == 384, "the complete prefix is resumable");
    // The chain is written before its state image, so a chain that does not fit publishes nothing:
    // the image that would have closed it is never written.
    expect(!spill_chain(small, small_digests, ninfer::DiskKVKind::MainKV, 0, 512, big_text.data(),
                        4096),
           "a prefix that does not fit fails instead of leaving a hole");
    small.wait_idle();
    const std::optional<q36::detail::DiskResumePoint> after_hole =
        small.plan_resume(small_digests, 512);
    expect(after_hole.has_value() && after_hole->frontier == 384,
           "a failed spill does not publish a frontier; the older one still serves");

    // A backend (MTP) chain is spilled and required one frontier behind the text chain, and each
    // family has its own budget. A text chain that fits while the backend chain does not leaves
    // nothing resumable.
    TempDir lag_dir("ninfer_lane_disk_tier_lag_test");
    const auto lag_options = [&](const std::filesystem::path& path) {
        q36::detail::LaneDiskTier::Options options;
        options.path           = path.string();
        options.capacity_bytes = 100000; // seven 8 KiB text records, three backend, one state
        options.text_stride    = 4096;
        options.backend_stride = 4096;
        options.state_bytes    = 4096;
        options.backend_lag    = 1;
        options.tag            = kTag;
        return options;
    };
    const q36::detail::PrefixDigests lag_digests = digests_for(640, 20);
    std::vector<std::byte> lag_text(6 * 4096);
    std::vector<std::byte> lag_backend(6 * 4096);
    for (std::uint32_t page = 0; page < 6; ++page) {
        const std::vector<std::byte> text_page = pattern(4096, static_cast<std::uint8_t>(page));
        const std::vector<std::byte> backend_page =
            pattern(4096, static_cast<std::uint8_t>(0x40 + page));
        std::copy(text_page.begin(), text_page.end(), lag_text.begin() + page * 4096);
        std::copy(backend_page.begin(), backend_page.end(), lag_backend.begin() + page * 4096);
    }
    const std::vector<std::byte> lag_state = pattern(4096, 0x90);

    q36::detail::LaneDiskTier lagged(lag_options(lag_dir.path()));
    expect(lagged.backend_frontier(320) == 319, "the backend chain trails by its lag");
    expect(spill_chain(lagged, lag_digests, ninfer::DiskKVKind::MainKV, 0, 256, lag_text.data(),
                       4096),
           "the text chain of the overflowing spill still fits");
    expect(!spill_chain(lagged, lag_digests, ninfer::DiskKVKind::BackendKV, 0, 255,
                        lag_backend.data(), 4096),
           "a spill whose backend chain does not fit fails");
    lagged.wait_idle();
    expect(!lagged.plan_resume(lag_digests, 320).has_value(),
           "a failed backend chain publishes no frontier");

    // The same spill one full block shorter fits both families, and the plan names both chains.
    TempDir lag_fit_dir("ninfer_lane_disk_tier_lag_fit_test");
    q36::detail::LaneDiskTier lagged_fit(lag_options(lag_fit_dir.path()));
    expect(spill_chain(lagged_fit, lag_digests, ninfer::DiskKVKind::MainKV, 0, 192,
                       lag_text.data(), 4096) &&
               spill_chain(lagged_fit, lag_digests, ninfer::DiskKVKind::BackendKV, 0, 191,
                           lag_backend.data(), 4096) &&
               lagged_fit.spill_state(lag_digests, 192, lag_state),
           "the spill whose both chains fit succeeds");
    lagged_fit.wait_idle();
    const std::optional<q36::detail::DiskResumePoint> lagged_plan =
        lagged_fit.plan_resume(lag_digests, 320);
    expect(lagged_plan.has_value() && lagged_plan->frontier == 192 &&
               lagged_plan->backend_frontier == 191 && lagged_plan->text_pages == 3 &&
               lagged_plan->backend_pages == 3,
           "the plan carries both chains and their frontiers");
    std::vector<std::byte> lag_loaded_text(3 * 4096);
    std::vector<std::byte> lag_loaded_backend(3 * 4096);
    std::vector<std::byte> lag_loaded_state(4096);
    expect(lagged_fit.load_kv_pages(lag_digests, *lagged_plan, ninfer::DiskKVKind::MainKV, 0, 0,
                                    lagged_plan->text_pages, lag_loaded_text.data()) &&
               lagged_fit.load_kv_pages(lag_digests, *lagged_plan, ninfer::DiskKVKind::BackendKV,
                                        0, 0, lagged_plan->backend_pages,
                                        lag_loaded_backend.data()) &&
               lagged_fit.load_state(lag_digests, *lagged_plan, lag_loaded_state),
           "the lagged plan loads both chains and its state");
    expect(lag_loaded_text == std::vector<std::byte>(lag_text.begin(), lag_text.begin() + 3 * 4096),
           "the text blocks read back byte-exactly");
    expect(lag_loaded_backend ==
               std::vector<std::byte>(lag_backend.begin(), lag_backend.begin() + 3 * 4096),
           "the backend blocks read back byte-exactly");
    expect(lag_loaded_state == lag_state, "the lagged state image reads back byte-exactly");

    // Tensor parallel: every rank holds a different shard of the same tokens under the same
    // digest, so a frontier is resumable only when EVERY rank holds its shard, and each rank's
    // bytes must come back as that rank's own.
    TempDir rank_dir("ninfer_lane_disk_tier_rank_test");
    q36::detail::LaneDiskTier::Options rank_options;
    rank_options.path           = rank_dir.path().string();
    rank_options.capacity_bytes = 64ULL * 1024ULL * 1024ULL;
    rank_options.text_stride    = kStride;
    rank_options.state_bytes    = kStateBytes;
    rank_options.tag            = kTag;
    rank_options.ranks          = 2;
    q36::detail::LaneDiskTier ranked(std::move(rank_options));
    const q36::detail::PrefixDigests rank_digests = digests_for(320, 77);
    std::vector<std::byte> rank_text[2];
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        rank_text[rank] = pattern(2 * kStride, static_cast<std::uint8_t>(0x11 + rank));
    }
    const std::vector<std::byte> rank_state = pattern(kStateBytes, 0xC0);
    expect(ranked.rank_tag(0) == kTag && ranked.rank_tag(1) == (kTag | (1U << 24U)),
           "each rank keys its shard under its own tag");
    expect(spill_chain(ranked, rank_digests, ninfer::DiskKVKind::MainKV, 0, 128,
                       rank_text[0].data(), kStride),
           "one rank's chain spills on its own");
    expect(!ranked.plan_resume(rank_digests, 320).has_value(),
           "a frontier one rank short is not resumable");
    expect(spill_chain(ranked, rank_digests, ninfer::DiskKVKind::MainKV, 1, 128,
                       rank_text[1].data(), kStride) &&
               ranked.spill_state(rank_digests, 128, rank_state),
           "the second rank's chain completes the frontier");
    ranked.wait_idle();
    const std::optional<q36::detail::DiskResumePoint> rank_plan =
        ranked.plan_resume(rank_digests, 320);
    expect(rank_plan.has_value() && rank_plan->frontier == 128 && rank_plan->text_pages == 2,
           "both ranks complete the plan");
    std::vector<std::byte> rank_loaded[2];
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        rank_loaded[rank].resize(2 * kStride);
        expect(ranked.load_kv_pages(rank_digests, *rank_plan, ninfer::DiskKVKind::MainKV, rank, 0,
                                    2, rank_loaded[rank].data()),
               "a rank's chain loads");
    }
    expect(rank_loaded[0] == rank_text[0] && rank_loaded[1] == rank_text[1],
           "each rank reads back its own shard");
    expect(!ranked.load_kv_pages(rank_digests, *rank_plan, ninfer::DiskKVKind::MainKV, 2, 0, 1,
                                 rank_loaded[0].data()),
           "a rank the tier does not hold is refused");

    if (failures != 0) {
        std::cerr << failures << " disk lane tier checks failed\n";
        return 1;
    }
    std::cout << "disk lane tier checks passed\n";
    return 0;
}
