// End-to-end gate for the NVMe (L3) lane tier on the real artifact, at tp2 with MTP.
//
// The disk tier is the tier below the host one. It holds content-addressed 64-token KV blocks plus
// one state image per resumable frontier -- and that image is what makes a resume legal, because the
// GDN recurrent state cannot be rebuilt from KV. It is addressed by the prompt's own content, so it
// is the only tier that survives the engine that produced it.
//
// The scenario: two unrelated long prefixes fill both lanes, a third request is not admissible until
// one lane is parked into host RAM and the other replaced, and both victims reach the NVMe tier on
// the way out. The engine is then DESTROYED, taking the host arena, the device pages, the parked
// images and the resident prefix metadata with it. A second engine opens the same directory and asks
// for the same prompts. Nothing else can answer those requests: with no resident prefix and no host
// image, a resume that computes no prefill token can only have come out of the disk record.
//
// The oracle is engine 1's own cold continuation: a restored prefix must decode greedy-identically to
// the run that produced it, and the prefill counter must not move.
//
// NINFER_TEST_TP2=1 NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=/path/to/model.ninfer \
//   ctest -R disk_tier_real
//
// The scratch directory must not be a tmpfs; its default is under /var/tmp, and
// NINFER_DISK_TIER_DIR overrides it.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kCapacity = 4096;
constexpr std::uint32_t kOutput   = 8;
// Prompts sized against a 64-page pool (4096 tokens / 64): two of them fill both lanes, so the
// third is only admissible once one lane has been parked and the other replaced.
constexpr int kResidentLines = 119;
constexpr int kEvictingLines = 170;
// Every evicted lane publishes two resumable frontiers -- the prompt frontier a re-sent prompt lands
// on, and the lane's own channel end -- and each carries its own state image, so the state family's
// 10% share of this budget has to hold four of them.
constexpr std::size_t kDiskBytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, const std::filesystem::path& disk_path) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.tp                        = 2;
    options.devices                   = {0, 1};
    options.max_context               = kCapacity;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk             = 256;
    options.kv_cache                  = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph            = true;
    options.max_concurrency           = 2;
    options.host_context_bytes        = 256ULL * 1024ULL * 1024ULL;
    options.disk_kv_path              = disk_path;
    options.disk_kv_bytes             = kDiskBytes;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

ninfer::RequestOptions request_options(bool reuse, std::uint32_t count = kOutput) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens    = count;
    options.execution.allow_prefix_reuse         = reuse;
    options.execution.sampling.temperature       = 0.0F;
    options.execution.sampling.presence_penalty  = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults          = false;
    return options;
}

ninfer::PromptInput long_prompt(const std::string& tag, int lines) {
    std::string text = "Context tag " + tag + ". Review these Python functions.\n\n";
    for (int i = 0; i < lines; ++i) {
        text += "def add_" + std::to_string(i) + "(value):\n    return value + " +
                std::to_string(i) + "\n\n";
    }
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back({ninfer::MessagePartKind::Text, std::move(text), {}});
    ninfer::PromptInput input;
    input.options.enable_thinking   = false;
    input.options.preserve_thinking = true;
    input.messages.push_back(std::move(message));
    return input;
}

std::string tokens_of(const std::vector<ninfer::TokenId>& tokens) {
    std::string out;
    for (const ninfer::TokenId token : tokens) { out += std::to_string(token) + " "; }
    return out;
}

// One prompt that the first engine prefilled cold and that the second engine then has to resume.
struct Victim {
    const char* tag        = nullptr;
    std::uint32_t tokens   = 0;
    double cold_ttft       = 0.0;
    std::vector<ninfer::TokenId> continuation;
};

// Engine 1's leg: fill both lanes, then force the eviction that pushes the two victims to NVMe.
void produce(ninfer::Engine& engine, Victim& alpha, Victim& bravo, std::size_t& persisted) {
    ninfer::PreparedPrompt alpha_prompt = engine.prepare(long_prompt(alpha.tag, kResidentLines));
    const std::uint32_t alpha_tokens =
        static_cast<std::uint32_t>(alpha_prompt.debug_token_ids().size());
    ninfer::PreparedPrompt bravo_prompt = engine.prepare(long_prompt(bravo.tag, kResidentLines));
    const std::uint32_t bravo_tokens =
        static_cast<std::uint32_t>(bravo_prompt.debug_token_ids().size());
    require(alpha_tokens > 1800 && alpha_tokens < 2050,
            "the resident prompt size is outside the 64-page pool scenario window");
    require(bravo_tokens > 1800 && bravo_tokens < 2050,
            "the resident prompt size is outside the 64-page pool scenario window");

    auto alpha_handle = engine.submit(std::move(alpha_prompt), request_options(true));
    auto bravo_handle = engine.submit(std::move(bravo_prompt), request_options(true));
    const ninfer::GenerationResult alpha_result = alpha_handle.wait();
    const ninfer::GenerationResult bravo_result = bravo_handle.wait();
    require(alpha_result.prefix_reuse_path == ninfer::PrefixReusePath::FullReset &&
                bravo_result.prefix_reuse_path == ninfer::PrefixReusePath::FullReset,
            "the concurrent pair did not both cold-prefill");
    require(alpha_result.generated_token_ids.size() == kOutput &&
                bravo_result.generated_token_ids.size() == kOutput,
            "a cold run produced the wrong number of output tokens");

    const ninfer::GenerationResult evicting = engine.generate(
        engine.prepare(long_prompt("gamma", kEvictingLines)), request_options(false));
    require(evicting.generated_token_ids.size() == kOutput, "the evicting prompt output count");

    const ninfer::MemorySummary summary = engine.memory_summary();
    require(summary.host_tier_parked_bytes > 0, "the eviction did not park a lane into host RAM");
    require(summary.disk_tier_used_bytes > 0, "the eviction spilled nothing to NVMe");
    persisted = summary.disk_tier_used_bytes;
    std::cout << "after eviction: host parked=" << summary.host_tier_parked_bytes
              << " bytes, NVMe live=" << summary.disk_tier_used_bytes << " bytes\n";

    alpha.tokens       = alpha_tokens;
    alpha.cold_ttft    = alpha_result.timings.first_token_seconds;
    alpha.continuation = alpha_result.generated_token_ids;
    bravo.tokens       = bravo_tokens;
    bravo.cold_ttft    = bravo_result.timings.first_token_seconds;
    bravo.continuation = bravo_result.generated_token_ids;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    if (std::getenv("NINFER_TEST_TP2") == nullptr || artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: set NINFER_TEST_TP2=1 and NINFER_QWEN3_6_27B_NVFP4_WEIGHTS\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: the NVMe tier gate needs two CUDA devices\n";
        return 77;
    }

    const char* configured = std::getenv("NINFER_DISK_TIER_DIR");
    const std::string pattern = configured != nullptr && *configured != '\0'
                                    ? std::string(configured)
                                    : std::string("/var/tmp/ninfer_disk_tier_XXXXXX");
    std::vector<char> directory(pattern.begin(), pattern.end());
    directory.push_back('\0');
    if (::mkdtemp(directory.data()) == nullptr) {
        std::cerr << "FAIL: cannot create the NVMe tier scratch directory from " << pattern << '\n';
        return 1;
    }
    const std::filesystem::path root(directory.data());

    try {
        Victim alpha{.tag = "alpha"};
        Victim bravo{.tag = "bravo"};
        std::size_t persisted = 0;
        {
            ninfer::Engine engine(engine_options(artifact, root));
            const ninfer::MemorySummary startup = engine.memory_summary();
            require(startup.host_tier_capacity_bytes > 0,
                    "the host lane tier allocated no capacity");
            require(startup.disk_tier_capacity_bytes == kDiskBytes,
                    "the NVMe tier did not take the configured budget");
            require(startup.disk_tier_used_bytes == 0,
                    "a freshly opened NVMe tier reported live records");
            produce(engine, alpha, bravo, persisted);
        }
        std::cout << "engine 1 released: the host arena, the device pages and the parked images went "
                     "with it\n";
        require(persisted > 0, "the first engine persisted nothing");

        // Engine 2 has no device prefix, no host image and no resident metadata for either prompt,
        // so only the NVMe records can serve a resume.
        ninfer::Engine engine(engine_options(artifact, root));
        const ninfer::MemorySummary reopened = engine.memory_summary();
        require(reopened.host_tier_parked_bytes == 0, "a fresh engine reported parked host images");
        require(reopened.disk_tier_used_bytes == persisted,
                "the reopened NVMe tier did not reload the records the first engine persisted");
        require(reopened.disk_tier_restores == 0, "a fresh engine reported NVMe resumes");
        std::cout << "reopened: NVMe live=" << reopened.disk_tier_used_bytes << " bytes\n";

        std::uint32_t resumed = 0;
        for (Victim* victim : {&alpha, &bravo}) {
            const auto before = engine.runtime_stats().computed_prefill_tokens;
            const ninfer::GenerationResult restored =
                engine.generate(engine.prepare(long_prompt(victim->tag, kResidentLines)),
                                request_options(true));
            const auto after = engine.runtime_stats().computed_prefill_tokens;
            require(restored.generated_token_ids.size() == kOutput, "restored output count");
            require(restored.prefix_reuse_path != ninfer::PrefixReusePath::FullReset,
                    std::string("the spilled ") + victim->tag +
                        " prefix was not reused after the restart");
            require(after == before,
                    std::string("resuming the spilled ") + victim->tag +
                        " prefix still computed prefill tokens");
            require(restored.reused_prompt_tokens == victim->tokens,
                    std::string("the restored ") + victim->tag +
                        " frontier is not the prompt frontier the record closed");
            if (restored.generated_token_ids != victim->continuation) {
                throw std::runtime_error(
                    std::string("the disk-restored ") + victim->tag +
                    " continuation diverged from the cold one:\n  cold=" +
                    tokens_of(victim->continuation) +
                    "\n  disk=" + tokens_of(restored.generated_token_ids));
            }
            // The cold reference prefilled under contention with its sibling lane, so this ratio is
            // the conservative direction: it understates what a lone disk resume saves.
            require(restored.timings.first_token_seconds < victim->cold_ttft,
                    std::string("resuming the spilled ") + victim->tag +
                        " prefix from NVMe was not faster than cold prefill");
            std::cout << "restored " << victim->tag << ": " << restored.reused_prompt_tokens
                      << " prompt tokens from NVMe, " << (after - before)
                      << " prefill tokens computed, ttft " << victim->cold_ttft << " s cold -> "
                      << restored.timings.first_token_seconds << " s restored\n";
            ++resumed;
        }
        require(engine.memory_summary().disk_tier_restores == resumed,
                "the resumed prefixes did not account for one NVMe restore each");
        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
    std::filesystem::remove_all(root);
    return 0;
}
