// End-to-end gate for the host lane tier on the real artifact, at tp2 with MTP.
//
// A retained prefix is only evicted when the page pool is committed across several lanes, so the
// scenario submits two unrelated long prefixes CONCURRENTLY (they must land on different lanes),
// then asks for a third, longer prompt that cannot be admitted without reclaiming one of them. That
// eviction is where the tier acts: the lane is mirrored into pinned host RAM -- both ranks' Main
// Text KV, the MTP KV, the GDN linear-attention slots, the hidden rows, and the prefix metadata --
// and its device KV pages are released. The fourth request asks for the second prefix again and the
// engine restores the parked image instead of paying a re-prefill.
//
// The oracle is the plan's own gate: the restored continuation must be greedy-identical to the same
// prefix resumed from the LIVE retained state (the control engine below), which is byte-identical
// by construction only if the park/restore round trip is exact.
//
// NINFER_TEST_TP2=1 NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=/path/to/model.ninfer \
//   ctest -R host_tier_real

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kCapacity = 4096;
constexpr std::uint32_t kOutput   = 8;
// Prompts sized against a 64-page pool (4096 tokens / 64): two of them must fit together and the
// third must not be admissible while they are both resident.
constexpr int kResidentLines = 119;
constexpr int kEvictingLines = 170;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, bool tier) {
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
    options.host_context_bytes        = tier ? 256ULL * 1024ULL * 1024ULL : 0ULL;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

ninfer::RequestOptions request_options(bool reuse, std::uint32_t count = kOutput) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens    = count;
    options.execution.allow_prefix_reuse          = reuse;
    options.execution.sampling.temperature        = 0.0F;
    options.execution.sampling.presence_penalty   = 0.0F;
    options.execution.sampling.frequency_penalty  = 0.0F;
    options.stop.include_model_defaults           = false;
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

std::string tokens_of(const std::vector<ninfer::TokenId>& tokens) {
    std::string out;
    for (const ninfer::TokenId token : tokens) { out += std::to_string(token) + " "; }
    return out;
}

struct Run {
    ninfer::GenerationResult alpha;
    ninfer::GenerationResult bravo;
    ninfer::GenerationResult evicting;
    ninfer::GenerationResult resumed;
    ninfer::GenerationResult resumed_gamma;
    std::size_t parked_after_eviction   = 0;
    std::size_t parked_after_resume     = 0;
    std::size_t parked_after_round_trip = 0;
    bool evicted                        = false;
};

// `evict_between` is the parked/never-parked switch: with it, a third unrelated long prompt forces
// the second lane out of the pool; without it, the second prefix stays resident and the final
// request resumes it live.
//
// The parked run walks TWO round trips, not one. The first restores bravo and, to fit it, parks the
// evicting prefix in the same slot the tier just freed; the second restores that prefix again and
// re-parks bravo. Requiring the final parked byte count to equal the first park's is what proves a
// restore returns its host image instead of leaking it: a leak would strand a lane image in the
// fixed per-lane linear arena and force the second park to fall back to a discard (which the
// zero-re-prefill assertions would then catch).
Run run_sequence(ninfer::Engine& engine, std::uint32_t tokens_alpha, bool evict_between) {
    Run run;
    auto alpha = engine.submit(engine.prepare(long_prompt("alpha", kResidentLines)),
                               request_options(true));
    auto bravo = engine.submit(engine.prepare(long_prompt("bravo", kResidentLines)),
                               request_options(true));
    run.alpha = alpha.wait();
    run.bravo = bravo.wait();
    require(run.alpha.generated_token_ids.size() == kOutput, "alpha output count");
    require(run.bravo.generated_token_ids.size() == kOutput, "bravo output count");
    require(run.alpha.prefix_reuse_path == ninfer::PrefixReusePath::FullReset &&
                run.bravo.prefix_reuse_path == ninfer::PrefixReusePath::FullReset,
            "the concurrent pair did not both cold-prefill");

    const std::size_t parked_before = engine.memory_summary().host_tier_parked_bytes;
    if (evict_between) {
        run.evicting = engine.generate(engine.prepare(long_prompt("gamma", kEvictingLines)),
                                       request_options(false));
        require(run.evicting.generated_token_ids.size() == kOutput, "evicting output count");
        run.parked_after_eviction = engine.memory_summary().host_tier_parked_bytes;
        run.evicted               = run.parked_after_eviction > parked_before;
    }

    const auto prefill_before = engine.runtime_stats().computed_prefill_tokens;
    run.resumed = engine.generate(engine.prepare(long_prompt("bravo", kResidentLines)),
                                  request_options(true));
    const auto prefill_after = engine.runtime_stats().computed_prefill_tokens;
    require(run.resumed.generated_token_ids.size() == kOutput, "resumed output count");
    require(run.resumed.prefix_reuse_path != ninfer::PrefixReusePath::FullReset,
            "the resumed prefix was not reused");
    if (!evict_between) {
        require(prefill_after - prefill_before < tokens_alpha,
                "control resume did not reuse the live prefix");
        return run;
    }

    require(prefill_after == prefill_before,
            "resuming the parked prefix still computed prefill tokens");
    run.parked_after_resume = engine.memory_summary().host_tier_parked_bytes;

    run.resumed_gamma = engine.generate(engine.prepare(long_prompt("gamma", kEvictingLines)),
                                        request_options(true));
    require(run.resumed_gamma.generated_token_ids.size() == kOutput, "resumed gamma output count");
    require(run.resumed_gamma.prefix_reuse_path != ninfer::PrefixReusePath::FullReset,
            "the evicting prefix was not reused on its second request");
    if (run.resumed_gamma.generated_token_ids != run.evicting.generated_token_ids) {
        throw std::runtime_error(
            "restored evicting continuation diverged from its cold run:\n  cold=" +
            tokens_of(run.evicting.generated_token_ids) +
            "\n  parked=" + tokens_of(run.resumed_gamma.generated_token_ids));
    }
    require(engine.runtime_stats().computed_prefill_tokens == prefill_after,
            "the second restored prefix still computed prefill tokens");
    run.parked_after_round_trip = engine.memory_summary().host_tier_parked_bytes;
    return run;
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
        std::cout << "skip: host tier gate needs two CUDA devices\n";
        return 77;
    }

    try {
        std::uint32_t tokens_alpha = 0;
        Run control;
        {
            ninfer::Engine engine(engine_options(artifact, /*tier=*/false));
            require(engine.memory_summary().host_tier_capacity_bytes == 0,
                    "a tier-less engine reported host tier capacity");
            tokens_alpha = static_cast<std::uint32_t>(
                engine.prepare(long_prompt("alpha", kResidentLines)).debug_token_ids().size());
            std::cout << "resident prompt tokens=" << tokens_alpha << '\n';
            require(tokens_alpha > 1800 && tokens_alpha < 2050,
                    "resident prompt size is outside the 64-page pool scenario window");
            control = run_sequence(engine, tokens_alpha, /*evict_between=*/false);
        }

        const std::size_t rss_before = process_rss_bytes();
        ninfer::Engine engine(engine_options(artifact, /*tier=*/true));
        const ninfer::MemorySummary startup = engine.memory_summary();
        require(startup.host_tier_capacity_bytes > 0, "tier did not allocate host capacity");
        require(startup.host_tier_parked_bytes == 0, "tier parked something with no eviction yet");

        const Run tier = run_sequence(engine, tokens_alpha, /*evict_between=*/true);
        require(tier.evicted, "the evicting request did not park a retained lane into host RAM");
        // A lane image is its KV on both ranks PLUS both ranks' GDN linear-attention state. The KV
        // half is only ~64 MiB at this prefix length, so a >200 MiB image is direct evidence that
        // the linear state travelled with it.
        require(tier.parked_after_eviction > 200ULL * 1024ULL * 1024ULL,
                "the parked image is too small to contain the GDN linear-attention state");
        std::cout << "after eviction: parked=" << tier.parked_after_eviction << " bytes, rss="
                  << (process_rss_bytes() - rss_before) << " bytes over baseline\n";
        require(process_rss_bytes() > rss_before, "host RSS did not grow across the park");
        require(tier.resumed.prefix_reuse_path == control.resumed.prefix_reuse_path,
                "restored resume path differs from the live resume path");
        require(tier.resumed.reused_prompt_tokens == control.resumed.reused_prompt_tokens,
                "restored frontier differs from the live frontier");
        if (tier.resumed.generated_token_ids != control.resumed.generated_token_ids) {
            throw std::runtime_error(
                "restored continuation diverged from the live continuation:\n  live=" +
                tokens_of(control.resumed.generated_token_ids) +
                "\n  parked=" + tokens_of(tier.resumed.generated_token_ids));
        }
        require(tier.parked_after_resume > 0,
                "the first restore consumed every parked image; a lane went missing");
        require(tier.parked_after_round_trip == tier.parked_after_eviction,
                "a park/restore cycle did not return the lane image to the host arena");
        std::cout << "round trips: first=" << tier.parked_after_resume
                  << " final=" << tier.parked_after_round_trip << " bytes\n";
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
