// End-to-end gate for lane-free shared prefixes on the real artifact, at tp2 with MTP.
//
// A prefix is only worth keeping once its lane is gone, and that is exactly the case this gate
// builds: with two lanes, three different conversations are submitted in order. The third one lands
// on the first one's lane, so the first prefix is displaced -- and with shared state images enabled
// it is captured into the block index (its page groups held, its GDN state in pinned host RAM)
// before its lane is rebuilt. Asking for that first conversation again must then resume it with no
// lane holding it: a two-token prefill on a checkpoint restore, and a greedy-identical completion.
//
// The control run is the same sequence with the images disabled, where the displaced prefix is
// simply lost and the third request re-prefills the whole prompt. The pair is what separates "the
// payload answered" from "some lane still happened to hold the prefix".
//
// NINFER_TEST_TP2=1 NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=/path/to/model.ninfer \
//   ctest -R shared_payload_real

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kCapacity = 8192;
constexpr std::uint32_t kOutput   = 8;
// Sizes chosen so that all three prefixes fit the pool together: the point of the gate is the lane
// the third request displaces, not pool pressure.
constexpr int kFirstLines  = 119;
constexpr int kOtherLines  = 170;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, std::uint32_t images,
                                     std::size_t kv_mib) {
    ninfer::EngineOptions options;
    options.artifact_path            = artifact;
    options.tp                       = 2;
    options.devices                  = {0, 1};
    options.max_context              = kCapacity;
    options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk            = 256;
    options.kv_cache                 = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph           = true;
    options.max_concurrency          = 2;
    options.shared_state_images      = images;
    options.shared_kv_bytes          = kv_mib << 20;
    options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

ninfer::RequestOptions request_options() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens   = kOutput;
    options.execution.allow_prefix_reuse        = true;
    options.execution.sampling.temperature      = 0.0F;
    options.execution.sampling.presence_penalty = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults         = false;
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

struct Outcome {
    ninfer::GenerationResult first;
    ninfer::GenerationResult displaced_again;
    std::uint64_t prefill_tokens = 0;
    std::uint32_t payloads       = 0;
    std::uint32_t held_pages     = 0;
};

Outcome run(ninfer::Engine& engine) {
    Outcome out;
    // Conversation one and two fill both lanes; three then displaces one of them.
    auto first = engine.submit(engine.prepare(long_prompt("alpha", kFirstLines)), request_options());
    out.first  = first.wait();
    auto second = engine.submit(engine.prepare(long_prompt("bravo", kFirstLines)), request_options());
    (void)second.wait();
    auto third = engine.submit(engine.prepare(long_prompt("gamma", kOtherLines)), request_options());
    const ninfer::GenerationResult third_result = third.wait();
    require(third_result.generated_token_ids.size() == kOutput, "gamma output count");
    out.payloads     = engine.runtime_stats().shared_prefix.payloads;
    out.held_pages   = engine.runtime_stats().shared_prefix.held_pages;

    // The displaced conversation comes back. Nothing holds its chain any more, so what answers can
    // only be the payload the index captured when its lane was rebuilt.
    const std::uint64_t before = engine.runtime_stats().computed_prefill_tokens;
    auto again = engine.submit(engine.prepare(long_prompt("alpha", kFirstLines)), request_options());
    out.displaced_again = again.wait();
    out.prefill_tokens  = engine.runtime_stats().computed_prefill_tokens - before;
    return out;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    if (artifact == nullptr) {
        std::cerr << "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS is not set\n";
        return 77;
    }
    if (std::getenv("NINFER_TEST_TP2") == nullptr) {
        std::cerr << "NINFER_TEST_TP2 is not set\n";
        return 77;
    }
    try {
        std::uint32_t payloads = 0;
        {
            ninfer::Engine engine(engine_options(artifact, 2, 0));
            const Outcome shared = run(engine);
            payloads             = shared.payloads;

            require(shared.first.generated_token_ids.size() == kOutput, "alpha output count");
            require(shared.payloads >= 1,
                    "the displaced prefix was not captured into the shared index");
            require(shared.displaced_again.prefix_reuse_path != ninfer::PrefixReusePath::FullReset,
                    "the displaced prefix was re-prefilled instead of claimed");
            require(shared.displaced_again.reused_prompt_tokens > 0 &&
                        shared.displaced_again.reused_prompt_tokens < kCapacity,
                    "the displaced prefix reused no tokens");
            require(shared.prefill_tokens <= 2,
                    "the displaced prefix cost more than a two-token prefill");
            require(shared.displaced_again.generated_token_ids == shared.first.generated_token_ids,
                    "the claimed prefix did not decode the cold continuation");
            std::cout << "shared: reuse="
                      << static_cast<int>(shared.displaced_again.prefix_reuse_path)
                      << " payloads=" << shared.payloads << " prefill_tokens="
                      << shared.prefill_tokens
                      << " reused=" << shared.displaced_again.reused_prompt_tokens << '\n';
        }
        {
            // The same scenario with a per-node KV budget: the captured prefix parks its KV in host
            // RAM and releases the device pages, so the payload holds no page group at all. The
            // budget has to cover both ranks' shards of the prefix (about 35 MB per 1943 tokens).
            ninfer::Engine engine(engine_options(artifact, 2, 256));
            const Outcome host = run(engine);
            require(host.payloads >= 1, "the host KV payload was not captured");
            require(engine.runtime_stats().shared_prefix.held_pages == 0,
                    "a host KV payload still holds device pages");
            require(host.displaced_again.prefix_reuse_path != ninfer::PrefixReusePath::FullReset,
                    "the host KV payload did not answer");
            require(host.prefill_tokens <= 2, "the host KV claim cost more than a two-token prefill");
            require(host.displaced_again.generated_token_ids == host.first.generated_token_ids,
                    "the host KV claim did not decode the cold continuation");
            std::cout << "host-kv: reuse="
                      << static_cast<int>(host.displaced_again.prefix_reuse_path)
                      << " payloads=" << host.payloads << " held_pages=" << host.held_pages
                      << " prefill_tokens=" << host.prefill_tokens
                      << " reused=" << host.displaced_again.reused_prompt_tokens << '\n';
        }
        {
            ninfer::Engine engine(engine_options(artifact, 0, 0));
            const Outcome cold = run(engine);
            require(cold.payloads == 0, "a disabled index kept a payload");
            require(cold.displaced_again.prefix_reuse_path == ninfer::PrefixReusePath::FullReset,
                    "a disabled index still resumed the displaced prefix");
            std::cout << "control: reuse="
                      << static_cast<int>(cold.displaced_again.prefix_reuse_path)
                      << " prefill_tokens=" << cold.prefill_tokens << '\n';
        }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
