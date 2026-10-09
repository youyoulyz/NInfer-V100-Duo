// End-to-end gate for the block-level KV pool on the real artifact: N long sessions retained across
// the three tiers, every one of them resumable without re-prefilling.
//
// The scenario is the product question, not a synthetic one. A conversation is admitted, prefills its
// long prompt once, answers, and its prefix reaches the NVMe tier as that turn completes; the next
// conversation then displaces it from the paged pool. The engine is destroyed (taking the device pages, the host arena and every
// resident prefix metadata with it) and reopened against the same directory. Each conversation's
// prompt is then sent again, and the only thing that can answer it is the content-addressed record.
//
// The default shape is 8 sessions of 150K tokens against a pool that holds one of them at a time,
// because that is what the hardware admits: only `max_concurrency` requests are ever in flight, and
// the pool keeps every OTHER conversation's prefix on disk. The oracle is the first engine's own cold
// continuation -- a restored prefix must decode greedy-identically to the run that produced it --
// plus the prefill counter, which must not move at all on a resume.
//
// NINFER_TEST_TP2=1 NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=/path/to/model.ninfer \
//   ctest -R disk_pool_real
//
// Shrink it while iterating with NINFER_DISK_POOL_TOKENS / NINFER_DISK_POOL_LANES /
// NINFER_DISK_POOL_CAPACITY / NINFER_DISK_POOL_HOST_BYTES. The scratch directory must not be a
// tmpfs; its default is under /var/tmp and NINFER_DISK_TIER_DIR overrides it.
//
// NINFER_DISK_POOL_SPEC=none drops MTP; NINFER_DISK_POOL_CONCURRENCY (default 2) sets the lane count. At 1 the deployment has a
// single lane, so every session displaces its predecessor from that one lane instead of parking it
// to a free one: the record then has to reach the NVMe tier through the lane-replacement spill, and
// a resume has to land back on the lane the previous session just used. That shape is what a
// serial server runs, and it is the one the host tier alone cannot serve -- it only parks lanes
// other than the one being admitted.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kOutput = 8;

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') { return fallback; }
    return static_cast<std::uint32_t>(std::strtoul(raw, nullptr, 10));
}

std::size_t env_bytes(const char* name, std::size_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') { return fallback; }
    return static_cast<std::size_t>(std::strtoull(raw, nullptr, 10));
}

// The pool capacity the run was configured with, so the sizing checks compare against the same
// number the engine planned against.
std::uint32_t options_capacity() { return env_u32("NINFER_DISK_POOL_CAPACITY", 153600); }


void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, const std::filesystem::path& disk_path) {
    ninfer::EngineOptions options;
    options.artifact_path      = artifact;
    options.tp                 = 2;
    options.devices            = {0, 1};
    options.max_context        = env_u32("NINFER_DISK_POOL_CAPACITY", 153600);
    options.kv_capacity        = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
    options.prefill_chunk      = 1024;
    options.kv_cache           = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph     = true;
    options.max_concurrency    = env_u32("NINFER_DISK_POOL_CONCURRENCY", 2);
    options.host_context_bytes = env_bytes("NINFER_DISK_POOL_HOST_BYTES", 12ULL << 30);
    options.disk_kv_path       = disk_path;
    options.disk_kv_bytes      = env_bytes("NINFER_DISK_POOL_DISK_BYTES", 96ULL << 30);
    // The tier's publish hook sits on both resolution paths, so the gate runs the ordinary one too:
    // NINFER_DISK_POOL_SPEC=none drops the speculative backend that the other shapes exercise.
    const char* const spec = std::getenv("NINFER_DISK_POOL_SPEC");
    if (spec == nullptr || std::string(spec) != "none") {
        options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
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

// The tag is what makes two sessions distinct, and it sits at the very start, so their content
// differs from the first token -- each one is a cold prefill rather than a hit on its sibling --
// while the body is identical, which keeps every session the same number of tokens.
ninfer::PromptInput long_prompt(const std::string& tag, std::uint32_t lines) {
    // Fixed-width line numbers: a line's token count then does not depend on the index, so the
    // prompt's token count is exactly affine in the line count and the sizing fit below is exact.
    const auto label = [](std::uint32_t index) {
        std::string digits = std::to_string(index);
        return std::string(6U - std::min<std::size_t>(6, digits.size()), '0') + digits;
    };
    std::string text = "Context " + tag + ". Review these Python functions.\n\n";
    for (std::uint32_t i = 0; i < lines; ++i) {
        const std::string index = label(i);
        text += "def add_" + index + "(value):\n    return value + " + index + "\n\n";
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

// One conversation: the prompt the first engine prefilled cold and the second one has to resume.
struct Session {
    std::uint32_t id     = 0;
    std::uint32_t lines  = 0;
    std::uint32_t tokens = 0;
    double cold_ttft     = 0.0;
    double resume_ttft   = 0.0;
    std::vector<ninfer::TokenId> continuation;
};

std::vector<ninfer::TokenId> cold_run(ninfer::Engine& engine, Session& session,
                                      std::string& failure) {
    ninfer::PreparedPrompt prompt = engine.prepare(
        long_prompt("session-" + std::to_string(session.id), session.lines));
    session.tokens = prompt.summary().prompt_tokens;
    const ninfer::GenerationResult result = engine.generate(std::move(prompt), request_options(false));
    if (result.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        failure = "session " + std::to_string(session.id) + " was not a cold prefill";
    } else if (result.generated_token_ids.size() != kOutput) {
        failure = "session " + std::to_string(session.id) + " produced the wrong output count";
    }
    session.cold_ttft = result.timings.first_token_seconds;
    return result.generated_token_ids;
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
        std::cout << "skip: the disk pool gate needs two CUDA devices\n";
        return 77;
    }
    const std::uint32_t lanes = env_u32("NINFER_DISK_POOL_LANES", 8);
    const std::uint32_t target_tokens = env_u32("NINFER_DISK_POOL_TOKENS", 150000);
    if (lanes < 3) {
        std::cerr << "FAIL: the gate needs at least three sessions to force a displacement\n";
        return 1;
    }

    const char* configured = std::getenv("NINFER_DISK_TIER_DIR");
    const std::string pattern = configured != nullptr && *configured != '\0'
                                    ? std::string(configured)
                                    : std::string("/var/tmp/ninfer_disk_pool_XXXXXX");
    std::vector<char> directory(pattern.begin(), pattern.end());
    directory.push_back('\0');
    if (::mkdtemp(directory.data()) == nullptr) {
        std::cerr << "FAIL: cannot create the NVMe tier scratch directory from " << pattern << '\n';
        return 1;
    }
    const std::filesystem::path root(directory.data());

    try {
        std::vector<Session> sessions(lanes);
        for (std::uint32_t lane = 0; lane < lanes; ++lane) { sessions[lane].id = lane; }
        std::size_t persisted = 0;
        {
            ninfer::Engine engine(engine_options(artifact, root));
            const ninfer::MemorySummary startup = engine.memory_summary();
            std::cout << "pool: capacity=" << startup.kv_capacity << " tokens ("
                      << startup.kv_capacity / 1024 << "K), page groups "
                      << startup.kv_capacity_page_groups << ", host tier "
                      << startup.host_tier_capacity_bytes / (1ULL << 20) << " MiB, NVMe tier "
                      << startup.disk_tier_capacity_bytes / (1ULL << 30) << " GiB\n";
            require(startup.disk_tier_capacity_bytes == env_bytes("NINFER_DISK_POOL_DISK_BYTES", 96ULL << 30),
                    "the NVMe tier did not take the configured budget");
            require(startup.disk_tier_used_bytes == 0,
                    "a freshly opened NVMe tier reported live records");
            require(startup.kv_capacity >= target_tokens,
                    "the resident pool cannot hold one " + std::to_string(target_tokens) +
                        "-token session (capacity " + std::to_string(startup.kv_capacity) + ")");

            // Size the prompts to the target. The count is affine in the line count -- a fixed
            // template prologue plus a fixed-width body line -- and prepare() refuses a prompt above
            // max_context, so both probes are placed from a conservative density estimate that keeps
            // them well inside the pool. The two-point fit is then exact and lands on the target.
            require(target_tokens + 64U <= options_capacity(),
                    "the resident pool leaves no room for a " + std::to_string(target_tokens) +
                        "-token session");
            const auto count_for = [&](std::uint32_t probe_lines) {
                return engine.prepare(long_prompt("session-0", probe_lines)).summary().prompt_tokens;
            };
            const std::uint32_t seed_lines  = 32;
            const std::uint32_t seed_tokens = count_for(seed_lines);
            require(seed_tokens > seed_lines, "the seed prompt did not tokenize as expected");
            // The seed's average tokens per line overstates the marginal cost (it carries the fixed
            // prologue too), so sizing a probe from it can never push the probe past the pool.
            const double rough = static_cast<double>(seed_tokens) / static_cast<double>(seed_lines);
            const auto probe_lines = [&](double share) {
                return std::max(seed_lines, static_cast<std::uint32_t>(share *
                                                                      options_capacity() / rough));
            };
            const std::uint32_t small_lines  = probe_lines(0.2);
            const std::uint32_t big_lines    = std::max(small_lines + 1U, probe_lines(0.5));
            const std::uint32_t small_tokens = count_for(small_lines);
            const std::uint32_t big_tokens   = count_for(big_lines);
            require(big_tokens > small_tokens, "the prompt token count does not grow with the lines");
            const double per_line = static_cast<double>(big_tokens - small_tokens) /
                                    static_cast<double>(big_lines - small_lines);
            const double fixed = static_cast<double>(small_tokens) - per_line * small_lines;
            const std::uint32_t lines = static_cast<std::uint32_t>(
                std::ceil((static_cast<double>(target_tokens) + 16.0 - fixed) / per_line));
            for (Session& session : sessions) { session.lines = lines; }
            const std::uint32_t measured = count_for(lines);
            const std::uint32_t largest =
                engine.prepare(long_prompt("session-" + std::to_string(lanes - 1), lines))
                    .summary()
                    .prompt_tokens;
            std::cout << "prompts: " << lines << " lines -> " << measured << " tokens each (fit: "
                      << fixed << " + " << per_line << " x lines)\n";
            require(measured >= target_tokens,
                    "the prompts are shorter than the target: " + std::to_string(measured));
            require(largest <= options_capacity(),
                    "the longest prompt does not fit the resident pool: " +
                        std::to_string(largest));

            // Every session is admitted in turn and displaces the one before it, so each prefix
            // reaches the NVMe tier on its way out.
            for (Session& session : sessions) {
                std::string failure;
                session.continuation = cold_run(engine, session, failure);
                require(failure.empty(), failure);
                require(session.tokens >= target_tokens && session.tokens <= options_capacity(),
                        "session " + std::to_string(session.id) + " prefilled " +
                            std::to_string(session.tokens) + " tokens, outside [" +
                            std::to_string(target_tokens) + ", " +
                            std::to_string(options_capacity()) + "]");
                const ninfer::MemorySummary after = engine.memory_summary();
                std::cout << "session " << session.id << ": cold prefill " << session.tokens
                          << " tokens, ttft " << session.cold_ttft << " s, host parked "
                          << after.host_tier_parked_bytes / (1ULL << 20) << " MiB, NVMe "
                          << after.disk_tier_used_bytes / (1ULL << 20) << " MiB\n";
                std::cout.flush();
            }

            // Nothing displaces the last session, and no flush request stands in for one: the tier
            // has to hold each conversation as its turn completes, or the session a process happens
            // to be holding when it exits would be the one conversation that is not on it. The
            // resumed pass below is what proves it -- the last session's record can only come from
            // its own completion.
            const ninfer::MemorySummary summary = engine.memory_summary();
            persisted                            = summary.disk_tier_used_bytes;
            std::cout << "resident at exit: host parked "
                      << summary.host_tier_parked_bytes / (1ULL << 20) << " MiB, NVMe live "
                      << persisted / (1ULL << 20) << " MiB, restores " << summary.disk_tier_restores
                      << '\n';
            require(persisted > 0, "no session reached the NVMe tier");
        }
        std::cout << "engine 1 released: its device pages, host arena and resident prefixes are gone\n";

        ninfer::Engine engine(engine_options(artifact, root));
        const ninfer::MemorySummary reopened = engine.memory_summary();
        require(reopened.host_tier_parked_bytes == 0, "a fresh engine reported parked host images");
        require(reopened.disk_tier_used_bytes == persisted,
                "the reopened NVMe tier did not reload the records the first engine persisted");
        require(reopened.disk_tier_restores == 0, "a fresh engine reported NVMe resumes");

        std::uint32_t resumed = 0;
        for (Session& session : sessions) {
            const auto before = engine.runtime_stats().computed_prefill_tokens;
            const ninfer::GenerationResult result = engine.generate(
                engine.prepare(long_prompt("session-" + std::to_string(session.id), session.lines)),
                request_options(true));
            const auto after = engine.runtime_stats().computed_prefill_tokens;
            require(result.generated_token_ids.size() == kOutput,
                    "session " + std::to_string(session.id) + " restored output count");
            require(result.prefix_reuse_path != ninfer::PrefixReusePath::FullReset,
                    "session " + std::to_string(session.id) +
                        " was re-prefilled instead of resumed from the pool");
            require(after == before,
                    "resuming session " + std::to_string(session.id) +
                        " still computed prefill tokens");
            require(result.reused_prompt_tokens == session.tokens,
                    "session " + std::to_string(session.id) + " resumed at " +
                        std::to_string(result.reused_prompt_tokens) + " tokens, expected " +
                        std::to_string(session.tokens));
            if (result.generated_token_ids != session.continuation) {
                throw std::runtime_error(
                    "the resumed session " + std::to_string(session.id) +
                    " diverged from its cold continuation:\n  cold=" +
                    tokens_of(session.continuation) + "\n  pool=" +
                    tokens_of(result.generated_token_ids));
            }
            session.resume_ttft = result.timings.first_token_seconds;
            std::cout << "resumed session " << session.id << ": " << result.reused_prompt_tokens
                      << " prompt tokens from the pool, " << (after - before)
                      << " prefill tokens computed, ttft " << session.cold_ttft << " s cold -> "
                      << session.resume_ttft << " s resumed\n";
            std::cout.flush();
            ++resumed;
        }
        require(engine.memory_summary().disk_tier_restores == resumed,
                "the resumed sessions did not account for one pool restore each");
        std::cout << "ok: " << resumed << " x " << sessions.front().tokens
                  << "-token sessions resumed with zero prefill tokens\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
    std::filesystem::remove_all(root);
    return 0;
}
