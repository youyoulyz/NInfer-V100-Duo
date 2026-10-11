#include "serve/request_log.h"
#include "product/speculative_options.h"
#include "serve/console_log.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

std::uint64_t unix_time_ms() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::string new_server_instance_id() {
    const auto now    = std::chrono::system_clock::now().time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    return "serve-" + std::to_string(static_cast<long long>(::getpid())) + '-' +
           std::to_string(micros);
}

std::filesystem::path normalized_absolute_path(const std::string& value) {
    std::error_code error;
    std::filesystem::path path = std::filesystem::weakly_canonical(value, error);
    if (!error) { return path; }
    error.clear();
    path = std::filesystem::absolute(value, error);
    return error ? std::filesystem::path(value).lexically_normal() : path.lexically_normal();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { out << '-'; }
        out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

const char* finish_reason_name(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output_limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context_capacity";
    case ninfer::FinishReason::StopToken:
        return "stop_token";
    case ninfer::FinishReason::StopString:
        return "stop_string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

std::string tool_choice_name(const ToolChoice& choice) {
    switch (choice.mode) {
    case ToolChoiceMode::Auto:
        return "auto";
    case ToolChoiceMode::None:
        return "none";
    case ToolChoiceMode::Required:
        return "required";
    case ToolChoiceMode::Named:
        return choice.name.empty() ? "named" : choice.name;
    }
    return "unknown";
}

const char* kv_cache_name(ninfer::KvCacheStorage storage) {
    return storage == ninfer::KvCacheStorage::BFloat16 ? "bf16" : "int8-group64";
}

const char* kv_capacity_mode_name(ninfer::KvCapacityMode mode) {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

const char* proposal_head_name(ninfer::ProposalHead proposal) {
    return proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full";
}

const char* prefix_reuse_path_name(ninfer::PrefixReusePath path) {
    switch (path) {
    case ninfer::PrefixReusePath::FullReset:
        return "full_reset";
    case ninfer::PrefixReusePath::AppendAtFrontier:
        return "append_frontier";
    case ninfer::PrefixReusePath::RestoreTurnCheckpoint:
        return "restore_turn_checkpoint";
    case ninfer::PrefixReusePath::RestoreResponseCheckpoint:
        return "restore_response_checkpoint";
    // Planner-selected source path (materialization catalog); not produced by the Engine yet.
    case ninfer::PrefixReusePath::Root:
        return "root";
    case ninfer::PrefixReusePath::PrivateEndpoint:
        return "private_endpoint";
    case ninfer::PrefixReusePath::PrivateTurnClosure:
        return "private_turn_closure";
    case ninfer::PrefixReusePath::PrivateResponseReplay:
        return "private_response_replay";
    case ninfer::PrefixReusePath::PrivateLongAnchor:
        return "private_long_anchor";
    case ninfer::PrefixReusePath::SharedStablePrefix:
        return "shared_stable_prefix";
    }
    return "unknown";
}

Json event_base(const std::string& server_instance_id, std::uint64_t timestamp, const char* event) {
    return Json{{"artifact_type", kRequestLogArtifactType},
                {"schema_version", kRequestLogSchemaVersion},
                {"event", event},
                {"timestamp_unix_ms", timestamp},
                {"server_instance_id", server_instance_id}};
}

Json sampler_json(const ninfer::ResolvedSamplingParameters& sampling) {
    return Json{{"temperature", sampling.temperature},
                {"top_p", sampling.top_p},
                {"top_k", sampling.top_k},
                {"min_p", sampling.min_p},
                {"presence_penalty", sampling.presence_penalty},
                {"frequency_penalty", sampling.frequency_penalty},
                {"seed", sampling.seed}};
}

Json preset_json(const ninfer::SamplingPreset& preset) {
    return Json{{"temperature", preset.temperature},
                {"top_p", preset.top_p},
                {"top_k", preset.top_k},
                {"min_p", preset.min_p},
                {"presence_penalty", preset.presence_penalty},
                {"frequency_penalty", preset.frequency_penalty}};
}

Json overrides_json(const ninfer::SamplingOverrides& overrides) {
    Json result{{"temperature", nullptr},
                {"top_p", nullptr},
                {"top_k", nullptr},
                {"min_p", nullptr},
                {"presence_penalty", nullptr},
                {"frequency_penalty", nullptr},
                {"seed", nullptr}};
    if (overrides.temperature) { result["temperature"] = *overrides.temperature; }
    if (overrides.top_p) { result["top_p"] = *overrides.top_p; }
    if (overrides.top_k) { result["top_k"] = *overrides.top_k; }
    if (overrides.min_p) { result["min_p"] = *overrides.min_p; }
    if (overrides.presence_penalty) { result["presence_penalty"] = *overrides.presence_penalty; }
    if (overrides.frequency_penalty) { result["frequency_penalty"] = *overrides.frequency_penalty; }
    if (overrides.seed) { result["seed"] = *overrides.seed; }
    return result;
}

Json request_json(const RequestLogContext& context) {
    return Json{{"request_id", context.id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"enable_thinking", context.enable_thinking},
                {"preserve_thinking", context.preserve_thinking},
                {"preserve_thinking_semantic_change", context.preserve_thinking_semantic_change},
                {"sampling", sampler_json(context.sampling)}};
}

Json preparation_json(const RequestLogContext& context) {
    const PromptPreparationStats& stats = context.preparation;
    return Json{{"total", stats.seconds},
                {"acquisition", context.acquisition_seconds},
                {"media_preprocess", stats.media_preprocess_seconds},
                {"media_preprocess_work", stats.media_preprocess_work_seconds},
                {"tokenize", stats.tokenize_seconds},
                {"media_items", stats.media_items},
                {"media_bytes", stats.media_bytes},
                {"raw_patches", stats.raw_patches},
                {"vision_tokens", stats.vision_tokens},
                {"patch_bytes", stats.patch_bytes},
                {"cache_hits", stats.media_cache_hits},
                {"cache_misses", stats.media_cache_misses},
                {"singleflight_waits", stats.media_singleflight_waits},
                {"built_patch_bytes", stats.built_patch_bytes},
                {"reused_patch_bytes", stats.reused_patch_bytes}};
}

Json rejected_request_json(const RequestRejectionLogContext& context) {
    return Json{{"request_id", context.id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history}};
}

Json error_json(const ApiError& error) {
    Json code  = error.code.empty() ? Json(nullptr) : Json(error.code);
    Json param = error.param.empty() ? Json(nullptr) : Json(error.param);
    return Json{{"status", error.status},
                {"type", error.type},
                {"code", std::move(code)},
                {"param", std::move(param)},
                {"message", error.message}};
}

Json arena_json(const ninfer::ArenaMemorySummary& arena) {
    return Json{{"capacity_bytes", arena.capacity_bytes},
                {"used_bytes", arena.used_bytes},
                {"peak_used_bytes", arena.peak_used_bytes}};
}

Json speculative_json(const GenerationMetrics& metrics) {
    return Json{{"backend", product::speculative_backend_name(metrics.speculative_backend)},
                {"draft_window", metrics.speculative_draft_window},
                {"rounds", metrics.speculative_rounds},
                {"drafted_tokens", metrics.speculative_draft_tokens},
                {"accepted_tokens", metrics.speculative_accepted_tokens},
                {"fallback_steps", metrics.speculative_fallback_steps},
                {"accepted_per_position", metrics.speculative_accepted_per_position}};
}

// Tokens/second with fixed precision, or "n/a" when the interval is degenerate.
std::string rate(double tokens, double seconds) {
    std::ostringstream out;
    if (seconds > 0.0 && tokens > 0.0) {
        out << std::fixed << std::setprecision(1) << (tokens / seconds) << "tok/s";
    } else {
        out << "n/a";
    }
    return out.str();
}

std::string seconds_str(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << seconds << 's';
    return out.str();
}

// Compact resolved-sampler summary. temperature <= 0 is the exact-argmax path.
std::string sampler_str(const ninfer::ResolvedSamplingParameters& sampling) {
    if (sampling.temperature <= 0.0f) { return "greedy"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
        << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k;
    if (sampling.min_p > 0.0f) { out << " min_p=" << sampling.min_p; }
    if (sampling.presence_penalty != 0.0f) { out << " pres=" << sampling.presence_penalty; }
    if (sampling.frequency_penalty != 0.0f) { out << " freq=" << sampling.frequency_penalty; }
    out << " seed=" << sampling.seed;
    return out.str();
}

std::string speculative_str(const GenerationMetrics& metrics) {
    if (metrics.speculative_backend == SpeculativeBackend::None) { return "off"; }
    std::ostringstream out;
    out << product::speculative_backend_name(metrics.speculative_backend) << ' ' << std::fixed
        << std::setprecision(2);
    if (metrics.speculative_rounds > 0) {
        const double per_round = 1.0 + static_cast<double>(metrics.speculative_accepted_tokens) /
                                           static_cast<double>(metrics.speculative_rounds);
        out << per_round << "tok/round";
    } else {
        out << "n/a";
    }
    if (metrics.speculative_draft_tokens > 0) {
        const double accept_pct = 100.0 * static_cast<double>(metrics.speculative_accepted_tokens) /
                                  static_cast<double>(metrics.speculative_draft_tokens);
        out << " (" << std::setprecision(1) << accept_pct << "%)";
    }
    return out.str();
}

} // namespace

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const PreparedRequest& prepared) {
    RequestLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = request.model;
    context.stream                             = request.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = request.max_tokens_set;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.enable_thinking                    = prepared.enable_thinking;
    context.preserve_thinking                  = prepared.preserve_thinking;
    context.preserve_thinking_semantic_change  = prepared.preserve_thinking_semantic_change;
    context.sampling                           = prepared.sampling;
    context.acquisition_seconds                = prepared.acquisition_seconds;
    context.preparation                        = prepared.preparation;
    return context;
}

RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              ApiError error) {
    RequestRejectionLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = request.model;
    context.stream                             = request.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = request.max_tokens_set;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.error                              = std::move(error);
    return context;
}

std::string format_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "[req " << context.id << "] " << context.protocol << ' '
        << (context.stream ? "stream" : "non-stream") << " msgs=" << context.message_count
        << " max_tokens=" << context.requested_output_tokens << ' '
        << (context.requested_output_tokens_client_set ? "(client)" : "(server default)")
        << " tools=" << context.tool_count
        << " tool_choice=" << tool_choice_name(context.tool_choice)
        << " tool_history=" << (context.has_tool_history ? "yes" : "no")
        << " thinking=" << (context.enable_thinking ? "on" : "off")
        << " preserve_thinking=" << (context.preserve_thinking ? "on" : "off")
        << " preserve_change=" << (context.preserve_thinking_semantic_change ? "yes" : "no")
        << " sampler=[" << sampler_str(context.sampling) << ']';
    if (context.media_item_count != 0) {
        out << " prepare=" << seconds_str(context.preparation.seconds)
            << " acquire=" << seconds_str(context.acquisition_seconds)
            << " media=" << seconds_str(context.preparation.media_preprocess_seconds) << '/'
            << seconds_str(context.preparation.media_preprocess_work_seconds)
            << " tokenize=" << seconds_str(context.preparation.tokenize_seconds)
            << " media_cache=" << context.preparation.media_cache_hits << '/'
            << context.preparation.media_cache_misses << '/'
            << context.preparation.media_singleflight_waits;
    }
    out << " \xE2\x86\x92 submitted";
    return out.str();
}

std::string format_request_rejected(const RequestRejectionLogContext& context) {
    std::ostringstream out;
    out << "[req " << context.id << "] rejected phase=prepare protocol=" << context.protocol << ' '
        << (context.stream ? "stream" : "non-stream") << " msgs=" << context.message_count
        << " media=" << context.media_item_count << " tools=" << context.tool_count
        << " status=" << context.error.status;
    if (!context.error.code.empty()) { out << " code=" << context.error.code; }
    out << " message=" << context.error.message;
    return out.str();
}

std::string format_request_done(const RequestLogContext& context,
                                const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const double ttft_ms             = metrics.ttft_seconds * 1000.0;
    // Prefill emits the first token; the remaining (gen - 1) come from decode.
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;
    const double computed_prefill_tokens = static_cast<double>(
        std::max(0, outcome.prompt_tokens - static_cast<int>(metrics.prefix_cache_hit_tokens)));

    std::ostringstream out;
    out << "[req " << context.id << "] done finish="
        << (outcome.tool_calls.empty() ? finish_reason_name(outcome.finish_reason) : "tool_calls");
    if (!outcome.tool_calls.empty()) { out << " tool_calls=" << outcome.tool_calls.size(); }
    out << " prompt=" << outcome.prompt_tokens << " gen=" << outcome.completion_tokens
        << " cache=" << metrics.prefix_cache_hit_tokens
        << " reuse=" << prefix_reuse_path_name(metrics.prefix_reuse_path) << " ttft=" << std::fixed
        << std::setprecision(0) << ttft_ms << "ms"
        << " prefill=" << rate(computed_prefill_tokens, metrics.prefill_seconds)
        << " decode=" << rate(decode_tokens, metrics.decode_seconds)
        << " wall=" << seconds_str(metrics.total_seconds)
        << " speculative=" << speculative_str(metrics);
    return out.str();
}

std::string format_request_error(const RequestLogContext& context, const std::string& message) {
    std::ostringstream out;
    out << "[req " << context.id << "] error " << message;
    return out.str();
}

std::string format_throughput(const ThroughputReport& report) {
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    std::ostringstream out;
    out << "throughput interval=" << std::fixed << std::setprecision(3) << report.interval_seconds
        << "s prefill=" << std::setprecision(1) << prefill_rate << "tok/s decode=" << decode_rate
        << "tok/s running=" << report.scheduler.running_requests
        << " prefilling=" << report.scheduler.prefilling_requests
        << " decode_ready=" << report.scheduler.decode_ready_requests
        << " waiting=" << report.scheduler.waiting_requests << " avg_decode_batch=";
    if (report.decode_rounds == 0) {
        out << "n/a";
    } else {
        out << std::setprecision(2)
            << static_cast<double>(report.decode_row_rounds) /
                   static_cast<double>(report.decode_rounds);
    }
    // Cross-lane prefix sharing: how the content-addressed index answered, and what it holds now.
    const ninfer::SharedPrefixStats& shared = report.scheduler.shared_prefix;
    out << " shared_nodes=" << shared.nodes << " shared_owners=" << shared.owners
        << " shared_ownerless=" << shared.ownerless << " shared_payloads=" << shared.payloads
        << " shared_hits=" << shared.hits << "/" << shared.lookups
        << " shared_pruned=" << shared.pruned;
    return out.str();
}

std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp, const ServeOptions& options,
    const ninfer::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const ninfer::LoadSummary& load, const ninfer::MemorySummary& memory,
    const ServerLogEnvironment& environment, std::optional<std::uint64_t> artifact_size_bytes) {
    Json record = event_base(server_instance_id, timestamp, "server_start");

    Json artifact_size = nullptr;
    if (artifact_size_bytes.has_value()) { artifact_size = *artifact_size_bytes; }

    record["server"]   = Json{{"host", options.host},
                              {"port", options.port},
                              {"public_model_id", public_model_id},
                              {"api_key_configured", !options.api_key.empty()},
                              {"cors_enabled", options.enable_cors},
                              {"max_request_bytes", options.max_request_bytes},
                              {"media_cache_bytes", options.media_cache_bytes},
                              {"media_live_bytes", options.media_live_bytes},
                              {"media_preprocess_threads", options.media_preprocess_threads},
                              {"request_log_jsonl", options.request_log_jsonl},
                              {"access_log_dir", options.access_log_dir},
                              {"default_output_tokens", options.default_max_tokens},
                              {"default_thinking", options.enable_thinking},
                              {"default_preserve_thinking", options.preserve_thinking}};
    record["artifact"] = Json{{"path", options.artifact_path},
                              {"size_bytes", std::move(artifact_size)},
                              {"target", load.target},
                              {"weights_id", load.weights_id},
                              {"bytes_read", load.artifact_bytes_read},
                              {"host_to_device_bytes", load.host_to_device_bytes},
                              {"peak_staging_bytes", load.peak_staging_bytes},
                              {"tensor_count", load.tensor_count},
                              {"resource_count", load.resource_count},
                              {"load_seconds", load.load_seconds},
                              {"upload_seconds", load.upload_seconds}};
    record["engine"]   = Json{
          {"device", options.device},
          // Tensor-parallel width and the resolved per-rank device list. At `tp` 2 every other
          // engine and memory field in this record is rank 0's; `devices` is what names the
          // second GPU, so a downstream reader can tell a dual-device run from a single-device
          // one without parsing argv. `devices` always has exactly `tp` entries.
          {"tp", options.tp},
          {"devices", options.devices},
          {"max_context", options.max_context},
          {"rope_mode", options.rope_mode == RopeMode::Yarn ? "yarn" : "native"},
          {"yarn_factor", options.rope_mode == RopeMode::Yarn ? options.yarn_factor : 0.0},
          {"yarn_origin", options.rope_mode == RopeMode::Yarn ? options.yarn_origin : 0U},
          {"effective_max_context", memory.effective_max_context},
          {"yarn_mscale", memory.yarn_mscale},
          {"kv_capacity_mode", kv_capacity_mode_name(memory.kv_capacity_mode)},
          {"kv_capacity", memory.kv_capacity},
          {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
          {"kv_capacity_max_page_groups", memory.kv_capacity_max_page_groups},
          {"max_concurrency", options.max_concurrency},
          {"max_pending_requests", options.max_pending_requests},
          {"pending_timeout_ms", options.pending_timeout_ms},
          {"prefill_chunk", options.prefill_chunk},
          {"log_stats_interval_ms", options.log_stats_interval_ms},
          {"kv_cache", kv_cache_name(options.kv_cache)},
          {"vision", options.enable_vision},
          {"vision_max_tokens", std::min(options.max_context, options.vision_max_tokens)},
          {"cuda_graph", options.use_cuda_graph},
          {"prefix_reuse", options.allow_prefix_reuse},
          {"speculative_backend", product::speculative_backend_name(options.speculative.backend)},
          {"speculative_draft_window", options.speculative.draft_tokens},
          {"proposal_head", proposal_head_name(options.speculative.proposal_head)}};
    record["sampling_defaults"] =
        Json{{"thinking", preset_json(sampling_defaults.thinking)},
             {"non_thinking", preset_json(sampling_defaults.non_thinking)},
             {"server_overrides", overrides_json(options.sampling_overrides)},
             {"omitted_seed", "random"},
             {"greedy", options.greedy}};
    record["memory"] =
        Json{{"weights", arena_json(memory.weights)},
             {"sequence", arena_json(memory.sequence)},
             {"workspace", arena_json(memory.workspace)},
             {"request_transient", arena_json(memory.request_transient)},
             {"minimum_runtime_reservation_bytes", memory.minimum_runtime_reservation_bytes},
             {"kv_capacity_increment_bytes", memory.kv_capacity_increment_bytes},
             {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
             {"available_after_weights_bytes", memory.available_after_weights_bytes},
             {"available_after_startup_bytes", memory.available_after_startup_bytes},
             {"kv_capacity_headroom_bytes", memory.kv_capacity_headroom_bytes},
             {"planned_slack_bytes", memory.planned_slack_bytes},
             {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
             // Both observed figures are per device and are checked against the same
             // per-device allowance; the peer field is 0 at tp 1. `cuda_graph_node_count`
             // is one captured decode graph's node count, roughly 2x at tp 2 because one
             // graph holds both devices' nodes.
             {"cuda_graph_observed_bytes", memory.cuda_graph_observed_bytes},
             {"cuda_graph_peer_observed_bytes", memory.cuda_graph_peer_observed_bytes},
             {"cuda_graph_node_count", memory.cuda_graph_node_count},
             {"kv_payload_bytes", memory.kv_payload_bytes}};
    record["environment"] =
        Json{{"device", environment.device},
             {"gpu_name", environment.gpu_name},
             {"gpu_uuid", environment.gpu_uuid},
             {"total_device_memory_bytes", environment.total_device_memory_bytes},
             {"compute_capability_major", environment.compute_capability_major},
             {"compute_capability_minor", environment.compute_capability_minor},
             {"cuda_compile_version", environment.cuda_compile_version},
             {"cuda_runtime_version", environment.cuda_runtime_version},
             {"cuda_driver_version", environment.cuda_driver_version}};
    record["argv"] = options.startup_argv;
    return record.dump();
}

std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context) {
    Json record                   = event_base(server_instance_id, timestamp, "request_start");
    record["request"]             = request_json(context);
    record["preparation_seconds"] = preparation_json(context);
    return record.dump();
}

std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp,
                                         const RequestRejectionLogContext& context) {
    Json record       = event_base(server_instance_id, timestamp, "request_rejected");
    record["phase"]   = "prepare";
    record["request"] = rejected_request_json(context);
    record["error"]   = error_json(context.error);
    return record.dump();
}

std::string format_request_done_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    Json record       = event_base(server_instance_id, timestamp, "request_done");
    record["request"] = request_json(context);
    record["result"] =
        Json{{"finish_reason", finish_reason_name(outcome.finish_reason)},
             {"prompt_tokens", outcome.prompt_tokens},
             {"completion_tokens", outcome.completion_tokens},
             {"computed_prefill_tokens",
              std::max(0, outcome.prompt_tokens -
                              static_cast<int>(outcome.metrics.prefix_cache_hit_tokens))},
             {"prefix_cache_hit_tokens", outcome.metrics.prefix_cache_hit_tokens},
             {"prefix_reuse_path", prefix_reuse_path_name(outcome.metrics.prefix_reuse_path)},
             {"tool_call_count", outcome.tool_calls.size()}};
    record["timings_seconds"] = Json{
        {"prepare", outcome.metrics.prepare_seconds}, {"ttft", outcome.metrics.ttft_seconds},
        {"vision", outcome.metrics.vision_seconds},   {"prefill", outcome.metrics.prefill_seconds},
        {"decode", outcome.metrics.decode_seconds},   {"total", outcome.metrics.total_seconds}};
    record["speculative"] = speculative_json(outcome.metrics);
    return record.dump();
}

std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context,
                                      const std::string& message) {
    Json record       = event_base(server_instance_id, timestamp, "request_error");
    record["request"] = request_json(context);
    record["error"]   = Json{{"message", message}};
    return record.dump();
}

std::string format_throughput_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                   const ThroughputReport& report) {
    Json record = event_base(server_instance_id, timestamp, "throughput");
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    Json average_batch = nullptr;
    if (report.decode_rounds != 0) {
        average_batch = static_cast<double>(report.decode_row_rounds) /
                        static_cast<double>(report.decode_rounds);
    }
    record["interval_seconds"] = report.interval_seconds;
    record["tokens"]           = Json{{"computed_prefill", report.computed_prefill_tokens},
                                      {"committed_decode", report.committed_decode_tokens}};
    record["throughput_tokens_per_second"] =
        Json{{"prefill", prefill_rate}, {"decode", decode_rate}};
    record["scheduler"]    = Json{{"running", report.scheduler.running_requests},
                                  {"prefilling", report.scheduler.prefilling_requests},
                                  {"decode_ready", report.scheduler.decode_ready_requests},
                                  {"waiting", report.scheduler.waiting_requests}};
    record["decode_batch"] = Json{{"rounds", report.decode_rounds},
                                  {"row_rounds", report.decode_row_rounds},
                                  {"average_size", std::move(average_batch)}};
    // Cross-lane prefix sharing: the content-addressed index's usage and current census.
    record["shared_prefix"] = Json{{"nodes", report.scheduler.shared_prefix.nodes},
                                   {"owners", report.scheduler.shared_prefix.owners},
                                   {"ownerless", report.scheduler.shared_prefix.ownerless},
                                   {"payloads", report.scheduler.shared_prefix.payloads},
                                   {"hits", report.scheduler.shared_prefix.hits},
                                   {"lookups", report.scheduler.shared_prefix.lookups},
                                   {"pruned", report.scheduler.shared_prefix.pruned}};
    return record.dump();
}

ServerLogEnvironment query_server_log_environment(int device) {
    ServerLogEnvironment environment;
    environment.device               = device;
    environment.cuda_compile_version = cuda_version_string(CUDART_VERSION);

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        environment.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        environment.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        environment.gpu_name                  = properties.name;
        environment.gpu_uuid                  = cuda_uuid_string(properties.uuid);
        environment.total_device_memory_bytes = properties.totalGlobalMem;
        environment.compute_capability_major  = properties.major;
        environment.compute_capability_minor  = properties.minor;
    }
    return environment;
}

std::string make_server_instance_id() { return new_server_instance_id(); }

JsonlRequestLog::JsonlRequestLog(const std::string& path,
                                 const std::string& protected_artifact_path,
                                 std::string server_instance_id)
    : path_(path) {
    if (path_.empty()) { return; }
    if (!protected_artifact_path.empty() &&
        normalized_absolute_path(path_) == normalized_absolute_path(protected_artifact_path)) {
        throw std::invalid_argument("request JSONL log must not overwrite the model artifact");
    }
    server_instance_id_ = server_instance_id.empty() ? new_server_instance_id()
                                                     : std::move(server_instance_id);
    output_.open(path_, std::ios::out | std::ios::app);
    if (!output_) {
        throw std::runtime_error("failed to open request JSONL log for append: " + path_);
    }
}

void JsonlRequestLog::write_server_start(const ServeOptions& options,
                                         const ninfer::ModelSamplingDefaults& sampling_defaults,
                                         const std::string& public_model_id,
                                         const ninfer::LoadSummary& load,
                                         const ninfer::MemorySummary& memory) {
    if (!enabled()) { return; }
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(options.artifact_path, error);
    const std::optional<std::uint64_t> artifact_size =
        error ? std::nullopt : std::optional<std::uint64_t>(size);
    append(format_server_start_json(server_instance_id_, unix_time_ms(), options, sampling_defaults,
                                    public_model_id, load, memory,
                                    query_server_log_environment(options.device), artifact_size));
}

void JsonlRequestLog::write_request_start(const RequestLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_start_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_rejected(const RequestRejectionLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_rejected_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_done(const RequestLogContext& context,
                                         const GenerationOutcome& outcome) {
    if (!enabled()) { return; }
    append(format_request_done_json(server_instance_id_, unix_time_ms(), context, outcome));
}

void JsonlRequestLog::write_request_error(const RequestLogContext& context,
                                          const std::string& message) {
    if (!enabled()) { return; }
    append(format_request_error_json(server_instance_id_, unix_time_ms(), context, message));
}

void JsonlRequestLog::write_throughput(const ThroughputReport& report) {
    if (!enabled()) { return; }
    append(format_throughput_json(server_instance_id_, unix_time_ms(), report));
}

void JsonlRequestLog::append(std::string record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) { return; }
    output_ << record << '\n';
    output_.flush();
    if (!output_) {
        failed_ = true;
        write_console_log(ConsoleLogLevel::Error, "request JSONL logging failed for " + path_);
    }
}

std::string format_prometheus_metrics(const ninfer::RuntimeStats& stats,
                                      std::string_view model_id) {
    const ninfer::KvRuntimeStats& kv = stats.kv;
    const auto metric = [](std::ostringstream& out, std::string_view name, std::string_view help,
                           std::string_view type, std::uint64_t value) {
        out << "# HELP " << name << ' ' << help << '\n' << "# TYPE " << name << ' ' << type << '\n'
            << name << ' ' << value << '\n';
    };
    std::ostringstream out;
    const ninfer::SharedPrefixStats& shared = stats.shared_prefix;
    const std::uint64_t held_pages = shared.held_pages;

    metric(out, "ninfer_prefill_tokens_total", "Prompt tokens evaluated by prefill.", "counter",
           stats.computed_prefill_tokens);
    metric(out, "ninfer_decode_tokens_total", "Tokens committed by decode rounds.", "counter",
           stats.committed_decode_tokens);
    metric(out, "ninfer_decode_rounds_total", "Decode batch executions.", "counter",
           stats.decode_rounds);
    metric(out, "ninfer_decode_rows_total", "Sum of the decode batch sizes.", "counter",
           stats.decode_row_rounds);

    out << "# HELP ninfer_requests Requests by scheduler state.\n# TYPE ninfer_requests gauge\n"
        << "ninfer_requests{state=\"running\"} " << stats.running_requests << '\n'
        << "ninfer_requests{state=\"prefilling\"} " << stats.prefilling_requests << '\n'
        << "ninfer_requests{state=\"decode_ready\"} " << stats.decode_ready_requests << '\n'
        << "ninfer_requests{state=\"waiting\"} " << stats.waiting_requests << '\n';

    metric(out, "ninfer_speculative_rounds_total", "Speculative rounds the server has run.",
           "counter", stats.speculative_rounds);
    metric(out, "ninfer_speculative_drafted_tokens_total", "Draft tokens proposed.", "counter",
           stats.speculative_drafted_tokens);
    metric(out, "ninfer_speculative_accepted_tokens_total", "Draft tokens accepted.", "counter",
           stats.speculative_accepted_tokens);
    metric(out, "ninfer_speculative_fallback_steps_total", "Rounds with no accepted draft.",
           "counter", stats.speculative_fallback_steps);
    metric(out, "ninfer_speculative_live_rounds", "Speculative rounds of the running requests.",
           "gauge", stats.speculative_live.rounds);
    metric(out, "ninfer_speculative_live_drafted_tokens",
           "Drafted tokens of the running requests.", "gauge",
           stats.speculative_live.drafted_tokens);
    metric(out, "ninfer_speculative_live_accepted_tokens",
           "Accepted tokens of the running requests.", "gauge",
           stats.speculative_live.accepted_tokens);
    if (!stats.speculative_accepted_per_position.empty()) {
        out << "# HELP ninfer_speculative_accepted_per_position_total Rounds in which a draft at "
               "this position was accepted, over the server's life.\n"
               "# TYPE ninfer_speculative_accepted_per_position_total counter\n";
        for (std::size_t position = 0; position < stats.speculative_accepted_per_position.size();
             ++position) {
            out << "ninfer_speculative_accepted_per_position_total{position=\"" << (position + 1)
                << "\"} " << stats.speculative_accepted_per_position[position] << '\n';
        }
    }
    if (!stats.speculative_live.accepted_per_position.empty()) {
        out << "# HELP ninfer_speculative_live_accepted_per_position Accepted tokens per draft "
               "position of the running requests.\n"
               "# TYPE ninfer_speculative_live_accepted_per_position gauge\n";
        for (std::size_t position = 0; position < stats.speculative_live.accepted_per_position.size();
             ++position) {
            out << "ninfer_speculative_live_accepted_per_position{position=\"" << (position + 1)
                << "\"} " << stats.speculative_live.accepted_per_position[position] << '\n';
        }
    }

    out << "# HELP ninfer_kv_page_groups Paged KV page groups (one 64-token block) by pool and "
           "state.\n# TYPE ninfer_kv_page_groups gauge\n"
        << "ninfer_kv_page_groups{pool=\"main\",state=\"capacity\"} "
        << kv.main_capacity_page_groups << '\n'
        << "ninfer_kv_page_groups{pool=\"main\",state=\"entitled\"} "
        << kv.main_entitled_page_groups << '\n'
        << "ninfer_kv_page_groups{pool=\"main\",state=\"free\"} "
        << kv.main_free_page_groups << '\n'
        << "ninfer_kv_page_groups{pool=\"main\",state=\"shared_held\"} " << held_pages << '\n'
        << "ninfer_kv_page_groups{pool=\"backend\",state=\"capacity\"} "
        << kv.backend_capacity_page_groups << '\n'
        << "ninfer_kv_page_groups{pool=\"backend\",state=\"entitled\"} "
        << kv.backend_entitled_page_groups << '\n'
        << "ninfer_kv_page_groups{pool=\"backend\",state=\"free\"} "
        << kv.backend_free_page_groups << '\n';

    metric(out, "ninfer_kv_pool_payload_bytes", "Persistent KV payload this device holds, whole pool.",
           "gauge", kv.kv_payload_bytes);
    metric(out, "ninfer_gdn_state_bytes", "GDN linear-attention state bytes on one device.", "gauge",
           kv.gdn_state_bytes);
    out << "# HELP ninfer_state_slots GDN linear-attention slots.\n# TYPE ninfer_state_slots "
           "gauge\n"
        << "ninfer_state_slots{state=\"capacity\"} " << kv.state_capacity_slots << '\n'
        << "ninfer_state_slots{state=\"occupied\"} " << kv.state_occupied_slots << '\n';
    out << "# HELP ninfer_context_tokens Context ceilings in tokens.\n# TYPE ninfer_context_tokens "
           "gauge\n"
        << "ninfer_context_tokens{state=\"max\"} " << kv.max_context << '\n'
        << "ninfer_context_tokens{state=\"effective\"} " << kv.effective_max_context << '\n';

    out << "# HELP ninfer_host_kv_bytes Pinned host KV tier.\n# TYPE ninfer_host_kv_bytes gauge\n"
        << "ninfer_host_kv_bytes{state=\"capacity\"} " << kv.host_kv_capacity_bytes << '\n'
        << "ninfer_host_kv_bytes{state=\"occupied\"} " << kv.host_kv_occupied_bytes << '\n';
    out << "# HELP ninfer_host_tier_bytes Pinned host lane tier.\n# TYPE ninfer_host_tier_bytes "
           "gauge\n"
        << "ninfer_host_tier_bytes{state=\"capacity\"} " << kv.host_tier_capacity_bytes << '\n'
        << "ninfer_host_tier_bytes{state=\"parked\"} " << kv.host_tier_parked_bytes << '\n';
    out << "# HELP ninfer_disk_tier_bytes NVMe lane tier.\n# TYPE ninfer_disk_tier_bytes gauge\n"
        << "ninfer_disk_tier_bytes{state=\"capacity\"} " << kv.disk_tier_capacity_bytes << '\n'
        << "ninfer_disk_tier_bytes{state=\"used\"} " << kv.disk_tier_used_bytes << '\n';
    metric(out, "ninfer_disk_tier_restores_total", "Prefixes resumed from the NVMe tier.", "counter",
           kv.disk_tier_restores);

    metric(out, "ninfer_kv_cache_ttl_seconds", "Configured idle timeout for every KV cache.", "gauge",
           kv.kv_cache_ttl_seconds);
    metric(out, "ninfer_kv_cache_oldest_idle_seconds",
           "Idle time of the oldest retained prefix.", "gauge", kv.oldest_retained_idle_seconds);
    metric(out, "ninfer_kv_cache_maintenance_total", "Idle-cache maintenance passes run.", "counter",
           kv.kv_cache_maintenance_runs);
    out << "# HELP ninfer_kv_cache_expired_total KV cache entries released by the idle timeout.\n"
           "# TYPE ninfer_kv_cache_expired_total counter\n"
        << "ninfer_kv_cache_expired_total{kind=\"retained_lane\"} " << kv.expired_retained_lanes
        << '\n'
        << "ninfer_kv_cache_expired_total{kind=\"parked_image\"} " << kv.expired_parked_images
        << '\n'
        << "ninfer_kv_cache_expired_total{kind=\"disk_record\"} " << kv.expired_disk_records << '\n';

    metric(out, "ninfer_shared_nodes", "Content-addressed blocks the shared index holds.", "gauge",
           shared.nodes);
    metric(out, "ninfer_shared_owners", "Live lane promises on those blocks.", "gauge",
           shared.owners);
    metric(out, "ninfer_shared_ownerless", "Indexed blocks no lane serves any more.", "gauge",
           shared.ownerless);
    metric(out, "ninfer_shared_payloads", "Blocks holding a lane-free continuation image.", "gauge",
           shared.payloads);
    metric(out, "ninfer_shared_hits_total", "Plans the shared index answered.", "counter",
           shared.hits);
    metric(out, "ninfer_shared_lookups_total", "Plans that asked the shared index.", "counter",
           shared.lookups);
    metric(out, "ninfer_shared_pruned_total", "Blocks reclaimed once no lane served them.",
           "counter", shared.pruned);

    out << "# HELP ninfer_info Build identity of the served model.\n# TYPE ninfer_info gauge\n"
        << "ninfer_info{model=\"" << model_id << "\"} 1\n";
    return out.str();
}

} // namespace ninfer::serve
