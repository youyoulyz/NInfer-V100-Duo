#include "serve/access_log.h"
#include "serve/console_log.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

std::uint64_t unix_time_ms() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

// Request-in-flight scratch. httplib serves one request at a time per worker thread and runs the
// streaming provider and the post-response logger on that same thread.
thread_local std::chrono::steady_clock::time_point t_scope_start;
thread_local bool t_scope_active = false;
thread_local std::string t_stream_bytes;

// A body is stored as JSON when it parses, so a dataset builder reads the client's own objects
// instead of re-parsing a string. Anything else (SSE frames, a non-JSON error page) is stored as a
// string, and an oversized body as its head plus its true length.
Json embed_body(std::string_view bytes, std::size_t max_bytes, bool parse_json, bool& truncated) {
    if (bytes.empty()) { return nullptr; }
    if (bytes.size() > max_bytes) {
        truncated = true;
        return Json{{"truncated_head", std::string(bytes.substr(0, max_bytes))},
                    {"full_bytes", bytes.size()}};
    }
    if (parse_json) {
        Json parsed = Json::parse(bytes, nullptr, false);
        if (!parsed.is_discarded()) { return parsed; }
    }
    return std::string(bytes);
}

Json http_json(const HttpAccessFacts& facts) {
    return Json{{"method", facts.method},
                {"path", facts.path},
                {"target", facts.target},
                {"status", facts.status},
                {"remote_addr", facts.remote_addr},
                {"remote_port", facts.remote_port},
                {"user_agent", facts.user_agent},
                {"request_content_type", facts.request_content_type},
                {"response_content_type", facts.response_content_type},
                {"request_bytes", facts.request_bytes},
                {"response_bytes", facts.response_bytes},
                {"duration_seconds", facts.duration_seconds},
                {"streamed", facts.streamed}};
}

} // namespace

void begin_access_scope() {
    t_scope_start  = std::chrono::steady_clock::now();
    t_scope_active = true;
    t_stream_bytes.clear();
}

void append_access_stream_bytes(std::string_view bytes) { t_stream_bytes.append(bytes); }

AccessScopeDrain drain_access_scope() {
    AccessScopeDrain drain;
    if (!t_scope_active) {
        // The response never reached routing (a payload-limit rejection, for instance), so there is
        // nothing to time. Draining the buffer here is what keeps stale bytes from one request out
        // of the next one on this thread.
        t_stream_bytes.clear();
        return drain;
    }
    t_scope_active = false;
    const auto elapsed = std::chrono::steady_clock::now() - t_scope_start;
    drain.duration_seconds = std::chrono::duration<double>(elapsed).count();
    drain.streamed_bytes   = std::move(t_stream_bytes);
    t_stream_bytes.clear();
    return drain;
}

AccessLog::AccessLog(std::string directory, std::uint64_t max_file_bytes,
                     std::size_t max_body_bytes, std::string server_instance_id)
    : directory_(std::move(directory)),
      server_instance_id_(std::move(server_instance_id)),
      max_file_bytes_(max_file_bytes),
      max_body_bytes_(max_body_bytes) {
    if (directory_.empty()) { return; }
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    if (error) {
        throw std::runtime_error("failed to create the access log directory: " + directory_ + ": " +
                                 error.message());
    }
    active_path_ = (std::filesystem::path(directory_) / "access.jsonl").string();
    output_.open(active_path_, std::ios::out | std::ios::app);
    if (!output_) {
        throw std::runtime_error("failed to open the access log for append: " + active_path_);
    }
    const std::uintmax_t size = std::filesystem::file_size(active_path_, error);
    file_bytes_               = error ? 0 : static_cast<std::uint64_t>(size);
}

void AccessLog::write_access(const HttpAccessFacts& facts, std::string_view request_body,
                             std::string_view response_body) {
    if (!enabled()) { return; }
    bool request_truncated  = false;
    bool response_truncated = false;
    Json request            = embed_body(request_body, max_body_bytes_, true, request_truncated);
    Json response =
        embed_body(response_body, max_body_bytes_, !facts.streamed, response_truncated);

    std::lock_guard<std::mutex> lock(mutex_);
    if (!output_) { return; }
    const Json record{{"artifact_type", kAccessLogArtifactType},
                      {"schema_version", kAccessLogSchemaVersion},
                      {"event", "access"},
                      {"timestamp_unix_ms", unix_time_ms()},
                      {"server_instance_id", server_instance_id_},
                      {"access_id", ++access_id_},
                      {"http", http_json(facts)},
                      {"request_body", std::move(request)},
                      {"request_body_truncated", request_truncated},
                      {"response_body", std::move(response)},
                      {"response_body_truncated", response_truncated}};
    // A body is captured as the bytes that crossed the wire, so it can end inside a UTF-8
    // character: a client that gives up mid-frame leaves the teed SSE text cut after the first
    // bytes of a multi-byte sequence. JSON demands valid UTF-8, and the default dump throws
    // `type_error.316` on anything else -- which killed the server on every mid-stream disconnect
    // until this replaced the offending bytes. The log is a record of someone else's bytes, so
    // U+FFFD is the honest rendering of a sequence the wire never finished.
    append_locked(record.dump(-1, ' ', false, Json::error_handler_t::replace));
}

void AccessLog::append_locked(std::string_view record) {
    if (max_file_bytes_ != 0 && file_bytes_ != 0 &&
        file_bytes_ + record.size() + 1 > max_file_bytes_) {
        rotate_locked();
        if (!output_) { return; }
    }
    output_ << record << '\n';
    output_.flush();
    file_bytes_ += record.size() + 1;
    if (!output_) {
        write_console_log(ConsoleLogLevel::Error, "access log write failed for " + active_path_);
        output_.close();
    }
}

void AccessLog::rotate_locked() {
    output_.close();
    std::error_code error;
    const std::filesystem::path active(active_path_);
    for (std::uint64_t index = rotation_index_ + 1;; ++index) {
        const std::filesystem::path target =
            active.parent_path() / ("access-" + std::to_string(index) + ".jsonl");
        if (std::filesystem::exists(target, error)) { continue; }
        error.clear();
        std::filesystem::rename(active, target, error);
        if (error) {
            write_console_log(ConsoleLogLevel::Error,
                              "access log rotation failed for " + active_path_ + ": " +
                                  error.message());
        } else {
            rotation_index_ = index;
        }
        break;
    }
    output_.open(active_path_, std::ios::out | std::ios::app);
    file_bytes_ = 0;
    if (!output_) {
        write_console_log(ConsoleLogLevel::Error,
                          "access log reopen failed for " + active_path_);
    }
}

} // namespace ninfer::serve
