#pragma once

// Lossless access capture: one JSONL record per HTTP request, holding the client's body and the
// server's body verbatim. `--access-log-dir` enables it and datasets are built from its files
// (`tools/v100/ninfer-api/build_dataset.py`).
//
// This is deliberately NOT the measurement log (`--request-log-jsonl`). That one publishes a
// compact, versioned schema for benchmark consumers and stores no payload text; it is the record
// to quote when a number has to be defended. This one stores payload text and little else, so its
// records are large, its consumers are dataset builders, and nothing here should be read as a
// measurement. Both share one `server_instance_id` per process so a run is recognizable across
// them.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>

namespace ninfer::serve {

inline constexpr int kAccessLogSchemaVersion        = 1;
inline constexpr const char* kAccessLogArtifactType = "ninfer_serve_access_log";

// Identity and accounting of one request/response exchange, as the HTTP layer saw it.
struct HttpAccessFacts {
    std::string method;
    std::string path;
    std::string target; // request target, query string included
    int status             = 0;
    std::string remote_addr;
    int remote_port        = 0;
    std::string user_agent;
    std::string request_content_type;
    std::string response_content_type;
    std::size_t request_bytes  = 0;
    std::size_t response_bytes = 0;
    double duration_seconds    = 0.0;
    // A streaming response never materializes in `Response::body`; its bytes are teed into the
    // per-thread buffer below and the record stores them as one SSE string.
    bool streamed = false;
};

// One httplib worker thread handles a request, its streaming provider, and the post-response
// logger, so the request-in-flight scratch below needs no locking. `begin_access_scope()` opens it
// before routing; the SSE writers append to it; `drain_access_scope()` closes it and returns both
// the teed bytes and the wall time the request took.
struct AccessScopeDrain {
    std::string streamed_bytes;
    double duration_seconds = 0.0;
};

void begin_access_scope();
void append_access_stream_bytes(std::string_view bytes);
[[nodiscard]] AccessScopeDrain drain_access_scope();

class AccessLog {
public:
    // An empty `directory` disables the log. `max_file_bytes` rotates `access.jsonl` into
    // `access-<n>.jsonl`; `max_body_bytes` caps how much of one body a record stores, so an
    // oversized body is stored head-only with its length and a truncation flag.
    AccessLog(std::string directory, std::uint64_t max_file_bytes, std::size_t max_body_bytes,
              std::string server_instance_id);

    AccessLog(const AccessLog&)            = delete;
    AccessLog& operator=(const AccessLog&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return output_.is_open(); }

    // `request_body` and `response_body` are the exact bytes of the exchange, as received and as
    // written. A streamed response passes the bytes teed by the SSE writers.
    void write_access(const HttpAccessFacts& facts, std::string_view request_body,
                      std::string_view response_body);

private:
    // The three below require `mutex_` to be held.
    void append_locked(std::string_view record);
    void rotate_locked();

    std::string directory_;
    std::string active_path_;
    std::string server_instance_id_;
    std::uint64_t max_file_bytes_ = 0;
    std::size_t max_body_bytes_   = 0;
    std::uint64_t access_id_      = 0;
    std::uint64_t file_bytes_     = 0;
    std::uint64_t rotation_index_ = 0;
    std::ofstream output_;
    std::mutex mutex_;
};

} // namespace ninfer::serve
