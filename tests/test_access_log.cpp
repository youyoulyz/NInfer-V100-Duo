#include "serve/access_log.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
    std::vector<std::string> lines;
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) { lines.push_back(line); }
    return lines;
}

HttpAccessFacts facts_for(const std::string& path, bool streamed) {
    HttpAccessFacts facts;
    facts.method                = "POST";
    facts.path                  = path;
    facts.target                = path;
    facts.status                = 200;
    facts.remote_addr           = "192.168.1.20";
    facts.remote_port           = 51234;
    facts.user_agent            = "claude-cli/2.1.0";
    facts.request_content_type  = "application/json";
    facts.response_content_type = streamed ? "text/event-stream" : "application/json";
    facts.duration_seconds      = 1.25;
    facts.streamed              = streamed;
    return facts;
}

} // namespace

int main() {
    int failures = 0;

    // The per-request scope: a request that never reached routing drains empty and untimed, and
    // draining is what stops one request's frames from leaking into the next one.
    const AccessScopeDrain unscoped = drain_access_scope();
    failures += check(unscoped.streamed_bytes.empty() && unscoped.duration_seconds == 0.0,
                      "an unopened access scope did not drain empty");
    begin_access_scope();
    append_access_stream_bytes("data: {\"a\":1}\n\n");
    append_access_stream_bytes("data: [DONE]\n\n");
    const AccessScopeDrain scoped = drain_access_scope();
    failures += check(scoped.streamed_bytes == "data: {\"a\":1}\n\ndata: [DONE]\n\n" &&
                          scoped.duration_seconds >= 0.0,
                      "the access scope did not hand back the teed frames");
    failures += check(drain_access_scope().streamed_bytes.empty(),
                      "draining the access scope twice returned the same frames");

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("ninfer-access-log-test-" + std::to_string(static_cast<long long>(::getpid())));
    std::error_code error;
    std::filesystem::remove_all(root, error);

    // An empty directory disables the log without touching the filesystem.
    {
        AccessLog disabled("", 1 << 20, 1 << 20, "serve-test");
        failures += check(!disabled.enabled(), "an empty access log directory stayed enabled");
        disabled.write_access(facts_for("/v1/models", false), "{}", "{}");
    }

    // The directory is created, and one line carries the exchange: the client's body parsed, the
    // server's body parsed, and the HTTP facts of the request the logger saw.
    {
        const std::filesystem::path nested = root / "nested" / "access";
        AccessLog log(nested.string(), 1ULL << 20, 1ULL << 20, "serve-test");
        failures += check(log.enabled() && std::filesystem::is_directory(nested),
                          "the access log did not create its directory");
        const std::string request_body =
            R"({"model":"qwen3.8-27b-nvfp4","stream":false,"messages":[{"role":"user","content":"hi"}]})";
        const std::string response_body =
            R"({"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"hello"}}]})";
        log.write_access(facts_for("/v1/chat/completions", false), request_body, response_body);

        const std::vector<std::string> lines = read_lines(nested / "access.jsonl");
        failures += check(lines.size() == 1, "the access log did not append exactly one line");
        if (lines.size() == 1) {
            const Json record = Json::parse(lines[0]);
            failures += check(record.at("artifact_type") == kAccessLogArtifactType &&
                                  record.at("schema_version") == kAccessLogSchemaVersion &&
                                  record.at("event") == "access",
                              "access record preamble mismatch");
            failures += check(record.at("server_instance_id") == "serve-test" &&
                                  record.at("access_id") == 1,
                              "access record identity mismatch");
            failures += check(record.at("http").at("status") == 200 &&
                                  record.at("http").at("remote_addr") == "192.168.1.20" &&
                                  record.at("http").at("remote_port") == 51234 &&
                                  record.at("http").at("user_agent") == "claude-cli/2.1.0" &&
                                  record.at("http").at("streamed") == false &&
                                  record.at("http").at("duration_seconds") == 1.25,
                              "access record HTTP facts mismatch");
            // A JSON body is stored as the client's own object, not as an escaped string.
            failures += check(record.at("request_body").at("model") == "qwen3.8-27b-nvfp4" &&
                                  record.at("request_body").at("messages").at(0).at("content") ==
                                      "hi" &&
                                  record.at("response_body").at("choices").at(0).at("message").at(
                                      "content") == "hello",
                              "access record did not store the bodies as JSON");
            failures += check(record.at("request_body_truncated") == false &&
                                  record.at("response_body_truncated") == false,
                              "a small body was reported as truncated");
        }
    }

    // A streaming response is SSE text, which must survive as one string rather than a parse
    // attempt, and a body that is neither JSON nor SSE stays a string too.
    {
        const std::filesystem::path directory = root / "streamed";
        AccessLog log(directory.string(), 1ULL << 20, 1ULL << 20, "serve-test");
        const std::string sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        log.write_access(facts_for("/v1/chat/completions", true), "not json at all", sse);
        const std::vector<std::string> lines = read_lines(directory / "access.jsonl");
        failures += check(lines.size() == 1, "the streamed access record is missing");
        if (lines.size() == 1) {
            const Json record = Json::parse(lines[0]);
            failures += check(record.at("http").at("streamed") == true &&
                                  record.at("response_body") == sse,
                              "a streamed response was not stored as raw SSE text");
            failures += check(record.at("request_body") == "not json at all",
                              "a non-JSON request body was not stored as a string");
        }
    }

    // An oversized body is stored head-only, with the true length and a flag a reader can trust.
    {
        const std::filesystem::path directory = root / "capped";
        AccessLog log(directory.string(), 1ULL << 20, 64, "serve-test");
        const std::string big(500, 'x');
        log.write_access(facts_for("/v1/messages", false), "{\"a\":\"" + big + "\"}", "{}");
        const std::vector<std::string> lines = read_lines(directory / "access.jsonl");
        if (lines.size() == 1) {
            const Json record = Json::parse(lines[0]);
            failures += check(record.at("request_body_truncated") == true &&
                                  record.at("request_body").at("full_bytes") == 508 &&
                                  record.at("request_body").at("truncated_head").get<std::string>()
                                          .size() == 64,
                              "an oversized body did not keep its head and true length");
            failures += check(record.at("response_body_truncated") == false,
                              "the response body inherited the request's truncation");
        } else {
            failures += check(false, "the capped access record is missing");
        }
    }

    // Rotation keeps one bounded active file and moves the rest aside under numbered names.
    {
        const std::filesystem::path directory = root / "rotated";
        AccessLog log(directory.string(), 400, 1ULL << 20, "serve-test");
        for (int i = 0; i < 6; ++i) {
            log.write_access(facts_for("/v1/models", false), R"({"n":1})", R"({"ok":true})");
        }
        failures += check(std::filesystem::exists(directory / "access-1.jsonl"),
                          "the access log did not rotate at its size bound");
        failures += check(read_lines(directory / "access.jsonl").size() >= 1,
                          "the access log stopped writing after a rotation");
        std::size_t rotated = 0;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.path().filename() != "access.jsonl") { ++rotated; }
        }
        failures += check(rotated >= 1, "rotation left no numbered file behind");
    }

    std::filesystem::remove_all(root, error);

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
