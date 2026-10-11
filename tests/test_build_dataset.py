"""The access capture is raw wire traffic, so the dataset builder is where protocol knowledge
lives. These cases drive it with the exact framing the server writes: JSON bodies for
non-streaming responses and the three SSE dialects for streaming ones."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools/v100/ninfer-api"))

import build_dataset  # noqa: E402


def sse(*frames: tuple[str | None, Any]) -> str:
    text = ""
    for name, payload in frames:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        text += (f"event: {name}\n" if name else "") + f"data: {body}\n\n"
    return text


def access(
    path: str,
    request_body: Any,
    response_body: Any,
    *,
    streamed: bool = False,
    status: int = 200,
    access_id: int = 1,
) -> dict[str, Any]:
    return {
        "artifact_type": "ninfer_serve_access_log",
        "schema_version": 1,
        "event": "access",
        "timestamp_unix_ms": 1_700_000_000_000 + access_id,
        "server_instance_id": "serve-test",
        "access_id": access_id,
        "http": {
            "method": "POST",
            "path": path,
            "target": path,
            "status": status,
            "remote_addr": "192.168.1.20",
            "remote_port": 5000,
            "user_agent": "agent/1",
            "request_content_type": "application/json",
            "response_content_type": "text/event-stream" if streamed else "application/json",
            "request_bytes": 10,
            "response_bytes": 10,
            "duration_seconds": 1.5,
            "streamed": streamed,
        },
        "request_body": request_body,
        "request_body_truncated": False,
        "response_body": response_body,
        "response_body_truncated": False,
    }


def build(tmp_path: Path, records: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    capture = tmp_path / "capture"
    capture.mkdir(exist_ok=True)
    with (capture / "access.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    out = tmp_path / "dataset"
    stats = build_dataset.build([str(capture)], out, kwargs.get("min_completion_chars", 1), set())
    return {"stats": stats, "out": out}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_openai_chat_completions_non_streaming(tmp_path: Path) -> None:
    request = {
        "model": "qwen3.8-27b-nvfp4",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    response = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "let me look",
                    "reasoning_content": "thinking",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    }
    result = build(tmp_path, [access("/v1/chat/completions", request, response)])
    chat = read_jsonl(result["out"] / "chat.jsonl")
    assert len(chat) == 1
    assert chat[0]["protocol"] == "openai_chat_completions"
    assert chat[0]["messages"][0] == {"role": "user", "content": "hello"}
    assert chat[0]["completion"]["content"] == "let me look"
    assert chat[0]["completion"]["reasoning_content"] == "thinking"
    assert chat[0]["completion"]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
    assert chat[0]["usage"] == {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
    assert chat[0]["finish_reason"] == "tool_calls"
    assert chat[0]["tools"][0]["function"]["name"] == "f"

    sft = read_jsonl(result["out"] / "sft.jsonl")
    assert len(sft) == 1
    assert [message["role"] for message in sft[0]["messages"]] == ["user", "assistant"]


def test_anthropic_messages_normalize_system_thinking_and_tools(tmp_path: Path) -> None:
    request = {
        "model": "qwen3.8-27b-nvfp4",
        "system": [{"type": "text", "text": "be brief"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "source": {}}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}],
            },
        ],
        "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
    }
    response = {
        "content": [
            {"type": "thinking", "thinking": "why"},
            {"type": "text", "text": "done"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 30, "output_tokens": 5},
    }
    result = build(tmp_path, [access("/v1/messages", request, response)])
    chat = read_jsonl(result["out"] / "chat.jsonl")[0]
    roles = [message["role"] for message in chat["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert chat["messages"][0]["content"] == "be brief"
    assert chat["messages"][1]["content"] == "hi[image]"
    assert chat["messages"][2]["reasoning_content"] == "hmm"
    assert chat["messages"][2]["tool_calls"][0]["id"] == "toolu_1"
    assert chat["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert chat["messages"][3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "42"}
    assert chat["tools"] == [
        {
            "type": "function",
            "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
        }
    ]
    assert chat["completion"] == {"role": "assistant", "content": "done", "reasoning_content": "why"}
    assert chat["usage"] == {"prompt_tokens": 30, "completion_tokens": 5}
    assert chat["finish_reason"] == "end_turn"
    assert chat["warnings"] == ["1 non-text content part(s) replaced by a placeholder"]


def test_anthropic_streaming_reconstructs_deltas(tmp_path: Path) -> None:
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    stream = sse(
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 9}}}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "f", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "th"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "he"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "llo"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a"'}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ":1}"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}}),
        ("message_stop", {"type": "message_stop"}),
    )
    result = build(tmp_path, [access("/v1/messages", request, stream, streamed=True)])
    chat = read_jsonl(result["out"] / "chat.jsonl")[0]
    assert chat["completion"]["content"] == "hello"
    assert chat["completion"]["reasoning_content"] == "th"
    assert chat["completion"]["tool_calls"][0]["id"] == "toolu_9"
    assert chat["completion"]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
    assert chat["usage"] == {"prompt_tokens": 9, "completion_tokens": 4}
    assert chat["finish_reason"] == "tool_use"


def test_openai_chat_streaming_and_responses_streaming(tmp_path: Path) -> None:
    chat_stream = sse(
        (None, {"choices": [{"delta": {"role": "assistant"}}]}),
        (None, {"choices": [{"delta": {"reasoning_content": "r"}}]}),
        (None, {"choices": [{"delta": {"content": "a"}}]}),
        (None, {"choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}]}),
        (None, {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
    )
    responses_stream = sse(
        ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "ignored"}),
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
                        {"type": "message", "content": [{"type": "output_text", "text": "final"}]},
                        {"type": "function_call", "name": "f", "arguments": '{"b":2}', "call_id": "call_9"},
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 6},
                },
            },
        ),
    )
    result = build(
        tmp_path,
        [
            access("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "q"}]}, chat_stream, streamed=True, access_id=1),
            access("/v1/responses", {"model": "m", "input": "q"}, responses_stream, streamed=True, access_id=2),
        ],
    )
    chat = read_jsonl(result["out"] / "chat.jsonl")
    assert chat[0]["completion"] == {"role": "assistant", "content": "ab", "reasoning_content": "r"}
    assert chat[0]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert chat[0]["finish_reason"] == "stop"
    assert chat[1]["protocol"] == "openai_responses"
    assert chat[1]["messages"] == [{"role": "user", "content": "q"}]
    assert chat[1]["completion"]["content"] == "final"
    assert chat[1]["completion"]["reasoning_content"] == "why"
    assert chat[1]["completion"]["tool_calls"][0]["function"]["arguments"] == '{"b":2}'
    assert chat[1]["usage"] == {"prompt_tokens": 11, "completion_tokens": 6}


def test_responses_items_normalize_to_messages(tmp_path: Path) -> None:
    request = {
        "model": "m",
        "instructions": "be terse",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
            {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "call_1"},
            {"type": "function_call_output", "call_id": "call_1", "output": [{"type": "input_text", "text": "sunny"}]},
        ],
        "tools": [{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}],
    }
    response = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 5, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}},
    }
    result = build(tmp_path, [access("/v1/responses", request, response)])
    chat = read_jsonl(result["out"] / "chat.jsonl")[0]
    assert [message["role"] for message in chat["messages"]] == ["system", "user", "assistant", "tool"]
    assert chat["messages"][2]["tool_calls"][0]["id"] == "call_1"
    assert chat["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    assert chat["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert chat["usage"]["reasoning_tokens"] == 0


def test_dedup_conversations_and_non_model_requests(tmp_path: Path) -> None:
    request = {"model": "m", "messages": [{"role": "user", "content": "same"}]}
    response = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}]}
    records = [
        access("/v1/chat/completions", request, response, access_id=1),
        # The same turn again: agents retry, and the same prompt reaching the same answer twice is
        # one training sample.
        access("/v1/chat/completions", request, response, access_id=2),
        # A second turn of the same conversation: the history grows, the head does not.
        access(
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "same"}, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "more"}]},
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "second"}}]},
            access_id=3,
        ),
        # Not a model access at all.
        access("/metrics", {}, "ninfer_requests 0", access_id=4),
        # A rejected request: captured, counted, but never a training sample.
        access("/v1/chat/completions", request, {"error": {"message": "nope"}}, status=400, access_id=5),
    ]
    result = build(tmp_path, records)
    chat = read_jsonl(result["out"] / "chat.jsonl")
    assert len(chat) == 3
    assert {record["conversation_id"] for record in chat} == {chat[0]["conversation_id"]}
    assert len({record["conversation_id"] for record in chat}) == 1
    sft = read_jsonl(result["out"] / "sft.jsonl")
    assert [record["messages"][-1]["content"] for record in sft] == ["answer", "second"]
    stats = result["stats"]
    assert stats["accesses"]["kept"] == 3
    assert stats["accesses"]["dropped"] == 1
    assert stats["accesses"]["by_status"] == {"200": 3, "400": 1}
    assert stats["dataset"]["duplicates"] == 1
    assert stats["capture"]["non_model_requests"] == 1
    assert stats["conversations"]["count"] == 1


def test_short_or_broken_turns_are_captured_but_not_trained_on(tmp_path: Path) -> None:
    request = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
    truncated = access("/v1/chat/completions", request, {"choices": []}, access_id=1)
    truncated["response_body_truncated"] = True
    unparseable = access("/v1/messages", request, "not json", access_id=2)
    empty = access(
        "/v1/chat/completions",
        request,
        {"choices": [{"message": {"content": ""}}]},
        access_id=3,
    )
    result = build(tmp_path, [truncated, unparseable, empty])
    # chat.jsonl is the capture view: an access that produced nothing is still an access, and it
    # says why it could not be read.
    chat = read_jsonl(result["out"] / "chat.jsonl")
    assert len(chat) == 3
    assert chat[0]["warnings"] == ["response body truncated by the server"]
    assert chat[1]["warnings"] == ["response body did not parse as JSON"]
    # sft.jsonl is the training view, and there is nothing to train on here.
    assert read_jsonl(result["out"] / "sft.jsonl") == []
    stats = result["stats"]
    assert stats["accesses"]["kept"] == 3
    assert stats["accesses"]["dropped"] == 0
    assert stats["accesses"]["warnings"] == 2
    assert stats["dataset"]["empty_turns_skipped"] == 3


def test_conflicting_inputs_are_reported(tmp_path: Path) -> None:
    with pytest.raises(build_dataset.DatasetError):
        build_dataset.access_files([str(tmp_path / "missing")])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(build_dataset.DatasetError):
        build_dataset.access_files([str(empty)])
