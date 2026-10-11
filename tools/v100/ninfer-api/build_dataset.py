#!/usr/bin/env python3
"""Build a dataset from the lossless access capture of a running ninfer-serve.

`ninfer-serve --access-log-dir DIR` writes one `ninfer_serve_access_log` record per HTTP request,
holding the client's body and the server's body verbatim. This tool reads those files and turns
them into protocol-independent records:

  chat.jsonl   one line per successful model access, unfiltered and undeduplicated: the
               conversation the client sent, normalized to OpenAI-style messages, plus the
               assistant turn the server produced
  sft.jsonl    the training view: the same pairs, minus turns with no assistant output,
               deduplicated, with the assistant turn last
  stats.json   what was captured, by protocol, endpoint, tokens, latency and dedup

The access log is raw by design, so this is where protocol knowledge lives. A body that does not
parse is carried through as text with a `warnings` entry rather than dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


class DatasetError(Exception):
    """A capture that cannot be turned into a dataset."""


# --------------------------------------------------------------------------------------------
# Reading the capture
# --------------------------------------------------------------------------------------------


def access_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("access*.jsonl") if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise DatasetError(f"no access log at {path}")
    if not files:
        raise DatasetError("no access*.jsonl file found in the given paths")
    return files


def load_records(paths: Iterable[str]) -> Iterator[dict[str, Any]]:
    for path in access_files(paths):
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetError(f"{path}:{number}: {error}") from error
                if record.get("event") != "access":
                    continue
                record["_source"] = {"file": path.name, "line": number}
                yield record


def model_protocol(path: str) -> str | None:
    """The generation endpoints, and nothing else, produce a dataset record."""
    return {
        "/v1/chat/completions": "openai_chat_completions",
        "/v1/messages": "anthropic_messages",
        "/v1/responses": "openai_responses",
    }.get(path)


# --------------------------------------------------------------------------------------------
# Server-Sent Events
# --------------------------------------------------------------------------------------------


def parse_sse(text: str) -> list[tuple[str | None, Any]]:
    """Return (event name, parsed JSON) for every frame, keeping frames that do not parse."""
    frames: list[tuple[str | None, Any]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name: str | None = None
        data: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:") :].lstrip())
        if not data:
            continue
        payload = "\n".join(data)
        if payload == "[DONE]":
            continue
        try:
            frames.append((name, json.loads(payload)))
        except json.JSONDecodeError:
            frames.append((name, payload))
    return frames


# --------------------------------------------------------------------------------------------
# Canonical messages (OpenAI chat-completions shape)
# --------------------------------------------------------------------------------------------


def _text_of(parts: Any) -> tuple[str, int]:
    """Join a content-part list into text; images and documents become a counted placeholder."""
    if isinstance(parts, str):
        return parts, 0
    text_parts: list[str] = []
    dropped = 0
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text", "summary_text"):
            text_parts.append(part.get("text") or part.get("content") or "")
        elif kind in ("image", "input_image", "image_url", "document", "input_audio", "video"):
            dropped += 1
            text_parts.append(f"[{kind}]")
    return "".join(text_parts), dropped


def _function_call(name: str, arguments: str, call_id: str = "") -> dict[str, Any]:
    return {
        "id": call_id or name,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _tool_result_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        text, _ = _text_of(output)
        return text
    if output is None:
        return ""
    return json.dumps(output, ensure_ascii=False)


def messages_from_openai_chat(request: dict[str, Any]) -> tuple[list[Any], int]:
    """OpenAI chat messages are already the canonical shape; only parts are flattened."""
    messages: list[Any] = []
    dropped = 0
    for turn in request.get("messages") or []:
        message = dict(turn)
        if isinstance(message.get("content"), list):
            message["content"], lost = _text_of(message["content"])
            dropped += lost
        messages.append(message)
    return messages, dropped


def messages_from_anthropic(request: dict[str, Any]) -> tuple[list[Any], int]:
    messages: list[Any] = []
    dropped = 0
    system = request.get("system")
    if system:
        text, lost = _text_of(system)
        dropped += lost
        if text:
            messages.append({"role": "system", "content": text})
    for turn in request.get("messages") or []:
        role = turn.get("role", "user")
        content = turn.get("content")
        if isinstance(content, str) or content is None:
            messages.append({"role": role, "content": content or ""})
            continue
        text_parts: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text":
                text_parts.append(part.get("text", ""))
            elif kind == "thinking":
                reasoning.append(part.get("thinking", ""))
            elif kind == "tool_use":
                tool_calls.append(
                    _function_call(
                        part.get("name", ""),
                        json.dumps(part.get("input", {}), ensure_ascii=False),
                        part.get("id", ""),
                    )
                )
            elif kind == "tool_result":
                tool_results.append(part)
            elif kind in ("image", "document"):
                dropped += 1
                text_parts.append(f"[{kind}]")
        text = "".join(text_parts)
        # A user turn that only carries tool results produces no user message of its own: the
        # results are the message.
        if text or reasoning or tool_calls or not tool_results:
            message: dict[str, Any] = {"role": role, "content": text}
            if reasoning:
                message["reasoning_content"] = "".join(reasoning)
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
        # A tool result answers the assistant's call, so it is its own message, never part of the
        # user turn that carried it.
        for result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.get("tool_use_id", ""),
                    "content": _tool_result_text(result.get("content")),
                }
            )
    return messages, dropped


def messages_from_responses(request: dict[str, Any]) -> tuple[list[Any], int]:
    messages: list[Any] = []
    dropped = 0
    instructions = request.get("instructions")
    if instructions:
        text, lost = _text_of(instructions)
        dropped += lost
        if text:
            messages.append({"role": "system", "content": text})
    value = request.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "message" or ("role" in item and "content" in item):
                text, lost = _text_of(item.get("content"))
                dropped += lost
                messages.append({"role": item.get("role", "user"), "content": text})
            elif kind == "function_call":
                call = _function_call(
                    item.get("name", ""), item.get("arguments", ""), item.get("call_id", "")
                )
                if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                    messages[-1]["tool_calls"].append(call)
                else:
                    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            elif kind == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": _tool_result_text(item.get("output")),
                    }
                )
            elif kind == "reasoning":
                text, _ = _text_of(item.get("summary"))
                if not text:
                    text, _ = _text_of(item.get("content"))
                if text:
                    messages.append(
                        {"role": "assistant", "content": "", "reasoning_content": text}
                    )
    return messages, dropped


def messages_of(protocol: str, request: dict[str, Any]) -> tuple[list[Any], int]:
    if protocol == "openai_chat_completions":
        return messages_from_openai_chat(request)
    if protocol == "anthropic_messages":
        return messages_from_anthropic(request)
    if protocol == "openai_responses":
        return messages_from_responses(request)
    raise DatasetError(f"unknown protocol {protocol}")


def tools_of(protocol: str, request: dict[str, Any]) -> list[Any] | None:
    """Tools in the canonical OpenAI function shape, whatever the protocol sent."""
    tools = request.get("tools")
    if not tools:
        return None
    if protocol == "anthropic_messages":
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in tools
            if isinstance(tool, dict)
        ]
    if protocol == "openai_responses":
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") == "function"
        ]
    return tools


# --------------------------------------------------------------------------------------------
# The assistant turn
# --------------------------------------------------------------------------------------------


def _assistant(content: str, reasoning: str, tool_calls: list[Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _usage_of(value: Any) -> dict[str, Any]:
    """Normalize the three protocols' usage objects onto OpenAI's three counts."""
    if not isinstance(value, dict):
        return {}
    usage: dict[str, Any] = {}
    prompt = value.get("prompt_tokens", value.get("input_tokens"))
    completion = value.get("completion_tokens", value.get("output_tokens"))
    if isinstance(prompt, int):
        usage["prompt_tokens"] = prompt
    if isinstance(completion, int):
        usage["completion_tokens"] = completion
    if isinstance(value.get("total_tokens"), int):
        usage["total_tokens"] = value["total_tokens"]
    details = value.get("output_tokens_details") or value.get("completion_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
        usage["reasoning_tokens"] = details["reasoning_tokens"]
    return usage


def _tool_calls_of(value: Any) -> list[Any]:
    calls: list[Any] = []
    for call in value or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        calls.append(
            _function_call(
                function.get("name", call.get("name", "")),
                function.get("arguments", call.get("arguments", "")),
                call.get("id", call.get("call_id", "")),
            )
        )
    return calls


def completion_from_chat_json(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    choices = body.get("choices") or [{}]
    message = choices[0].get("message") or {}
    completion = _assistant(
        message.get("content") or "",
        message.get("reasoning_content") or message.get("reasoning") or "",
        _tool_calls_of(message.get("tool_calls")),
    )
    return completion, _usage_of(body.get("usage")), choices[0].get("finish_reason") or ""


def completion_from_chat_stream(text: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    finish = ""
    for _, frame in parse_sse(text):
        if not isinstance(frame, dict):
            continue
        if isinstance(frame.get("usage"), dict):
            usage = _usage_of(frame["usage"])
        for choice in frame.get("choices") or []:
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning.append(delta["reasoning_content"])
            for call in delta.get("tool_calls") or []:
                index = call.get("index", 0)
                slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    slot["arguments"] += function["arguments"]
    tool_calls = [
        _function_call(slot["name"], slot["arguments"], slot["id"]) for slot in calls.values()
    ]
    return (
        _assistant("".join(content), "".join(reasoning), tool_calls),
        usage,
        finish,
    )


def completion_from_anthropic_json(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    text: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[Any] = []
    for part in body.get("content") or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text.append(part.get("text", ""))
        elif part.get("type") == "thinking":
            reasoning.append(part.get("thinking", ""))
        elif part.get("type") == "tool_use":
            tool_calls.append(
                _function_call(
                    part.get("name", ""),
                    json.dumps(part.get("input", {}), ensure_ascii=False),
                    part.get("id", ""),
                )
            )
    usage = _usage_of(body.get("usage"))
    return (
        _assistant("".join(text), "".join(reasoning), tool_calls),
        usage,
        body.get("stop_reason") or "",
    )


def completion_from_anthropic_stream(text: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    finish = ""
    for _, frame in parse_sse(text):
        if not isinstance(frame, dict):
            continue
        kind = frame.get("type")
        if kind == "content_block_start":
            # The server names this `content_block`; `block` is what some SDKs use.
            block = frame.get("content_block") or frame.get("block") or {}
            if block.get("type") == "tool_use":
                calls[frame.get("index", 0)] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": "",
                }
        elif kind == "content_block_delta":
            delta = frame.get("delta") or {}
            if delta.get("type") == "text_delta":
                content.append(delta.get("text", ""))
            elif delta.get("type") == "thinking_delta":
                reasoning.append(delta.get("thinking", ""))
            elif delta.get("type") == "input_json_delta":
                slot = calls.setdefault(frame.get("index", 0), {"id": "", "name": "", "arguments": ""})
                slot["arguments"] += delta.get("partial_json", "")
        elif kind == "message_delta":
            if isinstance(frame.get("usage"), dict):
                usage.update(_usage_of(frame["usage"]))
            delta = frame.get("delta") or {}
            if delta.get("stop_reason"):
                finish = delta["stop_reason"]
        elif kind == "message_start":
            message = frame.get("message") or {}
            if isinstance(message.get("usage"), dict):
                usage.update(_usage_of(message["usage"]))
    tool_calls = [
        _function_call(slot["name"], slot["arguments"], slot["id"]) for slot in calls.values()
    ]
    return _assistant("".join(content), "".join(reasoning), tool_calls), usage, finish


def completion_from_responses_json(
    body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    text: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[Any] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            part, _ = _text_of(item.get("summary"))
            if not part:
                part, _ = _text_of(item.get("content"))
            reasoning.append(part)
        elif item.get("type") == "function_call":
            tool_calls.append(
                _function_call(
                    item.get("name", ""), item.get("arguments", ""), item.get("call_id", "")
                )
            )
        elif item.get("type") == "message":
            part, _ = _text_of(item.get("content"))
            text.append(part)
    return (
        _assistant("".join(text), "".join(reasoning), tool_calls),
        _usage_of(body.get("usage")),
        body.get("status") or "",
    )


def completion_from_responses_stream(text: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """The stream's terminal event carries the whole response object, so ask it, not the deltas."""
    for name, frame in reversed(parse_sse(text)):
        if isinstance(frame, dict) and isinstance(frame.get("response"), dict):
            return completion_from_responses_json(frame["response"])
        if name in ("response.failed", "error"):
            break
    return _assistant("", "", []), {}, ""


def completion_of(
    protocol: str, body: Any, streamed: bool
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if isinstance(body, str):
        if not streamed:
            raise DatasetError("response body did not parse as JSON")
        if protocol == "openai_chat_completions":
            return completion_from_chat_stream(body)
        if protocol == "anthropic_messages":
            return completion_from_anthropic_stream(body)
        if protocol == "openai_responses":
            return completion_from_responses_stream(body)
    if streamed:
        # A stream that produced no frames at all (an immediate disconnect) leaves nothing to read.
        return _assistant("", "", []), {}, ""
    if isinstance(body, dict):
        if protocol == "openai_chat_completions":
            return completion_from_chat_json(body)
        if protocol == "anthropic_messages":
            return completion_from_anthropic_json(body)
        if protocol == "openai_responses":
            return completion_from_responses_json(body)
    raise DatasetError(f"{protocol}: unusable response body")


# --------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------


def conversation_id(protocol: str, messages: list[Any]) -> str:
    """A conversation is what its system prompt and first user turn make it, so every turn of one
    agent session groups together even though each turn resends the whole history."""
    head: list[Any] = []
    for message in messages:
        if message.get("role") in ("system", "developer") and not head:
            head.append({"role": "system", "content": message.get("content")})
        elif message.get("role") in ("user", "tool"):
            head.append({"role": "user", "content": message.get("content")})
            break
    digest = hashlib.sha256(
        json.dumps([protocol, head], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def dataset_record(record: dict[str, Any]) -> dict[str, Any] | None:
    http = record.get("http") or {}
    protocol = model_protocol(http.get("path", ""))
    if protocol is None:
        return None
    request_body = record.get("request_body")
    if not isinstance(request_body, dict):
        return None
    warnings: list[str] = []
    if record.get("request_body_truncated"):
        warnings.append("request body truncated by the server")
    if record.get("response_body_truncated"):
        warnings.append("response body truncated by the server")
    messages, dropped = messages_of(protocol, request_body)
    if dropped:
        warnings.append(f"{dropped} non-text content part(s) replaced by a placeholder")
    try:
        completion, usage, finish = completion_of(
            protocol, record.get("response_body"), bool(http.get("streamed"))
        )
    except DatasetError as error:
        warnings.append(str(error))
        completion, usage, finish = _assistant("", "", []), {}, ""
    return {
        "id": f"{record.get('server_instance_id', 'serve')}:{record.get('access_id', 0)}",
        "timestamp_unix_ms": record.get("timestamp_unix_ms"),
        "protocol": protocol,
        "model": request_body.get("model", ""),
        "stream": bool(http.get("streamed")),
        "status": http.get("status", 0),
        "conversation_id": conversation_id(protocol, messages),
        "messages": messages,
        "tools": tools_of(protocol, request_body),
        "completion": completion,
        "usage": usage,
        "finish_reason": finish,
        "latency_seconds": http.get("duration_seconds"),
        "remote_addr": http.get("remote_addr", ""),
        "warnings": warnings,
    }


def has_text(record: dict[str, Any]) -> bool:
    completion = record["completion"]
    return bool(completion.get("content")) or bool(completion.get("tool_calls"))


def sft_key(record: dict[str, Any]) -> str:
    payload = {
        "messages": record["messages"],
        "tools": record["tools"],
        "completion": record["completion"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sft_record(record: dict[str, Any]) -> dict[str, Any]:
    """One trainable sample: the conversation as it arrived, then the assistant's answer."""
    messages = list(record["messages"]) + [record["completion"]]
    return {
        "conversation_id": record["conversation_id"],
        "id": record["id"],
        "messages": messages,
        "tools": record["tools"],
        "usage": record["usage"],
    }


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def summarize(
    scanned: dict[str, int],
    kept: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    chat_written: int,
    sft_written: int,
    duplicates: int,
    empty_turns: int,
) -> dict[str, Any]:
    def tally(records: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            value = str(record.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    prompt_tokens = [
        record["usage"]["prompt_tokens"]
        for record in kept
        if isinstance(record["usage"].get("prompt_tokens"), int)
    ]
    completion_tokens = [
        record["usage"].get("completion_tokens", 0) for record in kept
    ]
    latencies = [
        record["latency_seconds"]
        for record in kept
        if isinstance(record.get("latency_seconds"), (int, float))
    ]
    tool_calls = sum(1 for record in kept if record["completion"].get("tool_calls"))
    reasoning = sum(1 for record in kept if record["completion"].get("reasoning_content"))
    return {
        "capture": scanned,
        "accesses": {
            "dataset_records": len(kept) + len(dropped),
            "kept": len(kept),
            "dropped": len(dropped),
            "by_protocol": tally(kept, "protocol"),
            "by_model": tally(kept, "model"),
            "by_status": tally(kept + dropped, "status"),
            "streamed": sum(1 for record in kept if record["stream"]),
            "with_tool_calls": tool_calls,
            "with_reasoning": reasoning,
            "warnings": sum(1 for record in kept if record["warnings"]),
        },
        "conversations": {
            "count": len({record["conversation_id"] for record in kept}),
            "turns_per_conversation": round(
                len(kept) / max(1, len({record["conversation_id"] for record in kept})), 2
            ),
        },
        "dataset": {
            "chat_records": chat_written,
            "sft_records": sft_written,
            "duplicates": duplicates,
            "empty_turns_skipped": empty_turns,
        },
        "tokens": {
            "prompt_total": sum(prompt_tokens),
            "prompt_p50": percentile(prompt_tokens, 0.5),
            "prompt_p90": percentile(prompt_tokens, 0.9),
            "completion_total": sum(completion_tokens),
            "completion_p50": percentile(completion_tokens, 0.5),
            "completion_p90": percentile(completion_tokens, 0.9),
        },
        "latency_seconds": {
            "p50": percentile(latencies, 0.5),
            "p90": percentile(latencies, 0.9),
            "max": max(latencies) if latencies else 0.0,
        },
        "finish_reasons": tally(kept, "finish_reason"),
    }


def print_summary(stats: dict[str, Any], out: Path) -> None:
    accesses = stats["accesses"]
    print(f"scanned {stats['capture']['files']} file(s), {stats['capture']['lines']} record(s)")
    print(
        f"  model accesses kept {accesses['kept']}, dropped {accesses['dropped']} "
        f"({accesses['streamed']} streamed, {accesses['with_tool_calls']} with tool calls)"
    )
    print(f"  by protocol: {accesses['by_protocol'] or '{}'}")
    print(f"  conversations {stats['conversations']['count']} "
          f"({stats['conversations']['turns_per_conversation']} accesses each)")
    print(f"  prompt tokens total {stats['tokens']['prompt_total']} "
          f"p50 {stats['tokens']['prompt_p50']} p90 {stats['tokens']['prompt_p90']}")
    print(f"  completion tokens total {stats['tokens']['completion_total']} "
          f"p50 {stats['tokens']['completion_p50']} p90 {stats['tokens']['completion_p90']}")
    print(f"  dataset: chat {stats['dataset']['chat_records']}, "
          f"sft {stats['dataset']['sft_records']} "
          f"({stats['dataset']['duplicates']} duplicate turn(s) dropped)")
    print(f"wrote {out}/chat.jsonl, {out}/sft.jsonl, {out}/stats.json")


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def build(paths: list[str], out: Path, min_completion_chars: int, protocols: set[str]) -> dict[str, Any]:
    scanned = {"files": 0, "lines": 0, "other_events": 0, "non_model_requests": 0}
    scanned["files"] = len(access_files(paths))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for record in load_records(paths):
        scanned["lines"] += 1
        http = record.get("http") or {}
        if model_protocol(http.get("path", "")) is None:
            scanned["non_model_requests"] += 1
            continue
        parsed = dataset_record(record)
        if parsed is None:
            scanned["other_events"] += 1
            continue
        if http.get("status") != 200:
            dropped.append(parsed)
            continue
        if protocols and parsed["protocol"] not in protocols:
            continue
        kept.append(parsed)

    out.mkdir(parents=True, exist_ok=True)
    chat_written = 0
    with (out / "chat.jsonl").open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            chat_written += 1

    seen: set[str] = set()
    duplicates = 0
    empty_turns = 0
    sft_written = 0
    with (out / "sft.jsonl").open("w", encoding="utf-8") as handle:
        for record in kept:
            completion = record["completion"]
            if not has_text(record):
                empty_turns += 1
                continue
            if (
                len(completion.get("content") or "") < min_completion_chars
                and not completion.get("tool_calls")
            ):
                empty_turns += 1
                continue
            key = sft_key(record)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            handle.write(json.dumps(sft_record(record), ensure_ascii=False) + "\n")
            sft_written += 1

    stats = summarize(scanned, kept, dropped, chat_written, sft_written, duplicates, empty_turns)
    (out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a dataset from a ninfer-serve access capture (--access-log-dir).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  build_dataset.py ~/v100/ninfer-api/access --out ~/v100/ninfer-api/dataset\n"
        ),
    )
    parser.add_argument("paths", nargs="+", help="access log files, or directories holding them")
    parser.add_argument("--out", required=True, help="output directory for the dataset")
    parser.add_argument(
        "--min-completion-chars",
        type=int,
        default=1,
        help="drop sft turns whose assistant text is shorter than this (default 1); "
        "chat.jsonl keeps them",
    )
    parser.add_argument(
        "--protocol",
        action="append",
        default=[],
        choices=["openai_chat_completions", "anthropic_messages", "openai_responses"],
        help="keep only this protocol (repeatable)",
    )
    parser.add_argument("--quiet", action="store_true", help="write the files without a summary")
    args = parser.parse_args(argv)
    try:
        stats = build(
            args.paths,
            Path(args.out).expanduser(),
            args.min_completion_chars,
            set(args.protocol),
        )
    except DatasetError as error:
        print(f"build_dataset: {error}", file=sys.stderr)
        return 1
    if not args.quiet:
        print_summary(stats, Path(args.out).expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
