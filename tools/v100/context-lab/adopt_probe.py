#!/usr/bin/env python3
"""Cross-lane prefix adoption gate: an adopted prefix must decode the cold text byte for byte.

The probe builds one prompt, decodes it cold for a reference completion, decodes it once more with a
two-token output -- which leaves the prefix and the rewrite checkpoint that turn captured retained on
the lane that served it -- and then runs the same prompt twice concurrently. The second request of
that pair cannot start on the lane holding the prefix, so it reaches it by adoption (borrowed pages
plus a copied continuation state) or not at all. Both streams must reproduce the cold text exactly,
and both must report at most a two-token prefill on a reuse path.

Usage: a server started with ``--request-log-jsonl LOG`` must already be listening, then

    tools/v100/context-lab/adopt_probe.py --request-log LOG
"""
import argparse, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL_ID = "qwen3.8-27b-nvfp4"
REUSE_PATHS = {"append_frontier", "restore_turn_checkpoint"}


class Client:
    def __init__(self, base_url, model_id, timeout):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout

    def post(self, path, body):
        request = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def count_tokens(self, text):
        return self.post("/v1/messages/count_tokens", {
            "model": self.model_id,
            "messages": [{"role": "user", "content": text}]})["input_tokens"]

    def chat(self, text, max_tokens):
        return self.post("/v1/chat/completions", {
            "model": self.model_id, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": text}]})


def build_prompt(tag, lines):
    out = [f"# session {tag} working notes"]
    for index in range(lines):
        out.append(
            f"def {tag}_{index}(value, weight):\n"
            f"    total = value * {index} + weight\n"
            f"    if total % 7 == {index % 7}:\n"
            f"        return total - {index}\n"
            f"    return total + {index + 1}\n")
    return "\n".join(out)


def done_after(log_path, offset, expected, timeout):
    deadline = time.time() + timeout
    while True:
        records = []
        for line in log_path.read_text(errors="replace")[offset:].splitlines():
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "request_done":
                records.append(record)
        if len(records) >= expected:
            return records
        if time.time() > deadline:
            raise SystemExit(f"timeout waiting for {expected} request_done records")
        time.sleep(0.5)


def text_of(response):
    """Both output channels: a thinking model spends its first tokens on `reasoning_content`."""
    message = response["choices"][0]["message"]
    return (message.get("reasoning_content") or "") + "\x00" + (message.get("content") or "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--request-log", required=True, type=Path)
    parser.add_argument("--lines", type=int, default=19)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    args = parser.parse_args()

    client = Client(args.base_url, args.model_id, args.timeout_seconds)
    log_path = args.request_log
    tag = f"ad{int(time.time()) % 100000}"
    prompt = build_prompt(tag, args.lines)
    print("prompt tokens:", client.count_tokens(prompt), flush=True)

    offset = log_path.stat().st_size
    reference = text_of(client.chat(prompt, args.max_tokens))
    cold = done_after(log_path, offset, 1, args.timeout_seconds)
    print(f"cold     : reuse={cold[0]['result']['prefix_reuse_path']} "
          f"prefill={cold[0]['result']['computed_prefill_tokens']} "
          f"text={reference[:80]!r} chars={len(reference)}", flush=True)
    if len(reference) <= 1:
        print("FAIL: the reference completion is empty; nothing to compare", flush=True)
        return 1

    # A second identical prompt with a short output leaves the prefix -- and the rewrite
    # checkpoint the first turn captured -- retained on the lane that served it. The pair below
    # then runs on that lane plus a sibling that can only reach the prefix by adopting it.
    offset = log_path.stat().st_size
    client.chat(prompt, 2)
    done_after(log_path, offset, 1, args.timeout_seconds)

    offset = log_path.stat().st_size
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.chat, prompt, args.max_tokens) for _ in range(2)]
        texts = [text_of(future.result()) for future in futures]
    pair = done_after(log_path, offset, 2, args.timeout_seconds)

    failures = 0
    for record in pair:
        result = record["result"]
        ok = (result["computed_prefill_tokens"] <= 2 and
              result["prefix_reuse_path"] in REUSE_PATHS)
        failures += 0 if ok else 1
        print(f"concurrent req{record['request']['request_id']}: "
              f"prompt={result['prompt_tokens']} cache={result['prefix_cache_hit_tokens']} "
              f"reuse={result['prefix_reuse_path']} "
              f"prefill={result['computed_prefill_tokens']} {'OK' if ok else 'FAIL'}", flush=True)
    for index, text in enumerate(texts):
        same = text == reference
        failures += 0 if same else 1
        print(f"concurrent text[{index}]: {'IDENTICAL' if same else 'DIFFERS'} "
              f"({len(text)} chars)", flush=True)

    print("FAIL: an adopted prefix did not reproduce the cold output" if failures
          else "OK: adopted prefixes reused and decoded the cold text", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
