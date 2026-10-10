#!/usr/bin/env python3
"""Lane-admission reuse probe: a fresh prompt must not evict a retained prefix.

Regression probe for the admission choice in ``find_admission_lane``.  It drives ``SESSIONS``
distinct prompts through the retained-context pool and checks the schema-v10 request log after
each phase:

1. ``serial-warm`` sends every prompt once, so the pool ends up holding one retained prefix per
   occupied lane;
2. ``serial-replay`` sends them again one at a time -- each must resume its own prefix;
3. ``concurrent-replay`` sends them all at once -- every lane must still resume its own prefix;
4. ``serial-after-concurrent`` repeats the first prompt once more.

Before the admission fix a fresh prompt always took the lowest-numbered admissible lane, dropping
whatever prefix was retained there even while other lanes sat idle, so the replays fell back to a
full re-prefill.  Every replay record must therefore show ``prompt_tokens - prefix_cache_hit_tokens
<= 2`` with a reuse path of ``append_frontier`` or ``restore_turn_checkpoint``; the probe exits
non-zero otherwise.

Run it against a server started with tools/v100/context-lab/serve-batch-decode.sh and
``CONTEXT_LAB_REQUEST_LOG`` pointing at ``--request-log``:

    tools/v100/context-lab/reuse_probe.py --request-log logs/requests.jsonl
"""
import argparse
import json
import sys
import time
import urllib.request
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
    """A distinct, deterministic prompt of roughly ``lines`` short functions."""
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
    """The ``expected`` request_done records appended to ``log_path`` past ``offset``."""
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
            raise SystemExit(f"timeout waiting for {expected} request_done records in {log_path}")
        time.sleep(0.5)


def report(label, records, expect_reuse=True):
    """Print one line per record and return the number that failed to reuse.

    ``expect_reuse`` is false for the warm phase, which is what *creates* the retained prefixes
    and so legitimately reports a cold prefill for some of its requests.
    """
    failures = 0
    for record in records:
        result = record["result"]
        prompt = result["prompt_tokens"]
        prefill = result["computed_prefill_tokens"]
        path = result["prefix_reuse_path"]
        ok = (prefill <= 2 and path in REUSE_PATHS) if expect_reuse else True
        failures += 0 if ok else 1
        print(f"{label} req{record['request']['request_id']}: prompt={prompt} "
              f"cache={result['prefix_cache_hit_tokens']} reuse={path} prefill={prefill} "
              f"{'OK' if ok else 'FAIL'}", flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--request-log", required=True, type=Path)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--warm-tokens", type=int, default=2)
    parser.add_argument("--report-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()

    client = Client(args.base_url, args.model_id, args.timeout_seconds)
    log_path = args.request_log
    tag = f"pr{int(time.time()) % 100000}"
    prompts = [build_prompt(f"{tag}-{session}", 19) for session in range(args.sessions)]
    print("prompt tokens:", [client.count_tokens(prompt) for prompt in prompts], flush=True)

    failures = 0
    offset = log_path.stat().st_size
    for prompt in prompts:
        client.chat(prompt, args.warm_tokens)
    report("serial-warm", done_after(log_path, offset, args.sessions, args.timeout_seconds),
           expect_reuse=False)

    offset = log_path.stat().st_size
    for prompt in prompts:
        client.chat(prompt, args.report_tokens)
    failures += report("serial-replay", done_after(log_path, offset, args.sessions,
                                                   args.timeout_seconds))

    offset = log_path.stat().st_size
    with ThreadPoolExecutor(max_workers=args.sessions) as pool:
        list(pool.map(lambda prompt: client.chat(prompt, args.report_tokens), prompts))
    failures += report("concurrent-replay", done_after(log_path, offset, args.sessions,
                                                       args.timeout_seconds))

    offset = log_path.stat().st_size
    client.chat(prompts[0], args.report_tokens)
    failures += report("serial-after-concurrent", done_after(log_path, offset, 1,
                                                             args.timeout_seconds))

    print("FAIL: retained prefix was dropped" if failures else "OK: every replay reused its prefix",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
