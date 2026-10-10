#!/usr/bin/env python3
"""Serial long-context session pool driver for the retained-context tiers.

Builds ``SESSIONS`` distinct long prompts at a target token count and drives them one at a time
through the OpenAI chat-completions endpoint, so the server sees exactly one active request:

``cold``
    Send every session once, in order.  Each request displaces the previous lane, so every prefix
    is parked into the NVMe tier while it is displaced; the session still resident at exit is
    published by the engine as its turn completes.
``resume``
    Send the same sessions again, in the same order.  A retained prefix is restored instead of
    recomputed, so every request must report ``cache == prompt`` (modulo the frontier token) with
    ``reuse=append_frontier`` and a two-token prefill.  Restart the server between ``cold`` and
    ``resume`` to show that the tier survives a restart.

The line count that reaches the target token count is calibrated against
``/v1/messages/count_tokens`` and carried in the state file, so both phases build byte-identical
prompts.  Per-request numbers come from the operational log line that the server writes when a
request finishes:

    [req N] done finish=output_limit prompt=194950 gen=8 cache=194948 reuse=append_frontier \\
        ttft=5874ms prefill=0.3tok/s decode=78.1tok/s wall=6.01s speculative=mtp 3.50tok/round

Example (see README.md in this directory for the recorded 8 x 195K run):

    tools/v100/context-lab/serve-pool.sh > cold.log 2>&1 &
    python3 tools/v100/context-lab/pool_driver.py cold 8 192000 --serve-log cold.log --state s.json
    kill $(pgrep -x ninfer-serve)
    tools/v100/context-lab/serve-pool.sh > resume.log 2>&1 &
    python3 tools/v100/context-lab/pool_driver.py resume 8 192000 --serve-log resume.log --state s.json
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL_ID = "qwen3.8-27b-nvfp4"


class Client:
    def __init__(self, base_url, model_id, timeout):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout

    def post(self, path, body):
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def count_tokens(self, text):
        return self.post(
            "/v1/messages/count_tokens",
            {"model": self.model_id, "messages": [{"role": "user", "content": text}]},
        )["input_tokens"]

    def chat(self, text, max_tokens):
        return self.post(
            "/v1/chat/completions",
            {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": text}],
            },
        )


def body_text(tag, lines):
    """One deterministic line per code block, so the line count is the token knob."""
    out = [f"# session {tag} working notes"]
    for i in range(lines):
        out.append(
            f"def {tag}_{i}(value, weight):\n    total = value * {i} + weight\n"
            f"    if total % 7 == {i % 7}:\n        return total - {i}\n    return total + {i + 1}\n"
        )
    return "\n".join(out)


def parse_outcome(rest):
    fields = {}
    for key in ("finish", "reuse"):
        match = re.search(rf"{key}=(\S+)", rest)
        if match:
            fields[key] = match.group(1)
    for key in ("prompt", "gen", "cache"):
        match = re.search(rf"{key}=(\d+)", rest)
        if match:
            fields[key] = int(match.group(1))
    for key in ("prefill", "decode"):
        match = re.search(rf"{key}=([\d.]+)tok/s", rest)
        if match:
            fields[key] = float(match.group(1))
    match = re.search(r"ttft=(\d+)ms", rest)
    if match:
        fields["ttft_ms"] = int(match.group(1))
    return fields


class ServeLog:
    def __init__(self, path):
        self.path = Path(path)

    def text(self):
        return self.path.read_text(errors="replace")

    def frontier(self):
        return len(self.text())

    def next_outcome(self, before_len):
        """Block until a request finishes after ``before_len`` and return (request id, fields)."""
        while True:
            text = self.text()
            if len(text) > before_len:
                for line in text[before_len:].splitlines():
                    match = re.search(r"\[req (\d+)\] done (.+)$", line)
                    if match:
                        return int(match.group(1)), parse_outcome(match.group(2))
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("cold", "resume"))
    parser.add_argument("sessions", type=int)
    parser.add_argument("target_tokens", type=int)
    parser.add_argument("--serve-log", required=True, type=Path)
    parser.add_argument("--state", type=Path, default=Path("sessions.json"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()

    if args.serve_log.exists() and args.serve_log.stat().st_size == 0:
        print(f"warning: {args.serve_log} is empty; is the server redirecting its log there?",
              file=sys.stderr)

    client = Client(args.base_url, args.model_id, args.timeout_seconds)
    log = ServeLog(args.serve_log)

    if args.mode == "cold":
        probe_lines = 64
        per_line = client.count_tokens(body_text("probe", probe_lines)) / probe_lines
        lines = int(round(args.target_tokens / per_line))
        measured = client.count_tokens(body_text("probe", lines))
        lines = int(round(lines * args.target_tokens / measured))
        print(f"per_line={per_line:.3f} lines={lines} probe={measured}", flush=True)
        args.state.write_text(json.dumps({"lines": lines, "target": args.target_tokens,
                                         "n": args.sessions}) + "\n")
    else:
        state = json.loads(args.state.read_text())
        lines, args.sessions = state["lines"], state["n"]
        print(f"state lines={lines} sessions={args.sessions} target={state['target']}", flush=True)

    prompts = [body_text(f"pool{session}", lines) for session in range(args.sessions)]

    rows = []
    for step, prompt in enumerate(prompts):
        mark = log.frontier()
        started = time.time()
        client.chat(prompt, args.max_tokens)
        wall = time.time() - started
        request_id, outcome = log.next_outcome(mark)
        rows.append(outcome)
        print(f"{args.mode} step{step} session{step} req{request_id} "
              f"prompt={outcome.get('prompt')} cache={outcome.get('cache')} "
              f"reuse={outcome.get('reuse')} prefill={outcome.get('prefill')}tok/s "
              f"ttft={outcome.get('ttft_ms')}ms wall={wall:.1f}s", flush=True)

    if args.mode == "resume":
        missed = [i for i, row in enumerate(rows)
                  if row.get("reuse") != "append_frontier"
                  or row["prompt"] - row.get("cache", 0) > 2]
        if missed:
            print(f"RESUME-FAIL sessions={missed} {[rows[i] for i in missed]}", flush=True)
            sys.exit(1)
        print(f"RESUME-OK sessions={len(rows)} prefill_free=yes", flush=True)


main()
