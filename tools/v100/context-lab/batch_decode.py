#!/usr/bin/env python3
"""Concurrent decode-batch measurement on the retained-context pool.

For every shape ``N x LENGTH`` (N concurrent streams, each LENGTH prompt tokens) the driver

1. builds N distinct prompts calibrated to LENGTH tokens against ``/v1/messages/count_tokens``,
2. warms them -- all N are sent once so every lane already holds its prefix when the timed
   window opens,
3. times one round of N *simultaneous* requests with ``--max-tokens`` decode tokens and proves
   from the schema-v10 request log that all N resumed without re-prefill
   (``prefix_reuse_path == append_frontier`` and ``prompt - prefix_cache_hit_tokens <= 2``).

Reported per shape: every stream's own decode rate, the aggregate decode rate of the window,
and -- with ``--baseline`` -- the same numbers for one stream at the same prompt length, which is
the batch uplift.  The server must run tools/v100/context-lab/serve-batch-decode.sh with
``CONTEXT_LAB_REQUEST_LOG`` pointing at ``--request-log``.

    tools/v100/context-lab/batch_decode.py 8x10000 8x20000 6x30000 4x40000 2x80000 \
        --request-log logs/requests.jsonl --out results/batch_decode.json --baseline
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL_ID = "qwen3.8-27b-nvfp4"
MAX_CONCURRENCY = 8


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
        started = time.time()
        body = self.post(
            "/v1/chat/completions",
            {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": text}],
            },
        )
        return time.time() - started, body


def body_text(tag, lines):
    out = [f"# session {tag} working notes"]
    for i in range(lines):
        out.append(
            f"def {tag}_{i}(value, weight):\n    total = value * {i} + weight\n"
            f"    if total % 7 == {i % 7}:\n        return total - {i}\n    return total + {i + 1}\n"
        )
    return "\n".join(out)


class RequestLog:
    """Reader for the schema-v10 JSONL the server appends under --request-log-jsonl."""

    def __init__(self, path):
        self.path = Path(path)

    def offset(self):
        return self.path.stat().st_size if self.path.exists() else 0

    def records_after(self, offset):
        try:
            with self.path.open("rb") as handle:
                handle.seek(offset)
                raw = handle.read().decode(errors="replace")
        except FileNotFoundError:
            return []
        records = []
        for line in raw.splitlines():
            if not line.startswith("{"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially flushed trailing line
        return records

    def wait_for_done(self, offset, count, timeout):
        deadline = time.time() + timeout
        while True:
            done = [r for r in self.records_after(offset) if r.get("event") == "request_done"]
            if len(done) >= count:
                return done
            if time.time() >= deadline:
                raise SystemExit(f"only {len(done)}/{count} request_done records after "
                                 f"{timeout:.0f}s")
            time.sleep(0.5)


def send_all(client, prompts, max_tokens):
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        return list(pool.map(lambda prompt: client.chat(prompt, max_tokens), prompts))


def calibrate(client, tag, target):
    """Line count whose prompt reaches ``target`` tokens for ``tag`` (affine, two points)."""
    probe_lines = 64
    per_line = client.count_tokens(body_text(tag, probe_lines)) / probe_lines
    lines = int(round(target / per_line))
    measured = client.count_tokens(body_text(tag, lines))
    return int(round(lines * target / measured))


def summarize_done(records, window_seconds, throughput):
    """Turn request_done records into the per-shape numbers."""
    rows = []
    for record in records:
        result = record["result"]
        timings = record["timings_seconds"]
        decode_tokens = max(0, result["completion_tokens"] - 1)
        rows.append({
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "decode_tokens": decode_tokens,
            "prefix_cache_hit_tokens": result["prefix_cache_hit_tokens"],
            "computed_prefill_tokens": result["computed_prefill_tokens"],
            "prefix_reuse_path": result["prefix_reuse_path"],
            "decode_seconds": timings["decode"],
            "ttft_seconds": timings["ttft"],
            "decode_tok_s": decode_tokens / timings["decode"] if timings["decode"] > 0 else 0.0,
        })
    rows.sort(key=lambda row: row["decode_tok_s"])
    total_decode_tokens = sum(row["decode_tokens"] for row in rows)
    max_decode_seconds = max((row["decode_seconds"] for row in rows), default=0.0)
    return {
        "streams": len(rows),
        "prompt_tokens": [row["prompt_tokens"] for row in rows],
        "computed_prefill_tokens": [row["computed_prefill_tokens"] for row in rows],
        "prefix_cache_hit_tokens": [row["prefix_cache_hit_tokens"] for row in rows],
        "prefix_reuse_paths": sorted({row["prefix_reuse_path"] for row in rows}),
        "per_stream_decode_tok_s": [round(row["decode_tok_s"], 2) for row in rows],
        "per_stream_decode_tok_s_mean": statistics.fmean(
            row["decode_tok_s"] for row in rows) if rows else 0.0,
        "total_decode_tokens": total_decode_tokens,
        "decode_window_seconds": window_seconds,
        "max_stream_decode_seconds": max_decode_seconds,
        "aggregate_decode_tok_s_wall": total_decode_tokens / window_seconds
        if window_seconds > 0 else 0.0,
        "aggregate_decode_tok_s_lockstep": total_decode_tokens / max_decode_seconds
        if max_decode_seconds > 0 else 0.0,
        "throughput_samples": throughput,
    }


def run_shape(client, log, tag, streams, target, args, warm_tokens):
    lines = calibrate(client, f"{tag}0", target)
    prompts = [body_text(f"{tag}{stream}", lines) for stream in range(streams)]
    counted = [client.count_tokens(prompt) for prompt in prompts]
    if max(counted) - min(counted) > 1:
        raise SystemExit(f"prompt token counts diverge: {counted}")
    print(f"[{tag}] lines={lines} prompt_tokens={counted[0]} streams={streams} warming...",
          flush=True)

    mark = log.offset()
    warmed = send_all(client, prompts, warm_tokens)
    log.wait_for_done(mark, streams, args.timeout_seconds)

    rounds = []
    for repeat in range(args.repeat):
        mark = log.offset()
        started = time.time()
        responses = send_all(client, prompts, args.max_tokens)
        window = time.time() - started
        done = log.wait_for_done(mark, streams, args.timeout_seconds)
        all_records = log.records_after(mark)
        end_ms = int(time.time() * 1000)
        throughput = [
            {"average_size": record["decode_batch"]["average_size"],
             "decode_tok_s": record["throughput_tokens_per_second"]["decode"],
             "interval_seconds": record["interval_seconds"]}
            for record in all_records if record.get("event") == "throughput"
        ]
        summary = summarize_done(done, window, throughput)
        summary["repeat"] = repeat
        summary["prompt_wall_seconds"] = [
            round(elapsed, 2) for elapsed, _ in warmed] if repeat == 0 else None
        summary["client_window_seconds"] = window
        rounds.append(summary)
        print(f"[{tag}] round{repeat} streams={summary['streams']} "
              f"prompt={summary['prompt_tokens'][0]} "
              f"cache={summary['prefix_cache_hit_tokens']} "
              f"reuse={summary['prefix_reuse_paths']} "
              f"prefill_tokens={max(summary['computed_prefill_tokens'])} "
              f"per_stream={[round(v) for v in summary['per_stream_decode_tok_s']]}tok/s "
              f"mean={summary['per_stream_decode_tok_s_mean']:.1f}tok/s "
              f"aggregate_wall={summary['aggregate_decode_tok_s_wall']:.1f}tok/s "
              f"window={window:.2f}s", flush=True)
    return {"tag": tag, "streams": streams, "target_tokens": target, "prompt_tokens": counted[0],
            "lines": lines, "rounds": rounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("shapes", nargs="+", metavar="NxLENGTH",
                        help="concurrent stream count x prompt tokens, e.g. 8x10000")
    parser.add_argument("--request-log", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warm-tokens", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--baseline", action="store_true",
                        help="also time one stream at each prompt length")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()

    shapes = []
    for spec in args.shapes:
        match = re.fullmatch(r"(\d+)[xX](\d+)", spec)
        if not match:
            raise SystemExit(f"bad shape: {spec}")
        streams, target = int(match.group(1)), int(match.group(2))
        if not 1 <= streams <= MAX_CONCURRENCY:
            raise SystemExit(f"shape {spec} exceeds the 8-lane product ceiling")
        shapes.append((streams, target))

    client = Client(args.base_url, args.model_id, args.timeout_seconds)
    log = RequestLog(args.request_log)
    results = {"label": args.label, "max_tokens": args.max_tokens,
               "warm_tokens": args.warm_tokens, "repeat": args.repeat, "shapes": []}
    for index, (streams, target) in enumerate(shapes):
        shape = run_shape(client, log, f"s{index}x{streams}", streams, target, args,
                          args.warm_tokens)
        if args.baseline:
            shape["baseline"] = run_shape(client, log, f"b{index}", 1, target, args,
                                          args.warm_tokens)
        results["shapes"].append(shape)

    print("\nshape      streams  per-stream tok/s         aggregate tok/s   uplift")
    for shape in results["shapes"]:
        for round_summary in shape["rounds"]:
            mean = round_summary["per_stream_decode_tok_s_mean"]
            total = round_summary["aggregate_decode_tok_s_wall"]
            baseline = None
            if "baseline" in shape:
                baseline = shape["baseline"]["rounds"][0]["aggregate_decode_tok_s_wall"]
            uplift = f"{total / baseline:5.2f}x" if baseline else "     "
            print(f"{shape['tag']:10s} {round_summary['streams']:7d} "
                  f"{mean:7.1f} ({min(round_summary['per_stream_decode_tok_s']):.0f}-"
                  f"{max(round_summary['per_stream_decode_tok_s']):.0f})        "
                  f"{total:7.1f}        {uplift}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")


main()
