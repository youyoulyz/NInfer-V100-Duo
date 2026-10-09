#!/usr/bin/env python3
"""End-to-end accuracy gate for the V100 duo profile.

Serves a real `.ninfer` artifact through `ninfer-serve` and checks, with no Python and no reference
implementation in the loop, the end-to-end contracts this profile is responsible for:

  * twelve fixed greedy tasks answer semantically correctly and match the recorded golden answer
    byte-for-byte when this artifact has one;
  * generation is deterministic: repeating a request returns the identical answer;
  * a conversation continuation reuses the retained turn checkpoint instead of recomputing the
    full prefill (`prefix_cache_hit_tokens > 0`, non-`full_reset` path), while the cold first
    request reports a full reset.

The recorded goldens live in `tests/fixtures/v100/accuracy_golden.json`, keyed by artifact file
name. Record or refresh them with `--record`; a recorded answer is a regression baseline, so a
mismatch is a behaviour change that has to be justified before re-recording.

Exit status: 0 pass, 1 check failure, 2 setup or usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SERVE = REPO / "build-v100-duo" / "apps" / "ninfer-serve"
DEFAULT_RUNTIME_LIB_DIR = REPO / "build" / "_deps" / "install" / "lib"
DEFAULT_CUDA_LIB_DIR = Path("/usr/local/cuda-12.8/lib64")
DEFAULT_GOLDEN = REPO / "tests" / "fixtures" / "v100" / "accuracy_golden.json"
GOLDEN_VERSION = 1

# Fixed greedy task set. `accepts` is the semantic acceptance list; the recorded golden then pins
# one exact answer per task. Keep every entry unambiguous and wording-stable: these are capability
# probes, not a benchmark corpus.
TASKS: list[tuple[str, str, tuple[str, ...]]] = [
    ("mul", "What is 17 times 23? Reply with only the number.", ("391",)),
    ("capital", "What is the capital city of Australia? Reply with only the city name.",
     ("canberra",)),
    ("reverse", 'Reverse the string "abcdef". Reply with only the reversed string.', ("fedcba",)),
    ("count_r", 'How many times does the letter "r" appear in the word "strawberry"? '
                "Reply with only the number.", ("3",)),
    ("sum10", "What is the sum of the integers from 1 to 10? Reply with only the number.", ("55",)),
    ("french", 'Translate "Good morning" into French. Reply with only the translation.',
     ("bonjour",)),
    ("len", "In Python, what does len([1, 2, 3]) evaluate to? Reply with only the number.", ("3",)),
    ("power", "Which is larger, 9 squared or 2 to the power of 9? "
              "Reply with only the larger expression.",
     ("2^9", "2**9", "2 to the power of 9", "two to the power of 9", "512")),
    ("gold", "What is the chemical symbol for gold? Reply with only the symbol.", ("au",)),
    ("percent", "What is 15 percent of 200? Reply with only the number.", ("30",)),
    ("anagram", 'Are the words "listen" and "silent" anagrams of each other? '
                "Reply with only yes or no.", ("yes",)),
    ("german", 'What language is the phrase "Danke schoen" written in? '
               "Reply with only the language name.", ("german", "deutsch")),
]

# The continuation pair. `prefix_reuse` resumes a retained turn/response checkpoint, so the second
# turn of an exchanged conversation is what exercises it; an exact repeat of a cold single-turn
# prompt is a full reset by design and is checked as such.
FOLLOW_UP_TASK = "followup"
FOLLOW_UP_PROMPT = "Now do the same for France."
FOLLOW_UP_ACCEPTS = ("paris",)
TURN_BASE_TASK = "capital"


def load_golden(path: Path) -> dict:
    if not path.exists():
        return {"version": GOLDEN_VERSION, "artifacts": {}}
    with path.open() as handle:
        return json.load(handle)


def save_golden(path: Path, golden: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(golden, handle, indent=2, sort_keys=True)
        handle.write("\n")


def runtime_library_path(runtime_lib_dir: Path) -> str:
    existing = [str(part) for part in (runtime_lib_dir, DEFAULT_CUDA_LIB_DIR) if part.is_dir()]
    inherited = os.environ.get("LD_LIBRARY_PATH", "")
    if inherited:
        existing.append(inherited)
    return ":".join(existing)


def start_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    command = [
        str(args.serve), str(args.artifact),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--model-id", "accuracy-probe",
        "--tp", str(args.tp), "--devices", args.devices,
        "--kv-dtype", args.kv_dtype,
        "--max-context", str(args.max_context),
        "--prefill-chunk", str(args.prefill_chunk),
        "--max-concurrency", "1",
        "--no-thinking", "--greedy",
        "--request-log-jsonl", str(log_path),
    ]
    if args.spec != "off":
        command += ["--spec", args.spec, "--draft-tokens", str(args.draft_tokens),
                    "--lm-head-draft"]
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = runtime_library_path(args.runtime_lib_dir)
    log_path.write_bytes(b"")
    return subprocess.Popen(
        command, env=environment,
        stdout=open(f"{log_path}.stdout", "wb"), stderr=subprocess.STDOUT,
    )


def wait_ready(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as reply:
                if reply.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def ask(port: int, messages: list[dict], max_tokens: int, timeout: float) -> tuple[str, float, dict]:
    body = json.dumps({"model": "accuracy-probe", "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        payload = json.load(reply)
    text = (payload["choices"][0]["message"].get("content") or "").strip()
    return text, time.time() - started, payload.get("usage", {})


def read_requests(log_path: Path) -> list[dict]:
    records = []
    if not log_path.exists():
        return records
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "request_done":
            continue
        result = record["result"]
        records.append({
            "prefix_reuse_path": result["prefix_reuse_path"],
            "prefix_cache_hit_tokens": result["prefix_cache_hit_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "computed_prefill_tokens": result["computed_prefill_tokens"],
        })
    return records


def shutdown(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def accepted(text: str, accepts: tuple[str, ...]) -> bool:
    return any(candidate.lower() in text.lower() for candidate in accepts)


def run_artifact(args: argparse.Namespace, golden: dict) -> dict:
    key = args.artifact.name
    log_path = Path(args.log_dir) / f"accuracy_{key}.jsonl"
    process = start_server(args, log_path)
    rows: list[dict] = []
    failures: list[str] = []
    determinism: dict[str, bool | None] = {"cold_repeat": None, "turn_repeat": None}
    follow_up_answer: str | None = None

    def record(task: str, accepts: tuple[str, ...], messages: list[dict]) -> dict:
        nonlocal follow_up_answer
        try:
            text, seconds, usage = ask(args.port, messages, args.max_tokens, args.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            row = {"task": task, "ok": False, "answer": "", "error": str(error)}
            rows.append(row)
            failures.append(f"{task}: request failed: {error}")
            return row
        ok = accepted(text, accepts)
        row = {"task": task, "ok": ok, "answer": text, "seconds": round(seconds, 2),
               "usage": usage}
        rows.append(row)
        print(f"  [{'ok' if ok else 'FAIL'}] {task}: {text!r}", file=sys.stderr, flush=True)
        if not ok:
            failures.append(f"{task}: answer {text!r} matched none of {list(accepts)}")
        if task == FOLLOW_UP_TASK:
            follow_up_answer = text
        return row

    try:
        print(f"waiting for ninfer-serve on port {args.port} ({key})...",
              file=sys.stderr, flush=True)
        if not wait_ready(args.port, args.startup_timeout):
            return {"artifact": key, "error": "server did not become ready",
                    "failures": ["server did not become ready"]}

        single_turn = {name: ([{"role": "user", "content": prompt}], accepts)
                       for name, prompt, accepts in TASKS}
        for name, _prompt, accepts in TASKS:
            record(name, accepts, single_turn[name][0])

        first_row = rows[0]
        repeat = record("cold_repeat", TASKS[0][2], single_turn[TASKS[0][0]][0])
        determinism["cold_repeat"] = repeat["answer"] == first_row["answer"]
        if not determinism["cold_repeat"]:
            failures.append(f"determinism: repeated {TASKS[0][0]} returned {repeat['answer']!r} "
                            f"instead of {first_row['answer']!r}")

        # Reuse is decided against the retained state of the immediately preceding request on the
        # lane, so the base turn is re-issued right before its continuation.
        base_prompt = next(prompt for name, prompt, _ in TASKS if name == TURN_BASE_TASK)
        base = record("turn_base", next(accepts for name, _, accepts in TASKS
                                        if name == TURN_BASE_TASK),
                      [{"role": "user", "content": base_prompt}])
        continuation = [
            {"role": "user", "content": base_prompt},
            {"role": "assistant", "content": base["answer"]},
            {"role": "user", "content": FOLLOW_UP_PROMPT},
        ]
        record(FOLLOW_UP_TASK, FOLLOW_UP_ACCEPTS, continuation)
        turn_repeat = record("turn_repeat", FOLLOW_UP_ACCEPTS, continuation)
        determinism["turn_repeat"] = follow_up_answer is not None and \
            turn_repeat["answer"] == follow_up_answer
        if not determinism["turn_repeat"]:
            failures.append(f"determinism: repeated {FOLLOW_UP_TASK} returned "
                            f"{turn_repeat['answer']!r} instead of {follow_up_answer!r}")
    finally:
        shutdown(process)

    requests = read_requests(log_path)
    expected_requests = len(TASKS) + 4
    continuation_path: str | None = None
    if not args.no_prefix_reuse_check:
        if len(requests) != expected_requests:
            failures.append(f"prefix reuse: request log holds {len(requests)} completed requests, "
                            f"expected {expected_requests}")
        else:
            cold, repeated, continuation = (requests[0], requests[len(TASKS)],
                                            requests[len(TASKS) + 2])
            if (cold["prefix_cache_hit_tokens"], cold["prefix_reuse_path"]) != (0, "full_reset"):
                failures.append("prefix reuse: the cold first request reused "
                                f"{cold['prefix_cache_hit_tokens']} tokens via "
                                f"{cold['prefix_reuse_path']}")
            if (repeated["prefix_cache_hit_tokens"],
                    repeated["prefix_reuse_path"]) != (0, "full_reset"):
                failures.append("prefix reuse: an exact repeat of a single-turn prompt is expected "
                                "to be a full reset, but it reported "
                                f"{repeated['prefix_cache_hit_tokens']} tokens via "
                                f"{repeated['prefix_reuse_path']}")
            if (continuation["prefix_cache_hit_tokens"] <= 0 or
                    continuation["prefix_reuse_path"] == "full_reset"):
                failures.append("prefix reuse: the continuation recomputed the full prefill "
                                f"(path={continuation['prefix_reuse_path']}, "
                                f"hit_tokens={continuation['prefix_cache_hit_tokens']})")
            continuation_path = continuation["prefix_reuse_path"]

    recorded = golden["artifacts"].get(key, {})
    recorded_path = recorded.get("continuation_reuse_path")
    if (continuation_path is not None and recorded_path is not None and
            continuation_path != recorded_path):
        failures.append(f"prefix reuse: the continuation reported {continuation_path!r} where the "
                        f"recorded golden reports {recorded_path!r}")

    golden_answers = recorded.get("answers")
    if golden_answers is not None:
        for row in rows:
            expected = golden_answers.get(row["task"])
            if expected is not None and row["answer"] != expected:
                failures.append(f"{row['task']}: answer {row['answer']!r} differs from the "
                                f"recorded golden {expected!r}")

    return {
        "artifact": key,
        "identity": golden["artifacts"].get(key, {}).get("identity"),
        "correct": sum(1 for row in rows if row["ok"]),
        "total": len(rows),
        "determinism": determinism,
        "rows": rows,
        "requests": requests,
        "failures": failures,
        "continuation_reuse_path": continuation_path,
        "golden_compared": golden_answers is not None,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True, type=Path,
                        help="path to a .ninfer artifact to serve")
    parser.add_argument("--serve", type=Path, default=DEFAULT_SERVE,
                        help=f"ninfer-serve binary (default {DEFAULT_SERVE})")
    parser.add_argument("--runtime-lib-dir", type=Path, default=DEFAULT_RUNTIME_LIB_DIR,
                        help=f"runtime shared-library directory (default {DEFAULT_RUNTIME_LIB_DIR})")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--record", action="store_true",
                        help="record or refresh this artifact's golden answers instead of "
                             "comparing against them")
    parser.add_argument("--identity", default=None,
                        help="artifact identity to record with --record")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--kv-dtype", default="int8", choices=("bf16", "int8"))
    parser.add_argument("--spec", default="mtp", choices=("mtp", "dflash", "off"),
                        help="speculative backend; the production duo profile uses mtp "
                             "(tp2 prefix reuse requires --spec mtp)")
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--log-dir", default="/tmp")
    parser.add_argument("--no-prefix-reuse-check", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the full record as JSON")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.serve.is_file():
        print(f"error: ninfer-serve is missing: {args.serve}", file=sys.stderr)
        return 2
    if not args.artifact.is_file():
        print(f"error: artifact is missing: {args.artifact}", file=sys.stderr)
        return 2

    golden = load_golden(args.golden)
    golden.setdefault("version", GOLDEN_VERSION)
    golden.setdefault("artifacts", {})
    result = run_artifact(args, golden)

    if result.get("error"):
        print(f"FAIL {result['artifact']}: {result['error']}", file=sys.stderr)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2

    if args.record:
        if result["failures"]:
            print(f"refusing to record {result['artifact']}: the run already failed",
                  file=sys.stderr)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 1
        entry = golden["artifacts"].setdefault(result["artifact"], {})
        if args.identity:
            entry["identity"] = args.identity
        answers = {row["task"]: row["answer"] for row in result["rows"]
                   if row["task"] in {name for name, _, _ in TASKS} | {FOLLOW_UP_TASK}}
        if answers:
            entry["answers"] = answers
        entry["continuation_reuse_path"] = result["continuation_reuse_path"]
        save_golden(args.golden, golden)
        print(f"recorded {result['artifact']} ({result['correct']}/{result['total']}) "
              f"into {args.golden}")

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for row in result["rows"]:
            print(f"  {'ok  ' if row['ok'] else 'FAIL'} {row['task']:<11} {row['answer']!r}")
        source = "the recorded golden" if result["golden_compared"] else "accept lists only"
        print(f"{result['artifact']}: {result['correct']}/{result['total']} answers, "
              f"determinism={result['determinism']}, compared against {source}")

    if result["failures"]:
        for failure in result["failures"]:
            print(f"FAIL {result['artifact']}: {failure}", file=sys.stderr)
        return 1
    print(f"PASS {result['artifact']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
