# NInfer API service and live dashboard (2 × V100-SXM2-16GB)

Qwen3.8-27B NVFP4, TP2 + MTP3 (`--draft-tokens 3 --lm-head-draft`), INT8-G64 KV, **190 000-token
context at `--max-concurrency 2`**. The 190 000-token pool is shared: admission charges each
request its prompt plus its declared `max_tokens`, so two ~63K agent sessions run together (Claude
Code declares 32 000) while one 190 000-token request still fits alone. The lighter `gguf-blocks`
artifact is available and reaches a 440 000-token pool, but it gives up committing throughput on
this route -- see [Artifact choice](#artifact-choice). **Switching the artifact requires clearing
the tier** (`./stop.sh && rm -f tier/*`): records are keyed without the weights, so one
quantization's KV would otherwise be resumed by the next.

Every KV cache -- the retained device lane, the parked host image and the NVMe records -- has a
**4 h idle timeout** (`--kv-cache-ttl-seconds 14400`), so a conversation nobody comes back to
releases its memory and disk. A prefix the session displaces is parked in 2 GiB of pinned host RAM
and then on NVMe (400 GiB sparse bound), so a long conversation -- and a restarted server --
resumes it without recomputing it. The tier file is sparse but keeps the blocks of everything it
has held: an hourly reclaim pass hands the blocks of released records back to the filesystem (no
restart needed), and `ninfer_disk_tier_reclaimed_bytes_total` reports the total it has returned.

Everything here is run from the deployment directory `~/v100/ninfer-api`, whose scripts and monitoring
config are symlinks to this directory; its runtime state (`tier/`, `logs/`, the Prometheus and Grafana
data) stays there.

| | |
|---|---|
| Endpoint | `http://192.168.1.211:8080/v1` on every interface (OpenAI-compatible: `/v1/chat/completions`, `/v1/messages`, `/v1/models`, `/v1/responses`); `127.0.0.1` works locally too |
| Metrics | `http://127.0.0.1:8080/metrics` (Prometheus text; unauthenticated, never blocked by a request) |
| Model id | `qwen3.8-27b-nvfp4` |
| Artifact | `/home/luyzh/models/qwen3_8_27b_nvfp4.ninfer`; the lighter `gguf-blocks` profile is `/home/luyzh/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` |
| Start / stop | `./serve.sh` (foreground) / `./stop.sh`; background: `setsid nohup ./serve.sh > logs/serve.log 2>&1 &` |
| Dashboard | `http://<host>:3000` -> folder **NInfer** -> *NInfer (2xV100, 200k single session)*; anonymous viewer, or `admin` / `ninfer` |
| Prometheus | `http://127.0.0.1:9090` (scrape every 2 s) |
| Reset the tier | `./stop.sh && rm -f tier/*` (the tier is the only copy of a parked prefix, so deleting it only costs a re-prefill) |

Overrides: `NINFER_API_PORT`, `NINFER_API_ARTIFACT`, `NINFER_API_TIER_DIR`, `NINFER_API_DISK_MIB`,
`NINFER_API_ACCESS_DIR`. `serve.sh` admits two sessions at a time (`--max-concurrency 2`); extra
requests queue.

## Artifact choice

Both artifacts are `qwen3.8-27b` and both are already on this host. The choice was measured on this
deployment, not argued from the model cards, and it lands on NVFP4 for the opposite reason the
memory numbers suggest.

**What each one costs in memory** (TP2, per GPU, from the startup record):

| | NVFP4 | gguf-blocks (GSQ-RCO IQ3_S v3) |
|---|---:|---:|
| weights | 10.46 GiB | ~6.4 GiB |
| free after weights | 5.00 GiB | 9.46 GiB |
| pool at C=2 | 190 000 tok (2969 pages) | 440 000 tok (6875 pages) |
| slack at that shape | 81.4 MiB | 65.0 MiB |

**What each one does with a batch.** A decode round's cost and what it commits, measured here with
two counters sampled 120 s apart while every admitted lane was decoding:

| artifact | lanes | round | rows/round | committed/round | **aggregate** | per request |
|---|---:|---:|---:|---:|---:|---:|
| gguf-blocks | 1 (doc, matched prompts) | 27.6 ms | 1.00 | 2.9 | 105 tok/s | 105 |
| gguf-blocks | 2 (measured) | 50.6 ms | 2.00 | 5.7 | **112.6 tok/s** | 56.3 |
| gguf-blocks | 4 (measured) | 89.6 ms | 3.99 | 10.8 | **120.1 tok/s** | 30.1 |
| NVFP4 | 1 (measured) | — | 1.00 | — | ~85 tok/s | 85 |
| NVFP4 | 2 (measured) | — | 2.00 | — | **~172 tok/s** | 86 |

The gguf-blocks round costs `~4.6 ms + 23 ms x rows`: each extra row pays for its four MTP verify
positions again, because the GGUF/dp4a projections on this card re-stream the weights per position
instead of amortizing them across rows. Committed tokens grow linearly with rows, the round grows
almost as fast, and aggregate decode saturates near 120 tok/s no matter how many lanes are admitted.
The NVFP4 route is tensor-core and does amortize, so its second lane nearly doubles throughput.

**The same shape on a 5090** (`docs/performance.md`, concurrent decode-saturation campaign, 27B):
`groupwise-int` scales 1.00x / 1.33x / 1.67x / 2.88x at C = 1 / 2 / 4 / 8, against 1.00x / 1.97x /
3.46x / 5.67x for `nvfp4`. Our V100 measurement of NVFP4 (85 -> 172 from one lane to two, 2.02x)
matches that campaign's 1.97x, so this is the route's property rather than the card's.

**Verdict.** NVFP4, because committing tokens faster is what this deployment's agents wait on, and
it is what the extra lane buys: the lighter artifact would spend its freed 4 GiB on lanes
re-streaming the weights and on a pool the single-owner prefill cannot fill any faster. It remains
the right shape for a workload that is long-context-bound rather than decode-bound -- it prefills
55-65% faster (measured 950 -> 1630 tok/s on the same prompts) and its 440K pool holds a 240K
request plus another 205K -- and `serve.sh` documents the switch, including the tier clear it
requires.

## Dashboard

```bash
cd ~/v100/ninfer-api/monitoring && docker compose up -d     # Prometheus + Grafana
docker compose down                                          # stop them
```

The stack is three containers on the host network: Prometheus scrapes `127.0.0.1:8080/metrics`
directly, Grafana binds the host's 3000, and `grafana-image-renderer` serves Grafana's
`GET /render/d/ninfer/...png` so a dashboard snapshot can be captured without a browser
(`curl -u admin:ninfer 'http://127.0.0.1:3000/render/d/ninfer/ninfer-2xv100-200k-single-session?width=1600&height=2400&from=now-6h&to=now&kiosk' -o dash.png`).
From another machine, either open `http://<this-host>:3000` or forward it:
`ssh -L 3000:127.0.0.1:3000 luyzh@<this-host>`. The compose file honours `NINFER_REPO_DIR` (config,
defaults to this repository) and `NINFER_API_DIR` (runtime state, defaults to `~/v100/ninfer-api`).

The provisioned dashboard (15 panels, 2 s refresh) shows:

| Panel group | Metrics |
|---|---|
| Speed | `rate(ninfer_prefill_tokens_total)`, `rate(ninfer_decode_tokens_total)`, average decode batch (`rows/rounds`) |
| MTP | acceptance rate `accepted/drafted`, average accepted length `accepted/rounds`, both for the whole session and for the request running now, plus accepted-per-draft-position |
| KV cache | main and MTP KV page groups by state (`capacity` / `entitled` / `free` / `shared_held`), GDN state slots, host and NVMe tier bytes |
| Shared prefix index | indexed blocks, lane promises, lane-free payloads, pages held, hit rate |
| Requests | `ninfer_requests{state="running|prefilling|decode_ready|waiting"}`, context ceiling, KV bytes/token |

Panels default to a six-hour window, so a burst stays visible after it ends; the rate windows are
five minutes (thirty for the session acceptance) and the stat panels reduce over the whole window.
`MTP accepted per draft position` only fills while a request is running -- it is live-only. Empty
speed panels mean an idle server, not a broken dashboard.

`docs/serving.md` documents the endpoint itself; the scrape path reads published counters only, so
scraping cannot stall inference.

## Reachability (intranet)

`serve.sh` binds the API on **every interface** (`NINFER_API_HOST`, default `0.0.0.0`), Grafana
publishes `3000` and Prometheus `9090` on every interface, and the image renderer stays on
`127.0.0.1:8081` because it is Grafana's helper, not a service. The API is **unauthenticated**
unless `NINFER_API_KEY` is set (clients then send the same value as their bearer token), so anyone
who can reach the port can run the model. Keep it behind the host firewall
(`sudo ufw allow from <lan-cidr> to any port 8080 proto tcp`, the same for 3000 and 9090 for the
dashboard), or set `NINFER_API_HOST=127.0.0.1` to keep the API local. Verified through the host's
INPUT path -- a bridge-networked container reaching `http://192.168.1.211:8080/v1/models` gets 200,
as do Grafana and Prometheus, while the renderer stays unreachable off-host.

`clients/` carries both agent configurations, verified against this deployment:

| File | Use |
|---|---|
| `codex-ninfer.config.toml` | copy to `~/.codex/ninfer.config.toml`, then `codex --profile ninfer` (Codex 0.162 reads profiles from their own file; nothing else in your `config.toml` changes) |
| `claude-env.sh` | `source` it before `claude`, or put the same pairs in the `env` object of `~/.claude/settings.json` |

## Access capture and datasets

`serve.sh` passes `--access-log-dir ~/v100/ninfer-api/access` (override with
`NINFER_API_ACCESS_DIR`, or set it empty to serve without capturing), so every HTTP request is
stored verbatim: the client's body, the server's body (JSON, or the SSE text of a streaming
response), the peer, the status and the wall time. The active file rotates to `access-<n>.jsonl`
past 256 MiB. `docs/serving.md` documents the record.

Prometheus scrapes `/metrics` every two seconds and those bodies are captured too, which is most of
the capture's volume (~100 MB/day) and none of its value. Raise the scrape interval in
`monitoring/prometheus.yml` if the capture matters more than the dashboard's resolution.

Turn a capture into a dataset -- one conversation-per-access file, a deduplicated training file, and
a stats summary:

```bash
python3 tools/v100/ninfer-api/build_dataset.py ~/v100/ninfer-api/access \
  --out ~/v100/ninfer-api/dataset
```

It normalizes all three wires (OpenAI chat, Anthropic Messages, OpenAI Responses) onto OpenAI-style
messages, reconstructs a streaming response from its frames, groups turns into conversations, and
keeps anything it could not read with a warning rather than dropping it silently. The capture is
raw by design, so `chat.jsonl` is the archive view and `sft.jsonl` is the filtered, deduplicated
training view.

## Agent clients and the compatibility shim

Codex (Responses wire) and Claude Code (Anthropic Messages wire) run against this server unchanged;
`docs/serving.md` has both configurations. The engine validates both request bodies field by field,
which refuses a field a newer agent build starts sending -- that is how Codex 0.162 needed
`client_metadata`/`prompt_cache_key`/`include`/non-function tools/`reasoning.summary` and Claude Code
2.1 needed its `output_config.effort = high` translated. The engine now tolerates that set natively;
`compat-shim.py` is the forward-compatibility fallback for the next such field:

```bash
python3 compat-shim.py --port 8098 --upstream 127.0.0.1:8080   # point the client at :8098
```

It forwards bodies and streaming responses unchanged after dropping the fields the engine does not
act on, skipping tools it cannot execute and translating the effort vocabularies, logging every
change.

## opencode

`~/.config/opencode/opencode.json` declares the provider and makes it the default model:

```json
{ "provider": { "ninfer": { "npm": "@ai-sdk/openai-compatible",
    "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local" },
    "models": { "qwen3.8-27b-nvfp4": { "name": "Qwen3.8 27B NVFP4 (200k, local)" } } } },
  "model": "ninfer/qwen3.8-27b-nvfp4" }
```

```bash
cd <project> && opencode run "explain this repo"
opencode run --model ninfer/qwen3.8-27b-nvfp4 "list the TODOs in src/"
opencode            # interactive TUI
```

Verified through opencode: `glob`/`read`/`bash`/`write` tool calls round-trip
(`finish=tool_calls`, `tools=10..11` per request), a growing session reuses its prefix
(`cache=7504..7851` of `prompt=7760..7871`, `append_frontier` / `restore_turn_checkpoint`, ttft
82-429 ms against a 10 s cold prefill), and single-stream decode runs at 90-127 tok/s.
