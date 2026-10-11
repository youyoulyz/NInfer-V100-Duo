# NInfer API service and live dashboard (2 × V100-SXM2-16GB)

Qwen3.8-27B NVFP4, TP2 + MTP3 (`--draft-tokens 3 --lm-head-draft`), INT8-G64 KV, **one session up to
200 000 tokens**. A prefix the session displaces is parked in 2 GiB of pinned host RAM and then on
NVMe (400 GiB sparse bound), so a long conversation -- and a restarted server -- resumes it without
recomputing it.

Everything here is run from the deployment directory `~/v100/ninfer-api`, whose scripts and monitoring
config are symlinks to this directory; its runtime state (`tier/`, `logs/`, the Prometheus and Grafana
data) stays there.

| | |
|---|---|
| Endpoint | `http://127.0.0.1:8080/v1` (OpenAI-compatible: `/v1/chat/completions`, `/v1/messages`, `/v1/models`) |
| Metrics | `http://127.0.0.1:8080/metrics` (Prometheus text; unauthenticated, never blocked by a request) |
| Model id | `qwen3.8-27b-nvfp4` |
| Artifact | `/home/luyzh/models/qwen3_8_27b_nvfp4.ninfer` |
| Start / stop | `./serve.sh` (foreground) / `./stop.sh`; background: `setsid nohup ./serve.sh > logs/serve.log 2>&1 &` |
| Dashboard | `http://<host>:3000` -> folder **NInfer** -> *NInfer (2xV100, 200k single session)*; anonymous viewer, or `admin` / `ninfer` |
| Prometheus | `http://127.0.0.1:9090` (scrape every 2 s) |
| Reset the tier | `./stop.sh && rm -f tier/*` (the tier is the only copy of a parked prefix, so deleting it only costs a re-prefill) |

Overrides: `NINFER_API_PORT`, `NINFER_API_ARTIFACT`, `NINFER_API_TIER_DIR`, `NINFER_API_DISK_MIB`.
`serve.sh` runs one session at a time (`--max-concurrency 1`); extra requests queue.

## Dashboard

```bash
cd ~/v100/ninfer-api/monitoring && docker compose up -d     # Prometheus + Grafana
docker compose down                                          # stop them
```

Both containers use the host network, so Prometheus scrapes `127.0.0.1:8080/metrics` directly and
Grafana binds the host's 3000. From another machine, either open `http://<this-host>:3000` or forward
it: `ssh -L 3000:127.0.0.1:3000 luyzh@<this-host>`.

The provisioned dashboard (15 panels, 2 s refresh) shows:

| Panel group | Metrics |
|---|---|
| Speed | `rate(ninfer_prefill_tokens_total)`, `rate(ninfer_decode_tokens_total)`, average decode batch (`rows/rounds`) |
| MTP | acceptance rate `accepted/drafted`, average accepted length `accepted/rounds`, both for the whole session and for the request running now, plus accepted-per-draft-position |
| KV cache | main and MTP KV page groups by state (`capacity` / `entitled` / `free` / `shared_held`), GDN state slots, host and NVMe tier bytes |
| Shared prefix index | indexed blocks, lane promises, lane-free payloads, pages held, hit rate |
| Requests | `ninfer_requests{state="running|prefilling|decode_ready|waiting"}`, context ceiling, KV bytes/token |

`docs/serving.md` documents the endpoint itself; the scrape path reads published counters only, so
scraping cannot stall inference.

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
