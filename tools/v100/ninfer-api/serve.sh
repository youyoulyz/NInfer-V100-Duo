#!/usr/bin/env bash
# NInfer API service: two concurrent sessions sharing a 190K-token KV pool, backed by the pinned
# host tier and the local NVMe tier.
#
# The artifact is `qwen3.8-27b/nvfp4`. Its 10.46 GiB of weights per GPU leave 5.00 GiB for
# everything else, which caps the pool at 190 000 tokens with two lanes -- but it is the artifact
# whose decode scales with the batch: measured here, 85 tok/s for one row against 172 for two, so the
# second lane doubles committing throughput, while the `gguf-blocks` (GSQ-RCO IQ3_S) artifact that
# fits a 440K pool stops at ~120 tok/s however many lanes are admitted (see the artifact-choice
# section of the README, and docs/performance.md for the same shape on a 5090).
#
# `--max-context 190000` is the per-sequence ceiling and `--kv-capacity 190000` the shared pool:
# admission charges `prompt + declared max_tokens` per request, so two ~63K agent sessions run
# together (Claude Code declares 32 000), or one request at the full 190K. Each lane costs 2 GDN
# state slots (146.8 MiB) plus its share of the MTP graph allowance (82 MiB), which is why 190K at
# two lanes is the shape that fits (200K at one lane leaves 86.5 MiB; adding lane 2 to it does not).
#
# Switching the artifact invalidates the NVMe tier: its records are keyed by token digest and an
# execution tag that does not name the weights, so a record written under one quantization would be
# resumed under the next. Clear it (stop.sh, then `rm -f tier/*`) whenever the artifact changes.
#
#   ~/v100/ninfer-api/serve.sh > logs/serve.log 2>&1 &      # start (background)
#   kill $(pgrep -x ninfer-serve)                            # stop
#
# Every KV cache -- the retained device lane, the parked host image and the NVMe records -- is
# released after 4 h without use (`--kv-cache-ttl-seconds 14400`), so a conversation nobody comes
# back to stops holding memory or disk.
#
# Overrides: NINFER_API_HOST (default 0.0.0.0), NINFER_API_PORT, NINFER_API_ARTIFACT,
# NINFER_API_TIER_DIR, NINFER_API_DISK_MIB, NINFER_API_KEY, NINFER_API_ACCESS_DIR.
#
# Every HTTP request is captured losslessly under NINFER_API_ACCESS_DIR (default
# ~/v100/ninfer-api/access): the client's body and the server's body verbatim, which is what
# `build_dataset.py` turns into a dataset. Set it empty to serve without capturing.
#
# The API is served on every interface because this deployment is meant to be reached from the
# intranet, and it is UNAUTHENTICATED unless NINFER_API_KEY is set: anyone who can reach the port can
# run the model. Set NINFER_API_HOST=127.0.0.1 to keep it local, or NINFER_API_KEY=VALUE to require a
# key (clients then send the same value as their bearer token). The port still has to be open in the
# host firewall (`sudo ufw allow from <lan-cidr> to any port 8080 proto tcp`).
#
# A prefix the session displaces is parked in host RAM and then on NVMe, so a long conversation --
# or a restarted server -- resumes it without recomputing it.
set -euo pipefail

repo_dir=/home/luyzh/v100/NInfer-V100-Duo
executable="${repo_dir}/build-v100-duo/apps/ninfer-serve"
artifact="${NINFER_API_ARTIFACT:-/home/luyzh/models/qwen3_8_27b_nvfp4.ninfer}"
tier_dir="${NINFER_API_TIER_DIR:-$HOME/v100/ninfer-api/tier}"
host="${NINFER_API_HOST:-0.0.0.0}"
port="${NINFER_API_PORT:-8080}"
disk_mib="${NINFER_API_DISK_MIB:-409600}"   # 400 GiB bound; the tier file is sparse
access_dir="${NINFER_API_ACCESS_DIR:-$HOME/v100/ninfer-api/access}"
access_args=()
if [[ -n "${access_dir}" ]]; then
    access_args+=(--access-log-dir "${access_dir}")
fi

if [[ ! -x "${executable}" ]]; then
    echo "ninfer-serve is missing: ${executable} (cmake --build ${repo_dir}/build-v100-duo -j)" >&2
    exit 1
fi
export LD_LIBRARY_PATH="${repo_dir}/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${tier_dir}"

exec "${executable}" "${artifact}" \
    --tp 2 --devices 0,1 --host "${host}" --port "${port}" \
    --model-id qwen3.8-27b-nvfp4 \
    --max-context 190000 --kv-capacity 190000 --kv-dtype int8 --prefill-chunk 1024 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --max-concurrency 2 --max-pending-requests 16 --pending-timeout-ms 3600000 \
    --host-kv-mib 2048 --disk-kv-path "${tier_dir}" --disk-kv-mib "${disk_mib}" \
    --kv-cache-ttl-seconds 14400 --log-stats-interval-ms 60000 \
    "${access_args[@]}" \
    ${NINFER_API_KEY:+--api-key "${NINFER_API_KEY}"}
