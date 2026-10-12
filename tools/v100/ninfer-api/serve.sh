#!/usr/bin/env bash
# NInfer API service: two concurrent sessions sharing a 440K-token KV pool, backed by the pinned
# host tier and the local NVMe tier.
#
# The artifact is `qwen3.8-27b/gguf-blocks` (GSQ-RCO IQ3_S v3). It is the same Qwen3.8-27B model as
# the NVFP4 artifact, quantized to 3.4 bits instead of the NVFP4 hybrid, and that is what buys the
# concurrency: 5.66 GiB of weights per GPU against 10.46 GiB, leaving ~10.2 GiB where the NVFP4
# profile had 5.0. Measured on this host (docs/performance.md): prefill is 55-65% faster, decode
# 10-15% slower, and the NVFP4 shape could not host lane 2 at 200K at all.
#
# `--max-context 240000` is the per-sequence ceiling and `--kv-capacity 440000` the shared pool, which
# is what the smaller weights leave room for: the pool costs 1.096 MiB per 64 tokens per device, so
# 440 000 tokens (6875 pages) is 7.36 GiB against 1.80 GiB of fixed C=2 state -- about 0.3 GiB of
# slack per device. Two requests cannot both sit at the full 240K (that needs 7500 pages); one 240K
# plus a second up to ~205K fits.
#
# Two lanes is a measured choice, not a conservative one. With four rows a decode round cost 89.6 ms
# and committed 10.8 tokens (120 tok/s) against 27 ms and 2.7 tokens at one row: the GGUF/dp4a route
# on this card does not overlap weight streaming with the per-position compute that MTP's 1+3 verify
# pays for, so batching four rows pays that compute four times for the same weight reads. Two rows
# keeps the streaming term amortized without it, which is why the leftover memory goes into the pool
# instead of into more lanes.
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
artifact="${NINFER_API_ARTIFACT:-/home/luyzh/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer}"
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
    --max-context 240000 --kv-capacity 440000 --kv-dtype int8 --prefill-chunk 1024 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --max-concurrency 2 --max-pending-requests 16 --pending-timeout-ms 3600000 \
    --host-kv-mib 2048 --disk-kv-path "${tier_dir}" --disk-kv-mib "${disk_mib}" \
    --kv-cache-ttl-seconds 14400 --log-stats-interval-ms 60000 \
    "${access_args[@]}" \
    ${NINFER_API_KEY:+--api-key "${NINFER_API_KEY}"}
