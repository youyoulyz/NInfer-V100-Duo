#!/usr/bin/env bash
# NInfer API service: one 200K-token session, backed by the pinned host tier and the local NVMe tier.
#
#   ~/v100/ninfer-api/serve.sh > logs/serve.log 2>&1 &      # start (background)
#   kill $(pgrep -x ninfer-serve)                            # stop
#
# Every KV cache -- the retained device lane, the parked host image and the NVMe records -- is
# released after 24 h without use (`--kv-cache-ttl-seconds 86400`), so a conversation nobody comes
# back to stops holding memory or disk.
#
# Overrides: NINFER_API_HOST (default 0.0.0.0), NINFER_API_PORT, NINFER_API_ARTIFACT,
# NINFER_API_TIER_DIR, NINFER_API_DISK_MIB, NINFER_API_KEY.
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

if [[ ! -x "${executable}" ]]; then
    echo "ninfer-serve is missing: ${executable} (cmake --build ${repo_dir}/build-v100-duo -j)" >&2
    exit 1
fi
export LD_LIBRARY_PATH="${repo_dir}/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${tier_dir}"

exec "${executable}" "${artifact}" \
    --tp 2 --devices 0,1 --host "${host}" --port "${port}" \
    --model-id qwen3.8-27b-nvfp4 \
    --max-context 200000 --kv-capacity 200000 --kv-dtype int8 --prefill-chunk 1024 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 3600000 \
    --host-kv-mib 2048 --disk-kv-path "${tier_dir}" --disk-kv-mib "${disk_mib}" \
    --kv-cache-ttl-seconds 86400 --log-stats-interval-ms 60000 \
    ${NINFER_API_KEY:+--api-key "${NINFER_API_KEY}"}
