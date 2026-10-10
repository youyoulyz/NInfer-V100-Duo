#!/usr/bin/env bash
set -euo pipefail

# Long-context session pool profile: several full-length conversations retained on NVMe, one
# request at a time.  A prefix displaced from the paged KV pool is parked in the pinned host tier
# and then the NVMe tier instead of being discarded, so a conversation that comes back -- or a
# server that restarts -- resumes it with no re-prefill.  Measured with tools/v100/context-lab/
# pool_driver.py; see README.md in this directory for the recorded numbers.
#
#   CONTEXT_LAB_ARTIFACT  model artifact (default: the local Qwen3.8-27B NVFP4 build)
#   CONTEXT_LAB_TIER_DIR  NVMe tier directory; one directory holds one whole pool (default:
#                         /var/tmp/ninfer-context-lab-pool)
#   CONTEXT_LAB_PORT      HTTP port (default 8080)
# Resolve this script through any symlink layer, so a link kept in a scratch directory still
# finds the build tree it belongs to.
script_path=${BASH_SOURCE[0]}
while [[ -L "${script_path}" ]]; do
    link_target=$(readlink -- "${script_path}")
    case "${link_target}" in
        /*) script_path=${link_target} ;;
        *) script_path=$(dirname -- "${script_path}")/${link_target} ;;
    esac
done
script_dir=$(cd -- "$(dirname -- "${script_path}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../../.." && pwd)

readonly executable="${repo_dir}/build-v100-duo/apps/ninfer-serve"
readonly runtime_lib_dir="${repo_dir}/build/_deps/install/lib"
readonly cuda_lib_dir=/usr/local/cuda-12.8/lib64
readonly artifact="${CONTEXT_LAB_ARTIFACT:-/home/luyzh/models/qwen3_8_27b_nvfp4.ninfer}"
readonly tier_dir="${CONTEXT_LAB_TIER_DIR:-/var/tmp/ninfer-context-lab-pool}"
readonly port="${CONTEXT_LAB_PORT:-8080}"

if [[ ! -x "${executable}" ]]; then
    echo "ninfer-serve is missing: ${executable}" >&2
    echo "build it with: cmake --build ${repo_dir}/build-v100-duo -j" >&2
    exit 1
fi
if [[ ! -f "${artifact}" ]]; then
    echo "model artifact is missing: ${artifact}" >&2
    exit 1
fi

# Keep any caller-provided library path (the product links CUDA and FFmpeg out of the build tree).
ld_path="${runtime_lib_dir}:${cuda_lib_dir}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then ld_path="${ld_path}:${LD_LIBRARY_PATH}"; fi
export LD_LIBRARY_PATH="${ld_path}"

mkdir -p "${tier_dir}"

# One 200K-token INT8 session costs about 7.1 GiB in the tier; 96 GiB holds eight of them.
exec "${executable}" "${artifact}" \
    --tp 2 --devices 0,1 --host 127.0.0.1 --port "${port}" \
    --model-id qwen3.8-27b-nvfp4 \
    --max-context 200000 --kv-capacity 200000 --kv-dtype int8 --prefill-chunk 1024 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 3600000 \
    --host-kv-mib 2048 --disk-kv-path "${tier_dir}" --disk-kv-mib 98304 \
    --log-stats-interval-ms 60000
