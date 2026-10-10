#!/usr/bin/env bash
set -euo pipefail

# Concurrency / batch-decode measurement profile: eight lanes and the full 200K-token INT8 KV
# pool, no retention tiers.  Every prefix the driver times is already resident, so nothing is
# evicted or restored and the tiers would only add I/O to the window being measured.
#
#   CONTEXT_LAB_ARTIFACT     model artifact (default: the local Qwen3.8-27B NVFP4 build)
#   CONTEXT_LAB_PORT         HTTP port (default 8080)
#   CONTEXT_LAB_REQUEST_LOG  when set, append schema-v10 request records to this JSONL file
#                            (the driver needs it for per-request decode seconds)
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
readonly port="${CONTEXT_LAB_PORT:-8080}"
readonly request_log="${CONTEXT_LAB_REQUEST_LOG:-}"

if [[ ! -x "${executable}" ]]; then
    echo "ninfer-serve is missing: ${executable}" >&2
    echo "build it with: cmake --build ${repo_dir}/build-v100-duo -j" >&2
    exit 1
fi
if [[ ! -f "${artifact}" ]]; then
    echo "model artifact is missing: ${artifact}" >&2
    exit 1
fi

ld_path="${runtime_lib_dir}:${cuda_lib_dir}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then ld_path="${ld_path}:${LD_LIBRARY_PATH}"; fi
export LD_LIBRARY_PATH="${ld_path}"

options=(
    --tp 2 --devices 0,1 --host 127.0.0.1 --port "${port}"
    --model-id qwen3.8-27b-nvfp4
    --max-context 200000 --kv-capacity 200000 --kv-dtype int8 --prefill-chunk 1024
    --spec mtp --draft-tokens 3 --lm-head-draft
    --max-concurrency 8 --max-pending-requests 16 --pending-timeout-ms 3600000
    --log-stats-interval-ms 15000
)
if [[ -n "${request_log}" ]]; then
    mkdir -p "$(dirname -- "${request_log}")"
    options+=(--request-log-jsonl "${request_log}")
fi

exec "${executable}" "${artifact}" "${options[@]}"
