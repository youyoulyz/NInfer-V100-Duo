#!/usr/bin/env bash
# Does a shared prefix decode differently from one prefix per lane?
#
# For each concurrency N the runner times N streams twice at the same prompt length: once with N
# distinct prompts (a different cached prefix in every lane) and once with N byte-identical
# prompts (the same prefix everywhere, which is what a caller repeating one long prompt produces).
# The server is restarted between runs, so every set warms from an empty pool.
#
#   tools/v100/context-lab/run-prefix-compare.sh 10000 2 4 5 8
#
# CONTEXT_LAB_WORKDIR (default: $PWD) holds logs/ and results/.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 LENGTH N..." >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=server-control.sh
source "${here}/server-control.sh"

length=$1
shift
workdir=${CONTEXT_LAB_WORKDIR:-$PWD}
mkdir -p "${workdir}/logs" "${workdir}/results"

readonly request_log="${workdir}/logs/prefix-compare.jsonl"
readonly serve_log="${workdir}/logs/prefix-compare-serve.log"

trap 'stop_server >/dev/null 2>&1 || true' EXIT

for streams in "$@"; do
    for variant in distinct shared; do
        label="${variant}-${streams}x${length}"
        echo "=== ${label}: starting a fresh server ==="
        start_server "${request_log}" "${serve_log}"

        command=(python3 "${here}/batch_decode.py" "${streams}x${length}")
        if [[ ${variant} == shared ]]; then
            command+=(--shared-prefix)
        fi
        command+=(--request-log "${request_log}" --out "${workdir}/results/${label}.json"
                  --label "${label}")
        "${command[@]}" | tee -a "${workdir}/logs/prefix-compare.log"
    done
done

stop_server
echo "=== prefix comparison complete ==="
