#!/usr/bin/env bash
# Batch-decode ladder: measure one shape per server lifetime.
#
# Each shape is warmed with its own prefixes, timed, and then thrown away by restarting the
# server.  A single server cannot host the whole ladder: the prefixes of shape k are still
# resident when shape k+1 warms, so the pool is too full to admit every lane of the new shape and
# the admission pass evicts one of the lanes it just created, turning the timed round back into a
# re-prefill.  Restarting keeps every measured set resident from an empty pool, which is what the
# numbers are supposed to show.
#
#   tools/v100/context-lab/run-ladder.sh 8x10000 8x20000 5x30000 4x40000 2x80000
#
# CONTEXT_LAB_WORKDIR (default: $PWD) holds logs/ and results/.  Each shape is run with
# batch_decode.py --baseline, so the printed uplift is against one stream at the same prompt
# length.  CONTEXT_LAB_PORT is honoured by the launcher.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: $0 NxLENGTH..." >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workdir=${CONTEXT_LAB_WORKDIR:-$PWD}
mkdir -p "${workdir}/logs" "${workdir}/results"

readonly port="${CONTEXT_LAB_PORT:-8080}"
readonly request_log="${workdir}/logs/ladder.jsonl"
readonly serve_log="${workdir}/logs/ladder-serve.log"

server_pids() { pgrep -x ninfer-serve || true; }

stop_server() {
    local pids
    pids=$(server_pids)
    [[ -n "${pids}" ]] && kill ${pids} 2>/dev/null
    for _ in $(seq 1 60); do
        [[ -z "$(server_pids)" ]] && return 0
        sleep 1
    done
    echo "ninfer-serve did not exit" >&2
    return 1
}

wait_ready() {
    local deadline=$((SECONDS + 900))
    while ((SECONDS < deadline)); do
        if curl -fsS -m 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 3
    done
    echo "server did not become ready on port ${port}" >&2
    return 1
}

trap 'stop_server >/dev/null 2>&1 || true' EXIT

for shape in "$@"; do
    echo "=== ${shape}: starting a fresh server ==="
    stop_server
    CONTEXT_LAB_REQUEST_LOG="${request_log}" \
        setsid nohup "${here}/serve-batch-decode.sh" >>"${serve_log}" 2>&1 </dev/null &
    wait_ready

    python3 "${here}/batch_decode.py" "${shape}" \
        --request-log "${request_log}" \
        --out "${workdir}/results/${shape}.json" \
        --label "${shape}" --baseline \
        | tee -a "${workdir}/logs/ladder.log"
done

stop_server
echo "=== ladder complete ==="
