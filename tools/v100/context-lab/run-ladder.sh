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
# shellcheck source=server-control.sh
source "${here}/server-control.sh"

workdir=${CONTEXT_LAB_WORKDIR:-$PWD}
mkdir -p "${workdir}/logs" "${workdir}/results"

readonly request_log="${workdir}/logs/ladder.jsonl"
readonly serve_log="${workdir}/logs/ladder-serve.log"

trap 'stop_server >/dev/null 2>&1 || true' EXIT

for shape in "$@"; do
    echo "=== ${shape}: starting a fresh server ==="
    start_server "${request_log}" "${serve_log}"

    python3 "${here}/batch_decode.py" "${shape}" \
        --request-log "${request_log}" \
        --out "${workdir}/results/${shape}.json" \
        --label "${shape}" --baseline \
        | tee -a "${workdir}/logs/ladder.log"
done

stop_server
echo "=== ladder complete ==="
