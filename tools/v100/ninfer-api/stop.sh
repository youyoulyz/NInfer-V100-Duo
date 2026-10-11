#!/usr/bin/env bash
# Stops the NInfer API service (the model stays on disk; the tier keeps every parked prefix).
set -euo pipefail
pids=$(pgrep -x ninfer-serve || true)
if [[ -z "${pids}" ]]; then
    echo "ninfer-serve is not running"
    exit 0
fi
kill ${pids}
for _ in $(seq 1 60); do
    [[ -z "$(pgrep -x ninfer-serve || true)" ]] && { echo "stopped"; exit 0; }
    sleep 1
done
echo "ninfer-serve did not exit" >&2
exit 1
