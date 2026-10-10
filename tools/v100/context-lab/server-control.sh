#!/usr/bin/env bash
# Server lifecycle shared by the context-lab runners.  Source this file; it defines
#
#   start_server <request-log> <serve-log>   restart serve-batch-decode.sh and wait until ready
#   stop_server                              kill ninfer-serve and wait for it to exit
#
# and honours CONTEXT_LAB_PORT.  The launcher is resolved next to this file, so the server keeps
# running when the runner restarts it between measurements.
[[ -n "${CONTEXT_LAB_SERVER_CONTROL:-}" ]] && return 0
CONTEXT_LAB_SERVER_CONTROL=1

lab_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lab_port=${CONTEXT_LAB_PORT:-8080}

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
        if curl -fsS -m 2 "http://127.0.0.1:${lab_port}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 3
    done
    echo "server did not become ready on port ${lab_port}" >&2
    return 1
}

start_server() {  # <request-log path> <server-log path>
    local log_path=$1 server_log=$2
    stop_server
    CONTEXT_LAB_REQUEST_LOG="${log_path}" \
        setsid nohup "${lab_dir}/serve-batch-decode.sh" >>"${server_log}" 2>&1 </dev/null &
    wait_ready
}
