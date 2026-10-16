#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# discovery_fleet.sh — bring up the central-directory discovery plane
#
# Starts one NameServer + N_SHARDS DiscoveryAuthority processes. Each
# DiscoveryAuthority owns one shard of the keyspace; workers route events
# by `chunk_to_key(token_ids[:256]) % N_SHARDS`, and the lookup-side
# DataPlaneLudoTable inside the router mirrors all shards (`LOOKUP_SHARDING_MODE=full`).
#
# Sister script to sglang_fleet.sh. Same CLI surface.
#
# Usage
#   scripts/experiment/discovery_fleet.sh start  --shards 8        # one shard per worker
#   scripts/experiment/discovery_fleet.sh start  --shards 4 --ns-port 9999 --auth-base-port 8101
#   scripts/experiment/discovery_fleet.sh status
#   scripts/experiment/discovery_fleet.sh urls
#   scripts/experiment/discovery_fleet.sh stop
#
# Flags
#   --shards          N         number of authority processes (default 8)
#   --ns-port         P         NameServer port (default 9999)
#   --auth-base-port  P         authority#i listens on P + i (default 8101)
#   --workers         N         expected N_DP_WORKERS the authorities are sized for (default 8)
#   --log-dir         <path>    default logs/discovery
#   --persistence-dir <path>    AUTHORITY_LOG_DIR (default ./back)
#   --boot-timeout    S         per-process /alive wait (default 60)
#   --force                     wipe pidfiles + persistence on start
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SHARDS=8
NS_PORT=9999
AUTH_BASE_PORT=8101
WORKERS=8
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/discovery}"
PERSIST_DIR="${PERSIST_DIR:-$PROJECT_ROOT/back}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-60}"
FORCE=0

CMD="${1:-}"
[[ -z "$CMD" ]] && { echo "Usage: $0 {start|stop|status|urls} [flags]"; exit 1; }
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards)           SHARDS="$2";           shift 2 ;;
        --ns-port)          NS_PORT="$2";          shift 2 ;;
        --auth-base-port)   AUTH_BASE_PORT="$2";   shift 2 ;;
        --workers)          WORKERS="$2";          shift 2 ;;
        --log-dir)          LOG_DIR="$2";          shift 2 ;;
        --persistence-dir)  PERSIST_DIR="$2";      shift 2 ;;
        --boot-timeout)     BOOT_TIMEOUT="$2";     shift 2 ;;
        --force)            FORCE=1;               shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
PID_DIR="$LOG_DIR"

ns_pidfile()   { echo "$PID_DIR/nameserver.pid"; }
ns_log()       { echo "$LOG_DIR/nameserver.log"; }
auth_pidfile() { echo "$PID_DIR/authority_${1}.pid"; }
auth_log()     { echo "$LOG_DIR/authority_${1}.log"; }
auth_port()    { echo $((AUTH_BASE_PORT + $1)); }

build_authority_urls() {
    local urls=()
    for ((i=0; i<SHARDS; ++i)); do
        urls+=("http://127.0.0.1:$(auth_port $i)")
    done
    (IFS=, ; echo "${urls[*]}")
}
NS_URL="http://127.0.0.1:${NS_PORT}"
AUTH_URLS="$(build_authority_urls)"

wait_alive() {
    local url="$1"; local what="$2"
    local end=$(( SECONDS + BOOT_TIMEOUT ))
    while (( SECONDS < end )); do
        if curl -fsS -m 2 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    echo "    ! $what failed to come up at $url within ${BOOT_TIMEOUT}s" >&2
    return 1
}

is_running() {
    local pidfile="$1"
    [[ -f "$pidfile" ]] || return 1
    local pid="$(cat "$pidfile" 2>/dev/null || echo)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_one() {
    local name="$1"; local cmd="$2"; local pidfile="$3"; local log="$4"
    if is_running "$pidfile"; then
        echo "  [$name] already running (pid $(cat "$pidfile"))"
        return 0
    fi
    nohup bash -c "$cmd" >"$log" 2>&1 &
    echo $! > "$pidfile"
    echo "  [$name] pid=$(cat "$pidfile") log=$log"
}

stop_one() {
    local name="$1"; local pidfile="$2"
    if [[ ! -f "$pidfile" ]]; then echo "  [$name] no pidfile"; return; fi
    local pid="$(cat "$pidfile" 2>/dev/null || echo)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 0.5
        kill -9 "$pid" 2>/dev/null || true
        echo "  [$name] stopped (pid $pid)"
    else
        echo "  [$name] not running"
    fi
    rm -f "$pidfile"
}

cmd_start() {
    if (( FORCE )); then
        for f in "$PID_DIR"/*.pid; do [[ -f "$f" ]] && rm -f "$f"; done
        rm -rf "$PERSIST_DIR"
    fi
    mkdir -p "$PERSIST_DIR"

    echo "=== Starting discovery plane ==="
    echo "  shards=$SHARDS  ns=$NS_URL  auths=$AUTH_URLS"
    echo "  workers=$WORKERS  persistence=$PERSIST_DIR  logs=$LOG_DIR"

    local common_env="cd '$PROJECT_ROOT'; \
        export PYTHONPATH='$PROJECT_ROOT':\${PYTHONPATH:-}; \
        export MAINTENANCE_MODE=central; \
        export AUTHORITY_URLS='$AUTH_URLS'; \
        export NAMESERVER_URL='$NS_URL'; \
        export N_PREFILL_WORKERS='$WORKERS'; \
        export LOOKUP_SHARDING_MODE=full; \
        export AUTHORITY_LOG_DIR='$PERSIST_DIR'"

    start_one "name_server" \
        "$common_env; exec python3 -m ludo_sglang.authority.name_server" \
        "$(ns_pidfile)" "$(ns_log)"
    wait_alive "$NS_URL/topology" "name_server"

    for ((i=0; i<SHARDS; ++i)); do
        start_one "authority_${i}" \
            "$common_env; export SHARD_ID=$i; exec python3 -m ludo_sglang.authority.authority_main" \
            "$(auth_pidfile $i)" "$(auth_log $i)"
    done
    for ((i=0; i<SHARDS; ++i)); do
        wait_alive "http://127.0.0.1:$(auth_port $i)/alive" "authority_${i}"
    done

    echo ""
    echo "=== Discovery fleet up ==="
    echo "  NAMESERVER_URL=$NS_URL"
    echo "  AUTHORITY_URLS=$AUTH_URLS"
    echo ""
    echo "Export for the worker fleet:"
    echo "  export LUDO_AUTHORITY_URLS=\"$AUTH_URLS\""
    echo "  export NAMESERVER_URL=\"$NS_URL\""
    echo ""
    echo "Export for the router (lookup):"
    echo "  export MAINTENANCE_MODE=central"
    echo "  export AUTHORITY_URLS=\"$AUTH_URLS\""
    echo "  export NAMESERVER_URL=\"$NS_URL\""
    echo "  export LOOKUP_ID=0"
    echo "  export LOOKUP_SHARDING_MODE=full"
}

cmd_stop() {
    echo "=== Stopping discovery plane ==="
    shopt -s nullglob
    for f in "$PID_DIR"/authority_*.pid; do stop_one "$(basename "$f" .pid)" "$f"; done
    shopt -u nullglob
    stop_one "name_server" "$(ns_pidfile)"
}

cmd_status() {
    echo "=== Discovery plane status ==="
    local pidfile="$(ns_pidfile)"
    if is_running "$pidfile"; then
        echo "  name_server   pid=$(cat "$pidfile")  $NS_URL"
    else
        echo "  name_server   DOWN"
    fi
    shopt -s nullglob
    local pfiles=("$PID_DIR"/authority_*.pid)
    shopt -u nullglob
    if (( ${#pfiles[@]} == 0 )); then
        echo "  (no authorities)"
        return
    fi
    for pf in "${pfiles[@]}"; do
        local idx; idx="$(basename "$pf" .pid | sed 's/authority_//')"
        if is_running "$pf"; then
            echo "  authority_${idx}  pid=$(cat "$pf")  http://127.0.0.1:$((AUTH_BASE_PORT + idx))"
        else
            echo "  authority_${idx}  DOWN  (stale pidfile)"
        fi
    done
}

cmd_urls() {
    echo "NAMESERVER_URL=$NS_URL"
    echo "AUTHORITY_URLS=$AUTH_URLS"
}

case "$CMD" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    urls)   cmd_urls ;;
    *) echo "Unknown command: $CMD"; exit 1 ;;
esac
