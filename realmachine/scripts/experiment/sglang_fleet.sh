#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# sglang_fleet.sh — multi-port SGLang fleet launcher / stopper / status
#
# Same script for A6000 debug (N processes share GPU 0) and A100 prod
# (each process pinned to its own GPU). Each SGLang server is an independent
# process listening on BASE_PORT + i, addressed by the router as a separate
# worker. No --dp-size / data_parallel_rank involved anywhere.
#
# Usage
#   scripts/experiment/sglang_fleet.sh start --mode a6000 --workers 4 \
#       --model Qwen/Qwen2.5-1.5B-Instruct
#
#   scripts/experiment/sglang_fleet.sh start --mode a100  --workers 8 \
#       --model Qwen/Qwen2.5-7B-Instruct
#
#   scripts/experiment/sglang_fleet.sh status
#   scripts/experiment/sglang_fleet.sh stop
#
# Flags (all optional except --mode and --workers on `start`)
#   --mode       a6000 | a100         picks GPU pinning + default mem-fraction
#   --workers    N                    number of SGLang processes
#   --model      <hf-id-or-path>      passed to --model-path
#   --base-port  P                    workers listen on P, P+1, ... (default 30000)
#   --mem-frac   F                    --mem-fraction-static override
#   --log-dir    <path>               default logs/sglang
#   --extra      "<sglang flags>"     extra flags appended to every server
#   --force                           on `start`: clean stale PID files first
#
# Health check waits up to HEALTH_TIMEOUT seconds (default 600) per worker.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
MODE=""
WORKERS=""
MODEL=""
BASE_PORT="${BASE_PORT:-30000}"
MEM_FRAC=""
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/sglang}"
EXTRA_FLAGS=""
FORCE=0
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

# ── Arg parsing ──────────────────────────────────────────────────────────────
CMD="${1:-}"
[[ -z "$CMD" ]] && { echo "Usage: $0 {start|stop|status|urls} [flags]"; exit 1; }
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2";        shift 2 ;;
        --workers)    WORKERS="$2";     shift 2 ;;
        --model)      MODEL="$2";       shift 2 ;;
        --base-port)  BASE_PORT="$2";   shift 2 ;;
        --mem-frac)   MEM_FRAC="$2";    shift 2 ;;
        --log-dir)    LOG_DIR="$2";     shift 2 ;;
        --extra)      EXTRA_FLAGS="$2"; shift 2 ;;
        --force)      FORCE=1;          shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
PID_DIR="$LOG_DIR"

worker_pidfile() { echo "$PID_DIR/worker_${1}.pid"; }
worker_log()     { echo "$LOG_DIR/worker_${1}.log"; }
worker_port()    { echo $((BASE_PORT + $1)); }

list_running_workers() {
    shopt -s nullglob
    local files=("$PID_DIR"/worker_*.pid)
    shopt -u nullglob
    for f in "${files[@]}"; do
        local idx pid
        idx=$(basename "$f" .pid | sed 's/^worker_//')
        pid=$(cat "$f" 2>/dev/null || echo "")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$idx $pid"
        fi
    done
}

# ── Subcommands ──────────────────────────────────────────────────────────────

cmd_status() {
    local any=0
    while read -r idx pid; do
        any=1
        local port; port=$(worker_port "$idx")
        local health="?"
        if command -v curl >/dev/null && curl -fsS --max-time 1 \
                "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            health="OK"
        else
            health="DOWN"
        fi
        printf "  worker %s  pid=%s  port=%d  health=%s  log=%s\n" \
            "$idx" "$pid" "$port" "$health" "$(worker_log "$idx")"
    done < <(list_running_workers)
    [[ $any -eq 0 ]] && echo "  (no workers running)"
}

cmd_urls() {
    while read -r idx _; do
        echo "http://127.0.0.1:$(worker_port "$idx")"
    done < <(list_running_workers)
}

cmd_stop() {
    local any=0
    while read -r idx pid; do
        any=1
        echo "  stopping worker $idx (pid=$pid) ..."
        kill "$pid" 2>/dev/null || true
    done < <(list_running_workers)
    [[ $any -eq 0 ]] && { echo "  (no workers running)"; return; }
    # Give SGLang a few seconds to flush, then SIGKILL stragglers
    sleep 3
    while read -r idx pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  SIGKILL worker $idx (pid=$pid)"
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$(worker_pidfile "$idx")"
    done < <(list_running_workers)
    # Clean any remaining stale pidfiles
    rm -f "$PID_DIR"/worker_*.pid
}

cmd_start() {
    [[ -z "$MODE" ]]    && { echo "ERROR: --mode required (a6000 | a100)"; exit 1; }
    [[ -z "$WORKERS" ]] && { echo "ERROR: --workers required"; exit 1; }
    [[ -z "$MODEL" ]]   && { echo "ERROR: --model required"; exit 1; }

    case "$MODE" in
        a6000|a100) ;;
        *) echo "ERROR: --mode must be a6000 or a100"; exit 1 ;;
    esac

    # Default mem-fraction by mode
    if [[ -z "$MEM_FRAC" ]]; then
        if [[ "$MODE" == "a6000" ]]; then
            # Co-resident: leave headroom. 4 workers × 0.20 ≈ 38GB on 48GB.
            MEM_FRAC=$(awk -v n="$WORKERS" 'BEGIN{ printf "%.2f", 0.85/n }')
        else
            MEM_FRAC="0.85"
        fi
    fi

    # Detect existing
    local running; running=$(list_running_workers || true)
    if [[ -n "$running" ]]; then
        if [[ "$FORCE" -eq 1 ]]; then
            echo "  --force: cleaning up stale workers"
            cmd_stop
        else
            echo "ERROR: workers already running. Use --force to restart, or run: $0 stop"
            echo "$running" | sed 's/^/  /'
            exit 1
        fi
    fi

    # GPU sanity
    local n_gpus=0
    if command -v nvidia-smi >/dev/null; then
        n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
    fi
    if [[ "$MODE" == "a100" ]]; then
        if [[ "$n_gpus" -lt "$WORKERS" ]]; then
            echo "ERROR: --mode a100 with --workers $WORKERS but only $n_gpus GPU(s) visible"
            exit 1
        fi
    elif [[ "$MODE" == "a6000" ]]; then
        if [[ "$n_gpus" -lt 1 ]]; then
            echo "ERROR: --mode a6000 needs at least 1 GPU; nvidia-smi reports $n_gpus"
            exit 1
        fi
    fi

    if ! python3 -c "import sglang" 2>/dev/null; then
        echo "ERROR: 'import sglang' failed. Install matching version on this machine first."
        exit 1
    fi

    echo "=== Launching SGLang fleet ==="
    echo "  mode        : $MODE"
    echo "  workers     : $WORKERS"
    echo "  model       : $MODEL"
    echo "  base-port   : $BASE_PORT"
    echo "  mem-frac    : $MEM_FRAC"
    echo "  log-dir     : $LOG_DIR"
    [[ -n "$EXTRA_FLAGS" ]] && echo "  extra       : $EXTRA_FLAGS"
    echo ""

    for i in $(seq 0 $((WORKERS - 1))); do
        local port; port=$(worker_port "$i")
        local log;  log=$(worker_log "$i")
        local pidf; pidf=$(worker_pidfile "$i")
        local gpu_id
        if [[ "$MODE" == "a6000" ]]; then
            gpu_id=0
        else
            gpu_id="$i"
        fi

        echo "  -> worker $i  GPU=$gpu_id  port=$port  log=$log"
        # shellcheck disable=SC2086
        #
        # Discovery-plane env passed straight through to the SGLang process so
        # the patched HiRadixCache hooks (notify_advertise / notify_demote /
        # notify_promote / notify_evict in tier_notifier.py) auto-activate:
        #   - LUDO_AUTHORITY_URLS  → central mode (this script's intended use)
        #   - LUDO_ROUTER_URL      → gossip mode (legacy)
        #   - neither              → notifier is a no-op
        # If you want central mode, export both LUDO_AUTHORITY_URLS and
        # NAMESERVER_URL before calling this script.
        # PYTHONPATH is set so SGLang's patched hiradix_cache.py can
        # `from ludo_sglang.worker.tier_notifier import ...`.
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
        LUDO_WORKER_RANK="$i" \
        LUDO_AUTHORITY_URLS="${LUDO_AUTHORITY_URLS:-}" \
        LUDO_ROUTER_URL="${LUDO_ROUTER_URL:-}" \
        NAMESERVER_URL="${NAMESERVER_URL:-}" \
        CHUNK_SIZES="${CHUNK_SIZES:-}" \
        nohup python3 -m sglang.launch_server \
            --model-path "$MODEL" \
            --host 127.0.0.1 \
            --port "$port" \
            --mem-fraction-static "$MEM_FRAC" \
            $EXTRA_FLAGS \
            >"$log" 2>&1 &
        echo $! > "$pidf"
    done

    echo ""
    echo "=== Waiting for workers to become healthy (timeout ${HEALTH_TIMEOUT}s each) ==="
    local deadline=$((SECONDS + HEALTH_TIMEOUT))
    for i in $(seq 0 $((WORKERS - 1))); do
        local port; port=$(worker_port "$i")
        local pid;  pid=$(cat "$(worker_pidfile "$i")")
        printf "  worker %s (pid=%s, port=%d) " "$i" "$pid" "$port"
        while true; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo " CRASHED — see $(worker_log "$i")"
                exit 1
            fi
            if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
                echo " OK"
                break
            fi
            if [[ $SECONDS -ge $deadline ]]; then
                echo " TIMEOUT — see $(worker_log "$i")"
                exit 1
            fi
            printf "."
            sleep 2
        done
    done

    echo ""
    echo "=== Fleet up. Worker URLs ==="
    cmd_urls
    echo ""
    echo "Export for the router:"
    local urls
    urls=$(cmd_urls | paste -sd "," -)
    echo "  export SGLANG_WORKER_URLS=\"$urls\""
}

case "$CMD" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    urls)   cmd_urls ;;
    *) echo "Unknown command: $CMD (use start|stop|status|urls)"; exit 1 ;;
esac
