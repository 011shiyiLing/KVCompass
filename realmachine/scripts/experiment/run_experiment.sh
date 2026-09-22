#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_experiment.sh — sweep (scheme × rate), write per-cell CSV.
#
# This script assumes the SGLang fleet is already up (use
# scripts/experiment/sglang_fleet.sh to start it). For each (scheme, rate) cell
# it does:
#
#   1. Flush every SGLang worker's RadixCache so the previous
#      scheme's residual caches don't leak into this measurement.
#   2. Start the FastAPI router with SCHEME=$S in the background.
#   3. Wait for the router to be healthy.
#   4. Replay the trace at the target rate, write CSV to
#      results/<scheme>_r<rate>.csv.
#   5. Stop the router.
#
# Usage
# -----
#   # Full default sweep (6 schemes × N rates)
#   bash scripts/experiment/run_experiment.sh
#
#   # Custom subset
#   SCHEMES="ludo radix_tree consistent_hash" \
#   RATES="2 8 32" \
#   TRACE=traces/mix_500_seed42.jsonl \
#   bash scripts/experiment/run_experiment.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults (env-overridable) ──────────────────────────────────────────────
SCHEMES="${SCHEMES:-ludo radix_tree consistent_hash kademlia chord central_directory}"
RATES="${RATES:-8 16 32 64}"
TRACE="${TRACE:-traces/mix_3000_loogle30_seed42.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-results}"
WARMUP_S="${WARMUP_S:-10}"
MAX_CONCURRENT="${MAX_CONCURRENT:-512}"
ROUTER_PORT="${ROUTER_PORT:-8001}"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_BOOT_TIMEOUT="${ROUTER_BOOT_TIMEOUT:-60}"
WAN_PERCENTILE="${WAN_PERCENTILE:-p50}"
NUM_NODES_LOGICAL="${NUM_NODES_LOGICAL:-64}"

if [[ -z "${SGLANG_WORKER_URLS:-}" ]]; then
    echo "ERROR: SGLANG_WORKER_URLS not set. Start the fleet first:"
    echo "  bash scripts/experiment/sglang_fleet.sh start --mode a100 --workers 8 --model ..."
    echo "  export SGLANG_WORKER_URLS=\"...\""
    exit 1
fi
if [[ ! -f "$TRACE" ]]; then
    echo "ERROR: trace file not found: $TRACE"
    echo "  Generate first:  python -m ludo_sglang.workload.mixed_trace --out $TRACE --n 500"
    exit 1
fi
if [[ -z "${MODEL_NAME:-}" ]]; then
    echo "WARN: MODEL_NAME unset; relying on the default in config.py"
fi

mkdir -p "$RESULTS_DIR" logs

# ── Helpers ─────────────────────────────────────────────────────────────────
flush_caches() {
    IFS=',' read -ra urls <<< "$SGLANG_WORKER_URLS"
    for u in "${urls[@]}"; do
        # SGLang's /flush_cache returns 200 on success; ignore failures
        # (some sglang versions don't expose the endpoint, in which
        # case the worker simply keeps its RadixCache state).
        curl -sS -X POST -m 5 "${u%/}/flush_cache" >/dev/null 2>&1 || true
    done
}

wait_router_up() {
    # Probe /alive (lightweight, doesn't touch the SGLang fleet) so a
    # slow-to-respond worker doesn't block the boot probe. /health
    # remains for diagnostic use after the router is up.
    local deadline=$(( SECONDS + ROUTER_BOOT_TIMEOUT ))
    while (( SECONDS < deadline )); do
        if curl -fsS -m 2 "http://${ROUTER_HOST}:${ROUTER_PORT}/alive" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

stop_router() {
    if [[ -n "${ROUTER_PID:-}" ]] && kill -0 "$ROUTER_PID" 2>/dev/null; then
        kill "$ROUTER_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$ROUTER_PID" 2>/dev/null || true
    fi
    ROUTER_PID=""
}

trap stop_router EXIT INT TERM

# ── Sweep ───────────────────────────────────────────────────────────────────
echo "=== Experiment sweep ==="
echo "  schemes : $SCHEMES"
echo "  rates   : $RATES"
echo "  trace   : $TRACE"
echo "  results : $RESULTS_DIR/"
echo "  WAN_pct : $WAN_PERCENTILE   logical_N : $NUM_NODES_LOGICAL"
echo ""

cell_idx=0
for S in $SCHEMES; do
    echo "── scheme: $S ──────────────────────────────"
    for R in $RATES; do
        cell_idx=$(( cell_idx + 1 ))
        out_csv="$RESULTS_DIR/${S}_r${R}.csv"
        log_file="logs/router_${S}_r${R}.log"
        echo "  [cell $cell_idx]  rate=${R} rps  →  $(basename "$out_csv")"

        # 1) flush SGLang caches so each measurement starts from cold
        flush_caches

        # 2) start router
        SCHEME="$S" \
        WAN_PERCENTILE="$WAN_PERCENTILE" \
        NUM_NODES_LOGICAL="$NUM_NODES_LOGICAL" \
        ROUTER_HOST="$ROUTER_HOST" \
        ROUTER_PORT="$ROUTER_PORT" \
        nohup python3 -m ludo_sglang.router.ludo_router >"$log_file" 2>&1 &
        ROUTER_PID=$!

        if ! wait_router_up; then
            echo "    ERROR: router failed to come up; see $log_file"
            stop_router
            continue
        fi

        # 3) replay
        if ! python3 -m ludo_sglang.workload.trace_replayer \
                --trace "$TRACE" \
                --router "http://${ROUTER_HOST}:${ROUTER_PORT}" \
                --rate "$R" \
                --warmup "$WARMUP_S" \
                --max-concurrent "$MAX_CONCURRENT" \
                --out "$out_csv" \
                >"logs/replayer_${S}_r${R}.log" 2>&1
        then
            echo "    WARN: replayer reported failures (see logs/replayer_${S}_r${R}.log)"
        fi

        # 4) stop router
        stop_router
        sleep 1   # let port release before next iteration
    done
done

echo ""
echo "=== Sweep complete ==="
echo "  Result CSVs : $RESULTS_DIR/<scheme>_r<rate>.csv"
echo "  Router logs : logs/router_<scheme>_r<rate>.log"
echo ""
echo "Next step: produce fig1/fig2/fig2b plots:"
echo "  python3 scripts/plotting/plot_fig1_fig2.py --results-dir $RESULTS_DIR"
