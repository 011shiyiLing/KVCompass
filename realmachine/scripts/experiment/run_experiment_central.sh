#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_experiment_central.sh — sweep (scheme × rate) for the **real**
# central-directory discovery plane.
#
# Differences vs. scripts/experiment/run_experiment.sh:
#   * Brings up scripts/experiment/discovery_fleet.sh once before the sweep; tears it
#     down at the end.
#   * Each `ludo` cell runs the router with MAINTENANCE_MODE=central so the
#     LudoScheme builds ShardedDataPlaneLudoTable mirrors driven by the
#     authority broadcasts; non-ludo schemes still run in-process inside
#     the router (gossip plane is idle for them — they don't talk to the
#     authority).
#   * Wipes ./back/ between cells so each measurement starts with empty
#     authority state and a cold lookup mirror.
#
# Pre-conditions (caller responsibility):
#   * SGLang fleet is already up:  scripts/experiment/sglang_fleet.sh start --mode a100 ...
#   * The SGLang processes were spawned with LUDO_AUTHORITY_URLS +
#     NAMESERVER_URL exported, so the worker tier_notifier auto-activates.
#     The matching sglang_fleet.sh pass-through is already in place.
#   * SGLang HiRadixCache has been patched via scripts/setup/apply_sglang_tier_patch.py
#     so the four notify_* hooks fire on actual cache state-machine events.
#   * Workers were launched with the --enable-hierarchical-cache flag (DRAM
#     tier on) so demote / promote / evict actually happen.
#
# Usage
#   # Full default sweep (6 schemes × 4 rates, central mode for ludo)
#   bash scripts/experiment/run_experiment_central.sh
#
#   # Just verify central mode with one cell
#   SCHEMES=ludo RATES=8 bash scripts/experiment/run_experiment_central.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults (env-overridable) ──────────────────────────────────────────────
SCHEMES="${SCHEMES:-ludo radix_tree consistent_hash kademlia chord central_directory}"
RATES="${RATES:-8 12 16 20}"
TRACE="${TRACE:-traces/mix_3000_loogle30_seed42.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-results_central}"
WARMUP_S="${WARMUP_S:-10}"
MAX_CONCURRENT="${MAX_CONCURRENT:-512}"
ROUTER_PORT="${ROUTER_PORT:-8001}"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_BOOT_TIMEOUT="${ROUTER_BOOT_TIMEOUT:-60}"
WAN_PERCENTILE="${WAN_PERCENTILE:-p50}"
NUM_NODES_LOGICAL="${NUM_NODES_LOGICAL:-64}"

# Discovery-plane sizing
N_SHARDS="${N_SHARDS:-8}"                  # one authority per worker by default
NS_PORT="${NS_PORT:-9999}"
AUTH_BASE_PORT="${AUTH_BASE_PORT:-8101}"
PERSIST_DIR="${PERSIST_DIR:-$PROJECT_ROOT/back}"

NAMESERVER_URL="${NAMESERVER_URL:-http://127.0.0.1:${NS_PORT}}"
# Build AUTHORITY_URLS unless caller already supplied one
if [[ -z "${AUTHORITY_URLS:-}" ]]; then
    urls=()
    for ((i=0; i<N_SHARDS; ++i)); do urls+=("http://127.0.0.1:$((AUTH_BASE_PORT + i))"); done
    AUTHORITY_URLS="$(IFS=, ; echo "${urls[*]}")"
fi
export NAMESERVER_URL AUTHORITY_URLS

# Pre-condition checks
if [[ -z "${SGLANG_WORKER_URLS:-}" ]]; then
    echo "ERROR: SGLANG_WORKER_URLS not set. Start the fleet first:"
    echo "  scripts/experiment/sglang_fleet.sh start --mode a100 --workers 8 --model ..."
    exit 1
fi
if [[ ! -f "$TRACE" ]]; then
    echo "ERROR: trace not found: $TRACE"; exit 1
fi
if [[ -z "${MODEL_NAME:-}" ]]; then
    echo "WARN: MODEL_NAME unset; relying on the default in config.py"
fi

mkdir -p "$RESULTS_DIR" logs

# ── Helpers ─────────────────────────────────────────────────────────────────
flush_caches() {
    IFS=',' read -ra urls <<< "$SGLANG_WORKER_URLS"
    for u in "${urls[@]}"; do
        curl -sS -X POST -m 5 "${u%/}/flush_cache" >/dev/null 2>&1 || true
    done
}

wipe_persistence() {
    # Wipe authority log/snapshot dir so each cell starts with empty CP+DP
    # mirrors (the discovery fleet rebroadcasts a bootstrap snapshot to the
    # lookup on the next router boot). Be tolerant — callers should ensure
    # the discovery fleet is stopped first, but a stray .swp / journal file
    # shouldn't kill the sweep.
    rm -rf "$PERSIST_DIR" 2>/dev/null || true
    mkdir -p "$PERSIST_DIR"
}

wait_router_up() {
    local deadline=$(( SECONDS + ROUTER_BOOT_TIMEOUT ))
    while (( SECONDS < deadline )); do
        # Early-out if the router process died (SIGSEGV / OOM / exception).
        if [[ -n "${ROUTER_PID:-}" ]] && ! kill -0 "$ROUTER_PID" 2>/dev/null; then
            echo "    ! router pid ${ROUTER_PID} died during startup"
            return 1
        fi
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

stop_discovery() {
    bash "$SCRIPT_DIR/discovery_fleet.sh" stop || true
}

trap 'stop_router; stop_discovery' EXIT INT TERM

# ── Discovery fleet ─────────────────────────────────────────────────────────
echo "=== Booting discovery plane ==="
bash "$SCRIPT_DIR/discovery_fleet.sh" stop >/dev/null 2>&1 || true
wipe_persistence
N_WORKERS_DETECTED="$(awk -F, '{print NF}' <<< "$SGLANG_WORKER_URLS")"
bash "$SCRIPT_DIR/discovery_fleet.sh" start \
    --shards "$N_SHARDS" \
    --workers "$N_WORKERS_DETECTED" \
    --ns-port "$NS_PORT" \
    --auth-base-port "$AUTH_BASE_PORT" \
    --persistence-dir "$PERSIST_DIR" \
    --force
echo ""

# ── Sweep ───────────────────────────────────────────────────────────────────
echo "=== Sweep (central plane) ==="
echo "  schemes  : $SCHEMES"
echo "  rates    : $RATES"
echo "  trace    : $TRACE"
echo "  results  : $RESULTS_DIR/"
echo "  n_shards : $N_SHARDS  ns=$NAMESERVER_URL"
echo "  WAN_pct  : $WAN_PERCENTILE   logical_N : $NUM_NODES_LOGICAL"
echo ""

cell_idx=0
for S in $SCHEMES; do
    echo "── scheme: $S ──────────────────────────────"
    for R in $RATES; do
        cell_idx=$(( cell_idx + 1 ))
        out_csv="$RESULTS_DIR/${S}_r${R}.csv"
        log_file="logs/router_central_${S}_r${R}.log"
        echo "  [cell $cell_idx]  rate=${R} rps  →  $(basename "$out_csv")"

        # 1) Cold caches
        flush_caches

        # 2) For ludo cells, fully bounce the discovery plane: stop, wipe,
        #    sleep so ports fully release, restart, sleep again so authorities
        #    finish their /alive handshake before the router boots. Without
        #    these explicit waits the router process SIGSEGVs intermittently
        #    on cell ≥ 2 (race between router startup hooks and the
        #    name-server pushing stale lookup_addrs_for_shard pointing at
        #    the previous cell's dead router).
        if [[ "$S" == "ludo" ]]; then
            bash "$SCRIPT_DIR/discovery_fleet.sh" stop >/dev/null 2>&1 || true
            sleep 2
            wipe_persistence
            bash "$SCRIPT_DIR/discovery_fleet.sh" start \
                --shards "$N_SHARDS" \
                --workers "$N_WORKERS_DETECTED" \
                --ns-port "$NS_PORT" \
                --auth-base-port "$AUTH_BASE_PORT" \
                --persistence-dir "$PERSIST_DIR" \
                --force >/dev/null
            sleep 3
        else
            sleep 2
        fi

        # 3) Start router
        if [[ "$S" == "ludo" ]]; then
            ROUTER_ENV=( \
                "SCHEME=ludo" \
                "MAINTENANCE_MODE=central" \
                "AUTHORITY_URLS=$AUTHORITY_URLS" \
                "NAMESERVER_URL=$NAMESERVER_URL" \
                "LOOKUP_ID=0" \
                "LOOKUP_SHARDING_MODE=full" \
                "AUTHORITY_LOG_DIR=$PERSIST_DIR" \
            )
        else
            ROUTER_ENV=( \
                "SCHEME=$S" \
                "MAINTENANCE_MODE=gossip" \
            )
        fi

        env \
            "${ROUTER_ENV[@]}" \
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

        # 4) Replay
        if ! python3 -m ludo_sglang.workload.trace_replayer \
                --trace "$TRACE" \
                --router "http://${ROUTER_HOST}:${ROUTER_PORT}" \
                --rate "$R" \
                --warmup "$WARMUP_S" \
                --max-concurrent "$MAX_CONCURRENT" \
                --out "$out_csv" \
                >"logs/replayer_central_${S}_r${R}.log" 2>&1
        then
            echo "    WARN: replayer reported failures (see logs/replayer_central_${S}_r${R}.log)"
        fi

        # 5) Snapshot authority stats AND router-side scheme stats while
        #    everything is still up — helps spot cross-shard drops,
        #    broadcast failures, and (most importantly) whether the
        #    lookup mirror at the router actually has entries.
        if [[ "$S" == "ludo" ]]; then
            stats_log="logs/authority_stats_${S}_r${R}.json"
            : >"$stats_log"
            IFS=',' read -ra _auths <<< "$AUTHORITY_URLS"
            for u in "${_auths[@]}"; do
                echo "// $u" >>"$stats_log"
                curl -sS -m 3 "${u}/stats" >>"$stats_log" 2>/dev/null || true
                echo >>"$stats_log"
            done
            # Router-side scheme stats (lookup-mirror table sizes etc.)
            curl -sS -m 3 "http://${ROUTER_HOST}:${ROUTER_PORT}/scheme_stats" \
                >"logs/router_scheme_stats_${S}_r${R}.json" 2>/dev/null || true
        fi

        stop_router
        sleep 1
    done
done

echo ""
echo "=== Sweep complete ==="
echo "  Result CSVs : $RESULTS_DIR/<scheme>_r<rate>.csv"
echo "  Router logs : logs/router_central_<scheme>_r<rate>.log"
echo "  Authority stats per ludo cell : logs/authority_stats_ludo_r<rate>.json"
echo ""
echo "Plot:"
echo "  python3 scripts/plotting/plot_fig1_fig2.py --results-dir $RESULTS_DIR"
