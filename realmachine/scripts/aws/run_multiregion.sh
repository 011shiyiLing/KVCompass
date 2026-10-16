#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_multiregion.sh — 3-region eval-grid sweep driver (SkyWalker-style).
#
# Run from the CONTROL machine (laptop or a small VM) after:
#   1. instances launched, `hosts.env` filled in and sourced;
#   2. every worker node   : setup_worker.sh install && start;
#   3. every router node   : setup_router.sh (repo + ludo binding + tokenizer);
#   4. WAN matrix measured : measure_wan_matrix.py -> wan_matrix_real.json,
#      pushed to $REPO_DIR/ on every router node;
#   5. the trace pushed to $REPO_DIR/traces/ on every router node
#      (same file everywhere — per-region replayers filter by origin).
#
# For each (scheme × rate) cell it flushes all SGLang caches, restarts one
# router per region (clean routing-table state), fires one per-region
# replayer on each router node in parallel (--origin-filter i on the
# same global Poisson schedule), then collects and merges the CSVs.
# With MODE=tpt it runs the closed-loop throughput cell per scheme.
#
# Usage:
#   source scripts/aws/hosts.env
#   WORKLOAD=mixed TRACE=traces/mixed_6000_seed42.jsonl \
#     RATES="2 4 8 12 16 20 24" bash scripts/aws/run_multiregion.sh
#   WORKLOAD=mixed TRACE=traces/mixed_6000_seed42.jsonl \
#     MODE=tpt bash scripts/aws/run_multiregion.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Bash arrays don't survive `export` into a child shell — load hosts.env
# here so `bash scripts/aws/run_multiregion.sh` works standalone.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/hosts.env" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/hosts.env"
fi

SCHEMES="${SCHEMES:-central_directory ludo radix_tree consistent_hash chord skywalker dynamo}"
RATES="${RATES:-2 4 8 12 16 20 24}"
MODE="${MODE:-sweep}"                   # sweep | tpt
WORKLOAD="${WORKLOAD:-mixed}"
TRACE="${TRACE:?set TRACE=traces/<file>.jsonl (path on the router nodes)}"
WARMUP_S="${WARMUP_S:-20}"
DURATION_PER_RATE="${DURATION_PER_RATE:-120}"   # measured seconds per rate cell
TPT_CLIENTS_PER_REGION="${TPT_CLIENTS_PER_REGION:-32}"
TPT_DURATION="${TPT_DURATION:-90}"
RESULTS_DIR="${RESULTS_DIR:-results/results_${WORKLOAD}}"
ROUTER_BOOT_TIMEOUT="${ROUTER_BOOT_TIMEOUT:-90}"
SEED="${SEED:-0}"

: "${ROUTER_HOSTS:?source scripts/aws/hosts.env first}"
N_REGIONS=${#ROUTER_HOSTS[@]}
mkdir -p "$RESULTS_DIR"

# Per-node eviction policy — matches the sim (simulate.py EVICT_POLICY):
# mixed workload → LFU (stationary Zipf template reuse rewards frequency),
# wildchat / lmsys → LRU (session-lifetime reuse rewards recency).
# Override with EVICTION_POLICY=lru|lfu; SKIP_WORKER_RESTART=1 skips the
# worker restart (e.g. when re-running the same workload).
if [[ "${EVICTION_POLICY:-auto}" == "auto" ]]; then
  case "$WORKLOAD" in
    mixed) EVICTION_POLICY="lfu" ;;
    *)     EVICTION_POLICY="lru" ;;
  esac
fi

SSH() { ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_USER@$1" "${@:2}"; }
SCP() { scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$@"; }

region_router_urls() {   # "0:http://r0:port,1:http://r1:port,..."
  local out=""
  for i in $(seq 0 $((N_REGIONS-1))); do
    out+="${i}:http://${ROUTER_HOSTS[$i]}:${ROUTER_PORT},"
  done
  echo "${out%,}"
}

peer_urls_for() {        # peers of region $1
  local out=""
  for i in $(seq 0 $((N_REGIONS-1))); do
    [[ "$i" == "$1" ]] && continue
    out+="http://${ROUTER_HOSTS[$i]}:${ROUTER_PORT},"
  done
  echo "${out%,}"
}

flush_caches() {
  echo "  flushing SGLang caches on all workers ..."
  IFS=',' read -ra urls <<< "$SGLANG_WORKER_URLS"
  for u in "${urls[@]}"; do
    SSH "${ROUTER_HOSTS[0]}" "curl -sS -X POST -m 5 '${u%/}/flush_cache'" \
      >/dev/null 2>&1 || true
  done
}

restart_workers_with_policy() {
  # Relaunch every SGLang worker with --radix-eviction-policy $EVICTION_POLICY.
  echo "=== restarting all workers with eviction policy: $EVICTION_POLICY ==="
  local all=() pids=()
  for i in $(seq 0 $((N_REGIONS-1))); do
    local arr="WORKER_HOSTS_${i}[@]"
    all+=("${!arr}")
  done
  for w in "${all[@]}"; do
    SSH "$w" "MODEL_NAME='$MODEL_NAME' SGLANG_PORT=$SGLANG_PORT \
      EVICTION_POLICY=$EVICTION_POLICY bash setup_worker.sh restart" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== all workers healthy (policy=$EVICTION_POLICY) ==="
}

start_routers() {  # $1 = scheme
  local scheme="$1"
  for i in $(seq 0 $((N_REGIONS-1))); do
    SSH "${ROUTER_HOSTS[$i]}" "cd $REPO_DIR && \
      pkill -f ludo_sglang.router.ludo_router 2>/dev/null; sleep 1; \
      SCHEME=$scheme \
      MODEL_NAME=$MODEL_NAME \
      SGLANG_WORKER_URLS='$SGLANG_WORKER_URLS' \
      WORKER_REGIONS='$WORKER_REGIONS' \
      WAN_EMULATION=0 \
      WAN_MATRIX_FILE=$REPO_DIR/wan_matrix_real.json \
      ROUTER_REGION=$i ROUTER_ID=$i \
      ROUTER_HOST=0.0.0.0 ROUTER_PORT=$ROUTER_PORT \
      PEER_ROUTER_URLS='$(peer_urls_for $i)' \
      REGION_ROUTER_URLS='$(region_router_urls)' \
      nohup python3 -m ludo_sglang.router.ludo_router \
        > router_${scheme}.log 2>&1 & echo started"
  done
  echo "  waiting for routers ..."
  for i in $(seq 0 $((N_REGIONS-1))); do
    ok=0
    for t in $(seq 1 "$ROUTER_BOOT_TIMEOUT"); do
      if SSH "${ROUTER_HOSTS[$i]}" \
           "curl -sf -m 2 http://127.0.0.1:${ROUTER_PORT}/alive" >/dev/null 2>&1; then
        ok=1; break
      fi
      sleep 1
    done
    [[ "$ok" == 1 ]] || { echo "ERROR: router $i did not boot"; exit 1; }
  done
}

stop_routers() {
  for i in $(seq 0 $((N_REGIONS-1))); do
    SSH "${ROUTER_HOSTS[$i]}" \
      "pkill -f ludo_sglang.router.ludo_router 2>/dev/null" || true
  done
}

run_cell() {  # $1=scheme  $2=rate|tpt
  local scheme="$1" cell="$2" pids=() extra=""
  for i in $(seq 0 $((N_REGIONS-1))); do
    if [[ "$cell" == "tpt" ]]; then
      extra="--closed-loop --clients $TPT_CLIENTS_PER_REGION \
             --duration $TPT_DURATION \
             --summary-out results_tmp/${scheme}_tpt_summary.region${i}.json"
      out="results_tmp/${scheme}_tpt.region${i}.csv"
    else
      # Fixed-duration sizing: fire rate × (warmup + duration) requests.
      local lim
      lim=$(python3 -c "print(int($cell * ($WARMUP_S + $DURATION_PER_RATE)))")
      extra="--rate $cell --origin-filter $i --limit $lim"
      out="results_tmp/${scheme}_r${cell}.region${i}.csv"
    fi
    SSH "${ROUTER_HOSTS[$i]}" "cd $REPO_DIR && mkdir -p results_tmp && \
      python3 -m ludo_sglang.workload.trace_replayer \
        --trace $TRACE \
        --router http://127.0.0.1:${ROUTER_PORT} \
        --warmup $WARMUP_S --seed $SEED \
        --out $out $extra" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done

  # Collect + merge.
  local parts=()
  for i in $(seq 0 $((N_REGIONS-1))); do
    if [[ "$cell" == "tpt" ]]; then
      SCP "$SSH_USER@${ROUTER_HOSTS[$i]}:$REPO_DIR/results_tmp/${scheme}_tpt.region${i}.csv" "$RESULTS_DIR/"
      SCP "$SSH_USER@${ROUTER_HOSTS[$i]}:$REPO_DIR/results_tmp/${scheme}_tpt_summary.region${i}.json" "$RESULTS_DIR/"
    else
      SCP "$SSH_USER@${ROUTER_HOSTS[$i]}:$REPO_DIR/results_tmp/${scheme}_r${cell}.region${i}.csv" "$RESULTS_DIR/"
    fi
  done
  if [[ "$cell" == "tpt" ]]; then
    python3 scripts/aws/merge_region_results.py tpt \
      --out "$RESULTS_DIR/${scheme}_tpt_summary.json" \
      "$RESULTS_DIR/${scheme}"_tpt_summary.region*.json
    python3 scripts/aws/merge_region_results.py csv \
      --out "$RESULTS_DIR/${scheme}_tpt.csv" \
      "$RESULTS_DIR/${scheme}"_tpt.region*.csv
  else
    python3 scripts/aws/merge_region_results.py csv \
      --out "$RESULTS_DIR/${scheme}_r${cell}.csv" \
      "$RESULTS_DIR/${scheme}_r${cell}".region*.csv
  fi
}

if [[ "${SKIP_WORKER_RESTART:-0}" != "1" ]]; then
  restart_workers_with_policy
fi

for scheme in $SCHEMES; do
  if [[ "$MODE" == "tpt" ]]; then
    echo "=== [$WORKLOAD] $scheme closed-loop throughput ==="
    flush_caches
    start_routers "$scheme"
    run_cell "$scheme" "tpt"
    stop_routers
  else
    for rate in $RATES; do
      echo "=== [$WORKLOAD] $scheme @ ${rate} req/s ==="
      flush_caches
      start_routers "$scheme"
      run_cell "$scheme" "$rate"
      stop_routers
    done
  fi
done
echo "done — results in $RESULTS_DIR"
