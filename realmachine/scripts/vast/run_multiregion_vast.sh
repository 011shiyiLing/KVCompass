#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_multiregion_vast.sh — eval-grid sweep driver for the Vast.ai layout
# (one multi-GPU box per geographic region, real inter-continental WAN).
#
# Differences from scripts/aws/run_multiregion.sh:
#   * workers live on the SAME box as their region's router — each router
#     reaches its own region's workers via loopback (127.0.0.1:3000x) and
#     remote regions via the boxes' public port mappings;
#   * SSH goes to one box per region (root@IP -p SSH_PORT).
#
# Prereqs:
#   1. scripts/vast/hosts.env filled in;  2. repo rsynced to every box;
#   3. setup_box.sh install + start done on every box;
#   4. wan matrix measured (measure_wan_tcp.py) and pushed to every box;
#   5. traces/ pushed to every box.
#
# Usage:
#   WORKLOAD=mixed TRACE=traces/mixed_6000_seed42.jsonl \
#     RATES="2 4 8 12 16 20 24" bash scripts/vast/run_multiregion_vast.sh
#   WORKLOAD=mixed TRACE=... MODE=tpt bash scripts/vast/run_multiregion_vast.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hosts.env"

SCHEMES="${SCHEMES:-central_directory ludo radix_tree consistent_hash chord skywalker dynamo}"
RATES="${RATES:-2 4 8 12 16 20 24}"
MODE="${MODE:-sweep}"                   # sweep | tpt
WORKLOAD="${WORKLOAD:-mixed}"
TRACE="${TRACE:?set TRACE=traces/<file>.jsonl (path on the boxes)}"
WARMUP_S="${WARMUP_S:-20}"
DURATION_PER_RATE="${DURATION_PER_RATE:-120}"
TPT_CLIENTS_PER_REGION="${TPT_CLIENTS_PER_REGION:-32}"
TPT_DURATION="${TPT_DURATION:-90}"
RESULTS_DIR="${RESULTS_DIR:-results/results_${WORKLOAD}}"
ROUTER_BOOT_TIMEOUT="${ROUTER_BOOT_TIMEOUT:-90}"
SEED="${SEED:-0}"
ROUTER_PORT=8001

N_REGIONS=${#BOX_IP[@]}
mkdir -p "$RESULTS_DIR"

# Per-node eviction policy — matches the sim (mixed→LFU, else LRU).
if [[ "${EVICTION_POLICY:-auto}" == "auto" ]]; then
  case "$WORKLOAD" in
    mixed) EVICTION_POLICY="lfu" ;;
    *)     EVICTION_POLICY="lru" ;;
  esac
fi

# DRIVER_SELF_BOX: set to this box's index when the driver runs ON one
# of the boxes (overnight-robust mode — laptop can disconnect).  Commands
# for the self box run locally; the other boxes are reached with the
# mesh key.
SSHB() {  # $1 = box idx, rest = command
  if [[ "${DRIVER_SELF_BOX:-}" == "$1" ]]; then
    bash -c "${*:2}"
  else
    ssh -i "$SSH_KEY" -p "${BOX_SSH_PORT[$1]}" -o StrictHostKeyChecking=no \
        "$SSH_USER@${BOX_IP[$1]}" "${@:2}"
  fi
}
SCPB() {  # $1 = box idx, $2 = remote path, $3 = local path
  if [[ "${DRIVER_SELF_BOX:-}" == "$1" ]]; then
    cp "$2" "$3"
  else
    scp -i "$SSH_KEY" -P "${BOX_SSH_PORT[$1]}" -o StrictHostKeyChecking=no \
        "$SSH_USER@${BOX_IP[$1]}:$2" "$3"
  fi
}

# All cross-box traffic rides the SSH tunnel mesh (tunnel_mesh.sh):
# on every box, remote box b's worker i = 127.0.0.1:(40000+b*100+i),
# remote box b's router = 127.0.0.1:(40000+b*100+99).  The port scheme is
# identical on every box, so one URL string is valid mesh-wide.

worker_urls_for_region() {  # $1 = viewing region: own loopback, rest tunnels
  local view="$1" out="" b i
  for b in $(seq 0 $((N_REGIONS-1))); do
    for i in $(seq 0 $((${GPUS_PER_BOX[$b]}-1))); do
      if [[ "$b" == "$view" ]]; then
        out+="http://127.0.0.1:$((30000+i)),"
      else
        out+="http://127.0.0.1:$((40000+b*100+i)),"
      fi
    done
  done
  echo "${out%,}"
}

region_router_urls() {  # $1 = viewing region (own entry = local port)
  local view="$1" out="" i
  for i in $(seq 0 $((N_REGIONS-1))); do
    if [[ "$i" == "$view" ]]; then
      out+="${i}:http://127.0.0.1:${ROUTER_PORT},"
    else
      out+="${i}:http://127.0.0.1:$((40000+i*100+99)),"
    fi
  done
  echo "${out%,}"
}

peer_urls_for() {
  local out="" i
  for i in $(seq 0 $((N_REGIONS-1))); do
    [[ "$i" == "$1" ]] && continue
    out+="http://127.0.0.1:$((40000+i*100+99)),"
  done
  echo "${out%,}"
}

flush_caches() {
  echo "  flushing SGLang caches on all workers ..."
  local b i
  for b in $(seq 0 $((N_REGIONS-1))); do
    SSHB "$b" "$(for i in $(seq 0 $((${GPUS_PER_BOX[$b]}-1))); do
      echo -n "curl -sS -X POST -m 5 http://127.0.0.1:$((30000+i))/flush_cache >/dev/null 2>&1; "
    done) true" || true
  done
}

restart_workers_with_policy() {
  echo "=== restarting workers everywhere with eviction policy: $EVICTION_POLICY ==="
  local pids=() r
  for r in $(seq 0 $((N_REGIONS-1))); do
    SSHB "$r" "cd $REPO_DIR/scripts/vast && \
      N_GPUS=${GPUS_PER_BOX[$r]} MODEL_NAME='$MODEL_NAME' \
      HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} \
      EVICTION_POLICY=$EVICTION_POLICY bash setup_box.sh restart" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

stop_router_procs() {  # kill routers via runtime-assembled pattern —
  # a literal pattern would sit in the remote shell's own cmdline and
  # pkill would kill the session itself.
  local r
  for r in $(seq 0 $((N_REGIONS-1))); do
    SSHB "$r" "P=\$(printf '%s.router.ludo_router' ludo_sglang); \
      pkill -f \"\$P\" 2>/dev/null; true" || true
  done
}

start_routers() {  # $1 = scheme
  local scheme="$1" r
  stop_router_procs
  sleep 1
  for r in $(seq 0 $((N_REGIONS-1))); do
    # `cd X; cmd &` — NOT `cd X && cmd &`: with `&&` the `&` backgrounds
    # the whole and-list, leaving a wrapper subshell that holds sshd's
    # pipes open for the router's lifetime and the ssh call never returns.
    SSHB "$r" "cd $REPO_DIR; \
      SCHEME=$scheme \
      HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} \
      MODEL_NAME=$MODEL_NAME \
      SGLANG_WORKER_URLS='$(worker_urls_for_region "$r")' \
      WORKER_REGIONS='$WORKER_REGIONS' \
      WAN_EMULATION=0 \
      WAN_MATRIX_FILE=$REPO_DIR/wan_matrix_real.json \
      ROUTER_REGION=$r ROUTER_ID=$r \
      ROUTER_HOST=0.0.0.0 ROUTER_PORT=$ROUTER_PORT \
      PEER_ROUTER_URLS='$(peer_urls_for "$r")' \
      REGION_ROUTER_URLS='$(region_router_urls "$r")' \
      C_PREFILL_MS_PER_TOKEN_PRIOR=${C_PREFILL_MS_PER_TOKEN_PRIOR:-0.02} \
      PCIE_RELOAD_MS_PER_TOKEN=${PCIE_RELOAD_MS_PER_TOKEN:-0.0026} \
      ${ROUTER_EXTRA_ENV:-} \
      nohup python3 -m ludo_sglang.router.ludo_router \
        > router_${scheme}.log 2>&1 < /dev/null & echo started"
  done
  echo "  waiting for routers ..."
  for r in $(seq 0 $((N_REGIONS-1))); do
    local ok=0 t
    for t in $(seq 1 "$ROUTER_BOOT_TIMEOUT"); do
      if SSHB "$r" "curl -sf -m 2 http://127.0.0.1:${ROUTER_PORT}/alive" \
          >/dev/null 2>&1; then ok=1; break; fi
      sleep 1
    done
    [[ "$ok" == 1 ]] || { echo "ERROR: router $r did not boot "\
      "(check $REPO_DIR/router_${scheme}.log on box $r)"; exit 1; }
  done
}

stop_routers() {
  stop_router_procs
}

check_tunnels() {
  # Verify every box can reach every other box's worker0 through the
  # tunnel.  Supervisors self-heal, but REAL WAN outages on these paths
  # can last minutes (observed: US↔EU down >60 s) — wait out up to
  # 5 min per pair so the sweep pauses through an outage instead of
  # aborting between cells.
  local s b p t
  for s in $(seq 0 $((N_REGIONS-1))); do
    for b in $(seq 0 $((N_REGIONS-1))); do
      [[ "$b" == "$s" ]] && continue
      p=$((40000+b*100))
      for t in $(seq 1 60); do
        if SSHB "$s" "curl -sf -m 4 http://127.0.0.1:$p/get_model_info" \
            >/dev/null 2>&1; then break; fi
        [[ "$t" == 60 ]] && { echo "ERROR: tunnel box$s->box$b dead >5min"; return 1; }
        sleep 5
      done
    done
  done
  return 0
}

run_cell() {  # $1=scheme  $2=rate|tpt
  local scheme="$1" cell="$2" pids=() r extra out
  for r in $(seq 0 $((N_REGIONS-1))); do
    if [[ "$cell" == "tpt" ]]; then
      extra="--closed-loop --clients $TPT_CLIENTS_PER_REGION \
             --duration $TPT_DURATION \
             --warmup-requests ${WARMUP_REQUESTS:-0} \
             --summary-out results_tmp/${scheme}_tpt_summary.region${r}.json"
      out="results_tmp/${scheme}_tpt.region${r}.csv"
    else
      local lim
      lim=$(python3 -c "print(int($cell * ($WARMUP_S + $DURATION_PER_RATE)))")
      extra="--rate $cell --origin-filter $r --limit $lim \
             --warmup-requests ${WARMUP_REQUESTS_OPEN:-0} \
             --summary-out results_tmp/${scheme}_r${cell}_summary.region${r}.json"
      out="results_tmp/${scheme}_r${cell}.region${r}.csv"
    fi
    SSHB "$r" "cd $REPO_DIR && mkdir -p results_tmp && \
      python3 -m ludo_sglang.workload.trace_replayer \
        --trace $TRACE \
        --router http://127.0.0.1:${ROUTER_PORT} \
        --warmup $WARMUP_S --seed $SEED \
        --out $out $extra" &
    pids+=($!)
  done
  # Replayer exit 2 = some requests failed — record but DON'T abort the
  # sweep (failed rows are in the CSV as ok=0; analysis surfaces fail%).
  local rc_bad=0
  for p in "${pids[@]}"; do wait "$p" || rc_bad=$((rc_bad+1)); done
  [[ "$rc_bad" -gt 0 ]] && \
    echo "WARN: $rc_bad replayer(s) reported failed requests in ${scheme}/${cell}"

  for r in $(seq 0 $((N_REGIONS-1))); do
    if [[ "$cell" == "tpt" ]]; then
      SCPB "$r" "$REPO_DIR/results_tmp/${scheme}_tpt.region${r}.csv" "$RESULTS_DIR/"
      SCPB "$r" "$REPO_DIR/results_tmp/${scheme}_tpt_summary.region${r}.json" "$RESULTS_DIR/"
    else
      SCPB "$r" "$REPO_DIR/results_tmp/${scheme}_r${cell}.region${r}.csv" "$RESULTS_DIR/"
      SCPB "$r" "$REPO_DIR/results_tmp/${scheme}_r${cell}_summary.region${r}.json" "$RESULTS_DIR/" || true
    fi
  done
  if [[ "$cell" == "tpt" ]]; then
    python3 "$SCRIPT_DIR/../aws/merge_region_results.py" tpt \
      --out "$RESULTS_DIR/${scheme}_tpt_summary.json" \
      "$RESULTS_DIR/${scheme}"_tpt_summary.region*.json
    python3 "$SCRIPT_DIR/../aws/merge_region_results.py" csv \
      --out "$RESULTS_DIR/${scheme}_tpt.csv" \
      "$RESULTS_DIR/${scheme}"_tpt.region*.csv
  else
    python3 "$SCRIPT_DIR/../aws/merge_region_results.py" csv \
      --out "$RESULTS_DIR/${scheme}_r${cell}.csv" \
      "$RESULTS_DIR/${scheme}_r${cell}".region*.csv
  fi
}

if [[ "${SKIP_WORKER_RESTART:-0}" != "1" ]]; then
  restart_workers_with_policy
fi

for scheme in $SCHEMES; do
  if [[ "$MODE" == "tpt" ]]; then
    if [[ -f "$RESULTS_DIR/${scheme}_tpt_summary.json" ]]; then
      echo "=== [$WORKLOAD] $scheme tpt: already done, skipping ==="
      continue
    fi
    echo "=== [$WORKLOAD] $scheme closed-loop throughput ==="
    flush_caches
    check_tunnels
    start_routers "$scheme"
    run_cell "$scheme" "tpt"
    stop_routers
  else
    for rate in $RATES; do
      # Resume support: a cell whose merged CSV already exists is done.
      if [[ -f "$RESULTS_DIR/${scheme}_r${rate}.csv" ]]; then
        echo "=== [$WORKLOAD] $scheme @ ${rate} req/s: already done, skipping ==="
        continue
      fi
      echo "=== [$WORKLOAD] $scheme @ ${rate} req/s ==="
      flush_caches
      check_tunnels
      start_routers "$scheme"
      run_cell "$scheme" "$rate"
      stop_routers
    done
  fi
done
echo "done — results in $RESULTS_DIR"
