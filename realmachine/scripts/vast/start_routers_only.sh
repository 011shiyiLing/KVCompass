#!/usr/bin/env bash
# start_routers_only.sh — boot the 3 region routers (ludo, full system)
# WITHOUT running any workload. Used by the P1 propagation measurement.
#   bash scripts/vast/start_routers_only.sh [stop]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hosts.env"
N_REGIONS=${#BOX_IP[@]}
ROUTER_PORT=8001

SSHB() { ssh -i "$SSH_KEY" -p "${BOX_SSH_PORT[$1]}" \
             -o StrictHostKeyChecking=no "$SSH_USER@${BOX_IP[$1]}" "${@:2}"; }

worker_urls_for_region() {
  local view="$1" out="" b i
  for b in $(seq 0 $((N_REGIONS-1))); do
    for i in $(seq 0 $((${GPUS_PER_BOX[$b]}-1))); do
      if [[ "$b" == "$view" ]]; then out+="http://127.0.0.1:$((30000+i)),"
      else out+="http://127.0.0.1:$((40000+b*100+i)),"; fi
    done
  done
  echo "${out%,}"
}
region_router_urls() {
  local view="$1" out="" i
  for i in $(seq 0 $((N_REGIONS-1))); do
    if [[ "$i" == "$view" ]]; then out+="${i}:http://127.0.0.1:${ROUTER_PORT},"
    else out+="${i}:http://127.0.0.1:$((40000+i*100+99)),"; fi
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

stop_routers() {
  local r
  for r in $(seq 0 $((N_REGIONS-1))); do
    SSHB "$r" "P=\$(printf '%s.router.ludo_router' ludo_sglang); \
      pkill -f \"\$P\" 2>/dev/null; true" || true
  done
}

if [[ "${1:-start}" == "stop" ]]; then stop_routers; echo stopped; exit 0; fi

stop_routers; sleep 1
for r in $(seq 0 $((N_REGIONS-1))); do
  SSHB "$r" "cd $REPO_DIR; \
    SCHEME=ludo \
    SGLANG_WORKER_URLS='$(worker_urls_for_region "$r")' \
    WORKER_REGIONS='$WORKER_REGIONS' \
    WAN_EMULATION=0 WAN_MATRIX_FILE=$REPO_DIR/wan_matrix_real.json \
    ROUTER_REGION=$r ROUTER_ID=$r \
    ROUTER_HOST=0.0.0.0 ROUTER_PORT=$ROUTER_PORT \
    PEER_ROUTER_URLS='$(peer_urls_for "$r")' \
    REGION_ROUTER_URLS='$(region_router_urls "$r")' \
    C_PREFILL_MS_PER_TOKEN_PRIOR=$C_PREFILL_MS_PER_TOKEN_PRIOR \
    PCIE_RELOAD_MS_PER_TOKEN=$PCIE_RELOAD_MS_PER_TOKEN \
    nohup python3 -m ludo_sglang.router.ludo_router \
      > router_prop.log 2>&1 < /dev/null & echo \"router $r started\""
done
echo "waiting for routers ..."
for r in $(seq 0 $((N_REGIONS-1))); do
  for t in $(seq 1 90); do
    SSHB "$r" "curl -sf -m 2 http://127.0.0.1:${ROUTER_PORT}/alive" \
      >/dev/null 2>&1 && { echo "router $r alive"; break; }
    [[ "$t" == 90 ]] && { echo "ERROR router $r"; exit 1; }
    sleep 1
  done
done
