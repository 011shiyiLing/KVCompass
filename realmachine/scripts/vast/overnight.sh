#!/usr/bin/env bash
# overnight.sh — unattended pipeline: lmsys sweep → closed-loop tpt for
# mixed/wildchat/lmsys → eval_grid figure.  Every phase auto-resumes
# (completed cells are skipped by the driver) and auto-retries through
# network blips; remote stray processes are cleaned before each retry.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hosts.env"

LOG="${OVERNIGHT_LOG:-$REPO/overnight_status.log}"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

clean_strays() {
  local b
  for b in 0 1 2; do
    ssh -i "$SSH_KEY" -p "${BOX_SSH_PORT[$b]}" -o StrictHostKeyChecking=no \
      -o ConnectTimeout=20 "$SSH_USER@${BOX_IP[$b]}" \
      "P=\$(printf '%s.router.ludo_router' ludo_sglang); pkill -f \"\$P\" 2>/dev/null; \
       Q=\$(printf '%s_replayer' trace); pkill -f \"\$Q\" 2>/dev/null; true" \
      >/dev/null 2>&1 || true
  done
}

run_phase() {  # $1=name  $2...=env assignments for the driver
  local name="$1"; shift
  local attempt=1 max=40
  while true; do
    say "phase '$name' attempt $attempt"
    if env "$@" bash "$SCRIPT_DIR/run_multiregion_vast.sh" \
        >> "$REPO/overnight_${name}.log" 2>&1; then
      say "phase '$name' DONE"
      return 0
    fi
    say "phase '$name' attempt $attempt failed (see overnight_${name}.log)"
    clean_strays
    attempt=$((attempt+1))
    if [[ $attempt -gt $max ]]; then
      say "phase '$name' EXCEEDED $max attempts — moving on"
      return 1
    fi
    sleep 120
  done
}

say "=== overnight pipeline started ==="

# 1. LMSYS sweep (LRU already active from wildchat → skip worker restart)
run_phase lmsys_sweep \
  SKIP_WORKER_RESTART=1 WORKLOAD=lmsys TRACE=traces/lmsys_6000_seed42.jsonl \
  RATES="2 4 8 16 24 32 40" WARMUP_S=15 DURATION_PER_RATE=90

# 2. Closed-loop throughput — mixed first (needs LFU restart), then the
#    two LRU workloads (restart once for wildchat, skip for lmsys).
run_phase tpt_mixed \
  MODE=tpt WORKLOAD=mixed TRACE=traces/mixed_6000_seed42.jsonl \
  WARMUP_S=15 TPT_CLIENTS_PER_REGION=32 TPT_DURATION=90

run_phase tpt_wildchat \
  MODE=tpt WORKLOAD=wildchat TRACE=traces/wildchat_6000_seed42.jsonl \
  WARMUP_S=15 TPT_CLIENTS_PER_REGION=32 TPT_DURATION=90

run_phase tpt_lmsys \
  SKIP_WORKER_RESTART=1 MODE=tpt WORKLOAD=lmsys TRACE=traces/lmsys_6000_seed42.jsonl \
  WARMUP_S=15 TPT_CLIENTS_PER_REGION=32 TPT_DURATION=90

# 3. Figure — same 4×3 grid as the simulation's eval_grid.pdf.
say "rendering eval_grid ..."
python3 scripts/plotting/plot_eval_grid.py \
  --results-root results --box-rate 16 \
  --out results/figures/eval_grid.pdf >> "$LOG" 2>&1 \
  && say "figure written to results/figures/eval_grid.pdf" \
  || say "figure rendering FAILED (data intact; rerun plot manually)"

say "=== overnight pipeline COMPLETE ==="
