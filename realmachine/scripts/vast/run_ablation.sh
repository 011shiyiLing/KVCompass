#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_ablation.sh — ablation grid (one mechanism removed per config).
#
# Configs (each removes ONE mechanism from the full system):
#   A0 full          — no change
#   A1 no-loadscore  — ABL_NO_LOADSCORE=1 (pure affinity: no q_view, no λ)
#   A2 no-xregion    — ABL_LOCAL_ONLY=1  (remote holders never surface;
#                                         cross-migration structurally dead)
#   A3 no-migration  — MIG_ENABLE=0      (both trigger kinds off)
#   A4 fixed-grid    — ABL_FIXED_GRID=1  (intra chunk grid for all pairs)
#   A5 tier-blind    — ABL_TIER_BLIND=1  (DRAM priced as VRAM in score)
#
# Main metric: closed-loop throughput (MODE=tpt), 32 clients/region,
# 90 s measured AFTER a 500-completions/region warmup, workers restarted
# (cold KV cache) before every run.  3 seeds (42/43/44), each seed uses
# its own trace file.
#
#   CONFIGS="A0 A1" SEEDS="42" bash scripts/vast/run_ablation.sh   # subset
#   MODE_EXTRA=openloop bash ...      # adds the A0/A2/A3 open-loop runs
#
# Resume-safe: a (config, seed) whose merged tpt summary exists is skipped.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIGS="${CONFIGS:-A0 A1 A2 A3 A4 A5}"
SEEDS="${SEEDS:-42 43 44}"
WORKLOAD="${WORKLOAD:-mixed}"
TRACE_PREFIX="${TRACE_PREFIX:-traces/mixed_mt50_6000_seed}"
EXP_ID="${EXP_ID:-p0_ablation}"
ABL_ROOT="${ABL_ROOT:-results/ablation}"
ROWS_CSV="$ABL_ROOT/ablation_results.csv"
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Open-loop settings (A0/A2/A3 latency supplement).
OPENLOOP_CONFIGS="${OPENLOOP_CONFIGS:-A0 A2 A3}"
OPENLOOP_RATE="${OPENLOOP_RATE:-16}"
OPENLOOP_DURATION="${OPENLOOP_DURATION:-120}"
OPENLOOP_WARMUP_REQ="${OPENLOOP_WARMUP_REQ:-167}"   # ×3 regions ≈ 500 global

abl_env() {
  case "$1" in
    A0) echo "" ;;
    A1) echo "ABL_NO_LOADSCORE=1" ;;
    A2) echo "ABL_LOCAL_ONLY=1" ;;
    A3) echo "MIG_ENABLE=0" ;;
    A4) echo "ABL_FIXED_GRID=1" ;;
    A5) echo "ABL_TIER_BLIND=1" ;;
    *)  echo "unknown config $1" >&2; return 1 ;;
  esac
}

cd "$REPO_ROOT"
mkdir -p "$ABL_ROOT"

# Up to MAX_PASSES passes over the grid: a cell that fails (worker OOM
# during restart, WAN outage, ...) is retried on the next pass; completed
# cells are skipped by the summary-file check.
MAX_PASSES="${MAX_PASSES:-3}"
for pass in $(seq 1 "$MAX_PASSES"); do
  remaining=0
  for seed in $SEEDS; do
    for cfg in $CONFIGS; do
      rdir="$ABL_ROOT/${cfg}_seed${seed}"
      if [[ -f "$rdir/ludo_tpt_summary.json" ]]; then
        [[ "$pass" == 1 ]] && echo "=== [$cfg seed=$seed] tpt already done, skipping ==="
        continue
      fi
      echo "=== [pass $pass] [$cfg seed=$seed] closed-loop tpt ==="
      start_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      if ! WORKLOAD="$WORKLOAD" TRACE="${TRACE_PREFIX}${seed}.jsonl" \
        MODE=tpt SCHEMES=ludo SEED="$seed" \
        WARMUP_REQUESTS=500 \
        ROUTER_EXTRA_ENV="$(abl_env "$cfg")" \
        RESULTS_DIR="$rdir" \
        bash "$SCRIPT_DIR/run_multiregion_vast.sh" \
          2>&1 | tee "$rdir.tpt.log" | tail -4; then
        echo "WARN: [$cfg seed=$seed] failed on pass $pass — will retry"
        remaining=$((remaining+1))
        continue
      fi
      python3 "$SCRIPT_DIR/collect_result_row.py" \
        --out "$ROWS_CSV" --exp-id "$EXP_ID" --config "$cfg" \
        --workload "${WORKLOAD}_mt50" --seed "$seed" \
        --run-start-time "$start_ts" --git-commit "$GIT_COMMIT" \
        --warmup "500req/region" --duration-s 90 \
        --tpt-summary "$rdir/ludo_tpt_summary.json"
    done
  done
  [[ "$remaining" == 0 ]] && break
  echo "=== pass $pass done, $remaining cell(s) to retry ==="
done

if [[ "${MODE_EXTRA:-}" == "openloop" ]]; then
  for cfg in $OPENLOOP_CONFIGS; do
    seed=42
    rdir="$ABL_ROOT/${cfg}_seed${seed}_openloop"
    if [[ -f "$rdir/ludo_r${OPENLOOP_RATE}.csv" ]]; then
      echo "=== [$cfg openloop] already done, skipping ==="
      continue
    fi
    echo "=== [$cfg seed=$seed] open-loop @ ${OPENLOOP_RATE} rps ==="
    start_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    WORKLOAD="$WORKLOAD" TRACE="${TRACE_PREFIX}${seed}.jsonl" \
      MODE=sweep SCHEMES=ludo SEED="$seed" RATES="$OPENLOOP_RATE" \
      WARMUP_S=0 DURATION_PER_RATE="$OPENLOOP_DURATION" \
      WARMUP_REQUESTS_OPEN="$OPENLOOP_WARMUP_REQ" \
      ROUTER_EXTRA_ENV="$(abl_env "$cfg")" \
      RESULTS_DIR="$rdir" \
      bash "$SCRIPT_DIR/run_multiregion_vast.sh" \
        2>&1 | tee "$rdir.openloop.log" | tail -4
    python3 "$SCRIPT_DIR/collect_result_row.py" \
      --out "$ROWS_CSV" --exp-id "${EXP_ID}_openloop" --config "$cfg" \
      --workload "${WORKLOAD}_mt50" --seed "$seed" \
      --run-start-time "$start_ts" --git-commit "$GIT_COMMIT" \
      --warmup "${OPENLOOP_WARMUP_REQ}req/region" \
      --duration-s "$OPENLOOP_DURATION" \
      --openloop-csv "$rdir/ludo_r${OPENLOOP_RATE}.csv"
  done
fi

echo "done — rows in $ROWS_CSV"
