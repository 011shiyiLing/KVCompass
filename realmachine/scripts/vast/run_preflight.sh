#!/usr/bin/env bash
# run_preflight.sh — full-system (A0) closed-loop tpt on all 3 workloads,
# seed 42, 500-completions/region warmup, 90 s measured window.
# Purpose: confirm Mixed(mt50) is the reuse-favorable workload before P0.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
ROWS_CSV="results/preflight_batch2/preflight_results.csv"

run_one() {  # $1=workload  $2=trace
  local wl="$1" trace="$2"
  local rdir="results/preflight_batch2/${wl}"
  if [[ -f "$rdir/ludo_tpt_summary.json" ]]; then
    echo "=== [$wl] already done, skipping ==="
    return 0
  fi
  echo "=== [preflight $wl] full-system closed-loop tpt ==="
  local start_ts; start_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  WORKLOAD="$wl" TRACE="$trace" MODE=tpt SCHEMES=ludo SEED=42 \
    WARMUP_REQUESTS=500 RESULTS_DIR="$rdir" \
    bash "$SCRIPT_DIR/run_multiregion_vast.sh" 2>&1 | tail -3
  python3 "$SCRIPT_DIR/collect_result_row.py" \
    --out "$ROWS_CSV" --exp-id preflight --config A0-full \
    --workload "$wl" --seed 42 --run-start-time "$start_ts" \
    --git-commit "$GIT_COMMIT" --warmup "500req/region" --duration-s 90 \
    --tpt-summary "$rdir/ludo_tpt_summary.json"
}

run_one mixed    traces/mixed_mt50_6000_seed42.jsonl
run_one wildchat traces/wildchat_6000_seed42.jsonl
run_one lmsys    traces/lmsys_6000_seed42.jsonl
echo "preflight done — $ROWS_CSV"
