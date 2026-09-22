#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# fetch_cloudping_snapshot.sh — pull a CloudPing.co WAN-latency snapshot.
#
# CloudPing.co (https://www.cloudping.co) measures TCP round-trip time
# between every pair of AWS regions every 6 h.  This script archives the
# raw JSON for several percentiles so paper/code can cite a fixed
# snapshot rather than a moving target.
#
# Usage
#   bash scripts/setup/fetch_cloudping_snapshot.sh
#       → datasets/wan_snapshots/cloudping_<UTC-DATE>_p{50,90,99}_1M.json
#
#   bash scripts/setup/fetch_cloudping_snapshot.sh --timeframe 1Y
#       → 1-year window instead of the 1-month default.
#
# After fetching, run scripts/setup/build_wan_matrix.py to convert one of the
# JSON files into the WAN_LATENCY_MS Python block that goes into
# ludo_sglang/config.py.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$PROJECT_ROOT/datasets/wan_snapshots"

TIMEFRAME="1M"
PERCENTILES=("p_50" "p_90" "p_99")
API_BASE="https://api.cloudping.co/latencies"

# ── Args ─────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeframe)   TIMEFRAME="$2";              shift 2 ;;
        --percentiles) IFS=',' read -ra PERCENTILES <<< "$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2";                shift 2 ;;
        -h|--help)
            sed -n '1,/^# ─\+$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

case "$TIMEFRAME" in
    1D|1W|1M|1Y) ;;
    *) echo "ERROR: --timeframe must be one of {1D, 1W, 1M, 1Y}" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR"
TODAY="$(date -u +%Y-%m-%d)"

echo "=== Fetching CloudPing snapshot ==="
echo "  date       : $TODAY (UTC)"
echo "  timeframe  : $TIMEFRAME"
echo "  percentiles: ${PERCENTILES[*]}"
echo "  out dir    : $OUT_DIR"
echo ""

for P in "${PERCENTILES[@]}"; do
    url="${API_BASE}?percentile=${P}&timeframe=${TIMEFRAME}"
    out="$OUT_DIR/cloudping_${TODAY}_${P}_${TIMEFRAME}.json"
    printf "  %-8s → %s ... " "$P" "$(basename "$out")"

    # -f : fail on HTTP >=400 ; -sS : silent except errors
    if ! curl -fsS "$url" -o "$out.tmp"; then
        echo "FAIL (curl $url)"
        rm -f "$out.tmp"
        exit 1
    fi

    # Validate it's parseable JSON with the expected schema.
    if ! python3 - "$out.tmp" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert "metadata" in d and "data" in d, "missing top-level keys"
assert "percentile" in d["metadata"], "missing metadata.percentile"
assert d["metadata"].get("unit") == "milliseconds", "unit must be milliseconds"
assert len(d["data"]) >= 30, f"too few source regions: {len(d['data'])}"
PY
    then
        echo "FAIL (invalid JSON / unexpected schema)"
        rm -f "$out.tmp"
        exit 1
    fi

    mv "$out.tmp" "$out"
    bytes=$(wc -c <"$out")
    n_src=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['data']))" "$out")
    echo "OK (${bytes} B, ${n_src} source regions)"
done

echo ""
echo "=== Done ==="
echo "Next step:"
echo "  python3 scripts/setup/build_wan_matrix.py \\"
echo "    $OUT_DIR/cloudping_${TODAY}_p_50_${TIMEFRAME}.json"
