#!/usr/bin/env bash
# run_p2_scale.sh — P2 memory benchmark, scale extension to E = 1e8 / 1e9.
#
# Tiers:
#   SHA tier   (E ≤ 1e7)  : all 5 structures, SHA-256 keys — methodology
#                           parity with the original testbed runs.
#   fast tier  (E = 1e8)  : --fast-hash (validated equivalent: self_bytes
#                           and load factor identical to SHA at 1e6/1e7);
#                           radix skipped (≈70 GB — does not fit, cell is
#                           extrapolated downstream).
#   1e9 tier              : planetserve measured (small); ludo_view via
#                           ludo_view_alloc (exact view layout, sized with
#                           the distinct-key ratio measured at 1e8);
#                           ludo_cp / hashdir / radix extrapolated.
#
# Every process is guarded by --mem-cap-gb ${MEM_CAP:-13}: a cell that
# exceeds the cap emits an OOM_ABORT row instead of swapping the machine.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# macOS: pick up Homebrew OpenSSL if present (Linux relies on system paths)
OPENSSL_FLAGS=""
if command -v brew >/dev/null 2>&1; then
  P="$(brew --prefix openssl 2>/dev/null || true)"
  [[ -n "$P" ]] && OPENSSL_FLAGS="-I$P/include -L$P/lib"
fi
OUT="${OUT:-$ROOT/results_tmp/p2_mem_scale.csv}"
TOKENS="${TOKENS:-$ROOT/results_tmp/p2_tokens.bin}"
MEM_CAP="${MEM_CAP:-13}"
B="$HERE/p2_mem"

echo "structure,E_view,entries_native,seqs,tenants,rss_delta_bytes,self_bytes,rss_bits_per_entry,self_bits_per_entry,ludo_lf,entry_semantics,hash,tier" > "$OUT"

run_cell() {  # structure E extra_flags... ; appends hash/tier columns
  local S="$1" E="$2"; shift 2
  local hash=sha tier=measured
  for a in "$@"; do [[ "$a" == "--fast-hash" ]] && hash=fast; done
  echo "=== $S E=$E ($hash) ===" >&2
  local row
  row="$("$B" --structure "$S" --E "$E" --tokens "$TOKENS" \
              --mem-cap-gb "$MEM_CAP" "$@" 2>/dev/null)" || tier=oom_abort
  [[ -n "$row" ]] && echo "$row,$hash,$tier" >> "$OUT"
  echo "$row" >&2
}

# ── SHA tier: parity with the original run_p2.sh grid ──────────────────
# planetserve_eq = HRT carrying view-equivalent information (full 64-bit
# chunk hashes, depth to the 4096-tok cut, holders per (seq,cut)); the
# lossy gate 'planetserve' stays for reference.
for E in 100000 1000000 10000000; do
  for S in ludo_view ludo_cp hashdir planetserve planetserve_eq radix; do
    run_cell "$S" "$E"
  done
done

# ── fast tier: E = 1e8 (radix does not fit — extrapolated) ─────────────
for S in ludo_view ludo_cp hashdir planetserve; do
  run_cell "$S" 100000000 --fast-hash
done

# ── 1e9 tier ────────────────────────────────────────────────────────────
run_cell planetserve 1000000000 --fast-hash

# ludo_view_alloc@1e9: othello arrays sized by the distinct-key count,
# projected from the 1e8 ludo_view row's load factor (distinct = lf ×
# capacity(1e8); capacity(E) = 0.95·4·nb(E)).
ALLOC_KEYS=$(/usr/bin/python3 - "$OUT" <<'PY'
import csv, sys
row = next(r for r in csv.DictReader(open(sys.argv[1]))
           if r['structure'] == 'ludo_view'
           and abs(int(r['E_view']) - 100_000_000) < 100)
lf = float(row['ludo_lf'])
nb = 64
while nb * 0.95 * 4 < 100_000_000:
    nb <<= 1
print(int(lf * nb * 0.95 * 4 / 100_000_000 * 1_000_000_000))
PY
)
echo "=== ludo_view_alloc E=1e9 (alloc_keys=$ALLOC_KEYS) ===" >&2
run_cell ludo_view_alloc 1000000000 --fast-hash --alloc-keys "$ALLOC_KEYS"

echo "done -> $OUT"
