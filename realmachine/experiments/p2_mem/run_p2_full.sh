#!/usr/bin/env bash
# run_p2_full.sh — full-grid P2 memory benchmark on the 30 GB 5090 box.
# Measures as far out as RAM allows (--mem-cap-gb graceful OOM guard),
# extrapolating only where a cell exceeds the cap. FAITHFUL structures for
# the paper figure (ludo_view / hashdir / planetserve / radix); ludo_cp and
# ludo_view_alloc kept for the analytic reconciliation. --fast-hash at
# E >= 1e8 (matches the prior large-E rows). One process per (structure, E)
# for clean RSS deltas. Schema matches results_tmp/p2_mem_scale.csv (13
# cols: adds hash,tier).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# macOS: pick up Homebrew OpenSSL if present (Linux relies on system paths)
OPENSSL_FLAGS=""
if command -v brew >/dev/null 2>&1; then
  P="$(brew --prefix openssl 2>/dev/null || true)"
  [[ -n "$P" ]] && OPENSSL_FLAGS="-I$P/include -L$P/lib"
fi
OUT="${OUT:-$ROOT/results_tmp/p2_mem_scale_5090.csv}"
TOKENS="${TOKENS:-$ROOT/results_tmp/p2_tokens.bin}"
mkdir -p "$(dirname "$OUT")"
BIN="$HERE/p2_mem"
CAP_GB="${CAP_GB:-26}"          # leave headroom on 30 GB
FASTHASH_FROM=100000000         # use --fast-hash at E >= 1e8

# binary is already built; rebuild only if missing
if [[ ! -x "$BIN" ]]; then
  g++ -std=c++17 -O3 -march=native -DNDEBUG -ffast-math \
      -Wno-unused-result -Wno-deprecated-declarations \
      -I"$ROOT/deps/smash" $OPENSSL_FLAGS \
      "$HERE/p2_mem.cpp" \
      "$ROOT/deps/smash/utils/farmhash/farmhash.cc" \
      "$ROOT/deps/smash/utils/hashutil.cc" \
      "$ROOT/src/ludo/smash_runtime_stubs.cpp" \
      -lssl -lcrypto -lpthread -o "$BIN"
  echo "built"
fi

echo "structure,E_view,entries_native,seqs,tenants,rss_delta_bytes,self_bytes,rss_bits_per_entry,self_bits_per_entry,ludo_lf,entry_semantics,hash,tier" > "$OUT"

# per-structure E grids (skip cells that are pointless/guaranteed-OOM to
# save wall-clock; the cap still catches anything that slips through)
run_cell() {  # $1=structure  $2=E
  local S="$1" E="$2" FLAGS="" HASH="sha"
  if (( E >= FASTHASH_FROM )); then FLAGS="--fast-hash"; HASH="fast"; fi
  echo "=== $S E=$E ($HASH, cap ${CAP_GB}GB) ===" >&2
  local row
  row="$("$BIN" --structure "$S" --E "$E" --tokens "$TOKENS" \
                --mem-cap-gb "$CAP_GB" $FLAGS 2>>"$OUT.log")"
  local rc=$?
  # binary prints an OOM_ABORT row (rc=2) or a data row (rc=0)
  local TIER="measured"
  if [[ "$row" == *OOM_ABORT* || $rc -ne 0 ]]; then TIER="oom_abort"; fi
  if [[ -n "$row" ]]; then echo "${row},${HASH},${TIER}" >> "$OUT"; tail -1 "$OUT" >&2; fi
}

for E in 100000 1000000 10000000 100000000 1000000000; do
  run_cell ludo_view   "$E"
  run_cell hashdir     "$E"
  run_cell planetserve "$E"
done
# radix (token payload dominates) — only realistic up to ~1.6e7 on 30 GB
for E in 100000 1000000 10000000 100000000; do run_cell radix "$E"; done
# ludo maintenance-unit + exact view layout for the analytic reconciliation
for E in 100000 1000000 10000000 100000000 1000000000; do run_cell ludo_cp "$E"; done
for E in 10000000 100000000 1000000000; do run_cell ludo_view_alloc "$E"; done

echo "done -> $OUT" >&2