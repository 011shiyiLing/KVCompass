#!/usr/bin/env bash
# run_p2.sh — build + run the P2 memory benchmark on a box.
# One process per (structure, E): clean RSS deltas.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$ROOT/results_tmp/p2_mem.csv}"
TOKENS="${TOKENS:-$ROOT/results_tmp/p2_tokens.bin}"

g++ -std=c++17 -O3 -march=native -DNDEBUG -ffast-math \
    -Wno-unused-result -Wno-deprecated-declarations \
    -I"$ROOT/deps/smash" \
    "$HERE/p2_mem.cpp" \
    "$ROOT/deps/smash/utils/farmhash/farmhash.cc" \
    "$ROOT/deps/smash/utils/hashutil.cc" \
    "$ROOT/src/ludo/smash_runtime_stubs.cpp" \
    -lssl -lcrypto -lpthread -o "$HERE/p2_mem"
echo "built"

echo "structure,E_view,entries_native,seqs,tenants,rss_delta_bytes,self_bytes,rss_bits_per_entry,self_bits_per_entry,ludo_lf,entry_semantics" > "$OUT"
for E in 100000 1000000 10000000; do
  for S in ludo_view ludo_cp hashdir planetserve radix; do
    echo "=== $S E=$E ==="
    "$HERE/p2_mem" --structure "$S" --E "$E" --tokens "$TOKENS" >> "$OUT"
    tail -1 "$OUT"
  done
done
echo "done -> $OUT"
