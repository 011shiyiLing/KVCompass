#!/usr/bin/env bash
# Build p1_bench against the vendored Smash Ludo (deps/smash).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# macOS: pick up Homebrew OpenSSL if present (Linux relies on system paths)
OPENSSL_FLAGS=""
if command -v brew >/dev/null 2>&1; then
  P="$(brew --prefix openssl 2>/dev/null || true)"
  [[ -n "$P" ]] && OPENSSL_FLAGS="-I$P/include -L$P/lib"
fi
g++ -std=c++17 -O3 -march=native -DNDEBUG -ffast-math \
    -Wno-unused-result -Wno-deprecated-declarations \
    -I"$ROOT/deps/smash" $OPENSSL_FLAGS \
    "$HERE/p1_bench.cpp" \
    "$ROOT/deps/smash/utils/farmhash/farmhash.cc" \
    "$ROOT/deps/smash/utils/hashutil.cc" \
    "$ROOT/src/ludo/smash_runtime_stubs.cpp" \
    -lssl -lcrypto -lpthread \
    -o "$HERE/p1_bench"
echo "built -> $HERE/p1_bench"
