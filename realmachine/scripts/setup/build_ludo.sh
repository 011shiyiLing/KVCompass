#!/usr/bin/env bash
# build_ludo.sh — compile the pybind11 Ludo extension against deps/smash.
#
# Requirements (Ubuntu/Debian):
#   sudo apt-get install build-essential libssl-dev python3-dev
#   pip install pybind11 setuptools
#
# After this script succeeds, src/ludo/ludo_binding.*.so (and the 64-bit
# variant) exist and `import ludo_binding` works from src/ludo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"      # realmachine/
LUDO_BINDING_DIR="$PROJECT_ROOT/src/ludo"

echo "=== Building Ludo pybind11 extension ==="
echo "Project root : $PROJECT_ROOT"
echo "Build dir    : $LUDO_BINDING_DIR"

# Verify the vendored Smash Ludo source is present
if [[ ! -f "$PROJECT_ROOT/deps/smash/Ludo/ludo_cp_dp.h" ]]; then
    echo "ERROR: Smash Ludo source not found at $PROJECT_ROOT/deps/smash"
    exit 1
fi

python3 -c "import pybind11" 2>/dev/null || {
    echo "ERROR: pybind11 not found.  Run: pip install pybind11"; exit 1; }

cd "$LUDO_BINDING_DIR"
rm -rf build/ ludo_binding*.so
python3 setup.py build_ext --inplace

SO_FILE=$(ls ludo_binding.*.so 2>/dev/null | head -1)
[[ -n "$SO_FILE" ]] || { echo "ERROR: no .so produced."; exit 1; }
echo "=== Build successful: $LUDO_BINDING_DIR/$SO_FILE ==="

echo "Smoke test ..."
python3 - <<'PYEOF'
import sys; sys.path.insert(0, ".")
import ludo_binding as lb
cp = lb.LudoControlPlane(1000)
cp.insert(42, 1)
dp = lb.LudoDataPlane(); dp.deserialize(cp.serialize_dp())
assert dp.lookup(42) is not None
print("smoke test OK: native Ludo control/data plane round-trip")
PYEOF
