"""fp_microbench.py — structural false-positive rate of the keyless Ludo
data plane for never-inserted keys (paper Table 5, left half).

Builds a control plane sized for E entries, fills it to each load fraction,
serializes the data plane, then probes M never-inserted keys and counts how
often a non-empty locator is returned. Requires the compiled ludo_binding
(realmachine/scripts/setup/build_ludo.sh).

Usage: python3 fp_microbench.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "ludo"))
import ludo_binding as lb  # noqa: E402

E = 1_000_000          # table capacity (expected entries)
M = 100_000            # never-inserted probe keys
print(f"capacity E={E:,}  probes M={M:,}")
print(f"{'inserted':>10}{'load%':>8}{'FP-rate%':>10}")
for n in [200_000, 400_000, 600_000, 800_000, 950_000]:
    cp = lb.LudoControlPlane(E)
    for k in range(1, n + 1):
        cp.insert(k, 1)                       # non-zero value = occupied
    dp = lb.LudoDataPlane()
    dp.deserialize(cp.serialize_dp())
    base = 10**12                             # disjoint from inserted keys
    fp = sum(1 for i in range(M) if dp.lookup(base + i) is not None)
    print(f"{n:>10,}{100*n/E:>8.0f}{100*fp/M:>10.2f}")
