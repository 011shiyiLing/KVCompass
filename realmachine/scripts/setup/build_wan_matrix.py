#!/usr/bin/env python3
"""
build_wan_matrix.py — turn a CloudPing.co JSON snapshot into the
WAN_LATENCY_MS np.array() block consumed by ludo_sglang/config.py.

CloudPing returns TCP round-trip time (ms) between every pair of AWS
regions.  The simulator and router treat WAN as a one-way propagation
delay, so we (a) divide by 2 and (b) symmetrise by averaging A→B and
B→A so the matrix is exactly symmetric (router code does not assume
asymmetry).

Usage
-----
  # Default 8 regions (matches §3 simulation setup)
  python3 scripts/setup/build_wan_matrix.py \\
      datasets/wan_snapshots/cloudping_2026-04-27_p_50_1M.json

  # Custom region list, write straight into a .py file
  python3 scripts/setup/build_wan_matrix.py snap.json \\
      --regions us-east-1,eu-west-1,ap-northeast-1 \\
      --out wan_block.py

The output is a copy-pasteable Python block.  Paste it over the
existing WAN_LATENCY_MS / DEFAULT_REGIONS definitions in
ludo_sglang/config.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Eight AWS regions used by the §3 simulation.  Each is the closest
# real AWS region to one of the eight geographic regions in the WAN
# matrix (Virginia / Oregon / Ireland / Frankfurt / Tokyo / Singapore /
# Mumbai / São Paulo).
DEFAULT_REGIONS = [
    "us-east-1",        # N. Virginia
    "us-west-2",        # Oregon
    "eu-west-1",        # Ireland
    "eu-central-1",     # Frankfurt
    "ap-northeast-1",   # Tokyo
    "ap-southeast-1",   # Singapore
    "ap-south-1",       # Mumbai
    "sa-east-1",        # São Paulo
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("snapshot", type=Path,
                   help="CloudPing JSON snapshot from fetch_cloudping_snapshot.sh")
    p.add_argument("--regions",
                   default=",".join(DEFAULT_REGIONS),
                   help="Comma-separated AWS region IDs (default: 8 §3 regions)")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the Python block here instead of stdout.")
    p.add_argument("--no-symmetrize", action="store_true",
                   help="Keep A→B and B→A asymmetric (default: average them).")
    return p.parse_args()


def build_matrix(snapshot_data: dict, regions: list[str],
                 symmetrize: bool = True) -> np.ndarray:
    """Return an N×N one-way latency matrix (ms)."""
    n = len(regions)
    M = np.zeros((n, n), dtype=float)
    for i, src in enumerate(regions):
        if src not in snapshot_data:
            raise KeyError(f"region {src!r} not in snapshot 'data' map")
        row = snapshot_data[src]
        for j, dst in enumerate(regions):
            if i == j:
                continue                        # diagonal stays 0
            if dst not in row:
                raise KeyError(f"snapshot[{src!r}] missing destination {dst!r}")
            M[i, j] = row[dst] / 2.0            # RTT → one-way
    if symmetrize:
        M = 0.5 * (M + M.T)
    return M


def render_python_block(M: np.ndarray, regions: list[str],
                        snapshot_path: Path, metadata: dict,
                        symmetrized: bool) -> str:
    n = len(regions)
    lines: list[str] = []
    lines.append("# ── WAN_LATENCY_MS — auto-generated, do not hand-edit. ─────────────────")
    lines.append("# Source : CloudPing.co inter-region TCP RTT")
    lines.append(f"#          https://api.cloudping.co/latencies")
    lines.append(f"#          percentile = {metadata.get('percentile')!r}, "
                 f"timeframe = {metadata.get('timeframe')!r}, "
                 f"unit = {metadata.get('unit')!r}")
    lines.append(f"# Snapshot file : {snapshot_path.name}")
    lines.append(f"# Conversion    : RTT / 2 → one-way ms"
                 f"{', then symmetrised (A→B + B→A) / 2' if symmetrized else ''}")
    lines.append(f"# Region order  : {regions}")
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("DEFAULT_REGIONS = [")
    for r in regions:
        lines.append(f"    {r!r},")
    lines.append("]")
    lines.append("")

    # Header line: index → region legend (avoids alignment headaches when
    # region names have ambiguous abbreviations e.g. ap-south vs ap-southeast).
    legend_pairs = [f"[{i}] {r}" for i, r in enumerate(regions)]
    half = (len(legend_pairs) + 1) // 2
    lines.append("# Column / row order:")
    for left, right in zip(legend_pairs[:half],
                           legend_pairs[half:] + [""] * (half - (len(legend_pairs) - half))):
        lines.append(f"#   {left:<22}  {right}")
    lines.append("WAN_LATENCY_MS = np.array([")
    for i in range(n):
        cells = ", ".join(f"{M[i, j]:6.1f}" for j in range(n))
        lines.append(f"    [{cells}],   # [{i}] from {regions[i]}")
    lines.append("], dtype=float)")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    if len(set(regions)) != len(regions):
        print(f"ERROR: duplicate region in --regions: {regions}", file=sys.stderr)
        return 1
    if len(regions) < 2:
        print(f"ERROR: need at least 2 regions, got {regions}", file=sys.stderr)
        return 1

    if not args.snapshot.exists():
        print(f"ERROR: snapshot not found: {args.snapshot}", file=sys.stderr)
        return 1

    snap = json.loads(args.snapshot.read_text())
    if "data" not in snap or "metadata" not in snap:
        print(f"ERROR: snapshot has unexpected schema (missing 'data' / 'metadata')",
              file=sys.stderr)
        return 1

    try:
        M = build_matrix(snap["data"], regions, symmetrize=not args.no_symmetrize)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Hint: run with --regions to see which AWS region IDs are present:",
              file=sys.stderr)
        print(f"  available = {sorted(snap['data'].keys())}", file=sys.stderr)
        return 1

    block = render_python_block(M, regions, args.snapshot, snap["metadata"],
                                symmetrized=not args.no_symmetrize)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(block)
        print(f"  wrote {len(block)} bytes → {args.out}", file=sys.stderr)
        # Also print a one-line summary to stderr for sanity.
        print(f"  matrix shape = {M.shape}, "
              f"min off-diag = {M[M > 0].min():.1f} ms, "
              f"max = {M.max():.1f} ms", file=sys.stderr)
    else:
        print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
