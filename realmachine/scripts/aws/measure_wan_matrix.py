#!/usr/bin/env python3
"""measure_wan_matrix.py — build the measured 3-region WAN matrix.

SSHes into each region's router node, pings every other region's router
node, and writes the symmetrised one-way latency matrix as the JSON that
``WAN_MATRIX_FILE`` consumes.  Run from the control machine after
`source scripts/aws/hosts.env`.

Usage:
    python scripts/aws/measure_wan_matrix.py \
        --hosts "$ROUTER_HOSTS_CSV" \
        --regions "us-east-1,eu-central-1,ap-northeast-1" \
        --ssh-user "$SSH_USER" --ssh-key "$SSH_KEY" \
        --out wan_matrix_real.json

(ROUTER_HOSTS_CSV = the three router IPs comma-separated.)
Then push the file to every router node and set
    WAN_MATRIX_FILE=~/KVCompass/realmachine/wan_matrix_real.json
in the experiment environment.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess


def ping_rtt_ms(ssh_user: str, ssh_key: str, src_host: str,
                dst_host: str, count: int = 10) -> float:
    """Median RTT (ms) from src to dst, measured on src via SSH."""
    cmd = ["ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no",
           f"{ssh_user}@{src_host}",
           f"ping -c {count} -i 0.2 -q {dst_host}"]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=120).stdout
    m = re.search(r"= [\d.]+/([\d.]+)/", out)   # min/avg/max/mdev
    if not m:
        raise RuntimeError(f"ping {src_host} -> {dst_host} failed:\n{out}")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", required=True,
                    help="Comma-separated router host/IP per region.")
    ap.add_argument("--regions", required=True,
                    help="Comma-separated region names (same order).")
    ap.add_argument("--ssh-user", default="ubuntu")
    ap.add_argument("--ssh-key", required=True)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out", default="wan_matrix_real.json")
    args = ap.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    assert len(hosts) == len(regions), "hosts and regions length mismatch"
    n = len(hosts)

    rtt = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rtt[i][j] = ping_rtt_ms(args.ssh_user, args.ssh_key,
                                    hosts[i], hosts[j], args.count)
            print(f"  {regions[i]} -> {regions[j]}: RTT {rtt[i][j]:.1f} ms")

    # Symmetrise, then RTT/2 -> one-way (same convention as CloudPing
    # snapshots in ludo_sglang/wan_matrix_p*.py).
    one_way = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                one_way[i][j] = round((rtt[i][j] + rtt[j][i]) / 4.0, 1)

    doc = {"regions": regions, "one_way_ms": one_way,
           "source": "scripts/aws/measure_wan_matrix.py (ping, RTT/2, symmetrised)"}
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwrote {args.out}:")
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
