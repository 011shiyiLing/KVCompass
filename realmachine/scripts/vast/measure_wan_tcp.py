#!/usr/bin/env python3
"""measure_wan_tcp.py — measure the real inter-box WAN matrix (Vast
tunnel-mesh layout).

From each box, issue keep-alive HTTP GETs THROUGH the SSH tunnel to a
worker on every other box (127.0.0.1:40000+b*100).  The first request
warms the tunnel channel and is discarded; each subsequent request on
the persistent connection costs ≈ one WAN RTT (+ ~1 ms service time).
Median over N requests → RTT; one-way = RTT/2, symmetrised.  Writes the
WAN_MATRIX_FILE JSON that scoring and the latency-levelled cuts consume.

Driven by measure_wan.sh (which flattens the hosts.env arrays).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

# Raw TCP handshake to the peer box's public SSH port — the clean
# network RTT.  (HTTP probes through the tunnel measure server-side
# processing too — SGLang's /health even sleeps ~1 s — while the tunnel
# itself was verified to add ≈0 on top of this handshake RTT.)
PROBE_PY = r"""
import socket, time, statistics, sys
host, port, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ts = []
for _ in range(n):
    t0 = time.monotonic()
    s = socket.create_connection((host, port), timeout=10)
    ts.append((time.monotonic() - t0) * 1000.0)
    s.close()
    time.sleep(0.1)
print(statistics.median(ts) if ts else -1)
"""


def env_array(name: str) -> list[str]:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} not set — run via scripts/vast/measure_wan.sh")
    return v.split()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=15)
    ap.add_argument("--out", default="wan_matrix_real.json")
    args = ap.parse_args()

    ips = env_array("BOX_IP_FLAT")
    ssh_ports = env_array("BOX_SSH_PORT_FLAT")
    regions = env_array("REGION_NAMES_FLAT")
    ssh_user = os.environ.get("SSH_USER", "root")
    ssh_key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))
    n = len(ips)

    rtt = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Program travels via stdin (`python3 -`) — quoting survives
            # no shell round-trip that way.
            cmd = ["ssh", "-i", ssh_key, "-p", ssh_ports[i],
                   "-o", "StrictHostKeyChecking=no",
                   f"{ssh_user}@{ips[i]}",
                   f"python3 - {ips[j]} {ssh_ports[j]} {args.count}"]
            r = subprocess.run(cmd, input=PROBE_PY, capture_output=True,
                               text=True, timeout=300)
            out = r.stdout.strip()
            if not out:
                raise SystemExit(
                    f"probe {regions[i]} -> {regions[j]} failed "
                    f"(tunnels up? workers up?):\n{r.stderr[-400:]}")
            rtt[i][j] = float(out)
            print(f"  {regions[i]} -> {regions[j]}: RTT {rtt[i][j]:.1f} ms")

    one_way = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                one_way[i][j] = round((rtt[i][j] + rtt[j][i]) / 4.0, 1)

    doc = {"regions": regions, "one_way_ms": one_way,
           "source": "measure_wan_tcp.py (TCP handshake to peer SSH port, "
                     "RTT/2, symmetrised)"}
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwrote {args.out}:")
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
