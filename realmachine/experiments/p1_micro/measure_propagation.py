#!/usr/bin/env python3
"""measure_propagation.py — P1(2c): advertise-event propagation latency,
worker → every edge view, intra-region and cross-region separately.

Run ON a box (each box measures its own destination view via loopback):

    python3 experiments/p1_micro/measure_propagation.py \
        --self-region 1 \
        --origin-router http://127.0.0.1:40099 --origin-region 0 \
        --local-router http://127.0.0.1:8001 \
        --n 200 --cut 256 --out prop_us_to_eu.csv

Method per trial: POST /internal/debug_advertise at the ORIGIN router
(fresh random token prefix, so the key is new), then poll the LOCAL
router's /internal/debug_lookup via loopback every --poll-ms until the
holder appears.  One-way propagation estimate:

    dt = t_visible_local - t_post_start_local - post_rtt/2

(post_rtt measured per trial from the advertise POST itself; the RTT/2
correction removes the harness's own WAN leg to the origin.)
Intra-region runs point --origin-router at the local router: post_rtt
is sub-ms and dt is the pure local-index visibility delay.
"""
from __future__ import annotations

import argparse
import csv
import random
import time

import requests


def one_trial(origin: str, local: str, rank: int, cut: int,
              poll_ms: float, timeout_s: float) -> tuple[float, float]:
    token_ids = [random.randrange(0, 128000) for _ in range(cut)]
    t0 = time.monotonic()
    r = requests.post(f"{origin}/internal/debug_advertise",
                      json={"token_ids": token_ids, "rank": rank},
                      timeout=30)
    r.raise_for_status()
    post_rtt_ms = (time.monotonic() - t0) * 1000.0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        q = requests.post(f"{local}/internal/debug_lookup",
                          json={"token_ids": token_ids, "cut": cut},
                          timeout=10).json()
        if q["holders"]:
            dt = (time.monotonic() - t0) * 1000.0 - post_rtt_ms / 2.0
            return dt, post_rtt_ms
        time.sleep(poll_ms / 1000.0)
    return float("nan"), post_rtt_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin-router", required=True)
    ap.add_argument("--local-router", default="http://127.0.0.1:8001")
    ap.add_argument("--origin-region", type=int, required=True)
    ap.add_argument("--self-region", type=int, required=True)
    ap.add_argument("--rank", type=int, default=None,
                    help="worker rank to advertise as (default: first "
                         "worker of origin region = region index heuristic)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cut", type=int, default=256)
    ap.add_argument("--poll-ms", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rank = a.rank
    if rank is None:
        rank = {0: 0, 1: 8, 2: 10}.get(a.origin_region, 0)

    kind = "intra" if a.origin_region == a.self_region else "cross"
    rows, lost = [], 0
    for i in range(a.n):
        dt, rtt = one_trial(a.origin_router, a.local_router, rank,
                            a.cut, a.poll_ms, a.timeout_s)
        if dt != dt:  # NaN
            lost += 1
            continue
        rows.append({"trial": i, "kind": kind,
                     "origin_region": a.origin_region,
                     "dest_region": a.self_region,
                     "prop_ms": round(dt, 3),
                     "post_rtt_ms": round(rtt, 3)})
        time.sleep(0.05)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    vals = sorted(r["prop_ms"] for r in rows)
    p = lambda q: vals[min(len(vals) - 1, int(len(vals) * q))]
    print(f"{kind} region{a.origin_region}->region{a.self_region}: "
          f"n={len(vals)} lost={lost} p50={p(0.5):.1f}ms p99={p(0.99):.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
