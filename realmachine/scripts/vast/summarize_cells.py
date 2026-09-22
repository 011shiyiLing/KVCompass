#!/usr/bin/env python3
"""summarize_cells.py — live per-cell digest of a sweep results dir.

Prints, per (scheme, rate): n, fail%, effective-TTFT p50/p99, E2E p50,
observed service throughput (tok/s over the measured window), and the
real cache-hit rate (SGLang cached_tokens > template prefix).
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

TEMPLATE_PREFIX = 30      # chat-template tokens counted as cached on any hit


def cell_stats(path: Path) -> dict | None:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    ok = [r for r in rows if r.get("ok") == "1"]
    if not ok:
        return {"n": len(rows), "fail": 1.0}
    eff = sorted(float(r["ttft_ms"]) + float(r["router_overhead_ms"] or 0)
                 for r in ok)
    e2e = sorted(float(r["e2e_ms"]) + float(r["router_overhead_ms"] or 0)
                 for r in ok)
    toks = sum(int(r["prompt_tokens"] or 0) + int(r["completion_tokens"] or 0)
               for r in ok)
    span = (max(float(r["send_s"]) for r in ok)
            - min(float(r["send_s"]) for r in ok)) or 1.0
    hits = sum(1 for r in ok
               if int(r["cached_tokens"] or 0) > TEMPLATE_PREFIX + 20)
    xreg = sum(1 for r in ok if r["origin_region"] != r["worker_region"])
    q = lambda v, p: v[min(len(v) - 1, int(len(v) * p))]
    return {
        "n": len(rows), "fail": (len(rows) - len(ok)) / len(rows),
        "ttft_p50": q(eff, 0.50), "ttft_p99": q(eff, 0.99),
        "e2e_p50": q(e2e, 0.50),
        "tok_s": toks / span,
        "hit": hits / len(ok), "xreg": xreg / len(ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()

    table: dict[str, dict[float, dict]] = defaultdict(dict)
    for p in sorted(args.results_dir.glob("*_r*.csv")):
        m = re.match(r"(.+)_r([0-9.]+)\.csv$", p.name)
        if not m or ".region" in p.name:
            continue
        s = cell_stats(p)
        if s:
            table[m.group(1)][float(m.group(2))] = s

    hdr = (f'{"scheme":18s} {"rate":>5s} {"n":>4s} {"fail%":>6s} '
           f'{"TTFT p50":>9s} {"TTFT p99":>9s} {"E2E p50":>8s} '
           f'{"tok/s":>7s} {"hit%":>5s} {"xreg%":>6s}')
    print(hdr)
    print("-" * len(hdr))
    for scheme in sorted(table):
        for rate in sorted(table[scheme]):
            s = table[scheme][rate]
            if s.get("fail") == 1.0:
                print(f"{scheme:18s} {rate:5g}  ALL FAILED (n={s['n']})")
                continue
            print(f"{scheme:18s} {rate:5g} {s['n']:4d} {s['fail']*100:5.1f}% "
                  f"{s['ttft_p50']:8.0f}m {s['ttft_p99']:8.0f}m "
                  f"{s['e2e_p50']:7.0f}m {s['tok_s']:7.0f} "
                  f"{s['hit']*100:4.0f}% {s['xreg']*100:5.1f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
