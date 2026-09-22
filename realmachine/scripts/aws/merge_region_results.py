#!/usr/bin/env python3
"""merge_region_results.py — combine the 3 per-region replayer outputs
of one experiment cell into the single files plot_eval_grid.py reads.

  csv mode : header-aware concat of <cell>.region*.csv -> <cell>.csv
  tpt mode : closed-loop summaries — total throughput is the SUM of the
             per-region tokens over the common window.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def merge_csv(out: Path, parts: list[Path]) -> None:
    rows, fields = [], None
    for p in parts:
        with open(p) as f:
            rd = csv.DictReader(f)
            fields = fields or rd.fieldnames
            rows.extend(rd)
    rows.sort(key=lambda r: float(r.get("send_s") or 0))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"merged {len(parts)} files, {len(rows)} rows -> {out}")


def merge_tpt(out: Path, parts: list[Path]) -> None:
    docs = [json.load(open(p)) for p in parts]
    merged = {
        "mode": "closed_loop_merged",
        "regions": len(docs),
        "n_clients": sum(d.get("n_clients", 0) for d in docs),
        "n_ok": sum(d.get("n_ok", 0) for d in docs),
        "n_failed": sum(d.get("n_failed", 0) for d in docs),
        "tokens_total": sum(d.get("tokens_total", 0) for d in docs),
        "tpt_total_tok_s": sum(d.get("tpt_total_tok_s", 0.0) for d in docs),
        "per_region": docs,
    }
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"merged tpt -> {out}: {merged['tpt_total_tok_s']:.0f} tok/s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["csv", "tpt"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("parts", nargs="+", type=Path)
    a = ap.parse_args()
    (merge_csv if a.mode == "csv" else merge_tpt)(a.out, a.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
