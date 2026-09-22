#!/usr/bin/env python3
"""collect_result_row.py — append standardized result rows to a CSV.

One row per (run, metric):
  exp_id, config, workload, seed, metric, value,
  run_start_time, git_commit, warmup, duration_s, n_failed, notes

Two input modes:
  --tpt-summary <merged json>   → metric rows: tpt_total_tok_s (+ per-region)
  --openloop-csv <merged csv>   → metric rows: ttft_mean/p50/p99,
                                  e2e_mean/p50/p99 (effective = +router_overhead_ms)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _pct(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method), stdlib-only."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


FIELDS = ["exp_id", "config", "workload", "seed", "metric", "value",
          "run_start_time", "git_commit", "warmup", "duration_s",
          "n_failed", "notes"]


def append_rows(out: Path, base: dict, metrics: list[tuple[str, float]],
                notes: str = "") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for metric, value in metrics:
            w.writerow({**base, "metric": metric, "value": value,
                        "notes": notes})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--run-start-time", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--warmup", required=True,
                    help="e.g. '500req/region' or '30s+167req/region'")
    ap.add_argument("--duration-s", required=True)
    ap.add_argument("--tpt-summary", type=Path, default=None)
    ap.add_argument("--openloop-csv", type=Path, default=None)
    a = ap.parse_args()

    base = {"exp_id": a.exp_id, "config": a.config, "workload": a.workload,
            "seed": a.seed, "run_start_time": a.run_start_time,
            "git_commit": a.git_commit, "warmup": a.warmup,
            "duration_s": a.duration_s, "n_failed": 0}

    if a.tpt_summary:
        doc = json.load(open(a.tpt_summary))
        base["n_failed"] = doc.get("n_failed", 0)
        notes = ""
        unreached = [d for d in doc.get("per_region", [])
                     if not d.get("warmup_reached", True)]
        if unreached:
            notes = f"warmup_not_reached_in_{len(unreached)}_region(s)"
        metrics = [("tpt_total_tok_s", doc["tpt_total_tok_s"]),
                   ("tokens_total", doc["tokens_total"])]
        for i, d in enumerate(doc.get("per_region", [])):
            metrics.append((f"tpt_region{i}_tok_s", d.get("tpt_total_tok_s")))
            metrics.append((f"measured_window_region{i}_s",
                            d.get("measured_window_s", d.get("wall_s"))))
        append_rows(a.out, base, metrics, notes)

    if a.openloop_csv:
        rows = [r for r in csv.DictReader(open(a.openloop_csv))]
        ok = [r for r in rows if r.get("ok") == "1"]
        base["n_failed"] = len(rows) - len(ok)
        eff_ttft = sorted(float(r["ttft_ms"]) +
                          float(r.get("router_overhead_ms") or 0)
                          for r in ok if r.get("ttft_ms"))
        eff_e2e = sorted(float(r["e2e_ms"]) +
                         float(r.get("router_overhead_ms") or 0)
                         for r in ok if r.get("e2e_ms"))
        metrics = []
        for name, arr in (("ttft", eff_ttft), ("e2e", eff_e2e)):
            if not arr:
                continue
            metrics += [(f"{name}_mean_ms", sum(arr) / len(arr)),
                        (f"{name}_p50_ms", _pct(arr, 50)),
                        (f"{name}_p99_ms", _pct(arr, 99))]
        metrics.append(("n_requests_ok", len(ok)))
        append_rows(a.out, base, metrics)

    print(f"appended -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
