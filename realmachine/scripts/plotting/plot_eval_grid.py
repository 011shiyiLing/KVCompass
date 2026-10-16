"""plot_eval_grid.py — real-machine 4×3 master grid, layout/style-identical
to the simulation's the simulator's grid plotter
(figures/eval_grid.pdf).

Rows (metrics):   P99 TTFT vs rate · TTFT box · E2E-latency box · Throughput
Columns:          Mixed · WildChat · LMSYS
Panel captions:   "(a) Mixed, P99 TTFT." … row-major.

Data sources (all produced by trace_replayer.py):
  results_<workload>/<scheme>_r<rate>.csv          open-loop sweeps (rows 0-2)
  results_<workload>/<scheme>_tpt_summary.json     closed-loop throughput (row 3)
  (fallback: results_<workload>/<scheme>_tpt.csv — throughput recomputed
   from the CSV's send_s span)

Effective TTFT = ttft_ms + router_overhead_ms (SGLang prefill + whatever
lookup overhead the router injected; in real-WAN mode the propagation is
already inside ttft_ms and router_overhead_ms carries only the modelled
lookup cost — no double counting either way).

Usage:
  python scripts/plotting/plot_eval_grid.py \
      --results-root results \
      --box-rate 16 \
      --out figures/eval_grid.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style constants — copied verbatim from run_section3.py so the two
#    eval grids are visually interchangeable. ─────────────────────────────
ALL_SCHEMES = ["central_directory", "ludo", "radix_tree",
               "consistent_hash", "chord", "skywalker", "dynamo"]

LINE_COLORS = {
    "ludo":              "#985365",
    "radix_tree":        "#8a8c6f",
    "consistent_hash":   "#7e97ad",
    "kademlia":          "#b78D31",
    "chord":             "#b78D31",
    "skywalker":         "#6f9c8f",
    "dynamo":            "#9c7e97",
    "central_directory": "#7A6E5A",
}
LINE_FILL = {
    "ludo":              "#eba79c",
    "radix_tree":        "#a3b7a0",
    "consistent_hash":   "#ACBFEA",
    "kademlia":          "#FFDAA6",
    "chord":             "#FFDAA6",
    "skywalker":         "#b9d8cd",
    "dynamo":            "#d8c3d4",
    "central_directory": "#A5957D",
}
LINE_LABELS = {
    "ludo":              "KVCompass",
    "radix_tree":        "PlanetServe",
    "consistent_hash":   "Consistent Hash",
    "kademlia":          "Kademlia DHT",
    "chord":             "Chord DHT",
    "skywalker":         "SkyWalker",
    "dynamo":            "Dynamo",
    "central_directory": "Central Directory",
}
MARKERS = {"central_directory": "D", "ludo": "o", "radix_tree": "s",
           "consistent_hash": "^", "kademlia": "v", "chord": "P",
           "skywalker": "X", "dynamo": "*"}
SCHEME_HATCHES = {
    "ludo": "|||", "radix_tree": "\\\\", "consistent_hash": "xxx",
    "kademlia": "...", "chord": "///", "skywalker": "++",
    "dynamo": "--", "central_directory": "ooo",
}
SCHEME_LINESTYLES = {s: "-" for s in LINE_COLORS}

ORDER_LAST = [s for s in ALL_SCHEMES if s != "ludo"] + ["ludo"]
XTICK_2LINE = {
    "central_directory": "Central\nDirectory",
    "radix_tree":        "Planet\nServe",
    "consistent_hash":   "Consistent\nHash",
    "chord":             "Chord\nDHT",
    "skywalker":         "Sky\nWalker",
    "dynamo":            "Dynamo",
    "ludo":              "KV\nCompass",
}

FONT = "STIXGeneral"
matplotlib.rcParams.update({"font.family": FONT,
                            "axes.spines.top": False,
                            "axes.spines.right": False})
LBL_FS, TICK_FS, CAP_FS, LEG_FS = 20, 16.5, 21, 20
BOX_TICK_FS = 13

WORKLOAD_NAMES = [("Mixed", "mixed"), ("WildChat", "wildchat"),
                  ("LMSYS", "lmsys")]


# ── Data loading ─────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return [r for r in csv.DictReader(f)]


def _eff_ttft(row: dict) -> float | None:
    try:
        return float(row["ttft_ms"]) + float(row["router_overhead_ms"] or 0.0)
    except (KeyError, ValueError, TypeError):
        return None


def _e2e(row: dict) -> float | None:
    try:
        return float(row["e2e_ms"]) + float(row["router_overhead_ms"] or 0.0)
    except (KeyError, ValueError, TypeError):
        return None


def load_sweep(rdir: Path, scheme: str) -> list[tuple[float, list[dict]]]:
    """[(rate, ok_rows)] for every <scheme>_r<rate>.csv, ascending rate."""
    out = []
    for p in sorted(rdir.glob(f"{scheme}_r*.csv")):
        m = re.match(rf"{re.escape(scheme)}_r([0-9.]+)\.csv$", p.name)
        if not m:
            continue
        rows = [r for r in _read_csv(p) if r.get("ok") == "1"]
        if rows:
            out.append((float(m.group(1)), rows))
    return sorted(out)


def load_tpt(rdir: Path, scheme: str) -> float | None:
    """Closed-loop throughput (tok/s): summary JSON, else recompute."""
    js = rdir / f"{scheme}_tpt_summary.json"
    if js.exists():
        return float(json.load(open(js))["tpt_total_tok_s"])
    cf = rdir / f"{scheme}_tpt.csv"
    if cf.exists():
        rows = [r for r in _read_csv(cf) if r.get("ok") == "1"]
        if not rows:
            return None
        toks = sum(int(r["prompt_tokens"]) + int(r["completion_tokens"])
                   for r in rows)
        span = (max(float(r["send_s"]) for r in rows)
                - min(float(r["send_s"]) for r in rows))
        return toks / max(span, 1e-6)
    return None


# ── Plot ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "results",
                    help="Root holding results_mixed/ results_wildchat/ "
                         "results_lmsys/ (default: ./results)")
    ap.add_argument("--box-rate", type=float, default=None,
                    help="Arrival rate whose CSVs feed the TTFT/E2E box "
                         "rows (default: middle of the sweep).")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "results" / "figures" / "eval_grid.pdf")
    args = ap.parse_args()

    fig, axes = plt.subplots(4, 3, figsize=(16.5, 14.6))
    missing: list[str] = []

    workloads = []
    for wname, wkey in WORKLOAD_NAMES:
        rdir = args.results_root / f"results_{wkey}"
        if not rdir.is_dir():
            rdir = args.results_root  # single-workload layout fallback
        workloads.append((wname, rdir))

    # ── Row 0: P99 effective TTFT vs arrival rate ───────────────────────
    for col, (wname, rdir) in enumerate(workloads):
        ax = axes[0][col]
        for scheme in ALL_SCHEMES:
            sweep = load_sweep(rdir, scheme)
            if not sweep:
                missing.append(f"{wname}/{scheme} sweep")
                continue
            xs = [r for r, _ in sweep]
            ys = [float(np.percentile(
                    [t for t in (_eff_ttft(row) for row in rows)
                     if t is not None], 99))
                  for _, rows in sweep]
            is_hero = scheme == "ludo"
            ax.plot(xs, ys,
                    marker=MARKERS.get(scheme, "o"),
                    markersize=9 if is_hero else 7,
                    markeredgewidth=1.8 if is_hero else 1.2,
                    markerfacecolor=LINE_FILL[scheme],
                    markeredgecolor=LINE_COLORS[scheme],
                    markevery=max(1, len(xs) // 8),
                    linewidth=2.0 if is_hero else 1.4,
                    zorder=4 if is_hero else 3,
                    color=LINE_COLORS[scheme],
                    label=LINE_LABELS[scheme],
                    linestyle=SCHEME_LINESTYLES.get(scheme, "-"))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.tick_params(axis="y", which="minor", length=0)
        ax.set_xlabel("Arrival Rate (requests/s)", fontsize=LBL_FS,
                      fontname=FONT)
        if col == 0:
            ax.set_ylabel("P99 TTFT (ms)", fontsize=LBL_FS, fontname=FONT)
        ax.grid(True, axis="y", linestyle="dotted", alpha=0.8, linewidth=1)

    # ── Rows 1-2: TTFT / E2E box plots at the box rate ──────────────────
    def draw_box(ax, samples_by_scheme, ylabel):
        stats = []
        for s in ORDER_LAST:
            v = np.asarray(samples_by_scheme.get(s, [np.nan]))
            p1, p25, p50, p75, p99 = np.percentile(v, [1, 25, 50, 75, 99])
            stats.append({"med": p50, "q1": p25, "q3": p75,
                          "whislo": p1, "whishi": p99, "fliers": []})
        bp = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.62,
                    medianprops=dict(color="black", linewidth=1.5))
        for patch, s in zip(bp["boxes"], ORDER_LAST):
            patch.set_facecolor(LINE_FILL[s])
            patch.set_edgecolor(LINE_COLORS[s])
            patch.set_linewidth(1.5)
        for i, s in enumerate(ORDER_LAST):
            for ln in (bp["whiskers"][2 * i:2 * i + 2]
                       + bp["caps"][2 * i:2 * i + 2]):
                ln.set_color(LINE_COLORS[s])
                ln.set_linewidth(1.3)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
        ax.set_xticks(np.arange(1, len(ORDER_LAST) + 1))
        ax.set_xticklabels([XTICK_2LINE[s] for s in ORDER_LAST],
                           fontsize=BOX_TICK_FS, fontname=FONT,
                           linespacing=0.92)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.tick_params(axis="y", which="minor", length=0)
        ax.tick_params(axis="x", length=0)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=LBL_FS, fontname=FONT)
        ax.grid(True, axis="y", linestyle="dotted", alpha=0.8, linewidth=1)

    for col, (wname, rdir) in enumerate(workloads):
        ttft_by, e2e_by = {}, {}
        for scheme in ALL_SCHEMES:
            sweep = load_sweep(rdir, scheme)
            if not sweep:
                continue
            rates = [r for r, _ in sweep]
            if args.box_rate is not None and args.box_rate in rates:
                rows = dict(sweep)[args.box_rate]
            else:
                rows = sweep[len(sweep) // 2][1]   # middle of the sweep
            ttft_by[scheme] = [t for t in (_eff_ttft(r) for r in rows)
                               if t is not None]
            e2e_by[scheme] = [t for t in (_e2e(r) for r in rows)
                              if t is not None]
        if ttft_by:
            draw_box(axes[1][col], ttft_by,
                     "TTFT (ms)" if col == 0 else None)
            draw_box(axes[2][col], e2e_by,
                     "E2E Latency (ms)" if col == 0 else None)

    # ── Row 3: closed-loop throughput bars ──────────────────────────────
    for col, (wname, rdir) in enumerate(workloads):
        ax = axes[3][col]
        vals = []
        for s in ORDER_LAST:
            v = load_tpt(rdir, s)
            if v is None:
                missing.append(f"{wname}/{s} tpt")
                v = 0.0
            vals.append(v)
        vals = np.array(vals)
        # Auto unit: real 12-GPU testbeds land in Ktok/s, the sim in Mtok/s.
        if vals.max() >= 1e6:
            vals_u, unit = vals / 1e6, "Mtok/s"
        else:
            vals_u, unit = vals / 1e3, "Ktok/s"
        x = np.arange(len(ORDER_LAST))
        ax.set_axisbelow(True)
        for k, s in enumerate(ORDER_LAST):
            ax.bar(x[k], vals_u[k], 0.65,
                   facecolor=LINE_FILL[s], edgecolor=LINE_COLORS[s],
                   linewidth=2.0, hatch=SCHEME_HATCHES.get(s, ""))
            ax.text(x[k], vals_u[k] + 0.015 * max(vals_u.max(), 1e-9),
                    f"{vals_u[k]:.2f}", ha="center", va="bottom",
                    fontsize=14, fontname=FONT)
        ax.set_xticks(x)
        ax.set_xticklabels([XTICK_2LINE[s] for s in ORDER_LAST],
                           fontsize=BOX_TICK_FS, fontname=FONT,
                           linespacing=0.92)
        if vals_u.max() > 0:
            ax.set_ylim(0, vals_u.max() * 1.16)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.tick_params(axis="x", length=0)
        if col == 0:
            ax.set_ylabel(f"Throughput ({unit})", fontsize=LBL_FS,
                          fontname=FONT)
        ax.grid(True, axis="y", linestyle="dotted", alpha=0.8, linewidth=1)

    # ── Captions + legend (identical geometry to the sim grid) ──────────
    METRIC_NAMES = ["P99 TTFT", "TTFT", "E2E Latency", "Throughput"]
    fig.subplots_adjust(left=0.062, right=0.99, top=0.94, bottom=0.05,
                        hspace=0.78, wspace=0.19)
    fig.canvas.draw()
    letters = "abcdefghijkl"
    for r in range(4):
        for c in range(3):
            bb = axes[r][c].get_position()
            wname = workloads[c][0]
            fig.text((bb.x0 + bb.x1) / 2, bb.y0 - 0.048,
                     f"({letters[r * 3 + c]}) {wname}, {METRIC_NAMES[r]}.",
                     ha="center", va="top", fontsize=CAP_FS, fontname=FONT)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if "KVCompass" in labels:
        hero = labels.index("KVCompass")
        handles = [handles[hero]] + handles[:hero] + handles[hero + 1:]
        labels = [labels[hero]] + labels[:hero] + labels[hero + 1:]
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.998), ncol=7, frameon=False,
               prop={"family": FONT, "size": LEG_FS},
               handlelength=1.8, columnspacing=1.3, handletextpad=0.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"saved → {args.out}")
    if missing:
        print(f"[warn] missing data ({len(missing)}): "
              + ", ".join(missing[:20])
              + (" …" if len(missing) > 20 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
