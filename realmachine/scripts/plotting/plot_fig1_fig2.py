#!/usr/bin/env python3
"""
plot_fig1_fig2.py — real-machine fig1 / fig2 / fig2b plotter.

Faithfully reproduces the simulator driver ``simulation/run_section3.py``'s
figure styling: same Morandi palette, same STIXGeneral font, same log-axis
corner notes, same Ludo p50/p90/p99 crosshair, same legend layout.

Inputs:
  --results-dir : directory of <scheme>_r<rate>.csv files written by
                  ludo_sglang.workload.trace_replayer.

Outputs (under --out-dir, default = --results-dir):
  fig1_p99_ttft_vs_rate.pdf
  fig2_ttft_cdf.pdf
  fig2b_e2e_cdf.pdf
  fig3_cache_reuse.pdf       — bonus: cache reuse fraction by scheme
  table_summary.csv

The "TTFT" / "E2E" used everywhere is the *effective* value:
  effective_ttft = ttft_ms + router_overhead_ms
where router_overhead_ms is the injected DHT-walk + WAN-to-worker
latency. The SGLang client's own ttft_ms only covers the SGLang-side
prefill, so effective_ttft is what an end user would actually wait
under the modeled deployment topology.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ─── Sim-aligned palette / labels / markers ─────────────────────────────────
# Lifted byte-for-byte from
#   simulation/run_section3.py (scheme palette block)
# so any side-by-side comparison reads as the same figure family.

ALL_SCHEMES = ["central_directory", "ludo", "radix_tree",
               "consistent_hash", "kademlia", "chord"]

LINE_COLORS = {
    "ludo":              "#985365",
    "radix_tree":        "#8a8c6f",
    "consistent_hash":   "#7e97ad",
    "kademlia":          "#b78D31",
    "chord":             "#6F8A8C",
    "oracle":            "#7A6E5A",
    "central_directory": "#7A6E5A",
}

LINE_FILL = {
    "ludo":              "#eba79c",
    "radix_tree":        "#a3b7a0",
    "consistent_hash":   "#ACBFEA",
    "kademlia":          "#FFDAA6",
    "chord":             "#B8CCCD",
    "oracle":            "#A5957D",
    "central_directory": "#A5957D",
}

LINE_LABELS = {
    "ludo":              "KVCompass",
    "radix_tree":        "PlanetServe",
    "consistent_hash":   "Consistent Hash",
    "kademlia":          "Kademlia DHT",
    "chord":             "Chord DHT",
    "central_directory": "Central Directory",
    "oracle":            "Oracle",
}

SCHEME_LINESTYLES = {  # all solid in fig1/2/2b — colour+marker carries identity
    "central_directory": "-",
    "oracle":            "-",
    "ludo":              "-",
    "radix_tree":        "-",
    "consistent_hash":   "-",
    "kademlia":          "-",
    "chord":             "-",
}

MARKERS = {
    "central_directory": "D",
    "oracle":            "D",
    "ludo":              "^",
    "radix_tree":        "s",
    "consistent_hash":   "o",
    "kademlia":          "v",
    "chord":             "P",
}

CSV_NAME_RE = re.compile(r"^(?P<scheme>[a-z_]+)_r(?P<rate>[0-9.]+)\.csv$")


# ─── Helpers ported from sim run_section3.py ────────────────────────────────

def save_fig(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def int_like_fmt(x, _pos=None):
    """Tick formatter: integer values render w/o decimals, others natural."""
    if x == int(x):
        return f"{int(x)}"
    return f"{x:g}"


def pad_log_xaxis_for_note(ax):
    """Extend a log-x axis so the corner-pinned $\\log_{10}$ note fits past
    the rightmost tick without dragging in the next decade's tick."""
    xmin_axis, xmax_data = ax.get_xlim()
    xmax_data = max(xmax_data, xmin_axis * 10)
    max_dec = int(np.floor(np.log10(xmax_data)))
    min_dec = int(np.ceil(np.log10(max(xmin_axis, 1e-3))))
    if xmax_data > 10 ** (max_dec + 0.7):
        max_dec += 1
    right_edge = 10 ** (max_dec + 0.85)
    ax.set_xlim(xmin_axis, right_edge)
    ax.set_xticks(10.0 ** np.arange(min_dec, max_dec + 1))


def add_log_note(ax, axis: str, fontsize: int = 13, x_pad: float = 0.0):
    """Corner-pinned $\\log_{10}$ note just outside the plot rectangle.

    `x_pad` (axis='x' only) shifts the note rightward in axes-relative
    units so the combined panel can clear the rightmost tick label at
    the bumped fontsize."""
    LABEL = r"$\log_{10}$"
    kw = dict(fontsize=fontsize, transform=ax.transAxes,
              clip_on=False, zorder=100)
    if axis == "y":
        ax.text(0.0, 1.02, LABEL, ha="left",  va="bottom", **kw)
    elif axis == "x":
        ax.text(1.0 + x_pad, 0.0, LABEL, ha="right", va="top", **kw)


# ─── CSV → DataFrame ────────────────────────────────────────────────────────

def discover_csvs(results_dir: Path) -> list[tuple[str, float, Path]]:
    out = []
    for p in sorted(results_dir.glob("*.csv")):
        if "summary" in p.name:
            continue
        m = CSV_NAME_RE.match(p.name)
        if not m:
            continue
        out.append((m["scheme"], float(m["rate"]), p))
    return out


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for scheme, rate, path in discover_csvs(results_dir):
        df = pd.read_csv(path)
        df["scheme"]   = scheme
        df["rate_rps"] = rate
        df["src_file"] = path.name
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No <scheme>_r<rate>.csv files in {results_dir}")
    df = pd.concat(rows, ignore_index=True)
    for col in ("ok", "ttft_ms", "e2e_ms", "router_overhead_ms",
                "wan_to_worker_ms", "prompt_tokens", "completion_tokens",
                "cached_tokens"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Effective TTFT/E2E include the router-injected lookup + WAN
    # overhead, which the SGLang client's own ttft_ms / e2e_ms do not
    # see (sleep happens before SGLang is called). This is the
    # user-perceived latency under the modeled deployment topology.
    if "router_overhead_ms" in df.columns:
        df["effective_ttft_ms"] = (df["ttft_ms"].fillna(0)
                                    + df["router_overhead_ms"].fillna(0))
        df["effective_e2e_ms"]  = (df["e2e_ms"].fillna(0)
                                    + df["router_overhead_ms"].fillna(0))
    else:
        df["effective_ttft_ms"] = df["ttft_ms"]
        df["effective_e2e_ms"]  = df["e2e_ms"]
    return df


def per_cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["ok"] == 1].copy()
    if "prompt_tokens" in ok.columns and "cached_tokens" in ok.columns:
        ok["reuse_frac"] = ok["cached_tokens"] / ok["prompt_tokens"].clip(lower=1)
    grouped = ok.groupby(["scheme", "rate_rps"], as_index=False)
    out = grouped.agg(
        n_ok            = ("ok", "sum"),
        ttft_p50        = ("effective_ttft_ms", lambda s: float(np.percentile(s, 50)) if len(s) else np.nan),
        ttft_p95        = ("effective_ttft_ms", lambda s: float(np.percentile(s, 95)) if len(s) else np.nan),
        ttft_p99        = ("effective_ttft_ms", lambda s: float(np.percentile(s, 99)) if len(s) else np.nan),
        ttft_mean       = ("effective_ttft_ms", "mean"),
        e2e_p50         = ("effective_e2e_ms",  lambda s: float(np.percentile(s, 50)) if len(s) else np.nan),
        e2e_p95         = ("effective_e2e_ms",  lambda s: float(np.percentile(s, 95)) if len(s) else np.nan),
        e2e_p99         = ("effective_e2e_ms",  lambda s: float(np.percentile(s, 99)) if len(s) else np.nan),
        sglang_ttft_p50 = ("ttft_ms", lambda s: float(np.percentile(s, 50)) if len(s) else np.nan),
        sglang_ttft_p99 = ("ttft_ms", lambda s: float(np.percentile(s, 99)) if len(s) else np.nan),
        cached_frac_mean = ("reuse_frac", "mean") if "reuse_frac" in ok.columns
                          else ("ttft_ms", lambda s: np.nan),
    )
    totals = (df.groupby(["scheme", "rate_rps"], as_index=False)
                .size().rename(columns={"size": "n_total"}))
    return out.merge(totals, on=["scheme", "rate_rps"]) \
              .sort_values(["scheme", "rate_rps"]).reset_index(drop=True)


def _ordered_present(df_or_stats) -> list[str]:
    present = set(df_or_stats["scheme"].unique())
    return [s for s in ALL_SCHEMES if s in present] + \
           sorted(present - set(ALL_SCHEMES))


# ─── Fig 1 ──────────────────────────────────────────────────────────────────

def plot_fig1(stats: pd.DataFrame, out_path: Path) -> None:
    """P99 TTFT (log y) vs arrival rate (linear x)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    schemes = _ordered_present(stats)
    for scheme in schemes:
        sub = stats[stats["scheme"] == scheme].sort_values("rate_rps")
        xs = sub["rate_rps"].to_numpy()
        ys = sub["ttft_p99"].to_numpy()
        zorder = 4 if scheme in ("central_directory", "ludo", "radix_tree") else 3
        is_hero = (scheme == "ludo")
        msize = 12  if is_hero else 9
        ew    = 2.2 if is_hero else 1.5
        lw    = 2.2 if is_hero else 1.5
        markevery = max(1, len(xs) // 8)
        ax.plot(xs, ys,
                marker=MARKERS.get(scheme, "o"),
                markersize=msize, markeredgewidth=ew,
                markerfacecolor=LINE_FILL.get(scheme, "#cccccc"),
                markeredgecolor=LINE_COLORS.get(scheme, "#000000"),
                markevery=markevery,
                linewidth=lw, zorder=zorder,
                color=LINE_COLORS.get(scheme, "#000000"),
                label=LINE_LABELS.get(scheme, scheme),
                linestyle=SCHEME_LINESTYLES.get(scheme, "-"),
                clip_on=False)
    ax.set_xlabel("Arrival Rate (requests/s)", fontsize=17,
                  fontname="STIXGeneral")
    ax.set_ylabel("P99 TTFT (ms)", fontsize=16,
                  fontname="STIXGeneral")
    ax.set_yscale("log")
    # Pad the y-range a bit beyond the data so the log compression
    # has visual headroom — keeps the lowest curves from getting
    # squeezed against the bottom axis where small relative
    # differences look exaggerated. Pure cosmetic, no data change.
    y_data_min = float(stats["ttft_p99"].min())
    y_data_max = float(stats["ttft_p99"].max())
    ax.set_ylim(bottom=max(80.0, y_data_min / 1.5),
                top   =       y_data_max * 1.6)
    ax.tick_params(axis="both", labelsize=16)

    # ── Unified plain-integer tick labels on the log y-axis ─────────
    # The default LogLocator + sci-notation minor formatter mixes
    # "10²", "3 × 10²", "6 × 10²" etc., which looks inconsistent when
    # the y-range spans less than one decade. We instead label every
    # tick with its raw integer (e.g., 200, 300, 500, 1000) so all
    # tick labels share a single style. Major and minor ticks share
    # the same labelsize so the column of y labels reads as a single
    # uniform set of numbers.
    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10, subs=(1.0,), numticks=20))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10, subs=(2, 3, 5), numticks=40))
    plain_int = mticker.FuncFormatter(lambda y, _: f"{int(round(y))}")
    ax.yaxis.set_major_formatter(plain_int)
    ax.yaxis.set_minor_formatter(plain_int)
    ax.tick_params(axis="y", which="major", labelsize=16, length=5)
    ax.tick_params(axis="y", which="minor", labelsize=16, length=3)

    ax.legend(loc="lower center",
              bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False,
              borderaxespad=0,
              prop={"family": "STIXGeneral", "size": 15})
    # Dotted horizontal gridline at every labelled tick (major + minor)
    # so each y-axis number on the left visually corresponds to a
    # reference line crossing the plot, matching the existing 1000-line.
    ax.grid(True, axis="y", which="both",
            linestyle="dotted", alpha=0.8, linewidth=1)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    add_log_note(ax, "y")
    save_fig(fig, out_path)


# ─── Fig 2 / 2b ─────────────────────────────────────────────────────────────

def _cdf_percentile(sorted_vals, p):
    n = len(sorted_vals)
    idx = min(n - 1, int(round(n * p)))
    return sorted_vals[idx]


def _annotate_ludo_pcts(ax, sorted_ludo_vals, label_fontsize=17,
                        label_offsets: dict | None = None):
    """Crosshair p50/p90/p99 annotations on the Ludo CDF curve.

    Each label sits at the **lower-left** of its crosshair intersection
    point (i.e., inside the rectangle bounded by the dashed h/v-lines)
    and is rendered bold so it reads against the curves behind.

    `label_fontsize` defaults to the standalone-figure size (17 pt); the
    combined panel overrides it to keep the annotation visible after the
    per-panel width is shrunk."""
    if len(sorted_ludo_vals) == 0:
        return
    color   = LINE_COLORS["ludo"]
    xmin, _ = ax.get_xlim()
    # All three labels sit at the lower-left of their crosshair
    # intersection. Per-figure caller can override these via
    # `label_offsets`; defaults below work for fig2 (TTFT CDF).
    if label_offsets is None:
        label_offsets = {
            "p50": (-8,  -8),
            "p90": (-30, -8),
            "p99": (-30, -8),
        }
    label_bbox = dict(facecolor="lightgray", edgecolor="none",
                      alpha=0.55, pad=2.0)
    for p, tag in [(0.50, "p50"), (0.90, "p90"), (0.99, "p99")]:
        x_val = _cdf_percentile(sorted_ludo_vals, p)
        ax.hlines(y=p, xmin=xmin, xmax=x_val,
                  colors=color, linestyles="dashed",
                  linewidth=1.5, zorder=9)
        ax.vlines(x=x_val, ymin=0.0, ymax=p,
                  colors=color, linestyles="dashed",
                  linewidth=1.5, zorder=9)
        dx, dy = label_offsets[tag]
        ax.annotate(f"{tag.upper()}: {x_val:,.0f} ms",
                    xy=(x_val, p),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=label_fontsize, fontname="STIXGeneral",
                    fontweight="bold",
                    color=color, zorder=11,
                    horizontalalignment="right",
                    verticalalignment="top",
                    bbox=label_bbox)


def _plot_cdf(df: pd.DataFrame, rate: float, col: str, xlabel: str,
              out_path: Path, label_offsets: dict | None = None) -> None:
    ok = df[(df["ok"] == 1) & np.isclose(df["rate_rps"], rate)]
    if len(ok) == 0:
        print(f"  WARN: no successful requests at rate={rate}; skipping {out_path.name}")
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    schemes = _ordered_present(ok)

    # Pre-sort each scheme's values once.
    per_scheme: dict[str, np.ndarray] = {}
    for scheme in schemes:
        vals = ok[ok["scheme"] == scheme][col].dropna().to_numpy()
        per_scheme[scheme] = np.sort(vals)

    for scheme in schemes:
        vals = per_scheme[scheme]
        n    = len(vals)
        if n == 0:
            continue
        zorder = 4 if scheme in ("central_directory", "ludo", "radix_tree") else 3
        step = max(1, n // 500)
        is_hero = (scheme == "ludo")
        lw      = 4.0 if is_hero else 3.0
        ax.plot(vals[::step],
                (np.arange(0, n, step) + 1) / n,
                color=LINE_COLORS.get(scheme, "#000000"),
                linewidth=lw, zorder=zorder,
                linestyle=SCHEME_LINESTYLES.get(scheme, "-"),
                label=LINE_LABELS.get(scheme, scheme),
                clip_on=False)

    if "ludo" in per_scheme:
        _annotate_ludo_pcts(ax, per_scheme["ludo"],
                            label_offsets=label_offsets)
    ax.set_xscale("log")
    pad_log_xaxis_for_note(ax)
    ax.set_xlabel(xlabel, fontsize=17, fontname="STIXGeneral")
    ax.set_ylabel("CDF",  fontsize=16, fontname="STIXGeneral")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="both", labelsize=16)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(np.log10(x))}"))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(int_like_fmt))
    ax.legend(loc="lower center",
              bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False,
              borderaxespad=0,
              prop={"family": "STIXGeneral", "size": 15})
    ax.grid(True, axis="y", linestyle="dotted", alpha=0.8, linewidth=1)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    add_log_note(ax, "x")
    save_fig(fig, out_path)


def plot_fig2(df: pd.DataFrame, rate: float, out_path: Path) -> None:
    # fig2 (TTFT CDF) — p50 nudged a bit right (closer to its crosshair
    # so it sits just inside the L), p99 nudged a touch up + right
    # from the previous spot to clear the curve cluster on the right.
    fig2_offsets = {
        "p50": (-4,  -8),
        "p90": (-30, -8),
        "p99": (-30, -2),
    }
    _plot_cdf(df, rate, "effective_ttft_ms", "TTFT (ms)", out_path,
              label_offsets=fig2_offsets)


def plot_fig2b(df: pd.DataFrame, rate: float, out_path: Path) -> None:
    # fig2b (E2E CDF) — p99 nudged a touch up + right from the prior
    # spot so it sits just under the hline tail without floating too
    # far left into the curves.
    fig2b_offsets = {
        "p50": (-8,  -8),
        "p90": (-30, -8),
        "p99": (-30, -2),
    }
    _plot_cdf(df, rate, "effective_e2e_ms", "End-to-End Latency (ms)", out_path,
              label_offsets=fig2b_offsets)


# ─── Fig 3: cache reuse fraction by scheme (sim-style bar chart) ────────────

def plot_fig3(df: pd.DataFrame, rate: float, out_path: Path) -> None:
    ok = df[(df["ok"] == 1) & np.isclose(df["rate_rps"], rate)].copy()
    if "prompt_tokens" not in ok.columns or "cached_tokens" not in ok.columns:
        return
    ok["reuse_frac"] = ok["cached_tokens"] / ok["prompt_tokens"].clip(lower=1)
    schemes = _ordered_present(ok)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    means = []
    stds  = []
    for s in schemes:
        v = ok[ok["scheme"] == s]["reuse_frac"].to_numpy()
        means.append(v.mean() if len(v) else 0.0)
        stds.append(v.std()  if len(v) else 0.0)
    xs = np.arange(len(schemes))
    bars = ax.bar(xs, means, yerr=stds, capsize=3, width=0.7,
                  color=[LINE_FILL.get(s, "#cccccc")        for s in schemes],
                  edgecolor=[LINE_COLORS.get(s, "#000000")  for s in schemes],
                  linewidth=2.0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01,
                f"{m:.2f}", ha="center", va="bottom",
                fontsize=14, fontname="STIXGeneral")

    ax.set_xticks(xs)
    ax.set_xticklabels([LINE_LABELS.get(s, s) for s in schemes],
                       rotation=20, ha="right",
                       fontsize=15, fontname="STIXGeneral")
    ax.set_ylabel("KV Cache Reuse Fraction\n(cached_tokens / prompt_tokens)",
                  fontsize=16, fontname="STIXGeneral")
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, max(0.05, max(means) * 1.25 + 0.05))
    ax.grid(True, axis="y", linestyle="dotted", alpha=0.8, linewidth=1)
    fig.tight_layout()
    save_fig(fig, out_path)


# ─── Combined panel: fig5 + fig6 + fig6b side-by-side ──────────────────────

def plot_combined(stats: pd.DataFrame, df: pd.DataFrame,
                  cdf_rate: float, out_path: Path) -> None:
    """Render fig5 (P99 TTFT vs rate), fig6 (TTFT CDF), and fig6b (E2E CDF)
    side-by-side in a single figure with one shared top legend.

    Tuned for paper inclusion: each panel is roughly square (5.33 × 5.0 in)
    so the figure stays narrow when scaled to a column width, and every
    text element (axis labels, ticks, crosshair annotations, legend) is
    bumped well above the standalone-figure sizes so it remains readable
    at small print scales.  The legend sits flush against the panel tops
    to minimise inter-element whitespace.
    """
    ok = df[(df["ok"] == 1) & np.isclose(df["rate_rps"], cdf_rate)]
    if len(ok) == 0:
        print(f"  WARN: no successful requests at rate={cdf_rate}; "
              f"skipping {out_path.name}")
        return

    # Type-system shorthand for the bumped sizes — every panel reads from
    # this block so a downstream tweak only touches one place.  In-panel
    # text (labels, ticks, annotation, log-note) sits noticeably above
    # the standalone-figure sizes; the legend is held back to a size that
    # lets the full row strip stay strictly inside the figure width.
    LBL_FS      = 26   # axis label font (xlabel / ylabel)
    TICK_FS     = 24   # tick label font
    NOTE_FS     = 22   # corner-pinned log_{10} note
    ANNOT_FS    = 26   # Ludo crosshair "pNN: V ms" annotation
    LEGEND_FS   = 24   # shared top legend
    LEGEND_NCOL = 3    # 6 schemes laid out as 2 rows × 3 cols so the longest
                       # label ("Central Directory") fits within the panel-
                       # block width without overflowing.
    XTICK_N     = 5    # linear-x tick budget on fig5 panel — sized for the
                       # wider 6-in panel so up to 5 ticks lay out at ~1.5
                       # in spacing, neither crowded nor overly sparse.
    LOGNOTE_PAD = 0.06 # axes-relative right-shift for the x-axis log_{10}
                       # corner note — at 22-pt the note is wide enough to
                       # visually collide with the rightmost tick label
                       # when ha='right' against x=1.0; pushing it ~6 % of
                       # the panel width past the right spine clears the
                       # collision.

    # Total width 20 in (each panel ~6.67 in) gives the x-axis enough
    # physical length for 5 evenly-spaced ticks at the bumped 24-pt label
    # size with comfortable breathing room; the 5.5-in height absorbs the
    # 2-row legend so the per-panel plotted area stays comfortably large.
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # ── Panel 1: fig5 P99 TTFT vs arrival rate ──────────────────────────────
    ax1 = axes[0]
    schemes_fig1 = _ordered_present(stats)
    for scheme in schemes_fig1:
        sub = stats[stats["scheme"] == scheme].sort_values("rate_rps")
        xs = sub["rate_rps"].to_numpy()
        ys = sub["ttft_p99"].to_numpy()
        zorder = 4 if scheme in ("central_directory", "ludo", "radix_tree") else 3
        is_hero = (scheme == "ludo")
        msize = 12  if is_hero else 9
        ew    = 2.2 if is_hero else 1.5
        lw    = 2.2 if is_hero else 1.5
        markevery = max(1, len(xs) // 8)
        ax1.plot(xs, ys,
                 marker=MARKERS.get(scheme, "o"),
                 markersize=msize, markeredgewidth=ew,
                 markerfacecolor=LINE_FILL.get(scheme, "#cccccc"),
                 markeredgecolor=LINE_COLORS.get(scheme, "#000000"),
                 markevery=markevery,
                 linewidth=lw, zorder=zorder,
                 color=LINE_COLORS.get(scheme, "#000000"),
                 label=LINE_LABELS.get(scheme, scheme),
                 linestyle=SCHEME_LINESTYLES.get(scheme, "-"),
                 clip_on=False)
    ax1.set_xlabel("Arrival Rate (requests/s)", fontsize=LBL_FS,
                   fontname="STIXGeneral")
    ax1.set_ylabel("P99 TTFT (ms)", fontsize=LBL_FS, fontname="STIXGeneral")
    ax1.set_yscale("log")
    y_data_min = float(stats["ttft_p99"].min())
    y_data_max = float(stats["ttft_p99"].max())
    ax1.set_ylim(bottom=max(80.0, y_data_min / 1.5),
                 top   =       y_data_max * 1.6)
    # Cap the linear x-tick count so the bumped tick labels don't crowd
    # the narrower panel width.
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=XTICK_N))
    ax1.tick_params(axis="both", labelsize=TICK_FS)
    ax1.yaxis.set_major_locator(
        mticker.LogLocator(base=10, subs=(1.0,), numticks=20))
    ax1.yaxis.set_minor_locator(
        mticker.LogLocator(base=10, subs=(2, 3, 5), numticks=40))
    plain_int = mticker.FuncFormatter(lambda y, _: f"{int(round(y))}")
    ax1.yaxis.set_major_formatter(plain_int)
    ax1.yaxis.set_minor_formatter(plain_int)
    ax1.tick_params(axis="y", which="major", labelsize=TICK_FS, length=5)
    ax1.tick_params(axis="y", which="minor", labelsize=TICK_FS, length=3)
    ax1.grid(True, axis="y", which="both",
             linestyle="dotted", alpha=0.8, linewidth=1)
    add_log_note(ax1, "y", fontsize=NOTE_FS)

    # ── Panel 2 & 3: fig6 (TTFT) + fig6b (E2E) CDFs ─────────────────────────
    schemes_cdf = _ordered_present(ok)
    for ax, col, xlabel in (
            (axes[1], "effective_ttft_ms", "TTFT (ms)"),
            (axes[2], "effective_e2e_ms",  "End-to-End Latency (ms)"),
    ):
        per_scheme: dict[str, np.ndarray] = {}
        for scheme in schemes_cdf:
            vals = ok[ok["scheme"] == scheme][col].dropna().to_numpy()
            per_scheme[scheme] = np.sort(vals)
        for scheme in schemes_cdf:
            vals = per_scheme[scheme]
            n = len(vals)
            if n == 0:
                continue
            zorder = 4 if scheme in ("central_directory", "ludo", "radix_tree") else 3
            step = max(1, n // 500)
            is_hero = (scheme == "ludo")
            lw = 4.0 if is_hero else 3.0
            ax.plot(vals[::step],
                    (np.arange(0, n, step) + 1) / n,
                    color=LINE_COLORS.get(scheme, "#000000"),
                    linewidth=lw, zorder=zorder,
                    linestyle=SCHEME_LINESTYLES.get(scheme, "-"),
                    label=LINE_LABELS.get(scheme, scheme),
                    clip_on=False)
        if "ludo" in per_scheme:
            _annotate_ludo_pcts(ax, per_scheme["ludo"], label_fontsize=ANNOT_FS)
        ax.set_xscale("log")
        pad_log_xaxis_for_note(ax)
        ax.set_xlabel(xlabel, fontsize=LBL_FS, fontname="STIXGeneral")
        ax.set_ylabel("CDF",  fontsize=LBL_FS, fontname="STIXGeneral")
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(np.log10(x))}"))
        ax.tick_params(axis="x", which="minor", length=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(int_like_fmt))
        ax.grid(True, axis="y", linestyle="dotted", alpha=0.8, linewidth=1)
        add_log_note(ax, "x", fontsize=NOTE_FS, x_pad=LOGNOTE_PAD)

    # ── Shared top legend ───────────────────────────────────────────────────
    # Pull handles from the fig5 panel: it draws colour + linestyle + marker
    # for each scheme, the most informative single-handle representation.
    # 6 schemes lay out as 2 rows × 3 cols so each entry has ~5.3 in of
    # horizontal space — comfortable for the longest label ("Central
    # Directory") at 24 pt — while the legend strip stays strictly inside
    # the figure's 16-in width.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=LEGEND_NCOL, frameon=False,
               prop={"family": "STIXGeneral", "size": LEGEND_FS},
               handlelength=2.2, columnspacing=1.5,
               handletextpad=0.4)

    # rect[3]=0.83 leaves room for the 2-row legend strip at the top; the
    # panel tops sit just under the legend baseline so the legend reads as
    # part of the panel block rather than a detached caption.  The slightly
    # taller figure (5.5 vs 5.0 in) keeps each panel's plotted area at
    # ~4.6 in tall so the panels themselves do not visibly shrink.
    fig.tight_layout(rect=[0, 0, 1, 0.83])
    save_fig(fig, out_path)


# ─── CDF rate selection ─────────────────────────────────────────────────────

def pick_cdf_rate(stats: pd.DataFrame) -> float:
    """Default heuristic: highest rate at which Ludo p99 TTFT is still
    within 5× of its lowest-p99 (i.e. before the knee). Falls back to
    the maximum rate if Ludo isn't in the sweep."""
    ludo = stats[stats["scheme"] == "ludo"].sort_values("rate_rps")
    if len(ludo) == 0:
        return float(stats["rate_rps"].max())
    base = ludo["ttft_p99"].min()
    cap  = 5.0 * base
    knee = ludo[ludo["ttft_p99"] <= cap]["rate_rps"]
    if len(knee) == 0:
        return float(ludo["rate_rps"].iloc[0])
    return float(knee.max())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out-dir",     type=Path, default=None)
    p.add_argument("--cdf-rate",    type=float, default=None,
                   help="Rate (req/s) to use for fig2/fig2b/fig3 CDFs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Raw per-cell CSVs live in results/; derived figures + table_summary
    # go to results/figures/ unless --out-dir overrides it.
    out_dir = args.out_dir or (args.results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    print("=== plot_fig1_fig2 ===")
    print(f"  loaded {len(df)} rows from {args.results_dir}/")
    print(f"  schemes: {sorted(df['scheme'].unique())}")
    print(f"  rates  : {sorted(df['rate_rps'].unique())}")

    stats = per_cell_stats(df)
    table_path = out_dir / "table_summary.csv"
    stats.to_csv(table_path, index=False)
    print(f"  wrote {table_path}")

    cdf_rate = args.cdf_rate if args.cdf_rate is not None else pick_cdf_rate(stats)
    print(f"  CDF rate for fig2/2b/3: {cdf_rate:g} req/s")

    plot_fig1 (stats,    out_dir / "fig1_p99_ttft_vs_rate.pdf")
    plot_fig2 (df,  cdf_rate, out_dir / "fig2_ttft_cdf.pdf")
    plot_fig2b(df,  cdf_rate, out_dir / "fig2b_e2e_cdf.pdf")
    plot_fig3 (df,  cdf_rate, out_dir / "fig3_cache_reuse.pdf")
    plot_combined(stats, df, cdf_rate,
                  out_dir / "fig5_6_6b_combined.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
