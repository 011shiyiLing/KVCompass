"""motivation_cross_region.pdf — single horizontal grouped bar chart.

Replaces the old two-panel (scatter + %) figure.  For every shared prefix
(appears ≥2 times from ≥2 distinct users, ≥K tokens long) that is shared
across ≥2 countries, classify it:

  cross-country, same region  — sharers from >=2 countries whose nearest
                                deployment region is the SAME site (one
                                RDMA domain, e.g. FR+NL users served by DE)
  cross-region                — sharers map to >=2 deployment regions

A "region" here is one of the 8 RDMA-connected deployment sites of the
serving model (US/CN/DE/JP/SG/GB/KR/AU), not a continent: each sharer
country is mapped to its nearest region exactly as the simulator's
workload does.

Rows (y) bucket prefixes by how many countries share them (2 / 3 / 4 / ≥5);
bar length (x, log) is the total number of reuse occurrences (requests
hitting such a prefix).  K defaults to 1024 tokens — a level of the
simulator's chunk grid at/above the WAN break-even length, so every count
here is reuse deep enough to be worth moving across regions, not short
prompt boilerplate.

Usage: python regen_cross_region_bar.py <stats_dir> <out_dir> [K]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SP, OUTDIR = sys.argv[1], sys.argv[2]
K = sys.argv[3] if len(sys.argv) > 3 else "1024"
d = json.load(open(f"{SP}/wc_stats.json"))

FONT = "STIXGeneral"
matplotlib.rcParams.update({"font.family": FONT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False})
LBL_FS, TICK_FS, LEG_FS, SEG_FS, VAL_FS = 10, 9, 8, 7.5, 7

# Palette/style mirror motivation_geo_single.py.
C_SAMEREG = "#7FA3C4"   # cross-country, same region
C_XREGION = "#C17878"   # cross-region

sh = d["sharing"][K]
VOL_SHARED = sh["vol_shared"]                      # all reuse occurrences on shared prefixes
sc = np.array(sh["scatter"], float)   # [count, n_country, n_cont, n_users, n_serving_region]
cnt, nco, nreg = sc[:, 0], sc[:, 1], sc[:, 4]
xc = nco >= 2                                      # shared across >=2 countries
same_region = xc & (nreg == 1)
x_region    = xc & (nreg >= 2)

BUCKETS = [(2, "2"), (3, "3"), (4, "4"), (5, "≥5")]

def vol(mask, lo, hi=None):
    m = mask & (nco >= lo) & (nco < hi if hi else np.ones_like(xc, bool))
    return cnt[m].sum()

rows_same, rows_x = [], []
for i, (lo, lbl) in enumerate(BUCKETS):
    hi = BUCKETS[i + 1][0] if i + 1 < len(BUCKETS) else None
    # Percent of ALL shared-prefix reuse volume, so bar length reads as
    # "how much of the workload's reuse this class represents" instead of
    # an absolute count with no scale reference.
    rows_same.append(100.0 * vol(same_region, lo, hi) / VOL_SHARED)
    rows_x.append(100.0 * vol(x_region, lo, hi) / VOL_SHARED)

y = np.arange(len(BUCKETS))
BH = 0.38
fig, ax = plt.subplots(figsize=(4.03, 1.07))
ax.barh(y + BH/2, rows_same, height=BH, color=C_SAMEREG,
        edgecolor="white", linewidth=0.8, label="cross-country, same region")
ax.barh(y - BH/2, rows_x, height=BH, color=C_XREGION,
        edgecolor="white", linewidth=0.8, label="cross-region")
for yy, v in list(zip(y + BH/2, rows_same)) + list(zip(y - BH/2, rows_x)):
    if v > 0:
        ax.text(v * 1.12, yy, f"{v:.2g}%", va="center", ha="left",
                fontsize=VAL_FS, fontname=FONT)

ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels([b[1] for b in BUCKETS], fontsize=TICK_FS, fontname=FONT)
ax.set_ylabel("# countries sharing", fontsize=LBL_FS, fontname=FONT)
ax.set_xlabel(f"% of shared-prefix reuse volume (prefixes ≥{K} tokens)",
              fontsize=LBL_FS, fontname=FONT)
ax.tick_params(axis="x", labelsize=TICK_FS)
ax.margins(x=0.18)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
          frameon=False, prop={"family": FONT, "size": LEG_FS},
          handlelength=1.0, columnspacing=0.9, handletextpad=0.35)
fig.subplots_adjust(left=0.14, right=0.985, top=0.80, bottom=0.30)
fig.savefig(f"{OUTDIR}/motivation_cross_region.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUTDIR}/motivation_cross_region.png", dpi=150, bbox_inches="tight")
n_same, n_x = int(same_region.sum()), int(x_region.sum())
print(f"K={K}: same-region prefixes={n_same} (vol {int(cnt[same_region].sum()):,}), "
      f"cross-region prefixes={n_x} (vol {int(cnt[x_region].sum()):,})")
