"""motivation_remote_only.pdf — reuse that exists ONLY in other regions.

Replaying all WildChat conversations in timestamp order, each request's
K-token prefix is classified at its arrival moment: cold (nobody computed
it yet), local hit (a worker in the request's own serving region computed
it before), or remote-only hit (it exists, but only in OTHER regions).

This figure plots, per serving region, the remote-only share of that
region's hits: of the requests whose prefix is reusable at all, how many
could reuse it only across a region boundary.  Any per-node or per-region
index recomputes exactly these; demand-poor regions lose ~1/5 of their
reuse this way, demand-heavy ones almost none.

Serving region = nearest of the 8 deployment sites, the same mapping the
simulator's workload uses.  Bars: K = 256 and 512 tokens.

Usage: python regen_remote_only.py <stats_dir> <out_dir>
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP, OUTDIR = sys.argv[1], sys.argv[2]
d = json.load(open(f"{SP}/wc_remote_only.json"))

FONT = "STIXGeneral"
matplotlib.rcParams.update({"font.family": FONT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False})
LBL_FS, TICK_FS, LEG_FS, VAL_FS = 10, 9, 8, 7

REG = ["US", "CN", "DE", "JP", "SG", "GB", "KR", "AU"]
C_K256 = "#C17878"     # muted red, same as the cross-region palette
C_K512 = "#7FA3C4"     # muted blue
C_K1024 = "#B0A079"    # muted tan

def share(k):
    pr = d["classes"][str(k)]["per_region"]
    out = []
    for r in range(len(REG)):
        co, lo, ro = pr.get(str(r), [0, 0, 0])
        hits = lo + ro
        out.append(100.0 * ro / hits if hits else 0.0)
    return np.array(out)

s512 = share(512)

# Only the demand-poor regions, horizontal, styled after
# motivation_cross_region.pdf (same width, bar style, value labels).
SHOW = ["KR", "SG", "JP", "AU"]
idx = [REG.index(r) for r in SHOW]
vals = s512[idx]

y = np.arange(len(SHOW))[::-1]
fig, ax = plt.subplots(figsize=(4.03, 1.07))
ax.barh(y, vals, height=0.55, color="#C17878",
        edgecolor="white", linewidth=0.8)
for yy, v in zip(y, vals):
    ax.text(v + 0.5, yy, f"{v:.1f}%", va="center", ha="left",
            fontsize=VAL_FS, fontname=FONT)

ax.set_yticks(y)
ax.set_yticklabels(SHOW, fontsize=TICK_FS, fontname=FONT)
ax.set_ylabel("serving region", fontsize=LBL_FS, fontname=FONT)
ax.set_xlabel("% of cross-region hit only (prefixes \u2265512 tokens)",
              fontsize=LBL_FS, fontname=FONT)
ax.set_xlim(0, vals.max() * 1.18)
ax.tick_params(axis="x", labelsize=TICK_FS)
fig.subplots_adjust(left=0.14, right=0.985, top=0.95, bottom=0.30)
fig.savefig(f"{OUTDIR}/motivation_remote_only.pdf", dpi=150,
            bbox_inches="tight")
fig.savefig(f"{OUTDIR}/motivation_remote_only.png", dpi=150,
            bbox_inches="tight")

for k in d["ks"]:
    c = d["classes"][str(k)]
    print(f"K={k}: aggregate remote-only {c['pct_remote_only']:.1f}% of req, "
          f"{c['pct_remote_only_of_hits']:.1f}% of hits")
print("plotted (K=512):", {r: f"{v:.1f}%" for r, v in zip(SHOW, vals)})
