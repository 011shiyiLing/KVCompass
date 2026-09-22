import json, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Two horizontal 100%-stacked bars sharing one axes:
#   top bar    = Demand  (WildChat user requests, share of global total)
#   bottom bar = Supply  (LLM deployment census, share of global total)
# Each bar is segmented by country with a consistent per-country color, so the
# geographic demand/supply mismatch reads off directly (e.g. RU is a large
# demand slice but a sliver of supply; the US is the reverse). A grey
# "Rest of world" segment fills each bar out to 100%.
SP = sys.argv[1]; OUTDIR = sys.argv[2]
d = json.load(open(f"{SP}/wc_stats.json"))

FONT = "STIXGeneral"
matplotlib.rcParams.update({"font.family": FONT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False})
LBL_FS, TICK_FS, LEG_FS, SEG_FS = 10, 9, 8, 7.5

# ---------- demand (WildChat requests) ----------
total = d["total_rows"]; cc = d["country_counts"]
def dem(*names): return 100*sum(cc.get(n,0) for n in names)/total
DEMAND = {"US": dem("United States"), "CN": dem("China","Hong Kong","Taiwan"),
          "RU": dem("Russia"), "DE": dem("Germany"), "GB": dem("United Kingdom"),
          "FR": dem("France"), "JP": dem("Japan"), "IN": dem("India"),
          "EG": dem("Egypt"), "SG": dem("Singapore")}

# ---------- supply (LLM_Deployment census) ----------
SUP = {r["cc"]: float(r["supply_pct"]) for r in csv.DictReader(open(f"{SP}/supply_country_share.csv"))}

# Country plotting order (stacked left→right in both bars) and one color each.
ORDER = ["US","CN","RU","DE","GB","JP","SG","IN","FR","EG"]
NAME  = {"US":"USA","CN":"China","RU":"Russia","DE":"Germany","GB":"UK",
         "JP":"Japan","SG":"Singapore","IN":"India","FR":"France","EG":"Egypt"}
# One color per country (muted 9-color palette, in bar order; EG gets a
# matching 10th muted clay tone); "Rest" is neutral grey.
CC = {"US":"#7FA3C4",   # 矢车菊蓝
      "CN":"#8290C9",   # 粉雾紫
      "RU":"#A392C5",   # 灰绯红
      "DE":"#C17878",   # 灰绿
      "GB":"#90A88B",   # 雾青
      "JP":"#6BA3A5",   # 燕麦棕
      "SG":"#B39B84",   # 蓝灰
      "IN":"#96A0AB",   # 燕麦色
      "FR":"#E0D3B4",
      "EG":"#C7A9A0"}
REST_C = "#CFCbC4"; REST_LBL = "Rest of world"

def shares(src):
    """Per-country % for the named countries + a Rest remainder to reach 100."""
    vals = [src.get(c, 0.0) for c in ORDER]
    vals.append(max(0.0, 100.0 - sum(vals)))
    return np.array(vals)

DEM = shares(DEMAND)
SUP = shares(SUP)
SEG_KEYS   = ORDER + ["__REST__"]
SEG_COLORS = [CC[c] for c in ORDER] + [REST_C]

fig, ax = plt.subplots(figsize=(3.35, 1.3))
YPOS = {"Demand": 0.92, "Supply": 0.0}; BH = 0.72

def _text_color(hexc):
    r,g,b = (int(hexc[i:i+2],16)/255 for i in (1,3,5))
    return "black" if (0.299*r+0.587*g+0.114*b) > 0.6 else "white"

def draw_bar(y, data):
    left = 0.0
    for k, val, col in zip(SEG_KEYS, data, SEG_COLORS):
        ax.barh(y, val, height=BH, left=left, color=col,
                edgecolor="white", linewidth=0.8, zorder=3)
        # In-segment label: percentage only (country identity is carried by
        # the color legend).  Drawn wherever the "NN%" text fits (≥ 3 % of
        # axis width at 13 pt); slivers stay label-free.  GB is skipped by
        # request: its demand and supply shares are both 4 %, so the labels
        # add no contrast information.
        if k != "__REST__" and k != "GB" and val >= 5.0:
            ax.text(left + val/2, y, f"{val:.0f}%", ha="center", va="center",
                    fontsize=SEG_FS, fontname=FONT, color=_text_color(col),
                    zorder=4)
        left += val

draw_bar(YPOS["Demand"], DEM)
draw_bar(YPOS["Supply"], SUP)

ax.set_xlim(0, 100)
ax.set_ylim(-0.44, 1.42)
ax.set_yticks([YPOS["Supply"], YPOS["Demand"]])
ax.set_yticklabels(["Supply\n(deployments)", "Demand\n(requests)"],
                   fontsize=LBL_FS, fontname=FONT, linespacing=0.9)
ax.set_xlabel("% of global total", fontsize=LBL_FS, fontname=FONT)
ax.tick_params(axis="x", labelsize=TICK_FS)
ax.tick_params(axis="y", length=0)

# One legend: country color key (+ Rest of world) along the top.
handles = [Patch(facecolor=CC[c], label=NAME[c]) for c in ORDER]
handles.append(Patch(facecolor=REST_C, label=REST_LBL))
# Anchor the legend to the axes so it floats just above the bars (rather
# than to the figure), keeping the inter-element gap tight and fixed.
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.04),
          ncol=6, frameon=False, prop={"family":FONT,"size":LEG_FS},
          handlelength=0.9, columnspacing=0.6, handletextpad=0.3,
          labelspacing=0.22, borderpad=0.0)

fig.subplots_adjust(left=0.20, right=0.99, top=0.70, bottom=0.34)
fig.savefig(f"{OUTDIR}/motivation_geo_single.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUTDIR}/motivation_geo_single.png", dpi=150, bbox_inches="tight")
print("saved motivation_geo_single (two horizontal 100% bars: demand vs supply, colored by country)")
