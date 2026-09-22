import json, sys, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# motivation_geo: demand is globally dispersed (WildChat) but serving capacity
# is concentrated in a few countries AND a few clouds (LLM_Deployment census,
# 320k public endpoints). The demand!=supply gap is why requests are served
# cross-region; panel (b) adds the provider dimension on top of geography.
SP = sys.argv[1]; OUTDIR = sys.argv[2]
d = json.load(open(f"{SP}/wc_stats.json"))

FONT = "STIXGeneral"
matplotlib.rcParams.update({"font.family": FONT,
    "axes.spines.top": False, "axes.spines.right": False})
LBL_FS, TICK_FS, LEG_FS, PANEL_FS = 16, 13, 12, 16

# ---------- demand side (WildChat requests) ----------
total = d["total_rows"]; cc = d["country_counts"]
# WildChat uses names; fold Greater China so it matches the census country code (CN)
def dem(*names): return 100*sum(cc.get(n,0) for n in names)/total
DEMAND = {
 "US": dem("United States"), "CN": dem("China","Hong Kong","Taiwan"),
 "RU": dem("Russia"), "DE": dem("Germany"), "GB": dem("United Kingdom"),
 "FR": dem("France"), "JP": dem("Japan"), "IN": dem("India"),
 "EG": dem("Egypt"), "SG": dem("Singapore"),
}

# ---------- supply side (LLM_Deployment census) ----------
SUP = {r["cc"]: float(r["supply_pct"]) for r in csv.DictReader(open(f"{SP}/supply_country_share.csv"))}

NAME = {"US":"USA","CN":"China","DE":"Germany","RU":"Russia","GB":"UK","FR":"France",
        "JP":"Japan","IN":"India","EG":"Egypt","SG":"Singapore","KR":"S.Korea",
        "AU":"Australia"}

# countries to contrast: chosen to span net-importers and net-exporters of serving
ORDER_A = ["US","CN","RU","DE","GB","JP","SG","IN","FR","EG"]

C_DEMAND = "#5b8fb9"
C_SUPPLY = "#D55E00"

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                               gridspec_kw={"width_ratios":[1.05, 1.0]})

# ===== panel (a): demand vs supply share per country =====
x = np.arange(len(ORDER_A)); w = 0.40
dvals = [DEMAND[c] for c in ORDER_A]
svals = [SUP.get(c,0.0) for c in ORDER_A]
axa.bar(x-w/2, dvals, w, color=C_DEMAND, label="Request share (demand)",
        edgecolor="white", linewidth=0.3)
axa.bar(x+w/2, svals, w, color=C_SUPPLY, label="Deployment share (supply)",
        edgecolor="white", linewidth=0.3)
axa.set_ylabel("Share (%)", fontsize=LBL_FS, fontname=FONT)
axa.set_xticks(x)
axa.set_xticklabels([NAME[c] for c in ORDER_A], rotation=40, ha="right",
                    fontsize=TICK_FS, fontname=FONT)
axa.tick_params(axis="y", labelsize=TICK_FS)
axa.set_ylim(0, 37)
axa.legend(frameon=False, prop={"family":FONT,"size":LEG_FS},
           loc="upper right", handlelength=1.1, handletextpad=0.5, borderpad=0.2)
# mark the cross-region cases where demand >> local supply
for xi,c in zip(x, ORDER_A):
    if DEMAND[c] > SUP.get(c,0)*2.0 and DEMAND[c] > 3:
        axa.annotate("", xy=(xi-w/2, DEMAND[c]+0.6), xytext=(xi-w/2, SUP.get(c,0)+0.6),
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))

# ===== panel (b): provider mix within top serving countries =====
rows = list(csv.DictReader(open(f"{SP}/supply_country_provider.csv")))
PROVS = ["Amazon","Cloudflare","Akamai","Alibaba","Tencent","China Telco",
         "Microsoft","Google","Hetzner","Other"]
PC = {"Amazon":"#FF9900","Cloudflare":"#984EA3","Akamai":"#377EB8","Alibaba":"#E41A1C",
      "Tencent":"#4DAF4A","China Telco":"#A65628","Microsoft":"#66C2A5",
      "Google":"#F781BF","Hetzner":"#999933","Other":"#CCCCCC"}
ccs = [r["cc"] for r in rows]
xb = np.arange(len(ccs))
bottom = np.zeros(len(ccs))
for p in PROVS:
    vals = np.array([float(r[p]) for r in rows])
    axb.bar(xb, vals, 0.72, bottom=bottom, color=PC[p], edgecolor="white",
            linewidth=0.25, label=p)
    bottom += vals
axb.set_ylabel("Provider mix (%)", fontsize=LBL_FS, fontname=FONT)
axb.set_xticks(xb)
axb.set_xticklabels([NAME.get(c,c) for c in ccs], rotation=40, ha="right",
                    fontsize=TICK_FS, fontname=FONT)
axb.tick_params(axis="y", labelsize=TICK_FS)
axb.set_ylim(0, 100)

# provider legend: compact row beneath both panels
handles = [Patch(facecolor=PC[p], label=p) for p in PROVS]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           ncol=5, frameon=False, prop={"family":FONT,"size":LEG_FS},
           handlelength=1.1, columnspacing=1.2, handletextpad=0.4)

fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.44, wspace=0.30)
fig.canvas.draw()
caps = {axa:"(a) Demand is dispersed, supply is not",
        axb:"(b) A few clouds serve each region"}
for ax,lab in caps.items():
    bb = ax.get_position()
    fig.text((bb.x0+bb.x1)/2, 0.15, lab, ha="center", va="bottom",
             fontsize=PANEL_FS, fontname=FONT)

fig.savefig(f"{OUTDIR}/motivation_geo.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUTDIR}/motivation_geo.png", dpi=150, bbox_inches="tight")
print("saved motivation_geo (demand vs supply + provider mix)")
print("demand:", {c:round(DEMAND[c],1) for c in ORDER_A})
print("supply:", {c:round(SUP.get(c,0),1) for c in ORDER_A})
