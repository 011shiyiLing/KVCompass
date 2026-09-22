import json, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SP = sys.argv[1]; OUTDIR = sys.argv[2]
d = json.load(open(f"{SP}/wc_stats.json"))

FONT = "STIXGeneral"
matplotlib.rcParams.update({"font.family": FONT,
    "axes.spines.top": False, "axes.spines.right": False})
LBL_FS, TICK_FS, LEG_FS, PANEL_FS = 18, 16, 16, 18

C_WITHIN = "#b8b8b8"
C_XREGION = "#D55E00"

K = 256
s = d["sharing"][str(K)]
sc = np.array(s["scatter"], float)            # [count, n_country, n_cont, n_users]
cnt, nco, ncont = sc[:,0], sc[:,1], sc[:,2]
xreg = ncont >= 2

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.5),
                               gridspec_kw={"width_ratios":[1,1]})

# ---------- panel (a): scatter, 2-class, larger points ----------
axa.scatter(cnt[~xreg], nco[~xreg], s=26, c=C_WITHIN,  alpha=0.55, lw=0)
axa.scatter(cnt[xreg],  nco[xreg],  s=30, c=C_XREGION, alpha=0.65, lw=0)
axa.set_xscale("log"); axa.set_yscale("log")
axa.set_xlabel("Prefix reuse count", fontsize=LBL_FS, fontname=FONT)
axa.set_ylabel("# countries", fontsize=LBL_FS, fontname=FONT)
axa.tick_params(axis="both", labelsize=TICK_FS)
axa.set_ylim(0.8, max(nco)*1.6)
axa.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)

# ---------- panel (b): cross-region share rises with reuse count ----------
THR = [2,5,10,20,50]
frac = [100*(xreg[cnt>=t]).sum()/ (cnt>=t).sum() for t in THR]
xb = np.arange(len(THR))
axb.bar(xb, frac, width=0.66, color=C_XREGION, edgecolor="white", linewidth=0.4)
for x,v in zip(xb,frac):
    axb.text(x, v+1.2, f"{v:.0f}", ha="center", va="bottom",
             fontsize=TICK_FS-2, fontname=FONT)
axb.set_xticks(xb); axb.set_xticklabels([f"≥{t}" for t in THR],
                                         fontsize=TICK_FS, fontname=FONT)
axb.set_xlabel("Prefix reuse count", fontsize=LBL_FS, fontname=FONT)
axb.set_ylabel("% cross-region", fontsize=LBL_FS, fontname=FONT)
axb.tick_params(axis="y", labelsize=TICK_FS)
axb.set_ylim(0, max(frac)*1.20)

# ---------- shared 2-entry legend ----------
handles=[Patch(facecolor=C_WITHIN,label="within one region"),
         Patch(facecolor=C_XREGION,label="cross-region (≥2 regions)")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,0.99),
           ncol=2, frameon=False, prop={"family":FONT,"size":LEG_FS},
           handlelength=1.3, columnspacing=1.6, handletextpad=0.5)

# ---------- short neutral panel captions ----------
fig.subplots_adjust(left=0.10, right=0.985, top=0.85, bottom=0.30, wspace=0.34)
fig.canvas.draw()
caps={axa:"(a) Reuse count vs. countries", axb:"(b) Cross-region vs. reuse count"}
for ax,lab in caps.items():
    bb=ax.get_position()
    fig.text((bb.x0+bb.x1)/2, 0.045, lab, ha="center", va="bottom",
             fontsize=PANEL_FS, fontname=FONT)

fig.savefig(f"{OUTDIR}/motivation_cross_region.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUTDIR}/motivation_cross_region.png", dpi=150, bbox_inches="tight")
print("saved fig3; cross-region%% by thr:", [round(f) for f in frac])
