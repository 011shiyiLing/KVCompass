import json, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SP = sys.argv[1]
d = json.load(open(f"{SP}/wc_stats.json"))

# ---------- conference style ----------
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# Okabe-Ito-ish, fixed continent colors
CC = {"Asia":"#0072B2","Europe":"#D55E00","N.America":"#009E73",
      "S.America":"#CC79A7","Africa":"#E69F00","Oceania":"#56B4E9",
      "Other":"#999999","Unknown":"#cccccc"}
# crude country->continent for coloring the country bars
import pycountry_convert as pcc
CONT={"AF":"Africa","AS":"Asia","EU":"Europe","NA":"N.America","SA":"S.America","OC":"Oceania"}
FB={"Russia":"EU","Hong Kong":"AS","Taiwan":"AS","South Korea":"AS","Vietnam":"AS",
    "DR Congo":"AF","Iran":"AS","Czechia":"EU"}
def cont_of(name):
    if name in FB: return CONT[FB[name]]
    try: return CONT.get(pcc.country_alpha2_to_continent_code(pcc.country_name_to_country_alpha2(name)),"Other")
    except: return "Other"

total = d["total_rows"]
cc = d["country_counts"]
cont = d["continent_counts"]

# ============================================================
# FIGURE 1A : Pareto (country bars colored by continent + cumulative)
# ============================================================
TOPN = 15
items = list(cc.items())[:TOPN]
names = [k for k,_ in items]
vals = np.array([v for _,v in items], float)
pct = 100*vals/total
cum = np.cumsum([v for _,v in cc.items()])/total*100
cum_top = cum[:TOPN]
colors = [CC[cont_of(n)] for n in names]

fig, ax = plt.subplots(figsize=(3.4, 2.0))
x = np.arange(TOPN)
ax.bar(x, pct, color=colors, width=0.78, edgecolor="white", linewidth=0.3)
ax.set_ylabel("Share of requests (%)")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right")
ax.set_ylim(0, max(pct)*1.15)
ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
ax2.plot(x, cum_top, color="0.25", lw=1.1, marker="o", ms=2.4, zorder=5)
ax2.set_ylabel("Cumulative (%)", color="0.25")
ax2.set_ylim(0, 100); ax2.tick_params(colors="0.25")
# continent legend
from matplotlib.patches import Patch
order=["Asia","Europe","N.America","Africa","S.America","Oceania"]
handles=[Patch(facecolor=CC[c], label=f"{c} ({100*cont[c]/total:.0f}%)") for c in order if c in cont]
ax.legend(handles=handles, ncol=2, loc="upper right", handlelength=1.0,
          columnspacing=0.8, handletextpad=0.4, borderpad=0.2, bbox_to_anchor=(1.0,1.03))
ax.text(0.34, 0.50, f"{d['n_countries']} countries\n{d['n_users']/1000:.0f}K unique users",
        transform=ax.transAxes, fontsize=7, va="center", ha="left", color="0.35")
fig.savefig(f"{SP}/fig1a_pareto.png"); fig.savefig(f"{SP}/fig1a_pareto.pdf"); plt.close(fig)

# ============================================================
# FIGURE 1B : two-panel (continent shares | top countries)
# ============================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.6, 2.0), gridspec_kw={"width_ratios":[1,1.25]})
order2=[c for c in ["Asia","Europe","N.America","Africa","S.America","Oceania","Other"] if c in cont]
cv=[cont[c] for c in order2]
yL=np.arange(len(order2))[::-1]
axL.barh(yL, [100*v/total for v in cv], color=[CC[c] for c in order2], height=0.7)
axL.set_yticks(yL); axL.set_yticklabels(order2)
axL.set_xlabel("Share of requests (%)")
for y,v in zip(yL,cv): axL.text(100*v/total+0.6, y, f"{100*v/total:.0f}%", va="center", fontsize=6.5)
axL.set_xlim(0, max(100*v/total for v in cv)*1.18)
axL.set_title("(a) By world region", fontsize=8)
# right: top countries
TOP2=12
it=list(cc.items())[:TOP2]
nm=[k for k,_ in it]; vv=[100*v/total for _,v in it]
yR=np.arange(TOP2)[::-1]
axR.barh(yR, vv, color=[CC[cont_of(n)] for n in nm], height=0.72)
axR.set_yticks(yR); axR.set_yticklabels(nm)
axR.set_xlabel("Share of requests (%)")
axR.set_title("(b) Top countries", fontsize=8)
axR.set_xlim(0, max(vv)*1.12)
fig.tight_layout(w_pad=1.2)
fig.savefig(f"{SP}/fig1b_twopanel.png"); fig.savefig(f"{SP}/fig1b_twopanel.pdf"); plt.close(fig)

# ============================================================
# FIGURE 3 : cross-region prefix sharing (double panel)
# ============================================================
K = 256
s = d["sharing"][str(K)]
scat = np.array(s["scatter"], float)  # [count, ncountry, ncont, nusers]
cnt = scat[:,0]; ncountry = scat[:,1]; ncontv = scat[:,2]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"width_ratios":[1.15,1]})

# panel (a): popularity vs geographic spread + fraction-cross-region trend
xr = cnt; yr = ncountry
xreg = ncontv>=2
axa.scatter(xr[~xreg], yr[~xreg], s=7, c="0.72", alpha=0.5, lw=0, label="within one region")
axa.scatter(xr[xreg], yr[xreg], s=10, c="#D55E00", alpha=0.65, lw=0, label="spans ≥2 regions")
axa.set_xscale("log"); axa.set_yscale("log")
axa.set_xlabel("Prefix popularity (# requests)")
axa.set_ylabel("# distinct countries")
axa.set_ylim(0.8, max(ncountry)*1.5)
axa.set_title("(a) The more popular a prefix,\nthe more regions it spans", fontsize=8)
axa.legend(loc="upper left", handlelength=1.1, handletextpad=0.4, borderpad=0.2)
axa.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
# secondary axis: fraction spanning >=2 regions vs popularity bin
bins=np.logspace(np.log10(2), np.log10(cnt.max()), 8)
fb_x=[]; fb_y=[]
for b in range(1,len(bins)):
    m=(cnt>=bins[b-1])&(cnt<bins[b])
    if m.sum()>=15:
        fb_x.append(np.sqrt(bins[b-1]*bins[b])); fb_y.append(100*np.mean(ncontv[m]>=2))
axc=axa.twinx(); axc.spines["top"].set_visible(False)
axc.plot(fb_x, fb_y, color="0.15", lw=1.3, marker="s", ms=3.2, zorder=6,
         label="% cross-region (binned)")
axc.set_ylim(0,100); axc.set_ylabel("% spanning ≥2 regions", color="0.15")
axc.tick_params(colors="0.15")

# panel (b): reuse-volume decomposition across thresholds
Ks=[64,256,512]
single=[]; multi_same=[]; xcont=[]
for k in Ks:
    sk=d["sharing"][str(k)]
    vs=sk["vol_shared"]; vxc=sk["vol_xcountry"]; vxr=sk["vol_xregion"]
    single.append(100*(vs-vxc)/vs)
    multi_same.append(100*(vxc-vxr)/vs)
    xcont.append(100*vxr/vs)
xk=np.arange(len(Ks))
axb.bar(xk, single, color="0.78", label="within one country", width=0.6)
axb.bar(xk, multi_same, bottom=single, color="#56B4E9", label="cross-country, same region", width=0.6)
axb.bar(xk, xcont, bottom=np.array(single)+np.array(multi_same), color="#D55E00",
        label="cross-region (cross-continent)", width=0.6)
for i in range(len(Ks)):
    axb.text(i, 100-xcont[i]/2, f"{xcont[i]:.0f}%", ha="center", va="center",
             color="white", fontsize=7, fontweight="bold")
axb.set_xticks(xk); axb.set_xticklabels([f"{k}" for k in Ks])
axb.set_xlabel("Shared-prefix length (tokens)")
axb.set_ylabel("Reuse volume (%)")
axb.set_ylim(0,100)
axb.set_title("(b) A large share of prefix reuse\nis cross-region", fontsize=8)
hb,lb=axb.get_legend_handles_labels()
fig.legend(hb, lb, loc="lower center", ncol=3, handlelength=1.1, handletextpad=0.4,
           columnspacing=1.2, borderpad=0.2, bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(w_pad=2.0, rect=[0,0.08,1,1])
fig.savefig(f"{SP}/fig3_crossregion.png"); fig.savefig(f"{SP}/fig3_crossregion.pdf"); plt.close(fig)

print("rendered fig1a, fig1b, fig3")
