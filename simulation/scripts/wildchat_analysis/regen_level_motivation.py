"""regen_level_motivation.py — motivation figure for latency-derived Ludo levels.

Panels
  (a1/a2) Mechanism: net-TTFT-benefit line net(m) = m·C_prefill − WAN for a
          near pair (GB→DE) and a far pair (US→SG), with the credit staircase
          of fixed chunk levels vs latency-derived linear levels.  Shaded gap
          = benefit lost to quantisation; fixed levels below the break-even
          n₁ are dead (net < 0).
  (b)     Real-trace capture rate: fraction of achievable net TTFT benefit
          captured by each leveling scheme, over the WildChat multi-turn
          reuse-length distribution × all cross-region pairs (demand × supply
          weighted).  Annotated with average levels/pair (index+probe cost).
  (c)     Efficiency frontier: capture rate vs avg levels/pair for three
          scheme families — fixed chunk grids (latency-blind), latency-
          anchored exponential (n₁·b^j), latency-anchored linear (n₁ + j·α·n₁).

Data
  * Reuse lengths m: consecutive-turn events in datasets/wildchat.jsonl
    (turn k+1 reuses turn k's full prompt + generated reply — exact by
    construction of the per-turn cumulative-prefix expansion).
  * WAN / C_prefill / demand / supply: imported from the simulator so the
    figure and the system share one parameterisation.

Output: figures/motivation_levels.pdf (+ printed summary table)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulate import (                                    # type: ignore
    WAN_LATENCY_MS, C_PREFILL_MS_PER_TOKEN, CHUNK_SIZES,
    REGION_REQUEST_WEIGHTS, NODES_PER_REGION_LIST, REGION_NAMES,
    REUSE_FLOOR_TOKENS, LUDO_MAX_CUT, NUM_REGIONS,
)

CP      = C_PREFILL_MS_PER_TOKEN
CAP     = LUDO_MAX_CUT                       # 4096 — same cap for every scheme
FIGDIR  = ROOT / 'figures'
WC_PATH = ROOT / 'datasets' / 'wildchat.jsonl'

# ─── Reuse-length distribution from the WildChat multi-turn trace ────────────

def load_reuse_lengths():
    rows = [json.loads(l) for l in open(WC_PATH)]
    reuse = []
    prev = None
    for r in rows:                      # file order preserves turn order
        plen = max(1, len(r['workload']) // 4)
        if prev is not None and r['conv'] == prev['conv']:
            m = min(prev['plen'] + prev['gen'], plen)   # cached prefix of turn k+1
            reuse.append(m)
        prev = {'conv': r['conv'], 'plen': plen, 'gen': int(r.get('gen_len', 64))}
    return np.array(reuse, dtype=float)

# ─── Leveling schemes: cut chains per (S, X) pair ────────────────────────────

def n1(S, X):
    return float(REUSE_FLOOR_TOKENS[S][X])

def cuts_fixed(S, X, grid):
    return [c for c in grid if c <= CAP]

def cuts_exp(S, X, base):
    n = n1(S, X)
    if n <= 0 or n > CAP:
        return []
    out, c = [], n
    while c <= CAP and len(out) < 64:
        out.append(c)
        c *= base
    return out

def cuts_linear(S, X, alpha):
    n = n1(S, X)
    if n <= 0 or n > CAP:
        return []
    step = alpha * n
    out, c = [n], n + step
    while c <= CAP and len(out) < 256:
        out.append(c)
        c += step
    return out

# ─── Capture-rate evaluation ─────────────────────────────────────────────────

def credited(m_sorted, cuts):
    """Vector of credit c(m) = largest cut ≤ m (0 if none)."""
    if not cuts:
        return np.zeros_like(m_sorted)
    cuts_a = np.array(sorted(cuts))
    idx = np.searchsorted(cuts_a, m_sorted, side='right') - 1
    out = np.where(idx >= 0, cuts_a[np.clip(idx, 0, None)], 0.0)
    return out

def pair_weights():
    """Cross-pair weight ∝ demand(S) × supply(X)."""
    supply = np.array(NODES_PER_REGION_LIST, dtype=float)
    supply = supply / supply.sum()
    w = {}
    for S in range(NUM_REGIONS):
        for X in range(NUM_REGIONS):
            if S == X:
                continue
            w[(S, X)] = float(REGION_REQUEST_WEIGHTS[S] * supply[X])
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}

def evaluate(cut_fn, m, weights):
    """Absolute certified TTFT saving (ms/request), the exact-match upper
    bound (ms/request, scheme-independent), and avg levels/pair."""
    num = den = 0.0
    lvl_sum = 0.0
    for (S, X), w in weights.items():
        wan  = float(WAN_LATENCY_MS[S][X])
        cuts = cut_fn(S, X)
        c    = credited(m, cuts)
        cap_m = np.minimum(m, CAP)
        num += w * np.maximum(0.0, c * CP - wan).mean()
        den += w * np.maximum(0.0, cap_m * CP - wan).mean()
        lvl_sum += w * len(cuts)
    return num, den, lvl_sum

# ─── Figure (style matched to motivation_geo_single) ─────────────────────────

FONT = 'STIXGeneral'
matplotlib.rcParams.update({'font.family': FONT,
                            'axes.spines.top': False,
                            'axes.spines.right': False})
# Sized for one ACM column (3.35 in), same height as motivation_evict.
LBL_FS, TICK_FS, LEG_FS, CAP_FS = 20, 18, 19, 20

# Muted palette shared with motivation_geo_single / motivation_evict.
C_FIXED = '#C17878'   # dusty red — fixed chunk grid
C_MULT  = '#B39B84'   # oat brown — latency-anchored, multiplicative
C_ADD   = '#7FA3C4'   # blue      — latency-anchored, additive (ours)
C_GREY  = '#CFCbC4'

SHOW_PAIRS = [(5, 2), (0, 2), (0, 1), (0, 4)]   # GB→DE, US→DE, US→CN, US→SG


def panel_problem(ax):
    """One row per region pair: grey bar = match lengths too short for the
    forwarding to pay off on that pair; red dashed verticals = the
    one-size-fits-all fixed chunk grid."""
    pairs = sorted(SHOW_PAIRS, key=lambda p: n1(*p))     # near → far
    for c in CHUNK_SIZES:
        ax.axvline(c, color=C_FIXED, lw=2.4, ls=(0, (3, 2)), zorder=3)
    for row, (S, X) in enumerate(pairs):
        ax.barh(row, n1(S, X), height=0.58, color=C_GREY, alpha=0.9,
                edgecolor='none', zorder=1)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=C_GREY, alpha=0.9, label='too short to pay off'),
        Line2D([], [], color=C_FIXED, lw=2.4, ls=(0, (3, 2)),
               label='fixed chunk levels'),
    ]
    ax.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False,
              prop={'family': FONT, 'size': LEG_FS},
              handlelength=1.3, columnspacing=0.9, handletextpad=0.4,
              labelspacing=0.2, borderpad=0.0)
    # Slack past CAP so the 4096 line is not half-clipped by the frame.
    ax.set_xlim(0, CAP * 1.02)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_ylim(-0.55, len(pairs) - 0.45)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f'{REGION_NAMES[S]}→{REGION_NAMES[X]}'
                        for (S, X) in pairs],
                       fontsize=TICK_FS, fontname=FONT)
    ax.set_xlabel('matched prefix length (tokens)', fontsize=LBL_FS,
                  fontname=FONT, labelpad=1.5)
    ax.tick_params(axis='x', labelsize=TICK_FS)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)


def panel_sweep(ax, m, weights):
    pts_fixed = []
    for L in range(2, 13):
        grid = sorted({int(round(v)) for v in np.geomspace(128, CAP, L)})
        pts_fixed.append(
            evaluate(lambda S, X, g=grid: cuts_fixed(S, X, g), m, weights))
    bases  = (4.0, 3.0, 2.0, 1.7, 1.5, 1.3, 1.2, 1.1)
    pts_exp = [evaluate(lambda S, X, b=b: cuts_exp(S, X, b), m, weights)
               for b in bases]
    alphas = (4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
    pts_lin = [evaluate(lambda S, X, a=a: cuts_linear(S, X, a), m, weights)
               for a in alphas]

    achievable = pts_lin[0][1]          # exact-match bound
    ax.axhline(achievable, color='#666666', lw=2.2, ls='--')
    ax.text(12.7, achievable * 1.03, 'exact match', ha='right', va='bottom',
            fontsize=LEG_FS, fontname=FONT, color='#666666')
    for name, col, pts in (
            ('fixed grid', C_FIXED, pts_fixed),
            ('multiply', C_MULT, pts_exp),
            ('add', C_ADD, pts_lin)):
        ax.plot([p[2] for p in pts], [p[0] for p in pts], 'o-',
                color=col, lw=3.0, ms=6.4, label=name)
    k = alphas.index(0.5)
    ax.plot(pts_lin[k][2], pts_lin[k][0], '*', ms=26, color=C_ADD,
            markeredgecolor='black', markeredgewidth=1.4, zorder=6)
    ax.annotate(r'$\alpha=0.5$', (pts_lin[k][2], pts_lin[k][0]),
                xytext=(12, -28), textcoords='offset points',
                fontsize=LEG_FS, fontname=FONT)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, achievable * 1.28)
    ax.set_xlabel('table levels per region pair', fontsize=LBL_FS,
                  fontname=FONT, labelpad=1.5)
    ax.set_ylabel('TTFT saved\n(ms/req)', fontsize=LBL_FS, fontname=FONT,
                  labelpad=-2)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=3,
              frameon=False, prop={'family': FONT, 'size': LEG_FS},
              handlelength=1.3, columnspacing=0.9, handletextpad=0.4,
              labelspacing=0.2, borderpad=0.0)


def main():
    m = load_reuse_lengths()
    print(f'reuse events: {len(m)}  (p50={np.percentile(m,50):.0f}, '
          f'p90={np.percentile(m,90):.0f}, max={m.max():.0f} tokens)')
    weights = pair_weights()

    fig, axes = plt.subplots(2, 1, figsize=(6.7, 4.24))
    panel_problem(axes[0])
    panel_sweep(axes[1], m, weights)
    caps = ['(a) Fixed chunk sizes cannot fit all region pairs.',
            '(b) TTFT saved vs. levels per pair.']
    for ax, cap in zip(axes, caps):
        ax.text(0.5, -0.98, cap, transform=ax.transAxes,
                ha='center', va='top', fontsize=CAP_FS, fontname=FONT)

    fig.subplots_adjust(left=0.17, right=0.985, top=0.92, bottom=0.23,
                        hspace=2.1)
    out = FIGDIR / 'motivation_levels.pdf'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(FIGDIR / 'motivation_levels.png', dpi=300,
                bbox_inches='tight')
    print(f'saved → {out}')


if __name__ == '__main__':
    main()
