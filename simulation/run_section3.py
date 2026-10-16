"""
run_section3.py — Main driver for the large-scale simulation figures.

Usage:
    python3 run_section3.py [--quick] [--figs 125] [--n-tenants 50]
                            [--workload mixed|toolbench|loogle|aaps|wildchat|lmsys]

Flags:
    --quick      Use fewer requests / arrival-rate points (fast debug run).
    --workload   Workload for figs 1/2/5 (default: mixed).
    --figs       Which figures to generate (default: 1234; see --help).
    --seed       Random seed (default: 42).

Outputs (saved to results/):
    fig1_latency_throughput.pdf   — P99 TTFT vs arrival rate curves
    fig2_ttft_cdf.pdf (+ fig2b/fig2c CDFs and TTFT/E2E box plots)
    fig3b_cache_hit_rate.pdf      — normalized prefix-cache hit rate per scheme
                                    (FLOPs-saved stats printed + cached as JSON)
    fig4_heatmap.pdf              — steady-state per-node load-factor heatmap
    fig5_throughput.pdf           — closed-loop throughput bar chart
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, OrderedDict, Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Tuple

# ── project path ──────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import simulate as _sim          # module handle: sweepable globals (θ etc.)
from simulate import (
    Simulator, WorkloadGenerator,
    make_nodes, make_discovery,
    SCHEME_LABELS, SCHEME_COLORS,
    TTFT_SLO_MS, NUM_NODES, NUM_REGIONS,
    NODE_REGION, REGION_OFFSETS, REGION_REQUEST_WEIGHTS,
    NODE_VRAM_TOKENS, NODE_DRAM_TOKENS,
    Request, RequestRecord,
    C_DECODE_MS_PER_TOKEN, C_DECODE_STEP_MS_PER_TOKEN,
    C_PREFILL_MS_PER_TOKEN, WAN_LATENCY_MS,
    DIRECTORY_REGION,
    DIRECTORY_SERVICE_RATE_RPS, DIRECTORY_SERVICE_MS,
    CHUNK_SIZES,
    REUSE_FLOOR_TOKENS, LUDO_PAIR_CUTS,
    LUDO_LB_BCAST_MS, LUDO_LB_EWMA_BETA,
    SKYWALKER_PROBE_INTERVAL_MS,
    _nearest_least_loaded,
    _transfer_ms,
)

RESULTS_DIR = HERE / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SCHEME_HATCHES = {
    'ludo':              '|||',
    'ludo_coarse':       '///',
    'radix_tree':        '\\\\',
    'consistent_hash':   'xxx',
    'kademlia':          '...',
    'chord':             '///',
    'skywalker':         '++',
    'dynamo':            '--',
    'oracle':            'ooo',
    'central_directory': 'ooo',
}

SCHEME_LINESTYLES = {
    # All schemes use solid lines in the line-chart figures (fig1/2/2b).
    # Scheme identity is carried by color + marker + fill; using a single
    # linestyle avoids the mixed-dash clutter in the legend.
    'central_directory': '-',
    'ludo_coarse':       '-',
    'oracle':            '-',
    'ludo':              '-',
    'radix_tree':        '-',
    'consistent_hash':   '-',
    'kademlia':          '-',
    'chord':             '-',
    'skywalker':         '-',
    'dynamo':            '-',
}

# ─── Morandi palette for line/CDF plots (Figs 1, 2, 2b) ─────────────────────
# KVCompass carries the palette's red (swapped with consistent_hash's old
# blue); chord takes the yellow pair so it no longer clashes with dynamo's
# purple.  Edge colors here pair with the light fills in LINE_FILL below.
LINE_COLORS = {
    'ludo':              '#985365',
    'ludo_coarse':       '#c48b9f',   # lighter KVCompass red — ablation twin
    'radix_tree':        '#8a8c6f',
    'consistent_hash':   '#7e97ad',
    'kademlia':          '#b78D31',
    'chord':             '#b78D31',
    'skywalker':         '#6f9c8f',
    'dynamo':            '#9c7e97',
    'oracle':            '#7A6E5A',
    'central_directory': '#7A6E5A',
}
LINE_FILL = {
    'ludo':              '#eba79c',
    'ludo_coarse':       '#f3cdc5',
    'radix_tree':        '#a3b7a0',
    'consistent_hash':   '#ACBFEA',
    'kademlia':          '#FFDAA6',
    'chord':             '#FFDAA6',
    'skywalker':         '#b9d8cd',
    'dynamo':            '#d8c3d4',
    'oracle':            '#A5957D',
    'central_directory': '#A5957D',
}
LINE_LABELS = dict(SCHEME_LABELS)
LINE_LABELS['ludo']              = 'KVCompass'
LINE_LABELS['ludo_coarse']       = 'KVCompass-CG'
LINE_LABELS['radix_tree']        = 'PlanetServe'
LINE_LABELS['central_directory'] = 'Central Directory'

# ─── Experiment parameters ────────────────────────────────────────────────────

# Schemes for fig1/fig2/fig2b.  Central Directory replaces the old
# theoretical Oracle as the reference baseline — see _run_central_directory
# docstring for why (a realistic deployable upper bound, not a physically
# impossible 100 %-cache-hit lower bound).  Figures fig3b and fig4 still
# reference the old Oracle semantics via their own code paths.
# Chord is the DHT baseline in the run set; its Kademlia twin (identical
# hash-placement mechanics, cheaper parallel-lookup overhead constant) is kept
# implemented but not run — swap by exchanging 'chord'/'kademlia' here and in
# the fig3/fig3c/fig4 scheme lists below.
ALL_SCHEMES = ['central_directory', 'ludo', 'ludo_coarse', 'radix_tree',
               'consistent_hash', 'chord', 'skywalker', 'dynamo']

# Per-scheme marker glyph for fig1's latency-throughput line chart.  Defined
# at module level so the combined panel (fig1+fig2+fig2b) can pull legend
# handles consistent with the standalone fig1 PDF.
MARKERS = {'central_directory': 'D', 'oracle': 'D', 'ludo': 'o',
           'ludo_coarse': 'h', 'radix_tree': 's', 'consistent_hash': '^',
           'kademlia': 'v', 'chord': 'P', 'skywalker': 'X', 'dynamo': '*'}

# Arrival-rate sweep for Figure 1 (req/s, global across all 8 regions).
# Spans 200–6000 req/s.  With 4-way batching (simulate.MAX_BATCH_SIZE)
# the per-node capacity rises ~4×, so the schemes' saturation knees
# (≈1400 req/s for the DHTs, ≈2800 for the coordinator-bound Central
# Directory, ≈3200 for KVCompass) land in the *middle* of this range —
# each curve shows a visible flat region, knee, and post-saturation climb.
RATES_FULL  = [200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000, 2500,
               3000, 3500, 4000, 4500, 5000, 5500, 6000]
RATES_QUICK = [200, 1000, 2000, 3500, 5000, 6000]

# ── Fig 1 latency-throughput measurement ──────────────────────────────────────
# Each arrival-rate point simulates a fixed wall-clock window of Poisson
# arrivals (N = rate × DURATION_S).  Below saturation the p99 TTFT is the
# scheme's steady-state latency; past saturation the queue has no steady
# state, so the p99 reflects the backlog that builds up over the window —
# a curve that keeps climbing with arrival rate, the way a real
# fixed-duration load test behaves.  DURATION_S also feeds the fig2/3/4
# knee probe via `rate_to_n_warmup`.
DURATION_S_FULL  = 10.0     # per-rate sim time for --full
DURATION_S_QUICK = 2.0      # per-rate sim time for --quick
WARMUP_S_FULL    = 2.0      # warmup time prefix (discarded)
WARMUP_S_QUICK   = 0.5

# ── Fig 2 / 2b CDF operating point ────────────────────────────────────────────
# The TTFT / E2E CDFs are taken at a single arrival rate, chosen so every
# scheme's curve is informative:
#   - below KVCompass's knee (≈3300) → KVCompass runs in steady state,
#     its CDF a tight fast body plus a thin queue-burst tail rather than
#     the degenerate vertical line a rate at/above the knee produces;
#   - just below the Central Directory coordinator's knee (≈2800) → the
#     coordinator queue is loaded enough to spread CD's distribution
#     while keeping it healthy (not yet diverging);
#   - well past the DHT / PlanetServe knees (≈1400 / 2300) → those remain
#     clearly saturated.
FIG2_RATE = 2750

# ── Llama-70B rate rescale (SIM_MODEL=70b) ────────────────────────────────────
# 70B prefill is 3× slower per token (0.15 vs 0.05 ms), so every scheme's
# saturation knee shifts ~3× left.  Rescale the sweep and the fig2 CDF
# operating point so the knees land mid-sweep again, preserving each
# figure's methodology (flat region → knee → post-saturation climb; fig2
# below the KVCompass knee, past the DHT/PlanetServe knees).
if _sim.SIM_MODEL == '70b':
    RATES_FULL  = [100, 150, 200, 300, 400, 500, 600, 700, 800, 900,
                   1000, 1200, 1400, 1600, 1800, 2000]
    RATES_QUICK = [100, 300, 700, 1200, 1600, 2000]
    FIG2_RATE   = 900

# B1 16-region/128-worker: env override for the scaled operating point.
_f2 = os.environ.get('SIM_FIG2_RATE')
if _f2:
    FIG2_RATE = float(_f2)

# Legacy constants kept for Fig 3 (one-pass over the dataset, doesn't
# use rate-dependent sizing).
N_REQUESTS_QUICK = 2_000
N_REQUESTS_FULL  = 100_000
WARMUP_QUICK     = 500
WARMUP_FULL      = 10_000


def rate_to_n_warmup(rate_rps: float, quick: bool) -> Tuple[int, int]:
    """Return (n_requests, n_warmup) for a given arrival rate under the
    fixed-duration methodology.  Floors ensure even the lowest rate has
    enough samples for a stable p99 and a non-trivial warmup."""
    dur = DURATION_S_QUICK if quick else DURATION_S_FULL
    wup = WARMUP_S_QUICK   if quick else WARMUP_S_FULL
    n_req    = max(500, int(rate_rps * dur))
    n_warmup = max(100, int(rate_rps * wup))
    return n_req, n_warmup


# ─── Utilities ────────────────────────────────────────────────────────────────

def p99(values):
    return float(np.percentile(values, 99)) if values else float('inf')

def p50(values):
    return float(np.percentile(values, 50)) if values else float('inf')


# (slo_violation_rate was deleted along with the SLO removal — the
#  RequestRecord.slo_met field is retained only as an unused internal
#  flag for backward compat and is not consulted by any plotter or knee
#  finder in this file.)

def _run_central_directory(arrival_rate_rps, workload, n_requests, warmup, seed, wgen):
    """Central-directory routing: a centralised-coordinator baseline.

    Data-path topology (what the request actually traverses):
        user → edge → directory → edge → KV node → edge → user
    Compared to Ludo's shorter path (user → edge → KV → edge → user),
    Central Directory pays one extra WAN round-trip for the dispatch-time
    directory query.

    What it has, and what it pays:
      - **Global cache directory.**  The coordinator tracks where each
        prefix was last cached, so a lookup finds an existing copy
        whenever one is still resident — the structural advantage over
        the DHTs, which cannot index locally-generated caches.
      - **Realistic finite per-node cache.**  Caches are the same bounded
        VRAM/DRAM LRU as every other scheme: entries are evicted and
        demoted, so a prefix that has aged out of its holder's cache is a
        genuine miss (or a DRAM-tier reload).  This is the key change
        from the earlier idealised "infinite, never-evicted" model — it
        gives CD a realistic spread of hit/miss latencies.
      - **Single-coordinator bottleneck.**  Every dispatch is serialised
        through one M/D/1 coordinator (DIRECTORY_SERVICE_RATE_RPS); past
        that rate the queue grows unboundedly.  This is the scalability
        ceiling distributed planes (Ludo, DHTs) structurally avoid.
      - **Cross-region lookup RTT.**  Each request pays a WAN round-trip
        to the directory's home region (DIRECTORY_REGION); the cost
        varies with the request's own region (0 ms same-region up to
        ~300 ms farthest) rather than a flat worst-case constant.

    Latency accounting per request:
      ttft = directory_lookup_rtt(region)   # edge↔directory propagation
           + directory_queue_ms             # M/D/1 wait at coordinator
           + directory_service_ms           # coordinator processing
           + node_queue + prefill_or_attention
           + serving-leg WAN propagation / DRAM-reload cost
      e2e  = ttft + decode + return-path half-RTT

    Routing rule:
      1. Join coordinator queue (M/D/1, μ = DIRECTORY_SERVICE_RATE_RPS).
      2. Look the prefix up in the directory.  If it points to a node
         that *still* has the prefix cached, route there and prefill only
         the residual tokens.
      3. Otherwise (never seen, or aged out of its holder's cache) route
         to the globally least-loaded node and prefill the whole prompt.
      4. Insert the prompt into the chosen node's finite LRU cache and
         record it as the prefix's current holder.
    """
    rate_ms = arrival_rate_rps / 1000.0
    rng     = np.random.default_rng(seed)
    total   = warmup + n_requests

    # Save wgen RNG so subsequent scheme calls see the same stream
    wgen_state = wgen.rng.bit_generator.state

    nodes = make_nodes()
    for node in nodes:
        node.reset()

    # Global directory: prefix_id -> the node that most recently cached
    # the prefix.  A routing hint only — the *actual* match is read from
    # that node's finite LRU cache, so a prefix it has since evicted
    # simply yields a miss.
    directory: Dict[int, int] = {}

    # Coordinator M/D/1 queue: single-server, deterministic service time.
    # `directory_next_free` = earliest wall-clock time the coordinator can
    # begin processing a newly arriving request.  Persists across warmup.
    directory_next_free = 0.0

    records = []
    t = 0.0
    for i in range(total):
        t += rng.exponential(1.0 / rate_ms)
        req = wgen.sample(i, t, workload)
        ck  = req.content_keys or {}

        # Coordinator M/D/1: wait until the coordinator is free, then hold
        # the slot for DIRECTORY_SERVICE_MS.  Queue grows unboundedly once
        # arrival rate exceeds DIRECTORY_SERVICE_RATE_RPS.
        dir_service_start   = max(t, directory_next_free)
        dir_queue_ms        = dir_service_start - t
        directory_next_free = dir_service_start + DIRECTORY_SERVICE_MS

        # Resolve against the holder's *finite* cache: a hit only if the
        # holder still has (a contiguous prefix of) the prompt resident.
        hint = directory.get(req.prefix_id)
        if hint is not None:
            match_len, hit_tier = nodes[hint].cache_match_len(ck, req.prefix_len)
        else:
            match_len, hit_tier = 0, 'MISS'

        if match_len > 0:
            nid = hint
        else:
            # Never cached, or aged out — full prefill on the least-loaded node.
            nid       = min(range(NUM_NODES),
                            key=lambda n: nodes[n].queue_ms(t))
            match_len = 0
            hit_tier  = 'MISS'

        node               = nodes[nid]
        effective_residual = req.prefix_len - match_len
        ttft_ms = node.schedule(effective_residual, req.prefix_len, t)

        # Directory lookup: WAN round-trip to the directory's home region
        # (varies with the request's region) + coordinator queue + service.
        dir_lookup_ms = 2.0 * float(WAN_LATENCY_MS[DIRECTORY_REGION][req.region_id])
        ttft_ms += dir_lookup_ms + dir_queue_ms + DIRECTORY_SERVICE_MS

        # Serving-leg cost for the cached match: cross-region WAN
        # propagation, plus a DRAM reload penalty if the match was demoted.
        ttft_ms += _transfer_ms(hit_tier, node.region_id,
                                req.region_id, match_len)

        # Insert the prompt into the chosen node's finite LRU cache; it is
        # now the prefix's current holder.
        node.insert(ck, req.prefix_len, t)
        directory[req.prefix_id] = nid

        # Return-path: last generated token travels from serving node back
        # to user.  Same-region: free.  Cross-region: half the symmetric
        # RTT (matches Ludo's accounting in Simulator.run).
        if node.region_id == req.region_id:
            return_latency_ms = 0.0
        else:
            return_latency_ms = WAN_LATENCY_MS[node.region_id][req.region_id] / 2.0

        e2e_ms = (ttft_ms
                  + req.gen_len * C_DECODE_STEP_MS_PER_TOKEN
                  + return_latency_ms)

        if i >= warmup:
            records.append(RequestRecord(
                request_id       = i,
                prefix_id        = req.prefix_id,
                prefix_len       = req.prefix_len,
                arrival_time     = t,
                ttft_ms          = ttft_ms,
                e2e_latency_ms   = e2e_ms,
                match_len        = match_len,
                cache_hit        = match_len > 0,
                flops_saved_frac = (match_len / req.prefix_len
                                    if req.prefix_len else 0.0),
                node_id          = nid,
                slo_met          = ttft_ms <= TTFT_SLO_MS,
                hit_tier         = hit_tier,
                gen_len          = req.gen_len,
            ))

    # Restore wgen RNG state
    wgen.rng.bit_generator.state = wgen_state
    return records, nodes


def run_scheme(scheme, arrival_rate_rps, workload, n_requests, warmup, seed, wgen):
    # Paired comparisons: every (scheme, rate) run consumes a bit-identical
    # request stream instead of wherever the previous run left the shared RNG.
    wgen.reset_stream(seed)
    if scheme == 'central_directory':
        return _run_central_directory(arrival_rate_rps, workload, n_requests,
                                      warmup, seed, wgen)
    nodes     = make_nodes()
    discovery = make_discovery(scheme)
    sim = Simulator(nodes, discovery, wgen,
                    arrival_rate_rps=arrival_rate_rps,
                    scheme=scheme, workload=workload,
                    warmup=warmup, seed=seed)
    return sim.run(n_requests), nodes

def save_fig(fig, name):
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  saved → {path}')
    plt.close(fig)


def int_like_fmt(x, _pos=None):
    """Tick formatter: integer-valued labels render without decimals
    (0.0 → '0', 1.0 → '1'); non-integer values keep their natural form."""
    if x == int(x):
        return f'{int(x)}'
    return f'{x:g}'


def pad_log_xaxis_for_note(ax):
    """Extend a log-x axis so a corner-pinned $\\log_{10}$ note fits past
    the rightmost tick **without** introducing the next decade's tick.

    Strategy: explicitly set major x-ticks to exactly those decades
    covered by the data, then push the right edge to 10^(N+0.85) ≈ 7 ×
    last decade — enough breathing room for the note at axes-x = 1.0,
    yet strictly < 10 × last decade so LogLocator has no opportunity to
    render the next tick.
    """
    xmin_axis, xmax_data = ax.get_xlim()
    xmax_data = max(xmax_data, xmin_axis * 10)  # guard against degenerate range
    max_dec = int(np.floor(np.log10(xmax_data)))
    min_dec = int(np.ceil(np.log10(max(xmin_axis, 1e-3))))
    # If data crosses deep into the "next" decade (e.g. xmax_data at 9.5×10^N)
    # there's no room for the note between the last tick and the next decade,
    # so promote max_dec and accept the next tick.
    if xmax_data > 10 ** (max_dec + 0.7):
        max_dec += 1
    right_edge = 10 ** (max_dec + 0.85)
    ax.set_xlim(xmin_axis, right_edge)
    ax.set_xticks(10.0 ** np.arange(min_dec, max_dec + 1))


def add_log_note(ax, axis: str, fontsize: int = 13, x_pad: float = 0.0):
    """Corner-pinned $\\log_{10}$ note just OUTSIDE the plot rectangle.

    Placement mirrors where the log axis lives:
      - axis='y'  → above the top-left corner of the axes rectangle
      - axis='x'  → below the bottom-right corner of the axes rectangle

    Font matches the axis-label style (STIXGeneral, default black); size
    is a couple of points smaller than the axis labels so the subscript
    stays legible without crowding the corner.

    `x_pad` (axis='x' only) shifts the note rightward in axes-relative
    units; the combined panel uses a positive value at large fontsize to
    push the note clear of the rightmost tick label, which would
    otherwise visually collide.
    """
    # Use matplotlib mathtext for the subscript — STIXGeneral doesn't
    # ship Unicode subscript glyphs (U+2081 / U+2080) so we can't rely
    # on raw Unicode with the axis font.  `fontname` is intentionally
    # omitted here so mathtext renders in its default STIX math font
    # (visually identical to STIXGeneral for the "log" part).
    LABEL = r'$\log_{10}$'
    # Pinned flush-to the plot-rectangle edge.
    #   y-axis: above top-left corner, aligned left (outside top spine)
    #   x-axis: at the right end of the axis frame, tight against the
    #           right spine (ha='right', x=1.0+x_pad) and hugging the
    #           x-axis (va='top', y=0 → text sits at tick-label height).
    #           The plotter also trims xlim so the rightmost "10^N" tick
    #           is the last one on the axis — log_{10} fills the space
    #           where "10^(N+1)" would otherwise have appeared.
    kw = dict(fontsize=fontsize,
              transform=ax.transAxes, clip_on=False, zorder=100)
    if axis == 'y':
        ax.text(0.0, 1.02, LABEL,
                ha='left', va='bottom', **kw)
    elif axis == 'x':
        ax.text(1.0 + x_pad, 0.0, LABEL,
                ha='right', va='top', **kw)


# ─── Cache helpers (fig1 / fig2) ──────────────────────────────────────────────
# fig1's per-(rate, p99) sweep is expensive (17 rates × 6 schemes × duration);
# we cache it to JSON the same way fig2 caches its CDF arrays so plot-only
# tweaks and the combined panel can replay without re-simulation.

def _fig1_cache_path(args):
    mode = 'quick' if args.quick else 'full'
    if args.n_tenants > 1:
        tag = f'mt_n{args.n_tenants}_{mode}_s{args.seed}'
    else:
        tag = f'single_{mode}_s{args.seed}'
    return RESULTS_DIR / f'fig1_p99_{tag}.json'


def _load_fig1_cache(args, expected_rates=None):
    """Return (results, cd_max, ludo_max) from fig1 cache, or
    (None, None, None) if missing/stale.

    `results` is `{scheme: [(rate, p99_ttft), ...]}`.  Per-scheme curves
    end at each scheme's saturation knee, so they need not all cover the
    full sweep.  When `expected_rates` is given, the *attempted* sweep
    stored in the cache must match it exactly or the cache is treated as
    stale (so an edit to RATES_FULL forces a re-sim).
    """
    path = _fig1_cache_path(args)
    if not path.exists():
        return None, None, None
    with open(path) as f:
        cached = json.load(f)
    if not all(s in cached.get('results', {}) for s in ALL_SCHEMES):
        return None, None, None
    results = {s: [tuple(pt) for pt in cached['results'][s]] for s in ALL_SCHEMES}
    if expected_rates is not None:
        if sorted(cached.get('rates', [])) != sorted(expected_rates):
            return None, None, None
    return results, cached.get('cd_max'), cached.get('ludo_max')


def _load_fig2_cache(args, ludo_max_rate):
    """Return (all_ttfts, all_e2e) from the fig2 CDF cache, or (None, None)
    when missing.  Cache filename is reconstructed using the SAME
    (n_tenants, n_req, rate, seed) keying `run_fig2` uses — note that
    multi-tenant mode hard-overrides n_req to 500_000 regardless of rate
    (matching fig3b's methodology), so we mirror that branch here."""
    rate = float(FIG2_RATE)
    if args.n_tenants > 1:
        n_req = 500_000
        cache_tag = f'mt_n{args.n_tenants}_r{n_req}_rate{int(rate)}_s{args.seed}'
    else:
        n_req, _ = rate_to_n_warmup(rate, args.quick)
        cache_tag = f'single_rate{int(rate)}_s{args.seed}'
    path = RESULTS_DIR / f'fig2_cdf_{cache_tag}.json'
    if not path.exists():
        return None, None
    with open(path) as f:
        cached = json.load(f)
    if not all(s in cached.get('ttft_ms', {}) for s in ALL_SCHEMES):
        return None, None
    return cached['ttft_ms'], cached['e2e_ms']


# ─── Shared plotting helper ───────────────────────────────────────────────────

def _annotate_ludo_pcts(ax, sorted_ludo_vals, label_fontsize=17):
    """Mark Ludo's p50 / p90 / p99 on a CDF curve with crosshair guides
    instead of dots.  Each percentile gets:
      - a horizontal dashed line from the left y-axis to the CDF
      - a vertical dashed line from the x-axis up to the CDF
      - the label "pNN: V ms" pinned where the two lines meet
    All lines use the Ludo palette colour so the crosshair reads as a
    Ludo-specific reference grid against the other curves.

    `label_fontsize` defaults to the standalone-figure size (17 pt);
    the combined panel overrides it to keep the annotation visible after
    the per-panel width is shrunk.

    Module-level so both `run_fig2` and `_plot_combined` share one
    implementation.
    """
    color = LINE_COLORS['ludo']
    xmin, _ = ax.get_xlim()
    n = len(sorted_ludo_vals)
    for p, tag in [(0.50, 'p50'), (0.90, 'p90'), (0.99, 'p99')]:
        idx = min(n - 1, int(round(n * p)))
        x_val = sorted_ludo_vals[idx]
        ax.hlines(y=p, xmin=xmin, xmax=x_val,
                  colors=color, linestyles='dashed',
                  linewidth=1.5, zorder=9)
        ax.vlines(x=x_val, ymin=0.0, ymax=p,
                  colors=color, linestyles='dashed',
                  linewidth=1.5, zorder=9)
        ax.annotate(f'{tag}: {x_val:,.0f} ms',
                    xy=(x_val, p),
                    xytext=(8, -8),
                    textcoords='offset points',
                    fontsize=label_fontsize, fontname='STIXGeneral',
                    color=color, zorder=11)


# ─── Combined panel: fig1 + fig2 + fig2b side-by-side ─────────────────────────

def _plot_combined(fig1_results, all_ttfts, all_e2e,
                   schemes_to_plot=None,
                   filename='fig1_2_2b_combined.pdf'):
    """Render fig1 (latency-throughput), fig2 (TTFT CDF), and fig2b (E2E
    CDF) side-by-side in a single figure with one shared top legend.

    Tuned for paper inclusion: each panel is roughly square (5.33 × 5.0 in)
    so the figure stays narrow when scaled to a column width, and every
    text element (axis labels, ticks, crosshair annotations, legend) is
    bumped well above the standalone-figure sizes so it remains readable
    at small print scales.  The legend sits flush against the panel tops
    to minimise inter-element whitespace.
    """
    if schemes_to_plot is None:
        schemes_to_plot = ALL_SCHEMES

    # Type-system shorthand for the bumped sizes — every panel reads from
    # this block so a downstream tweak only touches one place.  In-panel
    # text (labels, ticks, annotation, log-note) sits noticeably above
    # the standalone-figure sizes; the legend is held back to a size that
    # lets the full row strip stay strictly inside the figure width.
    LBL_FS     = 26   # axis label font (xlabel / ylabel)
    TICK_FS    = 24   # tick label font
    NOTE_FS    = 22   # corner-pinned log_{10} note
    ANNOT_FS   = 26   # Ludo crosshair "pNN: V ms" annotation
    LEGEND_FS  = 24   # shared top legend
    LEGEND_NCOL = 3   # 6 schemes laid out as 2 rows × 3 cols so the longest
                      # label ("Central Directory") fits within the panel-block
                      # width without overflowing.
    XTICK_N    = 5    # linear-x tick budget on fig1 panel — sized for the
                      # wider 6-in panel so 5 ticks lay out at ~1.5 in
                      # spacing, neither crowded nor overly sparse.
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

    # ── Panel 1: fig1 latency-throughput ────────────────────────────────────
    ax1 = axes[0]
    for scheme in schemes_to_plot:
        xs = [r for r, _ in fig1_results[scheme]]
        ys = [t for _, t in fig1_results[scheme]]
        zorder = 4 if scheme in ('central_directory', 'ludo', 'radix_tree') else 3
        is_hero = (scheme == 'ludo')
        msize = 12  if is_hero else 9
        ew    = 2.2 if is_hero else 1.5
        lw    = 2.2 if is_hero else 1.5
        markevery = max(1, len(xs) // 8)
        ax1.plot(xs, ys,
                 marker=MARKERS.get(scheme, 'o'),
                 markersize=msize, markeredgewidth=ew,
                 markerfacecolor=LINE_FILL[scheme],
                 markeredgecolor=LINE_COLORS[scheme],
                 markevery=markevery,
                 linewidth=lw, zorder=zorder,
                 color=LINE_COLORS[scheme],
                 label=LINE_LABELS[scheme],
                 linestyle=SCHEME_LINESTYLES.get(scheme, '-'),
                 clip_on=False)
    ax1.set_xlabel('Arrival Rate (requests/s)', fontsize=LBL_FS,
                   fontname='STIXGeneral')
    ax1.set_ylabel('P99 TTFT (ms)', fontsize=LBL_FS, fontname='STIXGeneral')
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=TICK_FS)
    # Cap the x-tick count so the bumped tick labels don't crowd the
    # narrower panel width.
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=XTICK_N))
    ax1.yaxis.set_major_locator(
        mticker.LogLocator(base=10, subs=(1.0,), numticks=20))
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f'{int(np.log10(y))}'))
    ax1.tick_params(axis='y', which='minor', length=0)
    ax1.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
    add_log_note(ax1, 'y', fontsize=NOTE_FS)

    # ── Panel 2: fig2 TTFT CDF ──────────────────────────────────────────────
    ax2 = axes[1]
    for scheme in schemes_to_plot:
        vals = all_ttfts[scheme]
        n = len(vals)
        zorder = 4 if scheme in ('central_directory', 'ludo', 'radix_tree') else 3
        step = max(1, n // 500)
        is_hero = (scheme == 'ludo')
        lw = 4.0 if is_hero else 3.0
        ax2.plot(vals[::step],
                 (np.arange(0, n, step) + 1) / n,
                 color=LINE_COLORS[scheme],
                 linewidth=lw, zorder=zorder,
                 linestyle=SCHEME_LINESTYLES.get(scheme, '-'),
                 label=LINE_LABELS[scheme],
                 clip_on=False)
    _annotate_ludo_pcts(ax2, all_ttfts['ludo'], label_fontsize=ANNOT_FS)
    ax2.set_xscale('log')
    pad_log_xaxis_for_note(ax2)
    ax2.set_xlabel('TTFT (ms)', fontsize=LBL_FS, fontname='STIXGeneral')
    ax2.set_ylabel('CDF', fontsize=LBL_FS, fontname='STIXGeneral')
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(axis='both', labelsize=TICK_FS)
    ax2.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(np.log10(x))}'))
    ax2.tick_params(axis='x', which='minor', length=0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(int_like_fmt))
    ax2.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
    add_log_note(ax2, 'x', fontsize=NOTE_FS, x_pad=LOGNOTE_PAD)

    # ── Panel 3: fig2b End-to-End CDF ───────────────────────────────────────
    ax3 = axes[2]
    for scheme in schemes_to_plot:
        vals = all_e2e[scheme]
        n = len(vals)
        zorder = 4 if scheme in ('central_directory', 'ludo', 'radix_tree') else 3
        step = max(1, n // 500)
        is_hero = (scheme == 'ludo')
        lw = 4.0 if is_hero else 3.0
        ax3.plot(vals[::step],
                 (np.arange(0, n, step) + 1) / n,
                 color=LINE_COLORS[scheme],
                 linewidth=lw, zorder=zorder,
                 linestyle=SCHEME_LINESTYLES.get(scheme, '-'),
                 label=LINE_LABELS[scheme],
                 clip_on=False)
    _annotate_ludo_pcts(ax3, all_e2e['ludo'], label_fontsize=ANNOT_FS)
    ax3.set_xscale('log')
    pad_log_xaxis_for_note(ax3)
    ax3.set_xlabel('End-to-End Latency (ms)', fontsize=LBL_FS,
                   fontname='STIXGeneral')
    ax3.set_ylabel('CDF', fontsize=LBL_FS, fontname='STIXGeneral')
    ax3.set_ylim(0, 1.02)
    ax3.tick_params(axis='both', labelsize=TICK_FS)
    ax3.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(np.log10(x))}'))
    ax3.tick_params(axis='x', which='minor', length=0)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(int_like_fmt))
    ax3.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
    add_log_note(ax3, 'x', fontsize=NOTE_FS, x_pad=LOGNOTE_PAD)

    # ── Shared top legend ───────────────────────────────────────────────────
    # Pull handles from the fig1 panel: it draws colour + linestyle + marker
    # for each scheme, the most informative single-handle representation.
    # 6 schemes lay out as 2 rows × 3 cols so each entry has ~5.3 in of
    # horizontal space — comfortable for the longest label ("Central
    # Directory") at 24 pt — while the legend strip stays strictly inside
    # the figure's 16-in width.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=LEGEND_NCOL, frameon=False,
               prop={'family': 'STIXGeneral', 'size': LEGEND_FS},
               handlelength=2.2, columnspacing=1.5,
               handletextpad=0.4)

    # rect[3]=0.83 leaves room for the 2-row legend strip at the top; the
    # panel tops sit just under the legend baseline so the legend reads as
    # part of the panel block rather than a detached caption.  The slightly
    # taller figure (5.5 vs 5.0 in) keeps each panel's plotted area at
    # ~4.6 in tall so the panels themselves do not visibly shrink.
    fig.tight_layout(rect=[0, 0, 1, 0.83])
    save_fig(fig, filename)


# ─── Figure 1: Latency-Throughput Curves ──────────────────────────────────────

def run_fig1(args, wgen):
    print('\n[Figure 1] Latency-throughput curves …')
    if getattr(args, 'fig1_rates', ''):
        # Explicit sweep override (e.g. extend past 6000 on workloads where
        # every scheme is still flat there and the knees need more runway).
        rates = [int(x) for x in args.fig1_rates.split(',') if x.strip()]
    else:
        rates = RATES_QUICK if args.quick else RATES_FULL
    workload = args.workload

    # ── Cache lookup ─────────────────────────────────────────────────────────
    # Fig1 is the most expensive section3 step (16 rates × 6 schemes ×
    # 100 K-request steady-state runs).  Cache the per-scheme (rate,
    # p99_ttft) sweep + the derived knee rates so a plot-only re-run, or a
    # downstream combined panel rebuild, can skip the simulation entirely.
    cache_path = _fig1_cache_path(args)
    cached_results, cached_cd, cached_ludo = _load_fig1_cache(args, rates)

    fig1_seeds = ([int(x) for x in args.fig1_seeds.split(',') if x.strip()]
                  if getattr(args, 'fig1_seeds', '') else [args.seed])
    results_std = {}
    if cached_results is not None:
        # Multi-seed sweeps are keyed by the seed list too.
        with open(cache_path) as f:
            _raw = json.load(f)
        if _raw.get('fig1_seeds', [_raw.get('seed')]) != fig1_seeds:
            cached_results = None
        else:
            results_std = _raw.get('results_std', {})

    if cached_results is not None and cached_cd is not None and cached_ludo is not None:
        results  = cached_results
        cd_max   = cached_cd
        ludo_max = cached_ludo
        print(f'  loaded cached p99 sweeps from {cache_path.name}')
    else:
        dur_s = DURATION_S_QUICK if args.quick else DURATION_S_FULL
        print(f'  simulating {dur_s:.1f} s of Poisson arrivals per rate '
              '(N = rate × duration); below saturation the p99 is the '
              'steady-state latency, past saturation it is the backlog '
              'that builds up over the window.')

        results = {}   # scheme -> list of (rate, median p99 across seeds)
        results_raw = {}   # scheme -> {rate: [per-seed p99]} (kept in cache)
        for scheme in ALL_SCHEMES:
            print(f'  scheme={scheme}')
            pts, stds = [], []
            results_raw[scheme] = {}
            for rate in rates:
                n_req, warmup = rate_to_n_warmup(rate, args.quick)
                vals = []
                for sd in fig1_seeds:
                    recs, _ = run_scheme(scheme, rate, workload, n_req,
                                         warmup, sd, wgen)
                    vals.append(p99([r.ttft_ms for r in recs]))
                # Median across seeds, not mean: PlanetServe's gate-affinity
                # placement is heavy-tailed on conversational Zipf heads (one
                # unlucky seed can post a 10× p99), and a few-sample mean
                # inherits that jaggedness.  The median is the typical-
                # deployment statistic; for the low-variance schemes it
                # coincides with the mean.
                m = float(np.median(vals))
                pts.append((rate, m))
                stds.append((rate, float(np.std(vals))))
                results_raw[scheme][rate] = [float(v) for v in vals]
                tag = (f' (±{np.std(vals):,.0f}, {len(vals)} seeds)'
                       if len(vals) > 1 else '')
                print(f'    rate={rate}: p99={m:,.0f} ms{tag}')
            results[scheme] = pts
            results_std[scheme] = stds

        # SLO-free "knee" detection.  Fig 2 / 3 / 4 need an operating point
        # somewhere between "trivially idle" and "near saturation" so the
        # CDFs / FLOPs / heatmaps exercise real queue dynamics.  Without an
        # SLO threshold, we pick the *knee* of each scheme's p99-vs-rate
        # curve: the largest arrival rate at which p99 still sits within 3×
        # of the scheme's minimum p99 across the sweep.  Beyond that point
        # p99 climbs sharply as the queue approaches saturation.
        def _knee(pts):
            if not pts:
                return rates[0]
            pmin = min(t for _, t in pts)
            cap  = max(3.0 * pmin, pmin + 1.0)   # absolute floor for tiny pmin
            viable = [r for r, t in pts if t <= cap]
            return max(viable) if viable else pts[0][0]

        cd_max   = _knee(results['central_directory'])
        ludo_max = _knee(results['ludo'])

        with open(cache_path, 'w') as f:
            json.dump({
                'results':   {s: [list(pt) for pt in results[s]]
                              for s in ALL_SCHEMES},
                'cd_max':    cd_max,
                'ludo_max':  ludo_max,
                'rates':     list(rates),
                'duration_s': dur_s,
                'n_tenants': args.n_tenants,
                'workload':  args.workload,
                'seed':      args.seed,
                'fig1_seeds': fig1_seeds,
                'results_std': {s: [list(pt) for pt in results_std[s]]
                                for s in results_std},
                'results_per_seed': results_raw,
                'mode':      'quick' if args.quick else 'full',
            }, f, indent=2)
        print(f'  fig1 sweep cached → {cache_path.name}')

    print(f'  central directory knee rate = {cd_max} req/s')
    print(f'  ludo              knee rate = {ludo_max} req/s')

    def _plot_fig1(schemes_to_plot, filename):
        # Widen the frame (8 × 4.2 in, matching fig2/fig2b) and subsample
        # markers so the 17-point rate sweep stops reading as a dense cluster.
        # Aim for ~8 markers per curve regardless of sweep length.
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for scheme in schemes_to_plot:
            xs = [r for r, _ in results[scheme]]
            ys = [t for _, t in results[scheme]]
            zorder = 4 if scheme in ('central_directory', 'ludo', 'radix_tree') else 3
            # Ludo is the hero scheme — render with a slightly larger marker
            # AND a thicker line/marker edge so it reads first from the
            # legend; every other scheme uses the lighter common stroke.
            is_hero = (scheme == 'ludo')
            msize  = 12  if is_hero else 9
            ew     = 2.2 if is_hero else 1.5
            lw     = 2.2 if is_hero else 1.5
            markevery = max(1, len(xs) // 8)
            ax.plot(xs, ys,
                    marker=MARKERS.get(scheme, 'o'),
                    markersize=msize, markeredgewidth=ew,
                    markerfacecolor=LINE_FILL[scheme],
                    markeredgecolor=LINE_COLORS[scheme],
                    markevery=markevery,
                    linewidth=lw, zorder=zorder,
                    color=LINE_COLORS[scheme],
                    label=LINE_LABELS[scheme],
                    linestyle=SCHEME_LINESTYLES.get(scheme, '-'),
                    clip_on=False)
        ax.set_xlabel('Arrival Rate (requests/s)', fontsize=17,
                      fontname='STIXGeneral')
        ax.set_ylabel('P99 TTFT (ms)', fontsize=16,
                      fontname='STIXGeneral')
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=16)
        # Force a label at every decade — matplotlib's default LogLocator
        # drops every-other label once the tick font grows to 16pt,
        # leaving an ugly "10^2 / 10^4" skip pattern on a 6-decade axis.
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10, subs=(1.0,), numticks=20))
        # y-axis already carries the "log_10" header, so ticks show the
        # bare exponent (3, 4, 5, ...) instead of "10^3".  Minor ticks
        # suppressed so only labelled decades get a tick mark, matching
        # the x-axis's tick-only-at-label style.
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f'{int(np.log10(y))}'))
        ax.tick_params(axis='y', which='minor', length=0)
        # NOTE: matplotlib drops the `fontsize` kwarg when `prop` is given,
        # so the size MUST live inside the `prop` dict to take effect.
        ax.legend(loc='lower center',
                  bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False,
                  borderaxespad=0,
                  prop={'family': 'STIXGeneral', 'size': 15})
        ax.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        add_log_note(ax, 'y')
        save_fig(fig, filename)

    _plot_fig1(ALL_SCHEMES, 'fig1_latency_throughput.pdf')
    return cd_max, ludo_max


# ─── TBT (Time Between Tokens) — SGLang chunked-prefill interference model ──
#
# The base decode pace is the fixed C_DECODE_STEP_MS_PER_TOKEN.  Our serving
# engine is SGLang, whose chunked prefill executes every admitted prefill as
# SGLANG_CHUNK_TOKENS-sized chunk iterations with decode steps interleaving
# at chunk granularity: each chunk stretches ONE token gap of every request
# decoding on that GPU by the chunk's compute time (≤ 8192 × C_PREFILL
# ≈ 410 ms), and a longer prefill spills into consecutive gaps.  We measure
# this post-hoc: Node.prefill_events logs every non-zero residual prefill
# (start_ms, residual_tokens); a request decoding on that node during
# [arrival+TTFT, arrival+TTFT+gen_len·step] collects the chunk stalls of
# prefills starting inside its window.  Scheduling/TTFT/E2E numbers are
# untouched — TBT is a measurement overlay.  The metric rewards exactly our
# two levers: higher hit ratio → shorter residual prefills → fewer/smaller
# chunk stalls; better load balance → fewer prefills landing on hot nodes.

def _tbt_distribution(recs, nodes):
    """Pooled per-token TBT gaps for one scheme's run.

    Returns (n_base, gaps): count of unstalled gaps (exactly the base decode
    step) and a sorted array of stalled gap values in ms.  If a request
    collects more chunk stalls than it has decode iterations (deep
    saturation), the stall mass folds into its gen_len gaps round-robin.
    """
    step  = C_DECODE_STEP_MS_PER_TOKEN
    chunk = float(_sim.SGLANG_CHUNK_TOKENS)
    ev = {}
    for n in nodes:
        if n.prefill_events:
            a = np.asarray(n.prefill_events, dtype=float)
            o = np.argsort(a[:, 0], kind='stable')
            ev[n.node_id] = (a[o, 0], a[o, 1])
    n_base = 0
    chunks = []
    for r in recs:
        gen = r.gen_len
        if gen <= 0:
            continue
        starts_toks = ev.get(r.node_id)
        if starts_toks is None:
            n_base += gen
            continue
        starts, toks = starts_toks
        t1 = r.arrival_time + r.ttft_ms
        i = int(np.searchsorted(starts, t1, side='right'))
        j = int(np.searchsorted(starts, t1 + gen * step, side='left'))
        if j <= i:
            n_base += gen
            continue
        # Expand each overlapping prefill into its SGLang chunk stalls.
        w    = toks[i:j]
        full = (w // chunk).astype(int)
        rem  = w - full * chunk
        parts = np.concatenate([np.full(int(full.sum()), chunk),
                                rem[rem > 0]]) * C_PREFILL_MS_PER_TOKEN
        k = parts.size
        if k <= gen:
            chunks.append(parts + step)
            n_base += gen - k
        else:
            bounds = np.linspace(0, k, gen + 1).astype(int)
            chunks.append(np.add.reduceat(parts, bounds[:-1]) + step)
    gaps = np.sort(np.concatenate(chunks)) if chunks else np.empty(0)
    return n_base, gaps


def _tbt_compact(n_base, gaps, n_q=2001):
    """JSON-friendly summary: exact percentiles + a quantile grid of the
    stalled-gap distribution (raw pooled gaps run to 10^7-10^8 samples)."""
    base = C_DECODE_STEP_MS_PER_TOKEN
    N = n_base + gaps.size

    def pct(q):
        r = int(q / 100.0 * max(N - 1, 0))
        if r < n_base or not gaps.size:
            return base
        return float(gaps[min(r - n_base, gaps.size - 1)])

    grid = (np.quantile(gaps, np.linspace(0.0, 1.0, n_q))
            if gaps.size else np.empty(0))
    return {'n_base': int(n_base), 'n_gaps': int(gaps.size), 'base_ms': base,
            'pcts': {str(p): pct(p) for p in (50, 90, 95, 99, 99.9, 99.99)},
            'gap_q': [round(float(x), 4) for x in grid]}


def _tbt_curve(compact, n_pts=4001):
    """Reconstruct a sorted plot-ready sample array from a compact summary:
    an atom of weight n_base at base_ms followed by the stalled-gap
    quantile grid."""
    N = compact['n_base'] + compact['n_gaps']
    vals = np.full(n_pts, float(compact['base_ms']))
    if compact['n_gaps'] and N > 1:
        ranks = np.arange(n_pts) / (n_pts - 1) * (N - 1)
        m = ranks >= compact['n_base']
        gq = np.asarray(compact['gap_q'], dtype=float)
        rel = (ranks[m] - compact['n_base']) / max(compact['n_gaps'] - 1, 1)
        vals[m] = np.interp(rel, np.linspace(0.0, 1.0, gq.size), gq)
    return vals


# ─── Figure 2: TTFT CDF ───────────────────────────────────────────────────────

def run_fig2(args, wgen, oracle_max_rate, ludo_max_rate):
    print('\n[Figure 2] TTFT CDF …')
    # Operate at FIG2_RATE — set below KVCompass's saturation knee so its
    # CDF is a genuine steady-state latency distribution, while the DHTs
    # (knee ≈1400) are clearly saturated.  See FIG2_RATE for the rationale.
    rate = float(FIG2_RATE)
    if args.n_tenants > 1:
        # Match fig3b's 500 K-request methodology so fig2 / fig2b reflect
        # the same multi-tenant pool + total-request count fig3b uses.
        # Warmup is 10 % of the timed window.
        n_req  = 500_000
        warmup = n_req // 10
        print(f'  fixed rate = {rate:.0f} req/s (below ludo knee ≈ {ludo_max_rate}); '
              f'multi-tenant mode → N={n_req:,} requests, warmup={warmup:,}')
    else:
        n_req, warmup = rate_to_n_warmup(rate, args.quick)
        print(f'  fixed rate = {rate:.0f} req/s (below ludo knee ≈ {ludo_max_rate}), '
              f'N={n_req:,} requests')

    # Run each scheme once; reuse records for fig2 (TTFT CDF) and fig2b
    # (E2E CDF) so the 500 K-request simulation only pays the cost once.
    # Results are cached to JSON keyed on (n_tenants, n_req, rate, seed)
    # so plot-only tweaks skip the multi-minute replay.
    if args.n_tenants > 1:
        cache_tag = f'mt_n{args.n_tenants}_r{n_req}_rate{int(rate)}_s{args.seed}'
    else:
        cache_tag = f'single_rate{int(rate)}_s{args.seed}'
    fig2_cache = RESULTS_DIR / f'fig2_cdf_{cache_tag}.json'
    tbt_cache  = RESULTS_DIR / f'fig2_tbt_{cache_tag}.json'

    # Load cache only if it has every current scheme; otherwise an older
    # scheme-list (e.g. pre-Central-Directory runs) will KeyError at load.
    # TBT lives in its own JSON but is produced by the same simulation pass,
    # so a missing/stale TBT cache forces a re-simulation of both.
    cache_valid = False
    if fig2_cache.exists() and tbt_cache.exists():
        with open(fig2_cache) as f:
            cached = json.load(f)
        with open(tbt_cache) as f:
            cached_tbt = json.load(f)
        cache_valid = (all(s in cached.get('ttft_ms', {}) for s in ALL_SCHEMES)
                       and all(s in cached_tbt.get('tbt', {})
                               for s in ALL_SCHEMES))
        if not cache_valid:
            print(f'  cache {fig2_cache.name} / {tbt_cache.name} is stale '
                  '(missing schemes); re-simulating')
    elif fig2_cache.exists():
        print(f'  {tbt_cache.name} missing (pre-TBT cache); re-simulating')

    if cache_valid:
        all_ttfts = {s: cached['ttft_ms'][s] for s in ALL_SCHEMES}
        all_e2e   = {s: cached['e2e_ms' ][s] for s in ALL_SCHEMES}
        all_tbt   = {s: cached_tbt['tbt'][s] for s in ALL_SCHEMES}
        print(f'  loaded cached CDF arrays from {fig2_cache.name} + '
              f'{tbt_cache.name}')
    else:
        all_records = {}
        all_tbt     = {}
        for scheme in ALL_SCHEMES:
            print(f'  scheme={scheme}')
            recs, nodes_s = run_scheme(scheme, rate, args.workload, n_req,
                                       warmup, args.seed, wgen)
            all_records[scheme] = recs
            all_tbt[scheme] = _tbt_compact(*_tbt_distribution(recs, nodes_s))
            p = all_tbt[scheme]['pcts']
            print(f"    TBT p50={p['50']:.2f} p99={p['99']:.2f} "
                  f"p99.9={p['99.9']:.2f} ms")
        all_ttfts = {s: sorted(r.ttft_ms for r in all_records[s])
                     for s in ALL_SCHEMES}
        all_e2e   = {s: sorted(r.e2e_latency_ms for r in all_records[s])
                     for s in ALL_SCHEMES}
        with open(fig2_cache, 'w') as f:
            json.dump({'ttft_ms': all_ttfts, 'e2e_ms': all_e2e,
                       'rate': rate, 'n_req': n_req,
                       'n_tenants': args.n_tenants}, f)
        with open(tbt_cache, 'w') as f:
            json.dump({'tbt': all_tbt, 'rate': rate, 'n_req': n_req,
                       'n_tenants': args.n_tenants,
                       'base_step_ms': C_DECODE_STEP_MS_PER_TOKEN}, f)
        print(f'  CDF arrays cached → {fig2_cache.name} + {tbt_cache.name}')

    def _plot_fig2(schemes_to_plot, filename):
        # Wider frame (8 × 4.2 in) gives the corner-pinned log_{10} note
        # more horizontal breathing room against the rightmost tick.
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for scheme in schemes_to_plot:
            vals = all_ttfts[scheme]
            n    = len(vals)
            zorder = 4 if scheme in ('central_directory', 'ludo', 'radix_tree') else 3
            step = max(1, n // 500)
            is_hero = (scheme == 'ludo')
            lw      = 4.0 if is_hero else 3.0
            ax.plot(vals[::step],
                    (np.arange(0, n, step) + 1) / n,
                    color=LINE_COLORS[scheme],
                    linewidth=lw, zorder=zorder,
                    linestyle=SCHEME_LINESTYLES.get(scheme, '-'),
                    label=LINE_LABELS[scheme],
                    clip_on=False)
        _annotate_ludo_pcts(ax, all_ttfts['ludo'])
        ax.set_xscale('log')
        pad_log_xaxis_for_note(ax)
        ax.set_xlabel('TTFT (ms)', fontsize=17,
                      fontname='STIXGeneral')
        ax.set_ylabel('CDF', fontsize=16, fontname='STIXGeneral')
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis='both', labelsize=16)
        # x-axis carries the "log_10" corner note already, so ticks show
        # bare exponents (2, 3, 4, …) instead of "10^2" etc.  Minor tick
        # marks suppressed so tick lines only appear where there is a
        # labelled decade — matches the y-axis "only-at-labels" style.
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{int(np.log10(x))}'))
        ax.tick_params(axis='x', which='minor', length=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(int_like_fmt))
        # NOTE: matplotlib drops the `fontsize` kwarg when `prop` is given,
        # so the size MUST live inside the `prop` dict to take effect.
        ax.legend(loc='lower center',
                  bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False,
                  borderaxespad=0,
                  prop={'family': 'STIXGeneral', 'size': 15})
        ax.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        add_log_note(ax, 'x')
        save_fig(fig, filename)

    _plot_fig2(ALL_SCHEMES, 'fig2_ttft_cdf.pdf')

    # ── Figure 2b: End-to-End Latency CDF ────────────────────────────────────
    # E2E latency = TTFT + gen_len × C_DECODE_STEP_MS_PER_TOKEN.  Reuses the
    # records already simulated above — no need to re-run every scheme.
    print('\n[Figure 2b] E2E latency CDF …')

    def _plot_fig2b(schemes_to_plot, filename):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for scheme in schemes_to_plot:
            vals = all_e2e[scheme]
            n    = len(vals)
            zorder = 4 if scheme in ('central_directory', 'ludo', 'radix_tree') else 3
            step = max(1, n // 500)
            is_hero = (scheme == 'ludo')
            lw      = 4.0 if is_hero else 3.0
            ax.plot(vals[::step],
                    (np.arange(0, n, step) + 1) / n,
                    color=LINE_COLORS[scheme],
                    linewidth=lw, zorder=zorder,
                    linestyle=SCHEME_LINESTYLES.get(scheme, '-'),
                    label=LINE_LABELS[scheme],
                    clip_on=False)
        _annotate_ludo_pcts(ax, all_e2e['ludo'])
        ax.set_xscale('log')
        pad_log_xaxis_for_note(ax)
        ax.set_xlabel('End-to-End Latency (ms)', fontsize=17,
                      fontname='STIXGeneral')
        ax.set_ylabel('CDF', fontsize=16, fontname='STIXGeneral')
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis='both', labelsize=16)
        # Bare-exponent x ticks (log_10 header in the corner) + no minor ticks.
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{int(np.log10(x))}'))
        ax.tick_params(axis='x', which='minor', length=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(int_like_fmt))
        # NOTE: matplotlib drops the `fontsize` kwarg when `prop` is given,
        # so the size MUST live inside the `prop` dict to take effect.
        ax.legend(loc='lower center',
                  bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False,
                  borderaxespad=0,
                  prop={'family': 'STIXGeneral', 'size': 15})
        ax.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        add_log_note(ax, 'x')
        save_fig(fig, filename)

    _plot_fig2b(ALL_SCHEMES, 'fig2b_e2e_cdf.pdf')

    # ── Box-plot variants of fig2 / fig2b (SkyWalker-Fig.8-style) ───────────
    # Same cached samples, presented as one box per scheme at the fig2
    # operating point: box = IQR, line = median, whiskers = p1/p99 (the
    # default 1.5×IQR rule sprays 10^5 outlier dots on heavy-tailed latency
    # samples, so the tail is encoded in the upper whisker instead).
    def _plot_box(data, ylabel_, filename):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        stats = []
        for s in ALL_SCHEMES:
            v = np.asarray(data[s])
            p1, p25, p50, p75, p99 = np.percentile(v, [1, 25, 50, 75, 99])
            stats.append({'med': p50, 'q1': p25, 'q3': p75,
                          'whislo': p1, 'whishi': p99, 'fliers': []})
        bp = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.62,
                    medianprops=dict(color='black', linewidth=1.6))
        for patch, s in zip(bp['boxes'], ALL_SCHEMES):
            patch.set_facecolor(LINE_FILL[s])
            patch.set_edgecolor(LINE_COLORS[s])
            patch.set_linewidth(1.6)
        for i, s in enumerate(ALL_SCHEMES):
            for ln in bp['whiskers'][2 * i:2 * i + 2] + bp['caps'][2 * i:2 * i + 2]:
                ln.set_color(LINE_COLORS[s])
                ln.set_linewidth(1.4)
        ax.set_yscale('log')
        ax.set_ylabel(ylabel_, fontsize=16, fontname='STIXGeneral')
        ax.set_xticks(np.arange(1, len(ALL_SCHEMES) + 1))
        ax.set_xticklabels([LINE_LABELS[s] for s in ALL_SCHEMES],
                           rotation=25, ha='right', fontsize=13,
                           fontname='STIXGeneral')
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
        fig.tight_layout()
        save_fig(fig, filename)

    _plot_box(all_ttfts, 'TTFT (ms)', 'fig2_ttft_box.pdf')
    _plot_box(all_e2e, 'End-to-End Latency (ms)', 'fig2b_e2e_box.pdf')

    # ── Figure 2c: TBT CDF (prefill-stall interference model) ───────────────
    # Same operating point and plotting conventions as fig2 / fig2b; the
    # distribution is reconstructed from the compact per-scheme summaries
    # (base-step atom + stalled-gap quantile grid) in fig2_tbt_*.json.
    print('\n[Figure 2c] TBT CDF …')

    def _plot_fig2c(schemes_to_plot, filename):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for scheme in schemes_to_plot:
            vals = _tbt_curve(all_tbt[scheme])
            n    = len(vals)
            zorder = 4 if scheme in ('central_directory', 'ludo', 'radix_tree') else 3
            step = max(1, n // 500)
            is_hero = (scheme == 'ludo')
            lw      = 4.0 if is_hero else 3.0
            # clip_on stays True (unlike fig2/2b): the zoomed y-range cuts
            # each curve's initial rise, which must not paint outside the
            # axes or it blows up the figure bounding box.
            ax.plot(vals[::step],
                    (np.arange(0, n, step) + 1) / n,
                    color=LINE_COLORS[scheme],
                    linewidth=lw, zorder=zorder,
                    linestyle=SCHEME_LINESTYLES.get(scheme, '-'),
                    label=LINE_LABELS[scheme])
        ax.set_xscale('log')
        pad_log_xaxis_for_note(ax)
        ax.set_xlabel('TBT (ms)', fontsize=17,
                      fontname='STIXGeneral')
        ax.set_ylabel('CDF', fontsize=16, fontname='STIXGeneral')
        # Zoomed y-range: 95-97 % of token gaps sit exactly at the base
        # decode step for every scheme, so the whole comparison lives in
        # the top few percent of the CDF.  Full [0,1] framing renders all
        # curves visually coincident.
        ax.set_ylim(0.94, 1.003)
        ax.set_yticks([0.94, 0.96, 0.98, 1.0])
        # Ludo p99 crosshair (p50/p90 fall inside the base-step atom, so
        # the fig2/fig2b three-percentile annotation is meaningless here).
        _c   = LINE_COLORS['ludo']
        _x99 = all_tbt['ludo']['pcts']['99']
        _xlo, _ = ax.get_xlim()
        ax.hlines(y=0.99, xmin=_xlo, xmax=_x99, colors=_c,
                  linestyles='dashed', linewidth=1.5, zorder=9)
        ax.vlines(x=_x99, ymin=0.94, ymax=0.99, colors=_c,
                  linestyles='dashed', linewidth=1.5, zorder=9)
        ax.annotate(f'p99: {_x99:,.0f} ms', xy=(_x99, 0.99),
                    xytext=(7, -16), textcoords='offset points',
                    fontsize=15, fontname='STIXGeneral', color=_c)
        ax.tick_params(axis='both', labelsize=16)
        # Bare-exponent x ticks (log_10 header in the corner) + no minor ticks.
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{int(np.log10(x))}'))
        ax.tick_params(axis='x', which='minor', length=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(int_like_fmt))
        # NOTE: matplotlib drops the `fontsize` kwarg when `prop` is given,
        # so the size MUST live inside the `prop` dict to take effect.
        ax.legend(loc='lower center',
                  bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False,
                  borderaxespad=0,
                  prop={'family': 'STIXGeneral', 'size': 15})
        ax.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        add_log_note(ax, 'x')
        save_fig(fig, filename)

    _plot_fig2c(ALL_SCHEMES, 'fig2c_tbt_cdf.pdf')

    # ── Combined panel: fig1 + fig2 + fig2b side-by-side ────────────────────
    # Auto-emit whenever fig1's cache exists so a paper-grade 1×3 PDF with
    # one shared legend is produced "for free" alongside the standalone
    # fig1/fig2/fig2b PDFs.  When the cache is missing we just print a
    # hint and skip — running `--figs 1` once will populate it.
    fig1_results, _, _ = _load_fig1_cache(args)
    if fig1_results is not None:
        print('\n[Combined] fig1 + fig2 + fig2b panel …')
        _plot_combined(fig1_results, all_ttfts, all_e2e)
    else:
        print(f'\n[Combined] skipped — fig1 cache not found '
              f'({_fig1_cache_path(args).name}); '
              'run `--figs 1` once to populate.')


# ─── Figure 3: Prefill FLOPs Saved (content-aware routing) ──────────────────
#
# Methodology
# -----------
# Real prompts from ToolBench / LooGLE / AAPS are tokenised (TB: actual
# input_ids; LG/AAPS: 4-char ≈ 1-token approximation) and each request carries
# *content hashes at multiple chunk sizes*.  Routing is then driven by the
# real prefix bytes rather than an opaque dataset row id, so two requests that
# share a common prefix collide naturally:
#
#   ToolBench : ALL 6000 entries share a 320-token system prompt
#   LooGLE    : ~130 document groups, group members share thousands of tokens
#   AAPS      : ~249 problem groups, group members share ~244 tokens
#
# Per-scheme routing rules:
#   consistent_hash / kademlia / chord:
#       multi-level hash-placement — one independent hash ring per
#       CHUNK_SIZES level, hash-owner at level l is
#       hash((content_hash_at_cs, l)) % N.  The routing target is the
#       level-l owner with the longest contiguous cached prefix on it.
#       Hash-placement-without-migration is preserved: a cached prefix
#       appears on any given level's owner only when that node *is* the
#       level's hash-owner — so the match cap is LUDO_MAX in theory but
#       the effective hit distribution remains lower than Ludo's because
#       cross-level alignment happens only ≈ 1/N of the time.
#   radix_tree:
#       per-trie path of 10-token chunk hashes (PlanetServe geometry,
#       gate = 100 tokens minimum match), match_len grows with trie depth.
#   ludo:
#       multi-level table at all CHUNK_SIZES = [128, 256, 512, 1024, 2048, 4096];
#       finds the longest prefix shared with any node in the cluster.
#   oracle:
#       theoretical 100% reuse (geometric upper bound).

HASH_FIXED_CHUNK   = 128   # retained for corpus-hash precomputation only (levels set)
RADIX_CHUNK_TOKENS = 10    # PlanetServe HRT chunk size
RADIX_GATE_CHUNKS  = 10    # PlanetServe GATE — also the cap on trie depth
                           # (advertise_depth = min(GATE, cached_len//CHUNK_SIZE)
                           # in the upstream HashRadixTree implementation, so
                           # every hit yields exactly GATE × CHUNK_SIZE = 100
                           # tokens regardless of how long the actual shared
                           # prefix is — this is the single-granularity match
                           # limitation Ludo's multi-level table is designed
                           # to overcome).
RADIX_MAX_CHUNKS   = RADIX_GATE_CHUNKS

# Per-node KV-cache budget for Fig 3 (tokens).  Matches the system-wide
# simulate.py defaults (32 GB VRAM + 128 GB DRAM per node) — no artificial
# shrinkage.  Under real-scale VRAM, most prefixes fit without evicting, so
# the FLOPs-saved gap between schemes now reflects pure *discovery-layer
# match quality* (cache hit rate × reported match length), not LRU pressure.
FIG3_VRAM_TOKENS   = NODE_VRAM_TOKENS
FIG3_DRAM_TOKENS   = NODE_DRAM_TOKENS


def _entry_token_seq(entry, dataset_name):
    """Return a token-id sequence for the entry suitable for content hashing.

    ToolBench:  real input_ids (already tokenised by Llama tokenizer).
    LooGLE/AAPS: char-quad approximation — group every 4 chars as one
                 synthetic token.  Equal-content text yields equal token
                 sequences, which is all that hash-based routing needs.
    """
    if dataset_name == 'toolbench':
        return entry['input_ids']
    text = entry['workload']
    return [hash(text[i:i + 4]) & 0xFFFFFFFF for i in range(0, len(text), 4)]


def _build_content_corpus(workload='mixed'):
    """Load the workload's datasets and pre-compute per-entry content hashes.

    Returns: dict[dataset → list[entry-info]] where each entry-info has:
        prefix_len  : token count
        hashes      : {chunk_size: hash(tokens[:cs])} for each level
        radix_path  : list of 10-token chunk hashes (for the radix-tree trie)
        gen_len     : decode-side generation length (for completeness; unused
                      by Fig 3's FLOPs-saved metric)
    """
    levels = sorted(set(CHUNK_SIZES + [HASH_FIXED_CHUNK]))
    if workload == 'wildchat':
        files = {'wildchat': 'wildchat.jsonl'}
    elif workload == 'lmsys':
        files = {'lmsys': 'lmsys.jsonl'}
    else:
        files = {
            'toolbench': 'toolbench_zipf_1.1_prompts_6000.jsonl',
            'loogle':    'loogle.jsonl',
            'aaps':      'aaps.jsonl',
        }
    datasets_dir = WorkloadGenerator._DATASETS_DIR
    corpus = {}
    for ds, fn in files.items():
        entries = WorkloadGenerator._load_jsonl(datasets_dir / fn)
        infos = []
        for e in entries:
            toks = _entry_token_seq(e, ds)
            n = len(toks)
            hashes = {cs: hash(tuple(toks[:cs])) for cs in levels if cs <= n}
            # Full-length 10-token chunk path. RadixTree scheme uses only the
            # first RADIX_GATE_CHUNKS of this; oracle walks the entire path
            # to compute the longest-possible cached-prefix match.
            full_chunks = n // RADIX_CHUNK_TOKENS
            radix_path = [
                hash(tuple(toks[i * RADIX_CHUNK_TOKENS:(i + 1) * RADIX_CHUNK_TOKENS]))
                for i in range(full_chunks)
            ]
            # Rolling-hash block IDs — prefix_blocks[k] is a unique identifier
            # for the KV block spanning the first (k+1) × RADIX_CHUNK_TOKENS
            # tokens of this content.  Two requests sharing their first k+1
            # chunks get identical prefix_blocks[k] → block-level cache can
            # dedupe them naturally.
            prefix_blocks = []
            _rh = 0
            for c in radix_path:
                _rh = (_rh * 2654435761 ^ c) & ((1 << 64) - 1)
                prefix_blocks.append(_rh)
            gen_len = (e['sampling_params']['max_new_tokens']
                       if ds == 'toolbench' else e.get('gen_len', 50))
            infos.append({
                'prefix_len':    n,
                'hashes':        hashes,
                'radix_path':    radix_path,
                'prefix_blocks': prefix_blocks,
                'gen_len':       gen_len,
                'region_id':     e.get('region_id'),   # None unless wildchat
                'dataset':       ds,   # for workload-aware eviction
            })
        corpus[ds] = infos
        print(f'    {ds}: {len(infos)} entries')
    return corpus


def _zipf_request_stream(corpus, seed, rate_rps=1000.0):
    """Build a request stream from the corpus — one request per dataset entry.

    The datasets already encode their realistic access distributions:
      - ToolBench's 6000 prompts are pre-sampled with Zipf α=1.1 (hot entries
        appear as duplicate records, up to 6× each).
      - LooGLE / AAPS entries are distinct queries, one per row.
    Re-sampling with another Zipf layer would double-count the skew and
    artificially repeat the *same complete prompt* (including its unique
    suffix) → inflated cache-hit rates that don't reflect real per-user
    query uniqueness.  Instead we consume each row once.

    Each request carries:
      region_id    — WildChat demand-weighted region (drives WAN cost in scoring)
      arrival_time — Poisson timeline at rate_rps (drives queue dynamics)
    """
    rng     = np.random.default_rng(seed)
    rate_ms = rate_rps / 1000.0
    reqs = []
    for ds, infos in corpus.items():
        for info in infos:
            reqs.append({
                'prefix_len':    info['prefix_len'],
                'hashes':        info['hashes'],
                'radix_path':    info['radix_path'],
                'prefix_blocks': info['prefix_blocks'],
                'region_id':     (info['region_id']
                                  if info.get('region_id') is not None
                                  else int(rng.choice(
                                      NUM_REGIONS, p=REGION_REQUEST_WEIGHTS))),
                'dataset':       ds,
            })
    rng.shuffle(reqs)
    t = 0.0
    for r in reqs:
        t += rng.exponential(1.0 / rate_ms)
        r['arrival_time'] = t
    return reqs


def _simulate_fig3_central_directory(reqs):
    """Central Directory cache model for fig3b — aligned with the fig1/2/2b
    Central Directory baseline (see `_run_central_directory`).

    Semantics:
      - **Omniscient cache state.**  A single global directory tracks every
        block ever served anywhere in the fleet.
      - **Infinite per-node memory.**  No eviction ever happens; once a
        block is cached, it stays cached for the rest of the experiment.
      - **Infinite bandwidth.**  Any node can service any cached block, so
        we model the cache as a cluster-wide unified set rather than a
        per-node sharded view.
      - **No DRAM tier.**  With infinite VRAM, nothing is ever demoted, so
        every hit is a VRAM hit by definition.  `dram_hit = 0`.

    This replaces the earlier Bélády-eviction `_simulate_fig3_oracle`
    (an impossible physical upper bound) with a realistic network-optimal
    reference: the fleet's best achievable hit rate given that cold-start
    requests still pay a miss but repeats are always served from cache.
    """
    cached: set = set()
    flops_saved   = 0.0
    vram_hit_frac = 0.0

    for req in reqs:
        plen   = req['prefix_len']
        blocks = req['prefix_blocks']

        # Longest contiguous cached prefix — pure VRAM tier since cache
        # never evicts.
        v_chunks = 0
        for bid in blocks:
            if bid in cached:
                v_chunks += 1
            else:
                break

        if plen > 0:
            f = v_chunks * RADIX_CHUNK_TOKENS / plen
            vram_hit_frac += f
            flops_saved   += f

        # Insert every block into the global directory (infinite memory).
        for bid in blocks:
            cached.add(bid)

    n = len(reqs)
    return {'flops_saved': flops_saved   / n,
            'vram_hit':    vram_hit_frac / n,
            'dram_hit':    0.0,
            'miss':        0.0}


def _simulate_fig3_oracle(reqs):
    """Offline omniscient oracle: knows every future request's content AND
    every node's cache state at all times, then routes each request to the
    node whose cache holds the longest prefix that matches the request's
    real token content (= maximum possible FLOPs reuse).

    Implementation
    --------------
    A single global trie keyed on 10-token chunk hashes (the same radix path
    used by RadixTree, but here uncapped — full prefix length).  Each trie
    node stores the set of node_ids that have advertised a prefix reaching
    that depth.  For each new request:

      1. Walk the trie along the request's full radix_path; record the
         deepest reached node `last_hit` whose `holders` is non-empty.
         match_len = last_hit.depth × CHUNK_SIZE   (capped at prefix_len).
      2. If `last_hit` exists, route to any holder there (cache reuse is
         identical regardless of choice).  Otherwise (cold) route to the
         globally least-loaded node so that subsequent requests can find it.
      3. Insert the request's *full* radix path into the trie, marking every
         depth as "owned" by the chosen node.  This makes the freshly
         cached prefix discoverable at all granularities by future requests.

    Without cache eviction (the corpus fits in 64 × 512K-token VRAM), this
    schedule is provably optimal for the FLOPs-saved metric: any request
    whose content has been seen before in the stream achieves match_len =
    min(prefix_len, longest matching cached prefix), which is the geometric
    upper bound for that request.  Only first-occurrence (cold) requests
    contribute < 1.0 to the mean, exactly as expected for an online cache.
    """
    # Block-level unified cluster cache with offline-optimal (Bélády) eviction.
    # VRAM_BLOCKS = total cluster VRAM expressed in chunks; same for DRAM.
    # Bélády implemented via a lazy max-heap keyed on next-access index —
    # stale entries (whose block left the tier or whose cursor moved) are
    # popped and discarded on demand.
    import heapq
    VRAM_BLOCKS = NUM_NODES * (FIG3_VRAM_TOKENS // RADIX_CHUNK_TOKENS)
    DRAM_BLOCKS = NUM_NODES * (FIG3_DRAM_TOKENS // RADIX_CHUNK_TOKENS)

    # Pre-scan: per-block-id sorted list of arrival indices that will touch it.
    block_arrivals = defaultdict(list)
    for i, req in enumerate(reqs):
        for bid in req['prefix_blocks']:
            block_arrivals[bid].append(i)
    block_cursor = defaultdict(int)

    def _next_access(bid):
        accs = block_arrivals[bid]
        c = block_cursor[bid]
        return accs[c] if c < len(accs) else float('inf')

    vram_set  = set()   # block_ids in VRAM
    dram_set  = set()   # block_ids in DRAM
    vram_heap = []      # (-next_access, snap_cursor, bid) — lazy max-heap
    dram_heap = []

    def _push_v(bid):
        heapq.heappush(vram_heap,
                       (-_next_access(bid), block_cursor[bid], bid))

    def _push_d(bid):
        heapq.heappush(dram_heap,
                       (-_next_access(bid), block_cursor[bid], bid))

    def _pop_worst_v():
        while vram_heap:
            _, snap, bid = heapq.heappop(vram_heap)
            if bid in vram_set and block_cursor[bid] == snap:
                vram_set.discard(bid)
                return bid
        return None

    def _pop_worst_d():
        while dram_heap:
            _, snap, bid = heapq.heappop(dram_heap)
            if bid in dram_set and block_cursor[bid] == snap:
                dram_set.discard(bid)
                return bid
        return None

    def _insert_vram(bid):
        """Put bid into VRAM.  Touches if already there; promotes from DRAM;
        else inserts as fresh, demoting a Bélády-worst VRAM block if needed."""
        if bid in vram_set:
            _push_v(bid)   # refresh heap entry; old becomes stale
            return
        if bid in dram_set:
            dram_set.discard(bid)   # promote; old DRAM heap entry goes stale
        while len(vram_set) >= VRAM_BLOCKS:
            victim = _pop_worst_v()
            if victim is None:
                break
            if len(dram_set) >= DRAM_BLOCKS:
                _pop_worst_d()
            dram_set.add(victim)
            _push_d(victim)
        vram_set.add(bid)
        _push_v(bid)

    flops_saved   = 0.0
    vram_hit_frac = 0.0
    dram_hit_frac = 0.0

    for i, req in enumerate(reqs):
        plen    = req['prefix_len']
        blocks  = req['prefix_blocks']

        # Match: longest contiguous cached prefix (no per-scheme cap on oracle)
        v_chunks = d_chunks = 0
        for bid in blocks:
            if bid in vram_set:
                v_chunks += 1
            elif bid in dram_set:
                d_chunks += 1
            else:
                break

        if plen > 0:
            fv = v_chunks * RADIX_CHUNK_TOKENS / plen
            fd = d_chunks * RADIX_CHUNK_TOKENS / plen
            vram_hit_frac += fv
            dram_hit_frac += fd
            flops_saved   += fv + fd

        # Advance cursor for every block this request touches, then insert
        # (touch matched blocks + cache new ones).  Cursor advance must
        # happen before _insert_vram so the pushed heap entry reflects the
        # *next* future access of this block, not the current one.
        for bid in blocks:
            block_cursor[bid] += 1
        for bid in blocks:
            _insert_vram(bid)

    n = len(reqs)
    return {'flops_saved': flops_saved   / n,
            'vram_hit':    vram_hit_frac / n,
            'dram_hit':    dram_hit_frac / n,
            'miss':        0.0}


def _simulate_fig3(scheme, reqs):
    """Block-level content-aware Fig 3 simulator → per-tier prefix cache.

    Cache unit is one radix chunk (10 tokens); each node has an LRU VRAM cache
    (FIG3_VRAM_TOKENS // 10 blocks) and an LRU DRAM cache (4× that).  A block
    evicted from VRAM demotes to DRAM; a DRAM eviction fully discards.

    Routing:
      - hash schemes: nid = hash(tokens[:128]) % NUM_NODES (deterministic).
      - radix_tree:   candidates = nodes that currently cache the GATE-depth
                      block of the request's path (either tier); pick one.
      - ludo:         candidates = nodes that cache ANY of the request's
                      Ludo-level blocks; joint-score with T_q + T_p + T_t.
      - oracle:       unified cluster cache, Bélády eviction.

    Match on routed node = longest contiguous cached prefix of the request's
    prefix_blocks, capped by the scheme's max advertised depth.  Per-chunk
    tier is tracked, so match contributes to vram_hit and dram_hit in
    proportion to the # of VRAM vs DRAM chunks along the matched prefix.
    """
    if scheme == 'central_directory':
        return _simulate_fig3_central_directory(reqs)
    if scheme == 'oracle':
        # Legacy Bélády-oracle kept callable for backward compat; no longer
        # in fig3b's default scheme list.
        return _simulate_fig3_oracle(reqs)

    VRAM_BLOCKS = FIG3_VRAM_TOKENS // RADIX_CHUNK_TOKENS   # per-node
    DRAM_BLOCKS = FIG3_DRAM_TOKENS // RADIX_CHUNK_TOKENS

    # Scheme-specific cap on how deep a cached prefix the discovery layer
    # can report (in chunks).  Beyond this, even if physically cached, the
    # scheme wouldn't advertise it.
    LUDO_POS  = [cs // RADIX_CHUNK_TOKENS for cs in CHUNK_SIZES]  # [12, 25, 51, 102, 204, 409]
    LUDO_MAX  = max(LUDO_POS)
    # Hash-placement schemes now share Ludo's multi-granularity match cap —
    # see top-of-file docstring.  The hit distribution remains structurally
    # lower than Ludo's because each level's owner is determined by an
    # independent hash.
    HASH_POS  = LUDO_MAX
    RT_POS    = RADIX_GATE_CHUNKS                          # 10  (100 tok)

    vram = [OrderedDict() for _ in range(NUM_NODES)]
    dram = [OrderedDict() for _ in range(NUM_NODES)]
    node_next_free = [0.0] * NUM_NODES
    rr = 0

    # SkyWalker's per-region approximate prefix tries (block-addressed here),
    # bounded FIFO at ≈ a single node's cache scale (VRAM_BLOCKS).  Guides
    # routing only; the served match is recomputed from the replica's real
    # cache below.  sw_replica[r][bid]->{local nodes}, sw_snapshot[r][bid]->
    # {remote regions r has forwarded to}.
    sw_replica  = [OrderedDict() for _ in range(NUM_REGIONS)]
    sw_snapshot = [OrderedDict() for _ in range(NUM_REGIONS)]
    SW_MAX = VRAM_BLOCKS
    # 100 ms heartbeat: availability view refreshed once per interval, stale
    # in between.
    sw_avail: set = set()
    sw_last_probe = None
    # Ludo's two-channel load view (piggyback + suppressed broadcast),
    # per viewer region — mirrors LudoDiscovery.lb_probe.
    lb_view = [[None] * NUM_NODES for _ in range(NUM_REGIONS)]
    lb_pending = [[0.0] * NUM_NODES for _ in range(NUM_REGIONS)]
    lb_fresh = {}
    lb_pb = []                       # (t_back, S, X)
    lb_last_probe = None

    def _lb_fold3(S, X, now_):
        b = LUDO_LB_EWMA_BETA
        for n2 in range(NUM_NODES):
            if NODE_REGION[n2] != X:
                continue
            q2 = max(0.0, node_next_free[n2] - now_)
            lb_view[S][n2] = (q2 if lb_view[S][n2] is None
                              else b * q2 + (1.0 - b) * lb_view[S][n2])
            lb_pending[S][n2] = 0.0

    def _sw_match(trie_r, blocks_, avail):
        best = set()
        for bid in blocks_:
            holders = trie_r.get(bid)
            if not holders:
                break
            inter = holders & avail
            if not inter:
                break
            best = inter
        return best

    def _sw_insert(trie_r, blocks_, target):
        for bid in blocks_:
            s = trie_r.get(bid)
            if s is None:
                if len(trie_r) >= SW_MAX:
                    trie_r.popitem(last=False)      # FIFO evict earliest
                trie_r[bid] = {target}
            else:
                s.add(target)
                trie_r.move_to_end(bid)

    flops_saved   = 0.0
    vram_hit_frac = 0.0
    dram_hit_frac = 0.0

    # ── Ludo's workload-aware eviction ────────────────────────────────────
    # Only applied when scheme == 'ludo'.  Each cached block gets a
    # priority score combining workload classification + access-count
    # frequency.  Eviction picks the LOWEST-priority block among a
    # **random sample of SAMPLE_K** (Redis/Caffeine-style approximate
    # LFU) rather than scanning all blocks — full min-scan over the
    # 26 K-block VRAM would cost 50 M inserts × 26 K ≈ 10¹² ops and
    # take tens of hours in Python.  Sample K = 16 keeps the policy
    # behaviourally close to true lowest-priority eviction (E[min of K]
    # ≈ the K/(K+1) quantile of the true distribution) at O(K) per
    # eviction.
    #
    # Implementation: maintain a parallel `keys_list` alongside each
    # OrderedDict for O(1) random-index access.  On insert → append,
    # on remove → swap-with-last + pop.  Non-Ludo schemes never touch
    # these structures (membership / LRU semantics still flow through
    # the OrderedDict).
    is_ludo = (scheme == 'ludo')
    # Narrowed from {1.0, 1.5, 2.0} to {1.0, 1.2, 1.5} so log(hits)+recency
    # stays the dominant term.  Wider weights over-prioritize rare high-value
    # workloads and drop total hit rate, which hurts TTFT (prefill is ~10-30×
    # more expensive than DRAM load, so total-hit dominates latency even
    # when VRAM/DRAM mix improves).
    WORKLOAD_PRIORITY = {'toolbench': 1.0, 'loogle': 1.2, 'aaps': 1.5}
    SAMPLE_K = 16
    # Python's built-in `random` is 10-100× faster than numpy for
    # scalar/small-K calls (numpy's per-call overhead — array alloc,
    # arg parsing, dispatch — dominates at these sizes).  We hit this
    # RNG on every eviction (tens of millions of calls) so the
    # difference is hours vs. minutes.
    import random as _rand
    _evict_rng = _rand.Random(0xEE)

    vram_meta: List[dict]     = [{} for _ in range(NUM_NODES)]
    dram_meta: List[dict]     = [{} for _ in range(NUM_NODES)]
    vram_keys: List[list]     = [[] for _ in range(NUM_NODES)]
    dram_keys: List[list]     = [[] for _ in range(NUM_NODES)]
    vram_idx : List[dict]     = [{} for _ in range(NUM_NODES)]
    dram_idx : List[dict]     = [{} for _ in range(NUM_NODES)]

    def _vram_add_key(nid, bid):
        vram_idx[nid][bid] = len(vram_keys[nid])
        vram_keys[nid].append(bid)

    def _vram_remove_key(nid, bid):
        i = vram_idx[nid].pop(bid)
        last = vram_keys[nid][-1]
        if last != bid:
            vram_keys[nid][i]   = last
            vram_idx[nid][last] = i
        vram_keys[nid].pop()

    def _dram_add_key(nid, bid):
        dram_idx[nid][bid] = len(dram_keys[nid])
        dram_keys[nid].append(bid)

    def _dram_remove_key(nid, bid):
        i = dram_idx[nid].pop(bid)
        last = dram_keys[nid][-1]
        if last != bid:
            dram_keys[nid][i]   = last
            dram_idx[nid][last] = i
        dram_keys[nid].pop()

    # Pre-bind hot paths for the sampling hot loop: attribute + global
    # lookups inside a `min(..., key=lambda)` scan cost surprising amounts
    # when repeated 48 M times.  `math.log1p` is ~10× faster than
    # `np.log1p` for scalars (numpy pays array-alloc overhead per call).
    from math import log1p as _log1p
    _WP = WORKLOAD_PRIORITY
    _WP_DEFAULT = 1.0

    def _sample_min_priority(keys_list, nid, tier, now_):
        """Sampled approximate-LFU eviction — returns the lowest-priority
        block among SAMPLE_K random draws.  The score function is inlined
        here (instead of calling `_block_score`) so the hot `min`-over-
        sample loop avoids Python function-call overhead and dict-lookup
        repetition."""
        if tier == 'VRAM':
            meta_nid = vram_meta[nid]
        else:
            meta_nid = dram_meta[nid]

        n = len(keys_list)
        if n <= SAMPLE_K:
            sample = keys_list
        else:
            # random.choices: C-level with-replacement sampling.
            sample = _evict_rng.choices(keys_list, k=SAMPLE_K)

        best_bid   = sample[0]
        best_score = float('inf')
        for b in sample:
            m = meta_nid.get(b)
            if m is None:
                s = 0.0
            else:
                age_s = (now_ - m['last_t']) / 1000.0
                if age_s < 0.0:
                    age_s = 0.0
                s = _WP.get(m['ds'], _WP_DEFAULT) * (
                    _log1p(m['hits']) + 1.0 / (1.0 + age_s))
            if s < best_score:
                best_score = s
                best_bid   = b
        return best_bid

    def _evict_dram_lru(nid):
        bid = next(iter(dram[nid]))
        del dram[nid][bid]
        if is_ludo:
            dram_meta[nid].pop(bid, None)
            _dram_remove_key(nid, bid)

    def _evict_dram_workload(nid, now_):
        """Evict a sampled lowest-priority DRAM block (Ludo only)."""
        bid = _sample_min_priority(dram_keys[nid], nid, 'DRAM', now_)
        del dram[nid][bid]
        dram_meta[nid].pop(bid, None)
        _dram_remove_key(nid, bid)

    def _demote_vram_lru(nid):
        bid = next(iter(vram[nid]))
        del vram[nid][bid]
        if len(dram[nid]) >= DRAM_BLOCKS:
            _evict_dram_lru(nid)
        dram[nid][bid] = None

    def _demote_vram_workload(nid, now_):
        """Demote a sampled lowest-priority VRAM block to DRAM (Ludo)."""
        bid = _sample_min_priority(vram_keys[nid], nid, 'VRAM', now_)
        del vram[nid][bid]
        _vram_remove_key(nid, bid)
        meta = vram_meta[nid].pop(bid, None)
        if len(dram[nid]) >= DRAM_BLOCKS:
            _evict_dram_workload(nid, now_)
        dram[nid][bid] = None
        _dram_add_key(nid, bid)
        if meta is not None:
            dram_meta[nid][bid] = meta

    def _insert_vram(nid, bid, ds='toolbench', now_=0.0):
        if bid in vram[nid]:
            vram[nid].move_to_end(bid, last=True)
            if is_ludo and bid in vram_meta[nid]:
                vram_meta[nid][bid]['hits']   += 1
                vram_meta[nid][bid]['last_t'] = now_
            return
        if bid in dram[nid]:
            del dram[nid][bid]
            if is_ludo:
                _dram_remove_key(nid, bid)
                meta = dram_meta[nid].pop(bid, None)
            else:
                meta = None
        else:
            meta = None
        while len(vram[nid]) >= VRAM_BLOCKS:
            if is_ludo:
                _demote_vram_workload(nid, now_)
            else:
                _demote_vram_lru(nid)
        vram[nid][bid] = None
        if is_ludo:
            _vram_add_key(nid, bid)
            if meta is not None:
                meta['hits']   += 1
                meta['last_t'] = now_
                vram_meta[nid][bid] = meta
            else:
                vram_meta[nid][bid] = {
                    'ds':     ds,
                    'hits':   1,
                    'last_t': now_,
                }

    def _level_holders(bid):
        """On-the-fly discovery: nodes currently caching `bid` in either tier.
        Yields (nid, tier) pairs, 'VRAM' preferred over 'DRAM' per node."""
        for nid in range(NUM_NODES):
            if bid in vram[nid]:
                yield nid, 'VRAM'
            elif bid in dram[nid]:
                yield nid, 'DRAM'

    for req in reqs:
        plen    = req['prefix_len']
        hashes  = req['hashes']
        blocks  = req['prefix_blocks']
        region  = req.get('region_id', 0)
        ds_req  = req.get('dataset', 'toolbench')   # for workload-aware eviction
        now     = req.get('arrival_time', 0.0)
        nid     = None

        # ── Routing ────────────────────────────────────────────────────
        if scheme in ('consistent_hash', 'kademlia', 'chord'):
            # Multi-level hash-placement: for each applicable CHUNK_SIZES
            # level, determine that level's hash-owner via an independent
            # hash function, measure its currently-cached contiguous
            # match (up to this level's depth), and route to whichever
            # level's owner exposes the longest match.  This mirrors the
            # discovery-plane semantics in simulate.py exactly.
            best_nid = None
            best_m   = -1
            for l_idx, cs_tok in enumerate(CHUNK_SIZES):
                pos = LUDO_POS[l_idx]
                if pos == 0 or pos > len(blocks):
                    continue
                if cs_tok not in hashes:
                    continue
                owner = hash((hashes[cs_tok], l_idx)) % NUM_NODES
                # Longest contiguous cached prefix on this level's owner,
                # bounded by the level's depth.  VRAM or DRAM both count
                # here (the discovery layer does not distinguish tier for
                # routing — physical tier affects T_t downstream).
                m = 0
                for k in range(pos):
                    bid = blocks[k]
                    if bid in vram[owner] or bid in dram[owner]:
                        m += 1
                    else:
                        break
                if m > best_m:
                    best_m   = m
                    best_nid = owner
            if best_nid is None:
                best_nid = rr % NUM_NODES; rr += 1
            nid = best_nid

        elif scheme == 'ludo':
            # Candidate discovery along latency-derived cut chains: holders
            # in region X are probed at positions k·n₁(region,X)/CHUNK
            # (local chain = CHUNK_SIZES).  Level 1 of each cross chain IS
            # the latency floor, so too-short cross-region matches never
            # even surface as candidates.
            cand = {}   # nid -> best match_chunks across chain levels
            holders_memo = {}
            for X in range(NUM_REGIONS):
                for c in LUDO_PAIR_CUTS[region][X]:
                    pos = c // RADIX_CHUNK_TOKENS
                    if pos == 0:
                        continue
                    if pos > len(blocks):
                        break               # chain sorted ascending
                    bid = blocks[pos - 1]
                    if bid not in holders_memo:
                        holders_memo[bid] = list(_level_holders(bid))
                    for cnid, _tier in holders_memo[bid]:
                        if NODE_REGION[cnid] != X:
                            continue
                        if pos > cand.get(cnid, 0):
                            cand[cnid] = pos
            # Load-gated two-stage routing (mirrors route()): the pool is
            # every node that is not busy under the viewed load (own region
            # live, remote via the periodic EWMA broadcast); the chain scan
            # above only annotates pool nodes with reuse credit.
            if lb_pb:
                still_pb = []
                for t_back, S2, X2 in lb_pb:
                    if t_back > now:
                        still_pb.append((t_back, S2, X2))
                        continue
                    _lb_fold3(S2, X2, now)
                    lb_fresh[(S2, X2)] = t_back
                lb_pb[:] = still_pb
            if lb_last_probe is None or now - lb_last_probe >= LUDO_LB_BCAST_MS:
                for S2 in range(NUM_REGIONS):
                    for X2 in range(NUM_REGIONS):
                        if X2 == S2:
                            continue
                        if now - lb_fresh.get((S2, X2), -1e18) < LUDO_LB_BCAST_MS:
                            continue
                        _lb_fold3(S2, X2, now)
                        lb_fresh[(S2, X2)] = now
                lb_last_probe = now

            q_views = [(max(0.0, node_next_free[n] - now)
                        if NODE_REGION[n] == region
                        else (lb_view[region][n] or 0.0)
                             + lb_pending[region][n])
                       for n in range(NUM_NODES)]
            lam = min(_sim.LUDO_LAMBDA_MAX, min(q_views) / _sim.LUDO_LAMBDA_SCALE_MS)

            best_g = best_u = None          # gated / ungated optima
            best_g_score = best_u_score = float('inf')
            for cnid in range(NUM_NODES):
                creg = NODE_REGION[cnid]
                q_view = q_views[cnid]
                ml = cand.get(cnid, 0) * RADIX_CHUNK_TOKENS
                # Latency-aware floor: a cross-region match too short to cover
                # the network latency earns no reuse credit (full prefill).
                if creg != region and ml < REUSE_FLOOR_TOKENS[region][creg]:
                    ml = 0
                T_p  = (plen - ml) * C_PREFILL_MS_PER_TOKEN
                T_t  = (float(WAN_LATENCY_MS[region][creg])
                        if creg != region else 0.0)
                score = q_view + (1.0 + lam) * T_p + T_t
                if score < best_u_score:
                    best_u_score = score
                    best_u = cnid
                if (q_view <= _sim.LUDO_LB_BUSY_THRESH_MS
                        and score < best_g_score):
                    best_g_score = score
                    best_g = cnid
            # Pool-empty fallback: gate excluded everyone → ungated minimum.
            nid = best_g if best_g is not None else best_u
            # Self-dispatch accounting (fig3 nodes are single-slot).
            svc3 = ((plen - cand.get(nid, 0) * RADIX_CHUNK_TOKENS)
                    * C_PREFILL_MS_PER_TOKEN + plen * C_DECODE_MS_PER_TOKEN)
            lb_pending[region][nid] += svc3
            # Cross-region forward → response piggybacks the load stamp.
            creg3 = NODE_REGION[nid]
            if creg3 != region:
                lb_pb.append((now + q_views[nid] + svc3
                              + 1.5 * WAN_LATENCY_MS[region][creg3],
                              region, creg3))

        elif scheme == 'dynamo':
            # Accurate per-cluster index: local-first, cross-cluster on miss.
            local_cand = {}
            for pos in LUDO_POS:
                if pos == 0 or pos > len(blocks):
                    continue
                bid = blocks[pos - 1]
                for cnid, _t in _level_holders(bid):
                    if NODE_REGION[cnid] == region and pos > local_cand.get(cnid, 0):
                        local_cand[cnid] = pos
            local_hit = any(v > 0 for v in local_cand.values())
            local_ll  = min((n for n in range(NUM_NODES)
                             if NODE_REGION[n] == region),
                            key=lambda n: max(0.0, node_next_free[n] - now))
            cand = dict(local_cand)
            cand.setdefault(local_ll, 0)
            if not local_hit:
                # Local miss → cross-cluster fan-out (candidates in other regions).
                for pos in LUDO_POS:
                    if pos == 0 or pos > len(blocks):
                        continue
                    bid = blocks[pos - 1]
                    for cnid, _t in _level_holders(bid):
                        if NODE_REGION[cnid] != region and pos > cand.get(cnid, 0):
                            cand[cnid] = pos
            best_score = float('inf')
            for cnid, mpos in cand.items():
                ml   = mpos * RADIX_CHUNK_TOKENS
                creg = NODE_REGION[cnid]
                T_q  = max(0.0, node_next_free[cnid] - now)
                T_p  = (plen - ml) * C_PREFILL_MS_PER_TOKEN
                T_t  = (float(WAN_LATENCY_MS[region][creg])
                        if creg != region else 0.0)
                score = T_q + T_p + T_t
                if score < best_score:
                    best_score = score
                    nid = cnid

        elif scheme == 'skywalker':
            # Two-layer, locality-first, prefix-tree routing.  Availability
            # comes from a 100 ms heartbeat snapshot (stale between probes),
            # not the instantaneous free-slot state.
            if sw_last_probe is None or now - sw_last_probe >= SKYWALKER_PROBE_INTERVAL_MS:
                sw_avail = {n for n in range(NUM_NODES)
                            if node_next_free[n] <= now}
                sw_last_probe = now
            local_avail = {n for n in range(NUM_NODES)
                           if NODE_REGION[n] == region and n in sw_avail}
            if local_avail:
                cand = _sw_match(sw_replica[region], blocks, local_avail)
                pool = cand if cand else local_avail
                nid  = min(pool, key=lambda n: max(0.0, node_next_free[n] - now))
            else:
                # All local replicas full → available remote regions only.
                remote_avail = {}
                for n in range(NUM_NODES):
                    r = NODE_REGION[n]
                    if r != region and n in sw_avail:
                        remote_avail.setdefault(r, set()).add(n)
                if remote_avail:
                    rcand = _sw_match(sw_snapshot[region], blocks,
                                      set(remote_avail.keys()))
                    region_pool = rcand if rcand else set(remote_avail.keys())
                    tgt = min(region_pool,
                              key=lambda r: min(max(0.0, node_next_free[n] - now)
                                                for n in remote_avail[r]))
                    av    = remote_avail[tgt]
                    ncand = _sw_match(sw_replica[tgt], blocks, av)
                    pool  = ncand if ncand else av
                    nid   = min(pool,
                                key=lambda n: max(0.0, node_next_free[n] - now))
                else:
                    # Nothing available anywhere → queue at the local LB.
                    nid = min((n for n in range(NUM_NODES)
                               if NODE_REGION[n] == region),
                              key=lambda n: max(0.0, node_next_free[n] - now))
            # Selective push: the LB tracks its own dispatches — a probed-free
            # node stops looking available once pushed to (fig3 nodes are
            # single-slot, so one push consumes the whole credit), preventing
            # the rest of the heartbeat window from piling onto it.
            sw_avail.discard(nid)
            # Update the guiding tries with this routing decision.
            creg = NODE_REGION[nid]
            _sw_insert(sw_replica[creg], blocks, nid)
            if creg != region:
                _sw_insert(sw_snapshot[region], blocks, creg)

        elif scheme == 'radix_tree':
            if len(blocks) >= RT_POS:
                bid_gate = blocks[RT_POS - 1]
                holders = [cnid for cnid, _t in _level_holders(bid_gate)]
                if holders:
                    nid = min(holders,
                              key=lambda n: max(0.0, node_next_free[n] - now))
                else:
                    nid = min(range(NUM_NODES),
                              key=lambda n: max(0.0, node_next_free[n] - now))
            else:
                nid = min(range(NUM_NODES),
                          key=lambda n: max(0.0, node_next_free[n] - now))

        # ── Match computation — longest contiguous cached prefix on nid,
        #    capped by the scheme's discovery depth ────────────────────
        if scheme in ('consistent_hash', 'kademlia', 'chord'):
            cap = HASH_POS
        elif scheme == 'radix_tree':
            cap = RT_POS
        elif scheme == 'ludo':
            cap = LUDO_MAX
        else:
            cap = len(blocks)
        v_chunks = 0
        d_chunks = 0
        for k in range(min(cap, len(blocks))):
            bid = blocks[k]
            if bid in vram[nid]:
                v_chunks += 1
            elif bid in dram[nid]:
                d_chunks += 1
            else:
                break   # prefix must be contiguous
        match_len = (v_chunks + d_chunks) * RADIX_CHUNK_TOKENS

        # Latency-aware floor (ludo): if the chosen holder is cross-region and
        # the contiguous match is too short to cover the network latency, take
        # no reuse credit — pay full prefill, consistent with the routing gate.
        if scheme == 'ludo':
            creg = NODE_REGION[nid]
            if creg != region and match_len < REUSE_FLOOR_TOKENS[region][creg]:
                v_chunks = d_chunks = 0
                match_len = 0

        v_tokens = v_chunks * RADIX_CHUNK_TOKENS
        d_tokens = d_chunks * RADIX_CHUNK_TOKENS

        if plen > 0:
            frac_v = v_tokens / plen
            frac_d = d_tokens / plen
            vram_hit_frac += frac_v
            dram_hit_frac += frac_d
            flops_saved   += frac_v + frac_d

        # ── Cache update: touch matched blocks (may promote DRAM→VRAM),
        #    insert the remaining blocks of this request into VRAM ────
        matched_depth = v_chunks + d_chunks
        for k in range(matched_depth):
            _insert_vram(nid, blocks[k], ds_req, now)   # touch / promote
        for k in range(matched_depth, len(blocks)):
            _insert_vram(nid, blocks[k], ds_req, now)

        residual = plen - match_len
        service  = (residual * C_PREFILL_MS_PER_TOKEN
                    + plen   * C_DECODE_MS_PER_TOKEN)
        node_next_free[nid] = max(node_next_free[nid], now) + service

    n = len(reqs)
    return {'flops_saved': flops_saved   / n,
            'vram_hit':    vram_hit_frac / n,
            'dram_hit':    dram_hit_frac / n,
            'miss':        0.0}


def run_fig3(args, wgen, oracle_max_rate, ludo_max_rate):
    print('\n[Figure 3] Prefill FLOPs saved — content-aware single-pass …')
    # Central Directory replaces the old theoretical Oracle — matches fig1/
    # fig2/fig2b semantics.  See `_simulate_fig3_central_directory` docstring
    # for the three idealisations (omniscient / infinite memory / infinite
    # bandwidth).
    schemes = ['ludo', 'radix_tree', 'consistent_hash', 'chord', 'skywalker',
               'dynamo', 'central_directory']

    # Use the same operating point as Fig 2: 0.8 × Ludo's max sustainable rate.
    # At this load, queue dynamics matter — so Ludo's joint scoring is
    # exercised (longest-match vs queue-wait trade-off becomes non-trivial).
    rate_rps = max(5.0, ludo_max_rate * 0.8)
    print(f'  arrival rate = {rate_rps:.1f} req/s (0.8 × ludo_max)')

    # Cache the per-scheme simulation stats on disk so plot-only iterations
    # can skip the multi-minute replay loop.  Cache key is workload config;
    # if you change `_simulate_fig3` semantics, delete the cache file.
    if args.n_tenants > 1:
        n_fig3 = args.n_fig3_reqs or 10_000_000
        cache_path = RESULTS_DIR / (
            f'fig3_stats_mt_n{args.n_tenants}_r{n_fig3}_s{args.seed}.json')
    else:
        n_fig3 = None
        cache_path = RESULTS_DIR / f'fig3_stats_single_s{args.seed}.json'

    if cache_path.exists():
        with open(cache_path) as f:
            stats = json.load(f)
        if not all(s in stats for s in schemes):
            # Cache written by an older scheme list — regenerate.
            print(f'  cached stats {cache_path.name} missing schemes; '
                  're-running fig3')
            stats = None
    else:
        stats = None
    if stats is not None:
        print(f'  loaded cached stats from {cache_path.name}')
        for s in schemes:
            print(f'    {s:<18s} flops={stats[s]["flops_saved"]:.4f}  '
                  f'vram_pcache={stats[s]["vram_hit"]:.4f}  '
                  f'dram_pcache={stats[s]["dram_hit"]:.4f}')
    else:
        if args.n_tenants > 1:
            # Multi-tenant single-pass: generate N_fig3 requests from an
            # amplified (n_tenants × base_pool) pool via Zipf α = 1.1.  The
            # cache passes through VRAM → DRAM demotion early and spends
            # most of the run under eviction pressure, exposing realistic
            # tier-split hit rates (fig3b) that single-pass over 7,500
            # distinct entries cannot observe.
            from workload_multitenant import build_multitenant_fig3_requests
            print(f'  multi-tenant single-pass: n_tenants={args.n_tenants:,}, '
                  f'pool ≈ {args.n_tenants * 7500:,} entries, '
                  f'generating {n_fig3:,} requests …')
            reqs = build_multitenant_fig3_requests(
                n_tenants=args.n_tenants,
                n_requests=n_fig3,
                seed=args.seed)
            print(f'  total requests: {len(reqs):,}')
        else:
            print('  building content corpus …')
            corpus = _build_content_corpus(args.workload)
            reqs   = _zipf_request_stream(corpus, seed=args.seed, rate_rps=rate_rps)
            print(f'  total requests: {len(reqs):,}')

        stats = {}
        for s in schemes:
            stats[s] = _simulate_fig3(s, reqs)
            print(f'    {s:<18s} flops={stats[s]["flops_saved"]:.4f}  '
                  f'vram_pcache={stats[s]["vram_hit"]:.4f}  '
                  f'dram_pcache={stats[s]["dram_hit"]:.4f}')
        with open(cache_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'  stats cached → {cache_path.name}')

    # ── Unified Morandi-pink bar styling for Fig 3 / 3b ────────────────
    # Matches the consistent_hash line on Fig 1 / 2 / 2b — same LINE_FILL
    # face + LINE_COLORS edge, so the figure family reads as one palette.
    # Scheme identity is read from the x-axis tick labels, not from color
    # / hatch coding.  Edge linewidth thickened so each bar reads as a
    # distinct unit without the fill-pattern contrast.
    import matplotlib.patches as mpatches
    BAR_FACE  = LINE_FILL['consistent_hash']   # '#eba79c'  Morandi peach-pink
    BAR_EDGE  = LINE_COLORS['consistent_hash'] # '#985365'  mauve edge
    BAR_LW    = 2.5

    FIG3_LABELS = dict(SCHEME_LABELS)
    FIG3_LABELS['ludo']              = 'KVCompass'
    FIG3_LABELS['radix_tree']        = 'PlanetServe'
    FIG3_LABELS['central_directory'] = 'Central Directory'

    # ── Figure 3b: Stacked VRAM + DRAM cache hit rate (normalized) ─────
    # VRAM portion = solid pink; DRAM portion = same pink but translucent,
    # sitting on top.  No hatch.
    # Normalise to Central Directory's total hit rate (infinite-memory
    # network-optimal reference).  Every other scheme's VRAM/DRAM split is
    # reported as a fraction of this best achievable.
    cd_total = (stats['central_directory']['vram_hit']
                + stats['central_directory']['dram_hit'])
    if cd_total <= 0:
        cd_total = 1.0

    # DRAM segment uses the Ludo-blue palette (matches fig1/fig2 Ludo
    # line colour) so the two tiers are distinguishable by hue, not just
    # alpha.  VRAM keeps the Morandi peach-pink from the earlier design.
    DRAM_FACE = LINE_FILL['ludo']      # '#ACBFEA'  light blue
    DRAM_EDGE = LINE_COLORS['ludo']    # '#7e97ad'  Morandi blue edge

    x = np.arange(len(schemes))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    # Draw grid first (set_axisbelow pushes it behind bars) so the
    # dotted horizontal guide lines sit under the bars, not on top.
    ax.set_axisbelow(True)
    for k, s in enumerate(schemes):
        v = stats[s]['vram_hit'] / cd_total
        d = stats[s]['dram_hit'] / cd_total
        ax.bar(x[k], v, 0.65,
               facecolor=BAR_FACE, edgecolor=BAR_EDGE, linewidth=BAR_LW,
               clip_on=False)
        ax.bar(x[k], d, 0.65, bottom=v,
               facecolor=DRAM_FACE, edgecolor=DRAM_EDGE, linewidth=BAR_LW,
               clip_on=False)
        # Label each stack segment.  Segments tall enough (≳ 0.06) get
        # their value centred INSIDE the segment; thinner segments get
        # the value just above the top of the stacked bar so even a
        # hair-line DRAM slice (e.g. PlanetServe ~ 0.01) is still
        # annotated.  Font / size identical to the in-bar labels.
        if v >= 0.06:
            ax.text(x[k], v / 2, f'{v:.2f}',
                    ha='center', va='center', fontsize=16,
                    fontname='STIXGeneral')
        elif v > 0:
            ax.text(x[k], v + d + 0.015, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=16,
                    fontname='STIXGeneral')
        if d >= 0.06:
            ax.text(x[k], v + d / 2, f'{d:.2f}',
                    ha='center', va='center', fontsize=16,
                    fontname='STIXGeneral')
        elif d > 0:
            ax.text(x[k], v + d + 0.015, f'{d:.2f}',
                    ha='center', va='bottom', fontsize=16,
                    fontname='STIXGeneral')
    ax.set_xticks(x)
    ax.set_xticklabels([FIG3_LABELS[s] for s in schemes],
                       rotation=20, ha='right', fontsize=16,
                       fontname='STIXGeneral')
    ax.set_ylabel('Normalized Prefix Cache Hit Rate', fontsize=18,
                  fontname='STIXGeneral')
    # Cap y-axis at exactly 1.0 — oracle is the reference and never
    # exceeds 1, and the previous 1.15 padding left an ugly gap above
    # the legend.
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='y', labelsize=16)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(int_like_fmt))
    ax.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
    # Two legend entries — same patch size / edge thickness so they read
    # as visually equal-weight.  DRAM now uses the Ludo-blue palette,
    # VRAM keeps the Morandi peach-pink; hue carries the tier identity
    # instead of alpha shading.
    vram_patch = mpatches.Patch(
        facecolor=BAR_FACE, edgecolor=BAR_EDGE,
        linewidth=BAR_LW, label='VRAM Hit')
    dram_patch = mpatches.Patch(
        facecolor=DRAM_FACE, edgecolor=DRAM_EDGE,
        linewidth=BAR_LW, label='DRAM Hit')
    # `fontsize` is silently ignored when `prop` is supplied — size must
    # live inside the prop dict.
    fig.legend(handles=[vram_patch, dram_patch],
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False,
               prop={'family': 'STIXGeneral', 'size': 16})
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, 'fig3b_cache_hit_rate.pdf')


# ─── Figure 3c: Effective Prefill FLOPs Saved per Second ─────────────────────

def run_fig3c(args, wgen, oracle_max_rate, ludo_max_rate):
    """Effective FLOPs-saving *rate* under realistic queue dynamics.

    Fig 3 measures per-request savings (hit rate × prefix_len), which
    over-credits hash schemes: they get high cache locality but overloaded
    home-nodes queue requests for seconds, so the aggregate work actually
    delivered per second collapses.  Fig 3c closes that gap by weighting
    per-request savings against the *wall-clock rate* at which the scheme
    finishes work.  Formula:

        effective_tokens_saved_per_sec
            = Σ (flops_saved_frac_r × prefix_len_r)  over all records r
              ÷ (t_end_ms / 1000)

    where t_end_ms is the latest request-completion timestamp in the run
    (arrival_time + e2e_latency_ms).  Under overload t_end explodes, so
    effective rate crashes — exactly the penalty Fig 3 was missing.

    All values are normalized to oracle = 1.0 so the bar chart shape is
    directly comparable to Figure 3.
    """
    print('\n[Figure 3c] Effective FLOPs saved / sec — queue-aware …')
    schemes = ['ludo', 'radix_tree', 'consistent_hash', 'chord', 'skywalker', 'dynamo', 'oracle']

    # Same operating point as Fig 2 / Fig 3 so the figures read as one
    # side-by-side story.
    rate_rps = max(5.0, ludo_max_rate * 0.8)
    n_req, warmup = rate_to_n_warmup(rate_rps, args.quick)
    print(f'  arrival rate = {rate_rps:.1f} req/s (0.8 × ludo_max), '
          f'N={n_req:,} requests')

    stats = {}
    for s in schemes:
        recs, _ = run_scheme(s, rate_rps, args.workload, n_req, warmup,
                             args.seed, wgen)
        if not recs:
            stats[s] = {'saved': 0.0, 't_end_s': 1.0, 'rate': 0.0}
            continue
        total_saved = sum(r.flops_saved_frac * r.prefix_len for r in recs)
        t_end_ms    = max(r.arrival_time + r.e2e_latency_ms for r in recs)
        t_end_s     = max(t_end_ms / 1000.0, 1e-3)
        eff_rate    = total_saved / t_end_s
        stats[s] = {'saved': total_saved, 't_end_s': t_end_s, 'rate': eff_rate}
        print(f'    {s:<18s} saved_tokens={total_saved:>12,.0f}  '
              f't_end={t_end_s:>7.2f}s  '
              f'rate={eff_rate:>10,.0f} tok/s')

    # Horizontal bar chart with log-x — dynamic range spans ~3 orders of
    # magnitude (hash ≈ 50 TFLOPs/s, oracle ≈ 20 PFLOPs/s), so a linear
    # bar chart crushes the hash schemes to invisible stubs.  Log-x keeps
    # every bar readable while preserving the order-of-magnitude story.
    # Unified Morandi-pink fill + mauve edge (identical to fig3 / fig3b)
    # — scheme identity is read off the y-axis, not encoded in color.
    FIG3_LABELS = dict(SCHEME_LABELS)
    FIG3_LABELS['ludo']       = 'KVCompass'
    FIG3_LABELS['radix_tree'] = 'PlanetServe'

    BAR_FACE = LINE_FILL['consistent_hash']    # '#eba79c'
    BAR_EDGE = LINE_COLORS['consistent_hash']  # '#985365'
    BAR_LW   = 2.5

    # Llama-3.1-8B forward pass ≈ 2 × 8B params = 16 GFLOPs per token
    # (matches the 0.05 ms/token prefill rate declared in simulate.py at
    # ~70-80 % of A100 BF16 peak; see simulate.py lines 40-45).
    PREFILL_FLOPS_PER_TOKEN = 16e9

    # SI-prefix formatter so 2.05e16 FLOPs/s renders as "20.5P" rather
    # than a 17-digit string.  Mirrors engineering notation used in
    # systems papers.
    def _si(x: float) -> str:
        if x <= 0:
            return '0'
        for prefix, base in [('P', 1e15), ('T', 1e12),
                              ('G', 1e9), ('M', 1e6), ('K', 1e3)]:
            if x >= base:
                return f'{x / base:.2f}{prefix}'
        return f'{x:.0f}'

    # Bars drawn top-to-bottom in `schemes` order (same as fig3).  Ludo
    # sits at the top, oracle at the bottom.
    y_pos = np.arange(len(schemes))[::-1]
    flops_abs = [stats[s]['rate'] * PREFILL_FLOPS_PER_TOKEN for s in schemes]

    fig, ax = plt.subplots(figsize=(8, 5.0))
    for k, s in enumerate(schemes):
        ax.barh(y_pos[k], flops_abs[k],
                height=0.65,
                facecolor=BAR_FACE,
                edgecolor=BAR_EDGE,
                linewidth=BAR_LW)
        # Pinned just to the right of each bar.  fontsize=16 matches the
        # axis tick labels (and the bar-top annotations in fig3 / fig3b).
        ax.text(flops_abs[k] * 1.25, y_pos[k], _si(flops_abs[k]),
                ha='left', va='center', fontsize=16,
                fontname='STIXGeneral')

    ax.set_xscale('log')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FIG3_LABELS[s] for s in schemes],
                       fontsize=16, fontname='STIXGeneral')
    ax.set_xlabel('Effective Prefill FLOPs Saved per Second',
                  fontsize=18, fontname='STIXGeneral')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', length=0)  # hide y-axis tick marks (labels still show)
    # Breathing room above/below the first/last bar.
    ax.set_ylim(-0.7, len(schemes) - 0.3)
    # Leave log-x room for the right-side annotation (largest value ×30).
    xmin_auto, xmax_auto = ax.get_xlim()
    ax.set_xlim(max(xmin_auto, min(flops_abs) * 0.5), xmax_auto * 30)
    ax.xaxis.set_major_locator(
        mticker.LogLocator(base=10, subs=(1.0,), numticks=20))
    ax.grid(True, axis='x', linestyle='dotted', alpha=0.8, linewidth=1)
    add_log_note(ax, 'x')
    fig.tight_layout()
    save_fig(fig, 'fig3c_effective_flops_rate.pdf')


# ─── Figure 4: Steady-State Self-Partitioning Heatmap ────────────────────────

def _run_oracle_uniform_lb(rate_rps, workload, n_req, warmup, seed, wgen):
    """Fig 4-specific oracle: globally least-loaded routing.

    For the self-partitioning heatmap, the theoretical best is *perfect
    cluster-wide load balancing* — every request lands on the globally
    least-loaded node, regardless of region or cache state.  This is what
    minimises hot-spots and makes the per-node prefill rate most uniform.
    No cache-hit assumption (residual = prefix_len, full prefill every time),
    so the heatmap shows honest routing fairness, not inflated service times.
    """
    rate_ms = rate_rps / 1000.0
    rng     = np.random.default_rng(seed)
    total   = warmup + n_req

    wgen.reset_stream(seed)          # paired with run_scheme's stream
    wgen_state = wgen.rng.bit_generator.state
    nodes = make_nodes()
    for node in nodes:
        node.reset()

    records = []
    t = 0.0
    for i in range(total):
        t += rng.exponential(1.0 / rate_ms)
        req = wgen.sample(i, t, workload)
        # Globally least-loaded node (ignore region/cache)
        # Random tie-breaking: at low rates (Oracle heatmap panel), many
        # nodes have equal queue_ms = 0 early on; without jitter Python's
        # `min` always picks index 0, filling nodes monotonically and
        # producing a visibly skewed heatmap (rows 0–25 hot, 25–63 cold).
        # A sub-millisecond random term keeps the least-loaded ordering
        # intact while scattering ties uniformly across the 64-node ring.
        nid  = min(range(NUM_NODES),
                   key=lambda n: nodes[n].queue_ms(t) + rng.random() * 1e-6)
        node = nodes[nid]
        ttft_ms = node.schedule(req.prefix_len, req.prefix_len, t)
        e2e_ms  = ttft_ms + req.gen_len * C_DECODE_STEP_MS_PER_TOKEN
        if i >= warmup:
            records.append(RequestRecord(
                request_id       = i,
                prefix_id        = req.prefix_id,
                prefix_len       = req.prefix_len,
                arrival_time     = t,
                ttft_ms          = ttft_ms,
                e2e_latency_ms   = e2e_ms,
                match_len        = 0,
                cache_hit        = False,
                flops_saved_frac = 0.0,
                node_id          = nid,
                slo_met          = ttft_ms <= TTFT_SLO_MS,
                hit_tier         = 'MISS',
                gen_len          = req.gen_len,
            ))
    wgen.rng.bit_generator.state = wgen_state
    return records, nodes


def run_fig4(args, wgen, oracle_max_rate, ludo_max_rate):
    print('\n[Figure 4] Steady-state self-partitioning heatmap …')
    # Use the same operating point as fig2 / fig2b (= 1.0 × Ludo knee)
    # so all three cross-sectional figures depict the fleet under an
    # identical arrival pattern.  At this rate, Oracle's no-cache
    # uniform-LB is mildly saturated (~1.5× single-node capacity) and
    # hash home-nodes concentrate 3-5× that, producing the deep-red
    # hot-spot stripes the heatmap is designed to expose.  Sample count
    # per (node, time_bin) ≈ 18, large enough to suppress Poisson noise.
    rate           = max(5.0, ludo_max_rate * 1.0)
    # Heatmap-specific duration — 3× the fig1/2 duration.  With realistic
    # Llama-3.1-8B service times, Oracle knee ≈ 1 k req/s, so the default
    # 10 s × 64 nodes × 20 bins ≈ 6 requests per cell — Poisson noise
    # √6 ≈ 40 % CV is visible as speckled Oracle / Ludo panels.  Tripling
    # duration gives ~19 req/cell, CV √19/19 ≈ 23 %, smoothing the uniform
    # baseline without changing the underlying physics.
    FIG4_DURATION_S = 3.0 if args.quick else 30.0
    FIG4_WARMUP_S   = 1.0 if args.quick else 3.0
    n_req  = max(1, int(rate * FIG4_DURATION_S))
    warmup = max(1, int(rate * FIG4_WARMUP_S))
    print(f'  duration = {FIG4_DURATION_S:.0f} s (fig4-specific, '
          f'3× fig1 to smooth Poisson noise), N = {n_req:,}')

    def collect_heatmap(scheme):
        if scheme == 'oracle':
            # Per-metric oracle: globally-balanced routing (uniform heat).
            recs, nodes = _run_oracle_uniform_lb(rate, args.workload, n_req,
                                                 warmup, args.seed, wgen)
        else:
            recs, nodes = run_scheme(scheme, rate, args.workload, n_req,
                                     warmup, args.seed, wgen)
        if not recs:
            return np.zeros((NUM_NODES, 10))
        t_start = recs[0].arrival_time
        t_end   = recs[-1].arrival_time
        duration = t_end - t_start
        n_bins   = 20
        bin_size = duration / n_bins if duration > 0 else 1.0

        # Bin by *arrival time* of each request (= when the scheme made its
        # routing decision), not by the moment the node's slot opened.
        # Accumulator is the per-request GPU compute time (same formula
        # the simulator uses in Node.schedule for queue modelling):
        #     service_ms = (prefix_len − match_len) · C_PREFILL
        #                + prefix_len · C_DECODE_MS_PER_TOKEN
        # Cache hits automatically reduce the residual prefill term, so
        # the heatmap reflects actual GPU load (not routed-tokens).
        # Dividing per-node per-bin work by bin_size yields compute
        # utilization ∈ [0, 1] — the LLM-serving community's standard
        # load-balancing metric (see Preble NSDI '24, Orca, Splitwise).
        heatmap = np.zeros((NUM_NODES, n_bins))
        for r in recs:
            t = r.arrival_time
            if t < t_start or t >= t_end:
                continue
            b = int((t - t_start) / bin_size)
            if b >= n_bins:
                continue
            residual_tokens = int(r.prefix_len * (1.0 - r.flops_saved_frac))
            service_ms = (residual_tokens * C_PREFILL_MS_PER_TOKEN
                          + r.prefix_len * C_DECODE_MS_PER_TOKEN)
            heatmap[r.node_id, b] += service_ms
        # Normalise by the scheme's own mean cell — value = "load factor
        # relative to perfectly balanced".  Removes the "absolute work"
        # confound (cache hits lower Ludo's total work, which would
        # otherwise make Ludo's peak appear lower than Oracle's without
        # implying better balance) and isolates distribution uniformity.
        # 1.0 = balanced, >1 = hot, <1 = cold.  Gini-coefficient-style.
        mean_cell = heatmap.mean() if heatmap.sum() > 0 else 1.0
        if mean_cell <= 0:
            mean_cell = 1.0
        heatmap /= mean_cell
        return heatmap

    # Oracle first as the balanced reference, ludo beside it, then all
    # baselines (8 panels -> a 2x4 grid).
    all_schemes_fig4 = ['oracle', 'ludo', 'central_directory', 'radix_tree',
                        'consistent_hash', 'chord', 'skywalker', 'dynamo']

    # Cache the heatmap arrays so re-renders (font tweaks, layout changes)
    # don't require re-running the multi-minute simulation loop.
    cache_path = RESULTS_DIR / (
        f'fig4_heatmaps_{args.workload}_n{n_req}_w{warmup}_s{args.seed}.json'
    )
    heatmaps = {}
    if cache_path.exists():
        print(f'  loading cached heatmaps ← {cache_path.name}')
        with open(cache_path) as f:
            cached = json.load(f)
        for scheme in all_schemes_fig4:
            if scheme in cached:
                heatmaps[scheme] = np.array(cached[scheme], dtype=float)
    missing = [s for s in all_schemes_fig4 if s not in heatmaps]
    if missing:
        for scheme in missing:
            print(f'  running {scheme} …')
            heatmaps[scheme] = collect_heatmap(scheme)
        with open(cache_path, 'w') as f:
            json.dump({s: heatmaps[s].tolist() for s in all_schemes_fig4}, f)
        print(f'  heatmap arrays cached → {cache_path.name}')

    # Panel order: KVCompass first, everything else (incl. the oracle)
    # ascending by single-cell peak.
    all_schemes_fig4 = (['ludo'] +
                        sorted([s for s in all_schemes_fig4 if s != 'ludo'],
                               key=lambda s: float(
                                   np.max(heatmaps[s].mean(axis=1)))))

    sorted_ids = sorted(range(NUM_NODES), key=lambda i: NODE_REGION[i])

    # Grid sized to the scheme count (4 columns); any trailing empty cells are
    # hidden below.  (7 schemes → 4×2 with one blank.)
    ncols = 4
    nrows = (len(all_schemes_fig4) + ncols - 1) // ncols
    # Colour-scale: utilization values span 0 → ~3.7 across schemes
    # (hash home-nodes are ~3× their own scheme's mean under overload).
    # Using vmax = 1.0 or 1.2 compresses every overloaded cell to the
    # same saturated red and hides the real inter-scheme gap.  Setting
    # vmax to the 99th-percentile cell across ALL schemes stretches the
    # colour range across the observed load span: Oracle/Ludo's mean
    # (~0.3) maps to pale orange, hash home-nodes (~3.0) map to deep
    # crimson, and the gradient between the two is fully visible.
    # Fixed 0-2 scale: covers every node-level peak shown in the titles
    # (max 1.85) with a round top tick; the few transient cells above 2
    # saturate, exactly as they did under the old 99th-percentile cap.
    vmax_global = 2.0

    # Classic yellow→red heatmap (white-cream base → gold → orange → red →
    # deep crimson).  Reads as "temperature": cold cells pale yellow,
    # hot-spots saturate to dark red — standard heatmap convention.
    from matplotlib.colors import LinearSegmentedColormap
    heatmap_cmap = LinearSegmentedColormap.from_list(
        'classic_yr',
        ['#FFFBEA', '#FFD966', '#FF8C00', '#D7301F', '#7F0000'],
    )

    labels_fig4 = dict(SCHEME_LABELS)
    labels_fig4['ludo']              = 'KVCompass'
    labels_fig4['radix_tree']        = 'PlanetServe'
    labels_fig4['central_directory'] = 'Central Directory'

    # Fonts originally doubled (×2) for legibility; now bumped to ×3 of the
    # baseline (16/17pt → 48/50pt; xticks 16→40) so they match the 1.5×
    # enlarged subplots visually.  Figure height grows (12 → 14) to leave
    # room for the taller titles + xlabels while subplot inches stay fixed.
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.3 * ncols, 7.7 * nrows),
                             sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()
    # Hide any trailing cells not filled by a scheme.
    for j in range(len(all_schemes_fig4), len(axes_flat)):
        axes_flat[j].axis('off')

    for k, scheme in enumerate(all_schemes_fig4):
        ax = axes_flat[k]
        hm_sorted = heatmaps[scheme][sorted_ids]
        im = ax.imshow(hm_sorted, aspect='auto', interpolation='nearest',
                       cmap=heatmap_cmap, vmin=0, vmax=vmax_global)
        # Slightly wider-than-tall bounding box (w:h ≈ 1.43) — looks more
        # balanced than the original auto-fit rectangle without going all the
        # way to a square.  imshow's aspect='auto' still fills the box, so
        # cell colour mapping is unaffected.
        ax.set_box_aspect(0.7)
        # Peak utilization across all (node, time_bin) cells — included
        # in the subplot title so the magnitude of hot-spots is readable
        # even when the colorbar clips at vmax = 1.2.
        peak = float(np.max(hm_sorted)) if hm_sorted.size else 0.0
        # Node-level peak: hottest node's TIME-AVERAGED load factor.  The
        # cell peak flags transient single-window spikes; the node peak
        # flags sustained structural imbalance (what queueing punishes).
        node_peak = float(np.max(hm_sorted.mean(axis=1))) if hm_sorted.size else 0.0
        ax.set_title(f'{labels_fig4[scheme]}\n(peak: {node_peak:.2f})',
                     fontsize=62, fontname='STIXGeneral',
                     color='black', linespacing=0.85)
        ax.set_xlabel('Time bins', fontsize=64, fontname='STIXGeneral')
        if k % ncols == 0:
            ax.set_ylabel('Node (by region)', fontsize=62,
                          fontname='STIXGeneral')
        # Lock x-ticks to the original 0/5/10/15 grid.  X tick font kept
        # smaller than the rest so four labels still fit without overlap.
        ax.set_xticks([0, 5, 10, 15])
        ax.tick_params(axis='y', labelsize=58)
        ax.tick_params(axis='x', labelsize=52)
        for off in REGION_OFFSETS[1:]:
            ax.axhline(off - 0.5, color='white',
                       linewidth=0.8, linestyle=':')

    # Single shared colorbar on the right.  Position/width recomputed for
    # the (22,12) figure so the absolute cbar width (~0.24") and the gap
    # between the rightmost subplot and the cbar are unchanged from the
    # (16,9) layout.
    cbar_ax = fig.add_axes([0.92, 0.15, 0.011, 0.7])
    sm = plt.cm.ScalarMappable(cmap=heatmap_cmap,
                                norm=plt.Normalize(vmin=0, vmax=vmax_global))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Load factor', fontsize=62, fontname='STIXGeneral')
    cbar.ax.tick_params(labelsize=58)

    # Tight explicit spacing — only enough room for the 2× fonts (xlabel
    # below, 2-line title above) and a narrow gap between columns so the
    # title text doesn't touch the neighbour's title.  tight_layout is
    # avoided here because it bakes in extra padding around set_box_aspect.
    # Subplots_adjust recomputed for figsize=(22,14) and ×3 fonts so subplot
    # absolute size stays ~6.18×4.33 in; the extra figh goes into top/bottom
    # margins (taller title/xlabel) and a slightly bigger inter-row gap to
    # fit the larger 2-line bottom-row title under the top-row xlabel.
    plt.subplots_adjust(left=0.065, right=0.895, top=0.91, bottom=0.10,
                        wspace=0.10, hspace=0.72)
    save_fig(fig, 'fig4_heatmap.pdf')


# ─── Figure 5: Closed-loop service throughput (SkyWalker-style) ──────────────

def run_fig5(args, wgen):
    """Closed-loop service throughput, following the SkyWalker paper (§5.1).

    A fixed population of clients (allocated to regions by the demand
    distribution) each keeps exactly one request in flight; the next request
    is issued the moment the previous one completes end-to-end.  Service
    throughput = total tokens (prefix + generated) completed per second in
    the measurement window.  Reported per scheme as a bar chart.
    """
    print('\n[Figure 5] Closed-loop service throughput (SkyWalker-style) …')
    n_clients = args.tpt_clients
    dur_s     = 10 if args.quick else args.tpt_duration
    dur_ms    = dur_s * 1000.0
    warm_ms   = min(0.2 * dur_ms, 10_000.0)
    print(f'  clients = {n_clients} (∝ demand weights), '
          f'duration = {dur_s}s (warmup {warm_ms/1000:.0f}s)')

    mt_tag = f'_mt_n{args.n_tenants}' if args.n_tenants > 1 else ''
    cache_path = (RESULTS_DIR /
                  f'fig5_tpt_{args.workload}_c{n_clients}_d{dur_s}'
                  f'{mt_tag}_s{args.seed}.json')

    results = None
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if all(s in cached for s in ALL_SCHEMES):
            results = cached
            print(f'  loaded cached throughput from {cache_path.name}')
    if results is None:
        results = {}
        for scheme in ALL_SCHEMES:
            wgen.reset_stream(args.seed)   # paired streams across schemes
            nodes = make_nodes()
            disc  = make_discovery('centralized'
                                   if scheme == 'central_directory'
                                   else scheme)
            sim   = Simulator(nodes, disc, wgen,
                              arrival_rate_rps=1000.0,   # unused in closed loop
                              scheme=scheme, workload=args.workload,
                              warmup=0, seed=args.seed)
            r = sim.run_closed_loop(n_clients, dur_ms, warm_ms)
            results[scheme] = r
            print(f'    {scheme:<18} Tpt={r["tpt_total_tok_s"]:>12,.0f} tok/s '
                  f'(out {r["tpt_out_tok_s"]:>10,.0f} tok/s, '
                  f'{r["req_s"]:>7.1f} req/s, '
                  f'ttft p50={r["ttft_p50_ms"]:.0f}ms)')
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=1)
        print(f'  throughput cached → {cache_path.name}')

    # ── Bar chart ────────────────────────────────────────────────────────
    FIG5_LABELS = dict(SCHEME_LABELS)
    FIG5_LABELS['ludo']              = 'KVCompass'
    FIG5_LABELS['radix_tree']        = 'PlanetServe'
    FIG5_LABELS['central_directory'] = 'Central Directory'

    schemes = list(ALL_SCHEMES)
    vals    = np.array([results[s]['tpt_total_tok_s'] for s in schemes])
    x       = np.arange(len(schemes))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_axisbelow(True)
    for k, s in enumerate(schemes):
        ax.bar(x[k], vals[k], 0.65,
               facecolor=LINE_FILL.get(s, '#cccccc'),
               edgecolor=LINE_COLORS.get(s, '#555555'),
               linewidth=2.5, clip_on=False,
               hatch=SCHEME_HATCHES.get(s, ''))
        ax.text(x[k], vals[k] + 0.015 * vals.max(),
                f'{vals[k]/1e6:.2f}M' if vals[k] >= 1e6
                else f'{vals[k]/1e3:.0f}K',
                ha='center', va='bottom', fontsize=14,
                fontname='STIXGeneral')
    ax.set_xticks(x)
    ax.set_xticklabels([FIG5_LABELS[s] for s in schemes],
                       rotation=20, ha='right', fontsize=16,
                       fontname='STIXGeneral')
    ax.set_ylabel('Throughput (token/s)', fontsize=18,
                  fontname='STIXGeneral')
    ax.set_ylim(0, vals.max() * 1.12)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, axis='y', linestyle='dotted', alpha=0.8, linewidth=1)
    fig.tight_layout()
    save_fig(fig, 'fig5_throughput.pdf')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Section 3 simulation (Figures 1-4)')
    parser.add_argument('--quick',    action='store_true',
                        help='Fast run with fewer requests (for debugging)')
    parser.add_argument('--workload', default='mixed',
                        choices=['mixed', 'toolbench', 'loogle', 'aaps',
                                 'wildchat', 'lmsys'],
                        help='Workload for Figures 1-2 (default: mixed). '
                             'wildchat requires datasets/wildchat.jsonl '
                             '(run datasets/prep_wildchat.py first); '
                             'lmsys requires datasets/lmsys.jsonl '
                             '(datasets/prep_lmsys.py).')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--results-dir', default=None,
                        help='Output/cache directory for figures and JSON '
                             'caches (default: ./results next to this '
                             'script).  Relative paths are resolved against '
                             'the script directory.  Use a sibling folder '
                             '(e.g. results_wildchat) to keep a separate '
                             "workload's outputs from colliding with the "
                             'default results/ caches.')
    parser.add_argument('--figs',     default='1234',
                        help='Which figures to generate. Digits 1/2/3/4, '
                             "5 for the closed-loop throughput bar chart "
                             "(SkyWalker-style Tpt), plus optional 'm' for "
                             'the merged fig1+fig2+fig2b panel (loads from '
                             'JSON caches; requires fig1 + fig2 to have run '
                             "at least once). e.g. --figs 12m (default: 1234)")
    parser.add_argument('--fig1-rates', default='',
                        help='fig1: comma-separated arrival-rate sweep '
                             'override (default: built-in RATES_FULL/'
                             'RATES_QUICK).  Use to extend the sweep past '
                             '6000 req/s on workloads where the knees need '
                             'more runway.')
    parser.add_argument('--fig1-seeds', default='',
                        help='fig1: comma-separated seeds; the sweep runs '
                             'once per seed and plots the mean p99 (±std '
                             'band).  Empty = single seed (--seed).  p99 on '
                             'heavy-tailed placement (e.g. PlanetServe gate '
                             'affinity under conversational Zipf heads) is '
                             'high-variance; averaging is the honest curve.')
    parser.add_argument('--tpt-clients', type=int, default=1024,
                        help='fig5: closed-loop client population, allocated '
                             'to regions by demand weights (default 1024 — '
                             '4× the fleet batch slots, keeping replicas at '
                             'the sustained high utilisation the SkyWalker '
                             'paper evaluates at)')
    parser.add_argument('--tpt-duration', type=int, default=30,
                        help='fig5: simulated seconds per scheme (default 30; '
                             'first min(20%%, 10s) is cache warm-up excluded '
                             'from the measurement window)')
    parser.add_argument('--n-tenants', type=int, default=1,
                        help='Amplify the mixed-workload prefix pool via '
                             'multi-tenant replication (1 = single-tenant, '
                             "original §3 behaviour; 1400 ≈ 10 M-entry pool "
                             'matching §4 Internet-scale).  Each tenant has '
                             'a private 30–80-token system prompt; distinct '
                             'prefix_ids remain globally unique across the '
                             'n_tenants × 7,500 amplified pool.')
    parser.add_argument('--n-fig3-reqs', type=int, default=0,
                        help='Total requests fig3 / fig3b process in '
                             'single-pass mode.  0 = auto: 7,500 when '
                             'n_tenants=1 (legacy), 10 M when n_tenants>1.  '
                             'Ignored for fig1 / fig2 / fig2b / fig4 which '
                             'keep their fixed-duration arrival window.')
    args = parser.parse_args()

    # Apply --results-dir (was parsed but never wired up: every run wrote to
    # the default results/ and final outputs were copied out by hand).
    # Relative paths resolve against the script directory, per the help text.
    if args.results_dir:
        global RESULTS_DIR
        p = Path(args.results_dir)
        RESULTS_DIR = p if p.is_absolute() else HERE / p
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f'Results dir: {RESULTS_DIR}')

    # Per-workload node eviction policy (motivation eviction analysis):
    # LFU for the mixed workload's stationary Zipf template reuse, LRU for
    # conversational / everything else.  Uniform across nodes and schemes.
    _sim.EVICT_POLICY = 'lfu' if args.workload == 'mixed' else 'lru'

    print(f'Mode: {"QUICK" if args.quick else "FULL"} | '
          f'workload={args.workload} | seed={args.seed} | '
          f'n_tenants={args.n_tenants} | evict={_sim.EVICT_POLICY.upper()}')
    print(f'Cost model: Llama-{_sim.SIM_MODEL.upper()} '
          f'(C_prefill={_sim.C_PREFILL_MS_PER_TOKEN} ms/tok, '
          f'KV={_sim.KV_BYTES_PER_TOKEN // 1024} KB/tok, '
          f'VRAM={_sim.NODE_VRAM_TOKENS:,} tok/node) | '
          f'ludo table backend: {_sim.LUDO_TABLE_BACKEND}')
    print(f'Generating figures: {", ".join(args.figs)}')

    # Base WorkloadGenerator always exists (owns pools, content-key tables).
    # When n_tenants > 1, fig1 / fig2 / fig2b / fig4 wrap it with a
    # MultiTenantWorkloadGenerator so sample() produces Requests drawn from
    # the amplified (tenant × base) pool.
    base_wgen = WorkloadGenerator(seed=args.seed)
    if args.n_tenants > 1:
        from workload_multitenant import MultiTenantWorkloadGenerator
        wgen = MultiTenantWorkloadGenerator(
            base_wgen, n_tenants=args.n_tenants, seed=args.seed)
        print(f'  amplified pool size ≈ '
              f'{args.n_tenants * 7500:,} entries')
    else:
        wgen = base_wgen
    oracle_max_rate = None
    ludo_max_rate   = None

    t0 = time.time()

    if '1' in args.figs:
        oracle_max_rate, ludo_max_rate = run_fig1(args, wgen)

    # The probe below is only needed to feed fig2/3/4 a knee rate when
    # fig1 didn't run.  `--figs m` reads its knee straight from the fig1
    # cache, so we skip the probe in that case to save ~10 s.
    needs_knee = any(d in args.figs for d in '234')

    if needs_knee and (oracle_max_rate is None or ludo_max_rate is None):
        # Fig 1 wasn't run, so we don't have the sweep to derive a knee
        # from.  Estimate per-scheme knee rates directly by probing a
        # small set of arrival rates and picking the largest rate where
        # p99 is still within 3× of the minimum p99 across the probe.
        print('\nEstimating knee rates (no full sweep available) …')
        rates_probe = [200, 500, 1000, 2000, 4000]

        def _probe_knee(scheme: str) -> float:
            pts = []
            for rate in rates_probe:
                n_q, w_q = rate_to_n_warmup(rate, args.quick)
                recs, _ = run_scheme(scheme, rate, args.workload, n_q, w_q,
                                      args.seed, wgen)
                pts.append((rate, p99([r.ttft_ms for r in recs])))
            pmin = min(t for _, t in pts)
            cap  = max(3.0 * pmin, pmin + 1.0)
            viable = [r for r, t in pts if t <= cap]
            return max(viable) if viable else pts[0][0]

        # Note: `oracle_max_rate` is a legacy variable name carried
        # through fig2/3/4 signatures; it now holds the Central Directory
        # knee (fig1's reference scheme).  Fig3b/fig4 use their own oracle
        # code paths (_simulate_fig3_oracle / _run_oracle_uniform_lb) so
        # they don't read this probe — it just sizes their rate sweeps.
        if oracle_max_rate is None:
            oracle_max_rate = _probe_knee('central_directory')
            print(f'  central directory knee rate ≈ {oracle_max_rate} req/s')
        if ludo_max_rate is None:
            ludo_max_rate = _probe_knee('ludo')
            print(f'  ludo              knee rate ≈ {ludo_max_rate} req/s')

    if '2' in args.figs:
        run_fig2(args, wgen, oracle_max_rate, ludo_max_rate)

    if '3' in args.figs:
        run_fig3(args, wgen, oracle_max_rate, ludo_max_rate)

    if '4' in args.figs:
        run_fig4(args, wgen, oracle_max_rate, ludo_max_rate)

    if '5' in args.figs:
        run_fig5(args, wgen)

    # Merged fig1+fig2+fig2b panel — pure replot from cached JSONs, no
    # simulation.  Useful for "I already have the data, just want a fresh
    # combined PDF" workflows.  When invoked alongside `1` and `2`, the
    # auto-trigger inside run_fig2 already produced the same output, so
    # this branch only adds value for `--figs m` standalone.
    if 'm' in args.figs.lower():
        fig1_results, _, ludo_max = _load_fig1_cache(args)
        if fig1_results is None:
            print(f'\n[Merged] cannot replot — fig1 cache missing '
                  f'({_fig1_cache_path(args).name}); '
                  'run `--figs 1` first.')
        elif ludo_max is None:
            print('\n[Merged] cannot replot — fig1 cache lacks ludo knee rate.')
        else:
            all_ttfts, all_e2e = _load_fig2_cache(args, ludo_max)
            if all_ttfts is None:
                print('\n[Merged] cannot replot — fig2 cache missing '
                      f'(expected for n_tenants={args.n_tenants}, '
                      f'ludo_knee≈{ludo_max}); run `--figs 2` first.')
            else:
                print('\n[Merged] fig1 + fig2 + fig2b panel (from cache) …')
                _plot_combined(fig1_results, all_ttfts, all_e2e)

    print(f'\nDone in {time.time() - t0:.1f}s.  Results in {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
