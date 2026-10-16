"""evict_motivation.py — UniCache-Fig.1/Table-1-style eviction analysis.

Single-cache trace replay (no routing): each workload's Zipf trace is fed
through ONE prefix cache with the simulator's per-node budget
(NODE_VRAM_TOKENS), under four eviction policies:

  LRU  — evict least-recently-used (vLLM/SGLang default; our sim's Node)
  LFU  — evict lowest access count (tie: oldest)
  FIFO — evict oldest insertion; hits do not refresh
  WA   — Workload-Aware (Wang et al., ATC'25, as reproduced by UniCache):
         entries are partitioned by task category with an independent LRU
         per category; eviction takes the LRU victim of the *inserting*
         request's category (partition sizes converge to insert-volume
         shares).  Single-task workloads have one category, so WA
         degenerates to LRU — same statement as UniCache's.

Cache structure mirrors the simulator's Node: multi-level content hashes at
CHUNK_SIZES, contiguous-prefix hits, per-level token accounting.  Metric is
token-weighted hit ratio Σ matched / Σ prompt tokens (the flops_saved
convention; matches are capped by the 4096-token top level).

Workloads: Mixed (ToolBench+LooGLE+AAPS, 50/5/45), WildChat (375 K per-turn
entries), LMSYS (375 K per-turn entries).  Traces: 200 K Zipf(1.1) draws.

Outputs: figures/motivation_evict.pdf/png + printed & LaTeX Table.
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets'
FIGD = ROOT / 'results'
FIGD.mkdir(exist_ok=True)

# Policy display names (shared by the figure and the printed tables).
PL = {'lru': 'LRU', 'lfu': 'LFU', 'fifo': 'FIFO', 'wa': 'WA'}

CHUNK_SIZES = [128, 256, 512, 1024, 2048, 4096]
BUDGET      = (32 * 1024 ** 3) // (2 * 32 * 8 * 128 * 2)   # NODE_VRAM_TOKENS = 262144
N_REQ       = 200_000
ZIPF_A      = 1.1
SEED        = 42

# ─── Pools ───────────────────────────────────────────────────────────────────

def _load_jsonl(path, limit=None):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _tokens(entry):
    if 'input_ids' in entry:
        return list(entry['input_ids'])
    text = entry['workload']
    return [hash(text[i:i + 4]) & 0xFFFFFFFF for i in range(0, len(text), 4)]

class Pool:
    """Lazy content-key pool: keys computed only for sampled entries."""
    def __init__(self, entries, category):
        self.entries  = entries
        self.category = category
        self._keys    = {}
        w = np.arange(1, len(entries) + 1, dtype=float) ** (-ZIPF_A)
        self.cum = np.cumsum(w / w.sum())
        # Session structure (per-turn datasets): conv id -> entry indices in
        # turn order (prep scripts write turns consecutively).
        self.sessions = None
        if entries and 'conv' in entries[0]:
            sess, order = {}, []
            for i, e in enumerate(entries):
                c = e['conv']
                if c not in sess:
                    sess[c] = []
                    order.append(c)
                sess[c].append(i)
            self.sessions = [sess[c] for c in order]

    def draw(self, rng):
        return int(np.searchsorted(self.cum, rng.random(), side='right'))

    def keys_len(self, idx):
        got = self._keys.get(idx)
        if got is None:
            toks = _tokens(self.entries[idx])
            n = len(toks)
            got = ({cs: hash(tuple(toks[:cs])) & 0xFFFFFFFFFFFFFFFF
                    for cs in CHUNK_SIZES if cs <= n}, max(1, n))
            self._keys[idx] = got
        return got

def build_workloads():
    tb  = Pool(_load_jsonl(DATA / 'toolbench_zipf_1.1_prompts_6000.jsonl'), 'ToolBench')
    lg  = Pool(_load_jsonl(DATA / 'loogle.jsonl'), 'LooGLE')
    ap  = Pool(_load_jsonl(DATA / 'aaps.jsonl'), 'AAPS')
    wc  = Pool(_load_jsonl(DATA / 'wildchat.jsonl'), 'chat')
    lm  = Pool(_load_jsonl(DATA / 'lmsys.jsonl'), 'chat')
    return {'Mixed': ('mix', [tb, lg, ap]), 'WildChat': ('one', [wc]),
            'LMSYS': ('one', [lm])}

K_SESSIONS = 256      # concurrent active sessions for chat-session replay

def make_trace(kind, pools, rng):
    """Yield (content_keys, prefix_len, category) N_REQ times.

    'mix'  — Zipf(1.1) resampling per component (single-turn / template
             workloads: popular prompts genuinely recur — UniCache's
             MASH-QA-style replay).
    'one'  — SESSION replay (UniCache Algorithm-3-style): K concurrent
             active conversations; each step advances a random active
             session by one turn; finished sessions leave and are replaced.
             Each conversation occurs once, so within-session recency — not
             stationary popularity — is the reuse signal, and completed
             sessions genuinely die (the pattern that punishes LFU).
    """
    if kind == 'mix':
        for _ in range(N_REQ):
            r = rng.random()
            p = pools[0] if r < 0.50 else (pools[1] if r < 0.55 else pools[2])
            idx = p.draw(rng)
            ck, n = p.keys_len(idx)
            yield ck, n, p.category
        return

    p = pools[0]
    order = rng.permutation(len(p.sessions))
    nxt = 0
    active = []           # list of (session_indices, cursor)
    emitted = 0
    while emitted < N_REQ:
        while len(active) < K_SESSIONS and nxt < len(order):
            active.append([p.sessions[order[nxt]], 0])
            nxt += 1
        if not active:
            break
        j = int(rng.integers(0, len(active)))
        sess = active[j]
        idx = sess[0][sess[1]]
        sess[1] += 1
        if sess[1] >= len(sess[0]):
            active[j] = active[-1]
            active.pop()
        ck, n = p.keys_len(idx)
        emitted += 1
        yield ck, n, p.category

# ─── Eviction policies over the chunk-level cache ────────────────────────────

class Cache:
    """Multi-level contiguous-prefix cache with pluggable eviction."""
    def __init__(self, policy):
        self.policy = policy
        self.levels = [dict() for _ in CHUNK_SIZES]   # key -> tokens
        self.tokens = 0
        self.seq    = 0
        # policy structures
        self.od   = OrderedDict()          # LRU / FIFO order over (l, key)
        self.freq = {}                     # LFU counts
        self.heap = []                     # LFU lazy heap
        self.cats = {}                     # WA: cat -> OrderedDict
        self.catof = {}                    # (l,key) -> cat

    # -- policy hooks ---------------------------------------------------------
    def _touch(self, lk):
        if self.policy in ('lru',):
            self.od.move_to_end(lk)
        elif self.policy == 'wa':
            self.cats[self.catof[lk]].move_to_end(lk)
        elif self.policy == 'lfu':
            self.freq[lk] += 1
            import heapq
            heapq.heappush(self.heap, (self.freq[lk], self.seq, lk))

    def _add(self, lk, cat):
        if self.policy in ('lru', 'fifo'):
            self.od[lk] = None
        elif self.policy == 'wa':
            self.cats.setdefault(cat, OrderedDict())[lk] = None
            self.catof[lk] = cat
        elif self.policy == 'lfu':
            self.freq[lk] = 1
            import heapq
            heapq.heappush(self.heap, (1, self.seq, lk))

    def _victim(self, cat):
        if self.policy in ('lru', 'fifo'):
            return next(iter(self.od))
        if self.policy == 'wa':
            d = self.cats.get(cat)
            if not d:                          # own class empty → oldest class head
                d = max((c for c in self.cats.values() if c), key=len)
            return next(iter(d))
        # lfu: lazy heap — pop until a live, up-to-date entry surfaces
        import heapq
        while True:
            f, _s, lk = self.heap[0]
            if lk in self.freq and self.freq[lk] == f:
                return lk
            heapq.heappop(self.heap)

    def _remove(self, lk):
        l, key = lk
        self.tokens -= self.levels[l].pop(key)
        if self.policy in ('lru', 'fifo'):
            self.od.pop(lk, None)
        elif self.policy == 'wa':
            self.cats[self.catof.pop(lk)].pop(lk, None)
        elif self.policy == 'lfu':
            self.freq.pop(lk, None)

    # -- serve one request ----------------------------------------------------
    def serve(self, ck, plen, cat):
        self.seq += 1
        match = 0
        for l, cs in enumerate(CHUNK_SIZES):
            key = ck.get(cs)
            if key is None or key not in self.levels[l]:
                break
            match = cs
            self._touch((l, key))
        # insert missing levels (contiguous by construction of ck)
        for l, cs in enumerate(CHUNK_SIZES):
            key = ck.get(cs)
            if key is None:
                break
            if key in self.levels[l]:
                continue
            while self.tokens + cs > BUDGET:
                self._remove(self._victim(cat))
            self.levels[l][key] = cs
            self.tokens += cs
            self._add((l, key), cat)
        return min(match, plen)

# ─── Run the 3 × 4 grid ──────────────────────────────────────────────────────

def plot_figure(results, wnames):
    """Grouped bars, muted palette matching motivation_geo_single, one hatch
    per policy.  Sized for one column of the ACM two-column template
    (~3.35 in) so it is included at \\columnwidth without downscaling."""
    FONT = 'STIXGeneral'
    matplotlib.rcParams.update({'font.family': FONT,
                                'axes.spines.top': False,
                                'axes.spines.right': False,
                                'hatch.linewidth': 0.6})
    policies = ['lru', 'lfu', 'fifo', 'wa']
    PC = {'lru': '#7FA3C4', 'lfu': '#C17878', 'fifo': '#90A88B',
          'wa': '#A392C5'}
    PE = {'lru': '#5A7E9E', 'lfu': '#9E5A5A', 'fifo': '#6E8A69',
          'wa': '#7E6FA0'}
    PH = {'lru': '//', 'lfu': '--', 'fifo': '\\\\', 'wa': 'xx'}
    x = np.arange(len(wnames))
    bw = 0.19
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    for i, pol in enumerate(policies):
        ys = [results[(w, pol)] for w in wnames]
        xs = x + (i - 1.5) * bw
        ax.bar(xs, ys, bw, facecolor=PC[pol], edgecolor=PE[pol],
               linewidth=0.8, hatch=PH[pol], label=PL[pol])
        for xx, yy in zip(xs, ys):
            ax.text(xx, yy + 0.008, f'{yy:.2f}', ha='center', va='bottom',
                    fontsize=7.5, fontname=FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(wnames, fontsize=12, fontname=FONT)
    ax.set_ylabel('Cache hit ratio', fontsize=12, fontname=FONT)
    ax.set_ylim(0, 0.62)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', length=0)
    # Legend inside the axes, sitting at the 0.6 gridline height.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4,
              frameon=False, prop={'family': FONT, 'size': 10.5},
              handlelength=1.2, columnspacing=0.9, handletextpad=0.35)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.11)
    fig.savefig(FIGD / 'motivation_evict.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(FIGD / 'motivation_evict.png', dpi=300, bbox_inches='tight')
    print(f'saved → {FIGD / "motivation_evict.pdf"}')


def main():
    workloads = build_workloads()
    policies  = ['lru', 'lfu', 'fifo', 'wa']
    results   = {}          # (workload, policy) -> total hit ratio
    breakdown = {}          # policy -> {cat: ratio}   (Mixed only)

    for wname, (kind, pools) in workloads.items():
        for pol in policies:
            rng   = np.random.default_rng(SEED)
            cache = Cache(pol)
            hit_t = tot_t = 0
            cat_h, cat_t = {}, {}
            for ck, plen, cat in make_trace(kind, pools, rng):
                m = cache.serve(ck, plen, cat)
                hit_t += m
                tot_t += plen
                cat_h[cat] = cat_h.get(cat, 0) + m
                cat_t[cat] = cat_t.get(cat, 0) + plen
            ratio = hit_t / tot_t
            results[(wname, pol)] = ratio
            if wname == 'Mixed':
                breakdown[pol] = {c: cat_h[c] / cat_t[c] for c in cat_h}
            print(f'  {wname:<9} {pol.upper():<5} hit={ratio:.4f}', flush=True)

    with open(FIGD / 'motivation_evict_results.json', 'w') as f:
        json.dump({f'{w}|{p}': v for (w, p), v in results.items()}, f)
    plot_figure(results, list(workloads.keys()))

    # ── Table (UniCache Table 1 style: Mixed breakdown) ──────────────────────
    cats = ['ToolBench', 'LooGLE', 'AAPS']
    print('\nTable: hit-ratio breakdown of the Mixed workload')
    print(f'{"Policy":<8}{"Total":>8}' + ''.join(f'{c:>11}' for c in cats))
    for pol in policies:
        row = f'{PL[pol]:<8}{results[("Mixed", pol)]:>8.4f}'
        row += ''.join(f'{breakdown[pol].get(c, 0):>11.4f}' for c in cats)
        print(row)
    print('\nLaTeX:')
    print('\\begin{tabular}{lcccc}\\toprule')
    print('Eviction Policy & Total & ToolBench & LooGLE & AAPS \\\\ \\midrule')
    for pol in policies:
        cells = [results[('Mixed', pol)]] + [breakdown[pol].get(c, 0) for c in cats]
        print(PL[pol] + ' & ' + ' & '.join(f'{v:.4f}' for v in cells) + ' \\\\')
    print('\\bottomrule\\end{tabular}')


if __name__ == '__main__':
    main()
