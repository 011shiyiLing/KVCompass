"""replay_traces.py — remote-only / cross-metro reuse on production KV traces
(Qwen-Bailian, Mooncake) under a population-proportional origin model.

The traces carry real prefix-sharing structure (block hash_ids) and real
arrival times but NO geography.  Each root session is assigned a home
metro drawn from the simulator's 16-NA-metro demand distribution
(MSA-population weights, hardcoded below); follow-up turns
inherit the root's metro.  The replay then mirrors
extract_na16.py (same directory) exactly:

  remote-only : at arrival, the K-token prefix was computed before, but
                only by requests served in OTHER metros;
  local hit   : computed before in the request's own metro;
  cold        : never seen.

and the sharing stat: shared prefixes (>=2 distinct root sessions) bucketed
by the number of distinct metros of their sharers.

Primary mode = FIRST-TURN ONLY (root requests), the same filter WildChat
analysis uses (first user message per conversation), so same-session
history reuse never counts.  --all-turns adds the follow-up turns.

Randomness only enters via the metro draw, so every stat is reported as
mean ± std over --seeds independent assignments.

Bailian: chat_id / parent_chat_id give sessions; blocks are 16 tokens.
Mooncake: no session ids; a request whose hash_ids strictly extend a
previously seen request's full hash list is treated as that session's
next turn; blocks are 512 tokens (input_length / len(hash_ids)).

Usage: python replay_traces.py <trace.jsonl> [--block N] [--seeds 20]
       [--all-turns] [--out results/x.json]
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np

METROS = ['IAD', 'SJC', 'ORD', 'SEA', 'SAT', 'DSM', 'PHX', 'CYS',
          'NYC', 'LAX', 'ATL', 'MIA', 'DFW', 'DEN', 'SLC', 'PDX']
W = np.array([6.3, 7.6, 9.4, 4.0, 2.7, 0.75, 5.2, 0.10,
              19.9, 12.9, 6.4, 6.2, 8.1, 3.0, 1.3, 2.5]); W = W / W.sum()
KS = [64, 256, 512, 1024, 2048]

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--block", type=int, default=None)
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--all-turns", action="store_true")
ap.add_argument("--out", default=None)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.trace) if l.strip()]
has_sess = "chat_id" in rows[0]
if a.block is None:
    if has_sess:
        a.block = 16
    else:
        r = max(rows, key=lambda r: len(r["hash_ids"]))
        a.block = int(round(r["input_length"] / len(r["hash_ids"]) / 64) * 64) or 512
print(f"{a.trace}: {len(rows):,} requests, block={a.block} tok, sessions={'ids' if has_sess else 'prefix-extension heuristic'}")

# ── session / root resolution ──────────────────────────────────────────────
order = sorted(range(len(rows)), key=lambda i: rows[i]["timestamp"])   # stable
root_of = [None] * len(rows)          # index of root request per request
if has_sess:
    by_id = {r["chat_id"]: i for i, r in enumerate(rows)}
    for i in order:
        r = rows[i]; p = r["parent_chat_id"]
        if p == -1 or p not in by_id:
            root_of[i] = i
        else:
            root_of[i] = root_of[by_id[p]] if root_of[by_id[p]] is not None else i
else:
    full = {}                           # tuple(hash_ids) -> request idx (latest)
    for i in order:
        h = tuple(rows[i]["hash_ids"]); parent = None
        for cut in range(len(h) - 1, 0, -1):     # longest strict proper prefix seen as a FULL request
            j = full.get(h[:cut])
            if j is not None:
                parent = j; break
        root_of[i] = i if parent is None else root_of[parent]
        full[h] = i
roots = sorted({root_of[i] for i in range(len(rows))})
is_root = [root_of[i] == i for i in range(len(rows))]
sel = [i for i in order if (a.all_turns or is_root[i])]
print(f"  root sessions={len(roots):,}  requests replayed={len(sel):,}")

# prefix keys per K (block-aligned): K tokens = K/block blocks
keys = {k: {} for k in KS}
for i in sel:
    h = rows[i]["hash_ids"]
    for k in KS:
        nb = k // a.block
        if nb >= 1 and len(h) >= nb:
            keys[k][i] = hash(tuple(h[:nb]))

# ── geography-free upper bound: cross-SESSION hit fraction (any region) ───
geo_free = {}
for k in KS:
    seen = set(); hit = 0; n = 0
    for i in sel:
        kk = keys[k].get(i)
        if kk is None: continue
        n += 1
        if kk in seen: hit += 1
        seen.add(kk)
    geo_free[k] = {"n_req": n, "hits": hit, "pct_hit": 100.0 * hit / n if n else 0.0}

# ── replay under N random metro assignments ───────────────────────────────
per_seed = {k: [] for k in KS}
for s in range(a.seeds):
    rng = np.random.default_rng(1000 + s)
    root_reg = {ri: int(rng.choice(16, p=W)) for ri in roots}
    reg = [root_reg[root_of[i]] for i in range(len(rows))]
    for k in KS:
        seen = {}; cold = local = remote = 0
        pr = collections.defaultdict(lambda: [0, 0, 0])
        share = {}                       # key -> [count, set(root), set(metro)]
        for i in sel:
            kk = keys[k].get(i)
            if kk is None: continue
            r = reg[i]; b = 1 << r; m = seen.get(kk, 0)
            if m == 0: cold += 1; pr[r][0] += 1
            elif m & b: local += 1; pr[r][1] += 1
            else: remote += 1; pr[r][2] += 1
            seen[kk] = m | b
            e = share.get(kk)
            if e is None: share[kk] = [1, {root_of[i]}, {r}]
            else: e[0] += 1; e[1].add(root_of[i]); e[2].add(r)
        shared = [(c, len(rs), len(ms)) for c, rs, ms in share.values() if len(rs) >= 2]
        vol = sum(c for c, _, _ in shared)
        bucket = collections.Counter()
        for c, _, nm in shared: bucket[min(nm, 4)] += c
        hits = local + remote
        per_seed[k].append({
            "cold": cold, "local": local, "remote_only": remote,
            "pct_remote_only_of_req": 100.0 * remote / (cold + hits) if hits else 0.0,
            "pct_remote_only_of_hits": 100.0 * remote / hits if hits else 0.0,
            "per_region_pct_remote_of_hits": {
                METROS[r]: (100.0 * v[2] / (v[1] + v[2]) if v[1] + v[2] else None)
                for r, v in pr.items()},
            "n_shared_prefixes": len(shared), "vol_shared": vol,
            "pct_vol_by_nmetro": {str(b): 100.0 * bucket[b] / vol if vol else 0.0
                                  for b in (1, 2, 3, 4)},
            "pct_vol_xmetro": 100.0 * (vol - bucket[1]) / vol if vol else 0.0,
        })

def ms(vals):
    v = np.array([x for x in vals if x is not None], float)
    return [float(v.mean()), float(v.std())] if len(v) else None

summary = {}
for k in KS:
    ps = per_seed[k]
    summary[k] = {
        "n_req": geo_free[k]["n_req"],
        "geo_free_pct_cross_session_hit": geo_free[k]["pct_hit"],
        "pct_remote_only_of_req": ms([p["pct_remote_only_of_req"] for p in ps]),
        "pct_remote_only_of_hits": ms([p["pct_remote_only_of_hits"] for p in ps]),
        "per_region_pct_remote_of_hits": {
            m: ms([p["per_region_pct_remote_of_hits"].get(m) for p in ps]) for m in METROS},
        "n_shared_prefixes": ms([p["n_shared_prefixes"] for p in ps]),
        "vol_shared": ms([p["vol_shared"] for p in ps]),
        "pct_vol_by_nmetro": {b: ms([p["pct_vol_by_nmetro"][b] for p in ps]) for b in "1234"},
        "pct_vol_xmetro": ms([p["pct_vol_xmetro"] for p in ps]),
    }
    g = summary[k]
    print(f"  K={k:5d} n={g['n_req']:>8,}  cross-session hit {g['geo_free_pct_cross_session_hit']:5.1f}% | "
          f"remote-only {g['pct_remote_only_of_req'][0]:5.1f}±{g['pct_remote_only_of_req'][1]:.1f}% of req, "
          f"{g['pct_remote_only_of_hits'][0]:5.1f}±{g['pct_remote_only_of_hits'][1]:.1f}% of hits | "
          f"shared vol {g['vol_shared'][0]:,.0f}, x-metro {g['pct_vol_xmetro'][0]:5.1f}±{g['pct_vol_xmetro'][1]:.1f}%")

out = {"trace": a.trace, "block_tokens": a.block, "n_requests": len(rows),
       "n_root_sessions": len(roots), "mode": "all_turns" if a.all_turns else "first_turn_only",
       "seeds": a.seeds, "metros": METROS, "demand_weights": W.tolist(),
       "analytic_1_minus_sum_p2": float(1 - (W ** 2).sum()),
       "ks": KS, "summary": summary, "per_seed": per_seed}
outp = a.out or f"results/{Path(a.trace).stem}{'_allturns' if a.all_turns else ''}.json"
json.dump(out, open(outp, "w"), indent=1)
print("  wrote", outp)
