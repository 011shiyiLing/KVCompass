"""extract_na16.py — WildChat motivation analysis for the 16-NA-metro setting.

One pass over WildChat-1M producing BOTH motivation datasets with regions
redefined as the simulator's 16 North-American deployment metros
(simulate.py REGION_NAMES), using WildChat's per-conversation US *state*
field (GeoIP2) instead of the country field:

  wc_remote_only_na16.json — temporal cold / local-hit / remote-only
      classification per K (mirrors extract_remote_only.py exactly);
  wc_stats_na16.json       — shared-prefix sharing stats per K (mirrors
      extract.py's "sharing" block; countries → US states).

Scope: conversations with country == 'United States' and a mappable
state (the NA deployment's domestic demand — matching the simulator's
MSA-population demand model).  Non-US traffic is out of scope and only
counted for coverage reporting.

State→metro assignment: each state maps to the deployment metro nearest
its official 2020 Census population centroid (Centers of Population,
CenPop2020_Mean_ST.txt, census.gov; great-circle distance to the metro
city coordinates below) — instantiating the standard latency-based /
nearest-site routing rule (cf. AWS Route 53 latency-based routing,
Azure Traffic Manager 'performance' mode).

Sources (all verified 2026-08-10):
  Census Centers of Population 2020 (product page):
    https://www.census.gov/geographies/reference-files/time-series/geo/centers-population.html
  State centroid data file used here:
    https://www2.census.gov/geo/docs/reference/cenpop2020/CenPop2020_Mean_ST.txt
  Nearest-site routing rule:
    https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-policy-latency.html
    https://learn.microsoft.com/en-us/azure/traffic-manager/traffic-manager-routing-methods#performance
  WildChat-1M (state field per dataset card):
    https://huggingface.co/datasets/allenai/WildChat-1M
    https://arxiv.org/abs/2405.01470
  Tokenizer: https://github.com/openai/tiktoken (cl100k_base)  Granularity note: GeoIP2
resolves to states, so a state's whole traffic goes to one metro —
California (centroid → LAX) folds the SJC user base into LAX; the
cross-metro counts are therefore a conservative LOWER bound.

Usage: python extract_na16.py   (from repo root; needs
       datasets/wildchat-1m/data/*.parquet)
"""
import collections
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

HERE = Path(__file__).parent
OUT_REMOTE = HERE / "wc_remote_only_na16.json"
OUT_STATS = HERE / "wc_stats_na16.json"

enc = tiktoken.get_encoding("cl100k_base")
KS = [64, 256, 512, 1024, 2048]
MINLEN = 32

METROS = ['IAD', 'SJC', 'ORD', 'SEA', 'SAT', 'DSM', 'PHX', 'CYS',
          'NYC', 'LAX', 'ATL', 'MIA', 'DFW', 'DEN', 'SLC', 'PDX']
MID = {m: i for i, m in enumerate(METROS)}

# Metro city coordinates (deployment-site anchors, deg lat/lon).
METRO_XY = {
    'IAD': (39.04, -77.49), 'SJC': (37.34, -121.89), 'ORD': (41.88, -87.63),
    'SEA': (47.61, -122.33), 'SAT': (29.42, -98.49), 'DSM': (41.59, -93.62),
    'PHX': (33.45, -112.07), 'CYS': (41.14, -104.82), 'NYC': (40.71, -74.01),
    'LAX': (34.05, -118.24), 'ATL': (33.75, -84.39), 'MIA': (25.76, -80.19),
    'DFW': (32.78, -96.80), 'DEN': (39.74, -104.99), 'SLC': (40.76, -111.89),
    'PDX': (45.52, -122.68),
}

# Each US state/territory → deployment metro nearest its 2020 Census
# population centroid (CenPop2020_Mean_ST.txt, great-circle to METRO_XY;
# derivation script in this file's git history / regen instructions).
STATE_TO_METRO = {
    # ATL
    'Alabama': 'ATL', 'Georgia': 'ATL', 'Kentucky': 'ATL',
    'Mississippi': 'ATL', 'South Carolina': 'ATL', 'Tennessee': 'ATL',
    # CYS
    'Wyoming': 'CYS',
    # DEN
    'Colorado': 'DEN',
    # DFW
    'Arkansas': 'DFW', 'Louisiana': 'DFW', 'Oklahoma': 'DFW',
    # DSM
    'Iowa': 'DSM', 'Kansas': 'DSM', 'Minnesota': 'DSM', 'Missouri': 'DSM',
    'Nebraska': 'DSM', 'North Dakota': 'DSM', 'South Dakota': 'DSM',
    # IAD
    'Delaware': 'IAD', 'District of Columbia': 'IAD',
    'Washington, D.C.': 'IAD', 'Maryland': 'IAD', 'North Carolina': 'IAD',
    'Pennsylvania': 'IAD', 'Virginia': 'IAD', 'West Virginia': 'IAD',
    # LAX
    'California': 'LAX', 'Nevada': 'LAX',
    # MIA
    'Florida': 'MIA', 'Puerto Rico': 'MIA', 'U.S. Virgin Islands': 'MIA',
    # NYC
    'Connecticut': 'NYC', 'Maine': 'NYC', 'Massachusetts': 'NYC',
    'New Hampshire': 'NYC', 'New Jersey': 'NYC', 'New York': 'NYC',
    'Rhode Island': 'NYC', 'Vermont': 'NYC',
    # ORD
    'Illinois': 'ORD', 'Indiana': 'ORD', 'Michigan': 'ORD', 'Ohio': 'ORD',
    'Wisconsin': 'ORD',
    # PDX
    'Oregon': 'PDX',
    # PHX
    'Arizona': 'PHX', 'New Mexico': 'PHX',
    # SAT
    'Texas': 'SAT',
    # SEA
    'Alaska': 'SEA', 'Washington': 'SEA',
    # SJC
    'Hawaii': 'SJC',
    # SLC
    'Idaho': 'SLC', 'Montana': 'SLC', 'Utah': 'SLC',
}

# ── accumulators ────────────────────────────────────────────────────────────
# remote-only replay lists (per K)
ts_l = {k: [] for k in KS}
key_l = {k: [] for k in KS}
reg_l = {k: [] for k in KS}
# sharing accumulators (per K): key -> [count, set(state), set(ip), set(metro)]
share = {k: {} for k in KS}

n_total = n_us = n_us_state = 0
state_ct = collections.Counter()
metro_ct = collections.Counter()
users = set()

files = sorted(glob.glob("datasets/wildchat-1m/data/*.parquet"))
assert files, "run from the repo root (datasets/wildchat-1m/data missing)"
t0 = time.time()
for fi, f in enumerate(files):
    tbl = pq.read_table(
        f, columns=["conversation", "country", "state", "hashed_ip",
                    "timestamp"])
    convs = tbl.column("conversation").to_pylist()
    countries = tbl.column("country").to_pylist()
    states = tbl.column("state").to_pylist()
    ips = tbl.column("hashed_ip").to_pylist()
    stamps = tbl.column("timestamp").to_pylist()
    for conv, c, st, ip, ts in zip(convs, countries, states, ips, stamps):
        n_total += 1
        if c != 'United States':
            continue
        n_us += 1
        metro = STATE_TO_METRO.get(st)
        if metro is None or ts is None:
            continue
        n_us_state += 1
        state_ct[st] += 1
        metro_ct[metro] += 1
        if ip:
            users.add(ip)
        first = next((m["content"] for m in conv
                      if m.get("role") == "user" and m.get("content")), None)
        if not first:
            continue
        toks = enc.encode(first[:12000], disallowed_special=())
        if len(toks) < MINLEN:
            continue
        rid = MID[metro]
        epoch = ts.timestamp()
        for k in KS:
            if len(toks) < k:
                continue
            hd = hashlib.md5(bytes(str(toks[:k]), "utf8")).digest()
            key64 = int.from_bytes(hd[:8], "little")
            key_l[k].append(key64)
            ts_l[k].append(epoch)
            reg_l[k].append(rid)
            e = share[k].get(key64)
            if e is None:
                share[k][key64] = [1, {st}, {ip} if ip else set(), {rid}]
            else:
                e[0] += 1
                e[1].add(st)
                if ip:
                    e[2].add(ip)
                e[3].add(rid)
    print(f"shard {fi+1}/{len(files)} done, total={n_total} us={n_us} "
          f"mapped={n_us_state}, t={time.time()-t0:.0f}s", flush=True)

cov = {
    "total_rows": n_total, "us_rows": n_us, "us_state_mapped": n_us_state,
    "n_us_users": len(users),
    "metros": METROS,
    "state_to_metro": STATE_TO_METRO,
    "metro_counts": dict(metro_ct.most_common()),
    "state_counts": dict(state_ct.most_common()),
    "granularity_note": ("state-level GeoIP2: California's centroid maps "
                          "the whole state to LAX, so SJC receives only "
                          "Hawaii — cross-metro counts are a lower bound"),
}

# ── remote-only replay (mirrors extract_remote_only.py) ─────────────────────
out_r = {"coverage": cov, "ks": KS, "classes": {}}
for k in KS:
    ts = np.array(ts_l[k])
    key = np.array(key_l[k], dtype=np.uint64)
    reg = np.array(reg_l[k], dtype=np.int8)
    order = np.argsort(ts, kind="stable")
    seen: dict = {}
    cold = local = remote_only = 0
    remote_pairs = collections.Counter()
    per_region = collections.defaultdict(lambda: [0, 0, 0])
    for i in order:
        r = int(reg[i])
        kk = int(key[i])
        b = 1 << r
        mask = seen.get(kk, 0)
        if mask == 0:
            cold += 1
            per_region[r][0] += 1
        elif mask & b:
            local += 1
            per_region[r][1] += 1
        else:
            remote_only += 1
            per_region[r][2] += 1
            remote_pairs[(r, bin(mask).count('1'))] += 1
        seen[kk] = mask | b
    n = len(order)
    hits = local + remote_only
    out_r["classes"][k] = {
        "n_req": n, "cold": cold, "local_hit": local,
        "remote_only": remote_only,
        "pct_cold": 100.0 * cold / n if n else 0.0,
        "pct_local": 100.0 * local / n if n else 0.0,
        "pct_remote_only": 100.0 * remote_only / n if n else 0.0,
        "pct_remote_only_of_hits":
            (100.0 * remote_only / hits) if hits else 0.0,
        "remote_pairs": {f"{r}|{m}": v for (r, m), v in
                         sorted(remote_pairs.items())},
        "per_region": {METROS[r]: v for r, v in sorted(per_region.items())},
    }
    print(f"K={k}: n={n:,} cold={cold:,} local={local:,} "
          f"remote_only={remote_only:,} "
          f"({out_r['classes'][k]['pct_remote_only']:.1f}% of req, "
          f"{out_r['classes'][k]['pct_remote_only_of_hits']:.1f}% of hits)",
          flush=True)

with open(OUT_REMOTE, "w") as fh:
    json.dump(out_r, fh, indent=1)
print("WROTE", OUT_REMOTE)

# ── sharing stats (mirrors extract.py's sharing block) ──────────────────────
out_s = {"coverage": cov, "sharing": {}}
for k in KS:
    recs = [(cnt, len(sts), len(ips_), len(regs))
            for cnt, sts, ips_, regs in share[k].values()]
    total_req_with_prefix = sum(r[0] for r in recs)
    shared = [r for r in recs if r[0] >= 2 and r[2] >= 2]
    vol_shared = sum(r[0] for r in shared)
    xstate = [r for r in shared if r[1] >= 2]
    xmetro = [r for r in shared if r[3] >= 2]
    out_s["sharing"][k] = {
        "n_prefixes_with_minlen": len(recs),
        "vol_total_req": total_req_with_prefix,
        "n_shared_prefixes": len(shared),
        "n_xstate_prefixes": len(xstate),
        "n_xmetro_prefixes": len(xmetro),
        "vol_shared": vol_shared,
        "vol_xstate": sum(r[0] for r in xstate),
        "vol_xmetro": sum(r[0] for r in xmetro),
        # (count, n_state, n_users, n_metro_region)
        "scatter": [[r[0], r[1], r[2], r[3]] for r in shared],
    }
    pct = (100.0 * out_s["sharing"][k]["vol_xmetro"] / vol_shared) \
        if vol_shared else 0.0
    print(f"K={k}: shared={len(shared):,} vol_shared={vol_shared:,} "
          f"xmetro_vol={out_s['sharing'][k]['vol_xmetro']:,} ({pct:.1f}%)",
          flush=True)

with open(OUT_STATS, "w") as fh:
    json.dump(out_s, fh)
print("WROTE", OUT_STATS)
