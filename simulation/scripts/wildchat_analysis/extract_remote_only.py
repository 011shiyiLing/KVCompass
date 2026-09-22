"""extract_remote_only.py — temporal local-miss / remote-hit analysis.

For every WildChat request (first user turn of each conversation), replayed
in timestamp order, classify its K-token prefix at the moment it arrives:

  cold         nobody anywhere has computed this prefix before
  local hit    a worker in the SAME serving region computed it before
  remote-only  the prefix HAS been computed before, but only by workers in
               OTHER serving regions — a per-node or per-region index would
               recompute it from scratch; only a cross-region discovery
               plane can reuse it

Serving region = nearest of the 8 deployment sites (US/CN/DE/JP/SG/GB/KR/AU),
same mapping as the simulator's workload (datasets/prep_wildchat).

Output: wc_remote_only.json with per-K counts + percentages.

Usage: python3.13 scripts/wildchat_analysis/extract_remote_only.py [out.json]
       (run from the repo root; needs datasets/wildchat-1m/data/*.parquet)
"""
import collections
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'datasets'))
from prep_wildchat import country_to_region

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    str(Path(__file__).parent / "wc_remote_only.json")
enc = tiktoken.get_encoding("cl100k_base")
KS = [64, 256, 512, 1024, 2048]
MINLEN = 32                      # same trivial-prefix floor as extract.py

# per K: parallel lists (ts_epoch, key64, region_id)
ts_l = {k: [] for k in KS}
key_l = {k: [] for k in KS}
reg_l = {k: [] for k in KS}

files = sorted(glob.glob("datasets/wildchat-1m/data/*.parquet"))
assert files, "run from the repo root (datasets/wildchat-1m/data missing)"
t0 = time.time()
total_rows = 0
for fi, f in enumerate(files):
    tbl = pq.read_table(f, columns=["conversation", "country", "timestamp"])
    convs = tbl.column("conversation").to_pylist()
    countries = tbl.column("country").to_pylist()
    stamps = tbl.column("timestamp").to_pylist()
    for conv, c, ts in zip(convs, countries, stamps):
        total_rows += 1
        first = next((m["content"] for m in conv
                      if m.get("role") == "user" and m.get("content")), None)
        if not first or ts is None:
            continue
        toks = enc.encode(first[:12000], disallowed_special=())
        if len(toks) < MINLEN:
            continue
        rid = country_to_region(c)
        epoch = ts.timestamp()
        for k in KS:
            if len(toks) < k:
                continue
            h = hashlib.md5(bytes(str(toks[:k]), "utf8")).digest()
            key_l[k].append(int.from_bytes(h[:8], "little"))
            ts_l[k].append(epoch)
            reg_l[k].append(rid)
    print(f"shard {fi+1}/{len(files)} done, rows={total_rows}, "
          f"t={time.time()-t0:.0f}s", flush=True)

out = {"total_rows": total_rows, "ks": KS, "classes": {}}
for k in KS:
    ts = np.array(ts_l[k]);  key = np.array(key_l[k], dtype=np.uint64)
    reg = np.array(reg_l[k], dtype=np.int8)
    order = np.argsort(ts, kind="stable")     # replay in arrival order
    seen: dict = {}                            # key -> 8-bit region mask
    cold = local = remote_only = 0
    remote_pairs = collections.Counter()       # (my_region, #prior regions)
    # per serving region: [cold, local, remote_only]
    per_region = collections.defaultdict(lambda: [0, 0, 0])
    for i in order:
        r = int(reg[i])
        kk = int(key[i]); b = 1 << r
        mask = seen.get(kk, 0)
        if mask == 0:
            cold += 1;        per_region[r][0] += 1
        elif mask & b:
            local += 1;       per_region[r][1] += 1
        else:
            remote_only += 1; per_region[r][2] += 1
            remote_pairs[(r, bin(mask).count('1'))] += 1
        seen[kk] = mask | b
    n = len(order); hits = local + remote_only
    out["classes"][k] = {
        "n_req": n, "cold": cold, "local_hit": local,
        "remote_only": remote_only,
        "pct_cold": 100.0 * cold / n,
        "pct_local": 100.0 * local / n,
        "pct_remote_only": 100.0 * remote_only / n,
        "pct_remote_only_of_hits": (100.0 * remote_only / hits) if hits else 0.0,
        "remote_pairs": {f"{r}|{m}": v for (r, m), v in
                         sorted(remote_pairs.items())},
        "per_region": {str(r): v for r, v in sorted(per_region.items())},
    }
    print(f"K={k}: n={n:,} cold={cold:,} local={local:,} "
          f"remote_only={remote_only:,} "
          f"({out['classes'][k]['pct_remote_only']:.1f}% of req, "
          f"{out['classes'][k]['pct_remote_only_of_hits']:.1f}% of hits)",
          flush=True)

with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print("WROTE", OUT)
