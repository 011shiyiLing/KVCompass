"""extract_lmsys.py — prefix-sharing stats for LMSYS-Chat-1M.

LMSYS carries no user geography (no country, no hashed_ip), so the sharing
spread is measured over LANGUAGES: a prefix reused from >=2 languages is a
conservative lower bound on cross-locale reuse (same-language cross-country
pairs, e.g. US/UK, are not counted).  "Distinct users" is proxied by
distinct conversations (one first-turn per conversation).

Output json: {"total_rows": N, "sharing": {K: {"vol_shared": V,
              "scatter": [[count, n_language], ...]}}}
"""
import collections
import glob
import hashlib
import json
import sys
import time

import pyarrow.parquet as pq
import tiktoken

OUT = sys.argv[1] if len(sys.argv) > 1 else "lmsys_stats.json"
enc = tiktoken.get_encoding("cl100k_base")
KS = [1024, 2048]
MINLEN_FOR_SHARE = 32

share = {k: {} for k in KS}    # key -> [count, set(language)]
total_rows = 0
t0 = time.time()
files = sorted(glob.glob("datasets/lmsys-chat-1m/data/train-*.parquet"))
for fi, f in enumerate(files):
    tbl = pq.read_table(f, columns=["conversation", "language"])
    convs = tbl.column("conversation").to_pylist()
    langs = tbl.column("language").to_pylist()
    for conv, lang in zip(convs, langs):
        total_rows += 1
        first = next((m["content"] for m in conv
                      if m.get("role") == "user" and m.get("content")), None)
        if not first:
            continue
        toks = enc.encode(first[:12000], disallowed_special=())
        if len(toks) < MINLEN_FOR_SHARE:
            continue
        for k in KS:
            if len(toks) < k:
                continue
            key = hashlib.md5(bytes(str(toks[:k]), "utf8")).hexdigest()
            e = share[k].get(key)
            if e is None:
                share[k][key] = [1, {lang} if lang else set()]
            else:
                e[0] += 1
                if lang:
                    e[1].add(lang)
    print(f"shard {fi+1}/{len(files)} rows={total_rows} t={time.time()-t0:.0f}s",
          flush=True)

out = {"total_rows": total_rows, "sharing": {}}
for k in KS:
    # shared = appears >=2 times; each occurrence is a distinct conversation
    # (one first-turn per conversation), our best proxy for distinct users.
    recs = [(cnt, len(ls)) for cnt, ls in share[k].values() if cnt >= 2]
    out["sharing"][k] = {
        "n_shared_prefixes": len(recs),
        "vol_shared": sum(r[0] for r in recs),
        "scatter": [[r[0], r[1]] for r in recs],
    }
    xlang = [r for r in recs if r[1] >= 2]
    print(f"K={k}: shared={len(recs)} vol={sum(r[0] for r in recs):,} "
          f"xlang={len(xlang)} vol_xlang={sum(r[0] for r in xlang):,}")

with open(OUT, "w") as fh:
    json.dump(out, fh)
print("wrote", OUT)
