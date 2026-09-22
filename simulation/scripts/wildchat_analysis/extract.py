import glob, json, collections, hashlib, sys, time
import pyarrow.parquet as pq
import tiktoken
import pycountry_convert as pcc
from pathlib import Path

# Serving-region mapping: each country goes to its nearest of the 8
# deployment regions (US/CN/DE/JP/SG/GB/KR/AU), the same mapping the
# simulator's wildchat workload uses.  A "region" is one RDMA-connected
# deployment site, NOT a continent.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'datasets'))
from prep_wildchat import country_to_region

OUT = sys.argv[1] if len(sys.argv) > 1 else "wc_stats.json"
enc = tiktoken.get_encoding("cl100k_base")
KS = [64, 256, 512, 1024, 2048]   # token-prefix thresholds (1024/2048 align with the sim's chunk grid / cross-region break-even)
MINLEN_FOR_SHARE = 32        # ignore trivial prefixes (greetings) for sharing stats

# ---- country -> continent ----
CONT = {"AF":"Africa","AS":"Asia","EU":"Europe","NA":"N.America",
        "SA":"S.America","OC":"Oceania","AN":"Antarctica"}
FALLBACK = {  # names pycountry_convert may miss / WildChat spellings
 "Russia":"EU","South Korea":"AS","North Korea":"AS","Hong Kong":"AS","Taiwan":"AS",
 "Vietnam":"AS","Iran":"AS","Syria":"AS","Laos":"AS","Macao":"AS","Macau":"AS",
 "Brunei":"AS","Palestine":"AS","Czechia":"EU","Moldova":"EU","Kosovo":"EU",
 "Bolivia":"SA","Venezuela":"SA","Tanzania":"AF","Congo - Kinshasa":"AF",
 "Congo - Brazzaville":"AF","Côte d’Ivoire":"AF","Cote d'Ivoire":"AF",
 "Myanmar (Burma)":"AS","Myanmar":"AS","Cape Verde":"AF","Swaziland":"AF",
 "Eswatini":"AF","Réunion":"AF","Reunion":"AF","Curaçao":"SA","Curacao":"SA",
 "St. Lucia":"NA","St. Kitts & Nevis":"NA","Caribbean Netherlands":"NA",
 "U.S. Virgin Islands":"NA","Myanmar(Burma)":"AS"}
_cache = {}
def real_cont(name):
    """Return one of the 6 real continents, or None for missing/unmapped.
    Used for cross-region counting so Unknown/unmapped never inflate spread."""
    if not name: return None
    if name in _cache: return _cache[name]
    code = FALLBACK.get(name)
    if code is None:
        try:
            a2 = pcc.country_name_to_country_alpha2(name)
            code = pcc.country_alpha2_to_continent_code(a2)
        except Exception:
            code = None
    c = CONT.get(code)            # None if code missing/Antarctica
    _cache[name] = c
    return c

def continent(name):
    """Bucketed continent for fig1 totals (keeps Other/Unknown)."""
    if not name: return "Unknown"
    rc = real_cont(name)
    return rc if rc else "Other"

country_ct = collections.Counter()
continent_ct = collections.Counter()
users_global = set()
total_rows = 0

# prefix sharing accumulators, per k
# key -> [count, set(country), set(continent), set(ip), prefix_token_len,
#         set(serving_region)]
share = {k: {} for k in KS}

files = sorted(glob.glob("datasets/wildchat-1m/data/*.parquet"))
t0 = time.time()
for fi, f in enumerate(files):
    tbl = pq.read_table(f, columns=["conversation", "country", "hashed_ip"])
    convs = tbl.column("conversation").to_pylist()
    countries = tbl.column("country").to_pylist()
    ips = tbl.column("hashed_ip").to_pylist()
    for conv, c, ip in zip(convs, countries, ips):
        total_rows += 1
        country_ct[c] += 1
        continent_ct[continent(c)] += 1
        rc = real_cont(c)            # 6 real continents or None (for region spread)
        sr = country_to_region(c)    # nearest of the 8 deployment regions
        if ip: users_global.add(ip)
        first = next((m["content"] for m in conv if m.get("role") == "user" and m.get("content")), None)
        if not first:
            continue
        toks = enc.encode(first[:12000], disallowed_special=())
        L = len(toks)
        if L < MINLEN_FOR_SHARE:
            continue
        for k in KS:
            if L < k:
                continue
            key = hashlib.md5(bytes(str(toks[:k]), "utf8")).hexdigest()
            d = share[k]
            e = d.get(key)
            if e is None:
                d[key] = [1, {c} if c else set(), {rc} if rc else set(),
                          {ip} if ip else set(), k, {sr}]
            else:
                e[0] += 1
                if c: e[1].add(c)
                if rc: e[2].add(rc)
                if ip: e[3].add(ip)
                e[5].add(sr)
    print(f"shard {fi+1}/14 done, rows={total_rows}, t={time.time()-t0:.0f}s", flush=True)

# ---- summarize ----
out = {
    "total_rows": total_rows,
    "n_countries": len([k for k in country_ct if k]),
    "n_users": len(users_global),
    "country_counts": dict(country_ct.most_common()),
    "continent_counts": dict(continent_ct),
    "sharing": {},
}

for k in KS:
    d = share[k]
    # per shared prefix: occurrences, #countries, #continents, #users
    recs = []
    for key, (cnt, cs, conts, ips, plen, sregs) in d.items():
        recs.append((cnt, len(cs), len(conts), len(ips), len(sregs)))
    total_req_with_prefix = sum(r[0] for r in recs)
    # shared = appears >=2 times from >=2 distinct users (real sharing, not same user repeat)
    shared = [r for r in recs if r[0] >= 2 and r[3] >= 2]
    xregion = [r for r in shared if r[2] >= 2]            # spans >=2 continents
    xcountry = [r for r in shared if r[1] >= 2]           # spans >=2 countries
    vol_total = total_req_with_prefix
    vol_shared = sum(r[0] for r in shared)
    vol_xregion = sum(r[0] for r in xregion)
    vol_xcountry = sum(r[0] for r in xcountry)
    # distribution of #continents and #countries among shared prefixes (weighted by prefix, i.e. per shared-prefix)
    ncont_hist = collections.Counter(r[2] for r in shared)
    ncountry_dist = sorted(r[1] for r in shared)
    out["sharing"][k] = {
        "n_prefixes_with_minlen": len(recs),
        "vol_total_req": vol_total,
        "n_shared_prefixes": len(shared),
        "n_xregion_prefixes": len(xregion),
        "n_xcountry_prefixes": len(xcountry),
        "vol_shared": vol_shared,
        "vol_xregion": vol_xregion,
        "vol_xcountry": vol_xcountry,
        "ncont_hist": dict(sorted(ncont_hist.items())),
        "ncountry_dist": ncountry_dist,   # list, for CDF
        # per shared-prefix tuples
        # (count, n_country, n_continent, n_users, n_serving_region)
        "scatter": [[r[0], r[1], r[2], r[3], r[4]] for r in shared],
    }

with open(OUT, "w") as fh:
    json.dump(out, fh)
print("WROTE", OUT, flush=True)
