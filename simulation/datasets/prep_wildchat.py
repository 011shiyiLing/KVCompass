"""prep_wildchat.py — build a section3_sim workload from WildChat-1M.

WildChat-1M ships as HuggingFace parquet shards under
`datasets/wildchat-1m/data/`, one row per multi-turn conversation.  We turn it
into the flat `{"workload_id", "workload", ...}` JSONL that
`WorkloadGenerator` consumes, exposing the KV-cache prefix-sharing that makes
conversational traffic interesting:

  * Per-turn cumulative prefixes.  A K-turn conversation becomes K entries; the
    turn-k prompt is the full history up to and including the k-th user message
    (turns 1..k-1 user+assistant, then user_k).  Entry k's text is therefore an
    exact PREFIX of entry k+1's text, so consecutive turns collide at the
    content-hash levels and reproduce multi-turn KV reuse.  Different
    conversations sharing an opening (system prompt / common intro) also collide.

  * Real per-request geography.  Each entry carries region_id derived from the
    conversation's WildChat `country` (the 8 deployment regions US/CN/DE/JP/SG/
    GB/KR/AU; other countries fall back to the nearest region by continent).

  * Real decode length.  gen_len = length of the assistant's reply at that turn
    (in the same 4-char≈1-token unit the simulator uses for text workloads).

Output: datasets/wildchat.jsonl

Usage:
  python datasets/prep_wildchat.py [--shards N] [--max-entries M] [--seed S]
"""
import argparse
import glob
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import pycountry_convert as pcc

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "datasets"
WC_DIR    = DATA_DIR / "wildchat-1m" / "data"
OUT_PATH  = DATA_DIR / "wildchat.jsonl"

# 8 deployment regions (must match simulate.REGION_NAMES order → region_id).
REGION_NAMES = ['US', 'CN', 'DE', 'JP', 'SG', 'GB', 'KR', 'AU']
REGION_IDX   = {name: i for i, name in enumerate(REGION_NAMES)}

# Direct country-name → region for the 8 hosting countries.
DIRECT = {
    'United States': 'US', 'China': 'CN', 'Germany': 'DE', 'Japan': 'JP',
    'Singapore': 'SG', 'United Kingdom': 'GB', 'South Korea': 'KR',
    'Australia': 'AU',
}
# Continent → representative hosting region for everything else (nearest hub).
CONT_REGION = {
    'NA': 'US', 'SA': 'US',   # Americas  → US
    'EU': 'DE', 'AF': 'DE',   # Europe / Africa → Frankfurt
    'AS': 'CN',               # Asia      → CN (nearest large Asian hub)
    'OC': 'AU',               # Oceania   → AU
}
# WildChat spellings pycountry_convert tends to miss.
FALLBACK_CONT = {
    'Russia': 'EU', 'South Korea': 'AS', 'Hong Kong': 'AS', 'Taiwan': 'AS',
    'Vietnam': 'AS', 'Iran': 'AS', 'Syria': 'AS', 'Czechia': 'EU',
    'Moldova': 'EU', 'Kosovo': 'EU', 'Bolivia': 'SA', 'Venezuela': 'SA',
    'Tanzania': 'AF', "Côte d’Ivoire": 'AF', 'Myanmar (Burma)': 'AS',
    'Palestine': 'AS', 'Brunei': 'AS', 'Laos': 'AS', 'Macao': 'AS',
}
_region_cache = {}


def country_to_region(country):
    """Map a WildChat country name to one of the 8 region_ids (default US)."""
    if not country:
        return REGION_IDX['US']
    if country in _region_cache:
        return _region_cache[country]
    if country in DIRECT:
        rid = REGION_IDX[DIRECT[country]]
    else:
        cont = FALLBACK_CONT.get(country)
        if cont is None:
            try:
                a2   = pcc.country_name_to_country_alpha2(country)
                cont = pcc.country_alpha2_to_continent_code(a2)
            except Exception:
                cont = None
        rid = REGION_IDX[CONT_REGION.get(cont, 'US')]
    _region_cache[country] = rid
    return rid


def approx_tokens(text):
    """Same 4-char≈1-token approximation the simulator uses for text."""
    return max(1, len(text) // 4)


def expand_conversation(conv, country, max_prompt_chars):
    """Yield per-turn cumulative-prefix entries for one conversation.

    conv is the WildChat `conversation` list of {role, content, ...} in order.
    """
    region_id = country_to_region(country)
    running   = ""            # history text before the current user message
    turns     = list(conv)
    i = 0
    while i < len(turns):
        msg = turns[i]
        if msg.get('role') != 'user':
            i += 1
            continue
        user_txt = msg.get('content') or ""
        # Assistant reply (next message, if present) → decode length.
        asst_txt = ""
        if i + 1 < len(turns) and turns[i + 1].get('role') == 'assistant':
            asst_txt = turns[i + 1].get('content') or ""
        prompt = (running + user_txt)[:max_prompt_chars]
        if prompt.strip():
            yield {
                'workload':  prompt,
                'gen_len':   min(approx_tokens(asst_txt), 2048) if asst_txt else 64,
                'region_id': region_id,
                'country':   country or 'Unknown',
            }
        # Extend history with this turn's user + assistant content.
        running = (running + user_txt + asst_txt)[:max_prompt_chars]
        i += 2 if asst_txt else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=int, default=1,
                    help='number of parquet shards to read (each ≈60k convs)')
    ap.add_argument('--max-entries', type=int, default=15000,
                    help='cap on total emitted entries (pool size)')
    ap.add_argument('--max-prompt-chars', type=int, default=40000,
                    help='truncate very long cumulative prompts (~10k tokens)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    shard_files = sorted(glob.glob(str(WC_DIR / 'train-*.parquet')))[:args.shards]
    if not shard_files:
        raise SystemExit(f'no parquet shards under {WC_DIR}')
    print(f'reading {len(shard_files)} shard(s)…')

    # Collect conversations (hash, conv, country), then shuffle for diversity.
    convs = []
    for sf in shard_files:
        t = pq.read_table(sf, columns=['conversation_hash', 'conversation',
                                       'country'])
        d = t.to_pydict()
        for h, c, ctry in zip(d['conversation_hash'], d['conversation'],
                              d['country']):
            convs.append((h, c, ctry))
    print(f'  {len(convs):,} conversations loaded')

    rng = random.Random(args.seed)
    rng.shuffle(convs)

    out       = []
    region_hist = {i: 0 for i in range(len(REGION_NAMES))}
    for h, conv, ctry in convs:
        if len(out) >= args.max_entries:
            break
        if not conv:
            continue
        for e in expand_conversation(conv, ctry, args.max_prompt_chars):
            e['workload_id'] = len(out)
            e['conv'] = h
            out.append(e)
            region_hist[e['region_id']] += 1
            if len(out) >= args.max_entries:
                break

    with open(OUT_PATH, 'w') as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    print(f'wrote {len(out):,} entries → {OUT_PATH}')
    tot = sum(region_hist.values()) or 1
    print('region distribution (from real WildChat country):')
    for i, name in enumerate(REGION_NAMES):
        print(f'  {name}: {region_hist[i]:6d}  ({100*region_hist[i]/tot:4.1f}%)')


if __name__ == '__main__':
    main()
