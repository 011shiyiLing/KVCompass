"""prep_lmsys.py — build a section3_sim workload from LMSYS-Chat-1M.

Same per-turn cumulative-prefix expansion as prep_wildchat.py (turn-k prompt =
full history + user_k, so turn k is an exact prefix of turn k+1), minus the
geography: LMSYS-Chat-1M carries no country metadata, so entries have no
region_id (simulations fall back to demand-weighted region sampling, like the
mixed workload).

Output: datasets/lmsys.jsonl
"""
import argparse
import glob
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "datasets"
LM_DIR   = DATA_DIR / "lmsys-chat-1m" / "data"
OUT_PATH = DATA_DIR / "lmsys.jsonl"


def approx_tokens(text):
    return max(1, len(text) // 4)


def expand_conversation(conv, max_prompt_chars):
    running = ""
    turns = list(conv)
    i = 0
    while i < len(turns):
        msg = turns[i]
        if msg.get('role') != 'user':
            i += 1
            continue
        user_txt = msg.get('content') or ""
        asst_txt = ""
        if i + 1 < len(turns) and turns[i + 1].get('role') == 'assistant':
            asst_txt = turns[i + 1].get('content') or ""
        prompt = (running + user_txt)[:max_prompt_chars]
        if prompt.strip():
            yield {
                'workload': prompt,
                'gen_len':  min(approx_tokens(asst_txt), 2048) if asst_txt else 64,
            }
        running = (running + user_txt + asst_txt)[:max_prompt_chars]
        i += 2 if asst_txt else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--max-entries', type=int, default=375000)
    ap.add_argument('--max-prompt-chars', type=int, default=40000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    shard_files = sorted(glob.glob(str(LM_DIR / 'train-*.parquet')))[:args.shards]
    if not shard_files:
        raise SystemExit(f'no parquet shards under {LM_DIR}')
    print(f'reading {len(shard_files)} shard(s)…')

    convs = []
    for sf in shard_files:
        t = pq.read_table(sf, columns=['conversation_id', 'conversation'])
        d = t.to_pydict()
        for h, c in zip(d['conversation_id'], d['conversation']):
            convs.append((h, c))
    print(f'  {len(convs):,} conversations loaded')

    rng = random.Random(args.seed)
    rng.shuffle(convs)

    out = []
    for h, conv in convs:
        if len(out) >= args.max_entries:
            break
        if not conv:
            continue
        for e in expand_conversation(conv, args.max_prompt_chars):
            e['workload_id'] = len(out)
            e['conv'] = h
            out.append(e)
            if len(out) >= args.max_entries:
                break

    with open(OUT_PATH, 'w') as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    n_convs = len({e['conv'] for e in out})
    print(f'wrote {len(out):,} entries ({n_convs:,} conversations, '
          f'avg {len(out)/n_convs:.2f} turns) → {OUT_PATH}')


if __name__ == '__main__':
    main()
