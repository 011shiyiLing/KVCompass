#!/usr/bin/env python3
"""export_tokens.py — tokenize the mixed trace's prompts into a binary
corpus for the P2 memory benchmark (REAL token sequences; radix-tree
memory is meaningless on synthetic strings).

Format (little-endian): u32 n_seqs, then per sequence u32 len + len×i32.
Run ON a box (needs transformers + the cached Llama tokenizer):
    python3 experiments/p2_mem/export_tokens.py \
        --trace traces/mixed_mt50_6000_seed42.jsonl \
        --out results_tmp/p2_tokens.bin
"""
from __future__ import annotations

import argparse
import json
import struct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tokenizer-json", default="",
                    help="load PreTrainedTokenizerFast directly from this "
                         "tokenizer.json (offline; skips AutoTokenizer's "
                         "hub round-trip)")
    a = ap.parse_args()

    if a.tokenizer_json:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=a.tokenizer_json)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)

    # Distinct prompts only — the C++ side amplifies via tenant prefixes.
    seen, prompts = set(), []
    for line in open(a.trace):
        p = json.loads(line)["prompt"]
        if p not in seen:
            seen.add(p)
            prompts.append(p)
    print(f"{len(prompts)} distinct prompts; tokenizing ...")
    with open(a.out, "wb") as f:
        f.write(struct.pack("<I", len(prompts)))
        total = 0
        for i, p in enumerate(prompts):
            ids = tok(p, add_special_tokens=False)["input_ids"]
            f.write(struct.pack("<I", len(ids)))
            f.write(struct.pack(f"<{len(ids)}i", *ids))
            total += len(ids)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(prompts)}")
    print(f"wrote {a.out}: {len(prompts)} seqs, {total} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
