"""
mixed_trace.py — generate a deterministic, replayable mixed-workload
trace from the three §3 datasets.

Output format (JSONL, one request per line)::

    {"req_id": int, "dataset": str, "prefix_id": int,
     "prompt": str, "max_new_tokens": int}

The same trace is replayed across every (scheme, rate) combination so
that all measurements compare schemes on identical inputs in identical
order — the only thing that varies between runs is the routing
strategy.

Sampling matches §3 simulation:
  * Mix : 50 % toolbench / 5 % loogle / 45 % aaps  (override with --mix)
  * Per-dataset Zipf α = 1.1 popularity weighting
  * Tenant amplification is *not* used here; this is the single-tenant
    real-machine variant (the §4 multi-tenant scale-out is a separate
    experiment we do not run on a 2/8-GPU testbed).

LooGLE truncation:
  Real LooGLE prompts are 62 K–250 K characters (~15 K–60 K tokens).
  On a 2-A6000 testbed every full LooGLE prompt would single-handedly
  consume seconds of prefill, drowning out the rest of the workload.
  We truncate every prompt to ``--max-prompt-chars`` (default 8000,
  ≈ 2 K tokens) so LooGLE keeps its long-prefix character but stays
  within an order of magnitude of toolbench / aaps in compute weight.
  Set --max-prompt-chars 0 to disable truncation entirely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"

# (jsonl filename, field name holding the prompt, default gen_len)
# Same real datasets the §3 simulation loads (copy the jsonl files from
# `simulation/datasets/` into this repo's datasets/).
_DATASETS = {
    "toolbench": ("toolbench_zipf_1.1_prompts_6000.jsonl", "text",     32),
    "loogle":    ("loogle.jsonl",                          "workload", 50),
    "aaps":      ("aaps.jsonl",                            "workload", 50),
    # WildChat / LMSYS (eval-grid columns 2/3) — produced by the sim's
    # datasets/prep_wildchat.py / prep_lmsys.py; entries carry a real
    # per-turn `gen_len`, WildChat additionally a real `region_id`.
    "wildchat":  ("wildchat.jsonl",                        "workload", 64),
    "lmsys":     ("lmsys.jsonl",                           "workload", 64),
}

DEFAULT_MIX = (("toolbench", 0.50), ("loogle", 0.05), ("aaps", 0.45))

# --workload shorthands: the three eval-grid columns.
WORKLOAD_MIXES = {
    "mixed":    "toolbench:0.5,loogle:0.05,aaps:0.45",
    "wildchat": "wildchat:1.0",
    "lmsys":    "lmsys:1.0",
}

# WildChat's region_id indexes the simulation's 8-region space
# ['US','CN','DE','JP','SG','GB','KR','AU'].  For a smaller real
# deployment map each to the nearest testbed region.  Default targets the
# 3-region AWS layout [us-east-1, eu-central-1, ap-northeast-1]:
#   US→0;  DE,GB→1;  CN,JP,SG,KR,AU→2.
DEFAULT_WC_REGION_MAP = "0:0,1:2,2:1,3:2,4:2,5:1,6:2,7:2"

# The sim's REGION_REQUEST_WEIGHTS (WildChat demand census) aggregated by
# the same 8→3 mapping — use as --region-weights for mixed/lmsys so the
# demand skew matches the simulation:  US .453 | EU .159 | Asia .388.
DEFAULT_3REGION_WEIGHTS = "0.453,0.159,0.388"


def _load_pool(name: str) -> list[dict]:
    """Return [{'text': str, 'gen_len': int|None, 'region_id': int|None}]."""
    fn, field, _ = _DATASETS[name]
    path = DATASETS_DIR / fn
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {name} not found at {path} — copy it from "
            f"'simulation/datasets/'")
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # Field names differ across extractors — try the registered
            # field first, then the common alternatives.
            text = d.get(field) or d.get("text") or d.get("workload") \
                or d.get("prompt") or ""
            if not text:
                continue
            gen_len = d.get("gen_len")
            if gen_len is None and isinstance(d.get("sampling_params"), dict):
                gen_len = d["sampling_params"].get("max_new_tokens")
            out.append({
                "text":      text,
                "gen_len":   int(gen_len) if gen_len is not None else None,
                "region_id": (int(d["region_id"])
                              if d.get("region_id") is not None else None),
            })
    return out


def parse_region_map(spec: str) -> dict[int, int]:
    """Parse '0:0,1:2,...' into {dataset_region: deployment_region}."""
    out: dict[int, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        out[int(k)] = int(v)
    return out


def _zipf_weights(n: int, alpha: float) -> np.ndarray:
    w = np.arange(1, n + 1, dtype=float) ** (-alpha)
    return w / w.sum()


def parse_mix(spec: str) -> tuple[tuple[str, float], ...]:
    """Parse 'toolbench:0.5,loogle:0.05,aaps:0.45' into a tuple."""
    out = []
    for chunk in spec.split(","):
        name, w = chunk.split(":")
        name = name.strip()
        if name not in _DATASETS:
            raise ValueError(f"Unknown dataset {name!r}; valid: {list(_DATASETS)}")
        out.append((name, float(w)))
    total = sum(w for _, w in out)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--mix weights must sum to 1.0, got {total}")
    return tuple(out)


def build_trace(
    out_path: Path,
    n_requests: int,
    seed: int = 42,
    alpha: float = 1.1,
    mix: tuple[tuple[str, float], ...] = DEFAULT_MIX,
    max_prompt_chars: int = 8000,
    region_weights: list[float] | None = None,
    wc_region_map: dict[int, int] | None = None,
    n_tenants: int = 0,
) -> dict:
    """Generate the trace; return a stats dict for sanity logging."""
    rng = np.random.default_rng(seed)

    # Multi-tenant amplification (sim's workload_multitenant.py, §3 grid
    # uses n_tenants=50): each tenant gets a private ~30-80-token system
    # prefix PREPENDED TO THE PROMPT TEXT, so the real tokenizer/
    # RadixCache reproduces the sim's sharing structure exactly —
    # within-tenant sharing preserved, cross-tenant prompts never share
    # a prefix.  Tenant drawn uniformly per request (the harder case
    # for discovery, as in the sim).
    tenant_prefixes: list[str] = []
    if n_tenants > 0:
        trng = np.random.default_rng(seed + 101)
        for t in range(n_tenants):
            n_words = int(trng.integers(12, 31))     # ≈30-80 tokens
            words = ["".join(chr(97 + c) for c in trng.integers(0, 26, 5))
                     for _ in range(n_words)]
            tenant_prefixes.append(
                f"[tenant-{t}] system context: " + " ".join(words) + "\n")
        tenant_rng = np.random.default_rng(seed + 211)
    pools: dict[str, list[dict]] = {}
    for name, _ in mix:
        pools[name] = _load_pool(name)
        if len(pools[name]) == 0:
            raise RuntimeError(f"Dataset {name} loaded but is empty")

    # Per-dataset Zipf weights (over its own pool size).
    zipf: dict[str, np.ndarray] = {
        name: _zipf_weights(len(pools[name]), alpha) for name, _ in mix
    }

    # Inverse-CDF for the dataset selector — one searchsorted per draw.
    mix_names = [n for n, _ in mix]
    mix_cum = np.cumsum([w for _, w in mix])

    # Per-dataset prefix_id base so global ids stay unique across pools.
    base_ids = {}
    cursor = 0
    for name in mix_names:
        base_ids[name] = cursor
        cursor += len(pools[name])

    # Origin-region assignment per request: drawn uniformly from
    # NUM_REGIONS (matches §3 simulation). We sample once at trace
    # build time and write it into the JSONL so any subsequent replay
    # — across schemes or rates — sees identical geographic origins.
    # Using the same RNG instance keeps it tied to the trace seed.
    from ludo_sglang.config import NUM_REGIONS_LOGICAL

    # Origin-region sampling: uniform by default (legacy); pass
    # --region-weights to skew demand geographically (e.g. WildChat's
    # demand census for a 3-region deployment: "0.5,0.3,0.2").
    if region_weights is not None:
        rw = np.asarray(region_weights, dtype=float)
        if len(rw) != NUM_REGIONS_LOGICAL:
            raise ValueError(
                f"--region-weights has {len(rw)} entries but the WAN "
                f"matrix defines {NUM_REGIONS_LOGICAL} regions")
        rw = rw / rw.sum()
    else:
        rw = None

    counts = {name: 0 for name in mix_names}
    char_lens: list[int] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for req_id in range(n_requests):
            ds_idx = int(np.searchsorted(mix_cum, rng.random()))
            if ds_idx >= len(mix_names):
                ds_idx = len(mix_names) - 1
            ds = mix_names[ds_idx]
            pool = pools[ds]
            in_pool_idx = int(rng.choice(len(pool), p=zipf[ds]))
            entry = pool[in_pool_idx]
            prompt = entry["text"]
            prefix_id = base_ids[ds] + in_pool_idx
            if n_tenants > 0:
                tenant = int(tenant_rng.integers(0, n_tenants))
                prompt = tenant_prefixes[tenant] + prompt
                prefix_id = tenant * 100_000 + prefix_id
            if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
                prompt = prompt[:max_prompt_chars]
            char_lens.append(len(prompt))
            counts[ds] += 1
            _, _, default_gen = _DATASETS[ds]
            # gen_len: real per-entry value where the dataset carries one
            # (ToolBench sampling_params, WildChat/LMSYS per-turn gen_len)
            # — same rule as the sim's _genlen_for().
            gen_len = entry["gen_len"] if entry["gen_len"] else default_gen
            # Origin region: WildChat entries carry a REAL region_id
            # (from the conversation's country, in the sim's 8-region
            # space) — map it onto the deployment regions.  Everything
            # else falls through to demand-weighted sampling, mirroring
            # the sim's _region_for().
            origin_region = None
            if entry["region_id"] is not None and wc_region_map:
                mapped = wc_region_map.get(entry["region_id"])
                if mapped is not None and 0 <= mapped < NUM_REGIONS_LOGICAL:
                    origin_region = mapped
            if origin_region is None:
                if rw is not None:
                    origin_region = int(rng.choice(NUM_REGIONS_LOGICAL, p=rw))
                else:
                    origin_region = int(rng.integers(0, NUM_REGIONS_LOGICAL))
            row = {
                "req_id":         req_id,
                "dataset":        ds,
                "prefix_id":      prefix_id,
                "origin_region":  origin_region,
                "prompt":         prompt,
                "max_new_tokens": gen_len,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    char_arr = np.array(char_lens)
    stats = {
        "out_path":         str(out_path),
        "n_requests":       n_requests,
        "seed":             seed,
        "alpha":            alpha,
        "mix":              {n: c / n_requests for n, c in counts.items()},
        "raw_counts":       counts,
        "char_len":         {
            "p50":   int(np.percentile(char_arr, 50)),
            "p95":   int(np.percentile(char_arr, 95)),
            "max":   int(char_arr.max()),
            "mean":  float(char_arr.mean()),
        },
        "max_prompt_chars": max_prompt_chars,
    }
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out",  type=Path, required=True,
                   help="Output trace path (JSONL).")
    p.add_argument("--n",    type=int,  default=500,
                   help="Number of requests to generate (default: 500).")
    p.add_argument("--seed", type=int,  default=42)
    p.add_argument("--alpha", type=float, default=1.1,
                   help="Zipf alpha for per-dataset popularity (default: 1.1).")
    p.add_argument("--mix",  type=str, default="toolbench:0.5,loogle:0.05,aaps:0.45",
                   help='Mix spec, e.g. "toolbench:0.5,loogle:0.05,aaps:0.45".')
    p.add_argument("--workload", type=str, default=None,
                   choices=sorted(WORKLOAD_MIXES),
                   help="Eval-grid workload shorthand; overrides --mix "
                        "(mixed / wildchat / lmsys).")
    p.add_argument("--region-weights", type=str, default=None,
                   help="Comma-separated origin-region demand weights, one "
                        "per WAN-matrix region.  For the 3-region AWS "
                        f'deployment use "{DEFAULT_3REGION_WEIGHTS}" (the '
                        "sim's WildChat demand census aggregated US/EU/Asia)."
                        "  Default: uniform.")
    p.add_argument("--wc-region-map", type=str, default=DEFAULT_WC_REGION_MAP,
                   help="Mapping from WildChat's 8-region region_id to "
                        "deployment regions ('' disables and falls back to "
                        f"--region-weights).  Default: {DEFAULT_WC_REGION_MAP}")
    p.add_argument("--max-prompt-chars", type=int, default=8000,
                   help="Truncate prompts to this many characters (0 = no truncation, "
                        "default: 8000 ≈ 2 K tokens).")
    p.add_argument("--n-tenants", type=int, default=0,
                   help="Multi-tenant amplification (sim §3 grid: 50). "
                        "Each tenant gets a private system prefix prepended "
                        "to its prompts; 0 disables.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mix_spec = WORKLOAD_MIXES[args.workload] if args.workload else args.mix
    mix = parse_mix(mix_spec)
    region_weights = ([float(x) for x in args.region_weights.split(",")]
                      if args.region_weights else None)
    wc_region_map = (parse_region_map(args.wc_region_map)
                     if args.wc_region_map else None)
    stats = build_trace(
        out_path        = args.out,
        n_requests      = args.n,
        seed            = args.seed,
        alpha           = args.alpha,
        mix             = mix,
        max_prompt_chars = args.max_prompt_chars,
        region_weights  = region_weights,
        wc_region_map   = wc_region_map,
        n_tenants       = args.n_tenants,
    )
    print(f"=== Trace generated ===")
    print(f"  path        : {stats['out_path']}")
    print(f"  n_requests  : {stats['n_requests']}")
    print(f"  seed/alpha  : {stats['seed']} / {stats['alpha']}")
    print(f"  mix actual  : "
          + ", ".join(f"{n}={f:.3f}" for n, f in stats['mix'].items()))
    print(f"  prompt chars: p50={stats['char_len']['p50']:,}  "
          f"p95={stats['char_len']['p95']:,}  "
          f"max={stats['char_len']['max']:,}  "
          f"mean={stats['char_len']['mean']:.0f}")
    if stats["max_prompt_chars"] > 0:
        print(f"  truncated to: {stats['max_prompt_chars']:,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
