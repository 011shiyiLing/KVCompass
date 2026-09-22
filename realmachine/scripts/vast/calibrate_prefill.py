#!/usr/bin/env python3
"""calibrate_prefill.py — measure C_PREFILL_MS_PER_TOKEN on the actual GPU.

The scoring math and the latency-levelled Ludo cuts (n₁ = WAN/C_PREFILL)
use an A100-calibrated prior (0.05 ms/tok).  RTX 5090 prefill is
substantially faster, which lengthens the cross-region reuse floor — so
the prior must be re-measured on the deployed hardware and passed to the
routers (C_PREFILL_MS_PER_TOKEN_PRIOR env, see run_multiregion_vast.sh).

Run ON a box (or via ssh) against one local worker AFTER flushing:
    python3 scripts/vast/calibrate_prefill.py --worker http://127.0.0.1:30000
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def measure_once(worker: str, n_words: int) -> tuple[float, int]:
    # Distinct filler content each call → no radix-cache reuse between runs.
    salt = str(time.time_ns())
    text = " ".join(f"{salt}w{i}" for i in range(n_words))
    body = json.dumps({
        "text": text,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{worker}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    ms = (time.monotonic() - t0) * 1000.0
    meta = out.get("meta_info") or {}
    return ms, int(meta.get("prompt_tokens", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="http://127.0.0.1:30000")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--n-words", type=int, default=6000,  # ≈ 8-12K tokens
                    help="filler words per probe prompt")
    args = ap.parse_args()

    urllib.request.urlopen(urllib.request.Request(
        f"{args.worker}/flush_cache", data=b"", method="POST"), timeout=10)

    rates = []
    for i in range(args.rounds):
        ms, ptoks = measure_once(args.worker, args.n_words)
        if ptoks > 0:
            rates.append(ms / ptoks)
            print(f"round {i}: {ptoks} tokens in {ms:.0f} ms "
                  f"→ {ms/ptoks:.4f} ms/tok")
    rates.sort()
    med = rates[len(rates) // 2]
    print(f"\nC_PREFILL_MS_PER_TOKEN_PRIOR={med:.4f}")
    print("Put this in scripts/vast/hosts.env:")
    print(f'  export C_PREFILL_MS_PER_TOKEN_PRIOR={med:.4f}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
