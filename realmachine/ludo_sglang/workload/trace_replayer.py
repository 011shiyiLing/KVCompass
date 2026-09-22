"""
trace_replayer.py — Poisson-arrival load generator for the Ludo router.

Reads a trace file produced by ``mixed_trace.py``, fires every request
to the router according to an exponential inter-arrival schedule
(homogeneous Poisson at ``--rate`` req/s), captures every response, and
writes one CSV row per request — including TTFT, E2E, the router's own
score estimates, and the SGLang RadixCache hit count.

The replayer is deliberately decoupled from the router lifecycle: it
does NOT start or stop the router, just hits its HTTP endpoint. The
shell driver (``scripts/run_experiment.sh``) handles router/SGLang
process lifetime so that one router-per-(scheme, rate) restart gives
each measurement a clean state.

Usage
-----
    python -m ludo_sglang.workload.trace_replayer \\
        --trace traces/mix_500.jsonl \\
        --router http://127.0.0.1:8000 \\
        --rate 4 \\
        --out results/ludo_r4.csv \\
        --warmup 5

CSV columns
-----------
Per-request data flows through three time domains:
  client wall clock — what the user/replayer observes
  router            — scoring / load-tracker estimates
  sglang worker     — RadixCache hit, true prompt/completion tokens

All three are flattened into one row so downstream analysis is just
``pd.read_csv(path)``.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import aiohttp
import numpy as np


# Canonical CSV schema. Order matters — kept stable so analysis scripts
# don't have to introspect.
CSV_FIELDS: tuple[str, ...] = (
    # Replayer-side timing
    "req_id",
    "dataset",
    "prefix_id",
    "origin_region",
    "arrival_s",          # planned arrival time (s, since trace t=0)
    "send_s",             # actual time we POSTed
    "wall_ms",            # POST → response close (client-side)
    "ok",                 # 1 if HTTP 200, 0 otherwise
    "error",              # truncated error string when ok=0
    # Router-side
    "scheme",
    "dp_rank",
    "worker_region",
    "matched_tokens",
    "residual_tokens",
    "router_overhead_ms",   # = lookup overhead + WAN to picked worker
    "wan_to_worker_ms",
    "score_ms",
    "queue_wait_estimate_ms",
    "reserved_ms",
    "inferred_prefill_rate",
    # SGLang-side
    "ttft_ms",
    "e2e_ms",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
)


def _empty_row(req: dict, arrival_s: float, send_s: float,
               wall_ms: float, ok: int, error: str = "") -> dict:
    """Build a row with NaN/empty payload — used for failed requests."""
    return {
        "req_id":        req["req_id"],
        "dataset":       req["dataset"],
        "prefix_id":     req.get("prefix_id", -1),
        "origin_region": req.get("origin_region", ""),
        "arrival_s":     arrival_s,
        "send_s":        send_s,
        "wall_ms":       wall_ms,
        "ok":            ok,
        "error":         error[:200],
        "scheme":        "",
        **{k: "" for k in CSV_FIELDS
           if k not in ("req_id", "dataset", "prefix_id", "origin_region",
                        "arrival_s", "send_s", "wall_ms", "ok", "error",
                        "scheme")},
    }


def _full_row(req: dict, arrival_s: float, send_s: float,
              wall_ms: float, payload: dict) -> dict:
    return {
        "req_id":        req["req_id"],
        "dataset":       req["dataset"],
        "prefix_id":     req.get("prefix_id", -1),
        "origin_region": req.get("origin_region", payload.get("origin_region", "")),
        "arrival_s":     arrival_s,
        "send_s":        send_s,
        "wall_ms":       wall_ms,
        "ok":            1,
        "error":         "",
        "scheme":                  payload.get("scheme", ""),
        "dp_rank":                 payload.get("dp_rank", ""),
        "worker_region":           payload.get("worker_region", ""),
        "matched_tokens":          payload.get("matched_tokens", ""),
        "residual_tokens":         payload.get("residual_tokens", ""),
        "router_overhead_ms":      payload.get("router_overhead_ms", ""),
        "wan_to_worker_ms":        payload.get("wan_to_worker_ms", ""),
        "score_ms":                payload.get("score_ms", ""),
        "queue_wait_estimate_ms":  payload.get("queue_wait_estimate_ms", ""),
        "reserved_ms":             payload.get("reserved_ms", ""),
        "inferred_prefill_rate":   payload.get("inferred_prefill_rate", ""),
        "ttft_ms":                 payload.get("ttft_ms", ""),
        "e2e_ms":                  payload.get("e2e_ms", ""),
        "prompt_tokens":           payload.get("prompt_tokens", ""),
        "completion_tokens":       payload.get("completion_tokens", ""),
        "cached_tokens":           payload.get("cached_tokens", ""),
    }


async def _fire_one(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                    router_url: str, req: dict,
                    arrival_s: float, t_zero: float, http_timeout: float,
                    rows: list, n_done: list) -> None:
    """Sleep until ``arrival_s`` (relative to ``t_zero``), then POST."""
    target_wall = t_zero + arrival_s
    delay = target_wall - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)

    async with sem:
        send_s = time.monotonic() - t_zero
        t_send = time.monotonic()
        try:
            post_body = {
                "prompt":         req["prompt"],
                "max_new_tokens": req["max_new_tokens"],
                "request_id":     str(req["req_id"]),
            }
            # Forward origin_region so the router applies the
            # trace's geographic origin (not a fresh derivation).
            if "origin_region" in req:
                post_body["origin_region"] = req["origin_region"]
            async with session.post(
                f"{router_url}/generate",
                json=post_body,
                timeout=aiohttp.ClientTimeout(total=http_timeout),
            ) as resp:
                wall_ms = (time.monotonic() - t_send) * 1000.0
                if resp.status != 200:
                    body = (await resp.text())[:300]
                    rows.append(_empty_row(req, arrival_s, send_s, wall_ms,
                                           ok=0, error=f"HTTP {resp.status}: {body}"))
                else:
                    payload = await resp.json()
                    rows.append(_full_row(req, arrival_s, send_s, wall_ms, payload))
        except Exception as e:
            wall_ms = (time.monotonic() - t_send) * 1000.0
            rows.append(_empty_row(req, arrival_s, send_s, wall_ms,
                                   ok=0, error=f"{type(e).__name__}: {e}"))
        finally:
            n_done[0] += 1


async def replay(
    trace_path: Path,
    router_url: str,
    rate_rps: float,
    out_csv: Path,
    warmup_s: float = 0.0,
    max_concurrent: int = 1024,
    http_timeout: float = 600.0,
    seed: int = 0,
    progress_every: int = 50,
    origin_filter: int | None = None,
    limit: int | None = None,
    warmup_n: int = 0,
) -> dict:
    requests = [json.loads(l) for l in open(trace_path)]
    if limit is not None and limit > 0:
        # Fixed-duration methodology (sim's rate_to_n_warmup): fire only
        # the first `limit` = rate × duration requests.  Applied BEFORE
        # the arrival draw + origin filter so all per-region replayers
        # of one cell agree on the request set and schedule.
        requests = requests[:limit]
    n = len(requests)

    # Homogeneous Poisson via cumulative exponential inter-arrivals.
    # Computed over the FULL trace before any origin filtering, so that
    # three per-region replayers (--origin-filter 0/1/2) driven from the
    # same trace+seed jointly reproduce the same global arrival process
    # a single replayer would.
    rng = np.random.default_rng(seed)
    inter = rng.exponential(1.0 / rate_rps, size=n)
    arrivals = np.cumsum(inter)

    if origin_filter is not None:
        keep = [i for i, r in enumerate(requests)
                if int(r.get("origin_region", -1)) == origin_filter]
        requests = [requests[i] for i in keep]
        arrivals = arrivals[keep]
        print(f"[replayer] origin_filter={origin_filter}: "
              f"{len(requests)}/{n} requests")
        n = len(requests)

    rows: list[dict] = []
    n_done = [0]
    sem = asyncio.Semaphore(max_concurrent)

    t_zero = time.monotonic()
    print(f"[replayer] firing {n} requests at {rate_rps:.2f} rps, "
          f"warmup={warmup_s}s, max_concurrent={max_concurrent}")

    async def _progress():
        while n_done[0] < n:
            await asyncio.sleep(2.0)
            elapsed = time.monotonic() - t_zero
            print(f"[replayer]   t={elapsed:6.1f}s  done {n_done[0]}/{n}  "
                  f"({n_done[0]/max(elapsed,1e-6):.1f} req/s observed)",
                  flush=True)

    conn = aiohttp.TCPConnector(limit=max_concurrent + 32)
    async with aiohttp.ClientSession(connector=conn) as session:
        prog = asyncio.create_task(_progress())
        await asyncio.gather(*[
            _fire_one(session, sem, router_url, req,
                      float(arrivals[i]), t_zero, http_timeout, rows, n_done)
            for i, req in enumerate(requests)
        ])
        prog.cancel()
        try:
            await prog
        except asyncio.CancelledError:
            pass

    total_wall_s = time.monotonic() - t_zero
    rows.sort(key=lambda r: r["req_id"])

    # Drop warmup window. We use *arrival_s*, not send_s — warmup is a
    # property of the trace, independent of how queued the system was.
    if warmup_s > 0:
        rows_kept = [r for r in rows if r["arrival_s"] >= warmup_s]
    else:
        rows_kept = rows
    if warmup_n > 0:
        # Request-count warmup: additionally drop the warmup_n earliest
        # arrivals (per this replayer's origin slice).
        by_arr = sorted(rows_kept, key=lambda r: r["arrival_s"])
        cut_ids = {id(r) for r in by_arr[:warmup_n]}
        rows_kept = [r for r in rows_kept if id(r) not in cut_ids]
    n_dropped = len(rows) - len(rows_kept)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for r in rows_kept:
            w.writerow(r)

    n_ok = sum(1 for r in rows_kept if r["ok"] == 1)
    ttfts = [float(r["ttft_ms"]) for r in rows_kept
             if r["ok"] == 1 and r["ttft_ms"] != ""]
    e2es  = [float(r["e2e_ms"])  for r in rows_kept
             if r["ok"] == 1 and r["e2e_ms"]  != ""]

    # Service throughput over the post-warmup window (tokens/s of
    # prompt+completion work completed) — the eval-grid row-4 metric
    # for the open-loop runs; the closed-loop mode below is the
    # saturation-throughput variant matching the sim's fig5.
    tok_total = sum(
        int(r["prompt_tokens"] or 0) + int(r["completion_tokens"] or 0)
        for r in rows_kept if r["ok"] == 1
        and r["prompt_tokens"] != "" and r["completion_tokens"] != "")
    window_s = max(1e-6, total_wall_s - warmup_s)

    summary = {
        "trace":     str(trace_path),
        "out_csv":   str(out_csv),
        "rate_rps":  rate_rps,
        "origin_filter": origin_filter,
        "n_total":   n,
        "n_kept":    len(rows_kept),
        "n_dropped_warmup": n_dropped,
        "warmup_requests":  warmup_n,
        "n_ok":      n_ok,
        "n_failed":  len(rows_kept) - n_ok,
        "wall_s":    total_wall_s,
        "observed_rps": n / total_wall_s if total_wall_s > 0 else 0.0,
        "ttft_ms":   _percentile_dict(ttfts),
        "e2e_ms":    _percentile_dict(e2es),
        "tokens_total":    tok_total,
        "tpt_total_tok_s": tok_total / window_s,
    }
    return summary


async def closed_loop(
    trace_path: Path,
    router_url: str,
    n_clients: int,
    duration_s: float,
    out_csv: Path,
    warmup_s: float = 0.0,
    http_timeout: float = 600.0,
    seed: int = 0,
    warmup_n: int = 0,
    warmup_cap_s: float = 240.0,
) -> dict:
    """Closed-loop throughput measurement (sim fig5 / eval-grid row 4).

    ``n_clients`` concurrent clients each loop over the trace (own
    random order): fire a request, await completion, immediately fire
    the next — the offered load self-adjusts to the fleet's service
    capacity, so completed tokens/s IS the service throughput.

    ``warmup_n`` > 0: the first ``warmup_n`` COMPLETIONS are a warmup
    phase excluded from the stats; the measured window is the
    ``duration_s`` seconds after the warmup_n-th completion (so the
    reported window is always duration_s of steady state).  The warmup
    phase is capped at ``warmup_cap_s`` in case throughput is too low to
    ever complete warmup_n requests.
    """
    requests = [json.loads(l) for l in open(trace_path)]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    t_zero = time.monotonic()
    # Mutable deadline: with request-count warmup the measured window
    # starts only when the warmup_n-th completion lands.
    deadline = [t_zero + duration_s + (warmup_cap_s if warmup_n > 0 else 0.0)]
    t_warm = [None]     # monotonic time of the warmup_n-th completion
    n_done = [0]

    async def _client(cid: int):
        order = rng.permutation(len(requests))
        i = 0
        while time.monotonic() < deadline[0]:
            req = requests[int(order[i % len(order)])]
            i += 1
            send_s = time.monotonic() - t_zero
            t_send = time.monotonic()
            try:
                post_body = {
                    "prompt":         req["prompt"],
                    "max_new_tokens": req["max_new_tokens"],
                    "request_id":     str(req["req_id"]),
                }
                if "origin_region" in req:
                    post_body["origin_region"] = req["origin_region"]
                async with session.post(
                    f"{router_url}/generate", json=post_body,
                    timeout=aiohttp.ClientTimeout(total=http_timeout),
                ) as resp:
                    wall_ms = (time.monotonic() - t_send) * 1000.0
                    if resp.status != 200:
                        body = (await resp.text())[:300]
                        rows.append(_empty_row(req, send_s, send_s, wall_ms,
                                               ok=0, error=f"HTTP {resp.status}: {body}"))
                    else:
                        payload = await resp.json()
                        rows.append(_full_row(req, send_s, send_s, wall_ms, payload))
            except Exception as e:
                wall_ms = (time.monotonic() - t_send) * 1000.0
                rows.append(_empty_row(req, send_s, send_s, wall_ms,
                                       ok=0, error=f"{type(e).__name__}: {e}"))
            finally:
                n_done[0] += 1
                if (warmup_n > 0 and t_warm[0] is None
                        and n_done[0] >= warmup_n):
                    t_warm[0] = time.monotonic()
                    deadline[0] = t_warm[0] + duration_s

    async def _progress():
        while time.monotonic() < deadline[0]:
            await asyncio.sleep(2.0)
            elapsed = time.monotonic() - t_zero
            phase = "warmup" if (warmup_n > 0 and t_warm[0] is None) else "measure"
            print(f"[closed-loop] t={elapsed:5.1f}s  done {n_done[0]}  "
                  f"({n_done[0]/max(elapsed,1e-6):.1f} req/s) [{phase}]",
                  flush=True)

    print(f"[closed-loop] {n_clients} clients × {duration_s:.0f}s "
          f"against {router_url}")
    conn = aiohttp.TCPConnector(limit=n_clients + 32)
    async with aiohttp.ClientSession(connector=conn) as session:
        prog = asyncio.create_task(_progress())
        await asyncio.gather(*[_client(c) for c in range(n_clients)])
        prog.cancel()
        try:
            await prog
        except asyncio.CancelledError:
            pass

    total_wall_s = time.monotonic() - t_zero
    if warmup_n > 0:
        # rows append in completion order → the first warmup_n rows are
        # exactly the warmup completions.
        rows_kept = rows[warmup_n:]
        warmup_wall_s = ((t_warm[0] - t_zero) if t_warm[0] is not None
                         else total_wall_s)
    else:
        # Keep only requests COMPLETED after the warmup point (send_s is
        # the completion-side filter here: the request must have been
        # issued post-warmup to count toward steady-state throughput).
        rows_kept = [r for r in rows if r["send_s"] >= warmup_s]
        warmup_wall_s = warmup_s

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for r in rows_kept:
            w.writerow(r)

    n_ok = sum(1 for r in rows_kept if r["ok"] == 1)
    tok_total = sum(
        int(r["prompt_tokens"] or 0) + int(r["completion_tokens"] or 0)
        for r in rows_kept if r["ok"] == 1
        and r["prompt_tokens"] != "" and r["completion_tokens"] != "")
    window_s = max(1e-6, total_wall_s - warmup_wall_s)
    ttfts = [float(r["ttft_ms"]) for r in rows_kept
             if r["ok"] == 1 and r["ttft_ms"] != ""]

    return {
        "mode":            "closed_loop",
        "trace":           str(trace_path),
        "out_csv":         str(out_csv),
        "n_clients":       n_clients,
        "duration_s":      duration_s,
        "warmup_requests": warmup_n,
        "warmup_wall_s":   warmup_wall_s,
        "warmup_reached":  (warmup_n == 0 or t_warm[0] is not None),
        "measured_window_s": window_s,
        "n_completed":     len(rows_kept),
        "n_ok":            n_ok,
        "n_failed":        len(rows_kept) - n_ok,
        "wall_s":          total_wall_s,
        "tokens_total":    tok_total,
        "tpt_total_tok_s": tok_total / window_s,
        "ttft_ms":         _percentile_dict(ttfts),
    }


def _percentile_dict(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}
    a = np.array(values)
    return {
        "p50":  float(np.percentile(a, 50)),
        "p95":  float(np.percentile(a, 95)),
        "p99":  float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "n":    int(len(values)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace",  type=Path, required=True)
    p.add_argument("--router", type=str,  default="http://127.0.0.1:8000")
    p.add_argument("--rate",   type=float, default=None,
                   help="Target arrival rate (req/s). Required unless "
                        "--closed-loop.")
    p.add_argument("--out",    type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--warmup", type=float, default=0.0,
                   help="Drop requests whose planned arrival_s < warmup (s).")
    p.add_argument("--warmup-requests", type=int, default=0,
                   help="Request-count warmup: open-loop drops the N "
                        "earliest arrivals; closed-loop excludes the "
                        "first N completions and measures --duration "
                        "seconds AFTER the Nth completion.")
    p.add_argument("--max-concurrent", type=int, default=1024)
    p.add_argument("--http-timeout",   type=float, default=600.0)
    p.add_argument("--seed",   type=int, default=0,
                   help="Seed for the Poisson inter-arrival sequence.")
    p.add_argument("--limit", type=int, default=None,
                   help="Fire only the first K trace requests "
                        "(fixed-duration methodology: K = rate × duration).")
    p.add_argument("--origin-filter", type=int, default=None,
                   help="Only fire requests with this origin_region — one "
                        "per-region replayer per router in the real "
                        "multi-region deployment. Arrival times stay on "
                        "the global schedule so 3 filtered replayers "
                        "jointly reproduce the global Poisson process.")
    p.add_argument("--closed-loop", action="store_true",
                   help="Closed-loop saturation-throughput mode "
                        "(eval-grid row 4): --clients concurrent clients "
                        "loop over the trace for --duration seconds.")
    p.add_argument("--clients",  type=int,   default=64,
                   help="Closed-loop concurrent clients (default 64).")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Closed-loop duration in seconds (default 60).")
    p.add_argument("--summary-out", type=Path, default=None,
                   help="Also write the summary JSON here.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.closed_loop:
        summary = asyncio.run(closed_loop(
            trace_path   = args.trace,
            router_url   = args.router.rstrip("/"),
            n_clients    = args.clients,
            duration_s   = args.duration,
            out_csv      = args.out,
            warmup_s     = args.warmup,
            http_timeout = args.http_timeout,
            seed         = args.seed,
            warmup_n     = args.warmup_requests,
        ))
    else:
        if args.rate is None:
            raise SystemExit("--rate is required unless --closed-loop")
        summary = asyncio.run(replay(
            trace_path     = args.trace,
            router_url     = args.router.rstrip("/"),
            rate_rps       = args.rate,
            out_csv        = args.out,
            warmup_s       = args.warmup,
            max_concurrent = args.max_concurrent,
            http_timeout   = args.http_timeout,
            seed           = args.seed,
            origin_filter  = args.origin_filter,
            limit          = args.limit,
            warmup_n       = args.warmup_requests,
        ))
    print()
    print("=== replay summary ===")
    print(json.dumps(summary, indent=2))
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2)
    return 0 if summary["n_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
