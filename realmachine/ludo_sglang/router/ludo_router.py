"""
Ludo Router — scheme-pluggable FastAPI dispatcher in front of an
SGLang fleet.

Each request flow:
  1) Tokenise the prompt (no chat template — see ``_tokenize`` below).
  2) Ask the active ``Scheme`` for a routing decision (worker rank,
     claimed-prefix-match length, optional injected lookup overhead).
  3) Increment ``pending_tokens[rank]`` by the residual work estimate
     (prompt_len − matched_tokens) so subsequent decisions see the
     load.
  4) Forward the request to that worker's
     ``/v1/chat/completions`` endpoint over HTTP, streaming.
  5) Decrement ``pending_tokens[rank]`` and call ``scheme.advertise()``
     so future requests see this worker as a candidate for the same
     prefix.

Endpoints
---------
POST /generate     — main inference path; returns one record dict.
GET  /health       — fleet liveness (per-worker /health probe).
GET  /scheme_stats — stats from the currently active Scheme.
GET  /config       — effective config dump (for sanity-checking env
                     overrides at experiment time).

Launch
------
  SCHEME=ludo \\
  SGLANG_WORKER_URLS="http://127.0.0.1:30000,http://127.0.0.1:30001" \\
  python -m ludo_sglang.router.ludo_router

The router is single-process; ``pending_tokens`` and the Scheme's
internal state are protected only by the Python GIL. The router is
deliberately stateless across runs — restart it between schemes /
percentiles to start each measurement from a clean routing-table state.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make the project root importable when run as a script (``python ludo_router.py``)
# in addition to being run as a module.
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

from ludo_sglang import config
from ludo_sglang.client.sglang_client import async_send_request
from ludo_sglang.router.load_tracker import LoadTracker
from ludo_sglang.router.schemes import SCHEME_REGISTRY, Scheme


# ── Tokeniser ───────────────────────────────────────────────────────────────

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    return _tokenizer


def _tokenize(prompt: str) -> list[int]:
    """Tokenise the templated prompt so the router's lookup keys live
    in the same key space the SGLang worker advertises into.

    Central-mode rationale: the worker's HiRadixCache stores tokens
    AFTER ``apply_chat_template`` (SGLang tokenises the templated text
    we send it), so its ``cache_finished_req`` advertises tokens that
    include the template prefix. To match those keys we must template
    here too — otherwise the discovery plane gets a key-space mismatch
    and ``matched_tokens`` stays zero forever.

    The chat-template prefix shifts every request's tokens by the same
    constant prefix length (~13 tokens for Qwen2.5-Instruct), so
    chunk-size levels ≥ 256 are still informative: two requests sharing
    a long document prefix still match at level-0+ once the document
    overlap exceeds (chunk_size − prefix_len).
    """
    tok = _get_tokenizer()
    templated = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    return tok.encode(templated, add_special_tokens=False)


def _apply_chat_template(prompt: str) -> str:
    """Apply the model's chat template to wrap the user prompt.

    SGLang's OpenAI-compat endpoint does this internally, but we use
    the native ``/generate`` endpoint (which takes raw text) so we can
    receive ``meta_info.cached_tokens`` per request. The native
    endpoint requires us to format the prompt ourselves.

    Returns the templated prompt text — already includes the
    assistant-response-start marker, so SGLang can begin generating
    immediately.
    """
    tok = _get_tokenizer()
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ── App state ───────────────────────────────────────────────────────────────

scheme: Scheme | None = None
load_tracker: LoadTracker | None = None
gossip_manager = None  # type: Optional["GossipManager"]
lb_view = None          # type: Optional["RegionLoadView"]
_lb_broadcast_task: Optional["asyncio.Task"] = None
_forward_client = None  # type: Optional["httpx.AsyncClient"]
# Central-mode topology (filled in at startup when MAINTENANCE_MODE=='central')
_topology_refresh_task: Optional["asyncio.Task"] = None
_central_client = None  # type: Optional["httpx.AsyncClient"]
app = FastAPI(title=f"Ludo Router [{config.SCHEME}/{config.MAINTENANCE_MODE}]")


def _measure_chat_template_prefix_tokens() -> int:
    """How many tokens the chat template prepends before user content.

    SGLang's response ``cached_tokens`` counts templated tokens (chat
    template prefix + user prompt). The routing-plane's claimed match
    is keyed on RAW user-prompt tokens. To compare them on the same
    axis (for cache_belief EMA), subtract this offset from
    cached_tokens. We measure it empirically by templating a
    one-character prompt and counting the templated tokens minus 1.
    """
    tok = _get_tokenizer()
    try:
        templated = tok.apply_chat_template(
            [{"role": "user", "content": "X"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n_total = len(tok.encode(templated, add_special_tokens=False))
        return max(0, n_total - 1)
    except Exception:
        # Conservative fallback: 0. Worst case the belief is slightly
        # generous on tiny matches, never wrongly punitive.
        return 0


async def _relay_dispatch(relay_url: str, rank: int,
                          formatted_prompt: str,
                          max_new_tokens: int) -> dict:
    """Dispatch a Ludo cross-region request via the serving region's
    router and fold the piggybacked load stamp into the local view.

    Falls back to a direct worker dispatch if the relay is unreachable —
    correctness never depends on the data-path channel."""
    global _forward_client
    if _forward_client is None:
        _forward_client = httpx.AsyncClient(timeout=config.HTTP_TIMEOUT)
    try:
        resp = await _forward_client.post(f"{relay_url}/internal/ludo_relay", json={
            "rank":             rank,
            "formatted_prompt": formatted_prompt,
            "max_new_tokens":   max_new_tokens,
            "origin_router_id": config.ROUTER_ID,
        })
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return await async_send_request(
            worker_url       = config.SGLANG_WORKER_URLS[rank],
            formatted_prompt = formatted_prompt,
            max_new_tokens   = max_new_tokens,
            http_timeout     = config.HTTP_TIMEOUT,
        )
    stamp = payload.get("lb_stamp")
    if stamp and lb_view is not None:
        try:
            lb_view.apply_report(int(stamp["from_router_id"]),
                                 [float(x) for x in stamp["pending_ms"]],
                                 piggyback=True)
        except (KeyError, ValueError, TypeError):
            pass
    return payload["result"]


async def _migrate_prefix(dst_rank: int, token_slice: list[int]) -> None:
    """Explicit KV migration executor (prompt-shipping + re-prefill).

    Ships the hot prefix to ``dst_rank`` as a prefill-only request so the
    replica lands in that worker's RadixCache, then advertises it.  The
    occupancy is charged to the load tracker for its duration so routing
    sees the background prefill.  Runs off the request critical path.
    """
    tok = _get_tokenizer()
    text = tok.decode(token_slice, skip_special_tokens=False)
    est_ms, _qw = await load_tracker.reserve(dst_rank, len(token_slice))
    try:
        await async_send_request(
            worker_url       = config.SGLANG_WORKER_URLS[dst_rank],
            formatted_prompt = text,
            max_new_tokens   = 1,
            http_timeout     = config.HTTP_TIMEOUT,
        )
        from src.ludo.ludo_table import TIER_VRAM
        scheme.advertise(token_slice, dst_rank, tier=TIER_VRAM)
    finally:
        await load_tracker.release(dst_rank, est_ms, 0, 0.0, None)


@app.on_event("startup")
async def _startup():
    global scheme, load_tracker, gossip_manager, lb_view, _lb_broadcast_task
    scheme_cls = SCHEME_REGISTRY[config.SCHEME]
    scheme = scheme_cls(config.N_DP_WORKERS)
    template_prefix = _measure_chat_template_prefix_tokens()
    load_tracker = LoadTracker(
        n_workers              = config.N_DP_WORKERS,
        prior_ms_per_token     = config.C_PREFILL_MS_PER_TOKEN_PRIOR,
        ema_alpha              = config.PREFILL_RATE_EMA_ALPHA,
        min_samples            = config.PREFILL_RATE_MIN_SAMPLES,
        belief_alpha           = getattr(config, "CACHE_BELIEF_EMA_ALPHA", 0.2),
        template_prefix_tokens = template_prefix,
    )
    print(config.summary())
    print(f"[router] http://{config.ROUTER_HOST}:{config.ROUTER_PORT}  "
          f"scheme={scheme.name}  workers={config.N_DP_WORKERS}  "
          f"router_id={config.ROUTER_ID}")
    print(f"[router] prefill prior={config.C_PREFILL_MS_PER_TOKEN_PRIOR} ms/tok  "
          f"EMA α={config.PREFILL_RATE_EMA_ALPHA}  "
          f"min_samples={config.PREFILL_RATE_MIN_SAMPLES}")
    print(f"[router] cache_belief EMA α="
          f"{getattr(config, 'CACHE_BELIEF_EMA_ALPHA', 0.2)}  "
          f"chat_template_prefix={template_prefix} tokens")
    print(f"[router] worker→region map: "
          + ", ".join(f"w{i}->r{config.WORKER_REGIONS[i]}"
                       for i in range(config.N_DP_WORKERS)))
    for i, url in enumerate(config.SGLANG_WORKER_URLS):
        print(f"[router]   worker[{i}] -> {url}")
    print(f"[router] WAN mode: "
          f"{'EMULATED (injected sleeps)' if config.WAN_EMULATION else 'REAL (no injection)'}"
          f"  router_region={config.ROUTER_REGION}")

    # ── Cross-region load view (multi-router deployments) ────────────
    # Peer routers exchange their live pending vectors every
    # LUDO_LB_BCAST_MS; snapshots then carry the viewed queue that the
    # new-design joint score (Ludo) and Dynamo score against.
    if config.PEER_ROUTER_URLS:
        from ludo_sglang.router.lb_view import RegionLoadView, broadcaster
        lb_view = RegionLoadView(config.N_DP_WORKERS)
        load_tracker.attach_view(lb_view)
        _lb_broadcast_task = asyncio.create_task(broadcaster(
            lb_view, load_tracker, config.PEER_ROUTER_URLS, config.ROUTER_ID))
        print(f"[router] lb view: broadcasting every "
              f"{config.LUDO_LB_BCAST_MS:.0f} ms to "
              f"{len(config.PEER_ROUTER_URLS)} peer(s)")

    # ── Explicit KV-migration executor (Ludo only) ────────────────────
    if hasattr(scheme, "attach_migrator"):
        scheme.attach_migrator(_migrate_prefix)
        print(f"[router] migration: "
              f"{'ENABLED' if config.MIG_ENABLE else 'disabled'} "
              f"(hot_k={config.MIG_HOT_K}, window={config.MIG_WINDOW_MS:.0f}ms, "
              f"cooldown={config.MIG_COOLDOWN_MS:.0f}ms)")

    # ── Maintenance plane: gossip vs central directory ───────────────
    if config.MAINTENANCE_MODE == "central":
        # Central-directory mode: this router is a Lookup; register
        # with the NameServer so authorities can find us, then skip the
        # gossip layer entirely.  Authority pushes arrive on
        # ``/internal/apply_packed`` (defined later).
        await _register_as_lookup()
        global _topology_refresh_task
        _topology_refresh_task = asyncio.create_task(_periodic_topology_refresh())
        print(f"[router] central-mode lookup #{config.LOOKUP_ID}, "
              f"name-server={config.NAMESERVER_URL}, "
              f"authorities={len(config.AUTHORITY_URLS)}")
    elif config.PEER_ROUTER_URLS and hasattr(scheme, "table") and hasattr(
        scheme, "attach_gossip"
    ):
        # Gossip mode (legacy): only spawned when the scheme has a
        # LudoTable AND at least one peer is configured.  Other
        # schemes (Radix, DHT, etc.) don't replicate state across
        # routers in this fleet model.
        from ludo_sglang.router.gossip import GossipManager
        gossip_manager = GossipManager(
            table        = scheme.table,
            peer_urls    = config.PEER_ROUTER_URLS,
            router_id    = config.ROUTER_ID,
            period_s     = config.GOSSIP_PERIOD_S,
            http_timeout_s = config.GOSSIP_HTTP_TIMEOUT_S,
            max_batch    = config.GOSSIP_MAX_BATCH,
        )
        scheme.attach_gossip(gossip_manager)
        gossip_manager.start()
        print(f"[router] gossip enabled: {len(config.PEER_ROUTER_URLS)} peer(s), "
              f"period={config.GOSSIP_PERIOD_S}s")
        for url in config.PEER_ROUTER_URLS:
            print(f"[router]   peer → {url}")
    else:
        print("[router] discovery-plane replication disabled "
              "(gossip mode without peers, single edge)")


async def _register_as_lookup():
    """Register this lookup with the NameServer, retrying on failure.

    Idempotent on the NameServer side — re-registering with the same
    ``lookup_id`` updates ``addr`` instead of duplicating.  This makes
    a router restart transparent to the authorities.
    """
    global _central_client
    import httpx
    if _central_client is None:
        _central_client = httpx.AsyncClient(
            timeout=config.MAINTENANCE_HTTP_TIMEOUT_S,
        )
    own_addr = f"http://{config.ROUTER_HOST}:{config.ROUTER_PORT}"
    for attempt in range(60):
        try:
            await _central_client.post(
                f"{config.NAMESERVER_URL}/register/lookup",
                json={"addr": own_addr, "lookup_id": config.LOOKUP_ID},
            )
            print(f"[router] registered as lookup#{config.LOOKUP_ID} "
                  f"@ {own_addr}")
            return
        except Exception as exc:
            await asyncio.sleep(1.0)


async def _periodic_topology_refresh():
    """Pull /topology every 30s.  In v1 the lookup doesn't actually
    need authority URLs (authority pushes to us, not the other way around),
    but we still pull so /scheme/stats can include topology info."""
    global _central_client
    import httpx
    if _central_client is None:
        _central_client = httpx.AsyncClient(
            timeout=config.MAINTENANCE_HTTP_TIMEOUT_S,
        )
    while True:
        try:
            await _central_client.get(f"{config.NAMESERVER_URL}/topology")
        except Exception:
            pass
        await asyncio.sleep(30.0)


@app.on_event("shutdown")
async def _shutdown():
    global gossip_manager, _topology_refresh_task, _central_client
    global _lb_broadcast_task, _forward_client
    if _lb_broadcast_task is not None:
        _lb_broadcast_task.cancel()
        try:
            await _lb_broadcast_task
        except (asyncio.CancelledError, Exception):
            pass
        _lb_broadcast_task = None
    if _forward_client is not None:
        await _forward_client.aclose()
        _forward_client = None
    if gossip_manager is not None:
        await gossip_manager.stop()
        gossip_manager = None
    if _topology_refresh_task is not None:
        _topology_refresh_task.cancel()
        try:
            await _topology_refresh_task
        except (asyncio.CancelledError, Exception):
            pass
        _topology_refresh_task = None
    if _central_client is not None:
        await _central_client.aclose()
        _central_client = None


# ── Main request handler ────────────────────────────────────────────────────

@app.post("/generate")
async def generate(req: dict):
    prompt          = req.get("prompt", "")
    max_new_tokens  = int(req.get("max_new_tokens", config.MAX_NEW_TOKENS))
    request_id      = req.get("request_id", "")
    swk_hop         = int(req.get("swk_hop", 0))
    # origin_region travels with each request so the WAN penalty is
    # reproducible from the trace.  If the client doesn't set it:
    #   real mode      → this router's own region (clients talk to
    #                    their nearest router, like a DNS-routed LB);
    #   emulation mode → derive deterministically from req_id (legacy,
    #                    uniform across regions).
    if "origin_region" in req and req["origin_region"] is not None:
        origin_region = int(req["origin_region"])
    elif not config.WAN_EMULATION:
        origin_region = config.ROUTER_REGION
    else:
        try:
            origin_region = config.request_origin_region(int(request_id))
        except (TypeError, ValueError):
            origin_region = config.request_origin_region(0)
    if not (0 <= origin_region < config.NUM_REGIONS_LOGICAL):
        raise HTTPException(400, f"origin_region out of range: {origin_region}")

    token_ids = _tokenize(prompt)
    if not token_ids:
        raise HTTPException(400, "Empty prompt after tokenisation")
    prompt_len = len(token_ids)

    # 1) snapshot worker load + service rate — schemes score off this.
    load = await load_tracker.snapshot()
    decision = await scheme.route(token_ids, load, origin_region,
                                  ctx={"swk_hop": swk_hop})

    # 1b) SkyWalker two-layer forwarding: the scheme decided the request
    #     should be handled by another region's router (all local workers
    #     full).  Hand the raw request over; the remote router makes the
    #     final replica pick with its own fresh local state.  swk_hop=1
    #     stops a second forward.
    forward_url = decision.info.get("forward_url") if decision.info else None
    if forward_url and swk_hop == 0:
        global _forward_client
        if _forward_client is None:
            _forward_client = httpx.AsyncClient(timeout=config.HTTP_TIMEOUT)
        try:
            resp = await _forward_client.post(f"{forward_url}/generate", json={
                "prompt":         prompt,
                "max_new_tokens": max_new_tokens,
                "request_id":     request_id,
                "origin_region":  origin_region,
                "swk_hop":        1,
            })
            resp.raise_for_status()
            payload = resp.json()
            payload["forwarded_via"] = config.ROUTER_REGION
            return payload
        except Exception as exc:
            # Remote LB unreachable (real WAN blip) → the paper's LB
            # queues the request locally instead of failing it.  Fall
            # through to a normal dispatch on the least-loaded local
            # worker.  repr(): several httpx exceptions stringify to ''.
            local = [r for r in range(config.N_DP_WORKERS)
                     if config.WORKER_REGIONS[r] == config.ROUTER_REGION]
            if local:
                decision.rank = min(local, key=lambda r: load.pending_ms[r])
                decision.matched_tokens = 0
                decision.info["forward_failed"] = repr(exc)[:120]
            else:
                raise HTTPException(
                    502, f"forward to {forward_url} failed: {exc!r}")

    rank = decision.rank
    residual = max(0, prompt_len - decision.matched_tokens)

    # 2) inject the scheme's combined overhead (lookup walk + WAN
    #    propagation to the chosen worker) as one sleep. Schemes do
    #    not sleep themselves — the router owns this so it can also
    #    compose WAN uniformly across schemes.
    #    Real multi-region mode: the network itself delivers the WAN
    #    propagation (the worker is physically remote), so only the
    #    modelled *lookup* overhead is injected — injecting the WAN
    #    term again would double-count it.
    inject_ms = decision.extra_lookup_ms
    if not config.WAN_EMULATION:
        inject_ms -= decision.wan_to_worker_ms
    if inject_ms > 0:
        await asyncio.sleep(inject_ms / 1000.0)

    # 3) reserve estimated work on the chosen worker; remember the
    #    queue-wait-at-dispatch so we can infer the true prefill rate
    #    from the observed TTFT after the request completes.
    est_ms, queue_wait_at_dispatch = await load_tracker.reserve(rank, residual)

    # 4) dispatch to SGLang's native /generate, with the chat template
    #    applied so SGLang sees the same token sequence its RadixCache
    #    keys on. We use /generate (not /v1/chat/completions) to get
    #    cached_tokens back from meta_info.
    #
    #    Ludo cross-region dispatches in real multi-router mode travel
    #    THROUGH the serving region's router (/internal/ludo_relay) —
    #    the sim's data path.  The relay response piggybacks that
    #    router's live pending vector, which we fold into the load view
    #    below: busy region pairs refresh at their own traffic rate, and
    #    the periodic broadcast is suppressed for them (lb_view.py).
    #    Decision, reservation and advertise all stay HERE at the origin
    #    router, exactly as in the sim's model.
    formatted_prompt = _apply_chat_template(prompt)
    relay_url = None
    if (scheme.name == "ludo" and config.LUDO_PIGGYBACK
            and not config.WAN_EMULATION):
        w_region = config.WORKER_REGIONS[rank]
        if w_region != config.ROUTER_REGION:
            relay_url = config.REGION_ROUTER_URLS.get(w_region)
    try:
        if relay_url is not None:
            result = await _relay_dispatch(relay_url, rank,
                                           formatted_prompt, max_new_tokens)
        else:
            result = await async_send_request(
                worker_url       = config.SGLANG_WORKER_URLS[rank],
                formatted_prompt = formatted_prompt,
                max_new_tokens   = max_new_tokens,
                http_timeout     = config.HTTP_TIMEOUT,
            )
    except Exception as exc:
        # Release reservation without folding the failure into the EMA.
        await load_tracker.release(rank, est_ms, residual,
                                   queue_wait_at_dispatch, None)
        raise HTTPException(502, f"SGLang error: {exc}")

    # 4) release reservation + EMA-update prefill rate from observed TTFT.
    inferred_prefill_rate = await load_tracker.release(
        rank, est_ms, residual, queue_wait_at_dispatch, result["ttft_ms"],
    )

    # 4b) per-worker cache_belief EMA. Compares the routing-plane's
    #     claim (decision.matched_tokens) against SGLang's actual
    #     cache delivery (result.cached_tokens − template_prefix).
    #     Used only by Ludo's joint scoring, but updated for every
    #     scheme so the LoadTracker stays scheme-agnostic.
    belief_obs = await load_tracker.update_belief(
        rank,
        decision.matched_tokens,
        result.get("cached_tokens"),
    )

    # Discovery-plane advertise.
    #
    # Gossip mode: the router writes its own LudoTable here (the
    # advertise call also queues a delta for the gossip layer).  The
    # initial tier is VRAM because the worker just finished prefill
    # and freshly-computed KV lives on the GPU.
    #
    # Central mode: the worker emits notify_advertise() to the authority
    # itself (see ludo_sglang/worker/tier_notifier.py); the router
    # does NOT write the table here.  The authority will broadcast the
    # resulting (level, key, packed) deltas back to this lookup
    # asynchronously via /internal/apply_packed.
    if config.MAINTENANCE_MODE != "central":
        try:
            from src.ludo.ludo_table import TIER_VRAM
            scheme.advertise(token_ids, rank, tier=TIER_VRAM)
        except TypeError:
            # Non-Ludo schemes don't accept the tier kwarg.
            scheme.advertise(token_ids, rank)

    # 5) return one row of per-request data — this is what the replayer
    #    persists into the per-(scheme, rate) CSV.
    return {
        "request_id":               request_id,
        "scheme":                   scheme.name,
        "dp_rank":                  rank,
        "origin_region":            origin_region,
        "worker_region":            config.WORKER_REGIONS[rank],
        "matched_tokens":           decision.matched_tokens,
        "residual_tokens":          residual,
        # What was actually injected on top of wall-clock.  In real WAN
        # mode the network delivers propagation physically, so this is
        # the modelled lookup overhead only — downstream "effective
        # TTFT = ttft + router_overhead" stays double-count-free.
        "router_overhead_ms":       max(0.0, inject_ms),
        "wan_to_worker_ms":         decision.wan_to_worker_ms,
        "score_ms":                 decision.score_ms,
        "queue_wait_estimate_ms":   queue_wait_at_dispatch,
        "reserved_ms":              est_ms,
        "inferred_prefill_rate":    inferred_prefill_rate,
        "ttft_ms":                  result["ttft_ms"],
        "e2e_ms":                   result["e2e_ms"],
        "prompt_tokens":            result["prompt_tokens"],
        "completion_tokens":        result["completion_tokens"],
        "cached_tokens":            result["cached_tokens"],
        "belief_obs":               belief_obs,
        "output_text":              result["text"],
        # decision.info is scheme-specific debug payload (e.g. the
        # per-worker max_match_per_worker for Ludo, owners_count for
        # CentralDirectory). Useful for ad-hoc analysis but never
        # required by the canonical CSV columns above.
        "scheme_info":              decision.info,
    }


# ── Diagnostics ─────────────────────────────────────────────────────────────

@app.get("/scheme_stats")
async def scheme_stats():
    return {
        "scheme": scheme.stats(),
        "load":   load_tracker.stats(),
        "gossip": gossip_manager.stats() if gossip_manager else {"enabled": False},
    }


@app.get("/config")
async def get_config():
    return {
        "scheme":              config.SCHEME,
        "wan_percentile":      config.WAN_PERCENTILE,
        "wan_avg_ms":          config.WAN_AVG_MS,
        "wan_max_ms":          config.WAN_MAX_MS,
        "num_nodes_logical":   config.NUM_NODES_LOGICAL,
        "n_workers":           config.N_DP_WORKERS,
        "model":               config.MODEL_NAME,
        "ludo_chunk_sizes":    config.LUDO_CHUNK_SIZES,
        "chord_overhead_ms":     config.CHORD_LOOKUP_OVERHEAD_MS,
        "kademlia_overhead_ms":  config.KADEMLIA_LOOKUP_OVERHEAD_MS,
        "directory_rtt_ms":      config.DIRECTORY_LOOKUP_OVERHEAD_MS,
        "directory_service_ms":  config.DIRECTORY_SERVICE_MS,
    }


@app.get("/alive")
async def alive():
    """Lightweight liveness probe — proves the router process is up
    and serving HTTP. Does NOT touch the SGLang fleet, so it always
    responds in well under 1 second. Use this for boot-readiness
    checks (``run_experiment.sh:wait_router_up``)."""
    return {"ok": True, "scheme": scheme.name if scheme else None}


@app.post("/internal/gossip_apply")
async def gossip_apply(req: dict):
    """
    Inbound gossip from a peer router.  Each delta in the batch is
    LWW-merged into the local LudoTable.

    Expected JSON body:
        {
          "from_router_id": int,
          "deltas": [
            [key, worker_idx, tier, epoch],
            ...
          ]
        }

    Idempotent: replaying the same batch is safe (LWW filters duplicates).
    Returns counts: {"received": N, "applied": M, "stale": N-M}.
    """
    if gossip_manager is None:
        # Gossip not configured on this router; reject so peer learns.
        raise HTTPException(404, "gossip not enabled on this router")
    src_id = int(req.get("from_router_id", -1))
    if src_id < 0 or src_id == config.ROUTER_ID:
        raise HTTPException(
            400, f"invalid from_router_id={src_id} "
                 f"(self={config.ROUTER_ID})",
        )
    deltas = req.get("deltas") or []
    return await gossip_manager.apply_batch(deltas, src_id)


@app.post("/internal/debug_advertise")
async def debug_advertise(req: dict):
    """P1 propagation harness: inject an advertise WITHOUT running a
    request.  Body: {"token_ids": [...], "rank": int, "tier": int(opt)}.
    Returns the router's monotonic clock (ms) after the local index +
    gossip enqueue completed."""
    from src.ludo.ludo_table import TIER_VRAM
    token_ids = req["token_ids"]
    rank = int(req["rank"])
    scheme.advertise(token_ids, rank, tier=int(req.get("tier", TIER_VRAM)))
    return {"t_mono_ms": time.monotonic() * 1000.0}


@app.post("/internal/debug_lookup")
async def debug_lookup(req: dict):
    """P1 propagation harness: is `token_ids[:cut]` visible in THIS
    router's view?  Body: {"token_ids": [...], "cut": int}.
    Returns holders and the router's monotonic clock (ms)."""
    from src.ludo.ludo_table import chunk_to_key
    token_ids = req["token_ids"]
    cut = int(req.get("cut", len(token_ids)))
    key = chunk_to_key(token_ids[:cut])
    holders = scheme._holders_at(cut, key) if hasattr(scheme, "_holders_at") \
        else []
    return {"holders": holders, "t_mono_ms": time.monotonic() * 1000.0}


@app.post("/internal/lb_report")
async def lb_report(req: dict):
    """Peer router's periodic load report (see router/lb_view.py).

    Body: {"from_router_id": int, "pending_ms": [float × n_workers]}.
    EWMA-folded into this router's cross-region load view.
    """
    if lb_view is None:
        raise HTTPException(404, "lb view not enabled (no PEER_ROUTER_URLS)")
    src = int(req.get("from_router_id", -1))
    if src < 0 or src == config.ROUTER_ID:
        raise HTTPException(400, f"invalid from_router_id={src}")
    try:
        lb_view.apply_report(src, [float(x) for x in req.get("pending_ms", [])])
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


@app.post("/internal/ludo_relay")
async def ludo_relay(req: dict):
    """Ludo cross-region data path (the sim's piggyback carrier).

    A peer router chose a worker in THIS region; we forward the prompt to
    that local worker and stamp the response with this router's live
    pending vector.  The origin router keeps the reservation, EMA updates
    and advertise — this endpoint is a thin relay, not a re-decision
    (SkyWalker's /generate forwarding re-decides; Ludo's relay must not,
    matching the sim where the origin LB's argmin is final).
    """
    rank = int(req.get("rank", -1))
    if not (0 <= rank < config.N_DP_WORKERS):
        raise HTTPException(400, f"rank out of range: {rank}")
    if config.WORKER_REGIONS[rank] != config.ROUTER_REGION:
        raise HTTPException(400,
            f"worker {rank} (region {config.WORKER_REGIONS[rank]}) is not "
            f"local to this router (region {config.ROUTER_REGION})")
    result = await async_send_request(
        worker_url       = config.SGLANG_WORKER_URLS[rank],
        formatted_prompt = req.get("formatted_prompt", ""),
        max_new_tokens   = int(req.get("max_new_tokens", config.MAX_NEW_TOKENS)),
        http_timeout     = config.HTTP_TIMEOUT,
    )
    stamp = None
    if lb_view is not None and load_tracker is not None:
        stamp = {
            "from_router_id": config.ROUTER_ID,
            "pending_ms":     list(load_tracker.pending_ms),
        }
        # Sender-side suppression: this response IS a load report to the
        # origin — skip the next periodic broadcast to it inside the period.
        lb_view.note_sent(int(req.get("origin_router_id", -1)))
    return {"result": result, "lb_stamp": stamp}


@app.post("/internal/dynamo_lookup")
async def dynamo_lookup(req: dict):
    """Dynamo cross-cluster fan-out ingress: answer with this cluster's
    local cache holders for the given content keys.

    Body: {"keys": {"<chunk_size>": <uint64 key>, ...}}
    Reply: {"candidates": [[rank, match_len, tier], ...]}
    """
    if scheme is None or not hasattr(scheme, "peer_lookup"):
        return {"candidates": [], "noop_for_scheme": getattr(scheme, "name", None)}
    keys = {int(cs): int(k) for cs, k in (req.get("keys") or {}).items()}
    return {"candidates": scheme.peer_lookup(keys)}


@app.post("/internal/dynamo_advertise")
async def dynamo_advertise(req: dict):
    """Dynamo index feed: a peer router dispatched a request to a worker
    in THIS region; record the new holder in this cluster's index (the
    real-machine analogue of the worker's KV event feeding its own
    cluster's KvIndexer)."""
    if scheme is None or not hasattr(scheme, "apply_remote_advertise"):
        return {"ok": True, "noop_for_scheme": getattr(scheme, "name", None)}
    keys = {int(cs): int(k) for cs, k in (req.get("keys") or {}).items()}
    scheme.apply_remote_advertise(int(req.get("rank", -1)),
                                  int(req.get("tier", 0)), keys)
    return {"ok": True}


@app.post("/internal/tier_event")
async def tier_event(req: dict):
    """
    Path-B tier-notification endpoint — called by an SGLang-side hook
    when the worker's RadixCache moves a chunk between tiers.

    Expected JSON body:
        {
          "rank":      int,                  # which worker emitted this
          "token_ids": list[int],            # the chunk's full token sequence
                                             #   (router re-hashes per level)
          "event":     "demote" | "promote" | "evict",
        }

    Mapping to LudoScheme calls:
        demote  → notify_tier(token_ids, rank, TIER_DRAM)
        promote → notify_tier(token_ids, rank, TIER_VRAM)
        evict   → notify_evict(token_ids, rank)

    No-ops for non-Ludo schemes (those have no tier concept).
    """
    if scheme is None or not hasattr(scheme, "notify_tier"):
        return {"ok": True, "noop_for_scheme": getattr(scheme, "name", None)}

    rank      = int(req.get("rank", -1))
    token_ids = req.get("token_ids") or []
    event     = (req.get("event") or "").lower()

    if rank < 0 or rank >= config.N_DP_WORKERS:
        raise HTTPException(400, f"rank out of range: {rank}")
    if not token_ids:
        raise HTTPException(400, "token_ids empty")
    if event not in ("demote", "promote", "evict"):
        raise HTTPException(400, f"unknown event: {event!r}")

    # Lazy import — keep tier constants out of the hot request path.
    from src.ludo.ludo_table import TIER_VRAM, TIER_DRAM

    if event == "demote":
        scheme.notify_tier(token_ids, rank, TIER_DRAM)
    elif event == "promote":
        scheme.notify_tier(token_ids, rank, TIER_VRAM)
    elif event == "evict":
        scheme.notify_evict(token_ids, rank)

    return {"ok": True, "event": event, "rank": rank,
            "scheme": scheme.name}


@app.post("/internal/topology_push")
async def topology_push(req: dict):
    """NameServer fans the freshly-computed topology to every node when
    `ready` first flips True. The router doesn't need to act on the
    `lookup_for_shard` partition itself (that's authority concern) — we
    just ack so the broadcast doesn't 404."""
    return {"ok": True, "lookup_id": config.LOOKUP_ID,
            "n_lookups": len(req.get("lookups", []) or []),
            "ready": bool(req.get("ready"))}


@app.post("/internal/apply_delta_batch")
async def apply_delta_batch(req: dict):
    """
    Central-mode ingress: a discovery authority pushed a batched
    ``DeltaBatchMsg``.  Each delta is a Smash-style wire payload that
    DataPlaneLudoTable.apply_delta dispatches on ``kind``:

      * ``insert``  → DP.apply_insert(path, value)
      * ``update``  → DP.apply_update(bs, value)
      * ``fallback`` / ``remove_fallback`` → mirror authority's fallback dict
      * ``snapshot`` → DP.deserialize(snapshot_bytes); local fallback cleared

    Each wire delta has ``level`` set so the router routes to the right
    ``tables[level]`` mirror.  Idempotent for snapshot but NOT for
    insert/update — the authority serialises broadcasts under ``send_lock``
    so re-delivery is not expected.
    """
    if config.MAINTENANCE_MODE != "central":
        raise HTTPException(404, "central-mode endpoint not enabled")
    if scheme is None or not hasattr(scheme, "apply_delta_batch"):
        raise HTTPException(
            500,
            f"scheme {getattr(scheme, 'name', '?')!r} does not support "
            "apply_delta_batch (only LudoScheme does in v1)",
        )
    deltas = req.get("deltas") or []
    return scheme.apply_delta_batch(deltas)


@app.get("/health")
async def health():
    """Full health check: probes each SGLang worker. Slower (up to
    n_workers × 5 s under network failure) — for diagnostics, not
    boot-readiness."""
    workers: list[dict] = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for i, url in enumerate(config.SGLANG_WORKER_URLS):
            try:
                r = await client.get(f"{url}/health")
                ok = r.status_code == 200
            except Exception:
                ok = False
            workers.append({"rank": i, "url": url, "ok": ok})
    return {"ok": True,
            "workers_ok": all(w["ok"] for w in workers),
            "workers":    workers,
            "scheme":     scheme.name if scheme else None,
            "load":       load_tracker.stats() if load_tracker else None}


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app,
                host=config.ROUTER_HOST,
                port=config.ROUTER_PORT,
                log_level="info")
