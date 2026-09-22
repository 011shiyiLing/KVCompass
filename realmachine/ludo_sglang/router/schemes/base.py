"""
Routing-scheme abstract base class.

Every scheme implements the same minimal interface so the router can
hot-swap them via the ``SCHEME`` env var without any branching in the
request path:

    scheme = SCHEME_REGISTRY[config.SCHEME](N_DP_WORKERS)
    load   = await load_tracker.snapshot()
    decision = await scheme.route(token_ids, load)
    # ... dispatch to SGLang worker decision.rank ...
    scheme.advertise(token_ids, decision.rank)

A ``RouteDecision`` carries everything the router needs to (a) pick a
worker, (b) pass to the LoadTracker for reserve / release, and (c)
record the per-request CSV row.

``extra_lookup_ms`` is an *intentionally injected* sleep that models
the planet-scale cost a baseline would pay in a real geo-distributed
deployment. Computed in ``ludo_sglang.config`` from the WAN matrix
(CloudPing snapshot) — has full provenance, not a free parameter.

``WorkerLoad`` is an immutable snapshot of every worker's pending-ms
and per-token prefill rate, so schemes can do load-aware scoring (Ludo)
or load-aware tie-breaking (Radix, CentralDir) directly in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ludo_sglang.router.load_tracker import WorkerLoad


@dataclass
class RouteDecision:
    rank: int                             # selected worker index (0..N-1)
    matched_tokens: int = 0               # routing-table-claimed prefix-match length
    extra_lookup_ms: float = 0.0          # total injected sleep (lookup + WAN)
    score_ms: float = 0.0                 # estimated TTFT used for argmin (ms)
    wan_to_worker_ms: float = 0.0         # WAN component of extra_lookup_ms
    info: dict = field(default_factory=dict)   # scheme-specific debug info


class Scheme:
    """Stateful routing strategy. One instance per router lifetime.

    Schemes do not sleep — they compute a ``RouteDecision`` (including
    the total ``extra_lookup_ms`` they want injected) and the router
    issues exactly one ``await asyncio.sleep`` for the combined value
    before dispatching to SGLang. Keeping the sleep out of route() lets
    the router compose lookup overhead, WAN propagation, and any future
    overhead source uniformly across schemes."""

    name: str = "abstract"

    def __init__(self, n_workers: int):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers

    async def route(
        self,
        token_ids: list[int],
        load: WorkerLoad,
        origin_region: int,
        ctx: dict | None = None,
    ) -> RouteDecision:
        """Pick which worker should serve this request.

        ``token_ids``     : prompt token ids (router has already
                            tokenised it; schemes never tokenise).
        ``load``          : immutable snapshot of per-worker
                            pending_ms + prefill_rate (ms/token).
        ``origin_region`` : the request's origin region (in WAN matrix
                            index space) — used to compute WAN cost
                            from the origin to each candidate worker
                            and (for cache-aware schemes) to bias the
                            scoring toward geographically close workers.
        ``ctx``           : optional per-request context from the router
                            (e.g. ``{"swk_hop": 1}`` marks a request
                            already forwarded once by a peer SkyWalker
                            router, so it must be served locally).
                            Schemes that don't need it ignore it.
        """
        raise NotImplementedError

    def advertise(self, token_ids: list[int], rank: int) -> None:
        """Notify the scheme that ``rank`` has just been selected to
        serve ``token_ids``. Schemes that maintain a routing table
        (Ludo, Radix, CentralDirectory) update it here. Schemes that
        route purely from the input (ConsistentHash / DHT) no-op."""
        return

    def stats(self) -> dict:
        """Return scheme-specific counters for ``GET /scheme_stats``.
        Default: only the scheme name."""
        return {"scheme": self.name}
