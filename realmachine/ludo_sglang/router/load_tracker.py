"""
Per-worker load and service-rate tracking for the router.

The router needs two things to drive Ludo's joint scoring:
  (1) ``pending_ms[r]`` — milliseconds of work worker ``r`` still owes
      across all in-flight requests.
  (2) ``prefill_rate[r]`` — milliseconds per residual token, updated
      online from observed TTFTs (EMA).

Putting both behind a single ``LoadTracker`` lets the router reserve /
release work atomically and lets the scheme see a consistent immutable
snapshot for scoring.

Time accounting per request lifecycle:

    ┌── route() ────┐              ┌── SGLang call ──┐         ┌── update ──┐
    │ snapshot()    │              │                 │         │ release()  │
    │ scheme picks  │  reserve()   │  await async_   │ ttft_ms │ EMA on     │
    │ rank          │ ──────────►  │  send_request   │ ──────► │ prefill_rt │
    └───────────────┘              └─────────────────┘         └────────────┘
       reads          adds est_ms          measures             subtracts est_ms,
       pending_ms     to pending_ms        observed             folds inferred
       prefill_rate                        TTFT                 prefill rate

All operations are guarded by a single ``asyncio.Lock`` — order of
events stays well-defined under concurrent requests.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerLoad:
    """Immutable snapshot of all workers' load + service rate.

    Schemes receive this in ``route()`` and read by index. Because it is
    a frozen snapshot, scheme code can score in any order without
    worrying about mutations from concurrent requests.

    ``cache_belief[r]`` ∈ [0, 1] is the EMA of (real-cached / claimed-
    matched) for worker ``r``. Cache-aware schemes (Ludo) multiply
    their claimed match by belief in the joint score so workers whose
    routing-plane claims have historically not been honoured by
    SGLang's RadixCache lose the score discount they'd otherwise get
    from a "hit" — preventing the table-vs-RadixCache staleness from
    funnelling traffic onto an over-committed worker.
    """
    pending_ms:   list[float]   # estimated ms each worker still owes (this router)
    prefill_rate: list[float]   # ms/token, EMA per worker
    cache_belief: list[float]   # ∈ [0, 1], 1.0 = full trust in claims
    # Viewed queue (ms) per worker: own live pending + peer routers'
    # EWMA-reported pending (see router/lb_view.py).  Equal to
    # ``pending_ms`` in a single-router deployment.  This is the T_q_view
    # the new-design joint score (Ludo) and Dynamo score against.
    pending_view_ms: list[float] = None  # type: ignore[assignment]
    # In-flight request count per worker (reserve() .. release() window).
    # SkyWalker's selective pushing uses this as its "pending requests"
    # availability signal.
    inflight: list[int] = None  # type: ignore[assignment]

    @property
    def n_workers(self) -> int:
        return len(self.pending_ms)


class LoadTracker:
    """Mutable owner of ``pending_ms`` + ``prefill_rate`` + ``cache_belief``."""

    def __init__(
        self,
        n_workers: int,
        prior_ms_per_token: float,
        ema_alpha: float = 0.2,
        min_samples: int = 5,
        belief_alpha: float = 0.2,
        template_prefix_tokens: int = 0,
    ):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if not (0.0 < belief_alpha <= 1.0):
            raise ValueError(f"belief_alpha must be in (0, 1], got {belief_alpha}")
        self.n            = n_workers
        self.pending_ms   = [0.0] * n_workers
        self.prefill_rate = [prior_ms_per_token] * n_workers
        self.cache_belief = [1.0] * n_workers   # start fully trusting; learn down
        self.inflight     = [0] * n_workers
        self.n_samples    = [0] * n_workers
        self.belief_n     = [0] * n_workers
        # Optional RegionLoadView (multi-router deployments); when set,
        # snapshot() folds peer-reported pending into pending_view_ms.
        self.lb_view      = None
        self.prior        = prior_ms_per_token
        self.alpha        = ema_alpha
        self.belief_alpha = belief_alpha
        # SGLang's cached_tokens counts the templated prompt (chat
        # template prefix + user content). To turn it into a "fraction
        # of the claimed user-prompt match that was real", subtract
        # template_prefix_tokens before dividing by matched_tokens.
        self.template_prefix_tokens = max(0, int(template_prefix_tokens))
        self.min_samples  = min_samples
        self._lock        = asyncio.Lock()

    def attach_view(self, lb_view) -> None:
        """Wire in a RegionLoadView so snapshots carry the cross-router
        viewed queue (multi-router real deployments)."""
        self.lb_view = lb_view

    async def snapshot(self) -> WorkerLoad:
        async with self._lock:
            pending = list(self.pending_ms)
            if self.lb_view is not None:
                pending_view = self.lb_view.viewed_pending_ms(pending)
            else:
                pending_view = list(pending)
            return WorkerLoad(
                pending_ms      = pending,
                prefill_rate    = list(self.prefill_rate),
                cache_belief    = list(self.cache_belief),
                pending_view_ms = pending_view,
                inflight        = list(self.inflight),
            )

    async def reserve(self, rank: int, residual_tokens: int) -> tuple[float, float]:
        """Charge worker ``rank`` for an incoming request.

        Returns ``(est_ms, queue_wait_at_dispatch_ms)`` — the caller
        must pass both back to ``release()`` so the EMA update can
        infer the true prefill rate (= observed TTFT − queue wait).
        """
        residual_tokens = max(0, residual_tokens)
        async with self._lock:
            queue_wait = self.pending_ms[rank]
            est_ms = residual_tokens * self.prefill_rate[rank]
            self.pending_ms[rank] += est_ms
            self.inflight[rank] += 1
            return est_ms, queue_wait

    async def release(
        self,
        rank: int,
        est_ms: float,
        residual_tokens: int,
        queue_wait_at_dispatch_ms: float,
        observed_ttft_ms: float | None,
    ) -> float | None:
        """Subtract reservation; fold observation into the EMA.

        Returns the inferred prefill rate from this observation
        (ms/token), or ``None`` if no signal could be extracted.
        """
        async with self._lock:
            self.pending_ms[rank] = max(0.0, self.pending_ms[rank] - est_ms)
            self.inflight[rank] = max(0, self.inflight[rank] - 1)

            if observed_ttft_ms is None or residual_tokens <= 0:
                return None
            inferred_prefill_ms = observed_ttft_ms - queue_wait_at_dispatch_ms
            if inferred_prefill_ms <= 0:
                # Either the observed TTFT was below our queue estimate
                # (overestimate of queue wait, or noise), or the worker
                # served entirely from cache. Either way, no clean
                # signal for the prefill rate — skip the EMA update.
                return None
            rate_obs = inferred_prefill_ms / residual_tokens

            # Sanity clip: sim's calibrated prefill rate for ~7-8B BF16
            # dense models on A100 is 0.05 ms/tok; real measurements
            # (Qwen2.5-7B / Llama-3.1-8B) sit within ±10× of this. The
            # clip here defends the EMA against bursty-arrival
            # measurement noise — when observed_ttft - queue_estimate
            # diverges from residual × true_rate (e.g., reservation
            # under-credits a concurrent burst), the implied rate can
            # spike. Capping at 0.5 ms/tok keeps "miss penalty" within
            # the same order of magnitude as sim's joint-score formula
            # uses; without this cap the score collapses to "always
            # pick a hit" and the joint-scoring balance is lost.
            rate_obs = max(0.005, min(0.5, rate_obs))

            self.prefill_rate[rank] = (
                (1.0 - self.alpha) * self.prefill_rate[rank]
                + self.alpha       * rate_obs
            )
            self.n_samples[rank] += 1
            return rate_obs

    async def update_belief(
        self,
        rank: int,
        matched_tokens: int,
        cached_tokens: int | None,
    ) -> float | None:
        """Fold one (claimed, actually-cached) observation into the
        per-worker cache_belief EMA.

        Returns the per-request observation in [0, 1] (the EMA target),
        or None if the request carried no claim (matched_tokens == 0).

        The observation is::

            obs = clip( max(0, cached - template_prefix) / matched, 0, 1 )

        Subtracting the chat-template prefix length is what makes the
        observation comparable to the routing-plane claim, which is
        keyed on raw user-prompt tokens. ``cached`` from SGLang counts
        templated tokens (template prefix + user content), so without
        the subtraction every cached=template_prefix-only response
        (true user-prompt cache miss) would falsely inflate belief.
        """
        if matched_tokens <= 0:
            return None
        if cached_tokens is None:
            cached_tokens = 0
        async with self._lock:
            user_cached = max(0, cached_tokens - self.template_prefix_tokens)
            obs = user_cached / matched_tokens
            if obs > 1.0:
                obs = 1.0
            elif obs < 0.0:
                obs = 0.0
            import os
            if os.environ.get("LUDO_FORCE_BELIEF_ONE", "0") == "1":
                self.cache_belief[rank] = 1.0
            else:
                self.cache_belief[rank] = (
                    (1.0 - self.belief_alpha) * self.cache_belief[rank]
                    + self.belief_alpha       * obs
                )
            self.belief_n[rank] += 1
            return obs

    def stats(self) -> dict:
        return {
            "pending_ms":   list(self.pending_ms),
            "inflight":     list(self.inflight),
            "lb_view":      self.lb_view.snapshot() if self.lb_view else None,
            "prefill_rate": list(self.prefill_rate),
            "cache_belief": list(self.cache_belief),
            "n_samples":    list(self.n_samples),
            "belief_n":     list(self.belief_n),
            "prior_ms_per_token":     self.prior,
            "ema_alpha":              self.alpha,
            "belief_alpha":           self.belief_alpha,
            "template_prefix_tokens": self.template_prefix_tokens,
        }
