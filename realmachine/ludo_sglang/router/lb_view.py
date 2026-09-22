"""
Cross-region load view — the real-machine port of the sim's
``LudoDiscovery._lb_view / lb_probe / lb_view_queue`` dissemination model.

Topology: one router per region.  Every request dispatched to a worker
passes through exactly one router, so a worker's true queue is the sum of
the per-router pending each router charged it.  Each router therefore

  * sees its OWN contribution live (``LoadTracker.pending_ms``), and
  * learns the other routers' contributions through a periodic load
    report (``LUDO_LB_BCAST_MS``, default 100 ms — the same cadence as
    the sim's suppressed broadcast and SkyWalker's heartbeat), folded
    with an EWMA (``LUDO_LB_EWMA_BETA``) so one-off bursts are smoothed
    while persistent backlog is kept.

``viewed_pending_ms`` = own live pending + Σ_peers EWMA(reported pending).
For a single-router deployment (no peers) this degrades to the live
pending vector — bit-identical to the legacy behaviour.

Two delivery channels, mirroring the sim exactly:

* **Piggyback (data path)** — a Ludo cross-region dispatch is relayed
  through the serving region's router (``/internal/ludo_relay``); the
  relay response carries that router's live pending vector, folded here
  via ``apply_report(..., piggyback=True)``.  Busy (S,X) pairs refresh
  at their own traffic rate.
* **Suppressed periodic broadcast (control path)** — every
  ``LUDO_LB_BCAST_MS`` the broadcaster reports to each peer *unless* it
  piggybacked to that peer within the last period (sender-side
  suppression: the serving router marks ``note_sent(origin)`` on every
  relay it answers, so suppression needs no coordination).

Both channels carry the same payload (this router's pending-contribution
vector), so the fold logic is identical and the total view stays exact
when every contribution is fresh.
"""
from __future__ import annotations

import asyncio
import time

import httpx

from ludo_sglang import config


class RegionLoadView:
    """EWMA view of the pending-ms other routers have charged each worker."""

    def __init__(self, n_workers: int, beta: float | None = None):
        self.n = n_workers
        self.beta = config.LUDO_LB_EWMA_BETA if beta is None else beta
        # peer router_id -> EWMA-folded pending vector (ms)
        self._peer_pending: dict[int, list[float]] = {}
        self._peer_fresh: dict[int, float] = {}
        # peer router_id -> when we last piggybacked a report TO it
        # (drives sender-side broadcast suppression).
        self._sent_recent: dict[int, float] = {}
        self.stats = {"reports_rx": 0, "reports_tx": 0, "tx_errors": 0,
                      "piggyback_rx": 0, "piggyback_tx": 0,
                      "bcast_suppressed": 0}

    def apply_report(self, from_router_id: int, pending_ms: list[float],
                     piggyback: bool = False) -> None:
        if len(pending_ms) != self.n:
            raise ValueError(
                f"lb report from router {from_router_id} has "
                f"{len(pending_ms)} workers, expected {self.n}")
        prev = self._peer_pending.get(from_router_id)
        if prev is None:
            self._peer_pending[from_router_id] = [float(p) for p in pending_ms]
        else:
            b = self.beta
            self._peer_pending[from_router_id] = [
                b * float(p) + (1.0 - b) * q for p, q in zip(pending_ms, prev)
            ]
        self._peer_fresh[from_router_id] = time.monotonic()
        self.stats["piggyback_rx" if piggyback else "reports_rx"] += 1

    def note_sent(self, peer_router_id: int) -> None:
        """A report just rode to `peer_router_id` on the data path —
        the next periodic broadcast to it inside the period is redundant."""
        if peer_router_id < 0:
            return
        self._sent_recent[peer_router_id] = time.monotonic()
        self.stats["piggyback_tx"] += 1

    def recently_sent(self, peer_router_id: int, period_s: float) -> bool:
        return (time.monotonic()
                - self._sent_recent.get(peer_router_id, float("-inf"))
                < period_s)

    def viewed_pending_ms(self, own_pending_ms: list[float]) -> list[float]:
        """Own live pending + every peer's EWMA-reported pending."""
        out = [float(p) for p in own_pending_ms]
        for vec in self._peer_pending.values():
            for r in range(self.n):
                out[r] += vec[r]
        return out

    def snapshot(self) -> dict:
        return {
            "peers":       sorted(self._peer_pending),
            "peer_pending": {k: [round(v, 1) for v in vec]
                             for k, vec in self._peer_pending.items()},
            **self.stats,
        }


async def broadcaster(view: RegionLoadView, load_tracker,
                      peer_urls: list[str], router_id: int) -> None:
    """Background task: every LUDO_LB_BCAST_MS, report this router's own
    live pending vector to every peer router's /internal/lb_report —
    sender-side suppressed for peers a piggyback already refreshed within
    the period (the sim's control-path channel)."""
    period_s = config.LUDO_LB_BCAST_MS / 1000.0
    # peer URL -> router id (== region id in our deployment) for the
    # suppression bookkeeping; unmapped peers are never suppressed.
    url_to_id = {u.rstrip("/"): rid
                 for rid, u in config.REGION_ROUTER_URLS.items()}
    async with httpx.AsyncClient(timeout=max(0.5, period_s * 5)) as client:
        while True:
            body = {
                "from_router_id": router_id,
                "pending_ms": list(load_tracker.pending_ms),
            }
            for url in peer_urls:
                peer_id = url_to_id.get(url.rstrip("/"), -1)
                if peer_id >= 0 and view.recently_sent(peer_id, period_s):
                    view.stats["bcast_suppressed"] += 1
                    continue
                try:
                    await client.post(f"{url}/internal/lb_report", json=body)
                    view.stats["reports_tx"] += 1
                except Exception:
                    view.stats["tx_errors"] += 1
            await asyncio.sleep(period_s)
