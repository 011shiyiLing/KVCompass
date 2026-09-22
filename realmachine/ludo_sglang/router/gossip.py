"""
gossip.py — multi-edge eventual-consistency replication for the Ludo
discovery plane.

Problem
-------
Each region runs its own router process; they all serve the same
inference fleet but each holds a private LudoTable.  When router A
advertises a chunk on its local edge, router B's table doesn't know
about it — so B's lookups miss prefixes that A has already seen.

This module replicates the LudoTable across all peers via push-based
gossip with bounded staleness ``GOSSIP_PERIOD_S``:

  1. Every local advertise / notify_tier / notify_evict produces a
     delta tuple ``(key, worker_idx, tier, epoch)`` in the LudoTable's
     LWW format and enqueues it in this manager's outbound buffer.

  2. A background asyncio task fires every ``GOSSIP_PERIOD_S``: it
     swaps the buffer, dedups entries that targeted the same
     ``(key, worker_idx)`` (keeping the highest epoch), and POSTs the
     batch to each peer's ``/internal/gossip_apply`` endpoint.

  3. The receiving peer's ``apply_batch`` calls ``LudoTable.apply_remote``
     on each delta.  LWW + Lamport-clock semantics inside LudoTable
     handle out-of-order delivery, duplicates, and concurrent writes
     across routers (see ``ludo_table.py:apply_remote``).

Convergence model
-----------------
``Δ_gossip = GOSSIP_PERIOD_S + max(WAN propagation to any peer)`` —
beyond this window every peer has either applied the delta or
explicitly LWW-rejected it as stale.  No quorum, no leader, no
linearizable read — the discovery plane is a bounded-staleness eventual
consistency layer on top of LudoTable, exactly the trade-off the design
doc commits to.

Failure tolerance
-----------------
* Peer down → POST raises, error is logged but swallowed; the buffered
  delta is *lost* for that peer.  We don't re-queue: under the LWW
  model the next advertise on the same prefix will carry a higher
  epoch and supersede whatever the down peer eventually sees.
* Router crash → in-memory buffer lost.  Same recovery as above:
  natural advertise traffic re-warms the table.
* Bootstrap → a freshly-started router receives no historical state.
  Future extension: add ``GET /internal/gossip_snapshot`` that returns
  the full LRU shadow + slot epochs so a new router can pull-bootstrap.
  Not implemented — relies on natural convergence in the workshop
  experiment.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable, List, Optional, Tuple

import httpx

from src.ludo.ludo_table import LudoTable

_log = logging.getLogger("ludo.gossip")

# Outbound delta type — same shape LudoTable.insert returns:
#   (key, worker_idx, tier, epoch)
DeltaTuple = Tuple[int, int, int, int]


class GossipManager:
    """Outbound + inbound gossip coordinator.

    Lifecycle::

        mgr = GossipManager(table, peer_urls, router_id, period_s)
        mgr.start()                      # spawns the background push task
        ...
        mgr.enqueue(delta)               # called by every local write
        ...
        await mgr.apply_batch(deltas, from_router_id)
                                         # called by /internal/gossip_apply
        ...
        await mgr.stop()                 # cancels the task, flushes once
    """

    def __init__(
        self,
        table: LudoTable,
        peer_urls: List[str],
        router_id: int,
        period_s: float = 1.0,
        http_timeout_s: float = 1.0,
        max_batch: int = 1000,
    ):
        if router_id < 0:
            raise ValueError(f"router_id must be ≥ 0, got {router_id}")
        if period_s <= 0:
            raise ValueError(f"period_s must be > 0, got {period_s}")
        self._table = table
        self._peers: List[str] = [u.rstrip("/") for u in peer_urls if u]
        self._router_id: int = int(router_id)
        self._period_s: float = float(period_s)
        self._timeout_s: float = float(http_timeout_s)
        self._max_batch: int = int(max_batch)

        # Outbound buffer — list of DeltaTuple appended on every local write.
        # We swap-and-clear at gossip time to minimise lock-hold time.
        self._buffer: List[DeltaTuple] = []
        self._buffer_lock = asyncio.Lock()

        # HTTP client (lazy-built on first push so we tie to the running loop).
        self._client: Optional[httpx.AsyncClient] = None

        self._task: Optional[asyncio.Task] = None
        self._stopped: bool = False

        # Stats
        self._pushes_attempted: int = 0
        self._pushes_succeeded: int = 0
        self._pushes_failed: int = 0
        self._deltas_sent: int = 0
        self._deltas_received: int = 0
        self._deltas_applied_in: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True iff at least one peer is configured."""
        return bool(self._peers)

    def enqueue(self, delta: Optional[DeltaTuple]) -> None:
        """
        Add a delta to the outbound buffer.  ``None`` is allowed (no-op)
        so callers can pass through ``LudoTable.insert``'s return value
        unchecked.

        This call is sync (the buffer is a plain list), since it sits
        on every request's hot path.  We accept the tiny race against
        the gossip task's swap — at worst we miss this delta in the
        current round and ship it on the next one.
        """
        if not self.enabled or delta is None:
            return
        # Append is atomic in CPython under GIL; no lock needed for hot path.
        self._buffer.append(delta)

    async def apply_batch(
        self, deltas: Iterable[List[int]], from_router_id: int,
    ) -> dict:
        """
        Apply an inbound gossip batch from peer ``from_router_id`` to
        the local LudoTable.  Each delta is a 4-element list
        ``[key, worker_idx, tier, epoch]`` (JSON serialisation flattens
        the tuple).

        Returns counts: ``{"received": N, "applied": M, "stale": N-M}``.
        """
        applied = 0
        received = 0
        for d in deltas:
            received += 1
            if len(d) != 4:
                continue
            key, worker_idx, tier, epoch = d
            ok = self._table.apply_remote(
                int(key), int(worker_idx), int(tier), int(epoch),
                int(from_router_id),
            )
            if ok:
                applied += 1
        self._deltas_received += received
        self._deltas_applied_in += applied
        return {
            "received": received,
            "applied":  applied,
            "stale":    received - applied,
            "router_id": self._router_id,
        }

    def start(self) -> None:
        """Launch the background push task on the current event loop."""
        if not self.enabled:
            _log.info("gossip disabled (no peers configured)")
            return
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._push_loop())
        _log.info(
            "gossip started: router_id=%d, %d peer(s), period=%.2fs",
            self._router_id, len(self._peers), self._period_s,
        )

    async def stop(self) -> None:
        """Cancel the push task and flush the buffer one last time."""
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        # One last flush so anything queued just before shutdown gets out.
        try:
            await self._flush_once()
        except Exception:
            pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def stats(self) -> dict:
        return {
            "enabled":           self.enabled,
            "router_id":         self._router_id,
            "peers":             list(self._peers),
            "period_s":          self._period_s,
            "buffer_size":       len(self._buffer),
            "pushes_attempted":  self._pushes_attempted,
            "pushes_succeeded":  self._pushes_succeeded,
            "pushes_failed":     self._pushes_failed,
            "deltas_sent":       self._deltas_sent,
            "deltas_received":   self._deltas_received,
            "deltas_applied_in": self._deltas_applied_in,
        }

    # ── Internal: periodic push loop ─────────────────────────────────────────

    async def _push_loop(self) -> None:
        try:
            while not self._stopped:
                await asyncio.sleep(self._period_s)
                if self._stopped:
                    break
                try:
                    await self._flush_once()
                except Exception as exc:                        # noqa: BLE001
                    _log.warning("gossip push iteration failed: %s", exc)
        except asyncio.CancelledError:
            pass

    async def _flush_once(self) -> None:
        """Take a snapshot of the buffer and broadcast it to all peers."""
        async with self._buffer_lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []

        # Dedup within the batch: keep highest epoch per (key, worker_idx).
        # Multiple writes to the same slot in one window are common when
        # a request hits multiple chunk-size levels, but only the latest
        # tier/epoch matters for downstream peers.
        dedup: dict[Tuple[int, int], DeltaTuple] = {}
        for delta in batch:
            key, worker_idx, tier, epoch = delta
            slot_key = (key, worker_idx)
            existing = dedup.get(slot_key)
            if existing is None or epoch > existing[3]:
                dedup[slot_key] = delta
        deduped = list(dedup.values())

        # Cap the per-push batch size; oldest deltas first overflow into
        # the next push round.  (Workshop scale rarely hits this.)
        if len(deduped) > self._max_batch:
            # Sort by epoch ascending; keep the LATEST max_batch.
            deduped.sort(key=lambda d: d[3])
            overflow = deduped[: -self._max_batch]
            deduped = deduped[-self._max_batch :]
            # Push the overflow back onto the buffer for the next round.
            async with self._buffer_lock:
                self._buffer = overflow + self._buffer

        if not deduped:
            return

        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout_s,
                limits=httpx.Limits(
                    max_connections=2 * len(self._peers),
                    max_keepalive_connections=len(self._peers),
                ),
            )

        # Fire all peer POSTs in parallel.
        payload = {
            "from_router_id": self._router_id,
            "deltas": [list(d) for d in deduped],
        }
        await asyncio.gather(
            *(self._post_to_peer(peer, payload) for peer in self._peers),
            return_exceptions=True,
        )
        self._deltas_sent += len(deduped)

    async def _post_to_peer(self, peer_url: str, payload: dict) -> None:
        self._pushes_attempted += 1
        url = peer_url + "/internal/gossip_apply"
        try:
            assert self._client is not None
            r = await self._client.post(url, json=payload)
            if r.status_code == 200:
                self._pushes_succeeded += 1
            else:
                self._pushes_failed += 1
                _log.warning("gossip POST %s → %d %s",
                             url, r.status_code, r.text[:128])
        except Exception as exc:                                # noqa: BLE001
            # Peer down / network blip / timeout — swallow.  LWW means
            # the next advertise will supersede what we lost.
            self._pushes_failed += 1
            _log.debug("gossip POST %s failed: %s", url, exc)
