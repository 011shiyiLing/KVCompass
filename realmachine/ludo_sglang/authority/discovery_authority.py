"""
discovery_authority.py — single-shard authoritative writer.

Architecture
============
Every region's edge router (Lookup) holds a read-only mirror of this
authority's state.  When a worker emits an advertise or tier event, it
hashes the prefix key and POSTs to the authority that owns the shard;
the authority updates its L per-level :class:`AuthorityLudoTable` instances
and broadcasts the resulting (level, key, packed_value) to every
registered lookup.

Public HTTP surface
-------------------
``POST /advertise``
    Worker → authority.  Worker just finished prefill; record VRAM tier
    for every chunk-size level the prompt fits.
``POST /tier_event``
    Worker → authority.  ``event ∈ {demote, promote, evict}``.
``POST /internal/leave``
    NameServer → authority.  A worker's heartbeat timed out; clear all
    its slots in every entry.
``GET  /stats``
    Operational stats.
``GET  /topology_pull``
    Authority → NameServer (internally) called once at startup; exposed
    here for debug.

Concurrency
-----------
Per-shard writes are serialised by ``self.update_lock``.  The lock is
held only for the slot-update primitive itself; broadcast HTTP POSTs
to lookups are issued *after* releasing the lock so a slow lookup
cannot back-pressure other writes.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

from ludo_sglang import config
from ludo_sglang.authority import messages as msg
from ludo_sglang.authority.persistence import AuthorityPersistence
from src.ludo.authority_table import (
    AuthorityLudoTable, Delta, ApplyResult,
    TIER_VRAM, TIER_DRAM, TIER_NONE,
    chunk_to_key, tier_name,
)


# Map worker event names to (tier, accept_predicate). The predicate guards
# against state-machine violations (e.g. a "promote" arriving before any
# "advertise" — the prior tier is irrelevant; we just always set the slot
# to the event-implied tier).
_EVENT_TO_TIER = {
    "advertise": TIER_VRAM,
    "promote":   TIER_VRAM,
    "demote":    TIER_DRAM,
    "evict":     TIER_NONE,
}

_log = logging.getLogger("ludo.authority")


# ────────────────────────────────────────────────────────────────────────
# DiscoveryAuthority
# ────────────────────────────────────────────────────────────────────────
class DiscoveryAuthority:
    """One per shard.

    Holds:
      * ``self.tables[level]`` — :class:`AuthorityLudoTable`, one per
        chunk-size level (so a single advertise touches L tables).
      * ``self.last_seq[rank]`` — per-worker monotonic sequence
        de-duplication; a tier event with seq ≤ last_seq is dropped
        as out-of-order or duplicate.
      * ``self.lookup_addrs`` — list of edge router URLs to broadcast
        to.  Refreshed lazily from the NameServer when missing.
    """

    def __init__(
        self,
        shard_id: int,
        n_shards: int,
        n_workers: int,
        chunk_sizes: list[int],
        capacity: int,
        persistence_dir: Optional[str] = None,
    ):
        self.shard_id = shard_id
        self.n_shards = n_shards
        self.n_workers = n_workers
        self.chunk_sizes = list(chunk_sizes)

        worker_addrs = [str(i) for i in range(n_workers)]
        # Per-shard capacity: each shard's CP holds at most 1/N of the
        # global keyspace, matching Smash's `cap = sumCap / nShards`
        # (deps/smash/Smash/master_fn.h:166). Floor at 64 so tiny test
        # deployments still bootstrap a valid cuckoo table.
        per_shard_capacity = max(64, capacity // max(1, n_shards))
        self.tables: list[AuthorityLudoTable] = [
            AuthorityLudoTable(
                worker_addrs=worker_addrs,
                expected_entries=per_shard_capacity,
                max_entries=per_shard_capacity,
                shard_id=shard_id,
                n_shards=n_shards,
            )
            for _ in self.chunk_sizes
        ]

        # Per-worker monotonic seq.  Survives heartbeat-loss/restart
        # because a restarted worker resets its counter to 0; that's
        # accepted, the authority will see seq drop and start tracking
        # from there.  We treat 0 as "no prior writes" (any seq>=1 is
        # accepted on first sight).
        self.last_seq: dict[int, int] = {}

        # Full lookup roster (populated from /topology), and the subset
        # this authority is responsible for broadcasting to (computed from
        # the name-server's `lookup_for_shard`).  Until the partition
        # arrives we fall back to broadcasting to every lookup so a
        # single-shard deployment still works.
        self.lookup_addrs: list[str] = []
        self.lookup_addrs_for_shard: list[str] = []

        self.update_lock = threading.RLock()
        # send_lock serialises broadcasts so a snapshot Delta (shipped by
        # the rebuild task) can never be re-ordered with a regular Insert/
        # Update Delta. Mirrors Smash's `sendLock` in master_fn.h.
        self.send_lock: Optional[asyncio.Lock] = None

        # Per-level rebuild tasks; we never run two concurrent rebuilds
        # for the same level (apply_slot_change has to flip building back
        # to False before another rebuild can be triggered).
        self._rebuild_tasks: list[Optional[asyncio.Task]] = [None] * len(self.chunk_sizes)

        self._client: Optional[httpx.AsyncClient] = None
        self._broadcasts_attempted = 0
        self._broadcasts_succeeded = 0
        self._broadcasts_failed = 0
        self._stale_seq_drops = 0
        self._cross_shard_drops = 0
        self._rebuilds_done = 0
        self._events_processed = 0
        self._bootstraps_sent = 0

        # ── Persistence ──────────────────────────────────────────────────
        # One append-log + snapshot per (shard, level).  Replay populates
        # the tables on startup; every successful apply appends; the log
        # rotates into a snapshot when it exceeds log_max_bytes.
        self.persistence: list[Optional[AuthorityPersistence]] = [None] * len(self.chunk_sizes)
        if persistence_dir:
            from pathlib import Path as _P
            for level in range(len(self.chunk_sizes)):
                p = AuthorityPersistence(_P(persistence_dir), shard_id=shard_id, level=level)
                self.persistence[level] = p
                table = self.tables[level]
                replayed = p.replay(table.apply_packed)
                if replayed:
                    print(f"[authority#{shard_id}] level {level}: "
                          f"replayed {replayed} record(s) from persistence")

    # ── Sharding ────────────────────────────────────────────────────────────
    def owns_key(self, key: int) -> bool:
        return (key % self.n_shards) == self.shard_id

    # ── Worker → authority event ingress ───────────────────────────────────────
    async def on_events_batch(
        self, rank: int, events: list[dict],
    ) -> dict:
        """Process a batched set of worker events. Each event is
        ``{"event": str, "token_ids": list[int], "seq": int}``. Returns a
        dict with apply / stale / cross_shard / rebuild stats."""
        if not (0 <= rank < self.n_workers):
            raise HTTPException(400, f"rank out of range: {rank}")
        deltas:        list[tuple[int, Delta]] = []   # (level, delta)
        levels_dirty:  set[int] = set()
        rebuild_levels: set[int] = set()
        applied = stale = cross_shard = bad = 0

        with self.update_lock:
            for ev in events:
                event = str(ev.get("event", "")).lower()
                if event not in _EVENT_TO_TIER:
                    bad += 1
                    continue
                seq = int(ev.get("seq", 0))
                if not self._check_seq(rank, seq):
                    stale += 1
                    continue
                tids = ev.get("token_ids") or []
                new_tier = _EVENT_TO_TIER[event]
                prompt_len = len(tids)

                # Fan-out worker path tags each event with the level it's
                # meant for: skip the iterate-all-levels loop and write
                # exactly that slot. Legacy worker path (no ``level`` key)
                # falls back to the old behaviour — each level's key is
                # tried against ``owns_key`` and the misses count as
                # cross_shard drops.
                if "level" in ev:
                    try:
                        levels_to_try = [int(ev["level"])]
                    except (TypeError, ValueError):
                        bad += 1
                        continue
                else:
                    levels_to_try = list(range(len(self.chunk_sizes)))

                for level in levels_to_try:
                    if not (0 <= level < len(self.chunk_sizes)):
                        bad += 1
                        continue
                    cs = self.chunk_sizes[level]
                    if cs > prompt_len:
                        # Legacy path: smaller levels can't write either,
                        # break out of the ordered iteration. Fan-out path:
                        # would mean the worker fanned a level beyond the
                        # prompt's reach — drop as bad.
                        if "level" in ev:
                            bad += 1
                            continue
                        break
                    key = chunk_to_key(tids[:cs])
                    if not self.owns_key(key):
                        cross_shard += 1
                        continue
                    result: ApplyResult = self.tables[level].apply_slot_change(
                        key, rank, new_tier,
                    )
                    if result.delta is not None:
                        deltas.append((level, result.delta))
                        applied += 1
                        p = self.persistence[level]
                        if p is not None:
                            p.append(key, result.new_packed)
                            levels_dirty.add(level)
                    if result.triggered_rebuild:
                        rebuild_levels.add(level)

        self._events_processed += len(events)
        self._cross_shard_drops += cross_shard

        # Snapshot rotate outside update_lock so dump's I/O doesn't block writers.
        for level in levels_dirty:
            p = self.persistence[level]
            if p is not None and p.needs_dump():
                p.dump(self.tables[level].iter_live())

        # Broadcast under send_lock so a concurrent rebuild's snapshot Delta
        # never gets reordered with a normal Insert/Update.
        if deltas:
            send_lock = self._send_lock_handle()
            async with send_lock:
                await self._broadcast_deltas(deltas)

        # Background rebuild for any level that just hit fallback.
        for level in rebuild_levels:
            self._maybe_spawn_rebuild(level)

        return {
            "ok":          True,
            "applied":     applied,
            "stale":       stale,
            "cross_shard": cross_shard,
            "bad":         bad,
            "rebuilt_levels": list(rebuild_levels),
        }

    # Compatibility wrappers — old single-event endpoints fan into the batch path.
    async def on_advertise(
        self, rank: int, token_ids: list[int], seq: int,
    ) -> dict:
        return await self.on_events_batch(rank, [{
            "event": "advertise", "token_ids": token_ids, "seq": seq,
        }])

    async def on_tier_event(
        self, rank: int, token_ids: list[int], event: str, seq: int,
    ) -> dict:
        if event not in msg.TIER_EVENT_NAMES:
            raise HTTPException(400, f"unknown event: {event!r}")
        return await self.on_events_batch(rank, [{
            "event": event, "token_ids": token_ids, "seq": seq,
        }])

    def _check_seq(self, rank: int, seq: int) -> bool:
        prev = self.last_seq.get(rank, -1)
        if seq <= prev:
            self._stale_seq_drops += 1
            return False
        self.last_seq[rank] = seq
        return True

    # ── Worker leave (NameServer-triggered) ─────────────────────────────────
    async def on_leave(self, worker_id: int) -> dict:
        """Clear ``worker_id``'s slot in every live entry of every level."""
        if not (0 <= worker_id < self.n_workers):
            raise HTTPException(400, f"worker_id out of range: {worker_id}")

        deltas: list[tuple[int, Delta]] = []
        levels_dirty: set[int] = set()
        rebuild_levels: set[int] = set()
        cleared = 0
        with self.update_lock:
            for level, table in enumerate(self.tables):
                for key, _packed in list(table.iter_live()):
                    result = table.apply_slot_change(key, worker_id, TIER_NONE)
                    if result.delta is not None:
                        cleared += 1
                        deltas.append((level, result.delta))
                        p = self.persistence[level]
                        if p is not None:
                            p.append(key, result.new_packed)
                            levels_dirty.add(level)
                    if result.triggered_rebuild:
                        rebuild_levels.add(level)
        for level in levels_dirty:
            p = self.persistence[level]
            if p is not None and p.needs_dump():
                p.dump(self.tables[level].iter_live())
        if deltas:
            send_lock = self._send_lock_handle()
            async with send_lock:
                await self._broadcast_deltas(deltas)
        for level in rebuild_levels:
            self._maybe_spawn_rebuild(level)
        print(f"[authority#{self.shard_id}] leave(worker={worker_id}) "
              f"cleared {cleared} slot(s) across {len(self.tables)} level(s)")
        return {"ok": True, "cleared": cleared, "rebuilt_levels": list(rebuild_levels)}

    # ── Rebuild orchestration (per level) ───────────────────────────────────
    def _maybe_spawn_rebuild(self, level: int) -> None:
        existing = self._rebuild_tasks[level]
        if existing is not None and not existing.done():
            return
        self._rebuild_tasks[level] = asyncio.create_task(self._rebuild_level(level))

    async def _rebuild_level(self, level: int) -> None:
        try:
            with self.update_lock:
                snapshot_delta = self.tables[level].merge_fallback_and_snapshot()
            send_lock = self._send_lock_handle()
            async with send_lock:
                await self._broadcast_deltas([(level, snapshot_delta)])
            p = self.persistence[level]
            if p is not None:
                p.dump(self.tables[level].iter_live())
            self._rebuilds_done += 1
            print(f"[authority#{self.shard_id}] level {level} rebuild OK; "
                  f"snapshot bytes={len(snapshot_delta.snapshot)}")
        except Exception as exc:
            _log.exception("rebuild failed for level %d: %s", level, exc)

    # ── Broadcast (Delta protocol) ──────────────────────────────────────────
    def _send_lock_handle(self) -> asyncio.Lock:
        if self.send_lock is None:
            self.send_lock = asyncio.Lock()
        return self.send_lock

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        n = max(1, len(self.lookup_addrs))
        # Long-lived client with generous keepalive — mirrors Smash's
        # `updateChannels = vector<int>` of persistent fds authority→lookup.
        self._client = httpx.AsyncClient(
            timeout=config.MAINTENANCE_HTTP_TIMEOUT_S,
            limits=httpx.Limits(
                max_connections=8 * n,
                max_keepalive_connections=4 * n,
                keepalive_expiry=300.0,
            ),
            http2=False,
        )

    async def _broadcast_deltas(
        self, level_deltas: list[tuple[int, Delta]],
        targets: Optional[list[str]] = None,
    ) -> None:
        """Broadcast Deltas to lookups. ``targets=None`` (default) sends to
        every lookup in the shard's mirror set; passing an explicit list
        is used by ``_send_bootstrap`` to fan out a snapshot to just the
        newly-joined edges without re-shipping it to everyone."""
        if targets is None:
            targets = self.lookup_addrs_for_shard or self.lookup_addrs
        if not targets:
            return
        self._ensure_client()
        wire: list[dict] = []
        for level, delta in level_deltas:
            payload = delta.to_wire()
            payload["level"] = level
            wire.append(payload)
        body = {"deltas": wire}
        await asyncio.gather(*[
            self._post_delta_batch(addr, body) for addr in targets
        ], return_exceptions=True)

    async def _send_bootstrap(self, target_addrs: list[str]) -> None:
        """One-time push of every level's current snapshot to a freshly-
        joined edge so its DataPlaneLudoTable mirrors are bootstrapped
        before incremental Insert/Update Deltas would land on an empty DP
        (which the lookup-side ``_dp_ready`` guard would otherwise drop).

        Snapshot is built under ``update_lock`` so no concurrent write
        can sneak in mid-build, then shipped under ``send_lock`` so any
        subsequent regular broadcast queues behind it — guaranteeing the
        new edge sees snapshot first, then the post-snapshot delta tail
        in apply order."""
        if not target_addrs:
            return
        deltas: list[tuple[int, Delta]] = []
        with self.update_lock:
            for level in range(len(self.tables)):
                snap = Delta(
                    kind="snapshot",
                    snapshot=self.tables[level].serialize_dp(),
                    shard_id=self.shard_id,
                )
                deltas.append((level, snap))
        async with self._send_lock_handle():
            await self._broadcast_deltas(deltas, targets=target_addrs)
        self._bootstraps_sent += len(target_addrs)
        print(f"[authority#{self.shard_id}] bootstrap snapshot fanned out to "
              f"{len(target_addrs)} new lookup(s): {target_addrs}")

    async def _post_delta_batch(self, addr: str, body: dict) -> None:
        self._broadcasts_attempted += 1
        url = f"{addr}/internal/apply_delta_batch"
        try:
            assert self._client is not None
            r = await self._client.post(url, json=body)
            if r.status_code == 200:
                self._broadcasts_succeeded += 1
            else:
                self._broadcasts_failed += 1
                _log.warning("delta broadcast %s → %d %s",
                             url, r.status_code, r.text[:128])
        except Exception as exc:
            self._broadcasts_failed += 1
            _log.debug("delta broadcast %s failed: %s", url, exc)

    # ── Stats ───────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "shard_id":              self.shard_id,
            "n_shards":              self.n_shards,
            "n_workers":             self.n_workers,
            "chunk_sizes":           list(self.chunk_sizes),
            "tables": [
                {**t.stats(), "level": i, "chunk_size": cs}
                for i, (t, cs) in enumerate(zip(self.tables, self.chunk_sizes))
            ],
            "lookup_addrs":          list(self.lookup_addrs),
            "lookup_addrs_for_shard": list(self.lookup_addrs_for_shard),
            "last_seq":              dict(self.last_seq),
            "broadcasts_attempted":  self._broadcasts_attempted,
            "broadcasts_succeeded":  self._broadcasts_succeeded,
            "broadcasts_failed":     self._broadcasts_failed,
            "stale_seq_drops":       self._stale_seq_drops,
            "cross_shard_drops":     self._cross_shard_drops,
            "events_processed":      self._events_processed,
            "rebuilds_done":         self._rebuilds_done,
            "bootstraps_sent":       self._bootstraps_sent,
            "rebuild_in_flight": [
                bool(t and not t.done()) for t in self._rebuild_tasks
            ],
            "persistence": [
                p.stats() if p is not None else None
                for p in self.persistence
            ],
        }


# ────────────────────────────────────────────────────────────────────────
# FastAPI app
# ────────────────────────────────────────────────────────────────────────
app = FastAPI(title=f"KVCompass DiscoveryAuthority #{config.SHARD_ID}")
authority: Optional[DiscoveryAuthority] = None


@app.on_event("startup")
async def _startup():
    global authority
    authority = DiscoveryAuthority(
        shard_id        = config.SHARD_ID,
        n_shards        = config.N_SHARDS,
        n_workers       = config.N_DP_WORKERS,
        chunk_sizes     = config.LUDO_CHUNK_SIZES,
        capacity        = config.LUDO_MAX_ENTRIES,
        persistence_dir = config.AUTHORITY_LOG_DIR,
    )
    print(f"[authority#{config.SHARD_ID}] starting, "
          f"n_shards={config.N_SHARDS}, n_workers={config.N_DP_WORKERS}, "
          f"L={len(config.LUDO_CHUNK_SIZES)} levels, "
          f"persistence_dir={config.AUTHORITY_LOG_DIR!r}")
    asyncio.create_task(_register_with_nameserver())
    asyncio.create_task(_periodic_topology_refresh())


@app.on_event("shutdown")
async def _shutdown():
    global authority
    if authority is None:
        return
    if authority._client is not None:
        await authority._client.aclose()
        authority._client = None
    for p in authority.persistence:
        if p is not None:
            try:
                p.flush()
                p.close()
            except Exception:
                pass


async def _register_with_nameserver():
    """Register self with the NameServer, then pull topology to find
    lookup addresses to broadcast to."""
    own_addr = _self_authority_addr()
    async with httpx.AsyncClient(timeout=config.MAINTENANCE_HTTP_TIMEOUT_S) as c:
        for attempt in range(60):
            try:
                await c.post(f"{config.NAMESERVER_URL}/register/authority",
                             json={"addr": own_addr,
                                   "shard_id": config.SHARD_ID})
                print(f"[authority#{config.SHARD_ID}] registered with name-server "
                      f"@ {config.NAMESERVER_URL}")
                return
            except Exception as exc:
                _log.warning("register attempt %d failed: %s", attempt, exc)
                await asyncio.sleep(1.0)


async def _periodic_topology_refresh():
    """Pull the topology once per second until the fleet is ready, then
    every 30s afterwards.  Updates ``authority.lookup_addrs`` AND
    ``authority.lookup_addrs_for_shard`` in place; the latter is what
    ``_broadcast`` actually targets."""
    interval = 1.0
    async with httpx.AsyncClient(timeout=config.MAINTENANCE_HTTP_TIMEOUT_S) as c:
        while True:
            try:
                r = await c.get(f"{config.NAMESERVER_URL}/topology")
                if r.status_code == 200 and authority is not None:
                    _ingest_topology(authority, r.json())
                    if r.json().get("ready"):
                        interval = 30.0
            except Exception as exc:
                _log.debug("topology refresh failed: %s", exc)
            await asyncio.sleep(interval)


def _ingest_topology(m: DiscoveryAuthority, body: dict) -> None:
    """Update authority's lookup roster and shard partition from a topology
    body (delivered either by polling /topology or by /internal/topology_push).

    Newly-joined lookups (present in the new subset but not the old) get
    a one-time bootstrap snapshot fan-out scheduled in the background
    so they don't have to wait for the next regular write to populate
    their DP mirror."""
    lookups = body.get("lookups", []) or []
    new_addrs = [l["addr"] for l in lookups]
    addr_changed = (new_addrs != m.lookup_addrs)
    m.lookup_addrs = new_addrs

    partition = body.get("lookup_for_shard") or []
    if 0 <= m.shard_id < len(partition):
        id_to_addr = {int(l["lookup_id"]): l["addr"] for l in lookups}
        new_subset = [
            id_to_addr[lid] for lid in partition[m.shard_id]
            if lid in id_to_addr
        ]
    else:
        # Partition not yet computed (fleet not ready) — fall back to
        # broadcasting to all known lookups so a single-shard deploy works.
        new_subset = list(new_addrs)

    old_subset = set(m.lookup_addrs_for_shard)
    newcomers = [a for a in new_subset if a not in old_subset]

    subset_changed = (new_subset != m.lookup_addrs_for_shard)
    m.lookup_addrs_for_shard = new_subset
    if addr_changed or subset_changed:
        print(f"[authority#{m.shard_id}] lookup roster: "
              f"{len(new_addrs)} known, {len(new_subset)} for this shard — "
              f"{new_subset}")

    if newcomers:
        # Schedule on the running loop; _ingest_topology is always called
        # from an async context (periodic refresh task or topology_push
        # HTTP handler), so get_running_loop is safe.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(m._send_bootstrap(newcomers))
        except RuntimeError:
            # No running loop (shouldn't happen in production paths) —
            # skip bootstrap; the newcomer will eventually be brought up
            # to date by the next authority rebuild snapshot or by anti-
            # entropy (缺口 2(b), not yet implemented).
            _log.warning("no running loop; skipping bootstrap for %s", newcomers)


def _self_authority_addr() -> str:
    # Authority URLs are configured in AUTHORITY_URLS; pick the entry that
    # matches our shard_id.  If that's not set (single-authority debug),
    # fall back to ROUTER_HOST:port.
    if config.SHARD_ID < len(config.AUTHORITY_URLS):
        return config.AUTHORITY_URLS[config.SHARD_ID].rstrip("/")
    # Default to AUTHORITY_PORT if env is set.
    import os
    port = int(os.environ.get("AUTHORITY_PORT", "8101"))
    return f"http://{config.ROUTER_HOST}:{port}"


# ── HTTP endpoints ─────────────────────────────────────────────────────────
@app.post("/events_batch")
async def events_batch(req: dict):
    """Worker → authority batched events. Body: ``EventBatchMsg``.
    All events in the batch share the same ``rank``; authority applies them
    in order under one ``update_lock`` and emits one batched broadcast
    per assigned lookup."""
    if authority is None:
        raise HTTPException(503, "authority not initialised")
    return await authority.on_events_batch(
        rank   = int(req["rank"]),
        events = list(req.get("events", [])),
    )


@app.post("/advertise")
async def advertise(req: dict):
    if authority is None:
        raise HTTPException(503, "authority not initialised")
    return await authority.on_advertise(
        rank      = int(req["rank"]),
        token_ids = list(req.get("token_ids", [])),
        seq       = int(req.get("seq", 0)),
    )


@app.post("/tier_event")
async def tier_event(req: dict):
    if authority is None:
        raise HTTPException(503, "authority not initialised")
    return await authority.on_tier_event(
        rank      = int(req["rank"]),
        token_ids = list(req.get("token_ids", [])),
        event     = str(req.get("event", "")),
        seq       = int(req.get("seq", 0)),
    )


@app.post("/internal/leave")
async def leave(req: dict):
    if authority is None:
        raise HTTPException(503, "authority not initialised")
    return await authority.on_leave(int(req["worker_id"]))


@app.get("/stats")
async def stats():
    if authority is None:
        return {"ok": False, "reason": "authority not initialised"}
    return authority.stats()


@app.get("/alive")
async def alive():
    return {"ok": True, "shard_id": config.SHARD_ID}


@app.post("/internal/topology_push")
async def topology_push(req: dict):
    """NameServer fans this out when the fleet first reaches `ready`,
    so the authority picks up its assigned lookup subset without waiting for
    the next periodic /topology pull."""
    if authority is None:
        raise HTTPException(503, "authority not initialised")
    _ingest_topology(authority, req)
    return {"ok": True, "shard_id": authority.shard_id,
            "lookups_for_this_shard": len(authority.lookup_addrs_for_shard)}
