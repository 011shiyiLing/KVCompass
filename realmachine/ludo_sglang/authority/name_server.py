"""
name_server.py — central-directory NameServer.

Roles
=====
1. Accept ``Register{Authority,Lookup,Worker}`` POSTs on startup, assign
   monotonically increasing IDs (or accept the ID the caller proposed
   if it is internally consistent), and store ``(role, id, addr,
   region)`` records.
2. Once every expected node has registered, mark the fleet ``ready``
   and start the heartbeat watcher.  Other nodes can pull the full
   topology via ``GET /topology`` at any time.
3. Watch worker heartbeats; on timeout, broadcast ``Leave(worker_id)``
   to every authority so each authority can clear that worker's slot in
   every entry it owns.

Compared to Smash's NameServer
==============================
* No bulk-borrowing / load-balancing daemon — KV cache doesn't migrate.
* No Hot block forwarding — workers report tier transitions directly
  to their authority, not through us.
* Topology pull is HTTP GET instead of TCP push, so any node (router,
  worker, authority) can call ``GET /topology`` and refresh on demand.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

from ludo_sglang import config

_log = logging.getLogger("ludo.name_server")


# ── In-memory registry (single-process; fine for v1) ────────────────────
class _State:
    authorities: list[dict] = []  # [{"shard_id", "addr"}]
    lookups: list[dict] = []     # [{"lookup_id", "addr"}]
    workers: list[dict] = []     # [{"rank", "addr", "region", "last_hb", "alive"}]
    n_shards_expected: int = 1
    n_lookups_expected: int = 0
    n_workers_expected: int = 0
    ready: bool = False
    register_lock: asyncio.Lock = None  # type: ignore  # initialised in startup
    watcher_task: Optional[asyncio.Task] = None
    # Static lookup→shard partition computed when `ready` first flips True.
    # ``lookup_for_shard[s]`` is the list of lookup_ids that mirror shard s.
    # Mirrors Smash's `lookupFunctionsForShard[i*nShards/nLookups].push_back(i)`
    # so each authority only broadcasts to its assigned subset.
    lookup_for_shard: list[list[int]] = []


state = _State()
app = FastAPI(title="KVCompass NameServer")


# ── Lifecycle ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    state.register_lock = asyncio.Lock()
    state.n_shards_expected = max(1, len(config.AUTHORITY_URLS))
    # n_lookups / n_workers come from env at start-time; defaults are
    # forgiving so the name server can still be useful in single-node
    # debug runs where you only want authority + worker.
    state.n_lookups_expected = int(
        # Falls back to 1 (the embedded edge router) if not set.
        getattr(config, "N_LOOKUPS_EXPECTED", 1) or 1
    )
    state.n_workers_expected = config.N_DP_WORKERS
    print(f"[name-server] expecting {state.n_shards_expected} authority(ies), "
          f"{state.n_lookups_expected} lookup(s), "
          f"{state.n_workers_expected} worker(s)")
    state.watcher_task = asyncio.create_task(_heartbeat_watcher())


@app.on_event("shutdown")
async def _shutdown():
    if state.watcher_task is not None:
        state.watcher_task.cancel()
        try:
            await state.watcher_task
        except (asyncio.CancelledError, Exception):
            pass


# ── Registration ───────────────────────────────────────────────────────
@app.post("/register/authority")
async def register_authority(req: dict):
    addr = str(req.get("addr", "")).rstrip("/")
    shard_id = int(req.get("shard_id", 0))
    if not addr:
        raise HTTPException(400, "addr required")
    async with state.register_lock:
        # Replace if shard_id already registered (authority restart).
        for m in state.authorities:
            if m["shard_id"] == shard_id:
                m["addr"] = addr
                _maybe_ready_locked()
                return {"ok": True, "shard_id": shard_id, "n_shards": state.n_shards_expected}
        state.authorities.append({"shard_id": shard_id, "addr": addr})
        _maybe_ready_locked()
    return {"ok": True, "shard_id": shard_id, "n_shards": state.n_shards_expected}


@app.post("/register/lookup")
async def register_lookup(req: dict):
    addr = str(req.get("addr", "")).rstrip("/")
    lookup_id = int(req.get("lookup_id", 0))
    if not addr:
        raise HTTPException(400, "addr required")
    async with state.register_lock:
        for l in state.lookups:
            if l["lookup_id"] == lookup_id:
                l["addr"] = addr
                _maybe_ready_locked()
                return {"ok": True, "lookup_id": lookup_id}
        state.lookups.append({"lookup_id": lookup_id, "addr": addr})
        _maybe_ready_locked()
    return {"ok": True, "lookup_id": lookup_id}


@app.post("/register/worker")
async def register_worker(req: dict):
    addr = str(req.get("addr", "")).rstrip("/")
    rank = int(req.get("rank", -1))
    region = int(req.get("region", 0))
    if rank < 0:
        raise HTTPException(400, "rank required")
    async with state.register_lock:
        for w in state.workers:
            if w["rank"] == rank:
                w["addr"] = addr
                w["region"] = region
                w["last_hb"] = time.monotonic()
                w["alive"] = True
                _maybe_ready_locked()
                return {"ok": True, "rank": rank}
        state.workers.append({
            "rank":    rank,
            "addr":    addr,
            "region":  region,
            "last_hb": time.monotonic(),
            "alive":   True,
        })
        _maybe_ready_locked()
    return {"ok": True, "rank": rank}


def _maybe_ready_locked():
    if (
        len(state.authorities) >= state.n_shards_expected
        and len(state.lookups) >= state.n_lookups_expected
        and len(state.workers) >= state.n_workers_expected
    ):
        first_ready = not state.ready
        state.ready = True
        if first_ready:
            _compute_lookup_for_shard_locked()
            print(f"[name-server] all expected nodes registered, fleet READY  "
                  f"lookup_for_shard={state.lookup_for_shard}")
            asyncio.create_task(_inform_all_topology())


def _compute_lookup_for_shard_locked() -> None:
    """Compute the lookup→shard mirror map.  Two modes:

    * ``full``      every edge mirrors every shard — required when the
                    edge router IS the request-routing gateway, because a
                    single inference request hits L chunk levels whose
                    hashes can land in different shards.
    * ``partition`` Smash-style client-side sharding — lookup i mirrors
                    shard ``i*nShards/nLookups``; clients route by shard.

    Recomputed every time the registry reaches ready; lookup IDs are
    stable across restarts so the partition is deterministic."""
    n_shards = max(1, state.n_shards_expected)
    n_lookups = len(state.lookups)
    all_ids = [int(l["lookup_id"]) for l in state.lookups]

    if config.LOOKUP_SHARDING_MODE == "full":
        state.lookup_for_shard = [list(all_ids) for _ in range(n_shards)]
        return

    # partition mode (Smash-faithful)
    out: list[list[int]] = [[] for _ in range(n_shards)]
    for lid in all_ids:
        shard = (lid * n_shards) // max(1, n_lookups)
        if shard >= n_shards:
            shard = n_shards - 1
        out[shard].append(lid)
    state.lookup_for_shard = out


async def _inform_all_topology() -> None:
    """Push the freshly-computed topology to every registered node so
    authorities can start broadcasting to their assigned lookup subset
    without waiting for the next periodic pull."""
    body = _topology_body()
    targets = [m["addr"] for m in state.authorities] + [l["addr"] for l in state.lookups]
    if not targets:
        return
    async with httpx.AsyncClient(timeout=config.MAINTENANCE_HTTP_TIMEOUT_S) as c:
        await asyncio.gather(*[
            _push_topology(c, addr, body) for addr in targets
        ], return_exceptions=True)


async def _push_topology(client: httpx.AsyncClient, addr: str, body: dict) -> None:
    try:
        await client.post(f"{addr}/internal/topology_push", json=body)
    except Exception as exc:
        _log.debug("topology push to %s failed: %s", addr, exc)


def _topology_body() -> dict:
    return {
        "authorities":  list(state.authorities),
        "lookups":  list(state.lookups),
        "workers":  [
            {k: w[k] for k in ("rank", "addr", "region", "alive")}
            for w in state.workers
        ],
        "n_shards": state.n_shards_expected,
        "ready":    state.ready,
        "lookup_for_shard": list(state.lookup_for_shard),
    }


# ── Heartbeat ──────────────────────────────────────────────────────────
@app.post("/heartbeat")
async def heartbeat(req: dict):
    rank = int(req.get("rank", -1))
    if rank < 0:
        raise HTTPException(400, "rank required")
    for w in state.workers:
        if w["rank"] == rank:
            w["last_hb"] = time.monotonic()
            if not w["alive"]:
                # Worker is back: mark alive, but authorities won't auto-
                # restore slots — they'll be repopulated by future
                # advertise calls.  (Same behaviour as a fresh worker.)
                w["alive"] = True
                print(f"[name-server] worker rank={rank} resumed heartbeats")
            return {"ok": True}
    raise HTTPException(404, f"unknown worker rank={rank}")


# ── Topology pull ──────────────────────────────────────────────────────
@app.get("/topology")
async def topology():
    return _topology_body()


# ── Heartbeat watcher daemon ───────────────────────────────────────────
async def _heartbeat_watcher():
    """Background daemon: every HEARTBEAT_CHECK_INTERVAL_S, find workers
    silent for more than HEARTBEAT_TIMEOUT_S and tell every authority to
    drop them.
    """
    try:
        async with httpx.AsyncClient(timeout=config.MAINTENANCE_HTTP_TIMEOUT_S) as client:
            while True:
                await asyncio.sleep(config.HEARTBEAT_CHECK_INTERVAL_S)
                if not state.ready:
                    continue
                now = time.monotonic()
                for w in state.workers:
                    if not w["alive"]:
                        continue
                    if now - w["last_hb"] > config.HEARTBEAT_TIMEOUT_S:
                        w["alive"] = False
                        print(f"[name-server] worker rank={w['rank']} timed out "
                              f"(last hb {now - w['last_hb']:.1f}s ago); "
                              f"broadcasting Leave to {len(state.authorities)} authority(ies)")
                        await asyncio.gather(*[
                            _post_leave(client, m["addr"], w["rank"])
                            for m in state.authorities
                        ], return_exceptions=True)
    except asyncio.CancelledError:
        pass


async def _post_leave(client: httpx.AsyncClient, authority_addr: str, worker_id: int):
    try:
        await client.post(f"{authority_addr}/internal/leave",
                          json={"worker_id": worker_id})
    except Exception as exc:
        _log.warning("leave POST to %s failed: %s", authority_addr, exc)


# ── Main entry ─────────────────────────────────────────────────────────
def main():
    import uvicorn
    host, port = _parse_url(config.NAMESERVER_URL)
    uvicorn.run(
        "ludo_sglang.authority.name_server:app",
        host=host, port=port, log_level="info",
    )


def _parse_url(url: str) -> tuple[str, int]:
    # http://127.0.0.1:9999 → ("127.0.0.1", 9999)
    no_scheme = url.split("://", 1)[-1]
    host, _, port = no_scheme.partition(":")
    return host or "0.0.0.0", int(port or "9999")


if __name__ == "__main__":
    main()
