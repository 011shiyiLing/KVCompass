"""
Dynamo scheme — NVIDIA-Dynamo-style hierarchical KV-cache-aware router.
Real-machine port of the sim's ``DynamoDiscovery`` + the
``scheme == 'dynamo'`` branch of ``route()``.

Each cluster (region) keeps an *accurate* per-cluster index of which
local worker holds which prefix — Dynamo's KvIndexer fed by worker KV
events.  A request first consults its local cluster index; on a *local
cache miss* it fans a lookup out to every other cluster and must wait
for the slowest response before it can pick the globally-best holder.
Candidate scoring is Dynamo's cache-overlap + load cost, realised as the
joint ``T_q_view + T_p + T_t`` (no latency-aware level floor — Dynamo
has no notion that a farther cluster needs a longer match; the index
uses the flat BASELINE_CHUNK_SIZES levels).

Fan-out realisation:
  * **Real multi-router mode** (``REGION_ROUTER_URLS`` set): the local
    miss triggers concurrent HTTP lookups to every peer router's
    ``/internal/dynamo_lookup``; the round-trip latency and the peers'
    service delays are physically paid (await-all = wait-for-slowest).
    To keep each per-cluster index accurate, an advertise for a worker
    in a *remote* region is also forwarded to that region's router
    (``/internal/dynamo_advertise``) — the real-machine equivalent of
    the worker's KV event feeding its own cluster's indexer.
  * **Single-router emulation**: the router already indexes the whole
    fleet, so the fan-out cost is injected instead — the propagation
    floor ``DYNAMO_XCLUSTER_LOOKUP_MS[S]`` (round-trip to the farthest
    region) plus M/D/1 queueing at each queried region's lookup
    coordinator (``DYNAMO_LOOKUP_SERVICE_MS`` service time), exactly as
    in the sim.
"""
from __future__ import annotations

import asyncio
import time

import httpx

from src.ludo.ludo_table import TIER_VRAM, TIER_DRAM, TIER_NONE, chunk_to_key

from ludo_sglang import config
from ludo_sglang.router.load_tracker import WorkerLoad
from ludo_sglang.router.schemes.base import RouteDecision, Scheme


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class DynamoScheme(Scheme):
    name = "dynamo"

    def __init__(self, n_workers: int):
        super().__init__(n_workers)
        self.chunk_sizes = list(config.BASELINE_CHUNK_SIZES)
        # index[level][key][rank] = tier  (accurate, event-fed)
        self.index: list[dict[int, dict[int, int]]] = [
            {} for _ in self.chunk_sizes]
        # Emulation-mode M/D/1 lookup-coordinator queues (next-free, ms).
        self._lookup_free = [0.0] * config.NUM_REGIONS_LOGICAL
        self._t0 = _now_ms()
        self._client: httpx.AsyncClient | None = None
        self.fanout_stats = {"fanouts": 0, "fanout_ms_sum": 0.0,
                             "peer_errors": 0, "adv_forwarded": 0}

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    # ── Index primitives ────────────────────────────────────────────────

    def _keys_for(self, token_ids: list[int]) -> dict[int, int]:
        plen = len(token_ids)
        out = {}
        for cs in self.chunk_sizes:
            if plen < cs:
                break
            out[cs] = chunk_to_key(token_ids[:cs])
        return out

    def _contig_lookup(self, keys: dict[int, int],
                       ranks: set[int] | None = None,
                       ) -> dict[int, tuple[int, int]]:
        """Accurate contiguous longest-prefix match per rank (no floor).
        Returns rank -> (match_len, tier); optionally restricted to
        ``ranks``."""
        contig: dict[int, tuple[int, int]] = {}   # rank -> (level, tier)
        for l, cs in enumerate(self.chunk_sizes):
            key = keys.get(cs)
            if key is None:
                break
            row = self.index[l].get(key)
            if not row:
                continue
            for rank, tier in row.items():
                if ranks is not None and rank not in ranks:
                    continue
                if l == 0:
                    contig[rank] = (0, tier)
                elif rank in contig and contig[rank][0] == l - 1:
                    prev_tier = contig[rank][1]
                    new_tier = (TIER_DRAM if TIER_DRAM in (prev_tier, tier)
                                else TIER_VRAM)
                    contig[rank] = (l, new_tier)
        return {rank: (self.chunk_sizes[lvl], tier)
                for rank, (lvl, tier) in contig.items()}

    # ── Cross-cluster fan-out ───────────────────────────────────────────

    async def _fanout_real(self, keys: dict[int, int],
                           ) -> dict[int, tuple[int, int]]:
        """Concurrent lookup at every peer router; awaits all (slowest
        gates, as in Dynamo).  Returns merged rank -> (ml, tier)."""
        peers = [(reg, url) for reg, url in config.REGION_ROUTER_URLS.items()
                 if reg != config.ROUTER_REGION]
        if not peers:
            return {}
        body = {"keys": {str(cs): k for cs, k in keys.items()}}

        async def _one(url: str):
            try:
                r = await self._http().post(f"{url}/internal/dynamo_lookup",
                                            json=body)
                return r.json().get("candidates", [])
            except Exception:
                self.fanout_stats["peer_errors"] += 1
                return []

        t0 = _now_ms()
        results = await asyncio.gather(*[_one(url) for _, url in peers])
        self.fanout_stats["fanouts"] += 1
        self.fanout_stats["fanout_ms_sum"] += _now_ms() - t0
        merged: dict[int, tuple[int, int]] = {}
        for cands in results:
            for rank, ml, tier in cands:
                rank, ml, tier = int(rank), int(ml), int(tier)
                if ml > merged.get(rank, (0, TIER_NONE))[0]:
                    merged[rank] = (ml, tier)
        return merged

    def _fanout_emulated_ms(self, origin_region: int) -> float:
        """Injected fan-out latency: propagation floor to the farthest
        region + M/D/1 queueing at each queried coordinator (sim's
        ``DynamoDiscovery.broadcast``)."""
        now = _now_ms() - self._t0
        worst = 0.0
        for r in range(config.NUM_REGIONS_LOGICAL):
            if r == origin_region:
                continue
            start = max(self._lookup_free[r], now)
            queue = start - now
            self._lookup_free[r] = start + config.DYNAMO_LOOKUP_SERVICE_MS
            rt = (2.0 * float(config.WAN_LATENCY_MS[origin_region][r])
                  + queue + config.DYNAMO_LOOKUP_SERVICE_MS)
            worst = max(worst, rt)
        return worst

    def peer_lookup(self, keys: dict[int, int]) -> list[list[int]]:
        """Answer a peer router's fan-out: this cluster's local holders.
        Called from the router's /internal/dynamo_lookup endpoint."""
        local_ranks = {r for r in range(self.n_workers)
                       if config.WORKER_REGIONS[r] == config.ROUTER_REGION}
        hits = self._contig_lookup(keys, ranks=local_ranks)
        return [[rank, ml, tier] for rank, (ml, tier) in hits.items()]

    # ── Routing ─────────────────────────────────────────────────────────

    async def route(self, token_ids, load: WorkerLoad, origin_region: int,
                    ctx: dict | None = None):
        prompt_len = len(token_ids)
        keys = self._keys_for(token_ids)
        regions = config.WORKER_REGIONS
        q_view = load.pending_view_ms
        real_mode = bool(config.REGION_ROUTER_URLS) and not config.WAN_EMULATION

        local_ranks = {r for r in range(self.n_workers)
                       if regions[r] == origin_region}
        local_hits = self._contig_lookup(keys, ranks=local_ranks)
        local_hit = any(ml > 0 for ml, _ in local_hits.values())

        # Local least-loaded is always a candidate (full local prefill).
        pool: dict[int, tuple[int, int]] = dict(local_hits)
        if local_ranks:
            local_ll = min(local_ranks, key=lambda r: q_view[r])
            pool.setdefault(local_ll, (0, TIER_NONE))

        lookup_ovh = 0.0
        if not local_hit:
            if real_mode:
                # Physically fan out to the peer routers — the await-all
                # round-trip is the real lookup cost (measured, and also
                # reported in fanout_stats).
                t0 = _now_ms()
                remote = await self._fanout_real(keys)
                lookup_ovh = 0.0   # already paid in wall-clock
                _ = t0
            else:
                remote = self._contig_lookup(
                    keys,
                    ranks={r for r in range(self.n_workers)
                           if regions[r] != origin_region})
                lookup_ovh = self._fanout_emulated_ms(origin_region)
            for rank, (ml, tier) in remote.items():
                if ml > pool.get(rank, (0, TIER_NONE))[0]:
                    pool[rank] = (ml, tier)

        # A request from a region with no local worker and no remote index
        # hit would otherwise leave the pool empty; fall back to the
        # globally least-loaded worker (what the other schemes do).
        if not pool:
            global_ll = min(range(self.n_workers), key=lambda r: q_view[r])
            pool[global_ll] = (0, TIER_NONE)

        # Joint T_q_view + T_p + T_t over the candidate pool.
        best_rank, best_score, best_ml = None, float("inf"), 0
        for rank, (ml, tier) in pool.items():
            em = ml * load.cache_belief[rank]
            t_p = max(0.0, prompt_len - em) * load.prefill_rate[rank]
            t_t = config.wan_ms_to_worker(origin_region, rank)
            if tier == TIER_DRAM and em > 0:
                t_t += em * config.PCIE_RELOAD_MS_PER_TOKEN
            score = q_view[rank] + t_p + t_t
            if score < best_score:
                best_score, best_rank, best_ml = score, rank, ml

        wan_ms = config.wan_ms_to_worker(origin_region, best_rank)
        return RouteDecision(
            rank             = best_rank,
            matched_tokens   = best_ml,
            extra_lookup_ms  = lookup_ovh + wan_ms,
            wan_to_worker_ms = wan_ms,
            score_ms         = best_score,
            info             = {
                "local_hit":       local_hit,
                "fanout_ms":       round(lookup_ovh, 2),
                "pool":            {str(r): [ml, t]
                                    for r, (ml, t) in pool.items()},
            },
        )

    # ── Index maintenance (event-fed, accurate) ─────────────────────────

    def advertise(self, token_ids, rank, tier: int = TIER_VRAM):
        keys = self._keys_for(token_ids)
        for l, cs in enumerate(self.chunk_sizes):
            key = keys.get(cs)
            if key is None:
                break
            self.index[l].setdefault(key, {})[rank] = tier
        # Real mode: a dispatch to a remote-region worker must feed THAT
        # cluster's indexer, or its region will never see the new holder.
        if (config.REGION_ROUTER_URLS
                and config.WORKER_REGIONS[rank] != config.ROUTER_REGION):
            url = config.REGION_ROUTER_URLS.get(config.WORKER_REGIONS[rank])
            if url:
                self.fanout_stats["adv_forwarded"] += 1
                asyncio.create_task(self._forward_advertise(url, rank, keys))

    async def _forward_advertise(self, url: str, rank: int,
                                 keys: dict[int, int]) -> None:
        try:
            await self._http().post(
                f"{url}/internal/dynamo_advertise",
                json={"rank": rank, "tier": int(TIER_VRAM),
                      "keys": {str(cs): k for cs, k in keys.items()}})
        except Exception:
            self.fanout_stats["peer_errors"] += 1

    def apply_remote_advertise(self, rank: int, tier: int,
                               keys: dict[int, int]) -> None:
        """Ingress for /internal/dynamo_advertise."""
        for l, cs in enumerate(self.chunk_sizes):
            key = keys.get(cs)
            if key is None:
                continue
            self.index[l].setdefault(key, {})[rank] = tier

    def notify_tier(self, token_ids, rank: int, new_tier: int) -> None:
        keys = self._keys_for(token_ids)
        for l, cs in enumerate(self.chunk_sizes):
            key = keys.get(cs)
            if key is None:
                break
            row = self.index[l].get(key)
            if not row:
                continue
            if new_tier == TIER_NONE:
                row.pop(rank, None)
                if not row:
                    del self.index[l][key]
            else:
                if rank in row:
                    row[rank] = new_tier

    def notify_evict(self, token_ids, rank: int) -> None:
        self.notify_tier(token_ids, rank, TIER_NONE)

    def stats(self):
        s = super().stats()
        fo = dict(self.fanout_stats)
        if fo["fanouts"]:
            fo["fanout_ms_avg"] = round(fo["fanout_ms_sum"] / fo["fanouts"], 2)
        s.update({
            "chunk_sizes": self.chunk_sizes,
            "entries_per_level": [len(t) for t in self.index],
            "fanout": fo,
        })
        return s
