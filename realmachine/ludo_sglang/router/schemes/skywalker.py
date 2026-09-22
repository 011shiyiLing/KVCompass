"""
SkyWalker scheme (EuroSys'26) — two-layer, locality-first, prefix-tree
load balancer.  Real-machine port of the sim's ``SkyWalkerDiscovery`` +
the ``scheme == 'skywalker'`` branch of ``route()``.

Mechanics reproduced from the paper's headline (prefix-tree) variant:

* **Availability from a stale heartbeat** — every
  ``SKYWALKER_PROBE_INTERVAL_MS`` (100 ms) the LB snapshots each local
  worker's free continuous-batch slots (``SKYWALKER_SLOTS − inflight``);
  between probes each selective push burns one credit, so a worker
  stops looking available the moment the LB has filled the capacity it
  saw.  Capacity freed mid-window is invisible until the next probe.

* **Locality-first** — if ANY local worker is available, serve locally;
  only when the whole local region is full forward to an available
  remote region (remote availability is judged from the viewed queue —
  the peer routers' 100 ms load reports — mirroring the paper's LB↔LB
  heartbeat).

* **Prefix-aware pick** — among the available candidates, choose the
  target with the longest matching prefix in the guiding trie: the
  *replica trie* for the local pick, the *snapshot trie* (which remote
  regions this LB forwarded which prefixes to) for the region hop.
  Ties / no-match fall back to least-loaded.  The tries are built
  incrementally from routed requests (bounded, FIFO-evicted) and are
  deliberately NOT synced with the workers' real caches.

Two-layer forwarding: in the real multi-region deployment (one router
per region, ``REGION_ROUTER_URLS`` set) a cross-region decision returns
``info["forward_url"]`` — the router POSTs the request to the target
region's router, which makes the final replica pick with its own fresh
local view (``ctx["swk_hop"] == 1`` marks the hop so a forwarded request
is always served in the receiving region).  In single-router emulation
the origin LB picks the remote worker itself and the WAN cost is
injected as usual.
"""
from __future__ import annotations

import time
from collections import OrderedDict

from src.ludo.ludo_table import chunk_to_key

from ludo_sglang import config
from ludo_sglang.router.load_tracker import WorkerLoad
from ludo_sglang.router.schemes.base import RouteDecision, Scheme


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class SkyWalkerScheme(Scheme):
    name = "skywalker"

    #: max entries per (region, level) trie — bounds LB memory, FIFO evict.
    TRIE_MAX = 65536

    def __init__(self, n_workers: int):
        super().__init__(n_workers)
        self.chunk_sizes = list(config.BASELINE_CHUNK_SIZES)
        n_regions = config.NUM_REGIONS_LOGICAL
        # replica_trie[region][level] : key -> {local worker ranks}
        self.replica_trie: list[list[OrderedDict]] = [
            [OrderedDict() for _ in self.chunk_sizes] for _ in range(n_regions)]
        # snapshot_trie[origin][level] : key -> {remote region ids}
        self.snapshot_trie: list[list[OrderedDict]] = [
            [OrderedDict() for _ in self.chunk_sizes] for _ in range(n_regions)]
        # Heartbeat availability snapshot (free-slot credits per worker).
        self._credits: dict[int, int] = {}
        self._last_probe: float | None = None

    # ── Heartbeat / selective pushing ───────────────────────────────────

    def _probe(self, load: WorkerLoad) -> None:
        now = _now_ms()
        if (self._last_probe is None
                or now - self._last_probe >= config.SKYWALKER_PROBE_INTERVAL_MS):
            self._credits = {
                r: max(0, config.SKYWALKER_SLOTS - load.inflight[r])
                for r in range(self.n_workers)
            }
            self._last_probe = now

    def _is_avail(self, rank: int) -> bool:
        return self._credits.get(rank, 0) > 0

    def _consume(self, rank: int) -> None:
        c = self._credits.get(rank, 0)
        if c > 0:
            self._credits[rank] = c - 1

    # ── Trie helpers ────────────────────────────────────────────────────

    def _keys_for(self, token_ids: list[int]) -> dict[int, int]:
        plen = len(token_ids)
        out = {}
        for cs in self.chunk_sizes:
            if plen < cs:
                break
            out[cs] = chunk_to_key(token_ids[:cs])
        return out

    @staticmethod
    def _insert(level_tries: list[OrderedDict], keys: dict[int, int],
                chunk_sizes: list[int], target, max_entries: int) -> None:
        for l, cs in enumerate(chunk_sizes):
            key = keys.get(cs)
            if key is None:
                break
            d = level_tries[l]
            if key in d:
                d[key].add(target)
                d.move_to_end(key)
            else:
                d[key] = {target}
                if len(d) > max_entries:
                    d.popitem(last=False)          # FIFO evict earliest

    @staticmethod
    def _match(level_tries: list[OrderedDict], keys: dict[int, int],
               chunk_sizes: list[int], avail: set) -> tuple[set, int]:
        """Longest contiguous prefix whose holders intersect `avail`.
        Returns (tied longest-match targets, matched token length)."""
        best: set = set()
        best_len = 0
        for l, cs in enumerate(chunk_sizes):
            key = keys.get(cs)
            if key is None:
                break
            holders = level_tries[l].get(key)
            if not holders:
                break
            inter = holders & avail
            if not inter:
                break
            best, best_len = inter, cs
        return best, best_len

    def _record(self, origin_region: int, rank: int, node_region: int,
                keys: dict[int, int]) -> None:
        self._insert(self.replica_trie[node_region], keys,
                     self.chunk_sizes, rank, self.TRIE_MAX)
        if node_region != origin_region:
            self._insert(self.snapshot_trie[origin_region], keys,
                         self.chunk_sizes, node_region, self.TRIE_MAX)

    # ── Routing ─────────────────────────────────────────────────────────

    async def route(self, token_ids, load: WorkerLoad, origin_region: int,
                    ctx: dict | None = None):
        keys = self._keys_for(token_ids)
        hop = int((ctx or {}).get("swk_hop", 0))
        self._probe(load)

        regions = config.WORKER_REGIONS
        q_live = load.pending_ms          # own dispatches, live
        q_view = load.pending_view_ms     # + peers' 100 ms load reports

        local_ranks = [r for r in range(self.n_workers)
                       if regions[r] == origin_region]
        local_avail = {r for r in local_ranks if self._is_avail(r)}

        chosen: int | None = None
        match_len = 0
        forward_region: int | None = None

        if local_avail:
            cand, match_len = self._match(
                self.replica_trie[origin_region], keys,
                self.chunk_sizes, local_avail)
            pool = cand if cand else local_avail
            chosen = min(pool, key=lambda r: q_live[r])
            if not cand:
                match_len = 0
        elif hop >= 1:
            # Forwarded request: the RECEIVING router makes the final
            # replica pick among ITS OWN region's workers (the paper's
            # second layer) — NOT the origin's.  Using origin_region here
            # bounced requests straight back to the saturated origin
            # region and melted it (77 s p99 before this fix).
            my_ranks = [r for r in range(self.n_workers)
                        if regions[r] == config.ROUTER_REGION] \
                       or list(range(self.n_workers))
            my_avail = {r for r in my_ranks if self._is_avail(r)}
            cand, match_len = self._match(
                self.replica_trie[config.ROUTER_REGION], keys,
                self.chunk_sizes, my_avail)
            pool = cand if cand else (my_avail or set(my_ranks))
            chosen = min(pool, key=lambda r: q_live[r])
            if not cand:
                match_len = 0
        else:
            # All local workers full → look at remote regions with
            # capacity, judged from the viewed queue (LB↔LB heartbeat).
            remote_avail: dict[int, set] = {}
            for r in range(self.n_workers):
                if regions[r] == origin_region:
                    continue
                if q_view[r] <= config.SKYWALKER_BUSY_MS:
                    remote_avail.setdefault(regions[r], set()).add(r)
            if remote_avail:
                rcand, match_len = self._match(
                    self.snapshot_trie[origin_region], keys,
                    self.chunk_sizes, set(remote_avail.keys()))
                region_pool = rcand if rcand else set(remote_avail.keys())
                tgt = min(region_pool,
                          key=lambda X: min(q_view[i] for i in remote_avail[X]))
                if not rcand:
                    match_len = 0
                # Two-layer: hand the request to the target region's
                # router when one exists; it re-decides with fresh local
                # state.  Otherwise (single-router emulation) pick the
                # remote worker here.
                if tgt in config.REGION_ROUTER_URLS and hop == 0:
                    forward_region = tgt
                av = remote_avail[tgt]
                ncand, nlen = self._match(
                    self.replica_trie[tgt], keys, self.chunk_sizes, av)
                pool = ncand if ncand else av
                chosen = min(pool, key=lambda r: q_view[r])
                if ncand:
                    match_len = nlen
            else:
                # Nothing available anywhere → queue at the least-loaded
                # local worker.
                chosen = min(local_ranks or list(range(self.n_workers)),
                             key=lambda r: q_live[r])

        self._consume(chosen)
        node_region = regions[chosen]
        self._record(origin_region, chosen, node_region, keys)

        wan_ms = config.wan_ms_to_worker(origin_region, chosen)
        info = {
            "swk_hop": hop,
            "local_avail": sorted(local_avail),
            "trie_match_len": match_len,
        }
        if forward_region is not None:
            info["forward_region"] = forward_region
            info["forward_url"] = config.REGION_ROUTER_URLS[forward_region]

        return RouteDecision(
            rank             = chosen,
            matched_tokens   = match_len,
            extra_lookup_ms  = wan_ms,      # WAN only; stripped in real mode
            wan_to_worker_ms = wan_ms,
            score_ms         = q_view[chosen],
            info             = info,
        )

    def stats(self):
        s = super().stats()
        s.update({
            "chunk_sizes": self.chunk_sizes,
            "credits": dict(self._credits),
            "replica_trie_entries": [
                sum(len(d) for d in lv) for lv in self.replica_trie],
            "snapshot_trie_entries": [
                sum(len(d) for d in lv) for lv in self.snapshot_trie],
        })
        return s
